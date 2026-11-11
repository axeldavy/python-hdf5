#pragma once

#include "file.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace python_hdf5 {

// Datatype names
//
// Python and C++ exchange HDF5 datatypes as ASCII names:
//   "<i4", ">u2", "|i1", "<f2", "<f8"   numeric types ("|b1" is the h5py boolean enum)
//   "<c8", "<c16"                      complex values stored as {r, i} compounds
//   "|S<n>[:utf8]"                     fixed-length strings
//   "vlen:utf8", "vlen:ascii"          variable-length strings
//   "vlen:<numeric>"                   variable-length numeric arrays
//   "enum:<integer>:<name>=<value>,..." enums with percent-escaped names, sorted by name
//   "opaque:<size>:<escaped tag>"      opaque data

// Hold dtype, shape, and C-order data for a transfer.
// Fixed-size elements live in data; variable-length elements (strings or raw numeric
// arrays) live in strings.
// Reads of fixed-size data fill block (malloc storage, uninitialized before the read).
// Writes may borrow caller-owned bytes through external, which must outlive the call;
// otherwise the bytes come from data.
struct AttributeData {
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<unsigned char> data;
    std::vector<std::string> strings;
    std::shared_ptr<unsigned char> block;
    std::size_t block_size = 0;
    const unsigned char* external = nullptr;
    std::size_t external_size = 0;

    // Return the bytes to write and store their count in size.
    const unsigned char* bytes(std::size_t& size) const {
        if (external) {
            size = external_size;
            return external;
        }
        size = data.size();
        return data.data();
    }
};

// Retrieve the rank and dimensions in one call; null dataspaces are checked by the caller.
inline Status dataspace_shape(hid_t space, std::vector<std::uint64_t>& shape,
                             std::string& error) {
    hsize_t dims[H5S_MAX_RANK];
    int rank = H5Sget_simple_extent_dims(space, dims, nullptr);
    if (rank < 0) {
        error = error_message();
        return Status::Failure;
    }
    if (rank > H5S_MAX_RANK) {
        error = "dataspace rank exceeds HDF5 limit";
        return Status::Unsupported;
    }
    shape.assign(dims, dims + rank);
    return Status::Ok;
}

// Numeric and boolean types

// Map a NumPy numeric dtype string to a borrowed predefined HDF5 numeric type.
inline hid_t numeric_type(const std::string& dtype) {
    if (dtype.size() != 3) return H5I_INVALID_HID;
    const bool big = dtype[0] == '>';
    switch (dtype[1]) {
        case 'i':
            switch (dtype[2]) {
                case '1': return H5T_STD_I8LE;
                case '2': return big ? H5T_STD_I16BE : H5T_STD_I16LE;
                case '4': return big ? H5T_STD_I32BE : H5T_STD_I32LE;
                case '8': return big ? H5T_STD_I64BE : H5T_STD_I64LE;
            }
            break;
        case 'u':
            switch (dtype[2]) {
                case '1': return H5T_STD_U8LE;
                case '2': return big ? H5T_STD_U16BE : H5T_STD_U16LE;
                case '4': return big ? H5T_STD_U32BE : H5T_STD_U32LE;
                case '8': return big ? H5T_STD_U64BE : H5T_STD_U64LE;
            }
            break;
        case 'f':
            switch (dtype[2]) {
                case '2': return big ? H5T_IEEE_F16BE : H5T_IEEE_F16LE;
                case '4': return big ? H5T_IEEE_F32BE : H5T_IEEE_F32LE;
                case '8': return big ? H5T_IEEE_F64BE : H5T_IEEE_F64LE;
            }
    }
    return H5I_INVALID_HID;
}

// Create an owned HDF5 enum compatible with h5py boolean values.
inline hid_t boolean_type() {
    ScopedId type(H5Tenum_create(H5T_STD_I8LE), H5Tclose);
    if (type.get() < 0) return H5I_INVALID_HID;
    unsigned char false_value = 0;
    unsigned char true_value = 1;
    if (H5Tenum_insert(type.get(), "FALSE", &false_value) < 0 ||
        H5Tenum_insert(type.get(), "TRUE", &true_value) < 0)
        return H5I_INVALID_HID;
    return type.release();
}

// Return the numeric dtype string of an integer or float type, or an empty string.
inline std::string numeric_dtype(hid_t type) {
    H5T_class_t kind = H5Tget_class(type);
    if (kind != H5T_INTEGER && kind != H5T_FLOAT) return {};
    size_t size = H5Tget_size(type);
    if (size != 1 && size != 2 && size != 4 && size != 8) return {};
    if (kind == H5T_FLOAT && size == 1) return {};

    char code = 'f';
    if (kind == H5T_INTEGER) {
        H5T_sign_t sign = H5Tget_sign(type);
        if (sign == H5T_SGN_2) code = 'i';
        else if (sign == H5T_SGN_NONE) code = 'u';
        else return {};
    }
    char order = '|';
    if (size > 1) {
        H5T_order_t byte_order = H5Tget_order(type);
        if (byte_order == H5T_ORDER_LE) order = '<';
        else if (byte_order == H5T_ORDER_BE) order = '>';
        else return {};
    }
    std::string name{order, code, static_cast<char>('0' + size)};
    return H5Tequal(type, numeric_type(name)) > 0 ? name : std::string();
}

// Return the item size in bytes encoded in a numeric dtype string such as "<i4".
inline size_t numeric_item_size(const std::string& name) {
    return static_cast<size_t>(name[2] - '0');
}

// Return true for numeric names that may be used as variable-length array elements.
inline bool is_vlen_base_name(const std::string& name) {
    return name.size() == 3 && numeric_type(name) >= 0;
}

// Name escaping

// Percent-escape every byte outside [A-Za-z0-9_.-] so names are free of separators.
inline std::string escape_name(const std::string& raw) {
    static const char* digits = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : raw) {
        if (std::isalnum(c) || c == '_' || c == '.' || c == '-') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(digits[c >> 4]);
            out.push_back(digits[c & 15]);
        }
    }
    return out;
}

// Reverse escape_name; return false for malformed escapes.
inline bool unescape_name(const std::string& text, std::string& out) {
    out.clear();
    for (size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '%') {
            out.push_back(text[i]);
            continue;
        }
        if (i + 2 >= text.size()) return false;
        char pair[3] = {text[i + 1], text[i + 2], 0};
        char* end = nullptr;
        unsigned long value = std::strtoul(pair, &end, 16);
        if (end != pair + 2) return false;
        out.push_back(static_cast<char>(value));
        i += 2;
    }
    return true;
}

// Complex types

// Return the float dtype of one complex component, or empty when name is not complex.
inline std::string complex_component_name(const std::string& name) {
    if (name.size() < 3 || name[1] != 'c' || (name[0] != '<' && name[0] != '>')) return {};
    if (name.substr(2) == "8") return std::string(1, name[0]) + "f4";
    if (name.substr(2) == "16") return std::string(1, name[0]) + "f8";
    return {};
}

// Create an owned h5py-compatible complex type: a compound of float members r and i.
inline hid_t complex_type(const std::string& name) {
    std::string component = complex_component_name(name);
    if (component.empty()) return H5I_INVALID_HID;
    ScopedId base(H5Tcopy(numeric_type(component)), H5Tclose);
    if (base.get() < 0) return H5I_INVALID_HID;
    size_t size = H5Tget_size(base.get());
    ScopedId type(H5Tcreate(H5T_COMPOUND, 2 * size), H5Tclose);
    if (type.get() < 0 || H5Tinsert(type.get(), "r", 0, base.get()) < 0 ||
        H5Tinsert(type.get(), "i", size, base.get()) < 0)
        return H5I_INVALID_HID;
    return type.release();
}

// Return the complex dtype of an h5py compound or native HDF5 complex type, or empty.
inline std::string complex_dtype(hid_t type) {
    // Recognize native complex types through their floating-point base.
    H5T_class_t type_class = H5Tget_class(type);
    std::string component;
    if (type_class == H5T_COMPLEX) {
        ScopedId super(H5Tget_super(type), H5Tclose);
        if (super.get() < 0) return {};
        component = numeric_dtype(super.get());
    } else {
        // Recognize compounds of exactly two equal floating-point members r and i.
        if (type_class != H5T_COMPOUND || H5Tget_nmembers(type) != 2) return {};
        ScopedId first(H5Tget_member_type(type, 0), H5Tclose);
        ScopedId second(H5Tget_member_type(type, 1), H5Tclose);
        if (first.get() < 0 || second.get() < 0) return {};
        component = numeric_dtype(first.get());
        size_t size = H5Tget_size(first.get());
        char* first_name = H5Tget_member_name(type, 0);
        char* second_name = H5Tget_member_name(type, 1);
        bool names = first_name && second_name && std::string(first_name) == "r" &&
                     std::string(second_name) == "i";
        if (first_name) H5free_memory(first_name);
        if (second_name) H5free_memory(second_name);
        if (!names || H5Tget_size(type) != 2 * size || H5Tget_member_offset(type, 0) != 0 ||
            H5Tget_member_offset(type, 1) != size || numeric_dtype(second.get()) != component)
            return {};
    }

    // Map the component type to the NumPy complex dtype.
    if (component.size() != 3 || component[1] != 'f') return {};
    if (component[2] == '4') return std::string(1, component[0]) + "c8";
    if (component[2] == '8') return std::string(1, component[0]) + "c16";
    return {};
}

// Enum types

// Describe one enum member with its value rendered as a decimal integer.
using EnumMember = std::pair<std::string, std::string>;

// Return the enum dtype name of a non-boolean integer enum, or empty if unsupported.
inline std::string enum_dtype_name(hid_t type) {
    // Recognize the h5py boolean enum separately.
    ScopedId boolean(boolean_type(), H5Tclose);
    if (boolean.get() >= 0 && H5Tequal(type, boolean.get()) > 0) return "|b1";

    // Require a little-endian or single-byte integer base type.
    ScopedId super(H5Tget_super(type), H5Tclose);
    if (super.get() < 0) return {};
    std::string base = numeric_dtype(super.get());
    if (base.size() != 3 || (base[1] != 'i' && base[1] != 'u') || base[0] == '>') return {};
    const bool is_signed = base[1] == 'i';
    const size_t size = numeric_item_size(base);

    // Collect members sorted by name for a canonical form.
    int count = H5Tget_nmembers(type);
    if (count < 0) return {};
    std::vector<EnumMember> members;
    for (int i = 0; i < count; ++i) {
        char* raw_name = H5Tget_member_name(type, static_cast<unsigned>(i));
        if (!raw_name) return {};
        std::string name(raw_name);
        H5free_memory(raw_name);
        unsigned char bytes[8] = {0};
        if (H5Tget_member_value(type, static_cast<unsigned>(i), bytes) < 0) return {};
        std::uint64_t value = 0;
        for (size_t k = 0; k < size; ++k) value |= static_cast<std::uint64_t>(bytes[k]) << (8 * k);
        std::string text;
        if (is_signed) {
            if (size < 8 && (value >> (8 * size - 1)) & 1) value |= ~std::uint64_t{0} << (8 * size);
            text = std::to_string(static_cast<std::int64_t>(value));
        } else {
            text = std::to_string(value);
        }
        members.emplace_back(escape_name(name), text);
    }
    std::sort(members.begin(), members.end());

    // Join the members into the dtype name.
    std::string out = "enum:" + base + ":";
    for (size_t i = 0; i < members.size(); ++i) {
        if (i) out.push_back(',');
        out += members[i].first + "=" + members[i].second;
    }
    return out;
}

// Create an owned HDF5 enum from an "enum:" dtype name, or return an invalid id.
inline hid_t enum_type(const std::string& name) {
    // Split the base type from the member list.
    if (name.rfind("enum:", 0) != 0) return H5I_INVALID_HID;
    size_t colon = name.find(':', 5);
    if (colon == std::string::npos) return H5I_INVALID_HID;
    std::string base = name.substr(5, colon - 5);
    if (base.size() != 3 || (base[1] != 'i' && base[1] != 'u') || base[0] == '>')
        return H5I_INVALID_HID;
    const bool is_signed = base[1] == 'i';
    const size_t size = numeric_item_size(base);
    ScopedId base_type(H5Tcopy(numeric_type(base)), H5Tclose);
    if (base_type.get() < 0) return H5I_INVALID_HID;
    ScopedId type(H5Tenum_create(base_type.get()), H5Tclose);
    if (type.get() < 0) return H5I_INVALID_HID;

    // Insert each member after range-checking its value for the base type.
    std::string rest = name.substr(colon + 1);
    size_t start = 0;
    while (start < rest.size()) {
        size_t end = rest.find(',', start);
        std::string member = rest.substr(start, end == std::string::npos ? end : end - start);
        start = end == std::string::npos ? rest.size() : end + 1;
        size_t equals = member.find('=');
        std::string member_name;
        if (equals == std::string::npos || !unescape_name(member.substr(0, equals), member_name))
            return H5I_INVALID_HID;
        std::string text = member.substr(equals + 1);
        errno = 0;
        char* parse_end = nullptr;
        std::uint64_t bits = 0;
        if (is_signed) {
            long long value = std::strtoll(text.c_str(), &parse_end, 10);
            long long limit = size == 8 ? std::numeric_limits<long long>::max()
                                        : (1LL << (8 * size - 1)) - 1;
            if (errno || value > limit || value < -limit - 1) return H5I_INVALID_HID;
            bits = static_cast<std::uint64_t>(value);
        } else {
            if (text.empty() || text[0] == '-') return H5I_INVALID_HID;
            unsigned long long value = std::strtoull(text.c_str(), &parse_end, 10);
            if (errno || (size < 8 && value >> (8 * size))) return H5I_INVALID_HID;
            bits = value;
        }
        if (text.empty() || *parse_end != '\0') return H5I_INVALID_HID;
        unsigned char bytes[8];
        for (size_t k = 0; k < 8; ++k) bytes[k] = static_cast<unsigned char>(bits >> (8 * k));
        if (H5Tenum_insert(type.get(), member_name.c_str(), bytes) < 0) return H5I_INVALID_HID;
    }
    return type.release();
}

// Opaque types

// Return "opaque:<size>:<escaped tag>" for an opaque type.
inline std::string opaque_dtype_name(hid_t type) {
    char* tag = H5Tget_tag(type);
    if (!tag) return {};
    std::string name = "opaque:" + std::to_string(H5Tget_size(type)) + ":" +
                       escape_name(tag);
    H5free_memory(tag);
    return name;
}

// Create an owned opaque type from an "opaque:" dtype name, or return an invalid id.
inline hid_t opaque_type(const std::string& name) {
    if (name.rfind("opaque:", 0) != 0) return H5I_INVALID_HID;
    size_t colon = name.find(':', 7);
    if (colon == std::string::npos) return H5I_INVALID_HID;
    std::string digits = name.substr(7, colon - 7);
    std::string tag;
    if (digits.empty() || digits.size() > 9 ||
        digits.find_first_not_of("0123456789") != std::string::npos ||
        !unescape_name(name.substr(colon + 1), tag) || tag.size() >= H5T_OPAQUE_TAG_MAX)
        return H5I_INVALID_HID;
    size_t size = static_cast<size_t>(std::stoul(digits));
    if (!size) return H5I_INVALID_HID;
    ScopedId type(H5Tcreate(H5T_OPAQUE, size), H5Tclose);
    if (type.get() < 0 || H5Tset_tag(type.get(), tag.c_str()) < 0) return H5I_INVALID_HID;
    return type.release();
}

// String types

// Describe a string datatype named "vlen:utf8", "vlen:ascii", "|S<n>" or "|S<n>:utf8".
struct StringSpec {
    bool variable = false;
    size_t size = 0;
    H5T_cset_t cset = H5T_CSET_ASCII;
};

// Return true when name is a variable-length string dtype name.
inline bool is_vlen_string_name(const std::string& name) {
    return name == "vlen:utf8" || name == "vlen:ascii";
}

// Return true when name is a variable-length numeric array dtype name.
inline bool is_vlen_array_name(const std::string& name) {
    return name.rfind("vlen:", 0) == 0 && !is_vlen_string_name(name);
}

// Return true when name is any variable-length dtype name.
inline bool is_vlen_name(const std::string& name) { return name.rfind("vlen:", 0) == 0; }

// Return true when name is a fixed-length string dtype name.
inline bool is_fixed_string_name(const std::string& name) { return name.rfind("|S", 0) == 0; }

// Parse a string dtype name; return false for other names or invalid sizes.
inline bool parse_string_dtype(const std::string& name, StringSpec& spec) {
    if (is_vlen_string_name(name)) {
        spec.variable = true;
        spec.cset = name == "vlen:utf8" ? H5T_CSET_UTF8 : H5T_CSET_ASCII;
        return true;
    }
    if (!is_fixed_string_name(name)) return false;
    size_t end = name.find(':');
    std::string digits = name.substr(2, end == std::string::npos ? end : end - 2);
    if (digits.empty() || digits.size() > 9 ||
        digits.find_first_not_of("0123456789") != std::string::npos)
        return false;
    spec.size = static_cast<size_t>(std::stoul(digits));
    if (!spec.size) return false;
    if (end != std::string::npos && name.substr(end) != ":utf8") return false;
    spec.variable = false;
    spec.cset = end == std::string::npos ? H5T_CSET_ASCII : H5T_CSET_UTF8;
    return true;
}

// Parse a decimal size without accepting empty, malformed, or overflowing values.
inline bool parse_size(const std::string& text, size_t& value) {
    if (text.empty()) return false;
    value = 0;
    for (unsigned char digit : text) {
        if (digit < '0' || digit > '9') return false;
        size_t number = digit - '0';
        if (value > (std::numeric_limits<size_t>::max() - number) / 10) return false;
        value = value * 10 + number;
    }
    return true;
}

inline hid_t owned_type(const std::string& name);

// Create an owned compound type from a recursively escaped dtype description.
inline hid_t compound_type(const std::string& name) {
    if (name.rfind("compound:", 0) != 0) return H5I_INVALID_HID;
    size_t size_end = name.find(':', 9);
    size_t size = 0;
    if (size_end == std::string::npos || !parse_size(name.substr(9, size_end - 9), size) ||
        !size)
        return H5I_INVALID_HID;
    ScopedId type(H5Tcreate(H5T_COMPOUND, size), H5Tclose);
    if (type.get() < 0) return H5I_INVALID_HID;

    // Decode and insert each member while checking its complete encoded layout.
    std::string members = name.substr(size_end + 1);
    size_t start = 0;
    size_t count = 0;
    while (start < members.size()) {
        size_t end = members.find(',', start);
        std::string member = members.substr(
            start, end == std::string::npos ? end : end - start);
        start = end == std::string::npos ? members.size() : end + 1;
        size_t first = member.find(':');
        size_t second = first == std::string::npos ? first : member.find(':', first + 1);
        size_t offset = 0;
        std::string field_name, child_name;
        if (first == std::string::npos || second == std::string::npos ||
            !parse_size(member.substr(0, first), offset) ||
            !unescape_name(member.substr(first + 1, second - first - 1), field_name) ||
            !unescape_name(member.substr(second + 1), child_name) || field_name.empty())
            return H5I_INVALID_HID;
        ScopedId field(owned_type(child_name), H5Tclose);
        if (field.get() < 0 || offset > size ||
            H5Tget_size(field.get()) > size - offset ||
            H5Tinsert(type.get(), field_name.c_str(), offset, field.get()) < 0)
            return H5I_INVALID_HID;
        ++count;
    }
    return count ? type.release() : H5I_INVALID_HID;
}

// Create an owned HDF5 array datatype from its dimensions and element type.
inline hid_t array_type(const std::string& name) {
    if (name.rfind("array:", 0) != 0) return H5I_INVALID_HID;
    size_t rank_end = name.find(':', 6);
    size_t dims_end = rank_end == std::string::npos ? rank_end : name.find(':', rank_end + 1);
    size_t rank = 0;
    if (rank_end == std::string::npos || dims_end == std::string::npos ||
        !parse_size(name.substr(6, rank_end - 6), rank) || !rank ||
        rank > H5S_MAX_RANK)
        return H5I_INVALID_HID;
    std::vector<hsize_t> dims;
    std::string dimensions = name.substr(rank_end + 1, dims_end - rank_end - 1);
    size_t start = 0;
    while (start <= dimensions.size()) {
        size_t end = dimensions.find(',', start);
        size_t dimension = 0;
        if (!parse_size(dimensions.substr(
                            start, end == std::string::npos ? end : end - start), dimension) ||
            !dimension)
            return H5I_INVALID_HID;
        dims.push_back(dimension);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    std::string child_name;
    if (dims.size() != rank || !unescape_name(name.substr(dims_end + 1), child_name))
        return H5I_INVALID_HID;
    ScopedId base(owned_type(child_name), H5Tclose);
    if (base.get() < 0) return H5I_INVALID_HID;
    return H5Tarray_create2(base.get(), static_cast<unsigned>(rank), dims.data());
}

// Create an owned HDF5 string type; fixed strings are null-padded like h5py.
inline hid_t string_type(const StringSpec& spec) {
    ScopedId type(H5Tcopy(H5T_C_S1), H5Tclose);
    if (type.get() < 0) return H5I_INVALID_HID;
    if (H5Tset_size(type.get(), spec.variable ? H5T_VARIABLE : spec.size) < 0 ||
        H5Tset_strpad(type.get(), spec.variable ? H5T_STR_NULLTERM : H5T_STR_NULLPAD) < 0 ||
        H5Tset_cset(type.get(), spec.cset) < 0)
        return H5I_INVALID_HID;
    return type.release();
}

// Dispatch between datatypes and names

// Reject composite types that require HDF5-managed variable-length allocations.
inline bool contains_variable_storage(hid_t type) {
    H5T_class_t kind = H5Tget_class(type);
    if (kind == H5T_VLEN) return true;
    if (kind == H5T_STRING) return H5Tis_variable_str(type) > 0;
    if (kind == H5T_ARRAY || kind == H5T_ENUM) {
        ScopedId base(H5Tget_super(type), H5Tclose);
        return base.get() < 0 || contains_variable_storage(base.get());
    }
    if (kind == H5T_COMPOUND) {
        int count = H5Tget_nmembers(type);
        if (count < 0) return true;
        for (int i = 0; i < count; ++i) {
            ScopedId member(H5Tget_member_type(type, static_cast<unsigned>(i)), H5Tclose);
            if (member.get() < 0 || contains_variable_storage(member.get())) return true;
        }
    }
    return false;
}

// Return the dtype name for a supported HDF5 datatype, or an empty string.
inline std::string dtype_name(hid_t type, bool nested = false) {
    H5T_class_t type_class = H5Tget_class(type);
    if (type_class == H5T_COMPOUND) {
        std::string complex = complex_dtype(type);
        if (!complex.empty()) return complex;
    }
    if (type_class == H5T_ARRAY && !nested) return {};
    if (type_class == H5T_COMPOUND || type_class == H5T_ARRAY) {
        if (contains_variable_storage(type)) return {};
        if (type_class == H5T_COMPOUND) {
            int count = H5Tget_nmembers(type);
            size_t size = H5Tget_size(type);
            if (count <= 0 || !size) return {};
            std::string name = "compound:" + std::to_string(size) + ":";
            for (int i = 0; i < count; ++i) {
                char* raw_name = H5Tget_member_name(type, static_cast<unsigned>(i));
                ScopedId member(H5Tget_member_type(type, static_cast<unsigned>(i)), H5Tclose);
                if (!raw_name || member.get() < 0) {
                    if (raw_name) H5free_memory(raw_name);
                    return {};
                }
                std::string field_name(raw_name);
                H5free_memory(raw_name);
                std::string child_name = dtype_name(member.get(), true);
                if (child_name.empty()) return {};
                if (i) name.push_back(',');
                name += std::to_string(H5Tget_member_offset(type, static_cast<unsigned>(i)));
                name.push_back(':');
                name += escape_name(field_name);
                name.push_back(':');
                name += escape_name(child_name);
            }
            return name;
        }

        int rank = H5Tget_array_ndims(type);
        if (rank <= 0 || rank > H5S_MAX_RANK) return {};
        hsize_t dims[H5S_MAX_RANK];
        if (H5Tget_array_dims2(type, dims) < 0) return {};
        ScopedId base(H5Tget_super(type), H5Tclose);
        if (base.get() < 0) return {};
        std::string child_name = dtype_name(base.get(), true);
        if (child_name.empty()) return {};
        std::string name = "array:" + std::to_string(rank) + ":";
        for (int i = 0; i < rank; ++i) {
            if (i) name.push_back(',');
            name += std::to_string(dims[i]);
        }
        name.push_back(':');
        name += escape_name(child_name);
        return name;
    }
    switch (H5Tget_class(type)) {
        case H5T_STRING: {
            // Name strings by variability, size, and character set.
            htri_t variable = H5Tis_variable_str(type);
            if (variable < 0) return {};
            bool utf8 = H5Tget_cset(type) == H5T_CSET_UTF8;
            if (variable) return utf8 ? "vlen:utf8" : "vlen:ascii";
            size_t size = H5Tget_size(type);
            if (!size) return {};
            return "|S" + std::to_string(size) + (utf8 ? ":utf8" : "");
        }
        case H5T_ENUM:
            return enum_dtype_name(type);
        case H5T_COMPOUND:
        case H5T_COMPLEX:
            return complex_dtype(type);
        case H5T_OPAQUE:
            return opaque_dtype_name(type);
        case H5T_VLEN: {
            // Accept only variable-length arrays of numeric values.
            ScopedId super(H5Tget_super(type), H5Tclose);
            if (super.get() < 0) return {};
            std::string base = numeric_dtype(super.get());
            return base.empty() ? std::string() : "vlen:" + base;
        }
        default:
            return numeric_dtype(type);
    }
}

// Create an owned file datatype from a supported dtype name, or return an invalid id.
inline hid_t owned_type(const std::string& name) {
    if (name.rfind("compound:", 0) == 0) return compound_type(name);
    if (name.rfind("array:", 0) == 0) return array_type(name);
    StringSpec spec;
    if (parse_string_dtype(name, spec)) return string_type(spec);
    if (name.rfind("enum:", 0) == 0) return enum_type(name);
    if (name.rfind("opaque:", 0) == 0) return opaque_type(name);
    if (is_vlen_array_name(name)) {
        std::string base = name.substr(5);
        if (!is_vlen_base_name(base)) return H5I_INVALID_HID;
        ScopedId element(H5Tcopy(numeric_type(base)), H5Tclose);
        return element.get() < 0 ? H5I_INVALID_HID : H5Tvlen_create(element.get());
    }
    if (!complex_component_name(name).empty()) return complex_type(name);
    hid_t builtin = numeric_type(name);
    if (builtin >= 0) return H5Tcopy(builtin);
    return name == "|b1" ? boolean_type() : H5I_INVALID_HID;
}

// Create an owned memory type for name, matching the character set of a string file type.
inline hid_t memory_type(const std::string& name, hid_t file_type) {
    StringSpec spec;
    if (parse_string_dtype(name, spec)) {
        if (file_type >= 0 && H5Tget_class(file_type) == H5T_STRING)
            spec.cset = H5Tget_cset(file_type);
        return string_type(spec);
    }
    return H5Tcopy(file_type);
}

// Return true when input data can be written to a stored type after Python-side casting.
inline bool dtype_compatible(const std::string& stored, const std::string& input) {
    if (is_vlen_string_name(stored) || is_vlen_string_name(input))
        return is_vlen_string_name(stored) && is_vlen_string_name(input);
    if (is_fixed_string_name(stored) || is_fixed_string_name(input))
        return is_fixed_string_name(stored) && is_fixed_string_name(input);
    return stored == input;
}

// Typed transfers

// Own a zero-initialized element array for variable-length transfers.
// Declare after the type and space it references so it is destroyed first.
template <class T>
class VlenBuffer {
public:
    // Allocate count zeroed elements.
    explicit VlenBuffer(size_t count) : items_(count) {}
    VlenBuffer(const VlenBuffer&) = delete;
    VlenBuffer& operator=(const VlenBuffer&) = delete;
    // Reclaim HDF5-allocated memory when armed.
    ~VlenBuffer() noexcept {
        if (type_ >= 0 && !items_.empty())
            H5Treclaim(type_, space_, H5P_DEFAULT, items_.data());
    }
    // Return the element array passed to HDF5.
    T* data() noexcept { return items_.data(); }
    // Reclaim read allocations for the given memory type and dataspace on destruction.
    void arm(hid_t type, hid_t space) noexcept {
        type_ = type;
        space_ = space;
    }

private:
    std::vector<T> items_;
    hid_t type_ = H5I_INVALID_HID;
    hid_t space_ = H5I_INVALID_HID;
};

// Build the memory type and validate the element size for a transfer of points elements.
// Return Ok, or Unsupported with error filled when the type is unsupported or too large.
inline Status transfer_type(const std::string& name, hid_t file_type, size_t points,
                            ScopedId& memory, hid_t& memory_id, size_t& item_size,
                            std::string& error, bool borrow_file_type = false) {
    if (borrow_file_type && !is_vlen_string_name(name) && !is_fixed_string_name(name)) {
        memory_id = file_type;
    } else {
        memory.reset(memory_type(name, file_type));
        memory_id = memory.get();
    }
    if (memory_id < 0) {
        error = "unsupported datatype";
        return Status::Unsupported;
    }
    if (is_vlen_string_name(name)) item_size = sizeof(char*);
    else if (is_vlen_array_name(name)) item_size = sizeof(hvl_t);
    else item_size = H5Tget_size(memory_id);
    if (!item_size || points > std::numeric_limits<size_t>::max() / item_size) {
        error = "transfer is too large";
        return Status::Unsupported;
    }
    return Status::Ok;
}

// Read points elements through byte or variable-length buffers into out.
// space describes the memory elements; io(memory_type, buffer) returns an HDF5 status.
// Return Ok, Unsupported for unsupported data, or Failure for an HDF5 failure, with error filled.
template <class Io>
inline Status typed_read(hid_t file_type, hid_t space, size_t points, AttributeData& out,
                      std::string& error, Io io) {
    // Build the memory type from the stored type.
    const std::string& name = out.dtype;
    ScopedId memory(H5I_INVALID_HID, H5Tclose);
    hid_t memory_id = H5I_INVALID_HID;
    size_t item_size = 0;
    if (Status status = transfer_type(name, file_type, points, memory, memory_id, item_size, error,
                                      true);
        status != Status::Ok)
        return status;

    // Read fixed-size elements through the byte buffer.
    if (!is_vlen_name(name)) {
        const size_t size = points * item_size;
        unsigned char dummy = 0;
        unsigned char* target = &dummy;
        if (size) {
            out.block.reset(static_cast<unsigned char*>(std::malloc(size)), std::free);
            if (!out.block) throw std::bad_alloc();
            out.block_size = size;
            target = out.block.get();
        }
        if (io(memory_id, target) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Read variable-length strings and copy them before HDF5 memory is reclaimed.
    out.strings.clear();
    out.strings.reserve(points);
    if (is_vlen_string_name(name)) {
        VlenBuffer<char*> buffer(points);
        if (points) buffer.arm(memory_id, space);
        char* empty_buffer = nullptr;
        if (io(memory_id, points ? static_cast<void*>(buffer.data())
                                    : static_cast<void*>(&empty_buffer)) < 0) {
            error = error_message();
            return Status::Failure;
        }
        for (size_t i = 0; i < points; ++i)
            out.strings.emplace_back(buffer.data()[i] ? buffer.data()[i] : "");
        return Status::Ok;
    }

    // Read variable-length numeric arrays as raw bytes per element.
    const size_t base_size = numeric_item_size(name.substr(5));
    VlenBuffer<hvl_t> buffer(points);
    if (points) buffer.arm(memory_id, space);
    hvl_t empty_buffer{0, nullptr};
    if (io(memory_id, points ? static_cast<void*>(buffer.data())
                                : static_cast<void*>(&empty_buffer)) < 0) {
        error = error_message();
        return Status::Failure;
    }
    for (size_t i = 0; i < points; ++i) {
        const hvl_t& item = buffer.data()[i];
        if (item.len && item.p)
            out.strings.emplace_back(static_cast<const char*>(item.p), item.len * base_size);
        else
            out.strings.emplace_back();
    }
    return Status::Ok;
}

// Write points elements from byte or variable-length buffers in input.
// io(memory_type, buffer) returns an HDF5 status; return codes match typed_read.
template <class Io>
inline Status typed_write(hid_t file_type, size_t points, const AttributeData& input,
                       std::string& error, Io io) {
    // Build the memory type from the supplied data.
    ScopedId memory(H5I_INVALID_HID, H5Tclose);
    hid_t memory_id = H5I_INVALID_HID;
    size_t item_size = 0;
    if (Status status =
            transfer_type(input.dtype, file_type, points, memory, memory_id, item_size, error);
        status != Status::Ok)
        return status;

    // Write fixed-size elements from the byte buffer.
    if (!is_vlen_name(input.dtype)) {
        std::size_t size = 0;
        const unsigned char* bytes = input.bytes(size);
        if (size != points * item_size) {
            error = "data size does not match the selected elements";
            return Status::Unsupported;
        }
        unsigned char dummy = 0;
        const unsigned char* data = size ? bytes : &dummy;
        if (io(memory.get(), data) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }
    if (input.strings.size() != points) {
        error = "data size does not match the selected elements";
        return Status::Unsupported;
    }

    // Write variable-length strings through a pointer array.
    if (is_vlen_string_name(input.dtype)) {
        std::vector<const char*> pointers(points);
        for (size_t i = 0; i < points; ++i) pointers[i] = input.strings[i].c_str();
        const char* empty_buffer = nullptr;
        if (io(memory.get(), points ? static_cast<const void*>(pointers.data())
                                    : static_cast<const void*>(&empty_buffer)) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Write variable-length numeric arrays through length and pointer pairs.
    const size_t base_size = numeric_item_size(input.dtype.substr(5));
    std::vector<hvl_t> items(points);
    for (size_t i = 0; i < points; ++i) {
        if (input.strings[i].size() % base_size) {
            error = "variable-length element size is not a multiple of the item size";
            return Status::Unsupported;
        }
        items[i].len = input.strings[i].size() / base_size;
        items[i].p = const_cast<char*>(input.strings[i].data());
    }
    hvl_t empty_buffer{0, nullptr};
    if (io(memory.get(), points ? static_cast<const void*>(items.data())
                                : static_cast<const void*>(&empty_buffer)) < 0) {
        error = error_message();
        return Status::Failure;
    }
    return Status::Ok;
}

}  // namespace python_hdf5
