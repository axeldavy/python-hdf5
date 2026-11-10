#pragma once

#include "datatype.hpp"
#include "group.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace python_hdf5 {

// Manage attributes while retaining the owning group or dataset.
class AttributeHandle {
public:
    // Bind to the file root or the supplied group.
    explicit AttributeHandle(std::shared_ptr<GroupHandle> group) noexcept
        : group_(std::move(group)) {}
    // Bind to an object whose identifier remains owned by owner.
    AttributeHandle(std::shared_ptr<GroupHandle> group, std::shared_ptr<void> owner,
                    hid_t object) noexcept
        : group_(std::move(group)), owner_(std::move(owner)), object_(object) {}

    // Read a supported numeric, boolean, or string attribute into owned output storage.
    Status read(const char* name, AttributeData& out, std::string& error) {
        // Lock the file and open the requested attribute.
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        std::unique_lock<std::mutex> attribute_lock(*group_->file_->attribute_mutex_);
        hid_t object = object_id();
        ScopedId attr(H5Aopen(object, name, H5P_DEFAULT), H5Aclose);
        if (attr.get() < 0) {
            // Distinguish a missing attribute from an HDF5 error without an extra
            // existence check on successful reads.
            error = error_message();
            htri_t exists = H5Aexists(object, name);
            if (exists < 0) return failure(error);
            return exists ? Status::Failure : Status::NotFound;
        }

        // Validate the attribute type and dataspace.
        ScopedId type(H5Aget_type(attr.get()), H5Tclose);
        ScopedId space(H5Aget_space(attr.get()), H5Sclose);
        if (type.get() < 0 || space.get() < 0) return failure(error);
        out.dtype = dtype_name(type.get());
        if (out.dtype.empty()) {
            error = "unsupported attribute datatype";
            return Status::Unsupported;
        }
        H5S_class_t space_type = H5Sget_simple_extent_type(space.get());
        if (space_type == H5S_NO_CLASS) return failure(error);
        if (space_type == H5S_NULL) {
            error = "null attribute dataspaces are not supported";
            return Status::Unsupported;
        }

        // Determine the shape and check the transfer size.
        if (Status status = dataspace_shape(space.get(), out.shape, error);
            status != Status::Ok) return status;
        hssize_t points = H5Sget_simple_extent_npoints(space.get());
        if (points < 0) return failure(error);

        // Read into owned storage, including zero-element attributes.
        Status status = typed_read(type.get(), space.get(), static_cast<size_t>(points), out, error,
                                   [&](hid_t memory, void* buffer) {
                                       return H5Aread(attr.get(), memory, buffer);
                                   });
        if (status != Status::Ok) return status;
        return Status::Ok;
    }

    // Replace an attribute with the supplied data; replacement is not atomic.
    // The attribute name is marked UTF-8 when utf8 is set, and ASCII otherwise.
    Status write(const char* name, const AttributeData& input, std::string& error,
              bool utf8 = false) {
        // Validate the type and rank before replacing an existing attribute.
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        std::unique_lock<std::mutex> attribute_lock(*group_->file_->attribute_mutex_);
        ScopedId type(owned_type(input.dtype), H5Tclose);
        if (type.get() < 0) {
            error = "unsupported attribute datatype";
            return Status::Unsupported;
        }
        if (input.shape.size() > H5S_MAX_RANK) {
            error = "attribute rank exceeds HDF5 limit";
            return Status::Unsupported;
        }

        // Construct the dataspace and attribute creation properties.
        std::vector<hsize_t> dims(input.shape.begin(), input.shape.end());
        ScopedId space(dims.empty() ? H5Screate(H5S_SCALAR)
                                    : H5Screate_simple(static_cast<int>(dims.size()),
                                                       dims.data(), nullptr), H5Sclose);
        ScopedId properties(H5Pcreate(H5P_ATTRIBUTE_CREATE), H5Pclose);
        if (space.get() < 0 || properties.get() < 0 ||
            H5Pset_char_encoding(properties.get(),
                                     utf8 ? H5T_CSET_UTF8 : H5T_CSET_ASCII) < 0)
            return failure(error);
        hssize_t points = H5Sget_simple_extent_npoints(space.get());
        if (points < 0) return failure(error);

        // Replace the existing attribute and write the supplied data.
        hid_t object = object_id();
        htri_t exists = H5Aexists(object, name);
        if (exists < 0) return failure(error);
        if (exists && H5Adelete(object, name) < 0) return failure(error);
        ScopedId attr(H5Acreate2(object, name, type.get(), space.get(), properties.get(),
                                 H5P_DEFAULT), H5Aclose);
        if (attr.get() < 0) return failure(error);
        return typed_write(type.get(), static_cast<size_t>(points), input, error,
                           [&](hid_t memory, const void* buffer) {
                               return H5Awrite(attr.get(), memory, buffer);
                           });
    }

    // Delete a named attribute, returning Status::NotFound when it is absent.
    Status remove(const char* name, std::string& error) {
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        std::unique_lock<std::mutex> attribute_lock(*group_->file_->attribute_mutex_);
        htri_t exists = H5Aexists(object_id(), name);
        if (exists < 0) return failure(error);
        if (!exists) return Status::NotFound;
        if (H5Adelete(object_id(), name) < 0) return failure(error);
        return Status::Ok;
    }

    // Check attribute existence, returning Status::NotFound when it is absent.
    Status contains(const char* name, std::string& error) {
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        std::unique_lock<std::mutex> attribute_lock(*group_->file_->attribute_mutex_);
        htri_t exists = H5Aexists(object_id(), name);
        if (exists < 0) return failure(error);
        return exists ? Status::Ok : Status::NotFound;
    }

    // Count attributes without collecting their names.
    Status count(std::uint64_t& out, std::string& error) {
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        std::unique_lock<std::mutex> attribute_lock(*group_->file_->attribute_mutex_);
        H5O_info2_t info;
        if (H5Oget_info3(object_id(), &info, H5O_INFO_NUM_ATTRS) < 0) return failure(error);
        out = info.num_attrs;
        return Status::Ok;
    }

    // Collect attribute names in HDF5 name order.
    Status names(std::vector<std::string>& out, std::string& error) {
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        std::unique_lock<std::mutex> attribute_lock(*group_->file_->attribute_mutex_);
        hsize_t index = 0;
        Collector collector{&out, nullptr};
        herr_t result = H5Aiterate2(object_id(), H5_INDEX_NAME, H5_ITER_INC, &index,
                                    collect, &collector);
        if (collector.exception) std::rethrow_exception(collector.exception);
        if (result < 0) return failure(error);
        return Status::Ok;
    }

private:
    struct Collector {
        std::vector<std::string>* names;
        std::exception_ptr exception;
    };
    // Append a name and defer any C++ exception until after HDF5 iteration.
    static herr_t collect(hid_t, const char* name, const H5A_info_t*, void* data) noexcept {
        auto* collector = static_cast<Collector*>(data);
        try {
            collector->names->emplace_back(name);
            return 0;
        } catch (...) {
            collector->exception = std::current_exception();
            return -1;
        }
    }
    // Check file state and configure error reporting with the file lock held.
    Status prepare(std::string& error) const {
        if (group_->file_->id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) return failure(error);
        return Status::Ok;
    }
    // Capture the current HDF5 error stack and return failure status.
    static Status failure(std::string& error) {
        error = error_message();
        return Status::Failure;
    }
    // Borrow the owner identifier while the file lock is held.
    hid_t object_id() const noexcept {
        if (owner_) return object_;
        return group_->id_ < 0 ? group_->file_->id_ : group_->id_;
    }
    std::shared_ptr<GroupHandle> group_;
    std::shared_ptr<void> owner_;
    hid_t object_ = H5I_INVALID_HID;
};

}  // namespace python_hdf5
