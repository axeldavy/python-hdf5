#pragma once

#include <atomic>
#include <shared_mutex>
#include <unistd.h>
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
        if (is_reserved(name)) return Status::NotFound;
        std::shared_lock<std::shared_mutex> swap_lock(swap_mutex());
        hid_t object = object_id();
        htri_t exists = H5Aexists(object, name);
        if (exists < 0) return failure(error);
        if (!exists) return Status::NotFound;
        ScopedId attr(H5Aopen(object, name, H5P_DEFAULT), H5Aclose);
        if (attr.get() < 0) return failure(error);

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
        if (space_type == H5S_NULL) {
            error = "null attribute dataspaces are not supported";
            return Status::Unsupported;
        }

        // Determine the shape and check the transfer size.
        int rank = H5Sget_simple_extent_ndims(space.get());
        if (rank < 0) return failure(error);
        std::vector<hsize_t> dims(static_cast<size_t>(rank));
        if (rank && H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr) < 0)
            return failure(error);
        out.shape.assign(dims.begin(), dims.end());
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

    // Replace an attribute with the supplied data. The new value is written under a
    // reserved temporary name and renamed over the old one, so a failure keeps the old value.
    // The attribute name is marked UTF-8 when utf8 is set, and ASCII otherwise.
    Status write(const char* name, const AttributeData& input, std::string& error,
              bool utf8 = false) {
        // Validate the type and rank before replacing an existing attribute.
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
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

        // Write the new value under the temporary name, discarding any crash leftover.
        if (is_reserved(name)) {
            error = "attribute names starting with '.python-hdf5-' are reserved";
            return Status::Unsupported;
        }
        hid_t object = object_id();
        htri_t exists = H5Aexists(object, name);
        if (exists < 0) return failure(error);
        std::string temporary = temporary_name();
        {
            ScopedId attr(H5Acreate2(object, temporary.c_str(), type.get(), space.get(),
                                     properties.get(), H5P_DEFAULT), H5Aclose);
            if (attr.get() < 0) return failure(error);
            Status status = typed_write(type.get(), static_cast<size_t>(points), input, error,
                                        [&](hid_t memory, const void* buffer) {
                                            return H5Awrite(attr.get(), memory, buffer);
                                        });
            if (status != Status::Ok) {
                attr.reset(H5I_INVALID_HID);
                H5Adelete(object, temporary.c_str());
                return status;
            }
        }

        // Swap the new attribute in, excluding readers on other file objects of the same file.
        std::unique_lock<std::shared_mutex> swap_lock(swap_mutex());
        if ((exists && H5Adelete(object, name) < 0) ||
            H5Arename(object, temporary.c_str(), name) < 0) {
            error = error_message();
            H5Adelete(object, temporary.c_str());
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Delete a named attribute, returning Status::NotFound when it is absent.
    Status remove(const char* name, std::string& error) {
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        if (is_reserved(name)) return Status::NotFound;
        std::shared_lock<std::shared_mutex> swap_lock(swap_mutex());
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
        if (is_reserved(name)) return Status::NotFound;
        std::shared_lock<std::shared_mutex> swap_lock(swap_mutex());
        htri_t exists = H5Aexists(object_id(), name);
        if (exists < 0) return failure(error);
        return exists ? Status::Ok : Status::NotFound;
    }

    // Collect attribute names in HDF5 name order.
    Status names(std::vector<std::string>& out, std::string& error) {
        std::unique_lock<std::mutex> lock(group_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        hsize_t index = 0;
        Collector collector{&out, nullptr};
        herr_t result = H5Aiterate2(object_id(), H5_INDEX_NAME, H5_ITER_INC, &index,
                                    collect, &collector);
        if (collector.exception) std::rethrow_exception(collector.exception);
        if (result < 0) return failure(error);
        return Status::Ok;
    }

private:
    // Return the process-wide mutex making the delete and rename swap invisible to readers.
    static std::shared_mutex& swap_mutex() {
        static std::shared_mutex mutex;
        return mutex;
    }

    // Return a process-unique reserved name, since several file objects may share one file.
    static std::string temporary_name() {
        static std::atomic<unsigned long long> counter{0};
        return ".python-hdf5-" + std::to_string(static_cast<unsigned long long>(getpid())) +
               "-" + std::to_string(counter.fetch_add(1));
    }

    // Return whether name is reserved for internal use.
    static bool is_reserved(const char* name) {
        return std::string(name).rfind(".python-hdf5-", 0) == 0;
    }

    struct Collector {
        std::vector<std::string>* names;
        std::exception_ptr exception;
    };
    // Append a name and defer any C++ exception until after HDF5 iteration.
    static herr_t collect(hid_t, const char* name, const H5A_info_t*, void* data) noexcept {
        auto* collector = static_cast<Collector*>(data);
        try {
            if (!is_reserved(name)) collector->names->emplace_back(name);
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
