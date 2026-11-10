#pragma once

#include "attribute.hpp"

namespace python_hdf5 {

// Describe a normalized hyperslab and the dataset shape used to construct it.
struct DatasetSelection {
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> start;
    std::vector<std::uint64_t> stride;
    std::vector<std::uint64_t> count;
    std::vector<std::uint64_t> result_shape;
    std::string dtype;
};

// Own a dataset identifier and keep its file and parent group alive.
class DatasetHandle : public std::enable_shared_from_this<DatasetHandle> {
public:
    // Adopt an open dataset identifier and retain its parent.
    DatasetHandle(std::shared_ptr<GroupHandle> parent, hid_t id) noexcept
        : parent_(std::move(parent)), id_(id) {}
    DatasetHandle(const DatasetHandle&) = delete;
    DatasetHandle& operator=(const DatasetHandle&) = delete;

    // Close the dataset unless strong file close already invalidated it.
    ~DatasetHandle() noexcept {
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (parent_->file_->id_ >= 0 && H5Dclose(id_) < 0)
            std::fputs("python-hdf5: failed to close dataset during cleanup\n", stderr);
    }

    // Return an attribute manager that shares dataset ownership.
    std::shared_ptr<AttributeHandle> attributes() {
        return std::make_shared<AttributeHandle>(parent_, shared_from_this(), id_);
    }

    // Return the absolute path used to open this dataset.
    Status name(std::string& output, std::string& error) {
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (parent_->file_->id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) return failure(error);
        ssize_t length = H5Iget_name(id_, nullptr, 0);
        if (length < 0) return failure(error);
        output.resize(static_cast<size_t>(length) + 1);
        ssize_t written = H5Iget_name(id_, output.data(), output.size());
        if (written < 0) return failure(error);
        output.resize(static_cast<size_t>(written));
        return Status::Ok;
    }

    // Return the group that contains this dataset.
    std::shared_ptr<GroupHandle> parent() const noexcept { return parent_; }

    // Return the file holder shared by this dataset.
    std::shared_ptr<FileHandle> file_handle() const noexcept { return parent_->file_; }

    // Open a hard-linked dataset or create one from the supplied data.
    // The new link is marked UTF-8 when utf8 is set, and ASCII otherwise.
    static Status open(const std::shared_ptr<GroupHandle>& parent, const char* name, bool create,
                    const AttributeData& input, std::shared_ptr<DatasetHandle>& output,
                    std::string& error, bool utf8 = false) {
        // Lock the file and inspect the target link.
        std::unique_lock<std::mutex> lock(parent->file_->mutex_);
        if (parent->file_->id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) return failure(error);
        hid_t location = parent->id_ < 0 ? parent->file_->id_ : parent->id_;
        // Open existing hard-linked datasets without following unsupported links.
        if (!create) {
            H5L_info2_t link;
            if (H5Lget_info2(location, name, &link, H5P_DEFAULT) < 0) {
                error = error_message();
                htri_t exists = H5Lexists(location, name, H5P_DEFAULT);
                if (exists < 0) return failure(error);
                return exists ? Status::Failure : Status::NotFound;
            }
            if (link.type != H5L_TYPE_HARD) {
                error = "soft, external, and user-defined links are not supported";
                return Status::Unsupported;
            }
            ScopedId dataset(H5Dopen2(location, name, H5P_DEFAULT), H5Dclose);
            if (dataset.get() < 0) {
                error = error_message();
                H5O_info2_t object;
                if (H5Oget_info_by_name3(location, name, &object, H5O_INFO_BASIC, H5P_DEFAULT) < 0)
                    return failure(error);
                if (object.type != H5O_TYPE_DATASET) {
                    error = "path does not refer to a dataset";
                    return Status::Unsupported;
                }
                return Status::Failure;
            }
            output = std::make_shared<DatasetHandle>(parent, dataset.get());
            dataset.release();
            return Status::Ok;
        }

        // Validate the type and dimensions before creating the dataset.
        htri_t exists = H5Lexists(location, name, H5P_DEFAULT);
        if (exists < 0) return failure(error);
        if (exists) {
            error = "a link with this name already exists";
            return Status::Failure;
        }
        if (input.shape.size() > H5S_MAX_RANK) {
            error = "dataset rank exceeds HDF5 limit";
            return Status::Unsupported;
        }
        ScopedId type(owned_type(input.dtype), H5Tclose);
        if (type.get() < 0) {
            error = "unsupported dataset datatype";
            return Status::Unsupported;
        }

        // Create the dataspace, datatype, and link.
        std::vector<hsize_t> dims(input.shape.begin(), input.shape.end());
        ScopedId space(dims.empty() ? H5Screate(H5S_SCALAR)
                                   : H5Screate_simple(static_cast<int>(dims.size()), dims.data(),
                                                      nullptr), H5Sclose);
        ScopedId properties(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
        if (type.get() < 0 || space.get() < 0 || properties.get() < 0 ||
            H5Pset_char_encoding(properties.get(),
                                     utf8 ? H5T_CSET_UTF8 : H5T_CSET_ASCII) < 0)
            return failure(error);
        ScopedId dataset(H5Dcreate2(location, name, type.get(), space.get(), properties.get(),
                                    H5P_DEFAULT, H5P_DEFAULT), H5Dclose);
        if (dataset.get() < 0) return failure(error);

        // Write the complete dataset and remove the link on write failure.
        hssize_t points = H5Sget_simple_extent_npoints(space.get());
        if (points < 0) return failure(error);
        Status status = typed_write(type.get(), static_cast<size_t>(points), input, error,
                                 [&](hid_t memory, const void* buffer) {
                                     return H5Dwrite(dataset.get(), memory, H5S_ALL, H5S_ALL,
                                                     H5P_DEFAULT, buffer);
                                 });
        if (status != Status::Ok) {
            if (H5Ldelete(location, name, H5P_DEFAULT) < 0)
                error += "; failed to remove the partially written dataset: " + error_message();
            return status;
        }

        // Transfer the open identifier to shared ownership.
        output = std::make_shared<DatasetHandle>(parent, dataset.get());
        dataset.release();
        return Status::Ok;
    }

    // Read metadata and optionally copy all data into output.
    Status read(AttributeData& output, bool contents, std::string& error) {
        // Lock the file and validate the stored type and dataspace.
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        ScopedId type(H5Dget_type(id_), H5Tclose);
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (type.get() < 0 || space.get() < 0) return failure(error);
        output.dtype = dtype_name(type.get());
        if (output.dtype.empty()) {
            error = "unsupported dataset datatype";
            return Status::Unsupported;
        }
        H5S_class_t space_type = H5Sget_simple_extent_type(space.get());
        if (space_type == H5S_NO_CLASS) return failure(error);
        if (space_type == H5S_NULL) {
            error = "null dataset dataspaces are not supported";
            return Status::Unsupported;
        }

        // Collect the shape and return early for metadata-only requests.
        if (Status status = dataspace_shape(space.get(), output.shape, error);
            status != Status::Ok) return status;
        if (!contents) return Status::Ok;

        // Check the transfer size and read into owned storage.
        hssize_t points = H5Sget_simple_extent_npoints(space.get());
        if (points < 0) return failure(error);
        return typed_read(type.get(), space.get(), static_cast<size_t>(points), output, error,
                          [&](hid_t memory, void* buffer) {
                              return H5Dread(id_, memory, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                             buffer);
                          });
    }

    // Replace all values after checking that dtype and shape still match.
    Status write(const AttributeData& input, std::string& error) {
        // Recheck the stored type and dimensions while holding the file lock.
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        ScopedId type(H5Dget_type(id_), H5Tclose);
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (type.get() < 0 || space.get() < 0) return failure(error);
        if (!dtype_compatible(dtype_name(type.get()), input.dtype)) {
            error = "dataset datatype changed during write";
            return Status::Unsupported;
        }
        int rank = H5Sget_simple_extent_ndims(space.get());
        if (rank < 0) return failure(error);
        std::vector<hsize_t> dims(static_cast<size_t>(rank));
        if (rank && H5Sget_simple_extent_dims(space.get(), dims.data(), nullptr) < 0)
            return failure(error);
        if (std::vector<std::uint64_t>(dims.begin(), dims.end()) != input.shape) {
            error = "dataset shape changed during write";
            return Status::Unsupported;
        }

        // Write the complete input, using a valid pointer for empty arrays.
        hssize_t points = H5Sget_simple_extent_npoints(space.get());
        if (points < 0) return failure(error);
        return typed_write(type.get(), static_cast<size_t>(points), input, error,
                           [&](hid_t memory, const void* buffer) {
                               return H5Dwrite(id_, memory, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                                               buffer);
                           });
    }

    // Transfer a normalized selection after revalidating its metadata under the file lock.
    Status transfer(const DatasetSelection& selection, AttributeData& data, bool writing,
                 std::string& error) {
        // Keep close, resize, and the actual transfer mutually exclusive.
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        ScopedId type(H5Dget_type(id_), H5Tclose);
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (type.get() < 0 || space.get() < 0) return failure(error);
        if (dtype_name(type.get()) != selection.dtype ||
            (writing && !dtype_compatible(selection.dtype, data.dtype))) {
            error = "dataset datatype changed during selection";
            return Status::Unsupported;
        }
        std::vector<std::uint64_t> dims;
        if (Status status = dataspace_shape(space.get(), dims, error);
            status != Status::Ok) return status;
        if (dims != selection.shape) {
            error = "dataset shape changed during selection";
            return Status::Unsupported;
        }

        // Select file coordinates or an empty selection for zero-length slices.
        if (!dims.empty()) {
            std::vector<hsize_t> start(selection.start.begin(), selection.start.end());
            std::vector<hsize_t> stride(selection.stride.begin(), selection.stride.end());
            std::vector<hsize_t> count(selection.count.begin(), selection.count.end());
            bool empty = false;
            for (hsize_t n : count) empty = empty || n == 0;
            if ((empty ? H5Sselect_none(space.get())
                       : H5Sselect_hyperslab(space.get(), H5S_SELECT_SET, start.data(),
                                             stride.data(), count.data(), nullptr)) < 0)
                return failure(error);
        }

        // Create a flat memory space with the same selected element count.
        hssize_t points = H5Sget_select_npoints(space.get());
        if (points < 0) return failure(error);
        hsize_t memory_size = points ? static_cast<hsize_t>(points) : 1;
        ScopedId memory(H5Screate_simple(1, &memory_size, nullptr), H5Sclose);
        if (memory.get() < 0 || (!points && H5Sselect_none(memory.get()) < 0))
            return failure(error);

        // Read or write even empty selections so access errors are not hidden.
        if (writing) {
            return typed_write(type.get(), static_cast<size_t>(points), data, error,
                               [&](hid_t memory_type_id, const void* buffer) {
                                   return H5Dwrite(id_, memory_type_id, memory.get(),
                                                   space.get(), H5P_DEFAULT, buffer);
                               });
        }
        data.dtype = selection.dtype;
        data.shape = selection.result_shape;
        return typed_read(type.get(), memory.get(), static_cast<size_t>(points), data, error,
                          [&](hid_t memory_type_id, void* buffer) {
                              return H5Dread(id_, memory_type_id, memory.get(), space.get(),
                                             H5P_DEFAULT, buffer);
                          });
    }

private:
    // Check file state and configure error reporting with the file lock held.
    Status prepare(std::string& error) const {
        if (parent_->file_->id_ < 0) {
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
    std::shared_ptr<GroupHandle> parent_;
    hid_t id_;
};

}  // namespace python_hdf5
