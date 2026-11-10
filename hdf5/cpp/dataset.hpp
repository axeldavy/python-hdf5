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

struct DatasetOptions {
    std::vector<std::uint64_t> chunks;
    std::vector<std::uint64_t> maxshape;
    bool has_maxshape = false;
    bool has_data = true;
    bool has_fill = false;
    AttributeData fill;
    int gzip_level = -1;
    bool shuffle = false;
    bool fletcher32 = false;
};

enum class SelectionKind { Index, Slice, Ellipsis };

struct SelectionPart {
    SelectionKind kind = SelectionKind::Slice;
    std::int64_t start = 0;
    std::int64_t stop = 0;
    std::int64_t step = 1;
    bool has_start = false;
    bool has_stop = false;
};

// Keep ordinary selection descriptions on the stack, bounded by the HDF5 rank limit.
struct SelectionRequest {
    std::array<SelectionPart, H5S_MAX_RANK + 1> parts;
    size_t size = 0;

    void add_index(std::int64_t value) {
        SelectionPart& part = parts[size++];
        part.kind = SelectionKind::Index;
        part.start = value;
    }

    void add_slice(bool has_start, std::int64_t start, bool has_stop, std::int64_t stop,
                   std::int64_t step) {
        SelectionPart& part = parts[size++];
        part.kind = SelectionKind::Slice;
        part.has_start = has_start;
        part.start = start;
        part.has_stop = has_stop;
        part.stop = stop;
        part.step = step;
    }

    void add_ellipsis() {
        parts[size++].kind = SelectionKind::Ellipsis;
    }
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

    // Flush this dataset or reload it from disk for SWMR readers.
    Status sync(bool refresh, std::string& error) {
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        std::unique_lock<std::mutex> attributes(*parent_->file_->attribute_mutex_);
        if ((refresh ? H5Drefresh(id_) : H5Dflush(id_)) < 0) return failure(error);
        return Status::Ok;
    }

    // Resolve a group or open a dataset while retaining the path-walk lock.
    static Status lookup(const std::shared_ptr<GroupHandle>& group, const char* path,
                         std::shared_ptr<GroupHandle>& group_output,
                         std::shared_ptr<DatasetHandle>& dataset_output, std::string& error) {
        const std::function<Status(const std::shared_ptr<GroupHandle>&, const char*, std::string&)>
            open_dataset = [&](const std::shared_ptr<GroupHandle>& parent, const char* name,
                               std::string& detail) {
                hid_t location = parent->id_ < 0 ? parent->file_->id_ : parent->id_;
                ScopedId dataset(H5Dopen2(location, name, H5P_DEFAULT), H5Dclose);
                if (dataset.get() < 0) return failure(detail);
                dataset_output = std::make_shared<DatasetHandle>(parent, dataset.get());
                dataset.release();
                return Status::Ok;
            };
        return group->resolve(path, false, false, group_output, error, false, false, &open_dataset);
    }

    // Open a hard-linked dataset or create one from the supplied data.
    // The new link is marked UTF-8 when utf8 is set, and ASCII otherwise.
    static Status open(const std::shared_ptr<GroupHandle>& parent, const char* name, bool create,
                    const AttributeData& input, const DatasetOptions& options,
                    std::shared_ptr<DatasetHandle>& output,
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
        std::vector<hsize_t> maxdims(options.maxshape.begin(), options.maxshape.end());
        ScopedId space(dims.empty() ? H5Screate(H5S_SCALAR)
                                   : H5Screate_simple(static_cast<int>(dims.size()), dims.data(),
                                                      options.has_maxshape ? maxdims.data()
                                                                           : nullptr), H5Sclose);
        ScopedId properties(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
        ScopedId creation(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
        if (type.get() < 0 || space.get() < 0 || properties.get() < 0 ||
            creation.get() < 0 ||
            H5Pset_char_encoding(properties.get(),
                                     utf8 ? H5T_CSET_UTF8 : H5T_CSET_ASCII) < 0)
            return failure(error);
        if (!options.chunks.empty()) {
            std::vector<hsize_t> chunks(options.chunks.begin(), options.chunks.end());
            if (H5Pset_chunk(creation.get(), static_cast<int>(chunks.size()), chunks.data()) < 0)
                return failure(error);
        }
        if (options.shuffle && H5Pset_shuffle(creation.get()) < 0) return failure(error);
        if (options.gzip_level >= 0) {
            unsigned int configuration = 0;
            if (H5Zfilter_avail(H5Z_FILTER_DEFLATE) <= 0 ||
                H5Zget_filter_info(H5Z_FILTER_DEFLATE, &configuration) < 0 ||
                !(configuration & H5Z_FILTER_CONFIG_ENCODE_ENABLED)) {
                error = "gzip compression filter is not available for encoding";
                return Status::Failure;
            }
            if (H5Pset_deflate(creation.get(),
                               static_cast<unsigned int>(options.gzip_level)) < 0)
                return failure(error);
        }
        if (options.fletcher32 && H5Pset_fletcher32(creation.get()) < 0) return failure(error);
        if (options.has_fill) {
            std::size_t size = 0;
            const unsigned char* bytes = options.fill.bytes(size);
            if (is_vlen_name(options.fill.dtype)) {
                error = "variable-length fill values are not supported";
                return Status::Unsupported;
            }
            if (!bytes || size != H5Tget_size(type.get())) {
                error = "invalid fill value";
                return Status::Unsupported;
            }
            if (H5Pset_fill_value(creation.get(), type.get(), bytes) < 0) return failure(error);
        }
        ScopedId dataset(H5Dcreate2(location, name, type.get(), space.get(), properties.get(),
                                    creation.get(), H5P_DEFAULT), H5Dclose);
        if (dataset.get() < 0) return failure(error);

        // Write the complete dataset and remove the link on write failure.
        if (!options.has_data) {
            output = std::make_shared<DatasetHandle>(parent, dataset.get());
            dataset.release();
            return Status::Ok;
        }
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

    // Resize a chunked dataset using the requested dimensions.
    Status resize(const std::vector<std::uint64_t>& shape, std::string& error) {
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (space.get() < 0) return failure(error);
        int rank = H5Sget_simple_extent_ndims(space.get());
        if (rank < 0) return failure(error);
        if (shape.size() != static_cast<size_t>(rank) || rank == 0) {
            error = "new shape must match the dataset rank";
            return Status::InvalidPath;
        }
        std::vector<hsize_t> dims(shape.begin(), shape.end());
        if (H5Dset_extent(id_, dims.data()) < 0) return failure(error);
        return Status::Ok;
    }

    // Read metadata and optionally copy all data into output.
    Status read(AttributeData& output, bool contents, std::string& error,
                bool include_shape = true) {
        // Lock the file and validate the stored type and dataspace.
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        ScopedId type(H5I_INVALID_HID, H5Tclose);
        if (Status status = dataset_type(type, contents, error); status != Status::Ok)
            return status;
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (space.get() < 0) return failure(error);
        output.dtype = dtype_;
        H5S_class_t space_type = H5Sget_simple_extent_type(space.get());
        if (space_type == H5S_NO_CLASS) return failure(error);
        if (space_type == H5S_NULL) {
            error = "null dataset dataspaces are not supported";
            return Status::Unsupported;
        }

        // Collect the shape and return early for metadata-only requests.
        if (include_shape || contents) {
            if (Status status = dataspace_shape(space.get(), output.shape, error);
                status != Status::Ok) return status;
        }
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
        ScopedId type(H5I_INVALID_HID, H5Tclose);
        if (Status status = dataset_type(type, true, error); status != Status::Ok) return status;
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (space.get() < 0) return failure(error);
        if (!dtype_compatible(dtype_, input.dtype)) {
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
        ScopedId type(H5I_INVALID_HID, H5Tclose);
        if (Status status = dataset_type(type, true, error); status != Status::Ok) return status;
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (space.get() < 0) return failure(error);
        if (dtype_ != selection.dtype ||
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
            hsize_t start[H5S_MAX_RANK], stride[H5S_MAX_RANK], count[H5S_MAX_RANK];
            bool empty = false;
            for (size_t i = 0; i < dims.size(); ++i) {
                start[i] = selection.start[i];
                stride[i] = selection.stride[i];
                count[i] = selection.count[i];
                empty = empty || count[i] == 0;
            }
            if ((empty ? H5Sselect_none(space.get())
                       : H5Sselect_hyperslab(space.get(), H5S_SELECT_SET, start,
                                             stride, count, nullptr)) < 0)
                return failure(error);
        }

        if (!writing) {
            data.dtype = selection.dtype;
            data.shape = selection.result_shape;
        }
        return selected_transfer(type.get(), space.get(), data, writing, error);
    }

    // Normalize ordinary selections and read from one fresh dataspace under one lock.
    Status read_selection(const SelectionRequest& request, AttributeData& data,
                          std::string& error) {
        std::unique_lock<std::mutex> lock(parent_->file_->mutex_);
        if (Status status = prepare(error); status != Status::Ok) return status;
        ScopedId type(H5I_INVALID_HID, H5Tclose);
        if (Status status = dataset_type(type, true, error); status != Status::Ok) return status;
        ScopedId space(H5Dget_space(id_), H5Sclose);
        if (space.get() < 0) return failure(error);
        H5S_class_t space_type = H5Sget_simple_extent_type(space.get());
        if (space_type == H5S_NO_CLASS) return failure(error);
        if (space_type == H5S_NULL) {
            error = "null dataset dataspaces are not supported";
            return Status::Unsupported;
        }
        hsize_t dims[H5S_MAX_RANK], start[H5S_MAX_RANK], stride[H5S_MAX_RANK];
        hsize_t count[H5S_MAX_RANK];
        int rank = H5Sget_simple_extent_dims(space.get(), dims, nullptr);
        if (rank < 0) return failure(error);
        size_t ellipses = 0;
        for (size_t i = 0; i < request.size; ++i)
            ellipses += request.parts[i].kind == SelectionKind::Ellipsis;
        if (ellipses > 1) {
            error = "only one ellipsis is allowed";
            return Status::InvalidSelection;
        }
        if (request.size - ellipses > static_cast<size_t>(rank)) {
            error = "too many indices for dataset";
            return Status::InvalidSelection;
        }
        for (int i = 0; i < rank; ++i)
            if (dims[i] > static_cast<hsize_t>(std::numeric_limits<std::int64_t>::max()))
                return Status::SelectionFallback;

        const size_t omitted = static_cast<size_t>(rank) - (request.size - ellipses);
        size_t axis = 0;
        bool empty = false;
        data.shape.clear();
        data.shape.reserve(static_cast<size_t>(rank));
        auto add_full_axis = [&]() {
            start[axis] = 0;
            stride[axis] = 1;
            count[axis] = dims[axis];
            data.shape.push_back(dims[axis]);
            empty = empty || count[axis] == 0;
            ++axis;
        };
        for (size_t i = 0; i < request.size; ++i) {
            const SelectionPart& part = request.parts[i];
            if (part.kind == SelectionKind::Ellipsis) {
                for (size_t n = 0; n < omitted; ++n) add_full_axis();
                continue;
            }
            const std::int64_t dim = static_cast<std::int64_t>(dims[axis]);
            if (part.kind == SelectionKind::Index) {
                std::int64_t index = part.start;
                if (index < 0 && index >= -dim) index += dim;
                if (index < 0 || index >= dim) {
                    error = "dataset index out of range";
                    return Status::InvalidSelection;
                }
                start[axis] = static_cast<hsize_t>(index);
                stride[axis] = count[axis] = 1;
            } else {
                auto bound = [dim](std::int64_t value) {
                    if (value < 0) return value < -dim ? std::int64_t(0) : value + dim;
                    return std::min(value, dim);
                };
                const std::int64_t first = part.has_start ? bound(part.start) : 0;
                const std::int64_t last = part.has_stop ? bound(part.stop) : dim;
                const std::int64_t length = last > first ? 1 + (last - first - 1) / part.step : 0;
                start[axis] = static_cast<hsize_t>(first);
                stride[axis] = length > 1 ? static_cast<hsize_t>(part.step) : 1;
                count[axis] = static_cast<hsize_t>(length);
                data.shape.push_back(static_cast<std::uint64_t>(length));
                empty = empty || length == 0;
            }
            ++axis;
        }
        while (axis < static_cast<size_t>(rank)) add_full_axis();
        if (rank && (empty ? H5Sselect_none(space.get())
                           : H5Sselect_hyperslab(space.get(), H5S_SELECT_SET, start, stride, count,
                                                 nullptr)) < 0)
            return failure(error);
        data.dtype = dtype_;
        return selected_transfer(type.get(), space.get(), data, false, error);
    }

private:
    // Dataset datatypes cannot change after creation; cache only their decoded description.
    Status dataset_type(ScopedId& type, bool need_type, std::string& error) {
        if (need_type || dtype_.empty()) {
            type.reset(H5Dget_type(id_));
            if (type.get() < 0) return failure(error);
            if (dtype_.empty()) {
                std::string name = dtype_name(type.get());
                if (name.empty()) {
                    error = "unsupported dataset datatype";
                    return Status::Unsupported;
                }
                dtype_ = std::move(name);
            }
        }
        return Status::Ok;
    }

    // Transfer selected points, including empty selections, through a flat memory space.
    Status selected_transfer(hid_t type, hid_t space, AttributeData& data, bool writing,
                             std::string& error) {
        hssize_t points = H5Sget_select_npoints(space);
        if (points < 0) return failure(error);
        hsize_t memory_size = points ? static_cast<hsize_t>(points) : 1;
        ScopedId memory(H5Screate_simple(1, &memory_size, nullptr), H5Sclose);
        if (memory.get() < 0 || (!points && H5Sselect_none(memory.get()) < 0))
            return failure(error);
        if (writing) {
            return typed_write(type, static_cast<size_t>(points), data, error,
                               [&](hid_t memory_type_id, const void* buffer) {
                                   return H5Dwrite(id_, memory_type_id, memory.get(),
                                                   space, H5P_DEFAULT, buffer);
                               });
        }
        return typed_read(type, memory.get(), static_cast<size_t>(points), data, error,
                          [&](hid_t memory_type_id, void* buffer) {
                              return H5Dread(id_, memory_type_id, memory.get(), space,
                                             H5P_DEFAULT, buffer);
                          });
    }

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
    std::string dtype_;
};

}  // namespace python_hdf5
