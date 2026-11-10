#pragma once

#include "file.hpp"

#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace python_hdf5 {

// Keep a group identifier and its file alive independently of Python GC order.
class GroupHandle {
public:
    // Adopt id; an invalid id denotes the file root.
    GroupHandle(std::shared_ptr<FileHandle> file, hid_t id) noexcept
        : file_(std::move(file)), id_(id) {}
    GroupHandle(const GroupHandle&) = delete;
    GroupHandle& operator=(const GroupHandle&) = delete;

    // Close the group unless strong file close has already invalidated it.
    ~GroupHandle() noexcept {
        std::unique_lock<std::mutex> lock(file_->mutex_);
        if (id_ >= 0 && file_->id_ >= 0 && H5Gclose(id_) < 0) {
            std::fputs("python-hdf5: failed to close an HDF5 group during cleanup\n", stderr);
        }
    }

    // Create a root-group view that shares file ownership.
    static std::shared_ptr<GroupHandle> root(const std::shared_ptr<FileHandle>& file) {
        return std::make_shared<GroupHandle>(file, H5I_INVALID_HID);
    }

    // Resolve or create a path; Status::NotFound means missing and Status::IsDataset means dataset.
    // Created links are marked UTF-8 when utf8 is set, and ASCII otherwise.
    Status resolve(const char* path, bool create, bool membership,
                std::shared_ptr<GroupHandle>& output, std::string& error, bool utf8 = false,
                bool require = false) {
        // Guard file state and configure the thread's HDF5 error handler.
        std::unique_lock<std::mutex> lock(file_->mutex_);
        if (file_->id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) {
            error = error_message();
            return Status::Failure;
        }

        // Normalize path components without permitting parent traversal.
        const std::string name(path);
        std::vector<std::string> components;
        for (size_t start = 0; start < name.size();) {
            size_t end = name.find('/', start);
            if (end == std::string::npos) end = name.size();
            std::string part = name.substr(start, end - start);
            if (part == "..") {
                error = "parent path components are not supported";
                return Status::InvalidPath;
            }
            if (!part.empty() && part != ".") components.push_back(std::move(part));
            start = end + 1;
        }
        if (name.empty()) {
            error = "group path must not be empty";
            return Status::InvalidPath;
        }
        if (create && components.empty() && !require) {
            error = "cannot create the root group";
            return Status::InvalidPath;
        }

        // Walk hard links and retain intermediate group identifiers.
        hid_t current = name.front() == '/' || id_ < 0 ? file_->id_ : id_;
        std::vector<std::unique_ptr<ScopedId>> opened;
        for (size_t index = 0; index < components.size(); ++index) {
            const std::string& part = components[index];
            const bool last = index + 1 == components.size();
            H5L_info2_t info;
            htri_t exists;
            if (!create) {
                if (H5Lget_info2(current, part.c_str(), &info, H5P_DEFAULT) >= 0) {
                    exists = 1;
                } else {
                    error = error_message();
                    exists = H5Lexists(current, part.c_str(), H5P_DEFAULT);
                    if (exists < 0) {
                        error = error_message();
                        return Status::Failure;
                    }
                    return exists ? Status::Failure : Status::NotFound;
                }
            } else {
                exists = H5Lexists(current, part.c_str(), H5P_DEFAULT);
                if (exists < 0) {
                    error = error_message();
                    return Status::Failure;
                }
                if (exists && H5Lget_info2(current, part.c_str(), &info, H5P_DEFAULT) < 0) {
                    error = error_message();
                    return Status::Failure;
                }
            }
            if (exists) {
                if (info.type != H5L_TYPE_HARD) {
                    error = "soft, external, and user-defined links are not supported";
                    return Status::Unsupported;
                }
                if (membership && last) return Status::Ok;
                if (create && last && !require) {
                    error = "a link with this name already exists";
                    return Status::Failure;
                }
                H5O_info2_t object;
                if (H5Oget_info_by_name3(current, part.c_str(), &object,
                                         H5O_INFO_BASIC, H5P_DEFAULT) < 0) {
                    error = error_message();
                    return Status::Failure;
                }
                if (object.type != H5O_TYPE_GROUP) {
                    if (membership) return Status::NotFound;
                    if (last && !create && object.type == H5O_TYPE_DATASET) {
                        // Retain the already resolved parent for the dataset open.
                        if (!opened.empty()) {
                            output = std::make_shared<GroupHandle>(file_, opened.back()->get());
                            opened.back()->release();
                        } else {
                            ScopedId parent(H5Gopen2(current, ".", H5P_DEFAULT), H5Gclose);
                            if (parent.get() < 0) {
                                error = error_message();
                                return Status::Failure;
                            }
                            output = std::make_shared<GroupHandle>(file_, parent.get());
                            parent.release();
                        }
                        return Status::IsDataset;
                    }
                    error = "path refers to an object that is not a group";
                    return Status::Unsupported;
                }
            }
            if (!exists) {
                ScopedId properties(H5Pcreate(H5P_LINK_CREATE), H5Pclose);
                if (properties.get() < 0 ||
                    H5Pset_char_encoding(properties.get(),
                                         utf8 ? H5T_CSET_UTF8 : H5T_CSET_ASCII) < 0) {
                    error = error_message();
                    return Status::Failure;
                }
                ScopedId group(H5Gcreate2(current, part.c_str(), properties.get(),
                                          H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
                if (group.get() < 0) {
                    error = error_message();
                    return Status::Failure;
                }
                auto saved = std::make_unique<ScopedId>(group.get(), H5Gclose);
                group.release();
                opened.push_back(std::move(saved));
            } else {
                ScopedId group(H5Gopen2(current, part.c_str(), H5P_DEFAULT), H5Gclose);
                if (group.get() < 0) {
                    error = error_message();
                    return Status::Failure;
                }
                auto saved = std::make_unique<ScopedId>(group.get(), H5Gclose);
                group.release();
                opened.push_back(std::move(saved));
            }
            current = opened.back()->get();
        }
        // Transfer the final group identifier to a shared holder.
        if (membership) return Status::Ok;
        if (components.empty()) {
            ScopedId group(H5Gopen2(current, ".", H5P_DEFAULT), H5Gclose);
            if (group.get() < 0) {
                error = error_message();
                return Status::Failure;
            }
            output = std::make_shared<GroupHandle>(file_, group.get());
            group.release();
        } else {
            output = std::make_shared<GroupHandle>(file_, opened.back()->get());
            opened.back()->release();
        }
        return Status::Ok;
    }

    // Collect link names in HDF5 name order while the file remains locked.
    Status names(std::vector<std::string>& output, std::string& error) {
        std::unique_lock<std::mutex> lock(file_->mutex_);
        if (file_->id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) {
            error = error_message();
            return Status::Failure;
        }

        // Preserve callback exceptions across the C API boundary.
        hsize_t index = 0;
        NameCollector collector{&output, nullptr};
        herr_t result = H5Literate2(id_ < 0 ? file_->id_ : id_, H5_INDEX_NAME, H5_ITER_INC,
                                    &index, collect_name, &collector);
        if (collector.exception) std::rethrow_exception(collector.exception);
        if (result < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Return the absolute path used to open this group.
    Status name(std::string& output, std::string& error) {
        std::unique_lock<std::mutex> lock(file_->mutex_);
        if (file_->id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (id_ < 0) {
            output = "/";
            return Status::Ok;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) {
            error = error_message();
            return Status::Failure;
        }
        ssize_t length = H5Iget_name(id_, nullptr, 0);
        if (length < 0) {
            error = error_message();
            return Status::Failure;
        }
        output.resize(static_cast<size_t>(length) + 1);
        ssize_t written = H5Iget_name(id_, output.data(), output.size());
        if (written < 0) {
            error = error_message();
            return Status::Failure;
        }
        output.resize(static_cast<size_t>(written));
        return Status::Ok;
    }

    // Delete a link relative to this group or at an absolute path.
    Status remove_link(const char* path, std::string& error) {
        std::unique_lock<std::mutex> lock(file_->mutex_);
        if (file_->id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) {
            error = error_message();
            return Status::Failure;
        }
        hid_t location = id_ < 0 ? file_->id_ : id_;
        htri_t exists = H5Lexists(location, path, H5P_DEFAULT);
        if (exists < 0) {
            error = error_message();
            return Status::Failure;
        }
        if (!exists) return Status::NotFound;
        if (H5Ldelete(location, path, H5P_DEFAULT) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Return the file holder shared by this group.
    std::shared_ptr<FileHandle> file_handle() const noexcept { return file_; }

private:
    friend class AttributeHandle;
    friend class DatasetHandle;
    struct NameCollector {
        std::vector<std::string>* names;
        std::exception_ptr exception;
    };

    // Append a link name and retain any exception for the calling method.
    static herr_t collect_name(hid_t, const char* name, const H5L_info2_t*, void* data) noexcept {
        auto* collector = static_cast<NameCollector*>(data);
        try {
            collector->names->emplace_back(name);
            return 0;
        } catch (...) {
            collector->exception = std::current_exception();
            return -1;
        }
    }

    std::shared_ptr<FileHandle> file_;
    hid_t id_;
};

}  // namespace python_hdf5
