#pragma once

#include <hdf5.h>

#include <array>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>

namespace python_hdf5 {

// Describe the outcome of a native operation; error text accompanies any non-Ok status.
enum class Status : int {
    Failure = -1,      // HDF5 or system failure (maps to OSError)
    Ok = 0,            // Operation succeeded
    Closed = 1,        // Owning file is closed (maps to ValueError)
    NotFound = 2,      // Link or attribute is missing (maps to KeyError)
    Unsupported = 3,   // Unsupported type, link or object (maps to TypeError)
    InvalidPath = 4,   // Malformed path (maps to ValueError)
    IsDataset = 5,     // Path resolved to a dataset rather than a group
    InvalidSelection = 6, // Invalid dataset index (maps to IndexError)
    SelectionFallback = 7, // Selection needs Python normalization
};

class GroupHandle;
class AttributeHandle;
class DatasetHandle;

enum class FileDriver { Default, Core };
enum class FileLocking { Default, Disabled, Enabled, BestEffort };
enum class FileVersion { Default, Earliest, V108, V110, V112, V114, V200, Latest };

// Hold validated file-access settings; unset values retain HDF5 defaults.
struct FileOptions {
    FileDriver driver = FileDriver::Default;
    FileLocking locking = FileLocking::Default;
    FileVersion low = FileVersion::Default;
    FileVersion high = FileVersion::Default;
    bool backing_store = true;
    size_t block_size = 64 * 1024;
    bool has_nslots = false;
    bool has_nbytes = false;
    bool has_w0 = false;
    bool swmr_read = false;
    size_t nslots = 0;
    size_t nbytes = 0;
    double w0 = 0;
};

inline H5F_libver_t hdf5_file_version(FileVersion version) {
    switch (version) {
        case FileVersion::Earliest: return H5F_LIBVER_EARLIEST;
        case FileVersion::V108: return H5F_LIBVER_V18;
        case FileVersion::V110: return H5F_LIBVER_V110;
        case FileVersion::V112: return H5F_LIBVER_V112;
        case FileVersion::V114: return H5F_LIBVER_V114;
        case FileVersion::V200: return H5F_LIBVER_V200;
        case FileVersion::Latest: return H5F_LIBVER_LATEST;
        case FileVersion::Default: return H5F_LIBVER_EARLIEST;
    }
    return H5F_LIBVER_EARLIEST;
}

// Own one HDF5 identifier and close it with its matching close function.
class ScopedId {
public:
    // Take ownership of id; closer must match its HDF5 identifier type.
    ScopedId(hid_t id, herr_t (*closer)(hid_t)) noexcept : id_(id), closer_(closer) {}
    ScopedId(const ScopedId&) = delete;
    ScopedId& operator=(const ScopedId&) = delete;
    // Report cleanup failures without throwing from a destructor.
    ~ScopedId() noexcept {
        if (id_ >= 0 && closer_(id_) < 0) {
            std::fputs("python-hdf5: failed to release a temporary HDF5 identifier\n", stderr);
        }
    }
    // Borrow the identifier without transferring ownership.
    hid_t get() const noexcept { return id_; }

    // Transfer ownership to the caller.
    hid_t release() noexcept {
        hid_t result = id_;
        id_ = H5I_INVALID_HID;
        return result;
    }
    // Close any held identifier and adopt id, which must use the same closer.
    void reset(hid_t id) noexcept {
        if (id_ >= 0 && closer_(id_) < 0)
            std::fputs("python-hdf5: failed to release a temporary HDF5 identifier\n", stderr);
        id_ = id;
    }
    // Close explicitly and retain the identifier if closing fails.
    bool close() noexcept {
        if (id_ < 0) return true;
        if (closer_(id_) < 0) return false;
        id_ = H5I_INVALID_HID;
        return true;
    }

private:
    hid_t id_;
    herr_t (*closer_)(hid_t);
};

// Append one HDF5 error description to the string passed as callback data.
inline herr_t append_error(unsigned int, const H5E_error2_t* entry, void* data) noexcept {
    try {
        auto& message = *static_cast<std::string*>(data);
        if (!message.empty()) {
            message += ": ";
        }
        message += entry->desc ? entry->desc : "unknown HDF5 error";
        return 0;
    } catch (...) {
        return -1;
    }
}

// Capture the calling thread's current HDF5 error stack.
inline std::string error_message() {
    std::string result;
    if (H5Ewalk2(H5E_DEFAULT, H5E_WALK_DOWNWARD, append_error, &result) < 0 ||
        result.empty()) {
        return "HDF5 operation failed; error details are unavailable";
    }
    return result;
}

// Serialize access to an owned HDF5 file and its descendants.
class FileHandle {
public:
    // Initialize a closed file holder.
    FileHandle() = default;
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    // Close the remaining file handle when the last owner releases it.
    ~FileHandle() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        if (id_ >= 0 && H5Fclose(id_) < 0) {
            std::fputs("python-hdf5: failed to close an HDF5 file during cleanup\n", stderr);
        }
    }

    // Open a file in the validated mode and report failures in error.
    Status open(const char* path, const char* mode, const FileOptions& options,
                std::string& error) {
        // Verify the HDF5 build and configure error reporting.
        std::unique_lock<std::mutex> lock(mutex_);
        hbool_t threadsafe = false;
        if (H5is_library_threadsafe(&threadsafe) < 0 || !threadsafe) {
            error = "HDF5 library is not thread-safe";
            return Status::Failure;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) {
            error = error_message();
            return Status::Failure;
        }

        // Configure strong close so explicit file close invalidates descendants.
        ScopedId access(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
        if (access.get() < 0) {
            error = error_message();
            return Status::Failure;
        }
        if (H5Pset_fclose_degree(access.get(), H5F_CLOSE_STRONG) < 0) {
            error = error_message();
            return Status::Failure;
        }

        // Apply only explicitly requested settings, retaining HDF5 defaults otherwise.
        if (options.driver == FileDriver::Core &&
            H5Pset_fapl_core(access.get(), options.block_size, options.backing_store) < 0) {
            error = error_message();
            return Status::Failure;
        }
        if (options.low != FileVersion::Default &&
            H5Pset_libver_bounds(access.get(), hdf5_file_version(options.low),
                                 hdf5_file_version(options.high)) < 0) {
            error = error_message();
            return Status::Failure;
        }
        if (options.locking != FileLocking::Default &&
            H5Pset_file_locking(access.get(), options.locking != FileLocking::Disabled,
                                options.locking == FileLocking::BestEffort) < 0) {
            error = error_message();
            return Status::Failure;
        }
        if (options.has_nslots || options.has_nbytes || options.has_w0) {
            int metadata_elements;
            size_t nslots, nbytes;
            double w0;
            if (H5Pget_cache(access.get(), &metadata_elements, &nslots, &nbytes, &w0) < 0 ||
                H5Pset_cache(access.get(), metadata_elements,
                             options.has_nslots ? options.nslots : nslots,
                             options.has_nbytes ? options.nbytes : nbytes,
                             options.has_w0 ? options.w0 : w0) < 0) {
                error = error_message();
                return Status::Failure;
            }
        }

        // Open or create the file according to the requested mode.
        const std::string requested(mode);
        if (requested == "r") {
            id_ = H5Fopen(path, H5F_ACC_RDONLY |
                          (options.swmr_read ? H5F_ACC_SWMR_READ : 0), access.get());
        } else if (requested == "r+") {
            id_ = H5Fopen(path, H5F_ACC_RDWR, access.get());
        } else if (requested == "w") {
            id_ = H5Fcreate(path, H5F_ACC_TRUNC, H5P_DEFAULT, access.get());
        } else if (requested == "w-" || requested == "x") {
            id_ = H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, access.get());
        } else if (requested == "a") {
            id_ = H5Fopen(path, H5F_ACC_RDWR, access.get());

            // Release setup resources and preserve errors from failed cleanup.
            if (id_ < 0) {
                id_ = H5Fcreate(path, H5F_ACC_EXCL, H5P_DEFAULT, access.get());
                if (id_ < 0) {
                    id_ = H5Fopen(path, H5F_ACC_RDWR, access.get());
                }
            }
        }
        if (id_ < 0) {
            error = error_message();
            return Status::Failure;
        }
        if (!access.close()) {
            error = error_message();
            if (H5Fclose(id_) >= 0) {
                id_ = H5I_INVALID_HID;
            } else {
                std::fputs("python-hdf5: failed to close file after setup failure\n", stderr);
            }
            return Status::Failure;
        }
        unsigned long file_number;
        if (H5Fget_fileno(id_, &file_number) < 0) {
            error = error_message();
            if (H5Fclose(id_) >= 0) {
                id_ = H5I_INVALID_HID;
            } else {
                std::fputs("python-hdf5: failed to close file after identity failure\n", stderr);
            }
            return Status::Failure;
        }
        attribute_mutex_ = &attribute_mutex(file_number);
        filename_ = path;
        return Status::Ok;
    }

    // Close the file once, invalidating all HDF5 child identifiers.
    Status close(std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (id_ < 0) {
            return Status::Ok;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0 || H5Fclose(id_) < 0) {
            error = error_message();
            return Status::Failure;
        }
        id_ = H5I_INVALID_HID;
        return Status::Ok;
    }

    // Flush HDF5 buffers while serializing with file operations.
    Status flush(std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0 ||
            H5Fflush(id_, H5F_SCOPE_GLOBAL) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Enable SWMR writing, synchronized with attribute operations across handles.
    Status start_swmr(std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        std::unique_lock<std::mutex> attributes(*attribute_mutex_);
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0 || H5Fstart_swmr_write(id_) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Query the actual SWMR reader or writer state of the file.
    Status swmr_mode(bool& enabled, std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        unsigned int flags;
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0 ||
            H5Fget_intent(id_, &flags) < 0) {
            error = error_message();
            return Status::Failure;
        }
        enabled = (flags & (H5F_ACC_SWMR_READ | H5F_ACC_SWMR_WRITE)) != 0;
        return Status::Ok;
    }

    // Return whether the file has been closed.
    bool closed() {
        std::unique_lock<std::mutex> lock(mutex_);
        return id_ < 0;
    }

    // Return the filesystem path used to open this file.
    std::string filename() const { return filename_; }

    // Retrieve the effective raw chunk cache settings of the open file.
    Status cache_settings(size_t& nslots, size_t& nbytes, double& w0, std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) {
            error = error_message();
            return Status::Failure;
        }
        ScopedId access(H5Fget_access_plist(id_), H5Pclose);
        int metadata_elements;
        if (access.get() < 0 ||
            H5Pget_cache(access.get(), &metadata_elements, &nslots, &nbytes, &w0) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

    // Retrieve the core driver's effective allocation increment and backing-store flag.
    Status core_settings(size_t& block_size, bool& backing_store, std::string& error) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (id_ < 0) {
            error = "file is closed";
            return Status::Closed;
        }
        if (H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr) < 0) {
            error = error_message();
            return Status::Failure;
        }
        ScopedId access(H5Fget_access_plist(id_), H5Pclose);
        if (access.get() < 0 ||
            H5Pget_fapl_core(access.get(), &block_size, &backing_store) < 0) {
            error = error_message();
            return Status::Failure;
        }
        return Status::Ok;
    }

private:
    friend class GroupHandle;
    friend class AttributeHandle;
    friend class DatasetHandle;
    // Share an attribute lock across all handles for an HDF5 file number.
    static std::mutex& attribute_mutex(unsigned long file_number) {
        static std::array<std::mutex, 97> mutexes;
        return mutexes[file_number % mutexes.size()];
    }

    std::mutex mutex_;
    std::mutex* attribute_mutex_ = nullptr;
    hid_t id_ = H5I_INVALID_HID;
    std::string filename_;
};

}  // namespace python_hdf5
