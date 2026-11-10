# cython: language_level=3
# distutils: language=c++
"""Implement the native HDF5 file interface."""

from libcpp cimport bool as cpp_bool
from libcpp.memory cimport make_shared, shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdint cimport int64_t, uint64_t
from libc.string cimport memcpy
from cpython.buffer cimport (
    PyBUF_WRITABLE,
    PyBuffer_FillInfo,
    PyBuffer_Release,
    PyObject_GetBuffer,
)
from cpython.bytes cimport PyBytes_FromStringAndSize

from collections import namedtuple
from functools import lru_cache
import os
import operator
import numpy as np


# Native handle declarations
cdef extern from "cpp/file.hpp" namespace "python_hdf5":
    cdef enum class Status:
        Failure
        Ok
        Closed
        NotFound
        Unsupported
        InvalidPath
        IsDataset
        InvalidSelection
        SelectionFallback


cdef extern from "cpp/file.hpp" namespace "python_hdf5" nogil:
    cdef enum class FileDriver:
        Default
        Core
    cdef enum class FileLocking:
        Default
        Disabled
        Enabled
        BestEffort
    cdef enum class FileVersion:
        Default
        Earliest
        V108
        V110
        V112
        V114
        V200
        Latest
    cdef cppclass FileOptions:
        FileOptions() except +
        FileDriver driver
        FileLocking locking
        FileVersion low
        FileVersion high
        cpp_bool backing_store
        size_t block_size
        cpp_bool has_nslots
        cpp_bool has_nbytes
        cpp_bool has_w0
        cpp_bool swmr_read
        size_t nslots
        size_t nbytes
        double w0
    cdef cppclass FileHandle:
        FileHandle() except +
        Status open(const char* path, const char* mode, const FileOptions& options,
                    string& error) except +
        Status close(string& error) except +
        Status flush(string& error) except +
        Status start_swmr(string& error) except +
        Status swmr_mode(cpp_bool& enabled, string& error) except +
        cpp_bool closed() except +
        string filename() except +
        Status cache_settings(size_t& nslots, size_t& nbytes, double& w0, string& error) except +
        Status core_settings(size_t& block_size, cpp_bool& backing_store, string& error) except +


cdef extern from "cpp/group.hpp" namespace "python_hdf5" nogil:
    cdef cppclass GroupHandle:
        @staticmethod
        shared_ptr[GroupHandle] root(const shared_ptr[FileHandle]& file) except +
        Status resolve(const char* path, cpp_bool create, cpp_bool membership,
                    shared_ptr[GroupHandle]& output, string& error,
                    cpp_bool utf8, cpp_bool require) except +
        Status names(vector[string]& output, string& error) except +
        Status count(uint64_t& output, string& error) except +
        Status name(string& output, string& error) except +
        Status remove_link(const char* path, string& error) except +
        shared_ptr[FileHandle] file_handle() except +

cdef extern from "cpp/attribute.hpp" namespace "python_hdf5" nogil:
    cdef cppclass AttributeData:
        string dtype
        vector[uint64_t] shape
        vector[unsigned char] data
        vector[string] strings
        shared_ptr[unsigned char] block
        size_t block_size
        const unsigned char* external
        size_t external_size
    cdef cppclass AttributeHandle:
        AttributeHandle(shared_ptr[GroupHandle] group) except +
        Status read(const char* name, AttributeData& out, string& error) except +
        Status write(const char* name, const AttributeData& input, string& error,
                  cpp_bool utf8) except +
        Status remove(const char* name, string& error) except +
        Status contains(const char* name, string& error) except +
        Status names(vector[string]& out, string& error) except +
        Status count(uint64_t& out, string& error) except +

cdef extern from "cpp/dataset.hpp" namespace "python_hdf5" nogil:
    cdef cppclass DatasetOptions:
        DatasetOptions() except +
        vector[uint64_t] chunks
        vector[uint64_t] maxshape
        cpp_bool has_maxshape
        cpp_bool has_data
        cpp_bool has_fill
        AttributeData fill
        int gzip_level
        cpp_bool shuffle
        cpp_bool fletcher32
    cdef cppclass DatasetSelection:
        vector[uint64_t] shape
        vector[uint64_t] start
        vector[uint64_t] stride
        vector[uint64_t] count
        vector[uint64_t] result_shape
        string dtype
    cdef cppclass SelectionRequest:
        SelectionRequest() except +
        void add_index(int64_t value)
        void add_slice(cpp_bool has_start, int64_t start, cpp_bool has_stop, int64_t stop,
                       int64_t step)
        void add_ellipsis()
    cdef cppclass DatasetHandle:
        @staticmethod
        Status lookup(const shared_ptr[GroupHandle]& group, const char* path,
                      shared_ptr[GroupHandle]& group_output,
                      shared_ptr[DatasetHandle]& dataset_output, string& error) except +
        @staticmethod
        Status open(const shared_ptr[GroupHandle]& parent, const char* name, cpp_bool create,
                 const AttributeData& input, const DatasetOptions& options,
                 shared_ptr[DatasetHandle]& output,
                 string& error, cpp_bool utf8) except +
        Status resize(const vector[uint64_t]& shape, string& error) except +
        shared_ptr[AttributeHandle] attributes() except +
        Status name(string& output, string& error) except +
        shared_ptr[GroupHandle] parent() except +
        shared_ptr[FileHandle] file_handle() except +
        Status sync(cpp_bool refresh, string& error) except +
        Status read(AttributeData& output, cpp_bool contents, string& error,
                    cpp_bool include_shape) except +
        Status write(const AttributeData& input, string& error) except +
        Status transfer(const DatasetSelection& selection, AttributeData& data, cpp_bool writing,
                     string& error) except +
        Status read_selection(const SelectionRequest& request, AttributeData& data,
                              string& error) except +


# Shared name, error, and numeric conversion helpers
cdef bytes _encode_name(object name):
    """Encode an HDF5 name without accepting embedded null bytes."""
    cdef bytes result
    if not isinstance(name, (str, bytes)):
        raise TypeError("group names must be str or bytes")
    result = name.encode("utf-8") if isinstance(name, str) else name
    if b"\0" in result:
        raise ValueError("group path contains a null byte")
    return result


cdef bint _needs_utf8(object name):
    """Return whether a name must be stored with the UTF-8 character set, like h5py."""
    return isinstance(name, str) and not name.isascii()


cdef object _decode_name(bytes name):
    """Return name as str when valid UTF-8, otherwise as bytes, like h5py."""
    try:
        return name.decode("utf-8")
    except UnicodeDecodeError:
        return name


cdef void _attribute_error(Status status, string& error, object name) except *:
    """Translate native operation status codes to Python exceptions."""
    if status == Status.Closed:
        raise ValueError(error.decode("utf-8", "replace"))
    if status == Status.NotFound:
        raise KeyError(name)
    if status == Status.Unsupported:
        raise TypeError(error.decode("utf-8", "replace"))
    if status != Status.Ok:
        raise OSError(error.decode("utf-8", "replace"))


cdef object _vlen_dtype(object element):
    """Return the h5py-style object dtype for variable-length data of element type."""
    return np.dtype("O", metadata={"vlen": element})


cdef bytes _escape_name(bytes raw):
    """Percent-escape bytes outside [A-Za-z0-9_.-] for use inside dtype names."""
    cdef bytearray out = bytearray()
    for byte in raw:
        if (48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122
                or byte in (45, 46, 95)):
            out.append(byte)
        else:
            out += b"%%%02X" % byte
    return bytes(out)


cdef bytes _unquote_bytes(str text):
    """Decode %XX escapes of an ASCII fragment into bytes."""
    cdef bytearray out = bytearray()
    cdef int index = 0
    raw = text.encode("ascii")
    while index < len(raw):
        if raw[index] == 37:
            out.append(int(raw[index + 1:index + 3], 16))
            index += 3
        else:
            out.append(raw[index])
            index += 1
    return bytes(out)


cdef bint _is_enum(object dtype):
    """Return whether dtype carries h5py enum metadata."""
    return dtype.kind in "iu" and bool(dtype.metadata) and "enum" in dtype.metadata


cdef bint _is_opaque(object dtype):
    """Return whether dtype is stored as an HDF5 opaque type."""
    if dtype.kind == "V":
        return True
    return bool(dtype.metadata) and bool(dtype.metadata.get("h5py_opaque"))


cdef str _dtype_class(object dtype):
    """Classify a dtype as numeric, fixed, vlenstr, vlenarr, or opaque."""
    if dtype.kind == "O":
        metadata = dtype.metadata or {}
        return "vlenstr" if metadata.get("vlen") in (str, bytes) else "vlenarr"
    if dtype.kind == "S":
        return "fixed"
    return "opaque" if _is_opaque(dtype) else "numeric"


cdef bytes _enum_name(object dtype):
    """Return the native name of an enum dtype, validating its member values."""
    base = np.dtype(dtype.str)
    info = np.iinfo(base)
    members = []
    for member, value in dtype.metadata["enum"].items():
        if not isinstance(member, str) or isinstance(value, (bool, np.bool_)):
            raise TypeError("enum members must map str names to integers")
        value = operator.index(value)
        if not info.min <= value <= info.max:
            raise ValueError(f"enum value {value} does not fit in {base}")
        members.append((_escape_name(member.encode("utf-8")), value))
    members.sort()
    body = b",".join(b"%s=%d" % item for item in members)
    return b"enum:" + base.str.encode("ascii") + b":" + body


cdef bytes _opaque_name(object dtype):
    """Return the native name of an opaque dtype, tagging NumPy dtypes like h5py."""
    tag = b"" if dtype.kind == "V" else b"NUMPY:" + dtype.str.encode("ascii")
    return b"opaque:%d:%s" % (dtype.itemsize, _escape_name(tag))


cdef bytes _vlen_name(object dtype):
    """Return the native name of a variable-length string or numeric array dtype."""
    element = (dtype.metadata or {}).get("vlen")
    if element is bytes:
        return b"vlen:ascii"
    if element is None or element is str:
        return b"vlen:utf8"
    base = np.dtype(element)
    if base.kind not in "iuf" or base.itemsize not in (1, 2, 4, 8) or base.metadata:
        raise TypeError(f"unsupported variable-length element dtype: {base}")
    if base.kind == "f" and base.itemsize == 1:
        raise TypeError(f"unsupported variable-length element dtype: {base}")
    return b"vlen:" + base.str.encode("ascii")


cdef bytes _dtype_name(object dtype):
    """Return the native dtype name used to describe dtype in transfers."""
    metadata = dtype.metadata or {}
    if dtype.kind == "O":
        return _vlen_name(dtype)
    if dtype.kind == "S" and metadata.get("h5py_encoding") == "utf-8":
        return dtype.str.encode("ascii") + b":utf8"
    if _is_opaque(dtype):
        return _opaque_name(dtype)
    if _is_enum(dtype):
        return _enum_name(dtype)
    return dtype.str.encode("ascii")


@lru_cache(maxsize=128)
def _plain_dtype(name: str) -> object:
    """Return a cached dtype without mutable special-type metadata."""
    return np.dtype(name)


cdef object _dtype_from_name(str name):
    """Return the NumPy dtype for a native dtype name, restoring special-type metadata."""
    if name == "vlen:utf8":
        return _vlen_dtype(str)
    if name == "vlen:ascii":
        return _vlen_dtype(bytes)
    if name.startswith("vlen:"):
        return _vlen_dtype(np.dtype(name[5:]))
    if name.startswith("enum:"):
        base, _, body = name[5:].partition(":")
        members = {}
        for item in filter(None, body.split(",")):
            member, _, value = item.partition("=")
            members[_unquote_bytes(member).decode("utf-8", "surrogateescape")] = int(value)
        return np.dtype(base, metadata={"enum": members})
    if name.startswith("opaque:"):
        size, _, tag = name[7:].partition(":")
        text = _unquote_bytes(tag).decode("utf-8", "surrogateescape")
        if text.startswith("NUMPY:"):
            return np.dtype(text[6:], metadata={"h5py_opaque": True})
        return np.dtype(f"V{size}")
    if name.endswith(":utf8"):
        return np.dtype(name[:-5], metadata={"h5py_encoding": "utf-8"})
    return _plain_dtype(name)


cdef object _read_dtype(str name):
    """Return the metadata-free dtype that describes the raw bytes of a native name."""
    if name.startswith("enum:"):
        return _plain_dtype(name[5:].partition(":")[0])
    if not name.startswith(("vlen:", "opaque:")) and not name.endswith(":utf8"):
        return _plain_dtype(name)
    dtype = _dtype_from_name(name)
    if dtype.kind == "O":
        return dtype
    return np.dtype(dtype.str) if dtype.kind != "V" else dtype


cdef object _check_metadata(object dtype, str label):
    """Reject dtype metadata that cannot be mapped onto HDF5, instead of dropping it."""
    metadata = dtype.metadata
    if not metadata:
        return
    keys = set(metadata)
    if dtype.kind == "O" and keys == {"vlen"}:
        _vlen_name(dtype)
        return
    if dtype.kind == "S" and keys == {"h5py_encoding"} and metadata["h5py_encoding"] in (
            "utf-8", "ascii"):
        return
    if dtype.kind in "iu" and keys == {"enum"}:
        _enum_name(dtype)
        return
    if keys == {"h5py_opaque"} and metadata["h5py_opaque"] is True and (
            dtype.kind in "MmV" or dtype.kind in "iufcb"):
        return
    raise TypeError(f"unsupported {label} dtype with special type metadata: {dtype}")


cdef object _string_objects(object array, str label):
    """Validate an object array of str or bytes and tag it with its encoding."""
    # Check element types and embedded null bytes.
    metadata = array.dtype.metadata or {}
    kinds = set()
    for item in array.ravel():
        if isinstance(item, str):
            kinds.add(str)
            has_null = "\0" in item
        elif isinstance(item, bytes):
            kinds.add(bytes)
            has_null = b"\0" in item
        else:
            raise TypeError(f"unsupported {label} dtype: object array of {type(item).__name__}")
        if has_null:
            raise ValueError("variable-length strings do not support embedded null bytes")

    # Select the encoding from explicit metadata or from the homogeneous elements.
    encoding = metadata.get("vlen")
    if encoding is None:
        if len(kinds) != 1:
            raise TypeError(f"unsupported {label} dtype: object arrays must contain only "
                            "str or only bytes")
        encoding = next(iter(kinds))
    return array.view(_vlen_dtype(encoding))


cdef object _vlen_arrays(object array, object element, str label):
    """Convert every element of an object array to a contiguous 1-D array of element."""
    base = np.dtype(element)
    result = np.empty(array.shape, dtype=_vlen_dtype(base))
    flat = result.reshape(-1)
    for index, item in enumerate(array.ravel()):
        item = np.ascontiguousarray(item, dtype=base)
        if item.ndim != 1:
            raise ValueError("variable-length elements must be one-dimensional")
        flat[index] = item
    return result


cdef object _prepare_array(object value, str label):
    """Copy a supported value into a C-order array carrying its special-type metadata.

    The label identifies the object in unsupported-dtype error messages.
    """
    # Store Python text and bytes as variable-length strings like h5py.
    if isinstance(value, str) or type(value) is bytes:
        array = np.empty((), dtype=object)
        array[()] = value
        return _string_objects(array, label)

    # Copy before releasing the GIL so native calls use stable storage.
    array = np.array(value, copy=True, order="C", subok=False)
    _check_metadata(array.dtype, label)
    kind = array.dtype.kind
    metadata = array.dtype.metadata or {}

    # Treat lists of str or bytes as variable-length strings, unlike explicit NumPy arrays.
    if kind in ("U", "S") and not isinstance(value, (np.ndarray, np.generic)):
        array = np.array(value, dtype=object)
        kind = "O"
    elif kind == "T":
        array = array.astype(object)
        kind = "O"
    if kind == "O":
        element = metadata.get("vlen")
        if element in (None, str, bytes):
            return _string_objects(array, label)
        return _vlen_arrays(array, element, label)
    if kind == "S":
        return array

    # Accept opaque raw bytes and values explicitly tagged as opaque.
    if kind == "V":
        if array.dtype.fields is not None or array.dtype.subdtype is not None:
            raise TypeError(f"unsupported {label} dtype: {array.dtype}")
        return array
    if metadata.get("h5py_opaque"):
        return array
    if kind in ("M", "m"):
        raise TypeError(f"unsupported {label} dtype: {array.dtype}; use opaque_dtype()")

    # Reject types that have no supported HDF5 representation.
    if kind not in ("i", "u", "f", "b", "c"):
        raise TypeError(f"unsupported {label} dtype: {array.dtype}")
    if (kind in ("i", "u") and array.dtype.itemsize not in (1, 2, 4, 8)
            or kind == "f" and array.dtype.itemsize not in (2, 4, 8)
            or kind == "c" and array.dtype.itemsize not in (8, 16)):
        raise TypeError(f"unsupported {label} dtype: {array.dtype}")
    return array


cdef object _fixed_strings(object array, int size):
    """Convert string data to a fixed-size bytes array, truncating by bytes."""
    if array.dtype.kind == "O":
        items = [item.encode("utf-8") if isinstance(item, str) else item
                 for item in array.ravel()]
        return np.array(items, dtype=f"S{size}").reshape(array.shape)
    return np.array(array, dtype=f"S{size}")


cdef object _vlen_values(object value, object dtype, tuple shape):
    """Coerce lists or single arrays into an object array of variable-length elements."""
    base = dtype.metadata["vlen"]
    if isinstance(value, (str, bytes)):
        raise TypeError(f"cannot convert text to variable-length {np.dtype(base)} arrays")
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        return value
    if not shape and not (isinstance(value, (list, tuple)) and value
                          and isinstance(value[0], (list, tuple, np.ndarray))):
        item = np.empty((), dtype=object)
        item[()] = np.asarray(value, dtype=base)
        return item
    result = np.empty(len(value), dtype=object)
    for index, item in enumerate(value):
        result[index] = np.asarray(item, dtype=base)
    return result


cdef object _convert_for_dataset(object value, object dtype, tuple shape, str label):
    """Convert value to the dataset dtype, broadcast it to shape, and copy it."""
    # Accept lists and single arrays for variable-length numeric elements.
    stored = _dtype_class(dtype)
    if stored == "vlenarr":
        array = _vlen_arrays(_vlen_values(value, dtype, shape), dtype.metadata["vlen"], label)
        return np.array(np.broadcast_to(array, shape), dtype=dtype, copy=True, order="C")

    # Write matching contiguous numeric arrays directly; the borrowed bytes are not copied.
    if (isinstance(value, np.ndarray) and stored == "numeric" and value.dtype == dtype
            and value.shape == shape and value.flags.c_contiguous
            and value.dtype.kind in "iufcb"):
        if value.dtype.metadata == dtype.metadata:
            return value
        return value.view(dtype)

    # Require values of the same class as the stored type; strings never parse as numbers.
    array = _prepare_array(value, label)
    given = _dtype_class(array.dtype)
    if stored != given and {stored, given} != {"fixed", "vlenstr"}:
        raise TypeError(f"cannot convert {array.dtype} to {label} dtype {dtype}")
    if stored == "opaque" and array.dtype.kind == "V" and array.dtype != dtype:
        raise TypeError(f"cannot convert {array.dtype} to {label} dtype {dtype}")

    # Convert between fixed and variable-length strings before broadcasting.
    if stored == "fixed":
        array = _fixed_strings(array, dtype.itemsize)
    elif stored == "vlenstr" and given == "fixed":
        array = array.astype(object)
    if array.shape == shape and array.dtype == dtype:
        result = array
    else:
        result = np.array(np.broadcast_to(array, shape), dtype=dtype, copy=True, order="C")

    # Restore special-type metadata that NumPy drops during conversion.
    if result.dtype.metadata != dtype.metadata:
        result = result.view(dtype)
    return result


cdef tuple _dataset_shape(object shape):
    """Validate dimensions for dataset shapes and chunks."""
    if isinstance(shape, int):
        shape = (shape,)
    try:
        dims = tuple(_file_size(dim, "dimension") for dim in shape)
    except TypeError:
        raise TypeError("shape must be an integer or a sequence of integers") from None
    if len(dims) > 32:
        raise TypeError("dataset rank exceeds HDF5 limit")
    return dims


cdef object _pack_array(object array, AttributeData& output):
    """Fill an initially empty native transfer object from a prepared array.

    Fixed-size bytes are borrowed, so the returned array must stay referenced until the
    native call that uses output has returned.
    """
    # Capture the dtype name and shape.
    output.dtype = _dtype_name(array.dtype)
    for dim in array.shape:
        output.shape.push_back(dim)

    # Store variable-length elements as encoded strings or raw array bytes.
    cdef bytes raw
    cls = _dtype_class(array.dtype)
    if cls == "vlenstr":
        for item in array.ravel():
            output.strings.push_back(item.encode("utf-8") if isinstance(item, str) else item)
        return array
    if cls == "vlenarr":
        for item in array.ravel():
            output.strings.push_back(item.tobytes())
        return array

    # Borrow the array bytes; callers keep the private array alive until the call returns.
    cdef const unsigned char[::1] view
    array = np.ascontiguousarray(array)
    if array.nbytes:
        view = array.reshape(-1).view(np.uint8)
        output.external = &view[0]
        output.external_size = array.nbytes
    return array


cdef class _Block:
    """Own a native memory block and expose it through the buffer protocol."""

    cdef shared_ptr[unsigned char] ptr
    cdef Py_ssize_t size

    def __getbuffer__(self, Py_buffer* view, int flags):
        """Expose the block as a writable byte buffer."""
        PyBuffer_FillInfo(view, self, self.ptr.get(), self.size, 0, flags)

    def __releasebuffer__(self, Py_buffer* view):
        """Release the buffer; the block lives as long as this object."""
        pass


cdef object _unpack_array(AttributeData& result, bint as_str):
    """Return a NumPy scalar or independent array from native data.

    Variable-length strings are returned as bytes, or as str when as_str is set.
    """
    # Restore variable-length data, decoding strings like h5py when requested.
    cdef bytes raw
    cdef size_t index
    cdef size_t byte_count
    cdef Py_buffer destination
    cdef _Block block
    name = (<bytes>result.dtype).decode("ascii")
    shape = tuple(result.shape)
    if name.startswith("vlen:"):
        array = np.empty(shape, dtype=object)
        flat = array.reshape(-1)
        if name in ("vlen:utf8", "vlen:ascii"):
            for index in range(result.strings.size()):
                raw = result.strings[index]
                flat[index] = raw.decode("utf-8", "surrogateescape") if as_str else raw
        else:
            element = _plain_dtype(name[5:])
            for index in range(result.strings.size()):
                byte_count = result.strings[index].size()
                if byte_count % element.itemsize:
                    raise OSError("invalid variable-length transfer size")
                item = np.empty(byte_count // element.itemsize, dtype=element)
                if byte_count:
                    PyObject_GetBuffer(item, &destination, PyBUF_WRITABLE)
                    try:
                        memcpy(destination.buf, result.strings[index].data(), byte_count)
                    finally:
                        PyBuffer_Release(&destination)
                flat[index] = item
        return array[()] if not shape else array

    # Adopt the native block without copying, including scalar dataspaces.
    dtype = _read_dtype(name)
    if result.block_size:
        block = _Block.__new__(_Block)
        block.ptr = result.block
        block.size = result.block_size
        array = np.frombuffer(block, dtype=dtype)
    else:
        array = np.empty(0, dtype=dtype)
    array = array.reshape(shape)
    return array[()] if not shape else array


cdef bint _selection_request(object selection, SelectionRequest& request) except -1:
    """Encode ordinary built-in indices without invoking user-defined index methods."""
    parts = selection if type(selection) is tuple else (selection,)
    if len(parts) > 33:
        return False
    for part in parts:
        if part is Ellipsis:
            request.add_ellipsis()
        elif type(part) is int:
            if part < -9223372036854775808 or part > 9223372036854775807:
                return False
            request.add_index(part)
        elif type(part) is slice:
            start, stop, step = part.start, part.stop, part.step
            if ((start is not None and (type(start) is not int or
                                       start < -9223372036854775808 or
                                       start > 9223372036854775807))
                    or (stop is not None and (type(stop) is not int or
                                              stop < -9223372036854775808 or
                                              stop > 9223372036854775807))
                    or (step is not None and (type(step) is not int or step <= 0 or
                                              step > 9223372036854775807))):
                return False
            request.add_slice(start is not None, 0 if start is None else start,
                              stop is not None, 0 if stop is None else stop,
                              1 if step is None else step)
        else:
            return False
    return True


cdef void _normalize_selection(object selection, tuple shape, object dtype,
                               DatasetSelection& output) except *:
    """Normalize integer and positive-slice selections for a dataset snapshot."""
    # Expand an ellipsis and supply omitted dimensions.
    parts = selection if isinstance(selection, tuple) else (selection,)
    ellipses = sum(part is Ellipsis for part in parts)
    if ellipses > 1:
        raise IndexError("only one ellipsis is allowed")
    if len(parts) - ellipses > len(shape):
        raise IndexError("too many indices for dataset")
    expanded = []
    for part in parts:
        if part is Ellipsis:
            expanded.extend([slice(None)] * (len(shape) - len(parts) + 1))
        else:
            expanded.append(part)
    expanded.extend([slice(None)] * (len(shape) - len(expanded)))
    output.dtype = _dtype_name(dtype)
    output.shape.reserve(len(shape))
    output.start.reserve(len(shape))
    output.stride.reserve(len(shape))
    output.count.reserve(len(shape))
    output.result_shape.reserve(len(shape))

    # Convert each dimension to an HDF5 start, stride, and element count.
    for dim, part in zip(shape, expanded):
        output.shape.push_back(dim)
        if isinstance(part, slice):
            start, stop, step = part.indices(dim)
            if step <= 0:
                raise ValueError("slice step must be positive")
            count = len(range(start, stop, step))
            output.start.push_back(start)
            output.stride.push_back(step if count > 1 else 1)
            output.count.push_back(count)
            output.result_shape.push_back(count)
        else:
            if isinstance(part, (bool, np.bool_)):
                raise TypeError("boolean indexing is not supported")
            try:
                index = operator.index(part)
            except TypeError:
                raise TypeError("only integers, slices, and ellipsis are supported") from None
            if index < 0:
                index += dim
            if index < 0 or index >= dim:
                raise IndexError("dataset index out of range")
            output.start.push_back(index)
            output.stride.push_back(1)
            output.count.push_back(1)


# Attribute mapping
cdef class AttributeManager:
    """Access numeric and string HDF5 attributes on a file, group, or dataset.

    The manager shares its owner's lifetime and synchronization. No attribute
    data is cached. Attribute operations through this library are synchronized
    across file handles in the same process, but callers should synchronize
    sequences of operations. Replacement is not transactional: an HDF5 error
    after deletion can lose the old value. Reads return independent values.
    """

    cdef shared_ptr[AttributeHandle] _handle

    def __init__(self):
        """Reject direct construction of an unbound manager."""
        raise TypeError("attribute managers must be obtained from a file or group")

    def keys(self):
        """Return a name-sorted snapshot of attribute names.

        Returns:
            Attribute names decoded with the filesystem encoding.

        Raises:
            ValueError: If the owning file is closed.
            OSError: If enumeration fails.
        """
        # Validate the owner and enumerate under its native lock.
        if not self._handle:
            raise ValueError("attribute manager is not initialized")
        cdef vector[string] names
        cdef string error
        cdef Status status
        with nogil:
            status = self._handle.get().names(names, error)

        # Translate errors and decode the name snapshot.
        _attribute_error(status, error, None)
        return [_decode_name(<bytes>name) for name in names]

    def create(self, name, data, shape=None, dtype=None):
        """Create or replace an attribute with an explicit shape and dtype.

        Args:
            name: Attribute name as str or bytes.
            data: Scalar or array containing the attribute values.
            shape: Optional output shape with the same number of elements as data.
            dtype: Optional supported NumPy or special HDF5-compatible dtype.

        Raises:
            TypeError: If the name, dtype, shape, or data is unsupported.
            ValueError: If the shape conflicts with the data, the name is invalid,
                or the owning file is closed.
            OverflowError: If a dimension is too large.
            OSError: If HDF5 cannot replace or write the attribute.
        """
        if not self._handle:
            raise ValueError("attribute manager is not initialized")
        cdef bytes encoded = _encode_name(name)
        if not encoded:
            raise ValueError("attribute name must not be empty")

        if dtype is None:
            array = _prepare_array(data, "attribute")
        else:
            requested = np.dtype(dtype)
            _check_metadata(requested, "attribute")
            if requested.fields is not None or requested.subdtype is not None:
                raise TypeError(f"unsupported attribute dtype: {requested}")
            if _dtype_class(requested) == "vlenarr":
                array = _vlen_values(data, requested, ())
                array = _vlen_arrays(array, requested.metadata["vlen"], "attribute")
            else:
                array = np.array(data, dtype=requested, copy=True, order="C", subok=False)
                if array.dtype.metadata != requested.metadata:
                    array = array.view(requested)
                array = _prepare_array(array, "attribute")
                if (requested.metadata or {}).get("vlen") is bytes:
                    for item in array.ravel():
                        if isinstance(item, str):
                            item.encode("ascii")

        if shape is not None:
            target_shape = _dataset_shape(shape)
            if np.prod(target_shape, dtype=object) != array.size:
                raise ValueError("shape of new attribute conflicts with shape of data")
            array = array.reshape(target_shape)

        cdef AttributeData input
        keep = _pack_array(array, input)
        cdef const char* name_ptr = encoded
        cdef bint utf8 = _needs_utf8(name)
        cdef string error
        cdef Status status
        with nogil:
            status = self._handle.get().write(name_ptr, input, error, utf8)
        _attribute_error(status, error, name)

    def __getitem__(self, name):
        """Read a named attribute into an independent scalar or array."""
        # Resolve the name and read into native-owned storage.
        if not self._handle:
            raise ValueError("attribute manager is not initialized")
        cdef bytes encoded = _encode_name(name)
        cdef const char* name_ptr = encoded
        cdef AttributeData result
        cdef string error
        cdef Status status
        with nogil:
            status = self._handle.get().read(name_ptr, result, error)

        # Translate failures before converting the result.
        _attribute_error(status, error, name)
        return _unpack_array(result, True)

    def __setitem__(self, name, value):
        """Create or replace a numeric or string attribute."""
        # Validate the owner and attribute name.
        if not self._handle:
            raise ValueError("attribute manager is not initialized")
        cdef bytes encoded = _encode_name(name)
        if not encoded:
            raise ValueError("attribute name must not be empty")

        # Prepare C-order data and its HDF5 dimensions.
        cdef AttributeData input
        keep = _pack_array(_prepare_array(value, "attribute"), input)

        # Keep the source alive until the native write completes.
        cdef const char* name_ptr = encoded
        cdef bint utf8 = _needs_utf8(name)
        cdef string error
        cdef Status status
        with nogil:
            status = self._handle.get().write(name_ptr, input, error, utf8)
        _attribute_error(status, error, name)

    def __delitem__(self, name):
        """Delete a named attribute or raise if it is absent."""
        if not self._handle:
            raise ValueError("attribute manager is not initialized")
        cdef bytes encoded = _encode_name(name)
        cdef const char* name_ptr = encoded
        cdef string error
        cdef Status status
        with nogil:
            status = self._handle.get().remove(name_ptr, error)
        _attribute_error(status, error, name)

    def __contains__(self, name):
        """Return whether the named attribute exists."""
        if not self._handle:
            raise ValueError("attribute manager is not initialized")
        cdef bytes encoded = _encode_name(name)
        cdef const char* name_ptr = encoded
        cdef string error
        cdef Status status
        with nogil:
            status = self._handle.get().contains(name_ptr, error)
        if status == Status.NotFound:
            return False
        _attribute_error(status, error, name)
        return True

    def __iter__(self):
        """Iterate over a snapshot of attribute names."""
        return iter(self.keys())

    def __len__(self):
        """Return the number of attributes."""
        if not self._handle:
            raise ValueError("attribute manager is not initialized")
        cdef uint64_t count
        cdef string error
        cdef Status status
        with nogil:
            status = self._handle.get().count(count, error)
        _attribute_error(status, error, None)
        return count


# String dtype helpers
StringInfo = namedtuple("StringInfo", ["encoding", "length"])


def string_dtype(encoding="utf-8", length=None):
    """Return a dtype describing an HDF5 string, like h5py.string_dtype.

    Args:
        encoding: Either "utf-8" or "ascii".
        length: Fixed length in bytes, or None for variable-length strings.

    Returns:
        An object dtype for variable-length strings, or an S dtype for fixed lengths.

    Raises:
        ValueError: If the encoding or length is invalid.
    """
    # Validate the arguments.
    if encoding not in ("utf-8", "ascii"):
        raise ValueError("encoding must be 'utf-8' or 'ascii'")
    if length is None:
        return _vlen_dtype(str if encoding == "utf-8" else bytes)
    length = operator.index(length)
    if length <= 0:
        raise ValueError("fixed-length strings must have a positive length")
    return np.dtype(f"S{length}", metadata={"h5py_encoding": encoding})


def check_string_dtype(dtype):
    """Return StringInfo(encoding, length) for string dtypes, else None.

    The length is None for variable-length strings.
    """
    dtype = np.dtype(dtype)
    metadata = dtype.metadata or {}
    if dtype.kind == "O" and metadata.get("vlen") in (str, bytes):
        return StringInfo("utf-8" if metadata["vlen"] is str else "ascii", None)
    if dtype.kind == "S":
        return StringInfo(metadata.get("h5py_encoding", "ascii"), dtype.itemsize)
    return None


def vlen_dtype(basetype):
    """Return a dtype describing variable-length arrays of basetype, like h5py.vlen_dtype.

    Raises:
        TypeError: If basetype cannot be stored as a variable-length element.
    """
    element = basetype if basetype in (str, bytes) else np.dtype(basetype)
    dtype = _vlen_dtype(element)
    _vlen_name(dtype)
    return dtype


def check_vlen_dtype(dtype):
    """Return the element type of a variable-length dtype, or None for other dtypes."""
    dtype = np.dtype(dtype)
    if dtype.kind != "O":
        return None
    return (dtype.metadata or {}).get("vlen")


def enum_dtype(values_dict, basetype=np.uint8):
    """Return an integer dtype carrying enum members, like h5py.enum_dtype.

    Raises:
        TypeError: If basetype is not an integer type or the members are invalid.
        ValueError: If a value does not fit in basetype.
    """
    base = np.dtype(basetype)
    if base.kind not in "iu":
        raise TypeError("enum base types must be integers")
    dtype = np.dtype(base, metadata={"enum": dict(values_dict)})
    _enum_name(dtype)
    return dtype


def check_enum_dtype(dtype):
    """Return the member dict of an enum dtype, or None for other dtypes."""
    dtype = np.dtype(dtype)
    if _is_enum(dtype):
        return dict(dtype.metadata["enum"])
    return None


def opaque_dtype(dtype):
    """Return dtype tagged so it is stored as an HDF5 opaque type, like h5py.opaque_dtype.

    Raises:
        TypeError: If dtype has fields or an unsupported kind.
    """
    dtype = np.dtype(dtype)
    if dtype.kind not in "MmVbiufc" or dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError(f"cannot store {dtype} as an opaque type")
    return np.dtype(dtype, metadata={"h5py_opaque": True})


def check_opaque_dtype(dtype):
    """Return whether dtype is stored as an HDF5 opaque type."""
    dtype = np.dtype(dtype)
    return dtype.kind in "MmV" or bool((dtype.metadata or {}).get("h5py_opaque"))


cdef object _decode_strings(object value, str codec, str errors):
    """Decode bytes or arrays of bytes into str values."""
    if isinstance(value, bytes):
        return value.decode(codec, errors)
    if isinstance(value, np.ndarray) and value.dtype.kind in ("O", "S"):
        items = [item.decode(codec, errors) for item in value.ravel()]
        result = np.empty(value.shape, dtype=object)
        result.reshape(-1)[:] = items
        return result
    return value


cdef object _convert_dtype(object value, object dtype):
    """Convert a read value to dtype, decoding strings for native string arrays."""
    if dtype.kind == "T":
        value = _decode_strings(value, "utf-8", "strict")
        return np.array(value, dtype=dtype)
    array = np.asarray(value)
    numeric = "biufc"
    textual = "SUOT"
    source_kind = array.dtype.kind
    if (source_kind in numeric and dtype.kind in numeric) or \
            (source_kind in textual and dtype.kind in textual) or \
            array.dtype == dtype:
        return array.astype(dtype)
    raise TypeError(f"cannot convert {array.dtype} data to {dtype}")


cdef class _ReadAdapter:
    """Convert dataset reads with a function, like h5py's asstr and astype views."""

    cdef object _dataset
    cdef object _convert

    def __init__(self, dataset, convert):
        """Bind to a dataset and a conversion function applied to each read."""
        self._dataset = dataset
        self._convert = convert

    def __getitem__(self, selection):
        """Read a selection and convert the result."""
        return self._convert(self._dataset[selection])

    def __len__(self):
        """Return the length of the first dataset dimension."""
        return len(self._dataset)

    @property
    def dtype(self):
        """Return the underlying dataset dtype."""
        return self._dataset.dtype

    @property
    def shape(self):
        """Return the underlying dataset shape."""
        return self._dataset.shape


# Dataset interface
cdef class Dataset:
    """Access a numeric or string HDF5 dataset with synchronized I/O."""

    cdef shared_ptr[DatasetHandle] _dataset

    def __init__(self):
        """Reject direct construction of an unbound dataset."""
        raise TypeError("datasets must be obtained from a file or group")

    cdef object _info(self, bint contents):
        """Read dataset metadata and, when requested, the contents."""
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef AttributeData output
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().read(output, contents, error, True)

        # Return either metadata or a Python-owned copy of the contents.
        _attribute_error(status, error, None)
        if contents:
            return _unpack_array(output, False)
        return (tuple(output.shape), _dtype_from_name((<bytes>output.dtype).decode("ascii")))

    @property
    def attrs(self):
        """Return a manager for the dataset's attributes."""
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef AttributeManager manager = AttributeManager.__new__(AttributeManager)
        manager._handle = self._dataset.get().attributes()
        return manager

    @property
    def shape(self):
        """Return the dataset shape."""
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef AttributeData output
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().read(output, False, error, True)
        _attribute_error(status, error, None)
        return tuple(output.shape)

    @property
    def name(self):
        """Return the dataset's absolute HDF5 path.

        Raises:
            ValueError: If the owning file is closed.
            OSError: If HDF5 cannot retrieve the path.
        """
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef string path
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().name(path, error)
        _attribute_error(status, error, None)
        return _decode_name(<bytes>path)

    @property
    def parent(self):
        """Return the group containing this dataset.

        Raises:
            ValueError: If the dataset is not initialized.
        """
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef Group group = Group.__new__(Group)
        group._group = self._dataset.get().parent()
        return group

    @property
    def file(self):
        """Return a file wrapper sharing this dataset's native file handle.

        Raises:
            ValueError: If the dataset is not initialized.
        """
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        return _file_from_handle(self._dataset.get().file_handle())

    @property
    def dtype(self):
        """Return the dataset NumPy dtype."""
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef AttributeData output
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().read(output, False, error, False)
        _attribute_error(status, error, None)
        return _dtype_from_name((<bytes>output.dtype).decode("ascii"))

    def __len__(self):
        """Return the length of the first dimension.

        Raises:
            TypeError: If the dataset is scalar.
        """
        shape = self.shape
        if not shape:
            raise TypeError("len() of a scalar dataset")
        return shape[0]

    def asstr(self, encoding=None, errors="strict"):
        """Return a read-only view that decodes string data to str.

        Args:
            encoding: Codec to use; defaults to the dataset's stored character set.
            errors: Codec error handling, such as "strict" or "surrogateescape".

        Returns:
            An object supporting indexing that returns str values or arrays of str.

        Raises:
            TypeError: If the dataset does not contain strings.
        """
        info = check_string_dtype(self.dtype)
        if info is None:
            raise TypeError("dataset does not contain strings")
        codec = encoding or info.encoding
        return _ReadAdapter(self, lambda value: _decode_strings(value, codec, errors))

    def astype(self, dtype):
        """Return a read-only view that converts values to dtype on read.

        Use "T" with NumPy 2 to read strings as native StringDType arrays.
        """
        target = np.dtype(dtype)
        return _ReadAdapter(self, lambda value: _convert_dtype(value, target))

    def read(self):
        """Return a copy of the complete dataset as a NumPy array or scalar.

        Raises:
            ValueError: If the file is closed.
            TypeError: If the dataset type or dataspace is unsupported.
            OSError: If HDF5 cannot read the data.
        """
        return self._info(True)

    def flush(self):
        """Flush this dataset for SWMR readers.

        Raises:
            ValueError: If the file is closed.
            OSError: If HDF5 cannot flush the dataset.
        """
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().sync(False, error)
        _attribute_error(status, error, None)

    def refresh(self):
        """Reload this dataset's metadata and data from its file.

        Raises:
            ValueError: If the file is closed.
            OSError: If HDF5 cannot refresh the dataset.
        """
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().sync(True, error)
        _attribute_error(status, error, None)

    def resize(self, size, axis=None):
        """Resize a chunked dataset, optionally changing only one axis.

        Args:
            size: New shape, or the new axis length when axis is given.
            axis: Optional axis to resize.

        Raises:
            ValueError: If the file is closed or the new shape is invalid.
            TypeError: If dimensions are invalid.
            OSError: If HDF5 cannot resize the dataset.
        """
        if not self._dataset:
            raise ValueError("dataset is not initialized")
        if axis is None:
            shape = _dataset_shape(size)
        else:
            axis = operator.index(axis)
            shape = list(self.shape)
            if axis < 0 or axis >= len(shape):
                raise ValueError("axis out of range")
            shape[axis] = _file_size(size, "size")
        cdef vector[uint64_t] dims
        for dim in shape:
            dims.push_back(dim)
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().resize(dims, error)
        if status == Status.InvalidPath:
            raise ValueError(error.decode("utf-8", "replace"))
        _attribute_error(status, error, None)

    def write(self, value):
        """Replace the complete dataset, broadcasting and converting the input.

        Args:
            value: Numeric value or array broadcastable to the dataset shape.

        Raises:
            ValueError: If the file is closed or the shape cannot be broadcast.
            TypeError: If the dataset type or input is unsupported.
            OSError: If HDF5 cannot write the data.
        """
        # Convert and broadcast into a private, C-contiguous buffer.
        shape, dtype = self._info(False)
        cdef AttributeData input
        keep = _pack_array(_convert_for_dataset(value, dtype, shape, "dataset"), input)

        # Revalidate metadata and write while holding the native file lock.
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().write(input, error)
        _attribute_error(status, error, None)

    def __getitem__(self, selection):
        """Read integer and positive-slice selections into independent storage."""
        if (selection is Ellipsis
                or isinstance(selection, tuple) and len(selection) == 0
                or isinstance(selection, slice) and selection == slice(None)):
            return self.read()

        if not self._dataset:
            raise ValueError("dataset is not initialized")
        cdef SelectionRequest request
        cdef DatasetSelection normalized
        cdef AttributeData output
        cdef string error
        cdef Status status
        if _selection_request(selection, request):
            with nogil:
                status = self._dataset.get().read_selection(request, output, error)
            if status == Status.InvalidSelection:
                raise IndexError(error.decode("utf-8", "replace"))
            if status != Status.SelectionFallback:
                _attribute_error(status, error, None)
                return _unpack_array(output, False)

        # Preserve Python index protocol and arbitrary-size index semantics.
        shape, dtype = self._info(False)
        _normalize_selection(selection, shape, dtype, normalized)
        with nogil:
            status = self._dataset.get().transfer(normalized, output, False, error)
        _attribute_error(status, error, None)
        return _unpack_array(output, False)

    def __setitem__(self, selection, value):
        """Write a broadcastable value to an integer or positive-slice selection."""
        if (isinstance(selection, tuple) and len(selection) == 0
                or isinstance(selection, slice) and selection == slice(None)):
            self.write(value)
            return

        # Normalize the selection and copy the converted, broadcast input.
        shape, dtype = self._info(False)
        cdef DatasetSelection normalized
        _normalize_selection(selection, shape, dtype, normalized)
        cdef AttributeData input
        keep = _pack_array(_convert_for_dataset(value, dtype, tuple(normalized.result_shape),
                                                "dataset"), input)

        # Revalidate the dataset snapshot and write only the selected elements.
        cdef string error
        cdef Status status
        with nogil:
            status = self._dataset.get().transfer(normalized, input, True, error)
        _attribute_error(status, error, None)


# Group interface
cdef class Group:
    """Access groups and links in an HDF5 file.

    Group operations are thread-safe with respect to the file instance that
    created them. Closing that file invalidates all its groups.

    Groups are obtained from a file or another group, not constructed directly.
    """

    cdef shared_ptr[GroupHandle] _group

    def __init__(self):
        """Reject direct construction of an unbound group."""
        raise TypeError("groups must be obtained from a file or group")

    @property
    def attrs(self):
        """Return a manager for this object's attributes."""
        if not self._group:
            raise ValueError("group is not initialized")
        cdef AttributeManager manager = AttributeManager.__new__(AttributeManager)
        manager._handle = make_shared[AttributeHandle](self._group)
        return manager

    cdef Status _resolve(self, object name, bint create, bint membership,
                      shared_ptr[GroupHandle]& output, bint require=False) except *:
        """Resolve a path under the file lock, optionally creating or testing it."""
        # Encode the path and invoke the native resolver without the GIL.
        if not self._group:
            raise ValueError("group is not initialized")
        cdef bytes encoded = _encode_name(name)
        cdef const char* path = encoded
        cdef bint utf8 = _needs_utf8(name)
        cdef string error
        cdef Status result
        with nogil:
            result = self._group.get().resolve(
                path, create, membership, output, error, utf8, require)

        # Preserve missing-link and dataset status for callers to handle.
        if result == Status.Closed:
            raise ValueError(error.decode("utf-8", "replace"))
        if result == Status.NotFound:
            return Status.NotFound
        if result == Status.IsDataset:
            return Status.IsDataset
        if result == Status.Unsupported:
            raise TypeError(error.decode("utf-8", "replace"))
        if result == Status.InvalidPath:
            raise ValueError(error.decode("utf-8", "replace"))
        if result != Status.Ok:
            raise OSError(error.decode("utf-8", "replace"))
        return Status.Ok

    def create_group(self, name):
        """Create a group, including missing intermediate groups.

        Args:
            name: Absolute or relative HDF5 group path.

        Returns:
            The newly created group.

        Raises:
            ValueError: If the file is closed or the path is invalid.
            TypeError: If the path traverses an unsupported link or object.
            OSError: If creation fails or the final name already exists.
        """
        # Create the native group before constructing its Python wrapper.
        cdef shared_ptr[GroupHandle] handle
        self._resolve(name, True, False, handle)
        cdef Group group = Group.__new__(Group)
        group._group = handle
        return group

    def require_group(self, name):
        """Return a group at name, creating missing groups along the path.

        Args:
            name: Absolute or relative HDF5 group path.

        Returns:
            The existing or newly created group.

        Raises:
            ValueError: If the file is closed or the path is invalid.
            TypeError: If the path refers to a non-group or unsupported link.
            OSError: If HDF5 cannot create or open the group.
        """
        cdef shared_ptr[GroupHandle] handle
        self._resolve(name, True, False, handle, True)
        cdef Group group = Group.__new__(Group)
        group._group = handle
        return group

    cdef object _dataset_at(self, object name, bint create, object value, object dtype,
                            shared_ptr[GroupHandle] parent, object settings):
        """Open or create a dataset under a safely resolved parent group."""
        # Validate the path and locate its parent.
        cdef bytes encoded = _encode_name(name)
        parts = [part for part in encoded.split(b"/") if part and part != b"."]
        if not encoded or not parts or b".." in parts or encoded.endswith(b"/"):
            raise ValueError("invalid dataset path")
        parent_path = b"/".join(parts[:-1])
        if encoded.startswith(b"/"):
            parent_path = b"/" + parent_path
        if not parent_path:
            parent_path = b"."
        cdef Status parent_status
        if not parent:
            parent_status = self._resolve(parent_path, False, False, parent)
            if parent_status == Status.NotFound:
                raise KeyError(name)
            if parent_status == Status.IsDataset:
                raise TypeError("dataset parent is not a group")
        if not parent:
            raise TypeError("dataset parent is not a group")

        # Prepare creation data, then open or create under the native lock.
        cdef AttributeData input
        cdef DatasetOptions options
        if create:
            shape, chunks, maxshape, compression, compression_opts, shuffle, fletcher32, fillvalue = settings
            if value is None:
                options.has_data = False
                requested = np.dtype("f4" if dtype is None else dtype)
                _check_metadata(requested, "dataset")
                if requested.fields is not None or requested.subdtype is not None:
                    raise TypeError(f"unsupported dataset dtype: {requested}")
                input.dtype = _dtype_name(requested)
                for dim in shape:
                    input.shape.push_back(dim)
            else:
                if dtype is None:
                    array = _prepare_array(value, "dataset")
                    if array.shape != shape:
                        array = array.reshape(shape)
                else:
                    requested = np.dtype(dtype)
                    _check_metadata(requested, "dataset")
                    if _dtype_class(requested) == "vlenarr":
                        value = _vlen_values(value, requested, ())
                    if np.shape(value) != shape:
                        value = np.asarray(value).reshape(shape)
                    array = _convert_for_dataset(value, requested, shape, "dataset")
                keep = _pack_array(array, input)
                requested = array.dtype
            if maxshape is not None:
                options.has_maxshape = True
                for dim in maxshape:
                    options.maxshape.push_back(18446744073709551615 if dim is None else dim)
            for dim in chunks:
                options.chunks.push_back(dim)
            if compression == "gzip":
                options.gzip_level = 4 if compression_opts is None else compression_opts
            options.shuffle = shuffle
            options.fletcher32 = fletcher32
            if fillvalue is not None:
                fill_keep = _pack_array(
                    _convert_for_dataset(fillvalue, requested, (), "dataset"), options.fill)
                options.has_fill = True
        cdef bytes leaf = parts[-1]
        cdef const char* leaf_ptr = leaf
        cdef shared_ptr[DatasetHandle] handle
        cdef bint utf8 = _needs_utf8(name)
        cdef string error
        cdef Status status
        with nogil:
            status = DatasetHandle.open(parent, leaf_ptr, create, input, options, handle, error, utf8)

        # Transfer shared native ownership to the Python wrapper.
        _attribute_error(status, error, name)
        cdef Dataset dataset = Dataset.__new__(Dataset)
        dataset._dataset = handle
        return dataset

    def create_dataset(self, name, shape=None, dtype=None, data=None, *, chunks=None,
                       maxshape=None, compression=None, compression_opts=None, shuffle=False,
                       fletcher32=False, fillvalue=None):
        """Create a dataset from data or a shape in an existing parent group.

        Args:
            name: Absolute or relative HDF5 dataset path.
            shape: Dataset dimensions. Required if data is omitted.
            data: Optional numeric or string scalar or array to store. Python str is stored as
                variable-length UTF-8, bytes as variable-length ASCII, and NumPy bytes
                arrays as fixed-length strings.
            dtype: Optional dtype to convert the data to, such as string_dtype().
            chunks: Chunk dimensions, True for automatic chunks, or None.
            maxshape: Maximum dimensions; None for an unlimited dimension.
            compression: "gzip" or None.
            compression_opts: gzip level from 0 to 9.
            shuffle: Enable the shuffle filter.
            fletcher32: Enable the Fletcher32 checksum filter.
            fillvalue: Optional scalar fill value for uninitialized elements.

        Returns:
            The newly created dataset.

        Raises:
            KeyError: If the parent group does not exist.
            ValueError: If the path is invalid or the file is closed.
            TypeError: If the datatype, parent, or link is unsupported.
            OSError: If the dataset cannot be created or written.
        """
        if data is not None and dtype is not None and _dtype_class(np.dtype(dtype)) == "vlenarr":
            data = _vlen_values(data, np.dtype(dtype), ())
        if shape is None:
            if data is None:
                raise TypeError("shape or data must be provided")
            shape = np.shape(data)
        else:
            shape = _dataset_shape(shape)
            if data is not None and np.prod(shape, dtype=object) != np.size(data):
                raise ValueError("shape is incompatible with data")
        if maxshape is not None:
            if isinstance(maxshape, int):
                maxshape = (maxshape,)
            maxshape = tuple(None if dim is None else _file_size(dim, "maxshape")
                             for dim in maxshape)
            if len(maxshape) != len(shape):
                raise ValueError("maxshape must match dataset rank")
            if any(limit is not None and dim > limit for dim, limit in zip(shape, maxshape)):
                raise ValueError("shape exceeds maxshape")
        if compression not in (None, "gzip"):
            raise ValueError("only gzip compression is supported")
        if compression_opts is not None:
            if compression != "gzip":
                raise ValueError("compression_opts requires gzip compression")
            compression_opts = operator.index(compression_opts)
            if not 0 <= compression_opts <= 9:
                raise ValueError("gzip compression level must be between 0 and 9")
        if not isinstance(shuffle, bool) or not isinstance(fletcher32, bool):
            raise TypeError("shuffle and fletcher32 must be bool")
        auto_chunk = maxshape is not None or compression is not None or shuffle or fletcher32
        if chunks is False and auto_chunk:
            raise ValueError("chunked storage is required for filters and maxshape")
        if not shape and (auto_chunk or chunks is True or bool(chunks)):
            raise ValueError("scalar datasets cannot be chunked")
        if chunks is True or (chunks is None and auto_chunk):
            itemsize = np.dtype(dtype).itemsize if dtype is not None else np.asarray(data).dtype.itemsize
            budget = max(1, 1024 * 1024 // max(1, itemsize))
            chosen = []
            for dim in reversed(shape):
                length = min(max(1, dim), 1024, budget)
                chosen.append(length)
                budget = max(1, budget // length)
            chunks = tuple(reversed(chosen))
        elif chunks is None or chunks is False:
            chunks = ()
        else:
            chunks = _dataset_shape(chunks)
        if chunks and (len(chunks) != len(shape) or any(dim == 0 for dim in chunks)):
            raise ValueError("chunks must contain one positive dimension per dataset axis")
        settings = (shape, chunks, maxshape, compression, compression_opts,
                    shuffle, fletcher32, fillvalue)
        cdef shared_ptr[GroupHandle] parent
        return self._dataset_at(name, True, data, dtype, parent, settings)

    def require_dataset(self, name, shape=None, dtype=None, *, data=None, exact=False,
                        **options):
        """Return a compatible dataset or create it from shape or data.

        An existing dataset must have the same shape. For shape-based requests,
        its dtype must match exactly when exact is True, or accept a safe cast
        from the requested dtype otherwise. Data-only calls retain exact dtype
        matching for compatibility.

        Args:
            name: Absolute or relative HDF5 dataset path.
            shape: Required shape when data is omitted.
            dtype: Requested dataset dtype.
            data: Optional data used to create or validate the dataset.
            exact: Require an exact dtype match for shape-based requests.
            **options: Dataset creation options when the dataset is missing.

        Returns:
            The existing compatible dataset or the newly created dataset.

        Raises:
            KeyError: If the parent group does not exist.
            ValueError: If the path is invalid or the file is closed.
            TypeError: If the existing dataset is incompatible or data is unsupported.
            OSError: If HDF5 cannot create or open the dataset.
        """
        if shape is None and data is None:
            raise TypeError("shape or data must be provided")
        if data is None and dtype is None:
            raise TypeError("dtype must be provided when data is omitted")
        expected = _prepare_array(data, "dataset") if data is not None else None
        requested_shape = _dataset_shape(shape) if shape is not None else expected.shape
        expected_dtype = np.dtype(dtype) if dtype is not None else expected.dtype
        if dtype is not None and data is not None:
            _check_metadata(expected_dtype, "dataset")
            _convert_for_dataset(data, expected_dtype, requested_shape, "dataset")

        try:
            dataset = self[name]
        except KeyError:
            return self.create_dataset(name, shape=shape, data=data, dtype=dtype, **options)

        if not isinstance(dataset, Dataset):
            raise TypeError("path does not refer to a dataset")
        if dataset.shape != requested_shape:
            raise TypeError(
                f"existing dataset has shape {dataset.shape}, expected {requested_shape}")
        if dtype is not None and (
                dataset.dtype != expected_dtype and
                (exact or shape is None or not np.can_cast(expected_dtype, dataset.dtype))):
            raise TypeError(
                f"existing dataset has dtype {dataset.dtype}, expected {expected_dtype}")
        if data is not None:
            _convert_for_dataset(data, dataset.dtype, dataset.shape, "dataset")
        return dataset

    def create_dataset_like(self, name, source, *, dtype=None):
        """Create a dataset using another dataset's data and dtype by default.

        Args:
            name: Absolute or relative HDF5 dataset path.
            source: Dataset whose data and dtype provide the defaults.
            dtype: Optional dtype to use instead of the source dtype.

        Returns:
            The newly created dataset.

        Raises:
            KeyError: If the parent group does not exist.
            ValueError: If the path is invalid or a file is closed.
            TypeError: If source is not a dataset or the data type is unsupported.
            OSError: If HDF5 cannot create or write the dataset.
        """
        if not isinstance(source, Dataset):
            raise TypeError("source must be a Dataset")
        target_dtype = source.dtype if dtype is None else dtype
        return self.create_dataset(name, data=source.read(), dtype=target_dtype)

    def keys(self):
        """Return a sorted snapshot of link names in this group.

        Returns:
            A list of names decoded with the filesystem encoding.

        Raises:
            ValueError: If the file is closed.
            OSError: If enumeration fails.
        """
        # Enumerate under the file lock before decoding names.
        if not self._group:
            raise ValueError("group is not initialized")
        cdef vector[string] names
        cdef string error
        cdef Status result
        with nogil:
            result = self._group.get().names(names, error)
        if result == Status.Closed:
            raise ValueError(error.decode("utf-8", "replace"))
        if result != Status.Ok:
            raise OSError(error.decode("utf-8", "replace"))
        return [_decode_name(<bytes>name) for name in names]

    def items(self):
        """Return a snapshot of link names and their opened objects.

        Returns:
            A list of ``(name, object)`` pairs.

        Raises:
            ValueError: If the file is closed.
            TypeError: If a listed link cannot be traversed.
            OSError: If enumeration or object opening fails.
        """
        return [(name, self[name]) for name in self.keys()]

    def values(self):
        """Return a snapshot of the objects linked from this group.

        Returns:
            A list of groups and datasets.

        Raises:
            ValueError: If the file is closed.
            TypeError: If a listed link cannot be traversed.
            OSError: If enumeration or object opening fails.
        """
        return [self[name] for name in self.keys()]

    def get(self, name, default=None):
        """Return the object at name, or default when the link is missing.

        Args:
            name: Absolute or relative HDF5 path.
            default: Value to return when name does not exist.

        Returns:
            The opened group or dataset, or default if the path is missing.

        Raises:
            ValueError: If the file is closed or the path is invalid.
            TypeError: If the path refers to an unsupported link or object.
            OSError: If HDF5 cannot open the object.
        """
        try:
            return self[name]
        except KeyError:
            return default

    @property
    def name(self):
        """Return this group's absolute HDF5 path.

        Raises:
            ValueError: If the owning file is closed.
            OSError: If HDF5 cannot retrieve the path.
        """
        if not self._group:
            raise ValueError("group is not initialized")
        cdef string path
        cdef string error
        cdef Status status
        with nogil:
            status = self._group.get().name(path, error)
        _attribute_error(status, error, None)
        return _decode_name(<bytes>path)

    @property
    def parent(self):
        """Return the parent group, or this root group for the file root.

        Raises:
            ValueError: If the owning file is closed.
            KeyError: If the parent group cannot be resolved.
        """
        cdef Group root
        cdef shared_ptr[GroupHandle] handle
        cdef Group parent
        path = self.name
        if path in ("/", b"/"):
            root = Group.__new__(Group)
            root._group = self._group
            return root
        separator = b"/" if isinstance(path, bytes) else "/"
        parent_path = path.rsplit(separator, 1)[0] or separator
        self._resolve(parent_path, False, False, handle)
        parent = Group.__new__(Group)
        parent._group = handle
        return parent

    @property
    def file(self):
        """Return a file wrapper sharing this group's native file handle.

        Raises:
            ValueError: If the group is not initialized.
        """
        if not self._group:
            raise ValueError("group is not initialized")
        return _file_from_handle(self._group.get().file_handle())

    def __getitem__(self, name):
        """Open a group or dataset at the requested path."""
        if not self._group:
            raise ValueError("group is not initialized")
        cdef bytes encoded = _encode_name(name)
        cdef const char* path = encoded
        cdef shared_ptr[GroupHandle] handle
        cdef shared_ptr[DatasetHandle] dataset_handle
        cdef string error
        cdef Status result
        with nogil:
            result = DatasetHandle.lookup(self._group, path, handle, dataset_handle, error)
        if result == Status.NotFound:
            raise KeyError(name)
        if result == Status.IsDataset:
            dataset = Dataset.__new__(Dataset)
            (<Dataset>dataset)._dataset = dataset_handle
            return dataset
        if result == Status.InvalidPath:
            raise ValueError(error.decode("utf-8", "replace"))
        _attribute_error(result, error, name)
        cdef Group group = Group.__new__(Group)
        group._group = handle
        return group

    def __delitem__(self, name):
        """Delete the link at name.

        Args:
            name: Absolute or relative HDF5 link path.

        Raises:
            KeyError: If the link does not exist.
            ValueError: If the file is closed or the path is invalid.
            TypeError: If name is not str or bytes.
            OSError: If HDF5 cannot remove the link.
        """
        if not self._group:
            raise ValueError("group is not initialized")
        cdef bytes encoded = _encode_name(name)
        parts = [part for part in encoded.split(b"/") if part and part != b"."]
        if not encoded or b".." in parts:
            raise ValueError("invalid link path")
        cdef const char* path = encoded
        cdef string error
        cdef Status status
        with nogil:
            status = self._group.get().remove_link(path, error)
        _attribute_error(status, error, name)

    def __contains__(self, name):
        """Return whether the requested supported link exists."""
        cdef shared_ptr[GroupHandle] unused
        return self._resolve(name, False, True, unused) == Status.Ok

    def __iter__(self):
        """Iterate over a snapshot of link names."""
        return iter(self.keys())

    def __len__(self):
        """Return the number of links in this group."""
        if not self._group:
            raise ValueError("group is not initialized")
        cdef uint64_t count
        cdef string error
        cdef Status status
        with nogil:
            status = self._group.get().count(count, error)
        _attribute_error(status, error, None)
        return count


# File interface
cdef FileVersion _file_version(object value) except *:
    """Translate a supported HDF5 format bound into its native enum."""
    if not isinstance(value, str):
        raise TypeError("libver bounds must be strings")
    if value == "earliest":
        return FileVersion.Earliest
    if value == "v108":
        return FileVersion.V108
    if value == "v110":
        return FileVersion.V110
    if value == "v112":
        return FileVersion.V112
    if value == "v114":
        return FileVersion.V114
    if value == "v200":
        return FileVersion.V200
    if value == "latest":
        return FileVersion.Latest
    raise ValueError(f"unsupported libver bound: {value!r}")


cdef size_t _file_size(object value, str label) except *:
    """Validate a nonnegative size without accepting booleans or floats."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer")
    try:
        number = operator.index(value)
    except TypeError:
        raise TypeError(f"{label} must be an integer") from None
    if number < 0:
        raise ValueError(f"{label} must not be negative")
    return number


cdef class File(Group):
    """Own an HDF5 file.

    Operations on the same instance are thread-safe. Close waits for in-flight
    operations, and later operations fail. This does not synchronize other
    libraries or processes accessing the file.

    Args:
        filename: Filesystem path of the file.
        mode: One of r, r+, w, w-, x, or a.
        driver: None for the default driver or "core" for an in-memory file.
        backing_store: Whether the core driver writes to disk; defaults to True.
        block_size: Core driver allocation increment in bytes; defaults to 65536.
        libver: Minimum format version, or a pair of minimum and maximum versions.
        rdcc_nslots: Raw chunk cache hash-table slots per dataset.
        rdcc_nbytes: Raw chunk cache capacity in bytes per dataset.
        rdcc_w0: Raw chunk cache preemption weight between 0 and 1.
        locking: None, a boolean, "true", "false", or "best-effort".
        swmr: Open a read-only SWMR reader.

    Raises:
        TypeError: If an option has an invalid type.
        ValueError: If an option, mode or filename is invalid.
        OverflowError: If an option exceeds the native size limit.
        OSError: If HDF5 cannot open or create the file.
    """

    cdef shared_ptr[FileHandle] _handle
    cdef bint _initialized

    def __cinit__(self):
        """Allocate the native file holder before Python initialization."""
        self._handle = make_shared[FileHandle]()
        self._initialized = False

    def __init__(self, filename, mode="r", *, driver=None, backing_store=None, block_size=None,
                 libver=None, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None, locking=None,
                 swmr=False):
        """Open or create the file with the requested mode."""
        # Validate arguments before attempting to open the native file.
        if self._initialized:
            raise RuntimeError("file is already initialized")
        if mode not in ("r", "r+", "w", "w-", "x", "a"):
            raise ValueError(f"invalid file mode: {mode!r}")

        cdef FileOptions options
        if not isinstance(swmr, bool):
            raise TypeError("swmr must be a bool")
        if swmr and mode != "r":
            raise ValueError("swmr=True is only supported for read-only mode")
        options.swmr_read = swmr
        if driver is not None and driver != "core":
            raise ValueError(f"unsupported file driver: {driver!r}")
        if driver == "core":
            options.driver = FileDriver.Core
            if backing_store is not None:
                if not isinstance(backing_store, bool):
                    raise TypeError("backing_store must be a bool")
                options.backing_store = backing_store
            if block_size is not None:
                options.block_size = _file_size(block_size, "block_size")
                if options.block_size == 0:
                    raise ValueError("block_size must be positive")
        elif backing_store is not None or block_size is not None:
            raise ValueError("backing_store and block_size require driver='core'")

        if libver is not None:
            if isinstance(libver, str):
                low, high = libver, "latest"
            elif isinstance(libver, tuple) and len(libver) == 2:
                low, high = libver
            else:
                raise TypeError("libver must be a version string or a pair of version strings")
            options.low = _file_version(low)
            options.high = _file_version(high)
            versions = {"earliest": 0, "v108": 1, "v110": 2, "v112": 3,
                        "v114": 4, "v200": 5, "latest": 5}
            if versions[high] == 0 or versions[low] > versions[high]:
                raise ValueError("invalid libver bounds")

        if locking is not None:
            if locking is True or locking == "true":
                options.locking = FileLocking.Enabled
            elif locking is False or locking == "false":
                options.locking = FileLocking.Disabled
            elif locking == "best-effort":
                options.locking = FileLocking.BestEffort
            else:
                raise ValueError(f"unsupported locking value: {locking!r}")
        if rdcc_nslots is not None:
            options.nslots = _file_size(rdcc_nslots, "rdcc_nslots")
            options.has_nslots = True
        if rdcc_nbytes is not None:
            options.nbytes = _file_size(rdcc_nbytes, "rdcc_nbytes")
            options.has_nbytes = True
        if rdcc_w0 is not None:
            options.w0 = float(rdcc_w0)
            if not 0 <= options.w0 <= 1:
                raise ValueError("rdcc_w0 must be between 0 and 1")
            options.has_w0 = True

        cdef bytes path = os.fsencode(filename)
        if b"\0" in path:
            raise ValueError("filename contains a null byte")
        cdef bytes encoded_mode = mode.encode("ascii")
        cdef const char* path_ptr = path
        cdef const char* mode_ptr = encoded_mode
        cdef string error
        cdef Status result

        # Open without the GIL and bind the root group on success.
        self._initialized = True
        with nogil:
            result = self._handle.get().open(path_ptr, mode_ptr, options, error)
        if result != Status.Ok:
            raise OSError(error.decode("utf-8", "replace"))
        self._group = GroupHandle.root(self._handle)

    def _cache_settings(self):
        """Return effective raw chunk cache settings for diagnostics."""
        cdef size_t nslots, nbytes
        cdef double w0
        cdef string error
        cdef Status result
        with nogil:
            result = self._handle.get().cache_settings(nslots, nbytes, w0, error)
        _attribute_error(result, error, None)
        return nslots, nbytes, w0

    def _core_settings(self):
        """Return effective core driver settings for diagnostics."""
        cdef size_t block_size
        cdef cpp_bool backing_store
        cdef string error
        cdef Status result
        with nogil:
            result = self._handle.get().core_settings(block_size, backing_store, error)
        _attribute_error(result, error, None)
        return block_size, bool(backing_store)

    @property
    def file(self):
        """Return this file object."""
        return self

    @property
    def name(self):
        """Return the filesystem path used to open this file."""
        if not self._handle:
            raise ValueError("file is not initialized")
        return os.fsdecode(<bytes>self._handle.get().filename())

    @property
    def parent(self):
        """Return this file object as the root group's parent."""
        return self

    @property
    def closed(self):
        """Return whether the file has been closed."""
        cdef cpp_bool result
        with nogil:
            result = self._handle.get().closed()
        return bool(result)

    @property
    def swmr_mode(self):
        """Return whether this file is open in SWMR reader or writer mode."""
        cdef cpp_bool enabled
        cdef string error
        cdef Status result
        with nogil:
            result = self._handle.get().swmr_mode(enabled, error)
        _attribute_error(result, error, None)
        return bool(enabled)

    @swmr_mode.setter
    def swmr_mode(self, value):
        """Start SWMR writing; disabling SWMR mode is not supported."""
        if value is not True:
            raise ValueError("SWMR mode can only be enabled")
        cdef string error
        cdef Status result
        with nogil:
            result = self._handle.get().start_swmr(error)
        _attribute_error(result, error, None)

    def close(self):
        """Close the file, waiting for operations on it to finish.

        Raises:
            OSError: If HDF5 cannot close the file.
        """
        cdef string error
        cdef Status result
        with nogil:
            result = self._handle.get().close(error)
        if result != Status.Ok:
            raise OSError(error.decode("utf-8", "replace"))

    def flush(self):
        """Flush HDF5 buffers for the file.

        Raises:
            ValueError: If the file is closed.
            OSError: If HDF5 cannot flush the file.
        """
        cdef string error
        cdef Status result
        with nogil:
            result = self._handle.get().flush(error)
        if result == Status.Closed:
            raise ValueError(error.decode("utf-8", "replace"))
        if result != Status.Ok:
            raise OSError(error.decode("utf-8", "replace"))

    def __enter__(self):
        """Return the open file for use in a context manager."""
        if self.closed:
            raise ValueError("file is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the file when leaving a context manager."""
        self.close()


cdef object _file_from_handle(shared_ptr[FileHandle] handle):
    """Create a Python file wrapper around an existing native file owner."""
    cdef File file = File.__new__(File)
    file._handle = handle
    file._group = GroupHandle.root(handle)
    file._initialized = True
    return file
