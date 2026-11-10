# python-hdf5

If you fell on this page by mistake you like want to look at [h5py](https://github.com/h5py/h5py) instead.
It is the reference, stable and widely used Python interface to HDF5.

However if you are an h5py user looking for a similar library with lower overhead and fewer crashes,
here is a project attempting to provide exactly that.

This project is an early-stage Python interface to HDF5, aiming for speed and thread-safety.
To achieve this, it uses the design I experimented with (python-nng)[https://github.com/axeldavy/python-nng]:
- Calls to the C native API of the hdf5 library
- Encapsulation in light modern C++ to control resource ownership and release order
- Cython wrapper to the C++ layer for Python exposure, with careful GIL management

The reason we use a C++ layer is because Python does not provide enough control on
the release of important resources in a deterministic manner. For instance,
no matter how you do it (`__del__`, finalizers, etc) you cannot guarantee the order
of resource release in Python. When Python exists, it will force all items to release
in whatever order it chooses. For many Python C extensions this is unacceptable and can lead
to crashes or corruption on Python exit. This also matters for HDF5 in order to properly release file,
group, and dataset resources. You don't want your file to end up locked and corrupted.

The lightweight C++ layer solves this issue with shared ownership, which ensures items are
released in a well-defined order, independent of Python's garbage collection. Its ownership
graph is acyclic: groups retain their file, datasets retain their parent group, and dataset
attribute handles retain their dataset owner. Python wrapper references do not determine the
native resource release order.
I wish more Python libraries adopted this approach as it is a really efficient pattern for resource management.

This project is far from feature complete, but it should have the basic operations needed for many use cases.
In addition it already proves faster than h5py for most implemented operations. See [Support summary](#support-summary) for what is and is not
supported.

```python
from hdf5 import File

with File("example.h5", "w") as file:
    group = file.create_group("measurements/run1")
    group.attrs["sample_count"] = 100
    dataset = group.create_dataset("values", data=[1.0, 2.0])
    dataset.attrs["channel"] = 1
    print(dataset[:])
    print(list(file["measurements"]))
    file.flush()
```

`File` accepts modes `r`, `r+`, `w`, `w-`, `x`, and `a`. Operations on the
same instance are synchronized; closing waits for in-flight operations, and
flushing a closed file raises `ValueError`. Use a context manager or `close()`
to report close errors. Native cleanup is a fallback and reports failures to
standard error.

### File-access options

Optional keyword arguments preserve the HDF5 defaults when omitted:

- `driver="core"` loads the named file into memory. `backing_store=True` (the
  default) writes changes to the named disk file on flush or close;
  `backing_store=False` discards changes instead. `block_size` sets the core
  driver's allocation increment in bytes (default 65536). These two options
  require the core driver. A filename is still required; opening a byte image
  or a Python file-like object is not supported.
- `rdcc_nslots`, `rdcc_nbytes`, and `rdcc_w0` configure the raw chunk cache
  defaults for each dataset. The first two are nonnegative integers; the
  preemption weight is between 0 and 1. Omitted values retain HDF5 defaults.
  These settings affect chunked datasets, not the metadata cache.
- `libver` sets HDF5 format bounds as a minimum string or a `(minimum, maximum)`
  tuple. Supported values are `"earliest"`, `"v108"`, `"v110"`, `"v112"`,
  `"v114"`, `"v200"`, and `"latest"`. A single value uses `"latest"` as
  the maximum. Newer formats may not be readable by older HDF5 libraries.
- `locking` accepts `True`/`"true"`, `False`/`"false"`, or `"best-effort"`.
  Best-effort still locks when available but tolerates filesystems without
  locking support. The `HDF5_USE_FILE_LOCKING` environment variable can override
  this setting. Disabling locks does not make concurrent process access safe:
  follow HDF5's access rules or risk corruption.

```python
with File("scratch.h5", "w", driver="core", backing_store=False,
          rdcc_nbytes=4 * 1024 * 1024, libver="latest") as file:
    file.create_dataset("values", data=[1, 2, 3])
```

`File` and `Group` provide `create_group`, `require_group`, path lookup,
membership, `keys`, `items`, `values`, `get`, iteration and `len`. These
collection methods return snapshots, with `items()` returning `(name, object)`
pairs and `values()` returning the linked objects. `get()` returns its default
only when the link is missing; unsupported links and other errors are still
reported. Group and dataset `name`, `parent` and `file` expose object paths and owners;
for a `File`, `name` is its filesystem path and `parent`/`file` return itself.
`del group[name]` removes a link. `keys()` returns a name-sorted snapshot. Group paths may
be absolute or relative, and creation makes missing intermediate groups.
Names may be strings or bytes (see "Special types and names" below).
Empty paths and `..` components are rejected. Hard-linked groups are supported;
soft, external and user-defined links are listed but cannot be traversed or
tested for membership. Dataset links resolve to `Dataset` objects. Closing a
file invalidates all its groups and datasets; otherwise child objects keep the
native file open even if the Python `File` object is collected.

`file.attrs`, `group.attrs` and `dataset.attrs` support reading, writing, deleting, membership,
iteration, `keys()` and `len`. Attributes may be numeric NumPy arrays or scalars:
signed and unsigned 8/16/32/64-bit integers, 16/32/64-bit floats, complex,
and booleans, plus the strings and special types described below.
Scalar reads return NumPy scalars; array reads return independent arrays. Byte
order and zero-length dimensions are preserved. Compound, array and reference
types and null dataspaces are not yet supported. Dtype metadata that cannot be
stored is rejected with `TypeError` rather than dropped.
`attrs.create(name, data, shape=None, dtype=None)` creates or replaces an attribute
with an explicit dtype and/or shape. A supplied shape must have the same number
of elements as the data; this operation reshapes rather than broadcasts. Supported
special dtypes retain their HDF5 metadata.
Dataset datatype descriptions are cached per open dataset handle because an HDF5 dataset's
datatype is immutable; its dataspace and data are not cached. Selected reads use the current
dataspace, and writes revalidate the shape after preparing the input.
Callers must not mutate input arrays concurrently with a write. Attribute managers do not cache
values or names: a completed change is visible to subsequent operations on other managers of the
same object, including through another file handle in the same process. Attribute operations
through this library are serialized across handles for the same HDF5 file, to guard against unsafe
concurrent deletion and access. Callers should still synchronize sequences of operations on the
same attribute: for example, a membership check followed by a read is not atomic. Replacement
deletes the old attribute before creating and writing the new one; it is not transactional, and
an HDF5 failure after deletion can leave the old value lost or the new value incomplete. HDF5
file locking and its cross-process access rules apply; file locking does not provide transactional
replacement.
`astype` converts numeric to numeric and text to text only; other conversions raise `TypeError`.
`"a/b" in file` is `False` when `a` is a dataset.
Attribute managers retain their group and file until released, and are
invalidated when the file is explicitly closed.

`create_dataset(name, data=...)` or `create_dataset(name, shape=..., dtype=...)`
creates a dataset in an existing parent group;
unlike `create_group`, it does not create missing intermediate groups.
Chunked creation supports `chunks`, `maxshape` (with `None` for unlimited axes),
`compression="gzip"` and compression levels, `shuffle`, `fletcher32`, and scalar
`fillvalue` for fixed-size types. Chunking is selected automatically when needed.
Chunked datasets can be extended or shrunk with `dataset.resize(shape)` or
`dataset.resize(size, axis=...)`. Data beyond a shrunk extent is discarded.
`require_dataset(name, shape, dtype, exact=False)` returns a same-shape dataset
whose dtype can safely accept the requested dtype, or creates one. The existing
`require_dataset(name, data=..., dtype=...)` form remains available and requires
an exact dtype match when a dtype is specified.
`create_dataset_like(name, source)` copies a dataset's data and dtype, with an
optional dtype override. Dataset
`shape` and `dtype` describe the stored values. `read()` and `write(value)`,
or `dataset[()]` and `dataset[:]`, operate on the complete dataset. Writes
broadcast and convert numeric values to the dataset shape and dtype using
NumPy; read results are independent arrays. Other selections, including partial
positive-step slices, integer indices, tuples and a single ellipsis, are supported
for reads and writes. Integer axes are removed; negative indices count from the
end, and empty slices are supported. Slice writes broadcast to the selected
shape. Negative and zero slice steps, boolean/fancy indexing, and new axes are
not supported. Basic selections access HDF5 hyperslabs without reading the whole
dataset. The supported types
match attribute types above. Compound, array, reference and other datatypes, null
dataspaces, and datasets requiring unavailable compression filters are not yet
supported. Writes are not transactional.

### SWMR access

Create objects first in a file opened with `libver="latest"`, then set
`file.swmr_mode = True` to start the single writer. Publish changes with
`dataset.flush()` (or `file.flush()`). Readers in other processes open with
`File(path, "r", swmr=True)` and call `dataset.refresh()` before observing
new data and shapes. Coordinate reader/writer notifications separately; flush
does not notify readers. Follow HDF5 SWMR restrictions, including creating the
necessary objects before entering writer mode and using a filesystem with the
required POSIX I/O semantics. SWMR is not a multi-writer protocol or a
transaction/durability guarantee.

### Special types and names

Special types follow the [h5py conventions](https://docs.h5py.org/en/stable/special.html).
Complex values are stored as HDF5 compounds with members `r` and `i`, `float16`
as a 2-byte float, and enums as HDF5 enums over an integer base (little-endian
or one byte). `enum_dtype(values, basetype)` and `check_enum_dtype(dtype)` attach
and read the member mapping; reads return plain integers whose dtype carries the
members. NumPy `V<n>` is stored as opaque data, and `opaque_dtype(dtype)` tags
datetimes and other dtypes as opaque (`NUMPY:<dtype>` tag), so untagged
`datetime64`/`timedelta64` are rejected. `vlen_dtype(base)` and `check_vlen_dtype(dtype)`
describe variable-length arrays of integers or floats, read as object arrays of
1-D arrays. Dtype metadata that cannot be stored raises `TypeError` instead of
being dropped. Enum values are range-checked against the base type.

Names are encoded like h5py: `str` as UTF-8 and `bytes` unchanged. Links and
attributes with non-ASCII `str` names are marked UTF-8, all others ASCII.
`keys()` returns `str` for valid UTF-8 names and `bytes` otherwise.

### Strings

String handling follows the [h5py conventions](https://docs.h5py.org/en/stable/strings.html).
Python `str` is stored as a variable-length UTF-8 string, `bytes` as variable-length
ASCII, lists of `str`/`bytes` and NumPy object arrays of them as variable-length arrays,
and NumPy `S` arrays as null-padded fixed-length strings; NumPy 2 `StringDType`
arrays are written as variable-length UTF-8. Fixed-width Unicode (`U`) arrays,
mixed `str`/`bytes` object arrays and strings with embedded null bytes are rejected.
Reads differ between datasets and attributes, as in h5py: dataset variable-length
strings are returned as `bytes` and attribute ones as `str` (UTF-8 with
`surrogateescape`); fixed-length strings are always NumPy `S` values. Use
`dataset.asstr(encoding=None, errors="strict")[...]` to decode dataset reads and
`dataset.astype("T")[...]` for NumPy 2 string arrays. `string_dtype(encoding, length)`
and `check_string_dtype(dtype)` mirror their h5py counterparts, and
`create_dataset(..., dtype=...)` converts data on creation. Writing to an
existing dataset converts strings to its stored class and truncates fixed-length
values by bytes; strings are never parsed as numbers.

## Support summary

| Area | Supported | Not supported yet |
|---|---|---|
| Files | Modes `r`, `r+`, `w`, `w-`, `x`, `a`; `flush`, `close`, context manager; core in-memory driver; raw chunk cache, libver and file-locking options; thread-safe per-file locking | Other drivers, file images and file-like objects, metadata cache configuration, user blocks |
| Groups | Create (with intermediate groups), lookup, membership, `keys`, iteration, `len`; absolute and relative paths | Soft, external and user-defined links (listed, not traversed), `..` paths, deleting or moving links, `visit`/`walk`, creation properties |
| Datasets | Create from data or shape and dtype; chunking, gzip, shuffle, Fletcher32, fixed-size fill values, resize; `shape`, `dtype`, `read`/`write`, positive-step slices, integers, tuples, ellipsis, `asstr`, `astype`, `len` | Variable-length fill values, other compression filters, creation-property introspection, negative or zero steps, fancy or boolean indexing, new axes, null dataspaces, region and virtual datasets |
| Attributes | Read, write, explicit `create` with dtype and shape, delete, membership, iteration, `keys`, `len` on files, groups and datasets | Atomic replacement, null dataspaces |
| Numeric types | 8/16/32/64-bit signed and unsigned integers, float16/32/64, complex64/128, bool, either byte order | 128-bit and extended floats, bit fields |
| Strings | Variable-length and fixed-length, UTF-8 and ASCII, h5py-compatible reads and writes | Fixed-width `U` arrays, mixed `str`/`bytes` arrays, embedded null bytes |
| Special types | Enums, opaque (including tagged `datetime64`/`timedelta64`), variable-length integer and float arrays | Compound, array, object and region references, time types, bitfields |
| Names | `str` as UTF-8, `bytes` unchanged, character set flags like h5py | Names containing null bytes |
| Convenience API | `keys`, `items`, `values`, `get`, `name`, `parent`, `file`, `require_group`, `require_dataset`, `create_dataset_like`, iteration, `len`, `in`, `[]`, `del group[name]` | |
| SWMR | Reader open, writer activation, dataset flush and refresh | Multi-writer access, process notifications |
| Not implemented | | Dimension scales, parallel HDF5, Python-level `Dataset.fields`/`dims`, h5py's low-level `h5py.h5*` API |

Interoperability tests use h5py when it is available. To use h5py from a
different interpreter, run `HDF5_TEST_H5PY_PYTHON=/path/to/python pytest`.

## Benchmarking

Install the optional benchmark dependency in the same interpreter as the
library and run the comparison from the repository root:

```sh
python -m pip install -e '.[benchmark]'
python benchmarks/compare.py --iterations 100 --repeats 5 --warmup 1 \
    --size-mib 8 --threads 4 --json benchmark-results.json
```

Use `--scenario` multiple times to select workloads, `--directory` to choose
the parent filesystem for temporary fixtures, or `--help` for all options.
The benchmark covers file open/close, group operations, scalar attributes,
dataset creation, whole-array and slice reads/writes, and threaded reads with a
shared handle, independently opened handles, or separate files. Each implementation
receives a copy of the same uncompressed fixture. Correctness checks run outside
timed batches; failures abort the run. Backend order alternates between samples.

The report includes the interpreter, GIL state, library versions, raw samples,
median wall time per operation, the h5py/python-hdf5 time ratio (above 1 favors
python-hdf5), and MiB/s for data operations. Threaded timings are aggregate batch
time divided by the completed operation count; they include scheduling and
synchronization, not individual request latency. The independent-handle case
includes opening and closing a handle per read. Large transfers use
`max(1, iterations // 20)` operations per sample. Threaded runs use
`threads * max(1, iterations // threads)` operations. All timings are warm-cache
workloads: setup, verification, and final flush/close are outside timed regions,
so writes do not measure durable storage. h5py and this library may bundle
different HDF5 versions and build options; compare results only with their
reported environment and configuration.

The extension bundles HDF5 2.2.0, built with thread safety enabled: it is
statically linked on Linux and macOS and shipped as a private DLL alongside the
extension on Windows, where HDF5 does not support static thread-safe builds.
Building from source requires a C and C++ compiler, CMake, and network access to
download the pinned HDF5 source. The build verifies its SHA-256 checksum. The
statically linked HDF5 symbols are hidden, and the Windows DLL has a unique
name, to reduce conflicts with other extensions. HDF5
file locks and cross-process access rules still apply. The initial build does
includes gzip/deflate using zlib; building requires zlib development files. HDF5 license notices are installed under
`hdf5/licenses`.

HDF5 operations may serialize internally even when called from different
threads. Free-threaded Python without the GIL has not yet been audited; importing
the extension enables the GIL on free-threaded builds. Neither C++ destructors
nor `flush()` can guarantee recovery or durability after a crash or power loss.
HDF5 may leave superblock status flags after an unclean shutdown; do not clear
them without checking the file and following HDF5 recovery guidance.
