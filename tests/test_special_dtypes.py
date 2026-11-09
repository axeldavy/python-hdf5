"""Test enum, complex, float16, opaque, variable-length numeric types and object names."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

import hdf5
from hdf5 import Dataset, File

_WRITE_WITH_H5PY = """
import h5py, numpy as np, sys
with h5py.File(sys.argv[1], 'w') as f:
    enum = h5py.enum_dtype({'B': 2, 'A': 1}, basetype='i2')
    f.create_dataset('enum', data=np.array([1, 2, 1], dtype=enum))
    f.attrs.create('enum', np.array([2], dtype=enum))
    f['cplx'] = np.array([1 + 2j, 3j], dtype='c8')
    f['cplx128'] = np.array([1 + 2j], dtype='c16')
    f.attrs['cplx'] = np.complex128(1 - 1j)
    f['half'] = np.array([0.5, 2], dtype='f2')
    f.attrs['half'] = np.float16(1.5)
    f['opaque'] = np.array([b'abcd', b'wxyz'], dtype='V4')
    f['date'] = np.array(['2020-01-02'], dtype='M8[s]').astype(h5py.opaque_dtype('M8[s]'))
    vl = h5py.vlen_dtype('<i4')
    ds = f.create_dataset('vlen', (2,), dtype=vl)
    ds[0] = np.array([1, 2, 3], dtype='<i4')
    ds[1] = np.array([], dtype='<i4')
    arr = np.empty(1, dtype=object)
    arr[0] = np.arange(2, dtype='<i4')
    f.attrs.create('vlen', arr, dtype=vl)
    f.attrs['\u00e9t\u00e9'] = 1
    f['caf\u00e9'] = 2
    f.create_group('g\u00e9')
"""


def _ds(file: File, name: str | bytes) -> Dataset:
    """Return the dataset stored under name."""
    dataset = file[name]
    assert isinstance(dataset, Dataset)
    return dataset


def _run(python: str, code: str, path: Path) -> None:
    """Run code in the h5py interpreter with the file path as argv[1]."""
    subprocess.run([python, "-c", code, str(path)], check=True)


def test_special_dtypes_read_h5py_files(tmp_path: Path, h5py_python: str) -> None:
    """Read every special type written by h5py with matching values and dtypes."""
    path = tmp_path / "h5py.h5"
    _run(h5py_python, _WRITE_WITH_H5PY, path)

    with File(path, "r") as file:
        # Enums read as integers and expose their members through dtype metadata
        assert _ds(file, "enum")[:].tolist() == [1, 2, 1]
        assert hdf5.check_enum_dtype(_ds(file, "enum").dtype) == {"A": 1, "B": 2}
        assert _ds(file, "enum").dtype == np.dtype("<i2")
        assert file.attrs["enum"].tolist() == [2]

        # Complex and half floats
        np.testing.assert_array_equal(_ds(file, "cplx")[:], [1 + 2j, 3j])
        assert _ds(file, "cplx").dtype == np.dtype("c8")
        assert _ds(file, "cplx128")[:].dtype == np.dtype("c16")
        assert file.attrs["cplx"] == 1 - 1j
        np.testing.assert_array_equal(_ds(file, "half")[:], np.array([0.5, 2], dtype="f2"))
        assert _ds(file, "half").dtype == np.dtype("f2")
        assert file.attrs["half"] == np.float16(1.5)

        # Opaque data
        assert _ds(file, "opaque")[:].tolist() == [b"abcd", b"wxyz"]
        assert _ds(file, "opaque").dtype == np.dtype("V4")
        assert hdf5.check_opaque_dtype(_ds(file, "date").dtype)
        assert _ds(file, "date")[:].dtype == np.dtype("M8[s]")
        assert _ds(file, "date")[0] == np.datetime64("2020-01-02", "s")

        # Variable-length numeric arrays
        vlen = list(np.asarray(_ds(file, "vlen")[:], dtype=object))
        assert vlen[0].tolist() == [1, 2, 3] and vlen[0].dtype == np.dtype("<i4")
        assert len(vlen[1]) == 0
        assert hdf5.check_vlen_dtype(_ds(file, "vlen").dtype) == np.dtype("<i4")
        assert list(np.asarray(file.attrs["vlen"], dtype=object)[0]) == [0, 1]

        # Non-ASCII names decode as str
        assert "\u00e9t\u00e9" in file.attrs.keys()
        assert _ds(file, "caf\u00e9")[()] == 2
        assert "g\u00e9" in file


def test_special_dtypes_written_for_h5py(tmp_path: Path, h5py_python: str) -> None:
    """Write special types that h5py reads back with identical storage and values."""
    path = tmp_path / "ours.h5"
    enum = hdf5.enum_dtype({"B": 2, "A": 1}, "i2")
    vlen = hdf5.vlen_dtype("<i4")
    items = np.empty(2, dtype=vlen)
    items[0] = np.array([1, 2, 3], dtype="<i4")
    items[1] = np.array([], dtype="<i4")
    with File(path, "w") as file:
        file.create_dataset("enum", data=np.array([1, 2], dtype=enum))
        file.attrs["enum"] = np.array([2], dtype=enum)
        file.create_dataset("cplx", data=np.array([1 + 2j], dtype="c8"))
        file.attrs["cplx"] = np.complex128(2j)
        file.create_dataset("half", data=np.array([0.25], dtype="f2"))
        file.create_dataset("opaque", data=np.array([b"abcd"], dtype="V4"))
        file.create_dataset("date", data=np.array(["2021-03-04"], dtype=hdf5.opaque_dtype("M8[s]")))
        file.create_dataset("vlen", data=items)
        file.attrs["vlen"] = items
        file.create_dataset("\u00e9", data=1)
        file.create_dataset("raw", data=1)
        file.attrs["caf\u00e9"] = 1
        file.attrs["plain"] = 1

    check = """
import h5py, numpy as np, sys
with h5py.File(sys.argv[1]) as f:
    assert h5py.check_enum_dtype(f['enum'].dtype) == {'A': 1, 'B': 2}
    assert f['enum'][:].tolist() == [1, 2]
    assert f.attrs['enum'].tolist() == [2]
    assert f['cplx'].dtype == np.dtype('c8') and f['cplx'][0] == 1 + 2j
    assert f.attrs['cplx'] == 2j
    assert f['half'].dtype == np.dtype('f2') and f['half'][0] == 0.25
    assert f['opaque'].dtype == np.dtype('V4') and f['opaque'][0] == np.void(b'abcd')
    assert h5py.check_opaque_dtype(f['date'].dtype)
    assert f['date'][0] == np.datetime64('2021-03-04', 's')
    assert h5py.check_vlen_dtype(f['vlen'].dtype) == np.dtype('<i4')
    assert f['vlen'][0].tolist() == [1, 2, 3] and len(f['vlen'][1]) == 0
    assert f.attrs['vlen'][0].tolist() == [1, 2, 3]
    assert list(f.keys()) == ['cplx', 'date', 'enum', 'half', 'opaque', 'raw', 'vlen', '\\u00e9']
    assert f.id.links.get_info(b'raw').cset == 0
    assert f.id.links.get_info('\\u00e9'.encode()).cset == 1
    assert f.attrs.get_id('plain').get_name() == b'plain'
    assert f.attrs.get_id('caf\\u00e9').get_name() == 'caf\\u00e9'.encode()
    assert h5py.h5a.get_info(f.id, b'plain').cset == 0
    assert h5py.h5a.get_info(f.id, 'caf\\u00e9'.encode()).cset == 1
"""
    _run(h5py_python, check, path)


def test_invalid_utf8_names_are_returned_as_bytes(tmp_path: Path, h5py_python: str) -> None:
    """Return names that are not valid UTF-8 as bytes and accept bytes names."""
    path = tmp_path / "bytes.h5"
    _run(h5py_python, "import h5py, sys\nh5py.File(sys.argv[1], 'w').create_group(b'\\xff')", path)
    with File(path, "r+") as file:
        assert file.keys() == [b"\xff"]
        assert b"\xff" in file
        file.create_dataset(b"\xfe", data=1)
        assert _ds(file, b"\xfe")[()] == 1


def test_enum_helpers_and_validation() -> None:
    """Validate enum construction and round-trip metadata."""
    dtype = hdf5.enum_dtype({"A": 1}, "u1")
    assert hdf5.check_enum_dtype(dtype) == {"A": 1}
    assert hdf5.check_enum_dtype("i4") is None
    with pytest.raises(ValueError):
        hdf5.enum_dtype({"A": 300}, "u1")
    with pytest.raises(TypeError):
        hdf5.enum_dtype({"A": 1}, "f4")


def test_enum_roundtrip_and_conversion(tmp_path: Path) -> None:
    """Preserve enum members and convert integers into an existing enum dataset."""
    enum = hdf5.enum_dtype({"X": 0, "Y": 1}, "u2")
    with File(tmp_path / "enum.h5", "w") as file:
        file.create_dataset("e", data=[0, 1], dtype=enum)
        dataset = _ds(file, "e")
        assert hdf5.check_enum_dtype(dataset.dtype) == {"X": 0, "Y": 1}
        dataset[0] = 1
        assert dataset[:].tolist() == [1, 1]


def test_vlen_helpers_and_errors(tmp_path: Path) -> None:
    """Round-trip lists of arrays and reject invalid variable-length elements."""
    assert hdf5.check_vlen_dtype(hdf5.vlen_dtype("f8")) == np.dtype("f8")
    assert hdf5.check_vlen_dtype("i4") is None
    with pytest.raises(TypeError):
        hdf5.vlen_dtype("c8")
    with File(tmp_path / "vlen.h5", "w") as file:
        file.create_dataset("v", data=[[1.0, 2.0], [3.0]], dtype=hdf5.vlen_dtype("f8"))
        values = list(np.asarray(_ds(file, "v")[:], dtype=object))
        assert values[0].tolist() == [1.0, 2.0] and values[1].tolist() == [3.0]
        with pytest.raises(ValueError):
            file.create_dataset("bad", data=[np.zeros((2, 2))], dtype=hdf5.vlen_dtype("f8"))


def test_datetime_requires_opaque_tag(tmp_path: Path) -> None:
    """Reject untagged datetimes and report opaque helper results."""
    assert hdf5.check_opaque_dtype(hdf5.opaque_dtype("m8[s]"))
    assert not hdf5.check_opaque_dtype("i4")
    with pytest.raises(TypeError):
        hdf5.opaque_dtype([("a", "i4")])
    with File(tmp_path / "dt.h5", "w") as file:
        with pytest.raises(TypeError, match="opaque_dtype"):
            file.create_dataset("d", data=np.array(["2020"], dtype="M8[Y]"))
