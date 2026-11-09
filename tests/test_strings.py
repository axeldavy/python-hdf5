"""Test string datasets and attributes against h5py conventions."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from hdf5 import Dataset, File, StringInfo, check_string_dtype, string_dtype


def _dataset(file: File, name: str) -> Dataset:
    """Return the named dataset after checking its type."""
    dataset = file[name]
    assert isinstance(dataset, Dataset)
    return dataset


def test_string_dtype_helpers_describe_encoding_and_length() -> None:
    """Create and inspect string dtypes like h5py."""
    assert check_string_dtype(string_dtype()) == StringInfo("utf-8", None)
    assert check_string_dtype(string_dtype("ascii")) == StringInfo("ascii", None)
    assert check_string_dtype(string_dtype("utf-8", 4)) == StringInfo("utf-8", 4)
    assert check_string_dtype(np.dtype("S3")) == StringInfo("ascii", 3)
    assert check_string_dtype(np.dtype("i4")) is None
    with pytest.raises(ValueError):
        string_dtype("latin-1")
    with pytest.raises(ValueError):
        string_dtype("utf-8", 0)


def test_dataset_vlen_strings_read_as_bytes_and_asstr(tmp_path: Path) -> None:
    """Return bytes from datasets by default and decoded text through asstr."""
    with File(tmp_path / "vlen.h5", "w") as file:
        dataset = file.create_dataset("text", data=["a", "b\u00e9", ""])
        assert dataset.shape == (3,)
        assert check_string_dtype(dataset.dtype) == StringInfo("utf-8", None)
        assert dataset[:].tolist() == [b"a", b"b\xc3\xa9", b""]
        assert dataset[1] == b"b\xc3\xa9"
        assert dataset.asstr()[:].tolist() == ["a", "b\u00e9", ""]
        assert dataset.asstr()[1] == "b\u00e9"
        assert len(dataset.asstr()) == 3
        with pytest.raises(TypeError):
            file.create_dataset("number", data=[1]).asstr()


def test_scalar_and_bytes_inputs_choose_h5py_encodings(tmp_path: Path) -> None:
    """Store str as UTF-8 and bytes as ASCII variable-length strings."""
    with File(tmp_path / "scalar.h5", "w") as file:
        text = file.create_dataset("text", data="caf\u00e9")
        raw = file.create_dataset("raw", data=b"abc")
        assert text.shape == ()
        assert text[()] == "caf\u00e9".encode()
        assert check_string_dtype(raw.dtype) == StringInfo("ascii", None)
        assert raw.read() == b"abc"


def test_fixed_strings_roundtrip_and_truncate_by_bytes(tmp_path: Path) -> None:
    """Round-trip NumPy S arrays and truncate larger assignments like h5py."""
    with File(tmp_path / "fixed.h5", "w") as file:
        dataset = file.create_dataset("fixed", data=np.array([b"a", b"bcd"]))
        assert dataset.dtype == np.dtype("S3")
        np.testing.assert_array_equal(dataset[:], np.array([b"a", b"bcd"]))
        dataset[0] = b"abcdef"
        dataset[1] = "\u00e9\u00e9"
        assert dataset[:].tolist() == [b"abc", "\u00e9\u00e9".encode()[:3]]
        assert file.create_dataset("empty", data=np.array([], dtype="S1")).shape == (0,)
        assert dataset.asstr(errors="ignore")[0] == "abc"


def test_strings_convert_between_fixed_and_variable_on_write(tmp_path: Path) -> None:
    """Convert string values to the stored class without parsing numbers."""
    with File(tmp_path / "convert.h5", "w") as file:
        variable = file.create_dataset("variable", data=["x", "y"])
        variable[:] = np.array([b"p", b"q"])
        assert variable[:].tolist() == [b"p", b"q"]
        fixed = file.create_dataset("fixed", data=np.array([b"x", b"y"]))
        fixed[:] = ["m", "n"]
        assert fixed[:].tolist() == [b"m", b"n"]
        numbers = file.create_dataset("numbers", data=[1, 2])
        with pytest.raises(TypeError):
            numbers[:] = ["1", "2"]
        with pytest.raises(TypeError):
            variable[:] = [1, 2]
        assert numbers[:].tolist() == [1, 2]


def test_dataset_dtype_argument_converts_data(tmp_path: Path) -> None:
    """Honor explicit string and numeric dtypes at creation."""
    with File(tmp_path / "dtype.h5", "w") as file:
        fixed = file.create_dataset("fixed", data=["abc", "de"], dtype=string_dtype("utf-8", 4))
        assert check_string_dtype(fixed.dtype) == StringInfo("utf-8", 4)
        assert fixed[:].tolist() == [b"abc", b"de"]
        ascii_text = file.create_dataset("ascii", data=[b"a"], dtype=string_dtype("ascii"))
        assert check_string_dtype(ascii_text.dtype) == StringInfo("ascii", None)
        assert file.create_dataset("float", data=[1, 2], dtype="f4").dtype == np.dtype("f4")


def test_vlen_selections_read_and_write_partial_ranges(tmp_path: Path) -> None:
    """Read and write hyperslab selections of variable-length strings."""
    with File(tmp_path / "selection.h5", "w") as file:
        dataset = file.create_dataset("grid", data=[["a", "b", "c"], ["d", "e", "f"]])
        assert dataset[1, 1:].tolist() == [b"e", b"f"]
        dataset[0, ::2] = ["x", "z"]
        assert dataset.asstr()[0].tolist() == ["x", "b", "z"]
        assert dataset[0:0].shape == (0, 3)


def test_string_attributes_return_str_and_arrays(tmp_path: Path) -> None:
    """Decode variable attributes to str and keep fixed attributes as bytes."""
    with File(tmp_path / "attrs.h5", "w") as file:
        file.attrs["title"] = "h\u00e9llo"
        file.attrs["names"] = ["a", "b"]
        file.attrs["raw"] = b"abc"
        file.attrs["fixed"] = np.array([b"a", b"bc"])
        dataset = file.create_dataset("data", data=[1])
        dataset.attrs["label"] = "tag"
        assert file.attrs["title"] == "h\u00e9llo"
        assert file.attrs["names"].tolist() == ["a", "b"]
        assert file.attrs["raw"] == "abc"
        assert file.attrs["fixed"].tolist() == [b"a", b"bc"]
        assert dataset.attrs["label"] == "tag"
        file.attrs["title"] = 3
        assert file.attrs["title"] == 3


def test_invalid_string_values_are_rejected_without_changes(tmp_path: Path) -> None:
    """Reject embedded nulls, fixed-width Unicode, and mixed or foreign objects."""
    with File(tmp_path / "invalid.h5", "w") as file:
        file.attrs["keep"] = "value"
        with pytest.raises(ValueError, match="null"):
            file.attrs["keep"] = "a\0b"
        with pytest.raises(ValueError, match="null"):
            file.create_dataset("nul", data=[b"a\0b"])
        for bad in (np.array(["a"]), np.array([1, "a"], dtype=object),
                    np.array(["a", b"b"], dtype=object), np.array([], dtype=object)):
            with pytest.raises(TypeError, match="unsupported"):
                file.attrs["keep"] = bad
        assert file.attrs["keep"] == "value"
        assert "nul" not in file


def test_string_reads_are_consistent_during_concurrent_access(tmp_path: Path) -> None:
    """Read variable-length strings concurrently without leaking or corrupting data."""
    expected = [f"item-{i}" * 5 for i in range(200)]
    with File(tmp_path / "threads.h5", "w") as file:
        dataset = file.create_dataset("text", data=expected)
        file.attrs["text"] = expected

        def work(_: int) -> bool:
            """Read dataset and attribute strings and compare them with the source."""
            return (dataset.asstr()[:].tolist() == expected
                    and file.attrs["text"].tolist() == expected)

        with ThreadPoolExecutor(8) as pool:
            assert all(pool.map(work, range(64)))


def test_strings_interoperate_with_h5py(tmp_path: Path, h5py_python: str) -> None:
    """Read h5py string data and write string data that h5py reads the same way."""
    path = tmp_path / "interop.h5"
    create = (
        "import h5py, numpy as np, sys\n"
        "with h5py.File(sys.argv[1], 'w') as f:\n"
        " f['vlen'] = ['a', 'b\\u00e9']\n"
        " f['fixed'] = np.array([b'ab', b'c'])\n"
        " f['utf8_fixed'] = np.array([b'x'], dtype=h5py.string_dtype('utf-8', 3))\n"
        " f.attrs['title'] = 'h\\u00e9'\n"
        " f.attrs['names'] = ['p', 'q']\n"
        " f.attrs['fixed'] = np.array([b'zz'])\n"
    )
    subprocess.run([h5py_python, "-c", create, str(path)], check=True)
    with File(path, "r+") as file:
        assert _dataset(file, "vlen")[:].tolist() == [b"a", b"b\xc3\xa9"]
        assert _dataset(file, "vlen").asstr()[:].tolist() == ["a", "b\u00e9"]
        assert _dataset(file, "fixed")[:].tolist() == [b"ab", b"c"]
        assert check_string_dtype(_dataset(file, "utf8_fixed").dtype) == StringInfo("utf-8", 3)
        assert file.attrs["title"] == "h\u00e9"
        assert file.attrs["names"].tolist() == ["p", "q"]
        assert file.attrs["fixed"].tolist() == [b"zz"]
        file.create_dataset("new_vlen", data=["n", "\u00e9"])
        file.create_dataset("new_ascii", data=b"raw")
        file.create_dataset("new_fixed", data=np.array([b"fg"]))
        file.create_dataset("new_utf8", data=["ab"], dtype=string_dtype("utf-8", 4))
        file.attrs["new_title"] = "t\u00e9"
        file.attrs["new_raw"] = b"r"
    check = (
        "import h5py, numpy as np, sys\n"
        "with h5py.File(sys.argv[1]) as f:\n"
        " assert h5py.check_string_dtype(f['new_vlen'].dtype).encoding == 'utf-8'\n"
        " assert f['new_vlen'].asstr()[:].tolist() == ['n', '\\u00e9']\n"
        " assert h5py.check_string_dtype(f['new_ascii'].dtype).encoding == 'ascii'\n"
        " assert f['new_ascii'][()] == b'raw'\n"
        " assert f['new_fixed'][:].tolist() == [b'fg']\n"
        " info = h5py.check_string_dtype(f['new_utf8'].dtype)\n"
        " assert (info.encoding, info.length) == ('utf-8', 4)\n"
        " assert f.attrs['new_title'] == 't\\u00e9'\n"
        " assert f.attrs['new_raw'] == 'r'\n"
    )
    subprocess.run([h5py_python, "-c", check, str(path)], check=True)


def test_dataset_astype_reads_converted_values(tmp_path: Path) -> None:
    """Convert numeric reads and, with NumPy 2, native string arrays."""
    with File(tmp_path / "astype.h5", "w") as file:
        numbers = file.create_dataset("numbers", data=np.array([1, 2], dtype="i4"))
        assert numbers.astype("f8")[:].dtype == np.dtype("f8")
        text = file.create_dataset("text", data=["a", "bc"])
        if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
            assert text.astype("T")[:].tolist() == ["a", "bc"]
