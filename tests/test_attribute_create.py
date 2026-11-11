"""Test explicit attribute creation and interoperability."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

import hdf5
from hdf5 import AttributeManager, File, Group


@pytest.mark.parametrize("owner", ["file", "group", "dataset"])
def test_attribute_create_shape_and_dtype(tmp_path: Path, owner: str) -> None:
    """Create and replace attributes using explicit dtype and shape on all owners."""
    with File(tmp_path / "attributes.h5", "w") as file:
        group = file.create_group("group")
        dataset = file.create_dataset("data", data=[1])
        attrs = {"file": file, "group": group, "dataset": dataset}[owner].attrs
        attrs.create("values", [1, 2, 3, 4], (2, 2), ">i2")
        values = np.asarray(attrs["values"])
        assert values.shape == (2, 2) and values.dtype == np.dtype(">i2")
        np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
        attrs.create("values", [2.5], shape=(), dtype="f4")
        assert attrs["values"] == np.float32(2.5)
        attrs.create(b"\xff", np.array([], dtype="i2"), shape=(2, 0), dtype=">i4")
        assert np.asarray(attrs[b"\xff"]).shape == (2, 0)
        assert np.asarray(attrs[b"\xff"]).dtype == np.dtype(">i4")
        attrs.create("inferred", [1, 2, 3], shape=3)
        assert np.asarray(attrs["inferred"]).shape == (3,)


def test_attribute_create_special_interoperability(tmp_path: Path, h5py_python: str) -> None:
    """Preserve enum, string, opaque and vlen storage metadata for h5py."""
    path = tmp_path / "special.h5"
    vlen = hdf5.vlen_dtype("<i4")
    items = np.empty(2, dtype=vlen)
    items[0] = np.array([1, 2], dtype="<i4")
    items[1] = np.array([], dtype="<i4")
    with File(path, "w") as file:
        attrs = file.attrs
        attrs.create("enum", [1, 2], dtype=hdf5.enum_dtype({"one": 1, "two": 2}, "i2"))
        attrs.create("text", ["hello", "world"], dtype=hdf5.string_dtype())
        attrs.create("fixed", ["abcd", "xy"], dtype=hdf5.string_dtype("utf-8", 3))
        attrs.create("vlen", items, dtype=vlen)
        attrs.create("date", np.array(["2020-01-02"], dtype="M8[s]"),
                     dtype=hdf5.opaque_dtype("M8[s]"))
        attrs.create("raw", [b"ab", b"cd"], dtype="V2")
        assert np.asarray(attrs["text"]).tolist() == ["hello", "world"]
        assert np.asarray(attrs["fixed"]).tolist() == [b"abc", b"xy"]
        assert np.asarray(attrs["vlen"])[0].tolist() == [1, 2]
    script = (
        "import h5py, numpy as np, sys\n"
        "with h5py.File(sys.argv[1]) as f:\n"
        " a=f.attrs\n"
        " assert h5py.check_enum_dtype(a.get_id('enum').dtype)=={'one':1,'two':2}\n"
        " assert a['enum'].tolist()==[1,2]\n"
        " assert h5py.check_string_dtype(a.get_id('text').dtype).encoding=='utf-8'\n"
        " assert a['text'].tolist()==['hello','world']\n"
        " assert h5py.check_string_dtype(a.get_id('fixed').dtype).length==3\n"
        " assert a['fixed'].tolist()==[b'abc',b'xy']\n"
        " assert h5py.check_vlen_dtype(a.get_id('vlen').dtype)==np.dtype('<i4')\n"
        " assert a['vlen'][0].tolist()==[1,2] and len(a['vlen'][1])==0\n"
        " assert h5py.check_opaque_dtype(a.get_id('date').dtype)\n"
        " assert a['date'][0]==np.datetime64('2020-01-02','s')\n"
        " assert a['raw'].tolist()==[b'ab',b'cd']\n"
    )
    subprocess.run([h5py_python, "-c", script, str(path)], check=True)


@pytest.mark.parametrize("dtype", ["<i4", ">i4"])
def test_attribute_create_vlen_numeric_shape_and_lifetime(tmp_path: Path, dtype: str) -> None:
    """Keep scalar and reshaped vlen values independent, including empty elements."""
    items = np.empty(2, dtype=hdf5.vlen_dtype(dtype))
    items[0] = np.array([3, 5], dtype=dtype)
    items[1] = np.array([], dtype=dtype)
    with File(tmp_path / "vlen.h5", "w") as file:
        file.attrs.create("array", items, shape=(1, 2), dtype=hdf5.vlen_dtype(dtype))
        file.attrs.create("scalar", np.array([6, 7], dtype=dtype),
                          dtype=hdf5.vlen_dtype(dtype))
        array = np.asarray(file.attrs["array"])
        scalar = np.asarray(file.attrs["scalar"])
        assert array.shape == (1, 2)
        first = np.asarray(array[0, 0])
        assert first.dtype == np.dtype(dtype) and first.flags.writeable
        assert np.asarray(array[0, 1]).size == 0
        assert scalar.dtype == np.dtype(dtype) and scalar.tolist() == [6, 7]
        first[0] = 9
        assert np.asarray(file.attrs["array"])[0, 0].tolist() == [3, 5]
    assert first.tolist() == [9, 5] and scalar.tolist() == [6, 7]


def test_attribute_create_validation_preserves_existing(tmp_path: Path) -> None:
    """Reject invalid inputs before deleting an existing attribute."""
    with File(tmp_path / "errors.h5", "w") as file:
        attrs = file.attrs
        attrs["value"] = 42
        for kwargs in (
            {"shape": (2,)},
            {"shape": (-1,)},
            {"shape": (1,) * 33},
            {"dtype": np.dtype("i4", metadata={"unknown": True})},
        ):
            with pytest.raises((TypeError, ValueError)):
                attrs.create("value", [1], **kwargs)  # type: ignore[arg-type]
            assert attrs["value"] == 42
        with pytest.raises(ValueError):
            attrs.create("", 1)
        with pytest.raises(ValueError):
            attrs.create("bad\0name", 1)
        with pytest.raises(TypeError):
            attrs.create(1, 1)  # type: ignore[arg-type]
        with pytest.raises(UnicodeEncodeError):
            attrs.create("value", "\u00e9", dtype=hdf5.string_dtype("ascii"))
        assert attrs["value"] == 42
        with pytest.raises(ValueError, match="not initialized"):
            AttributeManager.__new__(AttributeManager).create("x", 1)
    with pytest.raises(ValueError, match="closed"):
        attrs.create("value", 1)
    with File(tmp_path / "errors.h5") as file:
        with pytest.raises(OSError):
            file.attrs.create("value", [1, 2])
        assert file.attrs["value"] == 42


def test_attribute_create_cross_handle_visibility_and_race(tmp_path: Path) -> None:
    """Serialize replacements across file handles and expose completed writes."""
    path = tmp_path / "race.h5"
    with File(path, "w") as first:
        first.create_group("group")
        with File(path, "r+") as second:
            left = first["group"]
            right = second["group"]
            assert isinstance(left, Group) and isinstance(right, Group)
            left.attrs.create("value", [1, 1], dtype="i4")
            np.testing.assert_array_equal(right.attrs["value"], [1, 1])
            barrier = Barrier(2)

            def writer(marker: int, owner: Group) -> None:
                """Replace the shared attribute with an untorn marker."""
                barrier.wait()
                for _ in range(100):
                    owner.attrs.create("value", [marker] * 16, shape=(4, 4), dtype="i4")
                    current = np.asarray(owner.attrs["value"])
                    assert current.shape == (4, 4)
                    assert np.all(current == current.flat[0])

            with ThreadPoolExecutor(max_workers=2) as executor:
                jobs = [executor.submit(writer, 1, left), executor.submit(writer, 2, right)]
                for job in jobs:
                    job.result(timeout=30)
            np.testing.assert_array_equal(left.attrs["value"], right.attrs["value"])
