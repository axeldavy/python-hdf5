"""Test zero-copy transfers, new dtype round trips and name handling."""

import gc
from pathlib import Path

import numpy as np
import pytest

import hdf5
from hdf5 import Dataset, File


def _ds(file: File, name: str) -> Dataset:
    """Return the dataset stored under name."""
    dataset = file[name]
    assert isinstance(dataset, Dataset)
    return dataset


@pytest.mark.parametrize("dtype", ["f2", "<c8", ">c16", ">f2", "?", "i1"])
def test_transfer_new_numeric_dtypes_roundtrip(tmp_path: Path, dtype: str) -> None:
    """Round-trip datasets and attributes, preserving dtype, shape and values."""
    values = (np.arange(12).reshape(3, 4) * (1 + 1j if "c" in dtype else 1)).astype(dtype)
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=values)
        file.attrs["a"] = values
        dataset = _ds(file, "d")
        for result in (dataset[:], dataset[()], file.attrs["a"]):
            assert isinstance(result, np.ndarray)
            assert result.dtype == values.dtype
            np.testing.assert_array_equal(result, values)
        np.testing.assert_array_equal(dataset[1:3, ::2], values[1:3, ::2])


def test_transfer_read_result_is_independent_and_writable(tmp_path: Path) -> None:
    """Return writable arrays that stay valid after the file and dataset are gone."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=np.arange(1000, dtype="i8"))
        dataset = _ds(file, "d")
        first = dataset[:]
        second = dataset[:]
        assert isinstance(first, np.ndarray) and isinstance(second, np.ndarray)
        assert first.flags.writeable and first.flags.c_contiguous
        first[0] = -1
        assert second[0] == 0
        assert np.asarray(dataset[:])[0] == 0
    del dataset
    gc.collect()
    assert first[0] == -1 and first[999] == 999


def test_transfer_write_does_not_alias_or_mutate_input(tmp_path: Path) -> None:
    """Leave the caller array untouched and unlinked from the stored data."""
    source = np.arange(10, dtype="i4")
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=source)
        dataset = _ds(file, "d")
        dataset[:] = source
        source[:] = 99
        assert dataset[:].tolist() == list(range(10))
        assert source.tolist() == [99] * 10


@pytest.mark.parametrize(
    "make",
    [
        lambda: np.arange(20, dtype="i4")[::2],
        lambda: np.arange(10, dtype=">i4"),
        lambda: np.arange(10, dtype="i8"),
        lambda: np.arange(10, dtype="f8"),
        lambda: np.asfortranarray(np.arange(10, dtype="i4").reshape(2, 5)).T.T,
        lambda: np.arange(10, dtype="i4").reshape(2, 5)[:, ::-1],
        lambda: list(range(10)),
    ],
)
def test_transfer_write_converts_mismatching_layouts(tmp_path: Path, make) -> None:
    """Convert strided, byte-swapped, wider or list inputs to the stored dtype."""
    value = make()
    shape = (2, 5) if np.ndim(value) == 2 else (10,)
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=np.zeros(shape, dtype="i4"))
        dataset = _ds(file, "d")
        dataset[:] = value
        np.testing.assert_array_equal(dataset[:], np.asarray(value, dtype="i4"))


def test_transfer_write_readonly_and_broadcast_inputs(tmp_path: Path) -> None:
    """Accept read-only arrays and broadcast views without copying errors."""
    readonly = np.arange(6, dtype="f8")
    readonly.flags.writeable = False
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=np.zeros((4, 6)))
        dataset = _ds(file, "d")
        dataset[:] = np.broadcast_to(readonly, (4, 6))
        assert dataset[3].tolist() == readonly.tolist()
        dataset[0] = readonly
        dataset[1:3, 0] = 7.0
        assert dataset[1, 0] == 7.0 and dataset[2, 0] == 7.0


def test_transfer_empty_and_scalar_reads(tmp_path: Path) -> None:
    """Read zero-size and scalar datasets without a native block."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("empty", data=np.zeros((0, 3), dtype="f4"))
        file.create_dataset("scalar", data=np.float32(2.5))
        file.create_dataset("cplx", data=np.complex64(1 + 2j))
        assert _ds(file, "empty")[:].shape == (0, 3)
        assert _ds(file, "empty")[:].dtype == np.dtype("f4")
        scalar = _ds(file, "scalar")[()]
        assert scalar == np.float32(2.5) and type(scalar) is np.float32
        assert _ds(file, "cplx")[()] == 1 + 2j
        assert _ds(file, "scalar")[...] == np.float32(2.5)


def test_transfer_metadata_dtype_arrays_are_written_in_place_safely(tmp_path: Path) -> None:
    """Keep enum metadata for matching arrays and store the same values."""
    enum = hdf5.enum_dtype({"A": 0, "B": 1}, "u1")
    values = np.array([0, 1, 1, 0], dtype=enum)
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("e", data=values)
        dataset = _ds(file, "e")
        assert hdf5.check_enum_dtype(dataset.dtype) == {"A": 0, "B": 1}
        dataset[:] = values[::-1]
        assert dataset[:].tolist() == [0, 1, 1, 0]
        dataset[1:3] = 0
        assert dataset[:].tolist() == [0, 0, 0, 0]
        dataset[0] = 1
        assert dataset[0] == 1


def test_transfer_special_dtype_selections_and_attributes(tmp_path: Path) -> None:
    """Support slices and attributes for complex, opaque and variable-length data."""
    vlen = hdf5.vlen_dtype("<i4")
    items = np.empty(3, dtype=vlen)
    for index in range(3):
        items[index] = np.arange(index, dtype="<i4")
    opaque = np.array([b"ab", b"cd", b"ef"], dtype="V2")
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("v", data=items)
        file.create_dataset("o", data=opaque)
        file.attrs["o"] = opaque
        file.attrs["v"] = items
        dataset = _ds(file, "v")
        part = dataset[1:]
        assert [item.tolist() for item in np.asarray(part, dtype=object)] == [[0], [0, 1]]
        dataset[0] = np.array([5, 6], dtype="<i4")
        assert dataset[0].tolist() == [5, 6]
        assert [item.tolist() for item in np.asarray(file.attrs["v"], dtype=object)] == [[], [0], [0, 1]]
        assert _ds(file, "o")[1:].tolist() == [b"cd", b"ef"]
        assert file.attrs["o"].tolist() == [b"ab", b"cd", b"ef"]
        _ds(file, "o")[0] = np.void(b"zz")
        assert _ds(file, "o")[0] == np.void(b"zz")


@pytest.mark.parametrize("dtype", ["<i4", ">i4"])
def test_transfer_vlen_results_are_independent(tmp_path: Path, dtype: str) -> None:
    """Keep whole, selected and attribute vlen arrays writable after file close."""
    items = np.empty(3, dtype=hdf5.vlen_dtype(dtype))
    items[0] = np.array([1, 2, 3], dtype=dtype)
    items[1] = np.array([], dtype=dtype)
    items[2] = np.array([4, 5], dtype=dtype)
    scalar = np.empty((), dtype=hdf5.vlen_dtype(dtype))
    scalar[()] = items[0]
    with File(tmp_path / "vlen-independent.h5", "w") as file:
        dataset = file.create_dataset("v", data=items)
        scalar_dataset = file.create_dataset("scalar", data=scalar)
        file.attrs["v"] = items
        whole = np.asarray(dataset[...])
        selected = np.asarray(dataset[1:])
        single = np.asarray(dataset[0])
        scalar_result = np.asarray(scalar_dataset[...])
        attribute = np.asarray(file.attrs["v"])
        whole_first = np.asarray(whole[0])
        selected_empty = np.asarray(selected[0])
        selected_last = np.asarray(selected[1])
        attribute_first = np.asarray(attribute[0])
        for value in (whole_first, np.asarray(whole[1]), selected_empty, selected_last,
                      single, scalar_result, attribute_first):
            assert value.dtype == np.dtype(dtype)
            assert value.flags.writeable and value.flags.owndata
        whole_first[0] = -1
        single[1] = -2
        scalar_result[2] = -3
        attribute_first[0] = -4
        assert np.asarray(dataset[0]).tolist() == [1, 2, 3]
        assert np.asarray(file.attrs["v"])[0].tolist() == [1, 2, 3]
    assert whole_first.tolist() == [-1, 2, 3]
    assert selected_empty.tolist() == [] and selected_last.tolist() == [4, 5]
    assert single.tolist() == [1, -2, 3]
    assert scalar_result.tolist() == [1, 2, -3]
    assert attribute_first.tolist() == [-4, 2, 3]


def test_transfer_special_dtype_errors(tmp_path: Path) -> None:
    """Reject invalid special-type inputs without changing the stored data."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("v", data=[[1], [2, 3]], dtype=hdf5.vlen_dtype("<i4"))
        file.create_dataset("i", data=np.arange(3, dtype="i4"))
        file.create_dataset("o", data=np.array([b"ab"], dtype="V2"))
        with pytest.raises(TypeError):
            _ds(file, "i")[:] = np.array([b"ab", b"cd", b"ef"], dtype="V2")
        with pytest.raises(TypeError):
            _ds(file, "o")[:] = np.array([b"abc"], dtype="V3")
        with pytest.raises(TypeError):
            _ds(file, "v")[0] = "text"
        with pytest.raises(TypeError):
            hdf5.opaque_dtype(np.dtype([("a", "i4")]))
        file.attrs["x"] = np.zeros(2, dtype=[("a", "i4")])
        np.testing.assert_array_equal(file.attrs["x"]["a"], [0, 0])
        with pytest.raises(TypeError):
            file.attrs["x"] = np.zeros(2, dtype="c32")
        assert "x" in file.attrs
        assert _ds(file, "i")[:].tolist() == [0, 1, 2]


def test_transfer_enum_member_validation() -> None:
    """Validate enum member names, values and bases."""
    with pytest.raises(TypeError):
        hdf5.enum_dtype({1: 1}, "u1")  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        hdf5.enum_dtype({"A": True}, "u1")
    with pytest.raises(ValueError):
        hdf5.enum_dtype({"A": -1}, "u1")
    names = {"a b": 1, "c%d": 2, "\u00e9": 3}
    assert hdf5.check_enum_dtype(hdf5.enum_dtype(names, "i2")) == names


def test_transfer_enum_member_names_roundtrip(tmp_path: Path) -> None:
    """Escape unusual member names and restore them on read."""
    names = {"a b": 1, "c%d,e=f": 2, "\u00e9": 3}
    enum = hdf5.enum_dtype(names, "i2")
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("e", data=np.array([1, 2, 3], dtype=enum))
        assert hdf5.check_enum_dtype(_ds(file, "e").dtype) == names


def test_transfer_opaque_datetime_roundtrip_and_selection(tmp_path: Path) -> None:
    """Round-trip tagged datetimes and timedeltas through datasets and slices."""
    dates = np.array(["2020-01-01", "2021-06-01", "NaT"], dtype="M8[D]")
    deltas = np.array([1, 2, 3], dtype="m8[ms]")
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=dates.astype(hdf5.opaque_dtype("M8[D]")))
        file.create_dataset("t", data=deltas.astype(hdf5.opaque_dtype("m8[ms]")))
        file.attrs["d"] = dates.astype(hdf5.opaque_dtype("M8[D]"))
        np.testing.assert_array_equal(_ds(file, "d")[:], dates)
        np.testing.assert_array_equal(_ds(file, "t")[1:], deltas[1:])
        np.testing.assert_array_equal(file.attrs["d"], dates)
        assert _ds(file, "d").dtype == np.dtype("M8[D]")
        assert hdf5.check_opaque_dtype(_ds(file, "t").dtype)


def test_names_unicode_and_character_sets(tmp_path: Path) -> None:
    """Round-trip non-ASCII group, dataset and attribute names, and bytes names."""
    with File(tmp_path / "a.h5", "w") as file:
        group = file.create_group("gr\u00fcn/\u00e9t\u00e9")
        group.create_dataset("\u03b1", data=1)
        group.attrs["\u00e9"] = 2
        file.create_dataset(b"raw", data=3)
        file.attrs[b"rawattr"] = 4
    with File(tmp_path / "a.h5", "r") as file:
        assert file.keys() == ["gr\u00fcn", "raw"]
        group = file["gr\u00fcn/\u00e9t\u00e9"]
        assert isinstance(group, hdf5.Group)
        assert group.keys() == ["\u03b1"]
        assert group.attrs["\u00e9"] == 2
        assert group.attrs.keys() == ["\u00e9"]
        assert file.attrs["rawattr"] == 4
        assert "\u03b1" in group and "missing" not in group
        assert _ds(file, "raw")[()] == 3


def test_names_reject_null_bytes_and_wrong_types(tmp_path: Path) -> None:
    """Reject names that cannot be represented."""
    with File(tmp_path / "a.h5", "w") as file:
        with pytest.raises(ValueError):
            file.create_group("a\0b")
        with pytest.raises(ValueError):
            file.attrs[b"a\0b"] = 1
        with pytest.raises(TypeError):
            file.create_group(1)  # type: ignore[arg-type]


def test_transfer_memory_is_released(tmp_path: Path) -> None:
    """Free native read blocks when arrays are dropped, for numeric and vlen data."""
    resource = pytest.importorskip("resource")
    vlen = hdf5.vlen_dtype("<i4")
    items = np.empty(500, dtype=vlen)
    for index in range(500):
        items[index] = np.arange(200, dtype="<i4")
    big = np.zeros(1_000_000, dtype="f8")
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("big", data=big)
        file.create_dataset("v", data=items)
        dataset, vdataset = _ds(file, "big"), _ds(file, "v")

        def cycle() -> None:
            """Read and write both datasets repeatedly."""
            for _ in range(20):
                dataset[:]
                dataset[:] = big
                vdataset[:]

        cycle()
        gc.collect()
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        cycle()
        gc.collect()
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert after - before < 50_000
