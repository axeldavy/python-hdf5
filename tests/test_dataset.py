"""Test numeric datasets and their native lifetime."""

import gc
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from hdf5 import Dataset, File


@pytest.mark.parametrize(
    "dtype",
    ["i1", "u1", "<i2", ">i2", "<u2", ">u2", "<i4", ">i4", "<u4", ">u4",
     "<i8", ">i8", "<u8", ">u8", "<f4", ">f4", "<f8", ">f8", "?"],
)
def test_dataset_numeric_roundtrip(tmp_path: Path, dtype: str) -> None:
    """Preserve values, byte order, shape, and dtype across file reopen."""
    path = tmp_path / "numeric.h5"
    data = np.array([[0, 1, 0], [1, 1, 0]], dtype=dtype)[:, ::2]
    with File(path, "w") as file:
        dataset = file.create_dataset("array", data=data)
        assert dataset.shape == data.shape
        assert dataset.dtype == data.dtype
        np.testing.assert_array_equal(dataset[:], data)
        np.testing.assert_array_equal(dataset[()], data)
        dataset.attrs["count"] = 4
        dataset[:] = 1
        np.testing.assert_array_equal(dataset.read(), np.ones(data.shape, dtype=dtype))
        dataset.write(data)
    with File(path) as file:
        reopened = file["array"]
        assert isinstance(reopened, Dataset)
        np.testing.assert_array_equal(reopened.read(), data)
        assert reopened.attrs["count"] == 4


def test_dataset_scalar_empty_and_broadcast(tmp_path: Path) -> None:
    """Handle scalar, empty, noncontiguous and broadcast writes."""
    with File(tmp_path / "shapes.h5", "w") as file:
        scalar = file.create_dataset("scalar", data=np.float32(2))
        assert scalar.shape == ()
        assert scalar[()] == np.float32(2)
        scalar[()] = 3
        assert scalar.read() == np.float32(3)
        empty = file.create_dataset("empty", data=np.empty((2, 0, 3), dtype=">i2"))
        assert empty[:].shape == (2, 0, 3)
        empty.write(1)
        array = file.create_dataset("array", data=np.zeros((2, 3), dtype="f4"))
        array[:] = np.arange(6)[::2]
        np.testing.assert_array_equal(array[:], [[0, 2, 4], [0, 2, 4]])
        with pytest.raises(ValueError):
            array[:] = np.arange(4)
        np.testing.assert_array_equal(array[0], [0, 2, 4])
        array[1:] = 0
        np.testing.assert_array_equal(array[:], [[0, 2, 4], [0, 0, 0]])
        with pytest.raises(TypeError):
            _ = getattr(array, "__getitem__")(np.array([0, 1]))


def test_dataset_read_maximum_rank(tmp_path: Path) -> None:
    """Read the maximum HDF5 rank through whole and selected transfers."""
    data = np.ones((1,) * 32, dtype="i2")
    with File(tmp_path / "maximum-rank.h5", "w") as file:
        dataset = file.create_dataset("value", data=data)
        dataset.attrs["value"] = data
        assert dataset.shape == data.shape
        np.testing.assert_array_equal(dataset[:], data)
        np.testing.assert_array_equal(dataset[(0,) + (slice(None),) * 31], data[0])
        np.testing.assert_array_equal(dataset.attrs["value"], data)


def test_dataset_paths_errors_and_readonly(tmp_path: Path) -> None:
    """Reject invalid or unsupported operations without modifying datasets."""
    path = tmp_path / "errors.h5"
    with File(path, "w") as file:
        group = file.create_group("group")
        dataset = group.create_dataset("/group/value", data=[1, 2])
        assert isinstance(file["group/value"], Dataset)
        assert isinstance(group["/group/value"], Dataset)
        with pytest.raises(KeyError):
            file.create_dataset("missing/value", data=[1])
        with pytest.raises(OSError):
            file.create_dataset("group/value", data=[0])
        for name in ("", "/", "a/../b", "group/", "bad\0name"):
            with pytest.raises(ValueError):
                file.create_dataset(name, data=[1])
        for value in (np.array(["text"]), np.array([object()]), np.array(["2020"], dtype="M8[s]")):
            with pytest.raises(TypeError):
                file.create_dataset("bad", data=value)
            with pytest.raises(TypeError):
                dataset.write(value)
        with pytest.raises(TypeError, match="rank"):
            file.create_dataset("rank", data=np.zeros((1,) * 33))
        assert group.keys() == ["value"]
        with pytest.raises(TypeError):
            Dataset()
        with pytest.raises(ValueError, match="not initialized"):
            Dataset.__new__(Dataset).read()
    with File(path) as file:
        dataset = file["group/value"]
        assert isinstance(dataset, Dataset)
        with pytest.raises(OSError):
            dataset.write([3, 4])
        np.testing.assert_array_equal(dataset.read(), [1, 2])


def test_dataset_lookup_retains_resolved_parent(tmp_path: Path) -> None:
    """Keep the correct parent for relative and absolute dataset lookups."""
    with File(tmp_path / "parents.h5", "w") as file:
        group = file.create_group("a/b")
        group.create_dataset("value", data=[4, 5])
        for owner, path in (
            (file, "a/b/value"),
            (group, "value"),
            (group, "./value"),
            (group, "/a/b/value"),
            (group, "/a/./b/value"),
        ):
            dataset = owner[path]
            assert isinstance(dataset, Dataset)
            assert dataset.parent.name == "/a/b"
            np.testing.assert_array_equal(dataset[:], [4, 5])
        with pytest.raises(TypeError):
            file.create_dataset("a/b/value/child", data=1)
        with pytest.raises(ValueError):
            _ = file["a/b/value/"]


def test_dataset_require_and_create_like(tmp_path: Path) -> None:
    """Reuse compatible datasets and copy another dataset's data and dtype."""
    with File(tmp_path / "require-dataset.h5", "w") as file:
        original = file.create_dataset("original", data=np.array([1, 2], dtype="i4"))
        existing = file.require_dataset("original", data=np.array([3, 4], dtype="i4"))
        assert existing.name == original.name
        np.testing.assert_array_equal(existing.read(), [1, 2])

        created = file.require_dataset("created", data=np.array([5, 6], dtype="i2"))
        assert created.dtype == np.dtype("i2")
        np.testing.assert_array_equal(created.read(), [5, 6])

        copied = file.create_dataset_like("copy", original)
        assert copied.dtype == original.dtype
        np.testing.assert_array_equal(copied.read(), original.read())
        overridden = file.create_dataset_like("copy-float", original, dtype="f4")
        assert overridden.dtype == np.dtype("f4")
        np.testing.assert_array_equal(overridden.read(), [1, 2])

        with pytest.raises(TypeError, match="shape"):
            file.require_dataset("original", data=[1, 2, 3])
        with pytest.raises(TypeError, match="dtype"):
            file.require_dataset("original", data=[1, 2], dtype="i2")
        with pytest.raises(TypeError, match="source"):
            file.create_dataset_like("invalid", [1, 2])


def test_dataset_lifetime_and_close(tmp_path: Path) -> None:
    """Keep dataset and attributes alive and invalidate both on file close."""
    path = tmp_path / "lifetime.h5"
    file = File(path, "w")
    dataset = file.create_dataset("values", data=[1, 2])
    attrs = dataset.attrs
    del file, dataset
    gc.collect()
    attrs["value"] = 3
    del attrs
    gc.collect()
    with File(path, "r+") as file:
        dataset = file["values"]
        assert isinstance(dataset, Dataset)
        attrs = dataset.attrs
        assert attrs["value"] == 3
    for action in (dataset.read, lambda: dataset.write([0, 0]), lambda: dataset.shape,
                   lambda: dataset.dtype, attrs.keys):
        with pytest.raises(ValueError, match="closed"):
            action()
    with File(path, "r+") as reopened:
        del dataset, attrs
        gc.collect()
        result = reopened["values"]
        assert isinstance(result, Dataset)
        assert result.read().tolist() == [1, 2]


def test_dataset_concurrent_close(tmp_path: Path) -> None:
    """Serialize dataset reads and writes with file close."""
    file = File(tmp_path / "race.h5", "w")
    dataset = file.create_dataset("values", data=np.arange(128))
    barrier = Barrier(7)

    def worker() -> None:
        """Read repeatedly until the file closes."""
        barrier.wait()
        for _ in range(100):
            try:
                np.testing.assert_array_equal(dataset.read(), np.arange(128))
                assert dataset.shape == (128,)
            except ValueError as error:
                assert "closed" in str(error)
                return

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(worker) for _ in range(6)]
        barrier.wait()
        file.close()
        for future in futures:
            future.result(timeout=10)


def test_dataset_concurrent_read_and_write(tmp_path: Path) -> None:
    """Keep each complete read or write serialized on a shared file."""
    with File(tmp_path / "concurrent.h5", "w") as file:
        dataset = file.create_dataset("values", data=np.zeros(512, dtype="i8"))
        barrier = Barrier(5)

        def writer(value: int) -> None:
            """Write one consistent value repeatedly."""
            barrier.wait()
            for _ in range(50):
                dataset.write(value)

        def reader() -> None:
            """Check that no read mixes complete writes."""
            barrier.wait()
            for _ in range(100):
                values = np.asarray(dataset.read())
                assert np.all(values == values[0])

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(writer, value) for value in (1, 2)]
            futures.extend(executor.submit(reader) for _ in range(2))
            barrier.wait()
            for future in futures:
                future.result(timeout=10)


def test_dataset_h5py_interoperability(tmp_path: Path, h5py_python: str) -> None:
    """Read and write datasets and attributes across h5py implementations."""
    path = tmp_path / "interop.h5"
    create = (
        "import h5py, numpy as np, sys\n"
        "with h5py.File(sys.argv[1], 'w') as f:\n"
        " d = f.create_dataset('existing', data=np.array([1, 2], dtype='>i4'))\n"
        " d.attrs['flag'] = True\n"
        " f.create_dataset('bool', data=np.array([True, False]))\n"
        " f.create_dataset('text', data=np.array([b'a']))\n"
        " f.create_dataset('null', data=h5py.Empty('f'))\n"
        " f['soft'] = h5py.SoftLink('/existing')\n"
    )
    subprocess.run([h5py_python, "-c", create, str(path)], check=True)
    with File(path, "r+") as file:
        existing = file["existing"]
        assert isinstance(existing, Dataset)
        assert existing.dtype == np.dtype(">i4")
        assert existing.attrs["flag"]
        np.testing.assert_array_equal(existing[:], [1, 2])
        existing[:] = [7, 8]
        boolean = file["bool"]
        assert isinstance(boolean, Dataset)
        np.testing.assert_array_equal(boolean[:], [True, False])
        text = file["text"]
        assert isinstance(text, Dataset)
        assert text[:].tolist() == [b"a"]
        unsupported = file["null"]
        assert isinstance(unsupported, Dataset)
        with pytest.raises(TypeError):
            unsupported.read()
        for action in (lambda: unsupported.shape, lambda: unsupported.dtype):
            with pytest.raises(TypeError, match="null"):
                action()
        with pytest.raises(TypeError, match="null"):
            _ = unsupported[0]
        with pytest.raises(TypeError, match="links"):
            _ = file["soft"]
        created = file.create_dataset("created", data=np.array([3, 4], dtype=">u2"))
        created.attrs["flag"] = np.bool_(True)
    check = (
        "import h5py, numpy as np, sys\n"
        "with h5py.File(sys.argv[1]) as f:\n"
        " assert f['existing'][:].tolist() == [7, 8]\n"
        " assert f['created'].dtype == np.dtype('>u2')\n"
        " assert f['created'][:].tolist() == [3, 4]\n"
        " assert f['created'].attrs['flag'] == True\n"
    )
    subprocess.run([h5py_python, "-c", check, str(path)], check=True)


def test_special_dtype_metadata_rejected_not_dropped(tmp_path: Path) -> None:
    """Reject unknown dtype metadata instead of silently dropping it."""
    odd = np.dtype("i2", metadata={"unit": "m"})
    with File(tmp_path / "odd.h5", "w") as file:
        with pytest.raises(TypeError, match="special type"):
            file.create_dataset("x", data=np.array([0, 1], dtype=odd))
        with pytest.raises(TypeError, match="special type"):
            file.attrs["x"] = np.array([0, 1], dtype=odd)
        assert "x" not in file
