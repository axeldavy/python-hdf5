"""Test basic HDF5 dataset selections and hyperslab transfers."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from threading import Barrier
from types import EllipsisType

import numpy as np
import pytest

from hdf5 import Dataset, File


type Selection = int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...]


@pytest.mark.parametrize(
    "selection",
    [
        0, -1, (1, 2, 3), (-1, -2, -3), (slice(1, 4), 2),
        (slice(None), slice(1, 5, 2), 3), (Ellipsis, 2), (1, Ellipsis),
        (slice(9, 15), slice(None), 2), (slice(None, None, 100000000000000000000),),
        (slice(None), slice(None, None, 2), slice(1, 6, 2)), Ellipsis,
    ],
)
def test_dataset_selection_read_and_write(tmp_path: Path, selection: Selection) -> None:
    """Match NumPy for selected reads and broadcast writes."""
    data = np.arange(4 * 5 * 6, dtype=">i4").reshape(4, 5, 6)
    with File(tmp_path / "select.h5", "w") as file:
        dataset = file.create_dataset("array", data=data)
        expected = data[selection]
        result = dataset[selection]
        np.testing.assert_array_equal(result, expected)
        assert np.shape(result) == np.shape(expected)
        assert np.asarray(result).dtype == np.asarray(expected).dtype
        dataset[selection] = 37
        data[selection] = 37
        np.testing.assert_array_equal(dataset.read(), data)


def test_dataset_selection_noncontiguous_broadcast(tmp_path: Path) -> None:
    """Broadcast and convert noncontiguous input into the selected coordinates."""
    data = np.arange(120, dtype=">f4").reshape(4, 5, 6)
    with File(tmp_path / "broadcast.h5", "w") as file:
        dataset = file.create_dataset("array", data=data)
        values = np.arange(12, dtype="i2")[::2]
        dataset[1:4:2, 2, :] = values
        data[1:4:2, 2, :] = values
        np.testing.assert_array_equal(dataset.read(), data)
        with pytest.raises(ValueError):
            dataset[1:4:2, 2, :] = np.arange(4)
        np.testing.assert_array_equal(dataset.read(), data)


@pytest.mark.parametrize("dtype", [">i2", "<f8", "?"])
def test_dataset_selection_combinations(tmp_path: Path, dtype: str) -> None:
    """Compare combinations of integer and slice axes against NumPy."""
    data = np.arange(60).reshape(3, 4, 5).astype(dtype)
    choices = (0, -1, slice(None), slice(1, None, 2), slice(5, 6), slice(None, None, 7))
    with File(tmp_path / "combinations.h5", "w") as file:
        dataset = file.create_dataset("array", data=data)
        for selection in product(choices, repeat=3):
            expected = data[selection]
            actual = dataset[selection]
            np.testing.assert_array_equal(actual, expected)
            assert np.shape(actual) == np.shape(expected)
            dataset[selection] = 1
            data[selection] = 1
            np.testing.assert_array_equal(dataset.read(), data)


def test_dataset_selection_bounds_and_fallback(tmp_path: Path) -> None:
    """Match NumPy for signed bounds, integer protocol and oversized Python integers."""
    data = np.arange(20, dtype=">i4").reshape(4, 5)
    slices = (
        slice(None), slice(-10, -1), slice(-4, 10, 2), slice(3, -9),
        slice(-9223372036854775808, 9223372036854775807, 2),
        slice(-10**30, 10**30, 10**30), slice(2, 2), slice(0, 0),
        slice(np.int64(-3), np.int64(5), np.int64(2)),
    )
    with File(tmp_path / "bounds.h5", "w") as file:
        dataset = file.create_dataset("array", data=data)
        for first, second in product(slices, repeat=2):
            np.testing.assert_array_equal(dataset[first, second], data[first, second])
        for selection in ((-4, -5), (-1, Ellipsis), (Ellipsis, -1),
                          (np.int64(1), slice(None)), (slice(None), np.int64(-1))):
            np.testing.assert_array_equal(getattr(dataset, "__getitem__")(selection), data[selection])
        for selection in ((-5, 0), (0, 5), (10**30,), (-10**30,),
                          (Ellipsis, Ellipsis), (0, 0, 0)):
            with pytest.raises(IndexError):
                _ = dataset[selection]
        for selection in (slice(None, None, 0), slice(None, None, -1)):
            with pytest.raises(ValueError):
                _ = dataset[selection]


def test_dataset_selection_index_protocol_and_closed(tmp_path: Path) -> None:
    """Retain index protocol fallback and check file closure before calling user code."""
    calls: list[int] = []

    class Index:
        """Track index conversion."""

        def __index__(self) -> int:
            """Record conversion and return an index."""
            calls.append(1)
            return -1

    file = File(tmp_path / "index-protocol.h5", "w")
    dataset = file.create_dataset("array", data=np.arange(4))
    assert getattr(dataset, "__getitem__")(Index()) == 3
    assert calls == [1]
    file.close()
    with pytest.raises(ValueError, match="closed"):
        _ = getattr(dataset, "__getitem__")(Index())
    assert calls == [1]
    with pytest.raises(ValueError, match="closed"):
        _ = dataset[0, 0]


def test_dataset_selection_dtype_cache_follows_open_object(tmp_path: Path) -> None:
    """Keep cached dtype bound to the dataset object, not its replaced link."""
    with File(tmp_path / "replacement.h5", "w") as file:
        old = file.create_dataset("data", data=np.arange(4, dtype=">i4"))
        assert old.dtype == np.dtype(">i4")
        del file["data"]
        new = file.create_dataset("data", data=np.arange(4, dtype="<f8"))
        assert old.dtype == np.dtype(">i4") and new.dtype == np.dtype("<f8")
        assert np.asarray(old[1:3]).dtype == np.dtype(">i4")
        assert np.asarray(new[1:3]).dtype == np.dtype("<f8")
        np.testing.assert_array_equal(old[1:3], new[1:3])


def test_dataset_selection_errors_and_empty(tmp_path: Path) -> None:
    """Reject invalid selections and preserve empty shapes and read-only errors."""
    path = tmp_path / "errors.h5"
    with File(path, "w") as file:
        dataset = file.create_dataset("array", data=np.zeros((4, 0, 3), dtype="i4"))
        assert dataset[1:3, :, 1].shape == (2, 0)
        dataset[1:3, :, 1] = 5
        with pytest.raises(IndexError):
            _ = dataset[0, 0]
        with pytest.raises(IndexError):
            _ = dataset[4]
        with pytest.raises(IndexError):
            _ = dataset[0, 0, 0, 0]
        with pytest.raises(IndexError):
            _ = dataset[Ellipsis, Ellipsis]
        for selection in (slice(None, None, -1), slice(None, None, 0)):
            with pytest.raises(ValueError):
                _ = dataset[selection]
        for selection in (True, None, [0, 1], np.array([0, 1])):
            with pytest.raises(TypeError):
                _ = getattr(dataset, "__getitem__")(selection)
    with File(path, "r") as file:
        dataset = file["array"]
        assert isinstance(dataset, Dataset)
        with pytest.raises(OSError):
            dataset[1:3, :, 1] = 5


def test_dataset_selection_scalar_and_closed(tmp_path: Path) -> None:
    """Preserve scalar access and reject operations after close."""
    file = File(tmp_path / "scalar.h5", "w")
    scalar = file.create_dataset("scalar", data=np.int16(5))
    assert scalar[()] == 5
    assert scalar[...] == 5
    scalar[...] = 7
    assert scalar[()] == 7
    with pytest.raises(IndexError):
        _ = scalar[0]
    file.close()
    with pytest.raises(ValueError, match="closed"):
        _ = scalar[...]
    with pytest.raises(ValueError, match="closed"):
        scalar[...] = 1


@pytest.mark.parametrize("shape", [(0, 3), (2, 3)])
def test_dataset_ellipsis_returns_independent_data(tmp_path: Path, shape: tuple[int, ...]) -> None:
    """Return a whole independent array for bare ellipsis, including empty shapes."""
    data = np.arange(np.prod(shape), dtype=">i4").reshape(shape)
    with File(tmp_path / "ellipsis.h5", "w") as file:
        dataset = file.create_dataset("array", data=data)
        result = np.asarray(dataset[...])
        assert result.shape == shape and result.dtype == data.dtype
        np.testing.assert_array_equal(result, data)
        if result.size:
            result[0, 0] = -1
            np.testing.assert_array_equal(dataset[...], data)
    if result.size:
        assert result[0, 0] == -1


def test_dataset_selection_close_race(tmp_path: Path) -> None:
    """Serialize selection transfers against file close."""
    file = File(tmp_path / "race.h5", "w")
    dataset = file.create_dataset("array", data=np.arange(128).reshape(16, 8))
    barrier = Barrier(5)

    def worker() -> None:
        """Read a hyperslab until close wins."""
        barrier.wait()
        for _ in range(100):
            try:
                expected = np.arange(128).reshape(16, 8)[1:12:2, 3]
                np.testing.assert_array_equal(dataset[1:12:2, 3], expected)
            except ValueError as error:
                assert "closed" in str(error)
                return

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker) for _ in range(4)]
        barrier.wait()
        file.close()
        for future in futures:
            future.result(timeout=10)


def test_dataset_selection_concurrent_reads_and_writes(tmp_path: Path) -> None:
    """Keep cached dtype and selected transfers consistent across shared threads."""
    data = np.arange(128, dtype=">i4").reshape(16, 8)
    with File(tmp_path / "concurrent-selection.h5", "w") as file:
        dataset = file.create_dataset("array", data=data)
        barrier = Barrier(4)

        def reader() -> None:
            """Read repeatedly while another thread modifies a separate column."""
            barrier.wait()
            for _ in range(200):
                np.testing.assert_array_equal(dataset[1:12:2, 3], data[1:12:2, 3])
                assert dataset.dtype == data.dtype

        def writer() -> None:
            """Write a separate column while selected reads are running."""
            barrier.wait()
            for index in range(200):
                dataset[1:12:2, 4] = index

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(reader) for _ in range(3)]
            futures.append(executor.submit(writer))
            for future in futures:
                future.result(timeout=30)


def test_dataset_selection_h5py_large_sparse(tmp_path: Path, h5py_python: str) -> None:
    """Read and write a tiny hyperslab in a large h5py-created sparse dataset."""
    path = tmp_path / "large.h5"
    create = (
        "import h5py, sys\n"
        "with h5py.File(sys.argv[1], 'w') as f:\n"
        " d = f.create_dataset('data', (1000000000,), dtype='>i4', chunks=(1024,))\n"
        " d[999999990:999999996:2] = [11, 12, 13]\n"
    )
    subprocess.run([h5py_python, "-c", create, str(path)], check=True)
    with File(path, "r+") as file:
        dataset = file["data"]
        assert isinstance(dataset, Dataset)
        np.testing.assert_array_equal(dataset[999999990:999999996:2], [11, 12, 13])
        dataset[999999991:999999996:2] = [21, 22, 23]
    check = (
        "import h5py, sys\n"
        "with h5py.File(sys.argv[1]) as f:\n"
        " assert f['data'][999999990:999999996].tolist() == [11, 21, 12, 22, 13, 23]\n"
    )
    subprocess.run([h5py_python, "-c", check, str(path)], check=True)
