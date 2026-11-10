"""Test dataset creation properties and resizable storage."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from hdf5 import Dataset, File


def test_dataset_chunked_resize_fill_and_gzip(tmp_path: Path, h5py_python: str) -> None:
    """Create and resize a filtered dataset readable by h5py."""
    path = tmp_path / "chunked.h5"
    with File(path, "w") as file:
        dataset = file.create_dataset(
            "data", shape=(0, 3), dtype=">i4", chunks=(2, 3), maxshape=(None, 3),
            compression="gzip", compression_opts=6, shuffle=True, fletcher32=True, fillvalue=7,
        )
        assert dataset.shape == (0, 3)
        dataset.resize((4, 3))
        np.testing.assert_array_equal(dataset[:], np.full((4, 3), 7))
        dataset[1] = [1, 2, 3]
        dataset.resize(3, axis=0)
        np.testing.assert_array_equal(dataset[1], [1, 2, 3])
        dataset.resize((5, 3))
        np.testing.assert_array_equal(dataset[3:], np.full((2, 3), 7))
    subprocess.run(
        [h5py_python, "-c",
         "import h5py,sys\n"
         "with h5py.File(sys.argv[1]) as f:\n"
         " d=f['data']; assert d.shape==(5,3) and d.maxshape==(None,3)\n"
         " assert d.chunks==(2,3) and d.compression=='gzip' and d.compression_opts==6\n"
         " assert d.shuffle and d.fletcher32 and d.fillvalue==7\n"
         " assert d[1].tolist()==[1,2,3] and d[4].tolist()==[7,7,7]\n"
         " assert d.id.read_direct_chunk((0,0))[0] == 0\n", str(path)],
        check=True,
    )


def test_dataset_storage_errors(tmp_path: Path) -> None:
    """Reject invalid chunk/filter combinations and invalid resize requests."""
    with File(tmp_path / "errors.h5", "w") as file:
        for options in (
            {"shape": (2,), "chunks": False, "compression": "gzip"},
            {"shape": (2,), "compression": "unknown"},
            {"shape": (2,), "compression_opts": 5},
            {"shape": (2,), "compression": "gzip", "compression_opts": 10},
            {"shape": (2, 2), "chunks": (1,)},
            {"shape": (2,), "maxshape": (1,)},
            {"shape": (), "chunks": True},
        ):
            with pytest.raises((ValueError, TypeError)):
                file.create_dataset("invalid", **options)  # type: ignore[arg-type]
            assert "invalid" not in file
        dataset = file.create_dataset("fixed", data=[1, 2])
        with pytest.raises(OSError):
            dataset.resize((3,))
        with pytest.raises(ValueError):
            dataset.resize(3, axis=2)


def test_dataset_read_h5py_gzip(tmp_path: Path, h5py_python: str) -> None:
    """Read gzip-compressed chunks created by h5py."""
    path = tmp_path / "gzip.h5"
    subprocess.run(
        [h5py_python, "-c",
         "import h5py,numpy as np,sys\n"
         "with h5py.File(sys.argv[1], 'w') as f:\n"
         " f.create_dataset('data', data=np.arange(100), chunks=(10,), "
         "compression='gzip', shuffle=True, fletcher32=True)\n", str(path)],
        check=True,
    )
    with File(path) as file:
        dataset = file["data"]
        assert isinstance(dataset, Dataset)
        np.testing.assert_array_equal(dataset[13:37], np.arange(13, 37))


def test_dataset_create_shape_and_require(tmp_path: Path) -> None:
    """Create by shape without data and check h5py-style dtype compatibility."""
    with File(tmp_path / "shape.h5", "w") as file:
        dataset = file.create_dataset("value", (2, 3), "i4", [1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(dataset[:], [[1, 2, 3], [4, 5, 6]])
        empty = file.require_dataset("empty", (2, 3), "i4", chunks=(1, 3))
        np.testing.assert_array_equal(empty[:], np.zeros((2, 3), dtype="i4"))
        assert file.require_dataset("empty", (2, 3), "i2").dtype == np.dtype("i4")
        with pytest.raises(TypeError, match="dtype"):
            file.require_dataset("empty", (2, 3), "i2", exact=True)
        with pytest.raises(ValueError, match="shape"):
            file.create_dataset("bad", (3, 3), "i4", [1, 2])
