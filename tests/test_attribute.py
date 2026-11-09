"""Test numeric HDF5 attributes."""

import gc
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

from hdf5 import AttributeManager, File


@pytest.mark.parametrize(
    "dtype",
    ["i1", "u1", "<i2", ">i2", "<u2", ">u2", "<i4", ">i4", "<u4", ">u4",
     "<i8", ">i8", "<u8", ">u8", "<f4", ">f4", "<f8", ">f8", "?"],
)
def test_attribute_numeric_roundtrip(tmp_path: Path, dtype: str) -> None:
    """Round-trip supported scalar and multidimensional numeric dtypes."""
    source = np.array([[0, 1, 1], [1, 0, 1]], dtype=dtype)[:, ::2]
    with File(tmp_path / "numeric.h5", "w") as file:
        group = file.create_group("group")
        for owner in (file, group):
            owner.attrs["array"] = source
            owner.attrs["scalar"] = source[0, 0]
            result = owner.attrs["array"]
            np.testing.assert_array_equal(result, source)
            assert result.dtype == source.dtype
            assert owner.attrs["scalar"] == source[0, 0]
            assert owner.attrs["scalar"].dtype == source[0, 0].dtype
        source[0, 0] = 1
        assert np.asarray(group.attrs["array"])[0, 0] == 0


def test_attribute_mapping_and_empty_arrays(tmp_path: Path) -> None:
    """Support mapping operations, overwrites, snapshots, and empty arrays."""
    with File(tmp_path / "mapping.h5", "w") as file:
        attrs = file.attrs
        attrs["z"] = np.empty((0, 3), dtype=">i4")
        snapshot = attrs.keys()
        attrs["caf\u00e9"] = 7
        attrs[b"a"] = np.array([3, 5], dtype="i2")
        assert snapshot == ["z"]
        assert list(attrs) == ["a", "caf\u00e9", "z"]
        assert len(attrs) == 3
        assert "caf\u00e9" in attrs
        assert "missing" not in attrs
        assert attrs["z"].shape == (0, 3)
        assert attrs["z"].dtype == np.dtype(">i4")
        attrs["a"] = np.array([[2.5]], dtype="f4")
        np.testing.assert_array_equal(attrs["a"], [[2.5]])
        del attrs["a"]
        with pytest.raises(KeyError):
            _ = attrs["a"]
        with pytest.raises(KeyError):
            del attrs["a"]


def test_attribute_unsupported_and_readonly(tmp_path: Path) -> None:
    """Reject unsupported values without replacing an existing attribute."""
    path = tmp_path / "unsupported.h5"
    with File(path, "w") as file:
        file.attrs["value"] = 23
        for value in (np.array(["text"]), np.array([object()]), np.zeros(1, dtype=[("a", "i4")]), np.array(["2020"], dtype="M8[s]")):
            with pytest.raises(TypeError, match="unsupported"):
                file.attrs["value"] = value
            assert file.attrs["value"] == 23
        with pytest.raises(ValueError):
            file.attrs[""] = 1
        with pytest.raises(ValueError):
            file.attrs["bad\0name"] = 1
        with pytest.raises(TypeError, match="rank"):
            file.attrs["value"] = np.zeros((1,) * 33)
        assert file.attrs["value"] == 23
        with pytest.raises(TypeError):
            AttributeManager()
        with pytest.raises(ValueError, match="not initialized"):
            AttributeManager.__new__(AttributeManager).keys()
    with File(path) as file:
        with pytest.raises(OSError):
            file.attrs["value"] = 3
        assert file.attrs["value"] == 23
        with pytest.raises(OSError):
            del file.attrs["value"]


def test_attribute_lifetime_and_close(tmp_path: Path) -> None:
    """Keep the group alive and invalidate the manager after explicit close."""
    path = tmp_path / "lifetime.h5"
    file = File(path, "w")
    manager = file.create_group("group").attrs
    del file
    gc.collect()
    manager["value"] = [1, 2]
    np.testing.assert_array_equal(manager["value"], [1, 2])
    del manager
    gc.collect()
    with File(path, "r+") as file:
        manager = file["group"].attrs
        np.testing.assert_array_equal(manager["value"], [1, 2])
    for operation in (
        lambda: manager.keys(),
        lambda: manager["value"],
        lambda: manager.__setitem__("value", 1),
        lambda: manager.__delitem__("value"),
        lambda: "value" in manager,
    ):
        with pytest.raises(ValueError, match="closed"):
            operation()


def test_attribute_concurrent_close(tmp_path: Path) -> None:
    """Serialize attribute operations against close."""
    file = File(tmp_path / "race.h5", "w")
    attrs = file.attrs
    attrs["value"] = np.arange(64)
    barrier = Barrier(7)

    def worker() -> None:
        """Read and enumerate until the file closes."""
        barrier.wait()
        for _ in range(100):
            try:
                np.testing.assert_array_equal(attrs["value"], np.arange(64))
                assert attrs.keys() == ["value"]
            except ValueError as error:
                assert "closed" in str(error)
                return

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(worker) for _ in range(6)]
        barrier.wait()
        file.close()
        for future in futures:
            future.result(timeout=10)


def test_attribute_h5py_interoperability(tmp_path: Path, h5py_python: str) -> None:
    """Read h5py attributes and write attributes understood by h5py."""
    path = tmp_path / "interop.h5"
    create = (
        "import h5py, numpy as np, sys\n"
        "with h5py.File(sys.argv[1], 'w') as f:\n"
        " f.attrs['be'] = np.array([1, 7], dtype='>i4')\n"
        " f.attrs['truth'] = np.array([True, False])\n"
        " f.attrs['text'] = 'unsupported'\n"
        " f.attrs.create('null', h5py.Empty('f'))\n"
        " f.attrs.create('enum', np.array([1], dtype=h5py.enum_dtype({'ONE': 1}, 'i1')))\n"
        " f.create_group('group').attrs['scalar'] = np.float32(1.5)\n"
    )
    subprocess.run([h5py_python, "-c", create, str(path)], check=True)
    with File(path, "r+") as file:
        np.testing.assert_array_equal(file.attrs["be"], np.array([1, 7], dtype=">i4"))
        assert file.attrs["be"].dtype == np.dtype(">i4")
        np.testing.assert_array_equal(file.attrs["truth"], [True, False])
        assert file.attrs["truth"].dtype == np.dtype("bool")
        assert file["group"].attrs["scalar"] == np.float32(1.5)
        assert file.attrs["text"] == "unsupported"
        with pytest.raises(TypeError, match="null"):
            _ = file.attrs["null"]
        enum = file.attrs["enum"]
        assert enum.tolist() == [1]
        file.attrs["new"] = np.array([2, 8], dtype=">u2")
        file["group"].attrs["bool"] = np.array([True, False])
    check = (
        "import h5py, numpy as np, sys\n"
        "with h5py.File(sys.argv[1]) as f:\n"
        " assert f.attrs['new'].dtype == np.dtype('>u2')\n"
        " assert f.attrs['new'].tolist() == [2, 8]\n"
        " assert f['group'].attrs['bool'].dtype == np.dtype('bool')\n"
        " assert f['group'].attrs['bool'].tolist() == [True, False]\n"
        " assert f.attrs['text'] == 'unsupported'\n"
    )
    subprocess.run([h5py_python, "-c", check, str(path)], check=True)
