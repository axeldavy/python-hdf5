"""Test HDF5 file lifetime and open modes."""

import gc
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from hdf5 import Dataset, File


def test_file_context_flush_and_close(tmp_path: Path) -> None:
    """Close the file at context exit and reject use after close."""
    path = tmp_path / "example.h5"
    with File(path, "w") as file:
        assert not file.closed
        file.flush()
    assert file.closed
    file.close()
    with pytest.raises(ValueError, match="closed"):
        file.flush()
    with pytest.raises(ValueError, match="closed"):
        file.__enter__()
    with File(path) as reopened:
        assert not reopened.closed


@pytest.mark.parametrize("mode", ["w-", "x"])
def test_file_exclusive_create(tmp_path: Path, mode: str) -> None:
    """Do not overwrite an existing file in exclusive modes."""
    path = tmp_path / "example.h5"
    with File(path, mode):
        pass
    with pytest.raises(OSError):
        File(path, mode)


def test_file_append_and_readwrite(tmp_path: Path) -> None:
    """Open or create with append, then reopen for read-write access."""
    path = tmp_path / "example.h5"
    with File(path, "a"):
        pass
    with File(path, "a"):
        pass
    with File(path, "r+"):
        pass


@pytest.mark.parametrize("backing_store", [False, True])
def test_file_core_driver_persistence(tmp_path: Path, *, backing_store: bool) -> None:
    """Keep core files in memory or persist them according to backing_store."""
    path = tmp_path / "core.h5"
    with File(path, "w", driver="core", backing_store=backing_store, block_size=8192) as file:
        assert file._core_settings() == (8192, backing_store)  # type: ignore[attr-defined]
        dataset = file.create_dataset("value", data=[1, 2, 3])
        file.attrs["answer"] = 42
        file.flush()
        assert path.exists() is backing_store
        assert dataset[:].tolist() == [1, 2, 3]
    with pytest.raises(ValueError, match="closed"):
        file._core_settings()  # type: ignore[attr-defined]
    assert path.exists() is backing_store
    if backing_store:
        with File(path) as file:
            dataset = file["value"]
            assert isinstance(dataset, Dataset)
            assert dataset[:].tolist() == [1, 2, 3]
            assert file.attrs["answer"] == 42


def test_file_core_existing_file_without_backing(tmp_path: Path) -> None:
    """Load an existing file into memory without writing changes to disk."""
    path = tmp_path / "existing.h5"
    with File(path, "w") as file:
        file.create_dataset("data", data=[1, 2])
    with File(path, "r+", driver="core", backing_store=False) as file:
        dataset = file["data"]
        assert isinstance(dataset, Dataset)
        dataset[0] = 9
        file.flush()
        assert dataset[:].tolist() == [9, 2]
    with File(path) as file:
        dataset = file["data"]
        assert isinstance(dataset, Dataset)
        assert dataset[:].tolist() == [1, 2]
    with File(tmp_path / "default-core.h5", "w", driver="core") as file:
        assert file._core_settings() == (65536, True)  # type: ignore[attr-defined]


def test_file_core_modes_and_close_invalidation(tmp_path: Path) -> None:
    """Keep normal open modes and descendant close semantics with the core driver."""
    path = tmp_path / "modes.h5"
    with File(path, "a", driver="core") as file:
        dataset = file.create_dataset("value", data=[4, 5])
        attrs = dataset.attrs
        attrs["value"] = 4
    with pytest.raises(ValueError, match="closed"):
        dataset.read()
    with pytest.raises(ValueError, match="closed"):
        _ = attrs["value"]
    with File(path, "a", driver="core") as file:
        assert file["value"].name == "/value"
    with File(path, "r", driver="core", backing_store=False) as file:
        dataset = file["value"]
        assert isinstance(dataset, Dataset)
        with pytest.raises(OSError):
            dataset[0] = 8
    with pytest.raises(OSError):
        File(path, "x", driver="core")
    with pytest.raises(OSError):
        File(tmp_path / "missing.h5", "r+", driver="core")


def test_file_chunk_cache_configuration(tmp_path: Path) -> None:
    """Apply cache settings and preserve HDF5 defaults for omitted values."""
    with File(tmp_path / "default.h5", "w") as file:
        defaults = file._cache_settings()  # type: ignore[attr-defined]
    with File(tmp_path / "cache.h5", "w", rdcc_nslots=127, rdcc_nbytes=123456,
              rdcc_w0=0.25) as file:
        assert file._cache_settings() == (127, 123456, 0.25)  # type: ignore[attr-defined]
    with File(tmp_path / "partial.h5", "w", rdcc_nbytes=765432) as file:
        assert file._cache_settings() == (defaults[0], 765432, defaults[2])  # type: ignore[attr-defined]
    with File(tmp_path / "zero.h5", "w", rdcc_nslots=0, rdcc_nbytes=0, rdcc_w0=0) as file:
        assert file._cache_settings() == (0, 0, 0)  # type: ignore[attr-defined]
    with File(tmp_path / "one.h5", "w", rdcc_w0=1) as file:
        assert file._cache_settings() == (defaults[0], defaults[1], 1)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="closed"):
        file._cache_settings()  # type: ignore[attr-defined]


@pytest.mark.parametrize("libver", ["earliest", "v108", "v110", "v112", "v114",
                                    "v200", "latest", ("earliest", "v108"),
                                    ("v110", "v114")])
def test_file_libver_interoperability(
    tmp_path: Path, h5py_python: str, libver: str | tuple[str, str]
) -> None:
    """Create readable files for supported minimum and maximum format bounds."""
    path = tmp_path / "version.h5"
    with File(path, "w", libver=libver) as file:
        file.create_dataset("data", data=[1, 2])
    with File(path) as file:
        dataset = file["data"]
        assert isinstance(dataset, Dataset)
        assert dataset[:].tolist() == [1, 2]
    if libver == "latest":
        assert path.read_bytes()[8] == 3
    if libver == ("earliest", "v108"):
        assert path.read_bytes()[8] == 0
    if libver in ("v200", "latest"):
        version = subprocess.run(
            [h5py_python, "-c", "import h5py; print(h5py.version.hdf5_version_tuple >= (2, 0))"],
            check=True, capture_output=True, text=True,
        )
        if version.stdout.strip() != "True":
            pytest.skip("h5py's HDF5 version may not support the 2.0 format")
    subprocess.run(
        [h5py_python, "-c",
         "import h5py,sys; f=h5py.File(sys.argv[1]); assert f['data'][:].tolist()==[1,2]; f.close()",
         str(path)],
        check=True,
    )


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX file locking")
def test_file_locking_options_cross_process(tmp_path: Path) -> None:
    """Check that explicit locking controls are applied across processes."""
    script = """
import subprocess, sys
from hdf5 import File
path = sys.argv[1]
child = "import sys; from hdf5 import File; f=File(sys.argv[1], 'r', locking=sys.argv[2]); f.close()"
with File(path, 'w', locking=True) as file:
    file.flush()
    locked = subprocess.run([sys.executable, '-c', child, path, 'true'], capture_output=True)
    unlocked = subprocess.run([sys.executable, '-c', child, path, 'false'], capture_output=True)
    assert locked.returncode != 0 and b'lock' in locked.stderr.lower(), locked.stderr
    assert unlocked.returncode == 0, unlocked.stderr
"""
    environment = os.environ.copy()
    environment.pop("HDF5_USE_FILE_LOCKING", None)
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "locked.h5")],
                   env=environment, check=True, timeout=30)


@pytest.mark.skipif(sys.platform != "linux", reason="requires POSIX file locking")
def test_file_locking_environment_overrides_option(tmp_path: Path) -> None:
    """Respect HDF5_USE_FILE_LOCKING over the file-access property setting."""
    script = """
import subprocess, sys
from hdf5 import File
path = sys.argv[1]
child = "import sys; from hdf5 import File; f=File(sys.argv[1], 'r', locking=True); f.close()"
with File(path, 'w', locking=True) as file:
    file.flush()
    result = subprocess.run([sys.executable, '-c', child, path], capture_output=True)
    assert result.returncode == 0, result.stderr
"""
    environment = os.environ.copy()
    environment["HDF5_USE_FILE_LOCKING"] = "FALSE"
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "unlocked.h5")],
                   env=environment, check=True, timeout=30)


@pytest.mark.parametrize("locking", [None, True, False, "true", "false", "best-effort"])
def test_file_locking_values(tmp_path: Path, locking: bool | str | None) -> None:
    """Accept each documented file-locking policy."""
    with File(tmp_path / "locking.h5", "w", locking=locking):  # type: ignore[arg-type]
        pass


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"driver": "sec2"}, ValueError),
        ({"backing_store": False}, ValueError),
        ({"block_size": 1024}, ValueError),
        ({"driver": "core", "backing_store": 1}, TypeError),
        ({"driver": "core", "block_size": 0}, ValueError),
        ({"driver": "core", "block_size": -1}, ValueError),
        ({"driver": "core", "block_size": 1.5}, TypeError),
        ({"libver": "v999"}, ValueError),
        ({"libver": ("latest", "v108")}, ValueError),
        ({"libver": ("earliest", "earliest")}, ValueError),
        ({"libver": ("earliest",)}, TypeError),
        ({"libver": ("earliest", 1)}, TypeError),
        ({"locking": "maybe"}, ValueError),
        ({"locking": 0}, ValueError),
        ({"rdcc_nslots": -1}, ValueError),
        ({"rdcc_nbytes": True}, TypeError),
        ({"rdcc_nbytes": 1.5}, TypeError),
        ({"rdcc_nbytes": 2**100}, OverflowError),
        ({"rdcc_w0": -0.1}, ValueError),
        ({"rdcc_w0": 1.1}, ValueError),
        ({"rdcc_w0": float("nan")}, ValueError),
    ],
)
def test_file_options_invalid_before_create(
    tmp_path: Path, options: dict[str, object], error: type[Exception]
) -> None:
    """Reject invalid configuration without creating or truncating a file."""
    path = tmp_path / "untouched.h5"
    path.write_bytes(b"existing")
    with pytest.raises(error):
        File(path, "w", **options)  # type: ignore[arg-type]
    assert path.read_bytes() == b"existing"


def test_file_invalid_inputs(tmp_path: Path) -> None:
    """Reject invalid modes, paths and files with explicit errors."""
    with pytest.raises(ValueError, match="mode"):
        File(tmp_path / "test.h5", "invalid")
    with pytest.raises(ValueError, match="null"):
        File("bad\0path", "w")
    with pytest.raises(OSError):
        File(tmp_path / "missing.h5")
    path = tmp_path / "invalid.h5"
    path.write_text("not hdf5")
    with pytest.raises(OSError):
        File(path, "a")
    assert path.read_text() == "not hdf5"


def test_file_exception_closes_and_reinitialization_is_rejected(tmp_path: Path) -> None:
    """Close on exceptional exit and prevent replacement of a live handle."""
    path = tmp_path / "example.h5"
    file = File(path, "w")
    with pytest.raises(RuntimeError, match="body"):
        with file:
            with pytest.raises(RuntimeError, match="initialized"):
                file.__init__(path, "w")
            raise RuntimeError("body")
    assert file.closed


def test_file_cleanup_releases_file(tmp_path: Path) -> None:
    """Release the file after the last Python reference is collected."""
    path = tmp_path / "cleanup.h5"
    file = File(path, "w")
    del file
    gc.collect()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from hdf5 import File; File(sys.argv[1]).close()",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_file_hdf5_tool_interoperability(tmp_path: Path) -> None:
    """Create a file readable by an independent HDF5 installation."""
    h5dump = shutil.which("h5dump")
    if h5dump is None:
        pytest.skip("h5dump is not installed")
    path = tmp_path / "interop.h5"
    with File(path, "w"):
        pass
    result = subprocess.run([h5dump, "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "/" in result.stdout


def test_file_concurrent_flush_and_close(tmp_path: Path) -> None:
    """Serialize simultaneous flush and close without hangs or crashes."""
    file = File(tmp_path / "concurrent.h5", "w")
    barrier = Barrier(9)

    def worker() -> None:
        """Flush until close wins the race."""
        barrier.wait()
        for _ in range(100):
            try:
                file.flush()
            except ValueError as error:
                assert "closed" in str(error)
                return

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker) for _ in range(8)]
        barrier.wait()
        file.close()
        for future in futures:
            future.result(timeout=10)
    assert file.closed
