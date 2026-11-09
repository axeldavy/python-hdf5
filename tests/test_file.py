"""Test HDF5 file lifetime and open modes."""

import gc
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from hdf5 import File


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
