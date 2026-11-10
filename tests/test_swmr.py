"""Test single-writer multiple-reader file and dataset lifecycle."""

import subprocess
import sys
from pathlib import Path

import pytest

from hdf5 import File


def test_swmr_reader_observes_flushed_update(tmp_path: Path) -> None:
    """Refresh a dataset in another process after the writer flushes an update."""
    path = tmp_path / "swmr.h5"
    code = """
import sys
from hdf5 import File
with File(sys.argv[1], 'r', swmr=True) as f:
    assert f.swmr_mode
    dataset = f['data']
    assert dataset[0] == 1 and dataset.shape == (2,)
    print('ready', flush=True)
    assert sys.stdin.readline().strip() == 'refresh'
    dataset.refresh()
    assert dataset[0] == 7 and dataset.shape == (3,) and dataset[2] == 9
    print('updated', flush=True)
"""
    with File(path, "w", libver="latest") as file:
        dataset = file.create_dataset("data", data=[1, 2], maxshape=(None,))
        assert not file.swmr_mode
        file.swmr_mode = True
        assert file.swmr_mode
        with subprocess.Popen(
            [sys.executable, "-u", "-c", code, str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) as process:
            assert process.stdout is not None and process.stdin is not None
            assert process.stdout.readline().strip() == "ready"
            dataset[0] = 7
            dataset.resize(3, axis=0)
            dataset[2] = 9
            dataset.flush()
            process.stdin.write("refresh\n")
            process.stdin.flush()
            assert process.stdout.readline().strip() == "updated"
            _, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr


def test_swmr_errors_and_close(tmp_path: Path) -> None:
    """Reject invalid SWMR transitions and operations on closed objects."""
    path = tmp_path / "file.h5"
    with pytest.raises(ValueError, match="read-only"):
        File(path, "w", swmr=True)
    with File(path, "w") as file:
        dataset = file.create_dataset("data", data=[1])
        with pytest.raises(OSError):
            file.swmr_mode = True
        with pytest.raises(ValueError):
            file.swmr_mode = False
    for action in (dataset.flush, dataset.refresh, lambda: file.swmr_mode):
        with pytest.raises(ValueError, match="closed"):
            action()
