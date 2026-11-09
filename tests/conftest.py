"""Provide shared test fixtures."""

import os
import subprocess
import sys

import pytest


@pytest.fixture
def h5py_python() -> str:
    """Return a Python interpreter with h5py, or skip interoperability tests."""
    executable = os.environ.get("HDF5_TEST_H5PY_PYTHON", sys.executable)
    result = subprocess.run(
        [executable, "-c", "import h5py"], capture_output=True, text=True, check=False
    )
    if result.returncode:
        if "HDF5_TEST_H5PY_PYTHON" in os.environ:
            pytest.fail(f"Cannot import h5py with {executable}: {result.stderr}")
        pytest.skip("h5py is not installed")
    return executable
