"""Test the installed extension and its bundled native dependencies."""

import subprocess
import sys
from pathlib import Path

import pytest

import hdf5


def test_packaging_windows_hdf5_dll() -> None:
    """Ship the uniquely named HDF5 DLL next to the extension on Windows."""
    if sys.platform != "win32":
        pytest.skip("Windows uses a shared HDF5 library")

    package_dir = Path(hdf5.__file__).parent
    assert (package_dir / "python_hdf5_hdf5_2_2_0.dll").is_file()


@pytest.mark.parametrize("first", ["hdf5", "h5py"])
def test_packaging_hdf5_import_order(tmp_path: Path, h5py_python: str, first: str) -> None:
    """Open files with both libraries regardless of their import order."""
    if h5py_python != sys.executable:
        pytest.skip("both libraries must be installed in the current interpreter")

    code = (
        f"import {first}\n"
        "import hdf5, h5py, sys\n"
        "with hdf5.File(sys.argv[1], 'w') as f:\n"
        " f.attrs['value'] = 1\n"
        " f.attrs['value'] = 2\n"
        " assert f.attrs['value'] == 2\n"
        "with h5py.File(sys.argv[1], 'r') as f:\n"
        " assert f.attrs['value'] == 2\n"
    )
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "import-order.h5")],
        check=True,
    )
