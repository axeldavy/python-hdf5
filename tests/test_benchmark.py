"""Test the benchmark command and its result schema."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


_SCRIPT: Path = Path(__file__).resolve().parents[1] / "benchmarks" / "compare.py"


def test_benchmark_all_scenarios(tmp_path: Path) -> None:
    """Run every scenario and verify sample counts and numeric result shapes."""
    pytest.importorskip("h5py")
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable, str(_SCRIPT), "--iterations", "2", "--repeats", "2",
            "--warmup", "1", "--size-mib", "1", "--threads", "2",
            "--directory", str(tmp_path), "--json", str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["results"]) == 17
    assert "dataset_lookup" in report["results"]
    assert "h5py/new" in result.stdout
    assert report["environment"]["h5py_hdf5"]
    assert report["config"]["threads"] == 2
    for measurement in report["results"].values():
        assert measurement["operations_per_sample"] > 0
        assert measurement["bytes_per_operation"] >= 0
        for backend in ("h5py", "python-hdf5"):
            assert len(measurement["samples_ns_per_operation"][backend]) == 2
            assert all(value > 0 for value in measurement["samples_ns_per_operation"][backend])
            assert measurement["median_ns_per_operation"][backend] > 0
    assert list(tmp_path.iterdir()) == [output]


def test_benchmark_select_and_invalid_input(tmp_path: Path) -> None:
    """Restrict the run to selected scenarios and reject invalid parameters."""
    pytest.importorskip("h5py")
    output = tmp_path / "selected.json"
    subprocess.run(
        [
            sys.executable, str(_SCRIPT), "--iterations", "1", "--repeats", "1",
            "--warmup", "0", "--size-mib", "1", "--threads", "2",
            "--scenario", "dataset_read_slice", "--json", str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert list(report["results"]) == ["dataset_read_slice"]
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--iterations", "0"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode != 0
    assert "positive integer" in result.stderr
