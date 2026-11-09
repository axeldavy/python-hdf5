"""Compare equivalent HDF5 operations in h5py and python-hdf5.

Run this script with both packages installed in the same interpreter. Timings
are warm-cache wall-clock measurements, not disk durability measurements.
"""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from threading import Barrier
from typing import Final

import h5py
import numpy as np

import hdf5


_MIB: Final[int] = 1024 * 1024
_SCENARIOS: Final[tuple[str, ...]] = (
    "file_open_close",
    "group_lookup",
    "group_list",
    "group_create",
    "attribute_read",
    "attribute_write",
    "dataset_create_small",
    "dataset_read_small",
    "dataset_write_small",
    "dataset_read_large",
    "dataset_write_large",
    "dataset_read_slice",
    "dataset_write_slice",
    "threaded_read_shared",
    "threaded_read_independent",
    "threaded_read_separate_files",
)


@dataclass(frozen=True, kw_only=True)
class _Config:
    """Store benchmark controls."""

    iterations: int
    repeats: int
    warmup: int
    size_mib: int
    threads: int
    scenarios: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class _Workload:
    """Describe a timed batch and its untimed correctness check."""

    run: Callable[[], None]
    verify: Callable[[], None]
    operations: int
    bytes_per_operation: int


def _positive_int(value: str) -> int:
    """Parse a positive command-line integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    """Parse a nonnegative command-line integer."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _parse_args() -> argparse.Namespace:
    """Parse benchmark settings and scenario selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=_positive_int, default=100)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--warmup", type=_nonnegative_int, default=1)
    parser.add_argument("--size-mib", type=_positive_int, default=8)
    parser.add_argument("--threads", type=_positive_int, default=4)
    parser.add_argument("--scenario", action="append", choices=_SCENARIOS)
    parser.add_argument("--directory", type=Path, help="Parent directory for temporary test files")
    parser.add_argument("--json", type=Path, help="Write machine-readable results to this path")
    return parser.parse_args()


def _create_fixture(path: Path, large: np.ndarray) -> None:
    """Create the common fixture before copying it for each implementation."""
    with h5py.File(path, "w") as file:
        group = file.create_group("tree/branch")
        for index in range(32):
            group.create_group(f"child_{index:02d}")
        file.attrs["number"] = np.int64(42)
        file.create_dataset("small", data=np.arange(256, dtype=np.int64))
        file.create_dataset("large", data=large)
        file.create_group("created_groups")
        file.create_group("created_datasets")


class _BenchFile:
    """Own one implementation's fixture, workers, and benchmark operations."""

    __slots__ = {
        "_executor": "Worker pool for thread scenarios.",
        "_file": "Open benchmark file.",
        "_file_factory": "Factory used for reopening the file.",
        "_group_index": "Number of created benchmark groups.",
        "_independent": "Open per-thread independent files.",
        "_dataset_index": "Number of created benchmark datasets.",
        "_large": "Expected large dataset content.",
        "_path": "Path of the private fixture.",
        "_small": "Expected small dataset content.",
        "_threads": "Number of worker threads.",
    }

    def __init__(
        self, file_factory: Callable, path: Path, large: np.ndarray, *, threads: int
    ) -> None:
        """Open a private fixture and initialize the thread pool."""
        self._file_factory = file_factory
        self._path = path
        self._file = file_factory(path, "r+")
        self._independent = []
        self._large = large
        self._small = np.arange(256, dtype=np.int64)
        self._threads = threads
        self._executor = ThreadPoolExecutor(max_workers=threads)
        self._group_index = 0
        self._dataset_index = 0

    def close(self) -> None:
        """Stop workers and explicitly close the benchmark file."""
        self._executor.shutdown(wait=True)
        try:
            for file in self._independent:
                file.close()
        finally:
            self._file.close()

    def add_independent(self, path: Path) -> None:
        """Open an independent per-thread fixture."""
        self._independent.append(self._file_factory(path, "r"))

    def workload(self, name: str, *, iterations: int) -> _Workload:
        """Construct an equivalent operation batch and verification for a scenario."""
        file = self._file
        small = self._small
        large = self._large
        group = file["tree/branch"]
        small_dataset = file["small"]
        large_dataset = file["large"]
        selection = slice(128, 384)
        slice_data = large[selection].copy()

        # Large transfers use fewer repetitions to keep benchmark duration bounded.
        count = max(1, iterations // 20) if name.endswith("_large") else iterations
        byte_count = 0

        def verify_noop() -> None:
            """Verify that the shared fixture remains readable."""
            if int(file.attrs["number"]) != 42:
                raise AssertionError("fixture attribute changed unexpectedly")

        verify = verify_noop
        if name == "file_open_close":
            def operation() -> None:
                """Open and close the existing file."""
                with self._file_factory(self._path, "r"):
                    pass
        elif name == "group_lookup":
            def operation() -> None:
                """Resolve an existing nested group."""
                file["tree/branch"]
            def verify() -> None:
                """Verify the resolved group contains the expected children."""
                if len(file["tree/branch"]) != 32:
                    raise AssertionError("group lookup returned incorrect children")
        elif name == "group_list":
            def operation() -> None:
                """List existing group links."""
                list(group.keys())
            def verify() -> None:
                """Verify the complete set of group names."""
                if list(group.keys()) != [f"child_{index:02d}" for index in range(32)]:
                    raise AssertionError("group enumeration returned incorrect names")
        elif name == "group_create":
            def operation() -> None:
                """Create a distinct group."""
                self._group_index += 1
                file.create_group(f"created_groups/g{self._group_index}")
            def verify() -> None:
                """Check the final created group and group count."""
                created = file["created_groups"]
                if len(created) != self._group_index or f"g{self._group_index}" not in created:
                    raise AssertionError("group creation lost an entry")
        elif name == "attribute_read":
            def operation() -> None:
                """Read a scalar attribute."""
                file.attrs["number"]
        elif name == "attribute_write":
            def operation() -> None:
                """Replace a scalar attribute."""
                file.attrs["number"] = np.int64(42)
        elif name == "dataset_create_small":
            byte_count = int(small.nbytes)
            def operation() -> None:
                """Create a distinct small dataset."""
                self._dataset_index += 1
                file.create_dataset(f"created_datasets/d{self._dataset_index}", data=small)
            def verify() -> None:
                """Check the last dataset and number of created datasets."""
                group_created = file["created_datasets"]
                if len(group_created) != self._dataset_index:
                    raise AssertionError("dataset creation lost an entry")
                np.testing.assert_array_equal(group_created[f"d{self._dataset_index}"][:], small)
        elif name == "dataset_read_small":
            byte_count = int(small.nbytes)
            def operation() -> None:
                """Read the whole small dataset."""
                small_dataset[:]
            def verify() -> None:
                """Verify small dataset values."""
                np.testing.assert_array_equal(small_dataset[:], small)
        elif name == "dataset_write_small":
            byte_count = int(small.nbytes)
            def operation() -> None:
                """Write the whole small dataset."""
                small_dataset[:] = small
            def verify() -> None:
                """Verify the written small dataset."""
                np.testing.assert_array_equal(small_dataset[:], small)
        elif name == "dataset_read_large":
            byte_count = int(large.nbytes)
            def operation() -> None:
                """Read the whole large dataset."""
                large_dataset[:]
            def verify() -> None:
                """Verify large dataset values."""
                np.testing.assert_array_equal(large_dataset[:], large)
        elif name == "dataset_write_large":
            byte_count = int(large.nbytes)
            def operation() -> None:
                """Write the whole large dataset."""
                large_dataset[:] = large
            def verify() -> None:
                """Verify the written large dataset."""
                np.testing.assert_array_equal(large_dataset[:], large)
        elif name == "dataset_read_slice":
            byte_count = int(slice_data.nbytes)
            def operation() -> None:
                """Read a small contiguous hyperslab."""
                large_dataset[selection]
            def verify() -> None:
                """Verify the selected values."""
                np.testing.assert_array_equal(large_dataset[selection], slice_data)
        elif name == "dataset_write_slice":
            byte_count = int(slice_data.nbytes)
            def operation() -> None:
                """Write a small contiguous hyperslab."""
                large_dataset[selection] = slice_data
            def verify() -> None:
                """Verify that the entire dataset retains its expected contents."""
                np.testing.assert_array_equal(large_dataset[:], large)
        elif name in (
            "threaded_read_shared", "threaded_read_independent", "threaded_read_separate_files"
        ):
            byte_count = int(small.nbytes)
            per_worker = max(1, count // self._threads)
            count = per_worker * self._threads
            separate_datasets = [item["small"] for item in self._independent]

            def threaded_run() -> None:
                """Dispatch equal batches to synchronized workers."""
                barrier = Barrier(self._threads)

                def worker(index: int) -> None:
                    """Read from the scenario's assigned file or dataset."""
                    barrier.wait()
                    for _ in range(per_worker):
                        if name == "threaded_read_shared":
                            small_dataset[:]
                        elif name == "threaded_read_separate_files":
                            separate_datasets[index][:]
                        else:
                            with self._file_factory(self._path, "r") as separate:
                                separate["small"][:]

                futures = [self._executor.submit(worker, index) for index in range(self._threads)]
                for future in futures:
                    future.result()

            def verify() -> None:
                """Verify the shared or independently opened dataset."""
                if name == "threaded_read_shared":
                    np.testing.assert_array_equal(small_dataset[:], small)
                elif name == "threaded_read_separate_files":
                    for dataset in separate_datasets:
                        np.testing.assert_array_equal(dataset[:], small)
                else:
                    with self._file_factory(self._path, "r") as separate:
                        np.testing.assert_array_equal(separate["small"][:], small)
            return _Workload(run=threaded_run, verify=verify, operations=count,
                             bytes_per_operation=byte_count)
        else:
            raise ValueError(f"unknown benchmark scenario: {name}")

        def run() -> None:
            """Execute one timed batch of identical operations."""
            for _ in range(count):
                operation()

        return _Workload(run=run, verify=verify, operations=count, bytes_per_operation=byte_count)


def _run(config: _Config, *, directory: Path) -> dict:
    """Run alternating benchmark batches and return per-sample results."""
    # Create one reference fixture and independent copies for each implementation.
    elements = config.size_mib * _MIB // np.dtype(np.float64).itemsize
    large = np.arange(elements, dtype=np.float64)
    template = directory / "template.h5"
    _create_fixture(template, large)
    samples: dict[str, dict[str, list[float]]] = {
        scenario: {"h5py": [], "python-hdf5": []} for scenario in config.scenarios
    }
    results: dict[str, dict] = {}

    with ExitStack() as stack:
        backends: dict[str, _BenchFile] = {}
        for name, factory in (("h5py", h5py.File), ("python-hdf5", hdf5.File)):
            path = directory / f"{name}.h5"
            shutil.copyfile(template, path)
            backend = _BenchFile(factory, path, large, threads=config.threads)
            stack.callback(backend.close)
            backends[name] = backend
            if "threaded_read_separate_files" in config.scenarios:
                for index in range(config.threads):
                    other = directory / f"{name}-thread-{index}.h5"
                    shutil.copyfile(template, other)
                    backend.add_independent(other)

        # Alternate backend order across warmups and measured samples.
        for scenario in config.scenarios:
            workloads = {
                name: backend.workload(scenario, iterations=config.iterations)
                for name, backend in backends.items()
            }
            for sample in range(config.warmup + config.repeats):
                order = ("h5py", "python-hdf5") if sample % 2 == 0 else ("python-hdf5", "h5py")
                for name in order:
                    workload = workloads[name]
                    start = time.perf_counter_ns()
                    workload.run()
                    elapsed = time.perf_counter_ns() - start
                    workload.verify()
                    if sample >= config.warmup:
                        samples[scenario][name].append(elapsed / workload.operations)
            results[scenario] = {
                "operations_per_sample": workloads["h5py"].operations,
                "bytes_per_operation": workloads["h5py"].bytes_per_operation,
                "samples_ns_per_operation": samples[scenario],
                "median_ns_per_operation": {
                    name: statistics.median(values) for name, values in samples[scenario].items()
                },
            }

    # Record the environment needed to interpret or reproduce the comparison.
    return {
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "h5py_hdf5": h5py.version.hdf5_version,
            "python_hdf5": importlib.metadata.version("python-hdf5"),
            "python_hdf5_hdf5": "2.2.0 (bundled build)",
            "gil_enabled": sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True,
            "temporary_directory": str(directory),
        },
        "config": {
            "iterations": config.iterations,
            "repeats": config.repeats,
            "warmup": config.warmup,
            "size_mib": config.size_mib,
            "threads": config.threads,
            "scenarios": list(config.scenarios),
        },
        "results": results,
    }


def _print_report(report: dict) -> None:
    """Print per-operation medians, ratios, and available transfer throughput."""
    print(json.dumps(report["environment"], indent=2))
    print("\nScenario                     h5py us/op  python-hdf5 us/op  h5py/new  MiB/s (h5py/new)")
    for name, result in report["results"].items():
        medians = result["median_ns_per_operation"]
        baseline = medians["h5py"]
        candidate = medians["python-hdf5"]
        size = result["bytes_per_operation"]
        throughput = (
            f"{size / _MIB / (baseline / 1e9):.1f}/{size / _MIB / (candidate / 1e9):.1f}"
            if size else "-"
        )
        print(f"{name:28} {baseline / 1000:10.2f} {candidate / 1000:18.2f} "
              f"{baseline / candidate:9.2f}  {throughput}")


def main() -> None:
    """Run the benchmark and optionally persist the raw JSON samples."""
    args = _parse_args()
    scenarios = tuple(dict.fromkeys(args.scenario or _SCENARIOS))
    config = _Config(
        iterations=args.iterations,
        repeats=args.repeats,
        warmup=args.warmup,
        size_mib=args.size_mib,
        threads=args.threads,
        scenarios=scenarios,
    )
    with tempfile.TemporaryDirectory(prefix="python-hdf5-bench-", dir=args.directory) as temp:
        report = _run(config, directory=Path(temp))
    gc.collect()
    _print_report(report)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
