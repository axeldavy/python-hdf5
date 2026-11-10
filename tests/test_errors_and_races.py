"""Test erroneous API use and consistency when several objects share the same entries."""

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import hdf5
from hdf5 import Dataset, File, Group

_ITERATIONS = 300


def _ds(file: File | Group, name: str) -> Dataset:
    """Return the dataset stored under name."""
    dataset = file[name]
    assert isinstance(dataset, Dataset)
    return dataset


def _grp(file: File | Group, name: str) -> Group:
    """Return the group stored under name."""
    group = file[name]
    assert isinstance(group, Group)
    return group


def _run_threads(*workers: Callable[[], None]) -> None:
    """Run workers together, re-raising the first exception any of them raised."""
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(workers))

    def wrap(worker: Callable[[], None]) -> Callable[[], None]:
        """Return worker guarded by a start barrier and error collection."""

        def run() -> None:
            """Execute the worker and record failures."""
            try:
                barrier.wait()
                worker()
            except BaseException as error:  # noqa: BLE001 - reported to the test below
                errors.append(error)

        return run

    threads = [threading.Thread(target=wrap(worker)) for worker in workers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "deadlock"
    if errors:
        raise errors[0]


# Sequential visibility between independent objects


def test_attributes_second_manager_sees_add_replace_delete(tmp_path: Path) -> None:
    """Show every change made through one manager to another manager of the same object."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=[1])
        first, second = _ds(file, "d").attrs, _ds(file, "d").attrs

        # Add through the first manager
        first["x"] = 1
        assert second["x"] == 1 and "x" in second and second.keys() == ["x"]

        # Delete through the first manager and observe absence everywhere
        del first["x"]
        assert "x" not in second and len(second) == 0
        with pytest.raises(KeyError):
            second["x"]
        with pytest.raises(KeyError):
            del second["x"]

        # Re-add with another type through the second manager
        second["x"] = np.arange(3, dtype="i2")
        value = first["x"]
        assert isinstance(value, np.ndarray) and value.dtype == np.dtype("i2")

        # Replace with a different dtype, shape and class
        first["x"] = "text"
        assert second["x"] == "text"
        first["x"] = np.float64(2.5)
        assert second["x"] == 2.5 and len(first) == len(second) == 1


def test_counts_follow_changes_across_file_handles(tmp_path: Path) -> None:
    """Count links and attributes without caching names across file handles."""
    path = tmp_path / "counts.h5"
    first = File(path, "w")
    group = first.create_group("group")
    second = File(path, "r+")
    other = _grp(second, "group")
    attrs = group.attrs
    other_attrs = other.attrs
    assert len(group) == len(other) == len(attrs) == len(other_attrs) == 0
    for index in range(50):
        group.create_dataset(f"d{index}", data=index)
        attrs[f"a{index}"] = index
    group.create_group(b"\xff")
    attrs[b"\xff"] = 1
    assert len(other) == len(other_attrs) == 51
    del other[b"\xff"]
    del other_attrs[b"\xff"]
    del other["d0"]
    del other_attrs["a0"]
    assert len(group) == len(attrs) == 49
    first.close()
    for value in (group, attrs):
        with pytest.raises(ValueError, match="closed"):
            len(value)
    assert len(other) == len(other_attrs) == 49
    second.close()


def test_attributes_of_group_and_dataset_are_independent(tmp_path: Path) -> None:
    """Keep attributes with the same name on different objects separate."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=[1])
        file.create_group("g")
        file.attrs["n"] = 1
        _grp(file, "g").attrs["n"] = 2
        _ds(file, "d").attrs["n"] = 3
        del file.attrs["n"]
        assert _grp(file, "g").attrs["n"] == 2 and _ds(file, "d").attrs["n"] == 3
        assert len(file.attrs) == 0


def test_attribute_snapshot_does_not_change_after_keys_returned(tmp_path: Path) -> None:
    """Return snapshots, so later changes do not alter keys already taken."""
    with File(tmp_path / "a.h5", "w") as file:
        file.attrs["a"] = 1
        file.attrs["b"] = 2
        snapshot = file.attrs.keys()
        iterator = iter(file.attrs)
        del file.attrs["a"]
        file.attrs["c"] = 3
        assert snapshot == ["a", "b"]
        assert list(iterator) == ["a", "b"]
        assert file.attrs.keys() == ["b", "c"]


def test_datasets_second_object_sees_writes_and_new_links(tmp_path: Path) -> None:
    """Share data and group contents between separately opened objects."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=np.zeros(4, dtype="i4"))
        first, second = _ds(file, "d"), _ds(file, "d")
        first[1:3] = 5
        assert second[:].tolist() == [0, 5, 5, 0]
        second[:] = 7
        assert first[()].tolist() == [7] * 4

        # Groups opened twice see each other's children
        one, two = file.create_group("g"), _grp(file, "g")
        one.create_dataset("n", data=1)
        two.create_group("sub")
        assert one.keys() == two.keys() == ["n", "sub"]
        assert "n" in two and "sub" in one


def test_dataset_handle_stays_usable_after_sibling_changes(tmp_path: Path) -> None:
    """Keep other handles working when a sibling attribute or link set changes."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=[1, 2])
        dataset = _ds(file, "d")
        for index in range(20):
            file.create_dataset(f"other{index}", data=index)
            dataset.attrs[f"a{index}"] = index
        assert dataset[:].tolist() == [1, 2]
        assert len(dataset.attrs) == 20 and len(file) == 21


# Erroneous use: links, paths and objects


def test_errors_duplicate_and_conflicting_links(tmp_path: Path) -> None:
    """Refuse to overwrite links and keep the original object intact."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=[1])
        file.create_group("g")
        for create in (
            lambda: file.create_group("g"),
            lambda: file.create_group("d"),
            lambda: file.create_dataset("g", data=1),
            lambda: file.create_dataset("d", data=2),
        ):
            with pytest.raises(OSError, match="already exists"):
                create()
        assert _ds(file, "d")[:].tolist() == [1] and file.keys() == ["d", "g"]


def test_errors_paths_through_datasets_and_missing_parents(tmp_path: Path) -> None:
    """Report type and lookup errors for paths that cross non-groups or missing links."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=[1])
        for operation in (
            lambda: file.create_group("d/x"),
            lambda: file["d/x"],
            lambda: file.create_dataset("d/x", data=1),
        ):
            with pytest.raises(TypeError):
                operation()
        assert "d/x" not in file and "d" in file
        with pytest.raises(KeyError):
            file["missing"]
        with pytest.raises(KeyError):
            file["missing/deeper"]
        with pytest.raises(KeyError):
            file.create_dataset("missing/d", data=1)
        assert "missing" not in file and file.keys() == ["d"]


def test_errors_unsupported_operations_are_explicit(tmp_path: Path) -> None:
    """Fail loudly on unsupported API instead of silently ignoring it."""
    with File(tmp_path / "a.h5", "w") as file:
        with pytest.raises(KeyError):
            del file["x"]
        with pytest.raises(TypeError):
            file.create_group(1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            file.attrs[1] = 1  # type: ignore[index]
        with pytest.raises(TypeError):
            file.attrs["none"] = None  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            hdf5.Group()
        with pytest.raises(TypeError):
            hdf5.Dataset()
        with pytest.raises(TypeError):
            hdf5.AttributeManager()


@pytest.mark.parametrize(
    ("selection", "error"),
    [
        (1.5, TypeError),
        (None, TypeError),
        (True, TypeError),
        ([0, 1], TypeError),
        (10, IndexError),
        (-10, IndexError),
        ((0, 0), IndexError),
        ((Ellipsis, Ellipsis), IndexError),
        (slice(None, None, 0), ValueError),
        (slice(None, None, -1), ValueError),
    ],
)
def test_errors_invalid_selections_leave_data_unchanged(
    tmp_path: Path, selection: object, error: type[Exception]
) -> None:
    """Reject invalid selections for reads and writes without modifying the dataset."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=np.arange(4, dtype="i4"))
        dataset = _ds(file, "d")
        with pytest.raises(error):
            dataset[selection]  # type: ignore[index]
        with pytest.raises(error):
            dataset[selection] = 9  # type: ignore[index]
        assert dataset[:].tolist() == [0, 1, 2, 3]


def test_errors_invalid_write_values_leave_data_unchanged(tmp_path: Path) -> None:
    """Reject mismatching values without partially writing the dataset."""
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=np.arange(4, dtype="i4"))
        dataset = _ds(file, "d")
        with pytest.raises(TypeError):
            dataset[:] = "text"
        with pytest.raises(TypeError):
            dataset[:] = np.array([object()] * 4)
        with pytest.raises(ValueError):
            dataset.write(np.zeros(3))
        with pytest.raises(ValueError):
            dataset[0:2] = [1, 2, 3]
        with pytest.raises(TypeError):
            len(file.create_dataset("scalar", data=1))
        with pytest.raises(TypeError):
            dataset.asstr()
        assert dataset[:].tolist() == [0, 1, 2, 3]


def test_errors_read_only_file_rejects_every_mutation(tmp_path: Path) -> None:
    """Fail with OSError for each mutation and leave the file unchanged."""
    path = tmp_path / "a.h5"
    with File(path, "w") as file:
        file.create_dataset("d", data=[1, 2])
        file.attrs["a"] = 1
    with File(path, "r") as file:
        dataset = _ds(file, "d")
        for mutation in (
            lambda: file.create_group("g"),
            lambda: file.create_dataset("n", data=1),
            lambda: file.attrs.__setitem__("b", 1),
            lambda: file.attrs.__delitem__("a"),
            lambda: dataset.__setitem__(slice(None), 5),
            lambda: dataset.write([5, 5]),
        ):
            with pytest.raises(OSError):
                mutation()
        assert dataset[:].tolist() == [1, 2] and file.attrs["a"] == 1
        assert file.keys() == ["d"] and list(file.attrs) == ["a"]


def test_errors_file_open_failures(tmp_path: Path) -> None:
    """Report invalid files, directories and a second writer in the same process."""
    path = tmp_path / "a.h5"
    junk = tmp_path / "junk.h5"
    junk.write_bytes(b"not an hdf5 file")
    with File(path, "w"):
        with pytest.raises(OSError):
            File(path, "w")
        with pytest.raises(OSError):
            File(path, "w-")
    with pytest.raises(OSError):
        File(junk, "r")
    with pytest.raises(OSError):
        File(tmp_path, "r")
    with pytest.raises(OSError):
        File(tmp_path / "missing.h5", "r")
    with pytest.raises(TypeError):
        File(None, "r")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        File(path, "z")
    assert junk.read_bytes() == b"not an hdf5 file"


def test_errors_everything_raises_value_error_after_close(tmp_path: Path) -> None:
    """Invalidate every derived object when the file closes, without crashing."""
    file = File(tmp_path / "a.h5", "w")
    group = file.create_group("g")
    dataset = group.create_dataset("d", data=[1, 2])
    attrs = dataset.attrs
    attrs["a"] = 1
    file.close()
    operations: list[Callable[[], object]] = [
        lambda: group.keys(),
        lambda: group["d"],
        lambda: "d" in group,
        lambda: group.create_group("x"),
        lambda: group.attrs["a"],
        lambda: dataset.shape,
        lambda: dataset.dtype,
        lambda: dataset[:],
        lambda: dataset.__setitem__(slice(None), 1),
        lambda: dataset.read(),
        lambda: attrs["a"],
        lambda: attrs.__setitem__("b", 1),
        lambda: attrs.__delitem__("a"),
        lambda: attrs.keys(),
        lambda: "a" in attrs,
        lambda: file.keys(),
        lambda: file.flush(),
    ]
    for operation in operations:
        with pytest.raises(ValueError):
            operation()
    file.close()
    assert file.closed
    with pytest.raises(RuntimeError):
        file.__init__(tmp_path / "a.h5", "r")  # type: ignore[misc]


def test_errors_closing_one_file_does_not_affect_another(tmp_path: Path) -> None:
    """Keep objects of a different file working after closing a sibling file."""
    first, second = File(tmp_path / "1.h5", "w"), File(tmp_path / "2.h5", "w")
    for file in (first, second):
        file.create_dataset("d", data=[1])
        file.attrs["a"] = 1
    dataset = _ds(second, "d")
    first.close()
    assert dataset[:].tolist() == [1] and second.attrs["a"] == 1
    with pytest.raises(ValueError):
        first.attrs["a"]
    second.close()


# Concurrent use of the same entries through different objects


def test_race_attribute_add_delete_replace_versus_readers(tmp_path: Path) -> None:
    """Allow only complete values or KeyError while another manager changes an attribute."""
    valid = {1, 2, 3}
    stop = threading.Event()
    seen: list[int] = []

    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=[1])
        writer_attrs, reader_attrs, lister_attrs = (_ds(file, "d").attrs for _ in range(3))

        def writer() -> None:
            """Add, replace and delete the shared attribute repeatedly."""
            try:
                for index in range(_ITERATIONS):
                    writer_attrs["x"] = index % 3 + 1
                    writer_attrs["x"] = np.array([index % 3 + 1] * 4, dtype="i8")
                    if index % 2:
                        del writer_attrs["x"]
            finally:
                stop.set()

        def reader() -> None:
            """Read the attribute, tolerating only absence or a complete old or new value."""
            while not stop.is_set():
                try:
                    value = np.asarray(reader_attrs["x"])
                except KeyError:
                    continue
                assert value.shape in ((), (4,))
                assert len(set(value.reshape(-1).tolist())) == 1
                assert int(value.reshape(-1)[0]) in valid
                seen.append(1)

        def lister() -> None:
            """List and test membership while the attribute churns."""
            while not stop.is_set():
                assert set(lister_attrs.keys()) <= {"x"}
                assert len(lister_attrs) in (0, 1)
                "x" in lister_attrs  # noqa: B015 - exercised for crash safety only

        _run_threads(writer, reader, lister)
        assert set(file.attrs.keys()) == set() and len(_ds(file, "d").attrs) in (0, 1)


def test_race_deleting_attribute_from_two_managers(tmp_path: Path) -> None:
    """Let exactly one concurrent deleter win and the other see KeyError."""
    with File(tmp_path / "a.h5", "w") as file:
        file.attrs["x"] = 1
        results: list[str] = []
        lock = threading.Lock()

        def deleter() -> None:
            """Delete the shared attribute once through a private manager."""
            attrs = file.attrs
            outcome = "deleted"
            try:
                del attrs["x"]
            except KeyError:
                outcome = "missing"
            with lock:
                results.append(outcome)

        for _ in range(40):
            file.attrs["x"] = 1
            results.clear()
            _run_threads(deleter, deleter, deleter)
            assert sorted(results) == ["deleted", "missing", "missing"]
            assert "x" not in file.attrs


def test_race_attribute_writers_never_corrupt_each_other(tmp_path: Path) -> None:
    """Keep the final attribute equal to one writer's complete value."""
    with File(tmp_path / "a.h5", "w") as file:

        def writer(marker: int) -> Callable[[], None]:
            """Return a worker that repeatedly stores an array filled with marker."""

            def run() -> None:
                """Replace the shared attribute with marker-filled arrays."""
                attrs = file.attrs
                for _ in range(_ITERATIONS):
                    attrs["x"] = np.full(64, marker, dtype="i4")

            return run

        _run_threads(writer(1), writer(2), writer(3))
        final = np.asarray(file.attrs["x"])
        assert final.shape == (64,) and len(set(final.tolist())) == 1
        assert int(final[0]) in (1, 2, 3) and file.attrs.keys() == ["x"]


def test_race_dataset_writes_are_not_torn(tmp_path: Path) -> None:
    """Return only uniform arrays when two objects write whole arrays while one reads."""
    stop = threading.Event()
    with File(tmp_path / "a.h5", "w") as file:
        file.create_dataset("d", data=np.zeros((64, 64), dtype="i4"))
        first, second, reader_ds = (_ds(file, "d") for _ in range(3))

        def writer(dataset: Dataset, value: int) -> Callable[[], None]:
            """Return a worker writing complete arrays or slabs filled with value."""

            def run() -> None:
                """Write whole arrays repeatedly."""
                for _ in range(_ITERATIONS):
                    dataset[:] = np.full((64, 64), value, dtype="i4")

            return run

        def reader() -> None:
            """Check every read is entirely one writer's value."""
            while not stop.is_set():
                data = np.asarray(reader_ds[:])
                assert len(np.unique(data)) == 1, "torn read"
                assert np.unique(data)[0] in (0, 1, 2)

        def wrapper() -> None:
            """Stop the reader when both writers finish."""
            try:
                _run_threads(writer(first, 1), writer(second, 2))
            finally:
                stop.set()

        _run_threads(wrapper, reader)
        assert len(np.unique(first[:])) == 1


def test_race_group_creation_with_listing_and_lookup(tmp_path: Path) -> None:
    """Let two group objects create entries while a third lists and resolves them."""
    stop = threading.Event()
    with File(tmp_path / "a.h5", "w") as file:
        file.create_group("g")
        creators = [_grp(file, "g"), _grp(file, "g")]
        observer = _grp(file, "g")

        def creator(index: int) -> Callable[[], None]:
            """Return a worker creating uniquely named datasets and groups."""

            def run() -> None:
                """Create entries in the shared group."""
                for number in range(_ITERATIONS // 3):
                    creators[index].create_dataset(f"d{index}_{number}", data=number)
                    creators[index].create_group(f"g{index}_{number}")

            return run

        def watcher() -> None:
            """Check snapshots are sorted and every listed name resolves."""
            while not stop.is_set():
                names = observer.keys()
                assert names == sorted(names)
                for name in names[-3:]:
                    assert name in observer
                    assert observer[name] is not None

        def wrapper() -> None:
            """Stop the watcher after both creators complete."""
            try:
                _run_threads(creator(0), creator(1))
            finally:
                stop.set()

        _run_threads(wrapper, watcher)
        assert len(observer) == 2 * 2 * (_ITERATIONS // 3)


def test_race_duplicate_creation_has_exactly_one_winner(tmp_path: Path) -> None:
    """Allow one creator per name and report the loss as OSError for the others."""
    with File(tmp_path / "a.h5", "w") as file:
        for round_number in range(30):
            outcomes: list[str] = []
            lock = threading.Lock()

            def creator(name: str = f"n{round_number}", marker: int = 0) -> None:
                """Try to create the shared dataset and record the outcome."""
                try:
                    file.create_dataset(name, data=marker)
                    outcome = "created"
                except OSError:
                    outcome = "lost"
                with lock:
                    outcomes.append(outcome)

            _run_threads(creator, creator, creator, creator)
            assert sorted(outcomes) == ["created", "lost", "lost", "lost"]


def test_race_close_while_workers_use_shared_objects(tmp_path: Path) -> None:
    """Fail only with ValueError, never crash, when the file closes under live workers."""
    for _ in range(10):
        file = File(tmp_path / "a.h5", "w")
        dataset = file.create_dataset("d", data=np.arange(1000, dtype="i4"))
        attrs = dataset.attrs
        group = file.create_group("g")
        attrs["x"] = 1

        def churn(worker: int) -> Callable[[], None]:
            """Return a worker using every kind of object, with its own names, until close."""

            def run() -> None:
                """Write, churn attributes and create groups until the file closes."""
                try:
                    for index in range(10_000):
                        dataset[:10] = index
                        attrs[f"y{worker}"] = index
                        del attrs[f"y{worker}"]
                        group.create_group(f"n{worker}_{index}")
                        dataset[:]
                except ValueError as error:
                    assert "closed" in str(error)

            return run

        def closer() -> None:
            """Close the file while the workers are running."""
            threading.Event().wait(0.01)
            file.close()

        _run_threads(churn(0), churn(1), closer)
        assert file.closed


# Several File objects on the same path in one process


def test_two_file_objects_share_content_and_lifetime(tmp_path: Path) -> None:
    """Show changes across File objects and keep one usable after the other closes."""
    path = tmp_path / "a.h5"
    with File(path, "w") as file:
        file.create_dataset("d", data=[1, 2])
    first, second = File(path, "r+"), File(path, "r+")
    _ds(first, "d")[0] = 9
    first.attrs["a"] = 1
    first.create_group("g")
    first.flush()
    assert _ds(second, "d")[:].tolist() == [9, 2]
    assert second.attrs["a"] == 1 and second.keys() == ["d", "g"]
    del first.attrs["a"]
    assert "a" not in second.attrs

    # Closing one object must not invalidate the other or lose data
    dataset = _ds(second, "d")
    first.close()
    with pytest.raises(ValueError):
        first.keys()
    assert dataset[:].tolist() == [9, 2]
    dataset[1] = 7
    second.close()
    with File(path, "r") as check:
        assert _ds(check, "d")[:].tolist() == [9, 7] and check.keys() == ["d", "g"]


def test_race_two_file_objects_modifying_attributes_and_data(tmp_path: Path) -> None:
    """Keep two File objects consistent while both write and delete concurrently."""
    path = tmp_path / "a.h5"
    with File(path, "w") as file:
        file.create_dataset("d", data=np.zeros(32, dtype="i4"))
    first, second = File(path, "r+"), File(path, "r+")

    def worker(file: File, marker: int) -> Callable[[], None]:
        """Return a worker churning attributes and writing marker-filled data."""

        def run() -> None:
            """Alternate attribute adds, deletes and whole-array writes."""
            dataset = _ds(file, "d")
            for index in range(_ITERATIONS):
                dataset[:] = np.full(32, marker, dtype="i4")
                file.attrs[f"a{marker}"] = index
                value = np.asarray(dataset[:])
                assert len(np.unique(value)) == 1
                if index % 2:
                    try:
                        del file.attrs[f"a{marker}"]
                    except KeyError:
                        pytest.fail("a private attribute disappeared")

        return run

    _run_threads(worker(first, 1), worker(second, 2))
    assert first.attrs.keys() == second.attrs.keys()
    assert _ds(first, "d")[:].tolist() == _ds(second, "d")[:].tolist()
    first.close()
    second.close()


def test_contains_matches_getitem_for_paths_through_datasets(tmp_path: Path) -> None:
    """Return True from `in` exactly when indexing the same path succeeds."""
    with File(tmp_path / "c.h5", "w") as file:
        file.create_group("g").create_dataset("d", data=[1])
        for path in ("g", "g/d", "g/d/x", "g/missing", "nope/x"):
            try:
                file[path]
                found = True
            except (KeyError, TypeError):
                found = False
            assert (path in file) is found


def test_astype_rejects_unsafe_conversions(tmp_path: Path) -> None:
    """Allow numeric and text conversions only; reject reinterpretation as raw bytes."""
    with File(tmp_path / "t.h5", "w") as file:
        dataset = file.create_dataset("n", data=np.arange(3, dtype="<i4"))
        assert list(dataset.astype("<f8")[:]) == [0.0, 1.0, 2.0]
        for target in ("V3", "S4", "M8[s]"):
            with pytest.raises(TypeError):
                dataset.astype(target)[:]


def test_attribute_replace_validation_failure_keeps_old_value(tmp_path: Path) -> None:
    """Keep the old attribute when validation fails before replacement."""
    with File(tmp_path / "r.h5", "w") as file:
        file.attrs["a"] = [1, 2, 3]
        with pytest.raises((TypeError, ValueError)):
            file.attrs["a"] = np.array([object()], dtype=object)
        assert np.asarray(file.attrs["a"]).tolist() == [1, 2, 3]
        file.attrs["a"] = "new"
        assert file.attrs.keys() == ["a"] and len(file.attrs) == 1


@pytest.mark.parametrize("owner", ["file", "group", "dataset"])
def test_attribute_changes_visible_across_threads_and_file_handles(
    tmp_path: Path, owner: str
) -> None:
    """Observe completed changes without a flush or cached attribute data."""
    path = tmp_path / "visibility.h5"
    with File(path, "w") as first:
        first.create_group("g")
        first.create_dataset("d", data=[1])
        with File(path, "r+") as second:
            if owner == "group":
                writer, reader = _grp(first, "g").attrs, _grp(second, "g").attrs
            elif owner == "dataset":
                writer, reader = _ds(first, "d").attrs, _ds(second, "d").attrs
            else:
                writer, reader = first.attrs, second.attrs
            written = threading.Event()
            observed = threading.Event()

            def change() -> None:
                """Replace, then delete, waiting for each observation."""
                for value in (1, "new", None):
                    if value is None:
                        del writer["x"]
                    else:
                        writer["x"] = value
                    written.set()
                    assert observed.wait(timeout=10)
                    observed.clear()

            def check() -> None:
                """Read the state after each write has completed."""
                for value in (1, "new", None):
                    assert written.wait(timeout=10)
                    written.clear()
                    if value is None:
                        assert "x" not in reader and reader.keys() == [] and len(reader) == 0
                        with pytest.raises(KeyError):
                            reader["x"]
                    else:
                        assert reader["x"] == value
                        assert "x" in reader and reader.keys() == ["x"] and len(reader) == 1
                    observed.set()

            _run_threads(change, check)


@pytest.mark.parametrize("owner", ["file", "group", "dataset"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_attribute_concurrent_access_across_file_handles(
    tmp_path: Path, owner: str, hardlink: bool
) -> None:
    """Keep concurrent access safe across independently opened file handles."""
    path = tmp_path / "race-attrs.h5"
    with File(path, "w"):
        pass
    alias = path
    if hardlink:
        alias = tmp_path / "alias.h5"
        try:
            alias.hardlink_to(path)
        except OSError as error:
            pytest.skip(f"Hard links are unavailable: {error}")
    with File(path, "r+") as first, File(alias, "r+") as second:
        first.create_group("g")
        first.create_dataset("d", data=[1])
        if owner == "group":
            first_attrs, second_attrs = _grp(first, "g").attrs, _grp(second, "g").attrs
        elif owner == "dataset":
            first_attrs, second_attrs = _ds(first, "d").attrs, _ds(second, "d").attrs
        else:
            first_attrs, second_attrs = first.attrs, second.attrs
        first_attrs["x"] = np.ones(8, dtype="i4")
        stop = threading.Event()

        def writer(file: File, value: int) -> Callable[[], None]:
            """Return a worker replacing and deleting the shared attribute."""

            def run() -> None:
                """Write and delete, tolerating a concurrent deletion."""
                if owner == "group":
                    attrs = _grp(file, "g").attrs
                elif owner == "dataset":
                    attrs = _ds(file, "d").attrs
                else:
                    attrs = file.attrs
                for index in range(_ITERATIONS):
                    attrs["x"] = np.full(8, value, dtype="i4")
                    if index % 2:
                        try:
                            del attrs["x"]
                        except KeyError:
                            pass

            return run

        def reader() -> None:
            """Exercise read and listing while writes are in progress."""
            attrs = second_attrs
            while not stop.is_set():
                try:
                    value = np.asarray(attrs["x"])
                    assert value.shape == (8,) and value.dtype == np.dtype("i4")
                except KeyError:
                    pass
                assert set(attrs.keys()) <= {"x"}
                assert len(attrs) in (0, 1)
                _ = "x" in attrs

        def writers() -> None:
            """Run competing writers and stop the observer even on failure."""
            try:
                _run_threads(writer(first, 1), writer(second, 2))
            finally:
                stop.set()

        _run_threads(writers, reader)
        first_attrs["x"] = 42
        assert second_attrs["x"] == 42


@pytest.mark.parametrize("owner", ["file", "group", "dataset"])
@pytest.mark.parametrize("extra_attributes", [0, 10])
def test_attribute_delete_while_accessing_another_attribute(
    tmp_path: Path, owner: str, extra_attributes: int
) -> None:
    """Keep distinct attributes intact while another handle deletes and recreates one."""
    path = tmp_path / "different-attributes.h5"
    with File(path, "w") as first:
        first.create_group("g")
        first.create_dataset("d", data=[1])
        with File(path, "r+") as second:
            if owner == "group":
                changing, stable = _grp(first, "g").attrs, _grp(second, "g").attrs
            elif owner == "dataset":
                changing, stable = _ds(first, "d").attrs, _ds(second, "d").attrs
            else:
                changing, stable = first.attrs, second.attrs

            changing["a"] = -1
            stable["z"] = np.zeros(32, dtype="i8")
            for index in range(extra_attributes):
                changing[f"middle{index:02}"] = index

            phase = threading.Barrier(2)

            def replace_other() -> None:
                """Delete and recreate the attribute that sorts before the stable one."""
                for index in range(_ITERATIONS):
                    phase.wait(timeout=10)
                    del changing["a"]
                    changing["a"] = -index - 1
                    phase.wait(timeout=10)

            def update_stable() -> None:
                """Check each stable-attribute write and read during the churn."""
                for index in range(_ITERATIONS):
                    phase.wait(timeout=10)
                    expected = np.full(32, index + 1, dtype="i8")
                    stable["z"] = expected
                    np.testing.assert_array_equal(stable["z"], expected)
                    phase.wait(timeout=10)

            _run_threads(replace_other, update_stable)
            assert changing["a"] == -_ITERATIONS
            np.testing.assert_array_equal(changing["z"], np.full(32, _ITERATIONS, dtype="i8"))
            for index in range(extra_attributes):
                assert stable[f"middle{index:02}"] == index
