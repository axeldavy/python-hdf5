"""Test group traversal, creation, and ownership."""

import gc
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from hdf5 import Dataset, File, Group


def _group(parent: Group, name: str) -> Group:
    """Look up a group and verify its type."""
    result = parent[name]
    assert isinstance(result, Group)
    return result


def test_group_create_and_traverse(tmp_path: Path) -> None:
    """Create nested groups and navigate relative and absolute paths."""
    with File(tmp_path / "groups.h5", "w") as file:
        nested = file.create_group("zeta/inner")
        file.create_group("alpha")
        nested.create_group("leaf")
        assert isinstance(nested, Group)
        assert file.keys() == ["alpha", "zeta"]
        assert list(file) == ["alpha", "zeta"]
        assert len(file) == 2
        assert _group(file, "/zeta/inner").keys() == ["leaf"]
        assert _group(nested, "/alpha").keys() == []
        assert _group(nested, ".").keys() == ["leaf"]
        assert "zeta/inner/leaf" in file
        assert "zeta/missing" not in file
        assert "/" in nested
        with pytest.raises(KeyError):
            file["zeta/missing"]
        with pytest.raises(OSError):
            file.create_group("zeta")
    with File(tmp_path / "groups.h5") as file:
        assert _group(file, "zeta/inner").keys() == ["leaf"]


def test_group_snapshot_and_names(tmp_path: Path) -> None:
    """Return snapshots and support UTF-8 and bytes names."""
    with File(tmp_path / "names.h5", "w") as file:
        snapshot = file.keys()
        file.create_group("caf\u00e9")
        file.create_group(b"ascii")
        assert snapshot == []
        assert file.keys() == ["ascii", "caf\u00e9"]
        assert isinstance(file["caf\u00e9"], Group)
        assert isinstance(file[b"ascii"], Group)


def test_group_convenience_api_and_ownership(tmp_path: Path) -> None:
    """Expose snapshot lookups, path properties, and link deletion."""
    path = tmp_path / "convenience.h5"
    with File(path, "w") as file:
        assert file.name == str(path)
        assert file.parent is file
        nested = file.require_group("parent/child")
        assert nested.name == "/parent/child"
        assert nested.parent.name == "/parent"
        assert nested.parent.parent.name == "/"
        assert nested.file.keys() == ["parent"]
        assert file.file is file

        dataset = nested.create_dataset("values", data=[1, 2])
        assert dataset.name == "/parent/child/values"
        assert dataset.parent.name == nested.name
        assert dataset.file["parent/child/values"].read().tolist() == [1, 2]

        assert file.get("missing") is None
        marker = object()
        assert file.get("missing", marker) is marker
        assert file.get("parent").name == file["parent"].name
        assert [name for name, _ in nested.items()] == ["values"]
        assert [value.name for value in nested.values()] == [dataset.name]

        del nested["values"]
        assert "values" not in nested
        with pytest.raises(KeyError):
            del nested["values"]


def test_group_require_group_existing_and_wrong_type(tmp_path: Path) -> None:
    """Reuse existing groups and reject dataset paths."""
    with File(tmp_path / "require-group.h5", "w") as file:
        first = file.require_group("parent/child")
        second = file.require_group("parent/child")
        assert first.name == second.name == "/parent/child"
        assert file.require_group("/").name == "/"
        file.create_dataset("values", data=[1])
        with pytest.raises(TypeError, match="not a group"):
            file.require_group("values")


def test_group_invalid_inputs_and_readonly(tmp_path: Path) -> None:
    """Reject invalid paths without creating partial groups."""
    path = tmp_path / "invalid.h5"
    with File(path, "w") as file:
        for name in ("", "/", "new/../other", "name\0bad"):
            with pytest.raises(ValueError):
                file.create_group(name)
        assert file.keys() == []
        with pytest.raises(TypeError):
            getattr(file, "__getitem__")(42)
        with pytest.raises(TypeError):
            Group()
        with pytest.raises(ValueError, match="not initialized"):
            Group.__new__(Group).keys()
    with File(path, "r") as file:
        with pytest.raises(OSError):
            file.create_group("readonly")


def test_group_file_lifetime_and_close(tmp_path: Path) -> None:
    """Keep the native file alive through groups and invalidate them on close."""
    path = tmp_path / "lifetime.h5"
    file = File(path, "w")
    child = file.create_group("parent/child")
    del file
    gc.collect()
    child.create_group("still_open")
    del child
    gc.collect()
    with File(path, "r+") as reopened:
        assert _group(reopened, "parent/child").keys() == ["still_open"]
        child = _group(reopened, "parent/child")
    with pytest.raises(ValueError, match="closed"):
        child.keys()
    with pytest.raises(ValueError, match="closed"):
        child["still_open"]
    with pytest.raises(ValueError, match="closed"):
        child.create_group("new")
    with pytest.raises(ValueError, match="closed"):
        _ = "anything" in child
    del child
    gc.collect()


def test_group_stale_handle_does_not_close_reopened_file(tmp_path: Path) -> None:
    """Do not close a reused HDF5 identifier when an invalidated group is collected."""
    path = tmp_path / "reopened.h5"
    first = File(path, "w")
    stale = first.create_group("original")
    first.close()
    with File(path, "r+") as second:
        second.create_group("new")
        del stale
        gc.collect()
        assert second.keys() == ["new", "original"]


def test_group_independent_file_close(tmp_path: Path) -> None:
    """Keep independently opened file and group handles valid after close."""
    path = tmp_path / "multiple.h5"
    with File(path, "w") as first:
        first.create_group("group")
        with File(path, "r+") as second:
            group = _group(second, "group")
            first.close()
            assert group.keys() == []
            group.create_group("child")


def test_group_concurrent_operations_and_close(tmp_path: Path) -> None:
    """Serialize group operations with file close."""
    file = File(tmp_path / "concurrent.h5", "w")
    group = file.create_group("group")
    group.create_group("child")
    barrier = Barrier(9)

    def worker() -> None:
        """Read a group until the file closes."""
        barrier.wait()
        for _ in range(100):
            try:
                assert group.keys() == ["child"]
                assert "child" in group
                assert _group(group, "child").keys() == []
            except ValueError as error:
                assert "closed" in str(error)
                return

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker) for _ in range(8)]
        barrier.wait()
        file.close()
        for future in futures:
            future.result(timeout=10)


def test_group_h5py_interoperability(tmp_path: Path, h5py_python: str) -> None:
    """Read h5py groups and make groups that h5py can inspect."""
    path = tmp_path / "interop.h5"
    create = (
        "import h5py, sys\n"
        "with h5py.File(sys.argv[1], 'w') as f:\n"
        " f.create_group('existing/child')\n"
        " f['alias'] = f['existing']\n"
        " f.create_dataset('data', data=[1, 2])\n"
        " f['soft'] = h5py.SoftLink('/existing')\n"
        " f['external'] = h5py.ExternalLink('other.h5', '/')\n"
    )
    subprocess.run([h5py_python, "-c", create, str(path)], check=True)
    with File(path, "r+") as file:
        assert _group(file, "existing").keys() == ["child"]
        assert file.keys() == ["alias", "data", "existing", "external", "soft"]
        assert _group(file, "alias").keys() == ["child"]
        assert "data" in file
        assert isinstance(file["data"], Dataset)
        with pytest.raises(TypeError, match="not a group"):
            file["data/child"]
        for name in ("soft", "external", "soft/child"):
            with pytest.raises(TypeError, match="links"):
                file[name]
        with pytest.raises(TypeError, match="links"):
            _ = "soft" in file
        with pytest.raises(TypeError, match="links"):
            file.create_group("soft/new")
        file.create_group("new/nested")
        file.create_group("new/caf\u00e9")
    check = (
        "import h5py, sys\n"
        "with h5py.File(sys.argv[1], 'r') as f:\n"
        " assert 'new/nested' in f\n"
        " assert 'new/caf\\u00e9' in f\n"
        " assert f['data'][:].tolist() == [1, 2]\n"
    )
    subprocess.run([h5py_python, "-c", check, str(path)], check=True)
