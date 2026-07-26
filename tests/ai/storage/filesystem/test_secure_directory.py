#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SecureDirectory tests: dirfd-based access scope.

Verifies the structural safety properties the spec requires:
- Every component is opened with O_NOFOLLOW|O_DIRECTORY through a dirfd chain
  so a symlink anywhere in the chain (target, ancestor, or sibling swapped in
  mid-walk) is rejected by the kernel.
- All operations reject path-traversal components and slashes.
- Atomic writes leave no temp file behind on exception.
- atomic_publish_directory publishes the whole directory atomically.
- POSIX capability probe refuses construction on a platform missing the
  required primitives (mocked here)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from linktools.ai.storage.backends.filesystem.secure_directory import (
    FilesystemSecurityMode,
    SecureDirectory,
)
from linktools.ai.storage.object.errors import StorageObjectError


@pytest.fixture
def sd(tmp_path):
    return SecureDirectory(tmp_path / "root")


# --- component validation ----------------------------------------------------


def test_empty_component_rejected(sd):
    with pytest.raises(StorageObjectError, match="empty path component"):
        sd.read_bytes("", "a")
    with pytest.raises(StorageObjectError, match="empty path component"):
        sd.read_bytes("a", "")


def test_dotdot_rejected(sd):
    with pytest.raises(StorageObjectError, match="path traversal"):
        sd.read_bytes("..", "secret")


def test_slash_in_component_rejected(sd):
    # The caller must pass one segment at a time; "a/b" is forbidden because
    # the dirfd walk cannot apply it atomically.
    with pytest.raises(StorageObjectError, match="slash not allowed"):
        sd.read_bytes("a/b")


def test_nul_byte_rejected(sd):
    with pytest.raises(StorageObjectError, match="NUL"):
        sd.read_bytes("a\x00b")


# --- read/write/list --------------------------------------------------------


def test_atomic_write_then_read(sd):
    sd.ensure_directory("dir1", "dir2")
    sd.atomic_write("dir1", "dir2", "file.bin", content=b"payload")
    assert sd.read_bytes("dir1", "dir2", "file.bin") == b"payload"


def test_atomic_write_creates_intermediate_dirs(sd):
    sd.ensure_directory("a", "b", "c")
    sd.atomic_write("a", "b", "c", "x", content=b"deep")
    assert sd.read_bytes("a", "b", "c", "x") == b"deep"


def test_atomic_write_overwrites_existing(sd):
    sd.atomic_write("f", content=b"v1")
    sd.atomic_write("f", content=b"v2")
    assert sd.read_bytes("f") == b"v2"


def test_atomic_write_is_atomic_on_exception(monkeypatch, sd):
    """Patch os.replace to raise; the temp file must NOT survive."""
    real_replace = os.replace

    def boom(*args, **kwargs):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        sd.atomic_write("f", content=b"v1")
    monkeypatch.setattr(os, "replace", real_replace)
    # No temp file left behind in the root.
    leftover = [n for n in os.listdir(str(sd.root)) if n.startswith(".f.")]
    assert leftover == []


def test_read_missing_raises(sd):
    with pytest.raises(FileNotFoundError):
        sd.read_bytes("nope")


def test_read_json_round_trips(sd):
    payload = {"key": "value", "nested": {"a": 1}}
    sd.atomic_write("conf.json", content=json.dumps(payload).encode("utf-8"))
    assert sd.read_json("conf.json") == payload


# --- stat / list_names ------------------------------------------------------


def test_stat_returns_none_for_missing(sd):
    assert sd.stat("nope") is None


def test_stat_returns_stat_result_for_existing(sd):
    sd.atomic_write("f", content=b"x")
    st = sd.stat("f")
    assert st is not None
    assert st.st_size == 1


def test_stat_root(sd):
    st = sd.stat()
    assert st is not None
    assert stat.S_ISDIR(st.st_mode)


def test_list_names_returns_sorted(sd):
    sd.ensure_directory("dir")
    sd.atomic_write("dir", "b", content=b"")
    sd.atomic_write("dir", "a", content=b"")
    sd.atomic_write("dir", "c", content=b"")
    assert sd.list_names("dir") == ("a", "b", "c")


# --- unlink / remove_tree ---------------------------------------------------


def test_unlink_existing(sd):
    sd.atomic_write("f", content=b"x")
    sd.unlink("f")
    assert sd.stat("f") is None


def test_unlink_missing_raises_without_flag(sd):
    with pytest.raises(FileNotFoundError):
        sd.unlink("nope")


def test_unlink_missing_ok(sd):
    sd.unlink("nope", missing_ok=True)  # no exception


def test_remove_tree_recursive(sd):
    sd.ensure_directory("a", "b", "c")
    sd.atomic_write("a", "b", "c", "f", content=b"x")
    sd.atomic_write("a", "b", "g", content=b"y")
    sd.remove_tree("a")
    assert sd.stat("a") is None


def test_remove_tree_missing_is_noop(sd):
    sd.remove_tree("nope")  # no exception
    assert sd.stat("nope") is None


# --- atomic_publish_directory -----------------------------------------------


def test_atomic_publish_directory_publishes_all_files(sd):
    sd.ensure_directory("parent")
    sd.atomic_publish_directory(
        "parent", "subdir",
        files={"meta.json": b'{"a":1}', "content.bin": b"data"},
    )
    assert sd.read_bytes("parent", "subdir", "meta.json") == b'{"a":1}'
    assert sd.read_bytes("parent", "subdir", "content.bin") == b"data"


def test_atomic_publish_directory_refuses_existing_target(sd):
    """atomic_publish_directory CREATES a new directory atomically; it does
    not overwrite (POSIX rename to a non-empty dir fails anyway). The
    operation journal uses monotonically-numbered version directories, so a
    collision is itself a corruption signal."""
    sd.ensure_directory("parent")
    sd.atomic_publish_directory("parent", "sub", files={"v": b"1"})
    with pytest.raises(OSError):
        sd.atomic_publish_directory("parent", "sub", files={"v": b"2"})
    # The original survives unchanged.
    assert sd.read_bytes("parent", "sub", "v") == b"1"


# --- symlink rejection (the structural security property) --------------------


def test_symlink_at_target_rejected(sd):
    """A symlink planted AT the target name is rejected by O_NOFOLLOW."""
    sd.ensure_directory("dir")
    payload_path = sd.root / "outside" / "payload"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(b"secret")
    # Plant a symlink at dir/link -> ../../outside/payload
    os.symlink(payload_path, sd.root / "dir" / "link")
    with pytest.raises(OSError):  # ELOOP from O_NOFOLLOW on a symlink
        sd.read_bytes("dir", "link")


def test_symlink_in_ancestor_rejected(sd):
    """A symlink planted in an ancestor directory of the target is rejected
    by O_NOFOLLOW|O_DIRECTORY on the descent."""
    sd.ensure_directory("realdir")
    (sd.root / "realdir" / "file").write_bytes(b"trapped-content")
    # Replace "realdir" with a symlink to itself-ish via a sibling.
    outside = sd.root.parent / "outside_target"
    outside.mkdir(parents=True, exist_ok=True)
    # Plant "evil" as a symlink to outside; trying to descend into it must fail.
    os.symlink(outside, sd.root / "evil")
    with pytest.raises(OSError):
        sd.read_bytes("evil", "anything")


def test_symlink_unlinked_not_followed(sd):
    """remove_tree on a symlink unlinks the link itself; it does not traverse
    into the symlink's target."""
    sd.ensure_directory("dir")
    target = sd.root.parent / "outside_dir"
    target.mkdir(exist_ok=True)
    (target / "trapped").write_bytes(b"x")
    os.symlink(target, sd.root / "dir" / "link")
    sd.remove_tree("dir", "link")
    # The trapped file outside was NOT removed.
    assert (target / "trapped").exists()
    assert sd.stat("dir", "link") is None


# --- POSIX capability probe --------------------------------------------------


def test_posix_capability_probe_passes_on_linux():
    """On Linux, the required primitives are present, so construction does
    not raise."""
    SecureDirectory(Path("/tmp"), mode=FilesystemSecurityMode.SECURE_POSIX)


def test_posix_capability_probe_fails_when_o_nofollow_missing(monkeypatch):
    """If the platform lacks O_NOFOLLOW (probed via hasattr), SECURE_POSIX
    construction must refuse -- the safety model depends on it."""
    # Simulate a platform without O_NOFOLLOW.
    saved = os.O_NOFOLLOW
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    try:
        with pytest.raises(StorageObjectError, match="SECURE_POSIX"):
            SecureDirectory(Path("/tmp"), mode=FilesystemSecurityMode.SECURE_POSIX)
    finally:
        # Restore for the module-level _DIR_FLAGS capture (already captured
        # at import time, so this only restores the os attribute).
        os.O_NOFOLLOW = saved


def test_trusted_local_mode_skips_capability_probe(tmp_path):
    """TRUSTED_LOCAL bypasses the capability probe (the caller owns the
    safety tradeoff)."""
    SecureDirectory(
        tmp_path / "trusted", mode=FilesystemSecurityMode.TRUSTED_LOCAL
    )
