# -*- coding: utf-8 -*-
"""Tests for LockManager (§7.11) and atomic file utils (§17.1)."""
import os
import threading
from pathlib import Path

import pytest

from linktools.core._locks import LockManager, _sanitize
from linktools import utils


# --------------------------------------------------------------------------- #
# §7.11 LockManager
# --------------------------------------------------------------------------- #

@pytest.fixture
def manager(tmp_path):
    return LockManager(tmp_path / "locks")


def test_process_lock_is_a_context_manager(manager):
    with manager.process_lock("download:abc") as lock:
        assert lock.is_locked
    assert not lock.is_locked


def test_process_lock_sanitises_name_and_stays_in_dir(manager):
    # A hostile name must not traverse out of the lock directory.
    from pathlib import Path

    lock = manager.process_lock("../../etc/evil")
    assert manager.lock_dir.resolve() in Path(lock.lock_file).resolve().parents
    with lock:
        pass


def test_two_holders_of_same_process_lock_are_mutually_exclusive(manager):
    import filelock

    lock_a = manager.process_lock("job")
    lock_b = manager.process_lock("job")  # same name -> same lock file
    lock_a.acquire()
    try:
        # Same underlying file -> second holder cannot acquire without waiting;
        # filelock raises Timeout rather than returning False.
        with pytest.raises(filelock.Timeout):
            lock_b.acquire(timeout=0.1)
    finally:
        lock_a.release()


def test_different_process_locks_are_independent(manager):
    a = manager.process_lock("alpha")
    b = manager.process_lock("beta")
    a.acquire()
    try:
        assert b.acquire(timeout=0.1)  # truthy AcquireReturnProxy
    finally:
        b.release()
        a.release()


def test_file_lock_uses_lock_dir_not_business_file(manager, tmp_path):
    # §6.2: file_lock must NOT lock the business file itself; the lock lives
    # under lock_dir keyed by the sha256 of the absolute target path.
    target = tmp_path / "f.bin"
    lock = manager.file_lock(target)
    # lock_file is under lock_dir, NOT on the target
    assert manager.lock_dir in Path(lock.lock_file).parents
    assert str(target) not in str(lock.lock_file)
    # no .lock sidecar pollutes the target's directory
    assert not list(tmp_path.glob("*.lock"))


def test_sanitize_drops_separators():
    assert "/" not in _sanitize("a/b\\c")
    assert "\\" not in _sanitize("a/b\\c")
    assert _sanitize("   ") == "lock"


# --------------------------------------------------------------------------- #
# §17.1 atomic_write / atomic_replace
# --------------------------------------------------------------------------- #

def test_atomic_write_creates_file(tmp_path):
    target = tmp_path / "out" / "cfg.json"
    utils.atomic_write(target, '{"k": 1}')
    assert target.read_text() == '{"k": 1}'


def test_atomic_write_replaces_existing(tmp_path):
    target = tmp_path / "f"
    utils.atomic_write(target, "old")
    utils.atomic_write(target, "new")
    assert target.read_text() == "new"


def test_atomic_write_bytes(tmp_path):
    target = tmp_path / "b"
    utils.atomic_write(target, b"\x00\x01\x02")
    assert target.read_bytes() == b"\x00\x01\x02"


def test_atomic_write_leaves_no_tmp_on_failure(tmp_path):
    target = tmp_path / "f"
    # "☃" cannot encode as ascii -> UnicodeEncodeError AFTER mkstemp has
    # already created the temp file, so the cleanup branch must remove it.
    with pytest.raises(Exception):
        utils.atomic_write(target, "☃", encoding="ascii")
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_replace(tmp_path):
    src = tmp_path / "s"
    dst = tmp_path / "d"
    src.write_text("hello")
    utils.atomic_replace(src, dst)
    assert dst.read_text() == "hello"
    assert not src.exists()
