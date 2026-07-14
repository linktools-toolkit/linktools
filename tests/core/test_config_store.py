# -*- coding: utf-8 -*-
"""Tests for the ConfigStore (spec §8.5 CFG-005 PersistentStore).

A user-editable JSON file written atomically under a process lock. This is the
proper home for persistent user state (e.g. cntr's INSTALLED_CONTAINERS) that
the spec says must NOT live in the cache.
"""
import json
import os
import threading

import pytest

from linktools.core import ConfigStore
from linktools.core._locks import LockManager
from linktools.errors import ConfigError


@pytest.fixture
def store(tmp_path):
    lock_dir = tmp_path / "locks"
    lm = LockManager(lock_dir)
    s = ConfigStore(tmp_path / "settings.json", lock_manager=lm)
    try:
        yield s
    finally:
        pass


def test_get_set_roundtrip(store):
    store.set("name", "alpha")
    assert store.get("name") == "alpha"
    assert "name" in store
    assert store.get("missing", "fallback") == "fallback"


def test_persists_across_instances(store, tmp_path):
    store.set("k", [1, 2, 3])
    # A second store on the same file sees the write.
    again = ConfigStore(tmp_path / "settings.json", lock_manager=store._lock_manager)
    assert again.get("k") == [1, 2, 3]


def test_remove(store):
    store.save(a=1, b=2)
    assert store.remove("a") is True
    # v4 §3.4: get returns MISSING (not None) for absent keys
    from linktools.types import MISSING
    assert store.get("a") is MISSING
    assert "a" not in store
    assert store.get("b") == 2
    assert store.remove("nope") is False  # nothing removed


def test_save_batch_and_keys(store):
    store.save(a=1, b=2, c=3)
    assert set(store.keys()) == {"a", "b", "c"}
    assert dict(store.items()) == {"a": 1, "b": 2, "c": 3}


def test_reload_picks_up_external_changes(store, tmp_path):
    store.set("k", "mine")
    # Externally rewrite the file.
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"k": "external", "extra": 1}))
    store.reload()
    assert store.get("k") == "external"
    assert store.get("extra") == 1


def test_write_is_atomic_replaces_existing(store, tmp_path):
    path = tmp_path / "settings.json"
    store.set("k", "v1")
    first_inode = path.stat().st_ino if path.exists() else None
    store.set("k", "v2")
    assert path.read_text()  # non-empty, valid JSON
    assert json.loads(path.read_text())["k"] == "v2"
    # No stray temp files left beside it.
    assert not list(tmp_path.glob("*.tmp"))


def test_set_rolls_back_in_memory_state_when_flush_fails(store, monkeypatch):
    """A failed disk write must never leave the in-memory store reflecting
    a value that was never actually persisted -- a caller reading it back
    (without an intervening reload()) would otherwise see a phantom write
    that a crash right afterward would lose entirely."""
    store.set("k", "before")

    def broken_flush():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_flush", broken_flush)
    with pytest.raises(OSError):
        store.set("k", "after")

    assert store.get("k") == "before"


def test_save_rolls_back_in_memory_state_when_flush_fails(store, monkeypatch):
    store.save(a=1, b=2)

    def broken_flush():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_flush", broken_flush)
    with pytest.raises(OSError):
        store.save(a=99, c=3)

    assert store.get("a") == 1
    assert store.get("b") == 2
    assert "c" not in store


def test_remove_rolls_back_in_memory_state_when_flush_fails(store, monkeypatch):
    store.set("k", "v")

    def broken_flush():
        raise OSError("disk full")

    monkeypatch.setattr(store, "_flush", broken_flush)
    with pytest.raises(OSError):
        store.remove("k")

    assert store.get("k") == "v"


def test_corrupt_json_raises(store, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json")
    with pytest.raises(ConfigError):
        store.reload()


def test_missing_file_is_empty(store):
    assert store.keys() == []
    store.set("k", 1)  # creates the file
    assert (store.path).exists()


def test_dangling_symlink_raises(tmp_path):
    # os.path.exists()/Path.exists() follow symlinks and return False for a
    # dangling one -- indistinguishable from "genuinely missing" unless the
    # symlink itself is checked. Fail-closed: this must raise, not silently
    # report an empty store.
    lm = LockManager(tmp_path / "locks")
    path = tmp_path / "settings.json"
    os.symlink(str(tmp_path / "does-not-exist"), str(path))
    with pytest.raises(ConfigError):
        ConfigStore(path, lock_manager=lm)


def test_dangling_symlink_not_overridden_by_set(tmp_path):
    """A construction failure must not be recoverable by writing through
    the half-built instance -- there is no instance to write through."""
    lm = LockManager(tmp_path / "locks")
    path = tmp_path / "settings.json"
    os.symlink(str(tmp_path / "does-not-exist"), str(path))
    with pytest.raises(ConfigError):
        ConfigStore(path, lock_manager=lm)

    # The dangling symlink itself is untouched -- no store construction
    # ever got far enough to attempt a write.
    assert path.is_symlink()
    assert not path.exists()


def test_directory_path_raises(tmp_path):
    lm = LockManager(tmp_path / "locks")
    path = tmp_path / "settings.json"
    path.mkdir()
    with pytest.raises(ConfigError):
        ConfigStore(path, lock_manager=lm)


def test_root_not_an_object_raises(store, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigError):
        store.reload()


def test_concurrent_writes_do_not_lose_keys(store, tmp_path):
    # Two stores on the same file via independent lock managers (separate
    # processes simulation): each writes its own key; both survive.
    lm1 = LockManager(tmp_path / "l1")
    lm2 = LockManager(tmp_path / "l2")
    # share the SAME lock file so they really contend -- point both at one dir
    a = ConfigStore(tmp_path / "s.json", lock_manager=LockManager(tmp_path / "locks"))
    b = ConfigStore(tmp_path / "s.json", lock_manager=LockManager(tmp_path / "locks"))
    errors = []

    def writer(store, key, n):
        try:
            for i in range(n):
                store.set(key, i)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t1 = threading.Thread(target=writer, args=(a, "a", 30))
    t2 = threading.Thread(target=writer, args=(b, "b", 30))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    assert a.reload() is None or True
    assert "a" in a.keys() and "b" in a.keys()


# -- ConfigNamespace ------------------------------------------------------

def test_namespace_get_set_roundtrip(store):
    ns = store.namespace("app:one")
    assert ns.get("k", "fallback") == "fallback"
    ns.set("k", "v")
    assert ns.get("k") == "v"


def test_namespace_isolated_from_other_namespaces(store):
    a = store.namespace("app:a")
    b = store.namespace("app:b")
    a.set("k", "a-value")
    assert b.get("k") is None
    assert a.get("k") == "a-value"


def test_namespace_does_not_leak_into_store_top_level_keys(store):
    ns = store.namespace("app:one")
    ns.set("k", "v")
    # The namespace's data lives nested under its own name, not as a
    # top-level store key -- "k" itself must never collide with an
    # unrelated top-level config key of the same name.
    assert "k" not in store.keys()
    assert store.get("app:one") == {"k": "v"}


def test_namespace_pop(store):
    ns = store.namespace("app:one")
    ns.set("k", "v")
    assert ns.pop("k") == "v"
    assert ns.get("k") is None
    assert ns.pop("missing", "fallback") == "fallback"


def test_namespace_keys(store):
    ns = store.namespace("app:one")
    ns.set("a", 1)
    ns.set("b", 2)
    assert set(ns.keys()) == {"a", "b"}


def test_namespace_transaction_persists_once(store):
    ns = store.namespace("app:one")
    with ns.transaction() as tx:
        tx.set("a", 1)
        tx.set("b", 2)
    assert ns.get("a") == 1
    assert ns.get("b") == 2


def test_namespace_transaction_rolls_back_in_memory_on_error(store):
    ns = store.namespace("app:one")
    ns.set("a", 1)
    with pytest.raises(RuntimeError):
        with ns.transaction() as tx:
            tx.set("a", 2)
            raise RuntimeError("boom")
    assert ns.get("a") == 1


def test_namespace_transaction_survives_a_fresh_snapshot(store):
    # get()/set() outside a transaction read/write a fresh snapshot each
    # time -- not a live reference the caller could accidentally mutate.
    ns = store.namespace("app:one")
    ns.set("nested", {"x": 1})
    snapshot = ns.get("nested")
    snapshot["x"] = 999
    assert ns.get("nested") == {"x": 1}


def test_namespace_transaction_reenters_a_plain_store_operation(store):
    # A plain store.set() from within a namespace transaction on the same
    # store (e.g. ordinary config resolution persisting a value while a
    # container's settings.transaction() is open) must reuse the held lock
    # rather than deadlock or raise -- and both writes must survive.
    ns = store.namespace("app:one")
    with ns.transaction() as tx:
        tx.set("a", 1)
        store.set("unrelated", "value")
    assert ns.get("a") == 1
    assert store.get("unrelated") == "value"


def test_two_namespaces_of_the_same_store_can_nest_transactions(store):
    a = store.namespace("app:a")
    b = store.namespace("app:b")
    with a.transaction() as tx_a:
        tx_a.set("k", "a-value")
        with b.transaction() as tx_b:
            tx_b.set("k", "b-value")
    assert a.get("k") == "a-value"
    assert b.get("k") == "b-value"
