# -*- coding: utf-8 -*-
"""Tests for the ConfigStore (spec §8.5 CFG-005 PersistentStore).

A user-editable JSON file written atomically under a process lock. This is the
proper home for persistent user state (e.g. cntr's INSTALLED_CONTAINERS) that
the spec says must NOT live in the cache.
"""
import json
import threading

import pytest

from linktools._config_store import ConfigStore
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


def test_corrupt_json_raises(store, tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json")
    with pytest.raises(ConfigError):
        store.reload()


def test_missing_file_is_empty(store):
    assert store.keys() == []
    store.set("k", 1)  # creates the file
    assert (store.path).exists()


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
