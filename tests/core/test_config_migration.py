# -*- coding: utf-8 -*-
"""Tests for ConfigMigration (v2 §3.3)."""
import configparser
import os

import pytest

from linktools._config_store import ConfigStore
from linktools.core._locks import LockManager
from linktools.core._config_migration import ConfigMigration


@pytest.fixture
def setup(tmp_path):
    store = ConfigStore(tmp_path / "new.json", lock_manager=LockManager(tmp_path / "l"))
    old = tmp_path / "old.cfg"
    return store, old, tmp_path


def _write_old(path, entries):
    parser = configparser.ConfigParser()
    parser.optionxform = str  # preserve case (migration reads with optionxform=str)
    parser["CONTAINER.CACHE"] = entries
    with open(path, "w") as f:
        parser.write(f)


def test_inspect_finds_keys(setup):
    store, old, _ = setup
    _write_old(old, {"HOST": "1.2.3.4", "PORT": "8080"})
    info = ConfigMigration(store).inspect(old)
    assert info["file_exists"] is True
    assert info["count"] == 2
    assert "HOST" in info["keys"]


def test_inspect_missing_file(setup):
    store, old, _ = setup
    info = ConfigMigration(store).inspect(old)
    assert info["file_exists"] is False
    assert info["count"] == 0


def test_migrate_writes_to_config_store(setup):
    store, old, _ = setup
    _write_old(old, {"HOST": "1.2.3.4"})
    report = ConfigMigration(store).migrate(old, key_map={"HOST": "network.host"})
    assert "HOST" in report["migrated"]
    assert store.get("network.host") == "1.2.3.4"


def test_migrate_unmapped_keys_go_to_legacy(setup):
    store, old, _ = setup
    _write_old(old, {"UNKNOWN_KEY": "val"})
    report = ConfigMigration(store).migrate(old)
    assert "UNKNOWN_KEY" in report["legacy"]
    assert store.get("legacy.UNKNOWN_KEY") == "val"


def test_migrate_skips_existing(setup):
    store, old, _ = setup
    store.set("network.host", "newer")
    _write_old(old, {"HOST": "older"})
    report = ConfigMigration(store).migrate(old, key_map={"HOST": "network.host"})
    assert "HOST" in report["skipped"]
    assert store.get("network.host") == "newer"


def test_backup_and_rollback(setup):
    store, old, tmp = setup
    _write_old(old, {"X": "1"})
    mig = ConfigMigration(store)
    bak = mig.backup(old)
    assert os.path.isfile(bak)
    # destroy old, restore from backup
    os.remove(old)
    assert not old.exists()
    mig.rollback(bak, old)
    assert old.exists()


def test_verify(setup):
    store, old, _ = setup
    store.set("a", "1")
    assert ConfigMigration(store).verify() is True
