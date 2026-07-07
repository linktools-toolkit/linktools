# -*- coding: utf-8 -*-
"""Tests for the cntr legacy-settings migrator (spec §21.2).

Verifies user data (INSTALLED_CONTAINERS / INSTALLED_REPOS) is preserved across
the FileCache shelve -> ConfigStore migration, that newer ConfigStore data is
never overwritten, and that the migration is idempotent.
"""
import json
import os

import pytest

from linktools.cache import FileCache  # legacy, used to set up the source data
from linktools._config_store import ConfigStore
from linktools.core._locks import LockManager
from linktools.cntr._migrate import migrate_legacy_container_settings


class _FakeLogger(object):
    def __getattr__(self, name):
        # treat any level as a no-op sink for the test
        return lambda *_args, **_kwargs: None


@pytest.fixture
def stores(tmp_path):
    data_path = tmp_path / "data"
    setting_path = tmp_path / "data" / "container" / "setting"
    setting_path.mkdir(parents=True)
    lock_dir = tmp_path / "locks"
    config_store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(lock_dir))
    return config_store, data_path, setting_path


def _legacy_shelve(setting_path, **entries):
    cache = FileCache(str(setting_path / "manager"))
    with cache.session() as data:
        for k, v in entries.items():
            data.set(k, v)


def test_migrates_shelve_to_config_store(stores):
    config_store, data_path, setting_path = stores
    _legacy_shelve(setting_path, INSTALLED_CONTAINERS=["nginx", "authelia"],
                   INSTALLED_REPOS={"https://x": {"type": "git"}})
    migrated = migrate_legacy_container_settings(
        config_store, data_path, setting_path, _FakeLogger())
    assert migrated == {"INSTALLED_CONTAINERS", "INSTALLED_REPOS"}
    assert config_store.get("INSTALLED_CONTAINERS") == ["nginx", "authelia"]
    assert config_store.get("INSTALLED_REPOS") == {"https://x": {"type": "git"}}
    # legacy store removed
    assert not (setting_path / "manager").exists()


def test_does_not_overwrite_newer_config_store_data(stores):
    config_store, data_path, setting_path = stores
    config_store.set("INSTALLED_CONTAINERS", ["newer"])  # user already has data
    _legacy_shelve(setting_path, INSTALLED_CONTAINERS=["old"])
    migrate_legacy_container_settings(config_store, data_path, setting_path, _FakeLogger())
    assert config_store.get("INSTALLED_CONTAINERS") == ["newer"]  # not overwritten


def test_idempotent_second_run_is_noop(stores):
    config_store, data_path, setting_path = stores
    _legacy_shelve(setting_path, INSTALLED_CONTAINERS=["nginx"])
    first = migrate_legacy_container_settings(config_store, data_path, setting_path, _FakeLogger())
    second = migrate_legacy_container_settings(config_store, data_path, setting_path, _FakeLogger())
    assert first == {"INSTALLED_CONTAINERS"}
    assert second == set()  # legacy store already gone
    assert config_store.get("INSTALLED_CONTAINERS") == ["nginx"]


def test_migrates_containers_yml_and_repo_json(stores, tmp_path):
    config_store, data_path, setting_path = stores
    # Build the very-old/old source files.
    config_dir = data_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "containers.yml").write_text(json.dumps(["legacy-nginx"]))
    repo_dir = data_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "repo.json").write_text(json.dumps({"https://r": {"type": "git"}}))
    (repo_dir / "repo.lock").write_text("x")

    migrated = migrate_legacy_container_settings(
        config_store, data_path, setting_path, _FakeLogger())
    assert config_store.get("INSTALLED_CONTAINERS") == ["legacy-nginx"]
    assert config_store.get("INSTALLED_REPOS") == {"https://r": {"type": "git"}}
    assert migrated == {"INSTALLED_CONTAINERS", "INSTALLED_REPOS"}
    # legacy artefacts removed
    assert not config_dir.exists()
    assert not (repo_dir / "repo.json").exists()


def test_no_legacy_is_noop(stores):
    config_store, data_path, setting_path = stores
    migrated = migrate_legacy_container_settings(
        config_store, data_path, setting_path, _FakeLogger())
    assert migrated == set()
    assert config_store.keys() == []
