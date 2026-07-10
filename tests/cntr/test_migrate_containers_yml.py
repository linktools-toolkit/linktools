#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Migrating legacy containers.yml must not take unrelated files with it.

Regression: the migration removed the whole data/config directory via
also_remove=(config_dir,) -> ConfigMigration._remove -> remove_file() ->
shutil.rmtree() on a directory. Any unrelated file a user happened to keep
alongside containers.yml (a hand-written custom.json, a backup) was deleted
along with it. Only the migrated file should be removed; the directory itself
is only cleaned up if migration happens to leave it empty.
"""
import json

from linktools.core._environ import Environ
from linktools.cntr.manager import ContainerManager

from _harness import install_deterministic_interaction, _reset_global_config


def _make_environ(tmp_path, monkeypatch):
    install_deterministic_interaction()
    _reset_global_config()
    storage = str(tmp_path)
    monkeypatch.setenv("LINKTOOLS_PATH", storage)
    monkeypatch.setenv("LINKTOOLS_DATA_PATH", storage + "/data")
    monkeypatch.setenv("LINKTOOLS_TEMP_PATH", storage + "/temp")
    return Environ()


def test_migration_preserves_unrelated_config_files(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    config_dir = environ.get_data_path("container") / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "containers.yml").write_text(json.dumps(["nginx"]))
    unrelated = config_dir / "custom.json"
    unrelated.write_text(json.dumps({"keep": "me"}))

    ContainerManager(environ, name="aio")

    assert not (config_dir / "containers.yml").exists()  # migrated + removed
    assert unrelated.exists()  # untouched
    assert json.loads(unrelated.read_text()) == {"keep": "me"}
    assert environ.config_store.get("INSTALLED_CONTAINERS") == ["nginx"]


def test_migration_removes_config_dir_only_when_left_empty(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    config_dir = environ.get_data_path("container") / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "containers.yml").write_text(json.dumps(["nginx"]))

    ContainerManager(environ, name="aio")

    assert not config_dir.exists()  # nothing left behind, safe to drop the dir
