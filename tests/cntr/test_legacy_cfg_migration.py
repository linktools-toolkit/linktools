#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Constructing a real ContainerManager auto-migrates the legacy .cfg file.

Integration coverage for the end-to-end cntr user path (no manual command).
The actual migration of ``<data>/.config/<name>.cfg`` happens one layer down,
in ``Environ.config_store`` (see tests/core/test_environ_legacy_cfg_migration.py
for that unit coverage) -- ``ContainerManager.__init__`` forces
``env_config = environ.wrap_config(...)``, which constructs a
``PersistentSource(self.config_store, ...)``, which is what first touches
``config_store`` and triggers the migration. So by the time cntr's own
``_migrated`` (in _migrate.py, covering containers.yml/repo.json/FileCache
shelve) runs, the legacy .cfg is already gone.

This file guards against three real bugs found in the field:

1. ``ct-cntr config migrate`` pointed ``old_path`` at the *new* ConfigStore's own
   settings.json instead of the legacy ``<data>/.config/<name>.cfg`` file, so
   running it crashed with configparser.MissingSectionHeaderError. The command
   is now removed; migration runs automatically.
2. ConfigMigration.DEFAULT_KEY_MAP's target keys (e.g. "container.docker_type")
   did not match the verbatim "<namespace>.<FIELD_NAME>" format PersistentSource
   actually reads (e.g. "container.DOCKER_TYPE") -- migrated values were written
   but could never be read back. Fixed in _config.py; covered end-to-end here via
   a real ContainerManager resolving DOCKER_TYPE/DOCKER_USER post-migration.
3. Migration used to only be triggered from ContainerManager._load_setting,
   which "config set/list/edit" never call (they reach env_config directly) --
   those commands could run against stale pre-migration data. Now it happens as
   soon as anything (any sub-package, not just cntr) touches config_store.
"""
import configparser

from linktools.core._environ import Environ
from linktools.cntr.manager import ContainerManager

from _harness import install_deterministic_interaction, _reset_global_config


def _write_legacy_cfg(data_path, **entries):
    cfg_dir = data_path / ".config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser["CONTAINER.CACHE"] = entries
    with open(cfg_dir / "linktools.cfg", "w") as f:
        parser.write(f)
    return cfg_dir / "linktools.cfg"


def _make_environ(tmp_path, monkeypatch):
    install_deterministic_interaction()
    _reset_global_config()
    storage = str(tmp_path)
    monkeypatch.setenv("LINKTOOLS_PATH", storage)
    monkeypatch.setenv("LINKTOOLS_DATA_PATH", storage + "/data")
    monkeypatch.setenv("LINKTOOLS_TEMP_PATH", storage + "/temp")
    return Environ()


def test_auto_migrates_on_construction(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    cfg_path = _write_legacy_cfg(environ.get_data_path(), CONTAINER_TYPE="podman", DOCKER_USER="alice")

    ContainerManager(environ, name="aio")  # migration runs in __init__, no setting access needed

    assert not cfg_path.exists()  # migrated + removed
    assert environ.config_store.get("container.CONTAINER_TYPE") == "podman"
    assert environ.config_store.get("container.DOCKER_USER") == "alice"


def test_migrated_values_resolve_through_real_config_fields(tmp_path, monkeypatch):
    """The exact bug: migrated keys must be readable by the fields that use them,
    not just present in the raw store."""
    environ = _make_environ(tmp_path, monkeypatch)
    _write_legacy_cfg(environ.get_data_path(), CONTAINER_TYPE="podman", DOCKER_USER="alice")

    import linktools.rich as rich

    def fail(*_a, **_k):
        raise AssertionError("prompt/choose must not be reached; value should come from migration")

    monkeypatch.setattr(rich, "prompt", fail)
    monkeypatch.setattr(rich, "choose", fail)

    manager = ContainerManager(environ, name="aio")
    assert manager.env_config.get("DOCKER_TYPE", type=str) == "podman"
    assert manager.env_config.get("DOCKER_USER", type=str) == "alice"


def test_idempotent_second_run_is_noop(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    _write_legacy_cfg(environ.get_data_path(), CONTAINER_TYPE="docker")

    ContainerManager(environ, name="aio")
    assert environ.config_store.get("container.CONTAINER_TYPE") == "docker"

    # Second manager instance, same environ/store: no legacy file left, no crash.
    ContainerManager(environ, name="aio")
    assert environ.config_store.get("container.CONTAINER_TYPE") == "docker"


def test_does_not_overwrite_newer_config_store_data(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    _write_legacy_cfg(environ.get_data_path(), CONTAINER_TYPE="podman")
    environ.config_store.set("container.CONTAINER_TYPE", "docker")  # newer value already present

    ContainerManager(environ, name="aio")
    assert environ.config_store.get("container.CONTAINER_TYPE") == "docker"  # not overwritten


def test_no_legacy_cfg_is_noop(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    ContainerManager(environ, name="aio")  # must not raise when there is nothing to migrate
    assert "container.CONTAINER_TYPE" not in environ.config_store


def test_migrated_docker_app_path_is_not_reprompted(tmp_path, monkeypatch):
    """Regression: DOCKER_APP_PATH (and friends) used to lack cached=True on
    their PromptProvider, so every command re-prompted for a value that was
    already migrated/configured -- manager.configs's ConfigField.chain(...)
    fields must persist the migrated value and reuse it on the next manager
    instance (e.g. across separate `ct-cntr config list` invocations)."""
    environ = _make_environ(tmp_path, monkeypatch)
    _write_legacy_cfg(environ.get_data_path(), DOCKER_APP_PATH="/srv/legacy-app-path")

    ContainerManager(environ, name="aio")  # migration runs in __init__

    import linktools.rich as rich

    def fail(*_a, **_k):
        raise AssertionError("prompt must not be reached; value should come from the persisted store")

    monkeypatch.setattr(rich, "prompt", fail)

    manager = ContainerManager(environ, name="aio")  # a fresh manager instance
    assert manager.env_config.get("DOCKER_APP_PATH") == "/srv/legacy-app-path"
