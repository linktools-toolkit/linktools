# -*- coding: utf-8 -*-
"""Environ auto-migrates the legacy ConfigCacheParser .cfg file on first use.

Regression coverage for two bugs found while auditing cntr's config migration:

1. The only migration entry point was a manual CLI command (`ct env migrate` /
   the now-removed `ct-cntr config migrate`), so a pure-core install (no
   sub-package) had no automatic way to migrate its legacy
   ``<data>/.config/<name>.cfg``. Migration now runs the first time anything
   touches ``Environ.config_store`` -- before any sub-package (e.g. cntr) gets
   a chance to construct its own manager, so this is the single source of
   truth for that file (see BaseEnviron._migrate_legacy_cfg).
2. ConfigMigration's migrated target keys used to be lowercase/dotted
   (e.g. "container.docker_type") which does not match what PersistentSource
   actually reads ("<namespace>.<FIELD_NAME>" verbatim, e.g.
   "container.DOCKER_TYPE") -- migrated values were written but could never be
   read back. Covered here via a real Environ.get_config("DEBUG") round-trip.
"""
import configparser

from linktools.core._environ import BaseEnviron, Environ
from linktools.types import MISSING


def _reset_global_config():
    """Force the class-level cached global_config to re-read LINKTOOLS_* env
    vars on next access (see tests/cntr/_harness.py for the same pattern)."""
    descriptor = BaseEnviron.__dict__.get("global_config")
    if descriptor is not None and hasattr(descriptor, "val"):
        descriptor.val = MISSING


def _make_environ(tmp_path, monkeypatch):
    _reset_global_config()
    storage = str(tmp_path)
    monkeypatch.setenv("LINKTOOLS_PATH", storage)
    monkeypatch.setenv("LINKTOOLS_DATA_PATH", storage + "/data")
    monkeypatch.setenv("LINKTOOLS_TEMP_PATH", storage + "/temp")
    return Environ()


def _write_legacy_cfg(data_path, sections):
    cfg_dir = data_path / ".config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    parser.optionxform = str
    for name, entries in sections.items():
        parser[name] = entries
    with open(cfg_dir / "linktools.cfg", "w") as f:
        parser.write(f)
    return cfg_dir / "linktools.cfg"


def test_auto_migrates_on_first_config_store_access(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    cfg_path = _write_legacy_cfg(environ.get_data_path(), {
        "MAIN.CACHE": {"DEBUG": "true"},
        "CONTAINER.CACHE": {"CONTAINER_TYPE": "podman", "DOCKER_USER": "alice"},
    })

    environ.config_store  # first touch -- no sub-package, no manual command

    assert not cfg_path.exists()  # migrated + removed
    assert environ.config_store.get("main.DEBUG") == "true"
    assert environ.config_store.get("container.CONTAINER_TYPE") == "podman"
    assert environ.config_store.get("container.DOCKER_USER") == "alice"


def test_migrated_debug_resolves_through_get_config(tmp_path, monkeypatch):
    """The exact bug: a migrated key must be readable by the field that uses
    it, not just present in the raw store."""
    environ = _make_environ(tmp_path, monkeypatch)
    _write_legacy_cfg(environ.get_data_path(), {"MAIN.CACHE": {"DEBUG": "true"}})

    assert environ.get_config("DEBUG", bool) is True


def test_idempotent_second_environ_is_noop(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    _write_legacy_cfg(environ.get_data_path(), {"MAIN.CACHE": {"DEBUG": "true"}})

    environ.config_store  # migrates
    assert environ.config_store.get("main.DEBUG") == "true"

    _reset_global_config()
    second = Environ()
    second.config_store  # no legacy file left, no crash
    assert second.config_store.get("main.DEBUG") == "true"


def test_does_not_overwrite_newer_config_store_data(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    _write_legacy_cfg(environ.get_data_path(), {"MAIN.CACHE": {"DEBUG": "true"}})

    # Pre-seed the store file directly (before config_store is first touched
    # by this Environ instance) to simulate "user already has newer data".
    from linktools.core._config import ConfigStore
    environ.paths.ensure_config()
    ConfigStore(environ.paths.config / "settings.json", lock_manager=environ.locks).set(
        "main.DEBUG", "false")

    assert environ.config_store.get("main.DEBUG") == "false"  # not overwritten


def test_no_legacy_cfg_is_noop(tmp_path, monkeypatch):
    environ = _make_environ(tmp_path, monkeypatch)
    environ.config_store  # must not raise when there is nothing to migrate
    assert "main.DEBUG" not in environ.config_store
