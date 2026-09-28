# -*- coding: utf-8 -*-
"""Environ._config_store must not silently orphan a <=0.9.0 install's config.

Regression: in v0.9.0, every namespace's persisted config (any field, not
just specific known keys) lived as a section (<NAMESPACE>.CACHE) in one
INI file (<data_path>/.config/<name>.cfg), written by the old
configparser-based ConfigCache/PersistentSource. Today's PersistentSource
persists into a single JSON store (settings.json) instead. Without a
migration, every previously-persisted value (HOST, DOCKER_USER, any
cntr container's own settings, ...) becomes invisible the moment an
upgrading install builds its Config -- silently falling back to defaults/
re-prompting, as if the user had never configured anything.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path / "home")
    monkeypatch.setenv("LINKTOOLS_PATH", str(tmp_path))
    monkeypatch.delenv("LINKTOOLS_DATA_PATH", raising=False)
    monkeypatch.delenv("LINKTOOLS_TEMP_PATH", raising=False)
    yield tmp_path


def _write_legacy_cfg(environ, text):
    config_dir = environ.data_path / ".config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / f"{environ.name}.cfg").write_text(text)


def _migration_backups(environ, name):
    """Every migrations/<id>/<name> backup path under paths.config."""
    migrations_dir = environ.paths.config / "migrations"
    if not migrations_dir.is_dir():
        return []
    return [p for p in migrations_dir.glob(f"*/{name}") if p.exists()]


def test_every_field_in_every_section_is_migrated():
    from linktools.core._environ import Environ

    environ = Environ()
    _write_legacy_cfg(environ, """
[MAIN.CACHE]
HOST = 192.168.1.1
DEBUG = true

[CONTAINER.CACHE]
DOCKER_USER = myuser
NGINX_HTTPS_PORT = 8443
""")

    store = environ._config_store
    assert store.get("main.HOST") == "192.168.1.1"
    assert store.get("main.DEBUG") == "true"
    assert store.get("container.DOCKER_USER") == "myuser"
    assert store.get("container.NGINX_HTTPS_PORT") == "8443"


def test_old_cfg_file_backed_up_after_migration():
    from linktools.core._environ import Environ

    environ = Environ()
    _write_legacy_cfg(environ, "[MAIN.CACHE]\nHOST = 1.2.3.4\n")
    old_path = environ.data_path / ".config" / f"{environ.name}.cfg"

    environ._config_store
    # Moved into paths.config/migrations/<id>/, not deleted outright.
    assert not old_path.exists()
    backups = _migration_backups(environ, f"{environ.name}.cfg")
    assert len(backups) == 1
    assert "HOST = 1.2.3.4" in backups[0].read_text()
    assert (backups[0].parent / "report.json").exists()


def test_migration_never_overwrites_existing_store_data(monkeypatch, tmp_path):
    from linktools.core._environ import Environ

    first = Environ()
    first._config_store.set("main.HOST", "already-set")
    _write_legacy_cfg(first, "[MAIN.CACHE]\nHOST = from-legacy-file\n")

    second = Environ()
    assert second._config_store.get("main.HOST") == "already-set"


def test_fresh_install_never_creates_a_stray_config_dir():
    from linktools.core._environ import Environ

    environ = Environ()
    environ._config_store
    assert not (environ.data_path / ".config").exists()
    assert not (environ.paths.config / "migrations").exists()


def test_unrelated_sections_and_default_section_are_ignored():
    from linktools.core._environ import Environ

    environ = Environ()
    _write_legacy_cfg(environ, """
[ENV]
SOME_GLOBAL = ignored

[MAIN.CACHE]
HOST = 1.2.3.4
""")

    store = environ._config_store
    assert store.get("main.HOST") == "1.2.3.4"
    assert "ENV.SOME_GLOBAL" not in store
    assert "SOME_GLOBAL" not in store


def test_corrupt_legacy_cfg_does_not_block_environ_construction():
    from linktools.core._environ import Environ

    environ = Environ()
    _write_legacy_cfg(environ, "not a valid ini file [[[")

    # Must not raise -- best-effort migration, same fail-soft contract as
    # the cntr-side v0.9.0/pre-v0.9.0 migrations.
    store = environ._config_store
    assert store is not None
