# -*- coding: utf-8 -*-
"""ContainerManager.settings must not silently orphan an old install --
v0.9.0's FileCache format, or pre-v0.9.0's raw JSON files.

Regression: in v0.9.0, INSTALLED_CONTAINERS/INSTALLED_REPOS lived in a
FileCache (shelve-backed) at <data_path>/setting/manager. Before that (pre-
v0.9.0), they lived in raw files (<data_path>/config/containers.yml,
<data_path>/repo/repo.json) -- v0.9.0's own manager.py migrated those into
FileCache the first time it ran. Both now live in manager.settings (a
dedicated cntr.json). Without migrating each format directly, an install
jumping straight from pre-v0.9.0 to today (skipping v0.9.0 entirely, so
v0.9.0's own migration never runs) or upgrading from v0.9.0 itself would see
prepare_installed_containers()/repos.get_all() silently report nothing
installed, even though the user never removed anything.
"""
import pytest


def _migration_backups(manager, name):
    """Every migrations/<id>/<name> backup path under paths.config."""
    migrations_dir = manager.environ.paths.config / "migrations"
    if not migrations_dir.is_dir():
        return []
    return [p for p in migrations_dir.glob(f"*/{name}") if p.exists()]


def _make_manager(tmp_path):
    import os
    os.environ["LINKTOOLS_PATH"] = str(tmp_path)
    os.environ["LINKTOOLS_DATA_PATH"] = str(tmp_path / "data")
    os.environ["LINKTOOLS_TEMP_PATH"] = str(tmp_path / "temp")

    import _harness
    _harness.install_deterministic_interaction()
    _harness._reset_global_config()

    from linktools.core._environ import Environ
    from linktools.cntr.manager import ContainerManager

    return ContainerManager(Environ(), name="aio")


@pytest.fixture
def manager_env(tmp_path, monkeypatch):
    for key in ("LINKTOOLS_PATH", "LINKTOOLS_DATA_PATH", "LINKTOOLS_TEMP_PATH"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


def _write_v090_filecache(manager, containers=None, repos=None):
    # Replicates v0.9.0's now-removed linktools.cache.FileCache on-disk
    # format directly (shelve DB at <setting_path>/manager/data/data, each
    # record {"data": value, "ttl": None, "ts": ...}) -- the class itself is
    # gone (CacheStore replaced it everywhere it was still used), but the
    # migration code must still read a real v0.9.0 install's raw file.
    import shelve
    import time

    shelve_path = manager.data_path / "setting" / "manager" / "data" / "data"
    shelve_path.parent.mkdir(parents=True, exist_ok=True)
    with shelve.open(str(shelve_path)) as db:
        if containers is not None:
            db["INSTALLED_CONTAINERS"] = {"data": containers, "ttl": None, "ts": int(time.time())}
        if repos is not None:
            db["INSTALLED_REPOS"] = {"data": repos, "ttl": None, "ts": int(time.time())}


def test_v090_filecache_data_is_migrated(manager_env):
    manager = _make_manager(manager_env)
    _write_v090_filecache(
        manager,
        containers=["nginx"],
        repos={"/some/repo": {"type": "local"}},
    )

    fresh = _make_manager(manager_env)
    assert fresh.settings.get("INSTALLED_CONTAINERS") == ["nginx"]
    assert fresh.settings.get("INSTALLED_REPOS") == {"/some/repo": {"type": "local"}}
    # The old v0.9.0 setting/ directory is moved into paths.config/migrations/,
    # not left duplicated alongside the new store, and not deleted outright.
    assert not (fresh.data_path / "setting").exists()
    assert _migration_backups(fresh, "setting")


def test_migration_never_overwrites_existing_container_json_data(manager_env):
    manager = _make_manager(manager_env)
    manager.settings.set("INSTALLED_CONTAINERS", ["authelia"])
    _write_v090_filecache(manager, containers=["nginx"])

    fresh = _make_manager(manager_env)
    assert fresh.settings.get("INSTALLED_CONTAINERS") == ["authelia"]


def test_fresh_install_never_creates_a_stray_setting_directory(manager_env):
    manager = _make_manager(manager_env)
    _ = manager.settings
    assert not (manager.data_path / "setting").exists()
    assert not (manager.data_path / "config").exists()
    assert not (manager.data_path / "repo" / "repo.json").exists()
    assert not (manager.environ.paths.config / "migrations").exists()


def _write_pre_v090_files(manager, containers=None, repos=None):
    import json

    if containers is not None:
        config_dir = manager.data_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "containers.yml").write_text(json.dumps(containers))
    if repos is not None:
        repo_dir = manager.data_path / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        (repo_dir / "repo.json").write_text(json.dumps(repos))
        (repo_dir / "repo.lock").write_text("")


def test_pre_v090_raw_files_are_migrated(manager_env):
    manager = _make_manager(manager_env)
    _write_pre_v090_files(
        manager,
        containers=["nginx", "authelia"],
        repos={"/some/repo": {"type": "local"}},
    )
    # An actual cloned repository directory alongside the bookkeeping file
    # -- must never be swept away, only the repo.json/repo.lock files.
    (manager.data_path / "repo" / "some-clone").mkdir(parents=True, exist_ok=True)

    fresh = _make_manager(manager_env)
    assert fresh.settings.get("INSTALLED_CONTAINERS") == ["nginx", "authelia"]
    assert fresh.settings.get("INSTALLED_REPOS") == {"/some/repo": {"type": "local"}}
    assert not (fresh.data_path / "config").exists()
    assert not (fresh.data_path / "repo" / "repo.json").exists()
    assert not (fresh.data_path / "repo" / "repo.lock").exists()
    assert (fresh.data_path / "repo" / "some-clone").exists()
    # Old files are backed up into paths.config/migrations/, not deleted outright.
    config_backups = _migration_backups(fresh, "config")
    assert config_backups and (config_backups[0] / "containers.yml").exists()
    assert _migration_backups(fresh, "repo.json")
    assert _migration_backups(fresh, "repo.lock")


def test_pre_v090_migration_never_overwrites_existing_container_json_data(manager_env):
    manager = _make_manager(manager_env)
    manager.settings.set("INSTALLED_CONTAINERS", ["authelia"])
    _write_pre_v090_files(manager, containers=["nginx"])

    fresh = _make_manager(manager_env)
    assert fresh.settings.get("INSTALLED_CONTAINERS") == ["authelia"]
