#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import TYPE_CHECKING

from linktools.core import Config, backup_legacy_path

if TYPE_CHECKING:
    from linktools.core import ConfigStore
    from .container import ContainerManager


def migrate_legacy_settings(manager: "ContainerManager", new_store: "ConfigStore") -> "ConfigStore":
    if Config.is_read_only_resolution():
        return new_store

    # One-time migration from v0.9.0: INSTALLED_CONTAINERS/INSTALLED_REPOS
    # used to live in a shelve database at
    # <data_path>/setting/manager/data/data (v0.9.0's now-removed
    # linktools.cache.FileCache -- CacheStore replaced it everywhere
    # else, so this reads the raw shelve record directly instead of
    # keeping the whole legacy class around for one migration path). An
    # upgrading v0.9.0 installation's data there must move into this
    # store instead of silently becoming invisible
    # (prepare_installed_containers()/repos.get_all() would otherwise
    # see nothing installed). Guarded on the shelve file actually
    # existing first: shelve.open() creates it on first touch, and a
    # fresh (never-v0.9.0) install must never gain a stray setting/
    # directory just from accessing this property.
    try:
        old_setting_path = manager.data_path / "setting"
        old_shelve_path = old_setting_path / "manager" / "data" / "data"
        if old_shelve_path.parent.is_dir():
            import shelve

            manager.logger.warning("Found old v0.9.0 settings file, try to migrate.")
            migrated = []
            with shelve.open(str(old_shelve_path)) as old_db:
                for key in ("INSTALLED_CONTAINERS", "INSTALLED_REPOS"):
                    # FileCache's record shape: {"data": value, "ttl":
                    # ..., "ts": ...} -- both keys were persisted with no
                    # ttl (never expire), so no expiry check needed here.
                    if key not in new_store and key in old_db:
                        new_store.set(key, old_db[key]["data"])
                        migrated.append(key)
            backup_legacy_path(manager.environ.paths.config, old_setting_path)
            if migrated:
                manager.logger.info(f"Migrated old v0.9.0 settings: {', '.join(migrated)}")
    except Exception as e:
        # Best-effort: a corrupt/unreadable v0.9.0 shelve file must
        # never block construction of the manager itself -- same
        # fail-soft contract v0.9.0's own migration had.
        manager.logger.warning(f"Failed to migrate old settings: {e}")

    # One-time migration from pre-v0.9.0: INSTALLED_CONTAINERS/
    # INSTALLED_REPOS used to live in raw files (<data_path>/config/
    # containers.yml, <data_path>/repo/repo.json) before v0.9.0's own
    # FileCache migration existed. An install jumping straight from
    # pre-v0.9.0 to today skips v0.9.0 entirely, so its FileCache-based
    # migration (which handled this same jump) never runs -- this is
    # that same migration, ported forward so the chain isn't broken by
    # v0.9.0 no longer being an intermediate step anyone passes through.
    config_path = manager.data_path.joinpath("config", "containers.yml")
    repo_path = manager.data_path.joinpath("repo", "repo.json")
    repo_lock = manager.data_path.joinpath("repo", "repo.lock")

    if "INSTALLED_CONTAINERS" not in new_store and os.path.isfile(config_path):
        manager.logger.warning("Found old config file, try to migrate.")
        try:
            with open(config_path) as fd:
                new_store.set("INSTALLED_CONTAINERS", json.load(fd))
            backup_legacy_path(manager.environ.paths.config, config_path.parent)
            manager.logger.info(f"Migrated old config file: {config_path}")
        except Exception as e:
            manager.logger.warning(f"Failed to migrate old config file: {e}")

    if "INSTALLED_REPOS" not in new_store and os.path.isfile(repo_path):
        manager.logger.warning("Found old repo file, try to migrate.")
        try:
            with open(repo_path) as fd:
                new_store.set("INSTALLED_REPOS", json.load(fd))
            backup_legacy_path(manager.environ.paths.config, repo_path)
            backup_legacy_path(manager.environ.paths.config, repo_lock)
            manager.logger.info(f"Migrated old repo file: {repo_path}")
        except Exception as e:
            manager.logger.warning(f"Failed to migrate old repo file: {e}")

    return new_store
