#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One-time migration of cntr's legacy settings into ConfigStore (spec §21.2).

cntr historically stored its persistent user state (INSTALLED_CONTAINERS /
INSTALLED_REPOS) in three successive formats. This module consolidates all of
them into the persistent :class:`linktools._config_store.ConfigStore`, in age
order:

  1. ``<data>/config/containers.yml``   -- the original JSON file (very old)
  2. ``<data>/repo/repo.json``          -- the original repo-list file (old)
  3. legacy FileCache shelve ``<setting>/manager``  -- the format this refactor
     replaces

Each source is migrated ONLY if the target key is absent from ConfigStore (so a
user who already has newer data is never overwritten), and the source is removed
once migrated. Idempotent: a no-op when nothing legacy remains.

This is the ONLY module in the codebase that still imports the legacy
``FileCache``; both this migrator and ``linktools/cache.py`` are removed after
one release (spec §21.2 / §3.3).
"""
import json
import os
from typing import Set

from linktools.cache import FileCache  # legacy -- imported only for migration
from linktools.utils import remove_file

__all__ = ["PERSISTENT_KEYS", "migrate_legacy_container_settings"]

# Keys whose lifecycle is "persistent user state" -> ConfigStore (spec §8.5).
# Transient keys (RUNNING_CONTAINERS, per-container mount paths) are regenerable
# and are NOT migrated; they move straight to the CacheStore.
PERSISTENT_KEYS = ("INSTALLED_CONTAINERS", "INSTALLED_REPOS")


def _remove(path):
    # type: (str) -> None
    try:
        remove_file(path)
    except Exception:
        pass


def _migrate_json_file(config_store, source_path, key, also_remove, logger):
    # type: (object, str, str, tuple, object) -> bool
    """Migrate one legacy JSON file -> config_store[key]; clean up afterwards."""
    if not os.path.isfile(source_path):
        return False
    if key in config_store:
        # Newer data already present -- just drop the legacy artefact.
        for p in also_remove:
            _remove(p)
        return False
    try:
        with open(source_path) as fd:
            config_store.set(key, json.load(fd))
        logger.warning("Migrated %s from legacy %s", key, source_path)
    except Exception as exc:
        logger.warning("Failed to migrate %s from %s: %s", key, source_path, exc)
        return False
    for p in also_remove:
        _remove(p)
    return True


def _migrate_shelve(config_store, legacy_dir, logger):
    # type: (object, str, object) -> Set[str]
    """Migrate persistent keys from the legacy FileCache shelve -> config_store."""
    if not os.path.isdir(legacy_dir):
        return set()
    migrated = set()  # type: Set[str]
    try:
        cache = FileCache(legacy_dir)
        with cache.session() as data:
            for key in PERSISTENT_KEYS:
                if key in config_store:
                    continue
                value = data.get(key, None)
                if value is not None:
                    config_store.set(key, value)
                    migrated.add(key)
                    logger.warning("Migrated %s from legacy FileCache", key)
    except Exception as exc:
        logger.warning("Failed to migrate legacy FileCache at %s: %s", legacy_dir, exc)
        return migrated
    # Shelve fully consumed -> remove the legacy store directory.
    _remove(legacy_dir)
    return migrated


def migrate_legacy_container_settings(config_store, data_path, setting_path, logger):
    # type: (object, object, object, object) -> Set[str]
    """Migrate every legacy source of cntr's persistent settings into config_store.

    Args:
        config_store: the destination persistent store.
        data_path: cntr's data directory (contains ``config/`` and ``repo/``).
        setting_path: cntr's settings directory (contained the legacy shelve).
        logger: a logger for migration progress/warnings.

    Returns:
        The set of keys migrated in this call (empty on a clean, already-current
        install).
    """
    data_path = str(data_path)
    setting_path = str(setting_path)
    migrated = set()  # type: Set[str]

    if _migrate_json_file(
        config_store,
        os.path.join(data_path, "config", "containers.yml"),
        "INSTALLED_CONTAINERS",
        (os.path.join(data_path, "config"),),
        logger,
    ):
        migrated.add("INSTALLED_CONTAINERS")

    if _migrate_json_file(
        config_store,
        os.path.join(data_path, "repo", "repo.json"),
        "INSTALLED_REPOS",
        (os.path.join(data_path, "repo", "repo.json"),
         os.path.join(data_path, "repo", "repo.lock")),
        logger,
    ):
        migrated.add("INSTALLED_REPOS")

    migrated |= _migrate_shelve(
        config_store, os.path.join(setting_path, "manager"), logger
    )
    return migrated
