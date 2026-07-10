#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One-time migration of cntr's legacy settings into ConfigStore (spec §21.2).

cntr historically stored its persistent user state (INSTALLED_CONTAINERS /
INSTALLED_REPOS) in three successive formats. This module consolidates all of
them into the persistent :class:`linktools.core.ConfigStore`, in age
order:

  1. ``<data>/config/containers.yml``   -- the original JSON file (very old)
  2. ``<data>/repo/repo.json``          -- the original repo-list file (old)
  3. legacy FileCache shelve ``<setting>/manager``  -- the format this refactor
     replaces

The original, older ConfigCacheParser ini file (``<data>/.config/<name>.cfg``,
which also holds CONTAINER.CACHE.* tunables like HOST/DOCKER_TYPE/DOCKER_USER)
is migrated by core itself: ``Environ.config_store`` runs it the first time
anything touches the persistent store, before cntr's ContainerManager is even
constructed (see ``core/_environ.py: BaseEnviron._migrate_legacy_cfg``). By the
time this module runs, that file is already gone.

Each source is migrated ONLY if the target key is absent from ConfigStore (so a
user who already has newer data is never overwritten), and the source is removed
once migrated. Idempotent: a no-op when nothing legacy remains.

The actual read/write/cleanup mechanics are shared with core's own
``.cfg`` migration via ``ConfigMigration.migrate_json_file`` /
``migrate_shelve`` (core's config and cntr's config go through the same
``ConfigMigration`` methods, not two hand-rolled implementations) -- this
module only supplies cntr's own legacy source paths.

This is the ONLY module in the codebase that still (transitively, via
``ConfigMigration.migrate_shelve``) deals with the legacy ``FileCache``
shelve format; both this migrator and ``linktools/cache.py`` are removed
after one release (spec §21.2 / §3.3).
"""
import os

__all__ = ["PERSISTENT_KEYS", "migrate_legacy_container_settings"]

# Keys whose lifecycle is "persistent user state" -> ConfigStore.
# Transient keys (RUNNING_CONTAINERS, per-container mount paths) are regenerable
# and are NOT migrated; they move straight to the CacheStore.
PERSISTENT_KEYS = ("INSTALLED_CONTAINERS", "INSTALLED_REPOS")


def migrate_legacy_container_settings(config_store: object, data_path: object,
                                      setting_path: object, logger: object) -> "set[str]":
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
    from linktools.core import ConfigMigration

    data_path = str(data_path)
    setting_path = str(setting_path)
    mig = ConfigMigration(config_store, logger=logger)
    migrated: "set[str]" = set()

    if mig.migrate_json_file(
        os.path.join(data_path, "config", "containers.yml"),
        "INSTALLED_CONTAINERS",
        also_remove=(os.path.join(data_path, "config"),),
    ):
        migrated.add("INSTALLED_CONTAINERS")

    if mig.migrate_json_file(
        os.path.join(data_path, "repo", "repo.json"),
        "INSTALLED_REPOS",
        also_remove=(os.path.join(data_path, "repo", "repo.json"),
                     os.path.join(data_path, "repo", "repo.lock")),
    ):
        migrated.add("INSTALLED_REPOS")

    migrated |= mig.migrate_shelve(
        os.path.join(setting_path, "manager"), PERSISTENT_KEYS
    )

    return migrated
