#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linktools.core import ConfigStore
    from .container import ContainerManager


def _digest(path: "Path") -> str:
    h = hashlib.sha256()
    if path.is_dir():
        for sub in sorted(path.rglob("*")):
            if sub.is_file():
                h.update(sub.read_bytes())
    else:
        h.update(path.read_bytes())
    return h.hexdigest()[:8]


def _backup(config_dir: "Path", path: "Path") -> None:
    """Move ``path`` into ``<config_dir>/migrations/<migration_id>/``
    instead of deleting it outright -- a migration is a one-way, best-
    effort read of an old format; keep the original around in case
    something about the read turns out to be wrong.

    ``migration_id`` is ``<UTC timestamp>-<sha256 prefix>-<uuid4 prefix>``:
    unique per call, so repeated migrations never overwrite a previous
    backup (matches the ``ConfigMigration`` class this project used to
    have, before it was deleted outright in commit 043b058d).
    """
    if not path.exists() and not path.is_symlink():
        return
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    migration_id = f"{stamp}-{_digest(path)}-{uuid.uuid4().hex[:8]}"
    dest_dir = config_dir / "migrations" / migration_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    (dest_dir / "report.json").write_text(json.dumps({
        "source": str(path),
        "backup": str(dest),
        "migrated_at": stamp,
    }, indent=2))
    shutil.move(str(path), str(dest))


def migrate_legacy_settings(manager: "ContainerManager", new_store: "ConfigStore"):

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
            _backup(manager.environ.paths.config, old_setting_path)
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
            _backup(manager.environ.paths.config, config_path.parent)
            manager.logger.info(f"Migrated old config file: {config_path}")
        except Exception as e:
            manager.logger.warning(f"Failed to migrate old config file: {e}")

    if "INSTALLED_REPOS" not in new_store and os.path.isfile(repo_path):
        manager.logger.warning("Found old repo file, try to migrate.")
        try:
            with open(repo_path) as fd:
                new_store.set("INSTALLED_REPOS", json.load(fd))
            _backup(manager.environ.paths.config, repo_path)
            _backup(manager.environ.paths.config, repo_lock)
            manager.logger.info(f"Migrated old repo file: {repo_path}")
        except Exception as e:
            manager.logger.warning(f"Failed to migrate old repo file: {e}")

    return new_store
