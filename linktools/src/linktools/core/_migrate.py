#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-time migrations from older on-disk config formats."""

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._config_store import ConfigStore
    from ._environ import BaseEnviron


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


def migrate_legacy_config_cfg(environ: "BaseEnviron", store: "ConfigStore") -> None:
    """One-time migration from <=0.9.0: every namespace's persisted
    config lived as one section (``<NAMESPACE>.CACHE``) in a single INI
    file, ``<data_path>/.config/<name>.cfg``, written by the old
    configparser-based ``ConfigCache``/``PersistentSource``. Every field
    in every section -- not just specific known keys -- must move into
    this store instead of silently becoming invisible the moment a
    v0.9.0-or-older install upgrades. Values migrate as the raw strings
    the old INI format always stored (configparser has no other kind);
    each field's own ``cast=`` still applies the same way it always did
    at read time, so this needs no per-field knowledge.
    """
    old_path = environ.data_path / ".config" / f"{environ.name}.cfg"
    if not old_path.is_file():
        return
    logger = environ.get_logger("config")
    logger.warning(f"Found old config file, try to migrate: {old_path}")
    try:
        import configparser

        parser = configparser.ConfigParser()
        parser.optionxform = str  # preserve option-name casing (old ConfigParser subclass did the same)
        parser.read(str(old_path))
        migrated = []
        for section in parser.sections():
            if not section.upper().endswith(".CACHE"):
                continue
            namespace = section[:-len(".CACHE")].lower()
            prefix = f"{namespace}." if namespace else ""
            for key, value in parser.items(section):
                full_key = prefix + key
                if full_key not in store:
                    store.set(full_key, value)
                    migrated.append(full_key)
        # Not utils.remove_file(): it checks environ.debug, which (via
        # the process-wide singleton) can re-enter this same
        # not-yet-cached _config_store construction and recurse.
        _backup(environ.paths.config, old_path)
        if migrated:
            logger.info(f"Migrated old config file: {', '.join(migrated)}")
    except Exception as e:
        logger.warning(f"Failed to migrate old config file: {e}")
