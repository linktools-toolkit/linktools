#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Config data migration from old format to new ConfigStore (v2 §3.3).

Migrates user-saved config data from the legacy ConfigCacheParser (.cfg INI
file) to the new ConfigStore (JSON). This is the ONLY piece that must preserve
user data (v2 §0: "只保护用户已有配置数据").

Flow (v2 §3.3)::

    inspect()  -> report what would migrate (dry-run)
    backup()   -> copy old config to a backup file
    migrate()  -> read old, write to new ConfigStore
    verify()   -> check new config is readable
    rollback() -> restore old from backup if migration failed
"""

import configparser
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["ConfigMigration"]


class ConfigMigration(object):
    """One-time config data migration (v2 §3.3)."""

    def __init__(self, config_store, logger=None):
        # type: (Any, Any) -> None
        self._store = config_store
        self._logger = logger

    def _log(self, level, msg):
        if self._logger is not None:
            getattr(self._logger, level)(msg)

    # -- inspect -----------------------------------------------------------

    def inspect(self, old_path):
        # type: (PathLike) -> Dict[str, Any]
        """Read the old config file and report what would migrate (dry-run).

        Returns a dict with: keys found, total count, file exists.
        """
        old_path = str(old_path)
        result = {"file_exists": os.path.isfile(old_path), "keys": [], "count": 0}
        if not result["file_exists"]:
            return result
        parser = configparser.ConfigParser()
        parser.optionxform = str  # preserve key case
        parser.read(old_path)
        for section in parser.sections():
            for key in parser[section]:
                # ConfigCacheParser stores under "<NAMESPACE>.CACHE" sections.
                full_key = key
                result["keys"].append(full_key)
                result["count"] += 1
        self._log("info", "ConfigMigration.inspect: %d keys in %s" % (result["count"], old_path))
        return result

    # -- backup ------------------------------------------------------------

    def backup(self, old_path, backup_path=None):
        # type: (PathLike, Optional[PathLike]) -> str
        """Copy the old config to a backup file. Returns the backup path."""
        old_path = str(old_path)
        if not os.path.isfile(old_path):
            raise FileNotFoundError("old config not found: %s" % old_path)
        if backup_path is None:
            backup_path = old_path + ".backup"
        backup_path = str(backup_path)
        shutil.copy2(old_path, backup_path)
        self._log("info", "ConfigMigration.backup: %s -> %s" % (old_path, backup_path))
        return backup_path

    # -- migrate -----------------------------------------------------------

    def migrate(self, old_path, *, key_map=None):
        # type: (PathLike, Optional[Dict[str, str]]) -> Dict[str, Any]
        """Read old config and write to ConfigStore. Returns a report.

        ``key_map`` optionally maps old key names to new namespaced keys
        (e.g. {"LOG_LEVEL": "logging.level"}). Unmapped keys go to "legacy.<key>".
        """
        old_path = str(old_path)
        key_map = key_map or {}
        report = {"migrated": {}, "skipped": [], "legacy": []}

        if not os.path.isfile(old_path):
            self._log("warning", "ConfigMigration: old config not found: %s" % old_path)
            return report

        parser = configparser.ConfigParser()
        parser.optionxform = str  # preserve key case
        parser.read(old_path)
        for section in parser.sections():
            for key in parser[section]:
                old_key = key
                value = parser[section][key]
                # Map to new key or legacy.
                new_key = key_map.get(old_key, "legacy.%s" % old_key)
                if new_key.startswith("legacy."):
                    report["legacy"].append(old_key)
                    self._log("warning", "ConfigMigration: unmapped key %s -> %s" % (old_key, new_key))
                # Don't overwrite if already in the new store.
                if new_key in self._store:
                    report["skipped"].append(old_key)
                    continue
                self._store.set(new_key, value)
                report["migrated"][old_key] = new_key

        self._log("info", "ConfigMigration: migrated %d, skipped %d, legacy %d" % (
            len(report["migrated"]), len(report["skipped"]), len(report["legacy"])))
        return report

    # -- verify -----------------------------------------------------------

    def verify(self):
        # type: () -> bool
        """Verify the new ConfigStore is readable (keys can be retrieved)."""
        try:
            keys = self._store.keys()
            for key in keys[:10]:  # spot-check first 10
                _ = self._store.get(key)
            return True
        except Exception as exc:
            self._log("error", "ConfigMigration.verify failed: %s" % exc)
            return False

    # -- rollback ---------------------------------------------------------

    def rollback(self, backup_path, old_path):
        # type: (PathLike, PathLike) -> None
        """Restore the old config from a backup."""
        backup_path = str(backup_path)
        old_path = str(old_path)
        if not os.path.isfile(backup_path):
            raise FileNotFoundError("backup not found: %s" % backup_path)
        shutil.copy2(backup_path, old_path)
        self._log("warning", "ConfigMigration.rollback: restored %s from %s" % (old_path, backup_path))
