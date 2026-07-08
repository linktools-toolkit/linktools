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
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing import Any

__all__ = ["ConfigMigration"]

PathLike = Union[str, Path]

# Heuristic: keys whose name matches one of these are treated as secrets and
# masked in migration reports (spec §4.8: secret must not leak into reports).
_SECRET_HINTS = ("PASSWORD", "PASSWD", "PWD", "SECRET", "TOKEN", "API_KEY", "PRIVATE_KEY")


def _normalize(value: str) -> str:
    return (value or "").strip().lower()


def _is_secret(key: str) -> bool:
    upper = (key or "").upper()
    return any(hint in upper for hint in _SECRET_HINTS)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ConfigMigration(object):
    """One-time config data migration (v2 §3.3)."""

    def __init__(self, config_store: "Any", logger: "Any" = None,
                 config_dir: "PathLike | None" = None) -> None:
        self._store = config_store
        self._logger = logger
        # Where migration backups/reports live (spec §4.6). Defaults to the
        # directory holding the ConfigStore file.
        if config_dir is not None:
            self._config_dir = Path(str(config_dir))
        else:
            self._config_dir = Path(str(getattr(config_store, "path", "."))).parent

    # built-in key map covering core + cntr configuration keys.
    DEFAULT_KEY_MAP = {
        # Core
        "DEBUG": "debug",
        "DATA_PATH": "data.path",
        "TEMP_PATH": "temp.path",
        "STORAGE_PATH": "storage.path",
        "DEFAULT_USER_AGENT": "download.user_agent",
        "DEFAULT_WAN_IP_URL": "network.wan_ip_url",
        # Cntr container manager
        "HOST": "container.host",
        "DOCKER_HOST": "container.docker_host",
        "COMPOSE_PROJECT_NAME": "container.compose_project_name",
        "SERVICE_RESTART_POLICY": "container.service_restart_policy",
        "SERVICE_LOG_DRIVER": "container.service_log_driver",
        "SERVICE_LOG_MAX_SIZE": "container.service_log_max_size",
        "DOCKER_USER": "container.docker_user",
        "DOCKER_UID": "container.docker_uid",
        "DOCKER_GID": "container.docker_gid",
        "DOCKER_TYPE": "container.docker_type",
        "DOCKER_APP_PATH": "container.docker_app_path",
        "DOCKER_APP_DATA_PATH": "container.docker_app_data_path",
        "DOCKER_USER_DATA_PATH": "container.docker_user_data_path",
        "DOCKER_DOWNLOAD_PATH": "container.docker_download_path",
        # Cntr flare container: FLARE_DOMAIN is the canonical key; the legacy
        # misspelling FLARE_DOAMIN is mapped too so old user configs migrate.
        "FLARE_DOMAIN": "container.flare.domain",
        "FLARE_DOAMIN": "container.flare.domain",
        # Cntr installed state (already migrated via _migrate.py)
        "INSTALLED_CONTAINERS": "container.installed_containers",
        "INSTALLED_REPOS": "container.installed_repos",
        "RUNNING_CONTAINERS": "container.running_containers",
    }

    def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            getattr(self._logger, level)(msg)

    # -- inspect -----------------------------------------------------------

    def _read_old(self, old_path: "PathLike") -> "list[tuple[str, str, str]]":
        """Return [(section, key, value), ...] with key case preserved."""
        parser = configparser.ConfigParser()
        parser.optionxform = str  # preserve key case
        parser.read(str(old_path))
        entries = []
        for section in parser.sections():
            for key in parser[section]:
                entries.append((section, key, parser[section][key]))
        return entries

    @staticmethod
    def _merged_key_map(key_map):
        merged = dict(ConfigMigration.DEFAULT_KEY_MAP)
        if key_map:
            merged.update(key_map)
        return merged

    def _resolve_new_key(self, section, key, key_map):
        """Map an old (section, key) to a new namespaced key.

        Fallback order (spec §4.3): SECTION.KEY -> KEY -> legacy.SECTION.KEY.
        """
        full = "%s.%s" % (section, key)
        for candidate in (full, key, _normalize(full), _normalize(key)):
            if candidate in key_map:
                return key_map[candidate], "mapped"
        return ("legacy.%s.%s" % (_normalize(section), _normalize(key)),
                "unknown_key_preserved")

    def inspect(self, old_path: "PathLike") -> "dict[str, Any]":
        """Read the old config file and report what would migrate (dry-run).

        Keys are reported fully-qualified as ``<section>.<key>`` so same-named
        keys in different sections do not collide (spec §4.3).
        """
        old_path = str(old_path)
        result = {"file_exists": os.path.isfile(old_path), "keys": [], "count": 0}
        if not result["file_exists"]:
            return result
        for section, key, _ in self._read_old(old_path):
            result["keys"].append("%s.%s" % (section, key))
            result["count"] += 1
        self._log("info", "ConfigMigration.inspect: %d keys in %s" % (result["count"], old_path))
        return result

    # -- backup (§4.6: never overwrite) -----------------------------------

    def _migration_dir(self, migration_id):
        return self._config_dir / "migrations" / migration_id

    def _new_migration_id(self, old_path):
        # <UTC_TS>-<sha8>-<uuid8>: the uuid guarantees uniqueness even when two
        # backups of the same file land in the same second (§4.6: never overwrite).
        import uuid
        return "%s-%s-%s" % (_utc_stamp(), _sha256_file(old_path)[:8], uuid.uuid4().hex[:8])

    def backup(self, old_path: "PathLike", migration_id: "str | None" = None,
               backup_path: "PathLike | None" = None) -> str:
        """Copy the old config into a unique migrations/<id>/ dir (§4.6).

        Each call lands in ``<config_dir>/migrations/<UTC_TS>-<sha8>/`` with
        ``old-config.backup`` and ``report.json``, so repeated migrations never
        overwrite a previous backup. Returns the backup file path.
        """
        old_path = str(old_path)
        if not os.path.isfile(old_path):
            raise FileNotFoundError("old config not found: %s" % old_path)
        if backup_path is None:
            mid = migration_id or self._new_migration_id(old_path)
            dest_dir = self._migration_dir(mid)
            dest_dir.mkdir(parents=True, exist_ok=True)
            backup_path = dest_dir / "old-config.backup"
            report_path = dest_dir / "report.json"
            report_path.write_text(json.dumps({
                "source": old_path,
                "backup": str(backup_path),
                "sha256": _sha256_file(old_path),
                "created_at": _now_iso(),
                "migration_id": mid,
            }, indent=2))
        backup_path = str(backup_path)
        shutil.copy2(old_path, backup_path)
        self._log("info", "ConfigMigration.backup: %s -> %s" % (old_path, backup_path))
        return backup_path

    # -- migrate (§4.5/§4.7) ----------------------------------------------

    def migrate(
        self,
        old_path: "PathLike",
        *,
        key_map: "dict[str, str] | None" = None,
        dry_run: bool = False,
    ) -> "dict[str, Any]":
        """Read old config and write to ConfigStore. Returns a report.

        Each old ``<section>.<key>`` is mapped via ``key_map`` (fallback
        SECTION.KEY -> KEY), unmapped keys are preserved at
        ``legacy.<section>.<key>``, and keys already present are skipped. The
        report records every entry with old_key/new_key/reason; secret keys are
        flagged so callers can mask them.
        """
        old_path = str(old_path)
        key_map = self._merged_key_map(key_map)
        report = {"migrated": [], "skipped": [], "legacy": [], "entries": []}

        if not os.path.isfile(old_path):
            self._log("warning", "ConfigMigration: old config not found: %s" % old_path)
            return report

        for section, key, value in self._read_old(old_path):
            full = "%s.%s" % (section, key)
            new_key, reason = self._resolve_new_key(section, key, key_map)
            secret = _is_secret(full)
            if new_key in self._store:
                report["skipped"].append(full)
                report["entries"].append({"old_key": full, "new_key": new_key,
                                          "reason": "skipped_exists", "secret": secret})
                continue
            if not dry_run:
                self._store.set(new_key, value)
            if reason == "unknown_key_preserved":
                report["legacy"].append(full)
            else:
                report["migrated"].append(full)
            report["entries"].append({"old_key": full, "new_key": new_key,
                                      "reason": reason, "secret": secret})

        self._log("info", "ConfigMigration: migrated %d, skipped %d, legacy %d" % (
            len(report["migrated"]), len(report["skipped"]), len(report["legacy"])))
        return report

    # -- verify (§4.8: full, not a spot-check) ----------------------------

    def verify(self, report: "dict[str, Any] | None" = None) -> bool:
        """Verify the new ConfigStore is fully readable.

        Every key in the store must be retrievable. If a migration ``report`` is
        supplied, every mapped/legacy new_key it claims must also be present.
        """
        try:
            for key in self._store.keys():
                _ = self._store.get(key)
            if report is not None:
                for entry in report.get("entries", []):
                    if entry["reason"] in ("mapped", "unknown_key_preserved"):
                        if entry["new_key"] not in self._store:
                            self._log("error", "ConfigMigration.verify: missing %s" % entry["new_key"])
                            return False
            return True
        except Exception as exc:
            self._log("error", "ConfigMigration.verify failed: %s" % exc)
            return False

    # -- rollback ---------------------------------------------------------

    def rollback(self, backup_path: "PathLike", old_path: "PathLike") -> None:
        """Restore the old config from a backup."""
        backup_path = str(backup_path)
        old_path = str(old_path)
        if not os.path.isfile(backup_path):
            raise FileNotFoundError("backup not found: %s" % backup_path)
        shutil.copy2(backup_path, old_path)
        self._log("warning", "ConfigMigration.rollback: restored %s from %s" % (old_path, backup_path))
