"""Config subsystem: ConfigStore + ConfigMigration + Config schema/sources/resolver.

Merged (compact-layout spec §2.1) from the former _config_store.py,
core/_config_schema.py, core/_config_migration.py. Behaviour unchanged.
"""

import contextlib
import datetime
import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Union

from ..errors import (
    ConfigCastError,
    ConfigCycleError,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
)
from ..types import MISSING
from ..utils import atomic_write, get_file_hash

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "ConfigStore", "ConfigMigration", "Config", "ConfigField", "ConfigSchema",
    "ConfigResolver", "ConfigSource", "EnvironmentSource", "RuntimeOverrideSource",
    "PersistentSource", "FileSource", "DefaultSource", "AliasProvider",
    "LazyProvider", "PromptProvider", "ConfirmProvider", "ErrorProvider",
    "ChainProvider", "ResolvedConfig",
]


class ConfigStore(object):
    """A locked, atomically-written JSON key/value file."""

    def __init__(self, path: "Any", lock_manager: "Any | None" = None) -> None:
        self._path = Path(str(path))
        self._lock_manager = lock_manager
        self._data: "dict[str, Any]" = {}
        self.reload()

    @property
    def path(self) -> "Path":
        return self._path

    # -- load / flush -------------------------------------------------------

    def reload(self) -> None:
        """Re-read the file; missing -> empty, corrupt -> ConfigError."""
        if not self._path.exists():
            self._data = {}
            return
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError("cannot read config %s: %s" % (self._path, exc))
        try:
            data = json.loads(text)
        except ValueError as exc:
            # User-editable file: surface the corruption rather than silently
            # wiping it on the next write.
            raise ConfigError("config %s is not valid JSON: %s" % (self._path, exc))
        if not isinstance(data, dict):
            raise ConfigError("config %s must be a JSON object, got %s" % (self._path, type(data).__name__))
        self._data = data

    def _flush(self) -> None:
        atomic_write(
            self._path,
            json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=True),
        )

    # -- locking ------------------------------------------------------------

    @contextlib.contextmanager
    def _locked(self) -> "Iterator[None]":
        """Acquire the cross-process lock, reread, yield, then flush on exit."""
        if self._lock_manager is not None:
            lock = self._lock_manager.process_lock("config:" + self._path.name)
        else:
            # Fall back to a private filelock beside the config file.
            from filelock import FileLock

            lock = FileLock(str(self._path) + ".lock")
        with lock:
            self.reload()
            yield

    # -- read ---------------------------------------------------------------

    def get(self, key: str, default: "Any" = MISSING) -> "Any":
        """Return the value for ``key``, or ``default`` if absent (v4 §3.4).

        Uses MISSING as the sentinel so stored None is distinguishable from
        a missing key (``key in store`` vs ``store.get(key) is None``).
        """
        if key in self._data:
            return self._data[key]
        return default

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> "list[str]":
        return list(self._data.keys())

    def items(self) -> "list[tuple]":
        return list(self._data.items())

    # -- write (all go through the locked, atomic protocol) -----------------

    def set(self, key: str, value: "Any") -> None:
        with self._locked():
            self._data[key] = value
            self._flush()

    def save(self, **kwargs: "Any") -> None:
        with self._locked():
            self._data.update(kwargs)
            self._flush()

    def remove(self, *keys: str) -> bool:
        removed = False
        with self._locked():
            for key in keys:
                if key in self._data:
                    self._data.pop(key, None)
                    removed = True
            self._flush()
        return removed

    def delete(self, key: str) -> bool:
        """Alias for remove(key) (v5 P0-4: PersistentSource.delete calls this)."""
        return self.remove(key)

    def __repr__(self) -> str:
        return "ConfigStore(path=%r, keys=%d)" % (str(self._path), len(self._data))


PathLike = Union[str, Path]

# Heuristic: keys whose name matches one of these are treated as secrets and
# masked in migration reports (secret must not leak into reports).
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




class ConfigMigration(object):
    """One-time config data migration (v2 §3.3)."""

    def __init__(self, config_store: "Any", logger: "Any" = None,
                 config_dir: "PathLike | None" = None) -> None:
        self._store = config_store
        self._logger = logger
 # Where migration backups/reports live . Defaults to the
        # directory holding the ConfigStore file.
        if config_dir is not None:
            self._config_dir = Path(str(config_dir))
        else:
            self._config_dir = Path(str(getattr(config_store, "path", "."))).parent

    # built-in key map covering core + cntr configuration keys.
    # Section-qualified keys (SECTION.KEY) are authoritative; the bare-key
    # entries are a convenience fallback used ONLY when the bare key is
 # unambiguous .
    DEFAULT_KEY_MAP = {
        # -- explicit section-qualified mappings (preferred) --
        # Core (legacy ConfigCacheParser stored core under MAIN.CACHE)
        "MAIN.CACHE.DEBUG": "debug",
        "MAIN.CACHE.DATA_PATH": "data.path",
        "MAIN.CACHE.TEMP_PATH": "temp.path",
        "MAIN.CACHE.STORAGE_PATH": "storage.path",
        "MAIN.CACHE.DEFAULT_USER_AGENT": "download.user_agent",
        "MAIN.CACHE.DEFAULT_WAN_IP_URL": "network.wan_ip_url",
        # Cntr container manager (legacy CONTAINER.CACHE)
        "CONTAINER.CACHE.HOST": "container.host",
        "CONTAINER.CACHE.DOCKER_HOST": "container.docker_host",
        "CONTAINER.CACHE.COMPOSE_PROJECT_NAME": "container.compose_project_name",
        "CONTAINER.CACHE.SERVICE_RESTART_POLICY": "container.service_restart_policy",
        "CONTAINER.CACHE.SERVICE_LOG_DRIVER": "container.service_log_driver",
        "CONTAINER.CACHE.SERVICE_LOG_MAX_SIZE": "container.service_log_max_size",
        "CONTAINER.CACHE.DOCKER_USER": "container.docker_user",
        "CONTAINER.CACHE.DOCKER_UID": "container.docker_uid",
        "CONTAINER.CACHE.DOCKER_GID": "container.docker_gid",
        "CONTAINER.CACHE.DOCKER_TYPE": "container.docker_type",
        "CONTAINER.CACHE.DOCKER_APP_PATH": "container.docker_app_path",
        "CONTAINER.CACHE.DOCKER_APP_DATA_PATH": "container.docker_app_data_path",
        "CONTAINER.CACHE.DOCKER_USER_DATA_PATH": "container.docker_user_data_path",
        "CONTAINER.CACHE.DOCKER_DOWNLOAD_PATH": "container.docker_download_path",
        # Cntr flare container: FLARE_DOMAIN is canonical; the legacy
        # misspelling FLARE_DOAMIN maps to the same new key.
        "CONTAINER.CACHE.FLARE_DOMAIN": "container.flare.domain",
        "CONTAINER.CACHE.FLARE_DOAMIN": "container.flare.domain",
        # Cntr installed state (also migrated via _migrate.py)
        "CONTAINER.CACHE.INSTALLED_CONTAINERS": "container.installed_containers",
        "CONTAINER.CACHE.INSTALLED_REPOS": "container.installed_repos",
        "CONTAINER.CACHE.RUNNING_CONTAINERS": "container.running_containers",
        # -- bare-key fallback: only genuinely global core keys. Section-
        # sensitive keys (HOST, DOCKER_*, COMPOSE_*, SERVICE_*, FLARE_*,
        # INSTALLED_*) are deliberately NOT bare-mapped -- they must use their
        # full SECTION.KEY entry. Otherwise a stray MAIN.CACHE.HOST could be
        # pulled onto container.host via the (unambiguous) bare fallback.
        "DEBUG": "debug",
        "DATA_PATH": "data.path",
        "TEMP_PATH": "temp.path",
        "STORAGE_PATH": "storage.path",
        "DEFAULT_USER_AGENT": "download.user_agent",
        "DEFAULT_WAN_IP_URL": "network.wan_ip_url",
    }

    def _log(self, level: str, msg: str) -> None:
        if self._logger is not None:
            getattr(self._logger, level)(msg)

    # -- inspect -----------------------------------------------------------

    def _read_old(self, old_path: "PathLike") -> "list[tuple[str, str, str]]":
        """Return [(section, key, value), ...] with key case preserved."""
        import configparser  # only the migrate CLI path needs it
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

    def _resolve_new_key(self, section, key, key_map, ambiguous_keys=None):
        """Map an old (section, key) to a new namespaced key.

        Resolution order (fix-plan §1.3.2):
          1. explicit ``SECTION.KEY``
          2. normalized ``section.key``
          3. bare ``KEY`` -- only if that bare key is NOT ambiguous (i.e. it
             does not appear in more than one section); ambiguous bare keys
             must be mapped via the full ``SECTION.KEY`` or they are preserved
          4. otherwise ``legacy.<section>.<key>`` (never dropped)

        Returning ``mapped_legacy_bare`` for the bare-key fallback lets callers
        distinguish an explicit full-key mapping from an ambiguous-prone bare one.
        """
        ambiguous_keys = ambiguous_keys or set()
        full = "%s.%s" % (section, key)
        if full in key_map:
            return key_map[full], "mapped"
        nfull = _normalize(full)
        if nfull in key_map:
            return key_map[nfull], "mapped"
        if key in key_map and key not in ambiguous_keys:
            return key_map[key], "mapped_legacy_bare"
        nkey = _normalize(key)
        if nkey in key_map and nkey not in ambiguous_keys:
            return key_map[nkey], "mapped_legacy_bare"
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

 # -- backup (never overwrite) -----------------------------------

    def _migration_dir(self, migration_id):
        return self._config_dir / "migrations" / migration_id

    def _new_migration_id(self, old_path):
        # <UTC_TS>-<sha8>-<uuid8>: the uuid guarantees uniqueness even when two
 # backups of the same file land in the same second (never overwrite).
        import uuid
        return "%s-%s-%s" % (_utc_stamp(), get_file_hash(old_path, "sha256")[:8], uuid.uuid4().hex[:8])

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
                "sha256": get_file_hash(old_path, "sha256"),
                "created_at": _now_iso(),
                "migration_id": mid,
            }, indent=2))
        backup_path = str(backup_path)
        shutil.copy2(old_path, backup_path)
        self._log("info", "ConfigMigration.backup: %s -> %s" % (old_path, backup_path))
        return backup_path

 # -- migrate ----------------------------------------------

    def migrate(
        self,
        old_path: "PathLike",
        *,
        key_map: "dict[str, str] | None" = None,
        dry_run: bool = False,
    ) -> "dict[str, Any]":
        """Read old config and write to ConfigStore. Returns a report.

        Each old ``<section>.<key>`` is mapped via ``key_map``. Bare-key
        fallback is only used when the bare key is unambiguous (appears in a
        single section); otherwise the key must be mapped via its full
        ``SECTION.KEY`` or it is preserved at ``legacy.<section>.<key>`` so two
        same-named keys in different sections never collapse (fix-plan §1.3.2).

        Writes are planned first and applied in a single batch via
        ``store.save()`` so an interrupted migration cannot leave a half-written
        new store (fix-plan §1.3.4).
        """
        from collections import defaultdict

        old_path = str(old_path)
        key_map = self._merged_key_map(key_map)
        report = {"migrated": [], "skipped": [], "legacy": [], "entries": []}

        if not os.path.isfile(old_path):
            self._log("warning", "ConfigMigration: old config not found: %s" % old_path)
            return report

        entries = self._read_old(old_path)
        # A bare key present in >1 section is ambiguous: refuse to auto-map it
        # via the bare-key fallback (would collapse the sections onto one key).
        by_key = defaultdict(set)
        for section, key, _ in entries:
            by_key[key].add(section)
        ambiguous_keys = {k for k, secs in by_key.items() if len(secs) > 1}

        # Plan every entry first; apply as one batch write below.
        planned = {}  # new_key -> value
        for section, key, value in entries:
            full = "%s.%s" % (section, key)
            new_key, reason = self._resolve_new_key(
                section, key, key_map, ambiguous_keys)
            secret = _is_secret(full)
            # Skip if already in the store OR already claimed in this pass
            # (do not overwrite an existing/newer value or a sibling mapping).
            if new_key in self._store or new_key in planned:
                report["skipped"].append(full)
                report["entries"].append({"old_key": full, "new_key": new_key,
                                          "reason": "skipped_exists", "secret": secret})
                continue
            planned[new_key] = value
            if reason == "unknown_key_preserved":
                report["legacy"].append(full)
            else:
                report["migrated"].append(full)
            # NOTE: the raw value is intentionally NOT stored in the report, so
 # secret values can never leak into logs/CLI output .
            report["entries"].append({"old_key": full, "new_key": new_key,
                                      "reason": reason, "secret": secret})

        if planned and not dry_run:
            self._store.save(**planned)  # one locked, atomic batch write

        self._log("info", "ConfigMigration: migrated %d, skipped %d, legacy %d" % (
            len(report["migrated"]), len(report["skipped"]), len(report["legacy"])))
        return report

 # -- verify (full check) ----------------------

    def verify(self, report: "dict[str, Any] | None" = None) -> bool:
        """Verify the new ConfigStore is fully readable.

        Every key in the store must be retrievable. If a migration ``report`` is
        supplied, every mapped/legacy new_key it claims must be present and
        readable, and no secret entry may carry a raw ``value`` field.
        """
        try:
            for key in self._store.keys():
                _ = self._store.get(key)
            if report is not None:
                for entry in report.get("entries", []):
                    if entry["reason"] in ("mapped", "mapped_legacy_bare",
                                            "unknown_key_preserved"):
                        if entry["new_key"] not in self._store:
                            self._log("error", "ConfigMigration.verify: missing %s" % entry["new_key"])
                            return False
                        _ = self._store.get(entry["new_key"])  # must be readable
                    if entry.get("secret") and "value" in entry:
                        self._log("error", "ConfigMigration.verify: secret value leaked for %s"
                                  % entry["new_key"])
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


class AliasProvider:
    """Resolve a field's value from another field's value."""

    def __init__(self, target: str) -> None:
        self.target = target


class LazyProvider:
    """Compute a value from a callable. Stateless."""

    def __init__(self, func: "Callable[[ConfigResolver], Any]") -> None:
        self.func = func


class PromptProvider:
    """Prompt the user for a value. Stateless."""

    def __init__(self, message: str = None, default: "Any" = MISSING,
                 password: bool = False, choices: "list | None" = None) -> None:
        self.message = message
        self.default = default
        self.password = password
        self.choices = choices


class ConfirmProvider:
    """Ask the user for a yes/no confirmation."""

    def __init__(self, message: str = None, default: "Any" = MISSING) -> None:
        self.message = message
        self.default = default


class ErrorProvider:
    """Always raise ConfigError when resolved."""

    def __init__(self, message: str) -> None:
        self.message = message


class ChainProvider:
    """Try multiple providers in order until one yields a value."""

    def __init__(self, *providers: "Any") -> None:
        self.providers = list(providers)


class ConfigField:
    def __init__(self, name: str, default: "Any" = MISSING, cast: "Callable | None" = None,
                 validator: "Callable | None" = None, aliases: "Sequence[str]" = (),
                 required: bool = False, secret: bool = False, description: str = "",
                 deprecated: bool = False, provider: "Any" = None) -> None:
        self.name = name
        self.default = default
        self.cast = cast
        self.validator = validator
        self.aliases = tuple(aliases)
        self.required = required
        self.secret = secret
        self.description = description
        self.deprecated = deprecated
        self.provider = provider


class ConfigSchema:
    """A set of named ConfigFields with alias indexing.

    When ``allow_unknown`` is True, keys not in the schema are resolved
    through the source chain without field cast/validate.
    """

    def __init__(self, allow_unknown: bool = False) -> None:
        self._fields: "dict[str, ConfigField]" = {}
        self._alias_to_name: "dict[str, str]" = {}
        self.allow_unknown = allow_unknown

    def define(self, field: "ConfigField") -> "ConfigSchema":
        self._fields[field.name] = field
        for alias in field.aliases:
            self._alias_to_name[alias] = field.name
        return self

    def get(self, name: str) -> "ConfigField | None":
        if name in self._fields:
            return self._fields[name]
        target = self._alias_to_name.get(name)
        if target is not None:
            return self._fields.get(target)
        return None

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None


class ConfigSource:
    name = "source"

    def get(self, key: str) -> "tuple[Any, bool]":
        raise NotImplementedError


class EnvironmentSource(ConfigSource):
    name = "environment"

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    def get(self, key: str) -> "tuple[Any, bool]":
        value = os.environ.get(self._prefix + key, MISSING)
        if value is MISSING:
            return (MISSING, False)
        return (value, True)


class RuntimeOverrideSource(ConfigSource):
    name = "runtime-override"

    def __init__(self) -> None:
        self._data: "dict[str, Any]" = {}

    def set(self, key: str, value: "Any") -> None:
        self._data[key] = value

    def clear(self, key: "str | None" = None) -> None:
        if key is None:
            self._data.clear()
        else:
            self._data.pop(key, None)

    def get(self, key: str) -> "tuple[Any, bool]":
        if key in self._data:
            return (self._data[key], True)
        return (MISSING, False)


class PersistentSource(ConfigSource):
    """Reads/writes from a ConfigStore, key-prefixed by namespace."""

    name = "persistent"

    def __init__(self, store: "Any", namespace: str = "config") -> None:
        self._store = store
        self._prefix = (namespace + ".") if namespace else ""

    def _full(self, key: str) -> str:
        return self._prefix + key

    def get(self, key: str) -> "tuple[Any, bool]":
        full = self._full(key)
        if full in self._store:
            return (self._store.get(full), True)
        return (MISSING, False)

    def set(self, key: str, value: "Any") -> None:
        self._store.set(self._full(key), value)

    def delete(self, key: str) -> bool:
        return self._store.delete(self._full(key))

    def keys(self) -> "list[str]":
        """List this namespace's keys (store key minus the namespace prefix)."""
        result = []
        for full in self._store.keys():
            if not self._prefix or full.startswith(self._prefix):
                result.append(full[len(self._prefix):])
        return result

    def reload(self) -> None:
        self._store.reload()


class FileSource(ConfigSource):
    name = "file"

    def __init__(self, data: dict) -> None:
        self._data = dict(data)

    def get(self, key: str) -> "tuple[Any, bool]":
        if key in self._data:
            return (self._data[key], True)
        return (MISSING, False)


class DefaultSource(ConfigSource):
    name = "default"

    def __init__(self, schema: "ConfigSchema") -> None:
        self._schema = schema

    def get(self, key: str) -> "tuple[Any, bool]":
        field = self._schema.get(key)
        if field is not None and field.default is not MISSING:
            return (field.default, True)
        return (MISSING, False)


class ResolvedConfig:
    def __init__(self, value: "Any", field: "ConfigField | None",
                 source_name: str, raw_value: "Any") -> None:
        self.value = value
        self.field = field
        self.source_name = source_name
        self.raw_value = raw_value


class ConfigResolver:
    def __init__(self, schema: "ConfigSchema", sources: "Sequence[ConfigSource]") -> None:
        self._schema = schema
        self._sources = list(sources)
        self._memo: "dict[str, ResolvedConfig]" = {}

    def clear_memo(self, key: "str | None" = None) -> None:
        """Clear the resolution memo. If key given, invalidate only that key."""
        if key is not None:
            self._memo.pop(key, None)
        else:
            self._memo.clear()

    def get(self, key: str, type: "Any" = None, default: "Any" = MISSING) -> "Any":
        """Convenience: resolve and return the value (for LazyProvider lambdas)."""
        try:
            result = self.resolve(key)
            value = result.value
        except Exception:
            return default
        if type is not None and value is not MISSING:
            try:
                value = type(value)
            except (TypeError, ValueError):
                return default
        return value

    @staticmethod
    def _candidates(field: "ConfigField") -> "tuple[str, Ellipsis]":
        return (field.name,) + field.aliases

    def _cast_validate(self, field: "ConfigField", raw: "Any") -> "Any":
        value = raw
        if field.cast is not None:
            try:
                value = field.cast(raw)
            except Exception as exc:
                raise ConfigCastError("cannot cast %r for %s: %s" % (raw, field.name, exc))
        if field.validator is not None:
            try:
                ok = field.validator(value)
            except Exception as exc:
                raise ConfigValidationError("validator for %s raised: %s" % (field.name, exc))
            if not ok:
                raise ConfigValidationError("value %r failed validation for %s" % (value, field.name))
        return value

    def _first_present(self, field: "ConfigField") -> "tuple[str | None, Any, bool]":
        for source in self._sources:
            for candidate in self._candidates(field):
                raw, present = source.get(candidate)
                if present:
                    return (source.name, raw, True)
        return (None, MISSING, False)

    def _first_present_raw(self, key: str) -> "tuple[str | None, Any, bool]":
        for source in self._sources:
            raw, present = source.get(key)
            if present:
                return (source.name, raw, True)
        return (None, MISSING, False)

    def resolve(self, key: str, _stack: "list[str] | None" = None) -> "ResolvedConfig":
        if _stack is None and key in self._memo:
            return self._memo[key]
        result = self._resolve_inner(key, _stack)
        if _stack is None:
            self._memo[key] = result
        return result

    def _resolve_inner(self, key: str, _stack: "list[str] | None" = None) -> "ResolvedConfig":
        field = self._schema.get(key)
        if field is None:
            if self._schema.allow_unknown:
                source_name, raw, present = self._first_present_raw(key)
                if present:
                    return ResolvedConfig(raw, None, source_name, raw)
                raise ConfigNotFoundError("config %r is not set and has no default" % (key,))
            raise ConfigNotFoundError("unknown config key %r" % (key,))

        provider = field.provider
        if isinstance(provider, AliasProvider):
            stack = _stack if _stack is not None else []
            if key in stack:
                chain = stack[stack.index(key):] + [key]
                raise ConfigCycleError("config alias cycle: " + " -> ".join(chain))
            stack.append(key)
            return self.resolve(provider.target, _stack=stack)

        if isinstance(provider, LazyProvider):
            value = provider.func(self)
            return ResolvedConfig(self._cast_validate(field, value), field, "lazy", value)

        if isinstance(provider, PromptProvider):
            from ..rich import prompt
            value = prompt(provider.message or field.name, default=provider.default,
                           password=provider.password, choices=provider.choices)
            return ResolvedConfig(self._cast_validate(field, value), field, "prompt", value)

        if isinstance(provider, ConfirmProvider):
            from ..rich import confirm
            value = confirm(provider.message or field.name, default=provider.default)
            return ResolvedConfig(self._cast_validate(field, value), field, "confirm", value)

        if isinstance(provider, ErrorProvider):
            raise ConfigError(provider.message)

        if isinstance(provider, ChainProvider):
            for sub in provider.providers:
                try:
                    return self._try_provider(sub, field, key, _stack)
                except Exception:
                    continue

        source_name, raw, present = self._first_present(field)
        if present:
            return ResolvedConfig(self._cast_validate(field, raw), field, source_name, raw)
        if field.default is not MISSING:
            return ResolvedConfig(self._cast_validate(field, field.default), field, "default", field.default)
        if field.required:
            raise ConfigNotFoundError("required config %r is not set" % (key,))
        raise ConfigNotFoundError("config %r is not set and has no default" % (key,))

    def _try_provider(self, provider: "Any", field: "ConfigField",
                      key: str, _stack: "list[str] | None" = None) -> "ResolvedConfig":
        if isinstance(provider, AliasProvider):
            stack = list(_stack or [])
            if key in stack:
                chain = stack[stack.index(key):] + [key]
                raise ConfigCycleError("config alias cycle: " + " -> ".join(chain))
            stack.append(key)
            return self.resolve(provider.target, _stack=stack)

        if isinstance(provider, LazyProvider):
            value = provider.func(self)
            return ResolvedConfig(self._cast_validate(field, value), field, "lazy", value)

        if isinstance(provider, PromptProvider):
            from ..rich import prompt
            value = prompt(provider.message or field.name, default=provider.default,
                           password=provider.password, choices=provider.choices)
            return ResolvedConfig(self._cast_validate(field, value), field, "prompt", value)

        if isinstance(provider, ConfirmProvider):
            from ..rich import confirm
            value = confirm(provider.message or field.name, default=provider.default)
            return ResolvedConfig(self._cast_validate(field, value), field, "confirm", value)

        if isinstance(provider, ErrorProvider):
            raise ConfigError(provider.message)

        raise ConfigNotFoundError("unknown provider type: %r" % type(provider).__name__)

    def explain(self, key: str) -> dict:
        try:
            resolved = self.resolve(key)
        except ConfigNotFoundError:
            # Key is neither in the schema nor present in any source.
            return {
                "key": key,
                "unknown": True,
                "found": False,
                "resolved_value": MISSING,
                "selected_source": None,
                "raw_value": MISSING,
                "all_candidates": [],
                "secret": False,
                "deprecated": False,
                "description": "",
                "warnings": ["key is not set and has no default"],
            }
        field = resolved.field
        unknown = field is None
        candidate_keys = self._candidates(field) if field is not None else (key,)
        candidates: "list[dict]" = []
        for source in self._sources:
            for candidate in candidate_keys:
                raw, present = source.get(candidate)
                if present:
                    candidates.append({"source": source.name, "key": candidate, "raw": raw})
        secret = bool(field.secret) if field is not None else False
        if secret:
            shown_value = "***"
            shown_raw = "***"
        else:
            shown_value = resolved.value
            shown_raw = resolved.raw_value
        warnings: "list[str]" = []
        if unknown:
            warnings.append("key is not defined in schema")
        return {
            "key": key,
            "unknown": unknown,
            "resolved_value": shown_value,
            "selected_source": resolved.source_name,
            "raw_value": shown_raw,
            "all_candidates": candidates,
            "secret": secret,
            "deprecated": field.deprecated if field is not None else False,
            "description": field.description if field is not None else "",
            "warnings": warnings,
        }


class Config:
    """The user-facing config, backed by ConfigResolver."""

    def __init__(self, environ: "Any", schema: "ConfigSchema",
                 sources: "Sequence[ConfigSource]") -> None:
        self._environ = environ
        self._schema = schema
        self._sources = list(sources)
        self._resolver = ConfigResolver(schema, self._sources)

    @property
    def schema(self) -> "ConfigSchema":
        return self._schema

    def define(self, field: "ConfigField") -> "Config":
        self._schema.define(field)
        return self

    def _find(self, source_class: type) -> "ConfigSource | None":
        for s in self._sources:
            if isinstance(s, source_class):
                return s
        return None

    def get(self, key: str, type: "type | None" = None, default: "Any" = MISSING) -> "Any":
        try:
            result = self._resolver.resolve(key)
            value = result.value
        except ConfigNotFoundError:
            # Missing with no default -> surface the error instead of silently
            # propagating MISSING into business logic. Callers that accept
            # absence must pass an explicit default.
            if default is MISSING:
                raise
            return default
        if type is not None and value is not MISSING:
            try:
                value = type(value)
            except (TypeError, ValueError) as exc:
                if default is not MISSING:
                    return default
                raise ConfigCastError("cannot cast %r for %s: %s" % (value, key, exc))
        return value

    def require(self, key: str, type: "type | None" = None) -> "Any":
        """Resolve a must-exist key; raise ConfigNotFoundError if it is missing."""
        return self.get(key, type=type, default=MISSING)

    def set(self, key: str, value: "Any") -> None:
        runtime = self._find(RuntimeOverrideSource)
        if runtime is None:
            raise ConfigError("no RuntimeOverrideSource configured")
        runtime.set(key, value)
        self._resolver.clear_memo(key)

    def persist(self, key: str, value: "Any") -> None:
        persistent = self._find(PersistentSource)
        if persistent is None:
            raise ConfigError("no PersistentSource configured")
        persistent.set(key, value)
        self._resolver.clear_memo(key)

    def unset(self, key: str) -> None:
        runtime = self._find(RuntimeOverrideSource)
        if runtime:
            runtime.clear(key)
        self._resolver.clear_memo(key)

    def remove(self, key: str) -> None:
        persistent = self._find(PersistentSource)
        if persistent:
            persistent.delete(key)
        self._resolver.clear_memo(key)

    def explain(self, key: str) -> dict:
        return self._resolver.explain(key)

    def reload(self, clear_runtime: bool = False) -> None:
        if clear_runtime:
            runtime = self._find(RuntimeOverrideSource)
            if runtime:
                runtime.clear()
        for source in self._sources:
            reload_fn = getattr(source, "reload", None)
            if callable(reload_fn):
                reload_fn()
        self._resolver.clear_memo()

    def keys(self) -> "list[str]":
        known = set()
        for name in self._schema._fields:
            known.add(name)
        for source in self._sources:
            # Prefer a source-level keys() (PersistentSource knows its namespace
            # prefix); fall back to introspecting _data/_ns for older sources.
            keys_fn = getattr(source, "keys", None)
            if callable(keys_fn):
                try:
                    known.update(keys_fn())
                except Exception:
                    pass
            data = getattr(source, "_data", None)
            if isinstance(data, dict):
                known.update(data.keys())
            ns = getattr(source, "_ns", None)
            if ns is not None:
                try:
                    known.update(ns.keys())
                except Exception:
                    pass
        return sorted(known)

    def update_defaults(self, **kwargs: "Any") -> "Config":
        for key, value in kwargs.items():
            if isinstance(value, ConfigField):
                if value.name != key:
                    value = ConfigField(
                        name=key, default=value.default, cast=value.cast,
                        validator=value.validator, aliases=value.aliases,
                        required=value.required, secret=value.secret,
                        description=value.description, deprecated=value.deprecated,
                        provider=value.provider)
                self._schema.define(value)
            elif hasattr(value, "providers") or hasattr(value, "target") or \
                    hasattr(value, "func") or hasattr(value, "message"):
                self._schema.define(ConfigField(name=key, provider=value))
            elif hasattr(value, "get") and hasattr(value, "load") and \
                    hasattr(value, "_default"):
                default = getattr(value, "_default", MISSING)
                if default is MISSING:
                    default = None
                self._schema.define(ConfigField(name=key, default=default))
            else:
                self._schema.define(ConfigField(name=key, default=value))
        return self

    def cast(self, value: "Any", type: "Any" = None) -> "Any":
        if type is None or type is MISSING:
            return value
        if type == "path":
            return os.path.abspath(os.path.expanduser(str(value)))
        if callable(type):
            return type(value)
        return value
