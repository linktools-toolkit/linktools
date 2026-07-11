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
from typing import TYPE_CHECKING

from ..errors import (
    ConfigCastError,
    ConfigCycleError,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
)
from ..types import MISSING
from ..utils import atomic_write, get_file_hash, remove_file

if TYPE_CHECKING:
    from typing import Any, Callable, Iterator, Literal, Sequence, Union

    # The two string literals _cast_value special-cases (mirrors the old,
    # pre-v2 CONFIG_TYPES dispatch table): "path" expands/absolutizes a
    # filesystem path, "json" parses a JSON string. Anything else must be a
    # callable (including the builtins ``bool``/``str``, which _cast_value
    # also special-cases -- see _cast_bool/_cast_str).
    ConfigLiteralType = Literal["path", "json"]
    ConfigType = Union[ConfigLiteralType, Callable[[Any], Any], None]

    PathLike = Union[str, Path]


__all__ = [
    "ConfigStore", "ConfigMigration", "Config", "ConfigField", "ConfigSchema",
    "ConfigResolver", "ConfigSource", "EnvironmentSource", "RuntimeOverrideSource",
    "PersistentSource", "FileSource", "DefaultSource", "AliasProvider",
    "ConfigProvider", "LazyProvider", "PromptProvider", "ConfirmProvider", "ErrorProvider",
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
        return dict(key_map) if key_map else {}

    def _resolve_new_key(self, section, key, key_map, ambiguous_keys=None):
        """Map an old (section, key) to a new namespaced key.

        Resolution order:
          1. explicit ``SECTION.KEY`` (caller-supplied ``key_map`` overrides)
          2. normalized ``section.key``
          3. bare ``KEY`` -- only if that bare key is NOT ambiguous (i.e. it
             does not appear in more than one section); ambiguous bare keys
             must be mapped via the full ``SECTION.KEY`` or fall through below
          4. generic: any section following the legacy ConfigCacheParser
             "<NAMESPACE>.CACHE" convention maps to "<namespace-lower>.<KEY>"
             -- the same format ``Config.wrap_config(namespace=...)`` reads via
             PersistentSource. This is what lets ALL fields migrate (goal: not
             just the ones a caller happens to enumerate), including custom
             fields defined by sub-package/container configs this module has
             never heard of. Section-qualification already makes this
             collision-free, so no ambiguity check is needed here. Two cntr
             keys need special handling because the generic rule would land
             them somewhere nothing reads: the legacy misspelling
             FLARE_DOAMIN (renamed to FLARE_DOMAIN, the key the code actually
             reads) and INSTALLED_CONTAINERS/INSTALLED_REPOS (read bare, with
             no namespace prefix -- see ContainerManager._persistent_store /
             _migrate.py; RUNNING_CONTAINERS is deliberately excluded, it's
             transient state read from the *cache* store instead).
          5. otherwise ``legacy.<section>.<key>`` (never dropped) -- only hit
             for a section that doesn't even follow the "*.CACHE" convention.

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
        if section.upper().endswith(".CACHE"):
            namespace = section[:-len(".CACHE")].lower()
            if namespace:
                if key == "FLARE_DOAMIN":
                    return "%s.FLARE_DOMAIN" % namespace, "mapped"
                if key in ("INSTALLED_CONTAINERS", "INSTALLED_REPOS"):
                    return key, "mapped"
                return "%s.%s" % (namespace, key), "mapped_generic"
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

        Every old ``<section>.<key>`` is migrated -- not just the ones a
        caller happens to enumerate. An optional ``key_map`` of exceptions
        (renames, ambiguous-bare-key overrides) wins first; anything else
        under a "<NAMESPACE>.CACHE"-style section falls through to the
        generic "<namespace-lower>.<KEY>" mapping (with two built-in cntr
        exceptions -- FLARE_DOAMIN and INSTALLED_CONTAINERS/REPOS) so it is
        still migrated to a real, readable location (see
        ``_resolve_new_key``). Only a section that doesn't even follow that
        convention is preserved (never dropped) at ``legacy.<section>.<key>``.

        Writes are planned first and applied in a single batch via
        ``store.save()`` so an interrupted migration cannot leave a half-written
        new store.
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

    # -- migrate other legacy formats (shared with sub-packages) -----------
    #
    # ``migrate()`` above only understands the ini-style ConfigCacheParser
    # ``.cfg`` format. Sub-packages (e.g. cntr) have their own pre-ConfigStore
    # legacy formats -- a bare JSON blob file, or a legacy FileCache shelve --
    # that need the exact same "migrate once, never overwrite newer data,
    # clean up the source" behaviour. These two methods are that shared
    # mechanism, so core's and cntr's config migrations go through the same
    # ConfigMigration methods instead of each hand-rolling it.

    @staticmethod
    def _remove(path: "PathLike") -> None:
        try:
            remove_file(str(path))
        except Exception:
            pass

    def migrate_json_file(self, source_path: "PathLike", key: str,
                          also_remove: "Sequence[PathLike]" = ()) -> bool:
        """Migrate one legacy JSON blob file into ``store[key]``.

        Returns True if a value was migrated. If ``key`` is already present,
        the source is dropped (never overwritten) and this returns False.
        ``also_remove`` lets a source directory/lockfile be cleaned up
        alongside the primary file even when nothing was migrated.
        """
        source_path = str(source_path)
        if not os.path.isfile(source_path):
            return False
        cleanup = tuple(also_remove) or (source_path,)
        if key in self._store:
            for p in cleanup:
                self._remove(p)
            return False
        try:
            with open(source_path, encoding="utf-8") as fd:
                value = json.load(fd)
        except Exception as exc:
            self._log("warning", "ConfigMigration.migrate_json_file: failed to read %s: %s"
                      % (source_path, exc))
            return False
        self._store.set(key, value)
        self._log("warning", "ConfigMigration.migrate_json_file: migrated %s from %s"
                  % (key, source_path))
        for p in cleanup:
            self._remove(p)
        return True

    def migrate_shelve(self, legacy_dir: "PathLike", keys: "Sequence[str]") -> "set[str]":
        """Migrate the given ``keys`` out of a legacy FileCache shelve dir.

        Only the listed keys are moved (regenerable/transient state should be
        left for the caller to handle separately); the shelve directory is
        removed once considered, whether or not anything was migrated.
        """
        from ..cache import FileCache  # legacy structure; import kept local

        legacy_dir = str(legacy_dir)
        if not os.path.isdir(legacy_dir):
            return set()
        migrated: "set[str]" = set()
        try:
            cache = FileCache(legacy_dir)
            with cache.session() as data:
                for key in keys:
                    if key in self._store:
                        continue
                    value = data.get(key, None)
                    if value is not None:
                        self._store.set(key, value)
                        migrated.add(key)
                        self._log("warning",
                                  "ConfigMigration.migrate_shelve: migrated %s from legacy FileCache" % key)
        except Exception as exc:
            self._log("warning", "ConfigMigration.migrate_shelve: failed at %s: %s" % (legacy_dir, exc))
            return migrated
        self._remove(legacy_dir)
        return migrated

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
                                            "mapped_generic", "unknown_key_preserved"):
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


class ConfigProvider:
    """Base class for providers understood by ConfigResolver."""


class AliasProvider(ConfigProvider):
    """Resolve a field's value from another field's value."""

    def __init__(self, target: str) -> None:
        self.target = target


class LazyProvider(ConfigProvider):
    """Compute a value from a callable.

    ``cached=True`` persists the computed value (via the schema's
    PersistentSource, keyed by field name) the first time it is computed, so a
    non-deterministic ``func`` (e.g. one that generates a random secret) is
    only ever invoked once -- later resolutions reuse the persisted value
    instead of recomputing. Stateless otherwise.
    """

    def __init__(self, func: "Callable[[ConfigResolver], Any]", cached: bool = False) -> None:
        self.func = func
        self.cached = cached


class PromptProvider(ConfigProvider):
    """Prompt the user for a value.

    ``cached=True`` persists the answer (via the schema's PersistentSource,
    keyed by field name) so the user is only ever asked once -- later
    resolutions reuse the persisted answer instead of prompting again.
    Stateless otherwise.
    """

    def __init__(self, message: str = None, default: "Any" = MISSING,
                 password: bool = False, choices: "list | None" = None,
                 cached: bool = False, allow_empty: bool = False) -> None:
        self.message = message
        self.default = default
        self.password = password
        self.choices = choices
        self.cached = cached
        self.allow_empty = allow_empty


class ConfirmProvider(ConfigProvider):
    """Ask the user for a yes/no confirmation.

    ``cached=True`` persists the answer (via the schema's PersistentSource,
    keyed by field name) so the user is only ever asked once. Stateless
    otherwise.
    """

    def __init__(self, message: str = None, default: "Any" = MISSING, cached: bool = False) -> None:
        self.message = message
        self.default = default
        self.cached = cached


class ErrorProvider(ConfigProvider):
    """Always raise ConfigError when resolved."""

    def __init__(self, message: str) -> None:
        self.message = message


class ChainProvider(ConfigProvider):
    """Try multiple providers in order until one yields a value."""

    def __init__(self, *providers: "Any") -> None:
        self.providers = list(providers)


class ConfigField:
    def __init__(self, name: "str | None" = None, default: "Any" = MISSING, cast: "ConfigType" = None,
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

    @classmethod
    def chain(cls, *providers: "Any", **kwargs: "Any") -> "ConfigField":
        """Build a field whose provider is a ``ChainProvider`` of ``providers``.

        Shorthand for the common
        ``ConfigField(name=..., provider=ChainProvider(a, b, ...))`` spelling,
        which repeats the field name and adds a level of nesting for every
        chained field. ``name`` is normally omitted here: dict-based callers
        (``configs`` properties consumed via ``Config.update_defaults``) get it
        from the dict key automatically.

        Always wraps in ``ChainProvider`` even for a single provider: that
        wrapper is not a no-op -- ``ConfigResolver`` lets a ``ChainProvider``
        sub-provider's exception fall through to ``_first_present``/
        ``field.default``, while a bare provider's exception propagates
        directly. Collapsing a single-provider chain to the bare provider
        would silently drop that fallback.
        """
        return cls(provider=ChainProvider(*providers), **kwargs)


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
        previous = self._fields.get(field.name)
        if previous is not None:
            for alias in previous.aliases:
                if self._alias_to_name.get(alias) == field.name:
                    del self._alias_to_name[alias]
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

    def fields(self) -> "list[ConfigField]":
        """Every defined field, in definition order (insertion order of a
        Python dict). Lets a caller copy a schema's fields verbatim (e.g. a
        per-repository Config starting from the same base fields as its
        manager's Config, spec §35) without re-deriving them from scratch.
        """
        return list(self._fields.values())


class ConfigSource:
    name = "source"

    # Whether this source is consulted ahead of a field's provider
    # (Prompt/Lazy/Alias/Chain), not merely as its failure fallback -- see
    # ConfigResolver._first_present_before_provider. Source *order* (not
    # this flag) is the only thing that decides relative priority among
    # before_provider sources; this flag only decides the provider/no-provider
    # split.
    before_provider = False

    def get(self, key: str) -> "tuple[Any, bool]":
        raise NotImplementedError


class EnvironmentSource(ConfigSource):
    name = "environment"
    before_provider = True

    def __init__(self, prefix: str = "") -> None:
        self._prefix = prefix

    def get(self, key: str) -> "tuple[Any, bool]":
        value = os.environ.get(self._prefix + key, MISSING)
        if value is MISSING:
            return (MISSING, False)
        return (value, True)


class RuntimeOverrideSource(ConfigSource):
    name = "runtime-override"
    before_provider = True

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
    before_provider = True

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
    """A read-only source backed by a plain dict, e.g. a
    ``LinktoolsFileConfig.environment`` (spec §21).

    ``name`` lets two instances (local-file / global-file) report distinct
    ``explain()`` source names instead of both showing up as ``"file"``.
    ``reload_fn``, if given, is called by ``reload()`` to atomically replace
    this source's data (e.g. from a fresh ``LinktoolsFileConfigLoader.load()``)
    without ever leaving a half-updated state.
    """

    before_provider = True

    def __init__(self, data: dict, name: str = "file", reload_fn: "Callable[[], dict] | None" = None) -> None:
        self._data = dict(data)
        self.name = name
        self._reload_fn = reload_fn

    def get(self, key: str) -> "tuple[Any, bool]":
        if key in self._data:
            return (self._data[key], True)
        return (MISSING, False)

    def keys(self) -> "list[str]":
        return list(self._data.keys())

    def reload(self) -> None:
        if self._reload_fn is None:
            return
        data = self._reload_fn()
        self._data = dict(data)


class DefaultSource(ConfigSource):
    name = "default"
    before_provider = False

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


def _cast_bool(value: "Any") -> bool:
    """Cast a config value to bool without Python's ``bool("false") == True`` trap.

    The builtin ``bool()`` treats any non-empty string as truthy, so a bare
    ``cast=bool`` would turn an env var / persisted string like ``"false"``
    into ``True``. Recognize the common textual forms explicitly instead.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("true", "yes", "y", "on", "1"):
            return True
        if lowered in ("false", "no", "n", "off", "0"):
            return False
        raise TypeError("str %r cannot be converted to type bool" % (value,))
    return bool(value)


def _cast_str(value: "Any") -> str:
    """Cast a config value to str, JSON-dumping structured values instead of
    falling back to their Python ``repr`` (e.g. ``str(["a"])`` == ``"['a']"``,
    not valid JSON), and mapping ``None`` to ``""`` rather than ``"None"``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value)
    if value is None:
        return ""
    return str(value)


def _cast_json(value: "Any") -> "list | dict":
    """Cast a config value (typically a raw env-var/persisted string) to
    JSON-compatible data."""
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, (tuple, list, dict)):
        return value
    raise TypeError("%s cannot be converted to json" % (type(value).__name__,))


def _cast_value(cast: "ConfigType", value: "Any") -> "Any":
    """Apply a cast that may be a callable, ``bool``/``str``, or one of the
    string literals ``"path"``/``"json"``.

    ``ConfigField.cast`` accepts a callable or one of the ``ConfigType``
    literals (mirroring the explicit ``Config.cast(value, type=...)`` API);
    both must resolve identically. A bare ``field.cast(value)`` call fails on
    the string literal forms with ``TypeError: 'str' object is not
    callable``, so callers route through here. ``bool``/``str`` are
    special-cased the same way ``"path"``/``"json"`` are: passed straight to
    the builtin, ``bool`` would silently misparse strings (see
    ``_cast_bool``) and ``str`` would produce non-JSON reprs for structured
    values (see ``_cast_str``).
    """
    if cast == "path":
        return os.path.abspath(os.path.expanduser(str(value)))
    if cast == "json":
        return _cast_json(value)
    if cast is bool:
        return _cast_bool(value)
    if cast is str:
        return _cast_str(value)
    if callable(cast):
        return cast(value)
    return value


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
        except ConfigNotFoundError:
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

    @staticmethod
    def _prompt_value(provider: "PromptProvider", field: "ConfigField") -> "Any":
        message = provider.message or field.name
        default = provider.default if provider.default is not MISSING else field.default
        if provider.choices:
            from ..rich import choose
            return choose(message, provider.choices, default=default)
        from ..rich import prompt
        # Forward the field's cast as the prompt's target type (rich.prompt
        # only knows str/int/float/bool -- anything else, e.g. a custom
        # callable or the "path" literal, still gets a plain string prompt
        # and is cast afterwards by _cast_validate). Without this, an int/bool
        # field always got a bare string prompt, so a real user's typed answer
        # only worked by coincidence (str -> int/bool casts cleanly for valid
        # input) and any type-aware placeholder (e.g. in tests) had no way to
        # return a plausible value for the field it was actually asking about.
        type_hint = field.cast if field.cast in (str, int, float, bool) else str
        return prompt(message, type=type_hint, default=default,
                      password=provider.password, allow_empty=provider.allow_empty)

    def _persistent_source(self) -> "PersistentSource | None":
        for source in self._sources:
            if isinstance(source, PersistentSource):
                return source
        return None

    def _resolve_cached(self, field: "ConfigField", compute: "Callable[[], Any]") -> "tuple[Any, str]":
        """Resolve a ``cached=True`` provider: reuse a persisted value if one
        exists for this field, else compute it once and persist the result.

        Returns ``(value, source_name)``. Raises ``ConfigError`` if the schema
        has no PersistentSource configured (cached=True requires one) -- this
        surfaces the misconfiguration immediately rather than silently
        recomputing (e.g. re-prompting or regenerating a secret) every call.
        """
        persistent = self._persistent_source()
        if persistent is None:
            raise ConfigError(
                "cached=True provider for %r requires a PersistentSource" % (field.name,))
        raw, present = persistent.get(field.name)
        if present:
            return raw, "persistent"
        value = self._cast_validate(field, compute())
        persistent.set(field.name, value)
        return value, "persistent"

    def _cast_validate(self, field: "ConfigField", raw: "Any") -> "Any":
        value = raw
        if field.cast is not None:
            try:
                value = _cast_value(field.cast, raw)
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

    def _first_present_before_provider(self, field: "ConfigField") -> "tuple[str | None, Any, bool]":
        """Check every ``before_provider`` source (Environment, RuntimeOverride,
        Persistent, and now the local/global file sources) ahead of the
        field's provider.

        A field's ``provider`` must not silently outrank something already
        set: without this check, any field with a provider (nearly every
        meaningfully-configured field in practice) never even looks at these
        sources, because ``_resolve_inner`` tries the provider first and only
        falls back to ``_first_present`` if every sub-provider raises.

        This mirrors the pre-refactor legacy ``Config``, whose ``_map`` was a
        ``ChainMap(env_vars, persistent_cache, field_descriptors,
        global_config)`` -- the persisted cache was checked, for every field,
        *before* its ``Config.Prompt``/``Config.Lazy``/``Config.Alias``
        descriptor ever ran, regardless of that descriptor's own ``cached=``
        flag (which only controlled whether a freshly-computed answer got
        saved back, not whether an existing one was read first). Without
        Persistent here, a field whose provider lacks ``cached=True`` (e.g.
        HOST) never looks at an already-persisted/migrated value at all --
        every resolution re-prompts -- and even a ``cached=True`` field only
        checks the persisted value via its own ``_resolve_cached``, which
        still runs after ``AliasProvider``/other sub-providers earlier in a
        ``ChainProvider`` have already had a chance to run first.

        Also fixes: a same-named env var set for one particular invocation
        (e.g. ``NGINX_ROOT_DOMAIN=x ct-cntr up``) used to be permanently
        ignored once a cached=True provider had persisted an answer, since
        the provider ran (and "succeeded" from cache) before the environment
        was ever consulted.

        Source *order* (not this ``before_provider`` flag) is the only thing
        deciding relative priority among these sources -- this method never
        hardcodes a source class beyond the flag check, so a new
        ``before_provider`` source needs no change here (spec §22-23).
        """
        for source in self._sources:
            if not source.before_provider:
                continue
            for candidate in self._candidates(field):
                raw, present = source.get(candidate)
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

    def _resolve_alias(self, provider: "AliasProvider", key: str,
                       _stack: "list[str] | None") -> "ResolvedConfig":
        stack = list(_stack or [])
        if key in stack:
            chain = stack[stack.index(key):] + [key]
            raise ConfigCycleError("config alias cycle: " + " -> ".join(chain))
        stack.append(key)
        return self.resolve(provider.target, _stack=stack)

    def _resolve_leaf_provider(self, provider: "Any", field: "ConfigField") -> "ResolvedConfig":
        """Resolve a Lazy/Prompt/Confirm/Error provider (no Alias/Chain nesting).

        ``cached=True`` on Lazy/Prompt/Confirm routes the actual compute/ask
        through ``_resolve_cached`` so it only ever runs once per field.
        """
        if isinstance(provider, LazyProvider):
            if provider.cached:
                value, source_name = self._resolve_cached(field, lambda: provider.func(self))
                return ResolvedConfig(self._cast_validate(field, value), field, source_name, value)
            value = provider.func(self)
            return ResolvedConfig(self._cast_validate(field, value), field, "lazy", value)

        if isinstance(provider, PromptProvider):
            if provider.cached:
                value, source_name = self._resolve_cached(
                    field, lambda: self._prompt_value(provider, field))
                return ResolvedConfig(self._cast_validate(field, value), field, source_name, value)
            value = self._prompt_value(provider, field)
            return ResolvedConfig(self._cast_validate(field, value), field, "prompt", value)

        if isinstance(provider, ConfirmProvider):
            from ..rich import confirm
            default = provider.default if provider.default is not MISSING else field.default
            if provider.cached:
                value, source_name = self._resolve_cached(
                    field, lambda: confirm(provider.message or field.name, default=default))
                return ResolvedConfig(self._cast_validate(field, value), field, source_name, value)
            value = confirm(provider.message or field.name, default=default)
            return ResolvedConfig(self._cast_validate(field, value), field, "confirm", value)

        if isinstance(provider, ErrorProvider):
            raise ConfigError(provider.message)

        raise ConfigNotFoundError("unknown provider type: %r" % type(provider).__name__)

    def _resolve_inner(self, key: str, _stack: "list[str] | None" = None) -> "ResolvedConfig":
        field = self._schema.get(key)
        if field is None:
            if self._schema.allow_unknown:
                source_name, raw, present = self._first_present_raw(key)
                if present:
                    return ResolvedConfig(raw, None, source_name, raw)
                raise ConfigNotFoundError("config %r is not set and has no default" % (key,))
            raise ConfigNotFoundError("unknown config key %r" % (key,))

        # An explicit before_provider-source value (Environment/RuntimeOverride/
        # Persistent/local-file/global-file) always outranks the field's
        # provider (see _first_present_before_provider) -- checked ahead of
        # AliasProvider/ChainProvider/leaf-provider resolution, not merely as
        # their failure fallback.
        override_source, override_raw, override_present = self._first_present_before_provider(field)
        if override_present:
            return ResolvedConfig(self._cast_validate(field, override_raw), field, override_source, override_raw)

        provider = field.provider
        if isinstance(provider, AliasProvider):
            return self._resolve_alias(provider, key, _stack)

        if isinstance(provider, ChainProvider):
            for sub in provider.providers:
                try:
                    return self._try_provider(sub, field, key, _stack)
                except Exception:
                    continue
        elif provider is not None:
            return self._resolve_leaf_provider(provider, field)

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
            return self._resolve_alias(provider, key, _stack)
        return self._resolve_leaf_provider(provider, field)

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

    def persisted_keys(self) -> "list[str]":
        """Keys already answered/configured in the PersistentSource.

        Unlike ``keys()`` (every schema-declared field name, whether or not
        it has ever been set, plus every persisted/env/runtime key), this is
        only the keys someone has actually set. A caller that lists "every
        current config value" by resolving each key must use this instead of
        ``keys()`` for whatever it adds beyond its own explicitly-declared
        fields -- otherwise it forces resolution (and, for an uncached-so-far
        Prompt/Confirm-backed field, an interactive prompt) for every field
        merely because it's *possible* to set, not because it's been set.
        """
        persistent = self._find(PersistentSource)
        if persistent is None:
            return []
        return sorted(persistent.keys())

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
            elif isinstance(value, ConfigProvider):
                self._schema.define(ConfigField(name=key, provider=value))
            else:
                self._schema.define(ConfigField(name=key, default=value))
        return self

    def cast(self, value: "Any", type: "Any" = None) -> "Any":
        if type is None or type is MISSING:
            return value
        return _cast_value(type, value)
