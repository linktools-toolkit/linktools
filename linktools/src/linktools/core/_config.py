"""Config subsystem: Config schema/sources/resolver.

``ConfigStore`` (the persistence layer) lives in ``_config_store.py``;
``ProjectProfile``/loader live in ``_profile.py``.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import (
    CliError,
    ConfigCastError,
    ConfigCycleError,
    ConfigError,
    ConfigFieldError,
    ConfigNotFoundError,
    ConfigProviderUnavailable,
    ConfigPromptError,
    ConfigValidationError,
)
from ..types import MISSING

if TYPE_CHECKING:
    from typing import Any, Callable, Literal, Sequence, Union

    # The two string literals _cast_value special-cases: "path"
    # expands/absolutizes a filesystem path, "json" parses a JSON string.
    # Anything else must be a callable (including the builtins
    # ``bool``/``str``, which _cast_value also special-cases -- see
    # _cast_bool/_cast_str).
    ConfigLiteralType = Literal["path", "json"]
    ConfigType = Union[ConfigLiteralType, Callable[[Any], Any], None]

    PathLike = Union[str, Path]


__all__ = [
    "Config", "ConfigField", "ConfigSchema",
    "ConfigResolver", "ConfigSource", "EnvironmentSource", "RuntimeOverrideSource",
    "PersistentSource", "FileSource", "DictSource", "DefaultSource", "AliasProvider",
    "ConfigProvider", "LazyProvider", "PromptProvider", "ConfirmProvider", "ErrorProvider",
    "ChainProvider", "ResolvedConfig",
]


def redact_config_value(field: "ConfigField | None", value: "Any") -> "Any":
    """Mask ``value`` if ``field`` is a secret field; the single choke point
    every secret-redaction site (explain candidates/resolved/raw, Config
    list/get, CLI output) must go through."""
    if field is not None and field.secret:
        return "***"
    return value


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

    @classmethod
    def coerce(cls, name: str, value: "Any") -> "ConfigField":
        """Normalize one ``configs``-dict entry into a ``ConfigField`` named
        ``name``, without defining it on any schema.

        Shared by ``Config.update_defaults`` and any caller that needs the
        resulting ``ConfigField`` before deciding whether/how to define it
        (e.g. cntr's container config registration, which checks field
        ownership/collisions first) -- ``value`` is a ``ConfigField`` (kept
        as-is if already named ``name``, else copied under it), a bare
        ``ConfigProvider`` (wrapped with no default), or a plain default
        value.
        """
        if isinstance(value, ConfigField):
            if value.name == name:
                return value
            return cls(name=name, default=value.default, cast=value.cast,
                       validator=value.validator, aliases=value.aliases,
                       required=value.required, secret=value.secret,
                       description=value.description, deprecated=value.deprecated,
                       provider=value.provider)
        if isinstance(value, ConfigProvider):
            return cls(name=name, provider=value)
        return cls(name=name, default=value)


class ConfigSchema:
    """A set of named ConfigFields with alias indexing.

    Keys not in the schema are still resolved through the source chain,
    without field cast/validate.
    """

    def __init__(self) -> None:
        self._fields: "dict[str, ConfigField]" = {}
        self._alias_to_name: "dict[str, str]" = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def define(self, field: "ConfigField") -> "ConfigSchema":
        """Register (or re-register) ``field``.

        Re-defining the same ``field.name`` is the normal update path (a
        container/manager redeclaring its own field) and always succeeds,
        replacing the previous definition and dropping its now-stale
        aliases. Anything that would make two *different* fields collide --
        a bad name, a duplicate/self alias, or an alias claimed by (or
        claiming) another field's canonical name -- raises ConfigFieldError
        instead of silently overwriting one field's cast/validator/secret
        with another's, which is how a third party silently hijacking a
        Manager-owned key, or two sibling containers racing on scan order,
        used to go unnoticed.
        """
        if not isinstance(field.name, str) or not field.name:
            raise ConfigFieldError(
                "ConfigField.name must be a non-empty string, got %r" % (field.name,))

        seen_aliases: "set" = set()
        for alias in field.aliases:
            if not isinstance(alias, str) or not alias:
                raise ConfigFieldError(
                    "field %r has an invalid alias %r: aliases must be non-empty strings"
                    % (field.name, alias))
            if alias == field.name:
                raise ConfigFieldError("field %r cannot alias itself" % (field.name,))
            if alias in seen_aliases:
                raise ConfigFieldError(
                    "field %r declares duplicate alias %r" % (field.name, alias))
            seen_aliases.add(alias)
            if alias in self._fields:
                raise ConfigFieldError(
                    "alias %r for field %r collides with an existing field of the same name"
                    % (alias, field.name))
            existing_target = self._alias_to_name.get(alias)
            if existing_target is not None and existing_target != field.name:
                raise ConfigFieldError(
                    "alias %r for field %r already aliases field %r"
                    % (alias, field.name, existing_target))

        existing_alias_owner = self._alias_to_name.get(field.name)
        if existing_alias_owner is not None and existing_alias_owner != field.name:
            raise ConfigFieldError(
                "field name %r collides with an existing alias of field %r"
                % (field.name, existing_alias_owner))

        previous = self._fields.get(field.name)
        if previous is not None:
            for alias in previous.aliases:
                if self._alias_to_name.get(alias) == field.name:
                    del self._alias_to_name[alias]
        self._fields[field.name] = field
        for alias in field.aliases:
            self._alias_to_name[alias] = field.name
        self._revision += 1
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
        manager's Config) without re-deriving them from scratch.
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

    # Directory a source's relative ``cast="path"`` values resolve against;
    # None for sources with no filesystem origin (env/runtime/persistent).
    # Only FileSource instances ever set this to something else.
    base_path: "PathLike | None" = None

    @property
    def revision(self) -> "Any":
        """Opaque, comparable-by-equality token that changes whenever this
        source's data changes. ``ConfigResolver`` uses it to detect a write
        made through a *different* Config sharing this same source instance
        (e.g. sibling per-repository Configs sharing one
        RuntimeOverrideSource/PersistentSource) -- 0 for an immutable source
        that never changes after construction (Environment, Default)."""
        return 0

    def get(self, key: str) -> "tuple[Any, bool]":
        raise NotImplementedError

    def keys(self) -> "list[str]":
        return []

    def reload(self) -> None:
        return None


class _UnprefixedMapping(Mapping):
    def __init__(self, mapping: Mapping, prefix: str) -> None:
        self._mapping = mapping
        self._prefix = prefix

    def __getitem__(self, key: str):
        if key in self._mapping:
            return self._mapping[key]
        return self._mapping[self._prefix + key]

    def __iter__(self):
        prefix_length = len(self._prefix)
        for key in self._mapping:
            if key.startswith(self._prefix):
                yield key[prefix_length:]
            elif not self._prefix:
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)


class EnvironmentSource(ConfigSource):
    name = "environment"
    before_provider = True

    def __init__(self, *environment: "tuple[Mapping[str, Any], str] | None") -> None:
        """Each positional arg is a ``(mapping, prefix)`` layer, highest
        priority first; ``None`` is a no-op placeholder a caller can pass
        instead of conditionally omitting a layer. An empty ``prefix``
        reads ``mapping``'s keys as-is; a non-empty one strips it (e.g. the
        real ``os.environ`` with this source's env prefix).
        """
        self._layers = []
        for entry in environment:
            if not entry:
                continue
            mapping, prefix = entry
            self._layers.append(_UnprefixedMapping(mapping, prefix) if prefix else mapping)

    def get(self, key: str) -> "tuple[Any, bool]":
        # Each layer (e.g. the real os.environ) is read live, not snapshotted
        # at construction time -- a test (or a caller) that mutates it after
        # building this source must still see the new value.
        for layer in self._layers:
            if key in layer:
                return (layer[key], True)
        return (MISSING, False)


class RuntimeOverrideSource(ConfigSource):
    name = "runtime-override"
    before_provider = True

    def __init__(self) -> None:
        self._data: "dict[str, Any]" = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def _touch(self) -> None:
        self._revision += 1

    def set(self, key: str, value: "Any") -> None:
        self._data[key] = value
        self._touch()

    def clear(self, key: "str | None" = None) -> None:
        if key is None:
            if self._data:
                self._data.clear()
                self._touch()
            return
        if key in self._data:
            del self._data[key]
            self._touch()

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

    @property
    def revision(self) -> "Any":
        # Delegates to the underlying ConfigStore rather than maintaining a
        # separate counter: multiple PersistentSource instances (different
        # namespaces, or the same namespace built independently) can wrap
        # the very same ConfigStore, and the store's own revision is what
        # lets all of them observe a write made through any one of them.
        return self._store.revision

    def _full(self, key: str) -> str:
        return self._prefix + key

    def get(self, key: str) -> "tuple[Any, bool]":
        full = self._full(key)
        if full in self._store:
            return (self._store.get(full), True)
        return (MISSING, False)

    def set(self, key: str, value: "Any") -> None:
        self._store.set(self._full(key), value)

    def set_many(self, values: "dict[str, Any]") -> None:
        """Write every key in ``values`` in one atomic store flush -- unlike
        calling ``set`` in a loop, a failure partway through never leaves
        some keys written and others not."""
        full_values = {self._full(key): value for key, value in values.items()}
        self._store.save(**full_values)

    def delete(self, key: str) -> bool:
        return self._store.remove(self._full(key))

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
    profile-derived mapping data.

    ``name`` lets two instances (local-file / global-file) report distinct
    ``explain()`` source names instead of both showing up as ``"file"``.
    ``base_path`` is the directory the backing file lives in, used to resolve
    a ``cast="path"`` field's relative value. ``reload_fn``, if given, is
    called by ``reload()`` to atomically replace this source's data and
    base_path (e.g. from a fresh ``ProjectProfile(path...)``)
    without ever leaving a half-updated state.
    """

    before_provider = True

    def __init__(self, data: dict, name: str = "file",
                 reload_fn: "Callable[[], tuple[dict, PathLike | None]] | None" = None,
                 base_path: "PathLike | None" = None) -> None:
        self._data = dict(data)
        self.name = name
        self.base_path = base_path
        self._reload_fn = reload_fn
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def get(self, key: str) -> "tuple[Any, bool]":
        if key in self._data:
            return (self._data[key], True)
        return (MISSING, False)

    def keys(self) -> "list[str]":
        return list(self._data.keys())

    def replace(self, data: dict, base_path: "PathLike | None" = None) -> None:
        self._data = dict(data or {})
        self.base_path = base_path
        self._revision += 1

    def reload(self) -> None:
        if self._reload_fn is None:
            return
        data, base_path = self._reload_fn()
        self.replace(data, base_path=base_path)


class DictSource(ConfigSource):
    """A live read-only view over a mutable mapping."""

    before_provider = True

    def __init__(self, data: dict, name: str = "dict") -> None:
        self._data = data
        self.name = name

    @property
    def revision(self):
        return getattr(self._data, "revision", 0)

    def get(self, key: str) -> "tuple[Any, bool]":
        if key in self._data:
            return (self._data[key], True)
        return (MISSING, False)

    def keys(self) -> "list[str]":
        return list(self._data.keys())


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


def _cast_value(cast: "ConfigType", value: "Any", base_path: "PathLike | None" = None) -> "Any":
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

    A relative ``"path"`` value resolves against ``base_path`` (the
    FileSource's directory, when the value came from one) rather than always
    against the process CWD -- a local-file/global-file value written as
    ``"./data"`` means "relative to that config file", not to wherever the
    command happens to be invoked from.
    """
    if cast == "path":
        expanded = os.path.expanduser(str(value))
        if base_path is not None and not os.path.isabs(expanded):
            return os.path.abspath(os.path.join(str(base_path), expanded))
        return os.path.abspath(expanded)
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
        # key -> (revision_token_at_resolve_time, ResolvedConfig).
        self._memo: "dict[str, tuple[Any, ResolvedConfig]]" = {}

    def clear_memo(self, key: "str | None" = None) -> None:
        """Clear the resolution memo. If key given, invalidate only that key."""
        if key is not None:
            self._memo.pop(key, None)
        else:
            self._memo.clear()

    def revision_token(self) -> "Any":
        """A token that changes whenever this schema or any of its sources
        changes -- including a source shared with a sibling Config (e.g. a
        RuntimeOverrideSource/PersistentSource shared between a manager's
        Config and every per-repository Config), so a write made through
        that sibling is visible here without either Config needing to know
        the other exists. Not guaranteed stable across process runs or
        Source implementations -- comparable only by equality, never parsed."""
        return (self._schema.revision, tuple(source.revision for source in self._sources))

    def get(self, key: str, type: "Any" = None, default: "Any" = MISSING) -> "Any":
        """Convenience: resolve and return the value (for LazyProvider lambdas)."""
        try:
            result = self.resolve(key)
            value = result.value
        except ConfigNotFoundError:
            return default
        if type is not None and value is not MISSING:
            try:
                value = _cast_value(type, value)
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

    def _resolve_cached_provider(self, field: "ConfigField",
                                 compute: "Callable[[], Any]") -> "ResolvedConfig":
        """Resolve a ``cached=True`` provider: reuse a persisted value if one
        exists for this field, else compute it once and persist the result.

        Casts/validates the raw value exactly once, whichever branch is
        taken -- ``compute()`` (e.g. a prompt answer or a freshly-generated
        secret) must not be cast/validated twice just because it is also the
        value being persisted; a non-idempotent cast (or a validator with
        side effects) would otherwise run twice on first resolution but only
        once on every later, persisted-value resolution.

        Raises ``ConfigError`` if the schema has no PersistentSource
        configured (cached=True requires one) -- this surfaces the
        misconfiguration immediately rather than silently recomputing (e.g.
        re-prompting or regenerating a secret) every call.
        """
        persistent = self._persistent_source()
        if persistent is None:
            raise ConfigError(
                "cached=True provider for %r requires a PersistentSource" % (field.name,))
        raw, present = persistent.get(field.name)
        if present:
            value = self._cast_validate(field, raw)
            return ResolvedConfig(value, field, "persistent", raw)
        raw = compute()
        value = self._cast_validate(field, raw)
        persistent.set(field.name, value)
        return ResolvedConfig(value, field, "persistent", raw)

    def _cast_validate(self, field: "ConfigField", raw: "Any", base_path: "PathLike | None" = None) -> "Any":
        value = raw
        if field.cast is not None:
            try:
                value = _cast_value(field.cast, raw, base_path=base_path)
            except Exception as exc:
                shown_raw = redact_config_value(field, raw)
                # The underlying cast exception's OWN message may also embed
                # the raw value (e.g. Python's builtin `int("...")` ValueError
                # repeats its argument verbatim) -- redacting only shown_raw
                # above is not enough, the same secret would still leak back
                # in through this %s.
                detail = exc.__class__.__name__ if field.secret else str(exc)
                raise ConfigCastError("cannot cast %r for %s: %s" % (shown_raw, field.name, detail))
        if field.validator is not None:
            try:
                ok = field.validator(value)
            except Exception as exc:
                # A validator's own exception message may itself embed the
                # raw value (Core has no way to scrub arbitrary third-party
                # exception text) -- for a secret field, drop that message
                # entirely rather than risk forwarding a leaked value.
                detail = exc.__class__.__name__ if (field is not None and field.secret) else str(exc)
                raise ConfigValidationError("validator for %s raised: %s" % (field.name, detail))
            if not ok:
                shown_value = redact_config_value(field, value)
                raise ConfigValidationError("value %r failed validation for %s" % (shown_value, field.name))
        return value

    def validate_raw(self, field: "ConfigField", raw: "Any") -> "Any":
        """Cast+validate ``raw`` against ``field`` without touching any
        Source -- pure validation, no read or write side effect. Lets a
        caller (e.g. cntr's ``config set``) check a batch of raw values
        against every relevant schema BEFORE persisting any of them, so a
        later key's failure can never leave an earlier key already written.
        """
        return self._cast_validate(field, raw)

    def _first_present(self, field: "ConfigField") -> "tuple[ConfigSource | None, Any, bool]":
        for source in self._sources:
            for candidate in self._candidates(field):
                raw, present = source.get(candidate)
                if present:
                    return (source, raw, True)
        return (None, MISSING, False)

    def _first_present_raw(self, key: str) -> "tuple[str | None, Any, bool]":
        for source in self._sources:
            raw, present = source.get(key)
            if present:
                return (source.name, raw, True)
        return (None, MISSING, False)

    def _first_present_before_provider(self, field: "ConfigField") -> "tuple[ConfigSource | None, Any, bool]":
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
        ``before_provider`` source (e.g. ``ManagerConfigSource``) needs no
        change here.
        """
        for source in self._sources:
            if not source.before_provider:
                continue
            for candidate in self._candidates(field):
                raw, present = source.get(candidate)
                if present:
                    return (source, raw, True)
        return (None, MISSING, False)

    def resolve(self, key: str, _stack: "list[str] | None" = None) -> "ResolvedConfig":
        if _stack is None:
            cached = self._memo.get(key)
            if cached is not None:
                cached_token, cached_result = cached
                if cached_token == self.revision_token():
                    return cached_result
                # Some shared source (or this schema) changed since this was
                # memoized -- any other memoized key may have depended on it
                # through an Alias/Lazy/Chain provider, so the whole memo
                # (not just `key`) is stale now.
                self._memo.clear()

        result = self._resolve_inner(key, _stack)

        if _stack is None:
            # Recompute the token AFTER resolving, not before: a cached=True
            # provider can write PersistentSource as a side effect of this
            # very call, which would otherwise be memoized under a
            # now-stale pre-write token.
            self._memo[key] = (self.revision_token(), result)
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
                return self._resolve_cached_provider(field, lambda: provider.func(self))
            value = provider.func(self)
            return ResolvedConfig(self._cast_validate(field, value), field, "lazy", value)

        if isinstance(provider, PromptProvider):
            def compute_prompt():
                try:
                    return self._prompt_value(provider, field)
                except CliError as exc:
                    # No interactive channel to ask the user through (e.g.
                    # non-interactive/no-input mode) -- this provider simply
                    # has no value to offer, the same as an unset Alias
                    # target, not a data/program error. A ChainProvider
                    # containing this as a non-final sub-provider (e.g.
                    # "prompt, else auto-detect") relies on this to fall
                    # through instead of hard-failing whenever no TTY is
                    # attached.
                    raise ConfigPromptError(str(exc)) from exc
            if provider.cached:
                return self._resolve_cached_provider(field, compute_prompt)
            value = compute_prompt()
            return ResolvedConfig(self._cast_validate(field, value), field, "prompt", value)

        if isinstance(provider, ConfirmProvider):
            from ..rich import confirm
            default = provider.default if provider.default is not MISSING else field.default

            def compute_confirm():
                try:
                    return confirm(provider.message or field.name, default=default)
                except CliError as exc:
                    raise ConfigPromptError(str(exc)) from exc
            if provider.cached:
                return self._resolve_cached_provider(field, compute_confirm)
            value = compute_confirm()
            return ResolvedConfig(self._cast_validate(field, value), field, "confirm", value)

        if isinstance(provider, ErrorProvider):
            raise ConfigError(provider.message)

        raise ConfigNotFoundError("unknown provider type: %r" % type(provider).__name__)

    def _resolve_inner(self, key: str, _stack: "list[str] | None" = None) -> "ResolvedConfig":
        field = self._schema.get(key)
        if field is None:
            source_name, raw, present = self._first_present_raw(key)
            if present:
                return ResolvedConfig(raw, None, source_name, raw)
            raise ConfigNotFoundError("config %r is not set and has no default" % (key,))

        # An explicit before_provider-source value (Environment/RuntimeOverride/
        # Persistent/local-file/global-file) always outranks the field's
        # provider (see _first_present_before_provider) -- checked ahead of
        # AliasProvider/ChainProvider/leaf-provider resolution, not merely as
        # their failure fallback.
        override_source, override_raw, override_present = self._first_present_before_provider(field)
        if override_present:
            base_path = override_source.base_path
            return ResolvedConfig(self._cast_validate(field, override_raw, base_path=base_path),
                                   field, override_source.name, override_raw)

        provider = field.provider
        if isinstance(provider, AliasProvider):
            return self._resolve_alias(provider, key, _stack)

        if isinstance(provider, ChainProvider):
            for sub in provider.providers:
                try:
                    return self._try_provider(sub, field, key, _stack)
                except ConfigProviderUnavailable:
                    continue
        elif provider is not None:
            return self._resolve_leaf_provider(provider, field)

        source, raw, present = self._first_present(field)
        if present:
            return ResolvedConfig(self._cast_validate(field, raw, base_path=source.base_path),
                                   field, source.name, raw)
        if field.default is not MISSING:
            return ResolvedConfig(self._cast_validate(field, field.default), field, "default", field.default)
        if field.required:
            raise ConfigNotFoundError("required config %r is not set" % (key,))
        raise ConfigNotFoundError("config %r is not set and has no default" % (key,))

    def _try_provider(self, provider: "Any", field: "ConfigField",
                      key: str, _stack: "list[str] | None" = None) -> "ResolvedConfig":
        """Resolve one ChainProvider sub-provider.

        Only "this sub-provider has no value to offer" is turned into
        ``ConfigProviderUnavailable`` so the chain moves on to the next
        sub-provider: an AliasProvider whose target is unset
        (ConfigNotFoundError), or a Prompt/ConfirmProvider with no
        interactive channel to ask through (ConfigPromptError, e.g. the
        common "prompt, else auto-detect" chain shape). Everything else --
        a cycle, a cast/validator failure, an explicit ErrorProvider, or a
        bare program error -- must propagate immediately instead of being
        silently treated as "try the next one", or a high-priority
        provider's real failure would surface as a lower-priority
        provider's/default's value instead.
        """
        try:
            if isinstance(provider, AliasProvider):
                return self._resolve_alias(provider, key, _stack)
            return self._resolve_leaf_provider(provider, field)
        except (ConfigNotFoundError, ConfigPromptError) as exc:
            raise ConfigProviderUnavailable(str(exc)) from exc

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
                    candidates.append({
                        "source": source.name, "key": candidate,
                        "raw": redact_config_value(field, raw),
                    })
        secret = bool(field.secret) if field is not None else False
        shown_value = redact_config_value(field, resolved.value)
        shown_raw = redact_config_value(field, resolved.raw_value)
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

    @property
    def revision(self) -> "Any":
        """Opaque token, comparable only by equality, that changes whenever
        this Config's schema or any of its sources changes -- including a
        source shared with a sibling Config (see ``ManagerConfigSource``,
        which uses this to observe the owning manager's Config from a
        repository's own Config). Not a stable/parseable structure."""
        return self._resolver.revision_token()

    def define(self, field: "ConfigField") -> "Config":
        self._schema.define(field)
        # A schema change can affect any resolution that depends on this
        # field's aliases/provider/secret/cast, or on another field aliasing
        # it -- clear the whole memo, not just field.key.
        self._resolver.clear_memo()
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
                value = _cast_value(type, value)
            except (TypeError, ValueError) as exc:
                if default is not MISSING:
                    return default
                field = self._schema.get(key)
                shown = redact_config_value(field, value)
                # The underlying cast exception's OWN message may also embed
                # the raw value (e.g. Python's builtin `int("...")` ValueError
                # repeats its argument verbatim) -- for a secret field, drop
                # it entirely rather than risk leaking it back in via %s.
                detail = exc.__class__.__name__ if (field is not None and field.secret) else str(exc)
                raise ConfigCastError("cannot cast %r for %s: %s" % (shown, key, detail))
        return value

    def require(self, key: str, type: "type | None" = None) -> "Any":
        """Resolve a must-exist key; raise ConfigNotFoundError if it is missing."""
        return self.get(key, type=type, default=MISSING)

    def set(self, key: str, value: "Any") -> None:
        runtime = self._find(RuntimeOverrideSource)
        if runtime is None:
            raise ConfigError("no RuntimeOverrideSource configured")
        runtime.set(key, value)
        # Clears every memoized key, not just `key`: a field resolved via
        # AliasProvider/LazyProvider/ChainProvider (or any custom provider
        # reading another field through the resolver) may depend on this
        # value without that dependency being tracked anywhere, so a
        # narrower `clear_memo(key)` would leave those stale.
        self._resolver.clear_memo()

    def persist(self, key: str, value: "Any") -> None:
        persistent = self._find(PersistentSource)
        if persistent is None:
            raise ConfigError("no PersistentSource configured")
        persistent.set(key, value)
        self._resolver.clear_memo()

    def validate_value(self, key: str, raw_value: "Any") -> "Any":
        """Cast+validate ``raw_value`` against this Config's own schema for
        ``key``, without persisting or otherwise touching any Source.

        Lets a caller check a whole batch of raw values up front -- e.g.
        cntr's ``config set``, which must validate every key against every
        repository's own schema before writing any of them (a later key's
        failure must not leave an earlier key already persisted). An
        unknown key passes through as-is, matching how it would resolve.
        """
        field = self._schema.get(key)
        if field is None:
            return raw_value
        return self._resolver.validate_raw(field, raw_value)

    def persist_many(self, values: "dict[str, Any]") -> None:
        """Persist every key in ``values`` in one atomic write, clearing the
        memo once afterward -- unlike calling ``persist`` in a loop, this
        can never leave some keys written and others not if the underlying
        store write itself fails partway (``PersistentSource.set_many``/
        ``ConfigStore.save`` write everything in a single flush)."""
        persistent = self._find(PersistentSource)
        if persistent is None:
            raise ConfigError("no PersistentSource configured")
        persistent.set_many(values)
        self._resolver.clear_memo()

    def unset(self, key: str) -> None:
        runtime = self._find(RuntimeOverrideSource)
        if runtime:
            runtime.clear(key)
        self._resolver.clear_memo()

    def remove(self, key: str) -> None:
        persistent = self._find(PersistentSource)
        if persistent:
            persistent.delete(key)
        self._resolver.clear_memo()

    def explain(self, key: str) -> dict:
        return self._resolver.explain(key)

    def reload(self, clear_runtime: bool = False) -> None:
        if clear_runtime:
            runtime = self._find(RuntimeOverrideSource)
            if runtime:
                runtime.clear()
        for source in self._sources:
            source.reload()
        self._resolver.clear_memo()

    def keys(self) -> "list[str]":
        known = set()
        for field in self._schema.fields():
            known.add(field.name)
        for source in self._sources:
            known.update(source.keys())
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
            self._schema.define(ConfigField.coerce(key, value))
        # See Config.define -- any of the fields just (re)defined may change
        # what an already-memoized resolution should have returned.
        self._resolver.clear_memo()
        return self

    def cast(self, value: "Any", type: "Any" = None) -> "Any":
        if type is None or type is MISSING:
            return value
        return _cast_value(type, value)
