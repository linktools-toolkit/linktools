#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""New Config model: Schema / Field / Sources / Resolver (spec §8).

Standalone build (Phase 3 PR 07). The legacy ``Config`` (core/_config.py) stays
in place until cntr's DSL migrates to this model (PR 08); the two do not share
state. This module formalises the §8 contract:

* §8.2 source precedence (env > runtime-override > persistent > file > default)
  is encoded by the order of the ``sources`` list passed to ConfigResolver --
  first present wins.
* §8.3 ConfigField carries name/default/cast/validator/aliases/required/secret/
  description/deprecated/provider.
* §8.4 ValueProvider (AliasProvider here) is stateless -- resolution memo lives
  on the resolver, not the provider.
* §8.9 explain() reports resolved value, selected source, raw value, all
  candidates; secret fields are masked.
* §8.10 Alias cycles raise ConfigCycleError with the full chain.
* §8.12 raises the Config* error subtree.
"""

import os
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..errors import (
    ConfigCastError, ConfigCycleError, ConfigError, ConfigNotFoundError,
    ConfigValidationError,
)
from ..types import MISSING, MissingType

__all__ = [
    "ConfigField", "ConfigSchema",
    "ConfigSource", "EnvironmentSource", "RuntimeOverrideSource",
    "PersistentSource", "FileSource", "DefaultSource",
    "AliasProvider", "LazyProvider", "PromptProvider", "ConfirmProvider",
    "ErrorProvider", "ChainProvider",
    "ConfigResolver", "ResolvedConfig", "Config",
]


# --------------------------------------------------------------------------- #
# Providers (§8.4) -- stateless
# --------------------------------------------------------------------------- #

class AliasProvider(object):
    """Resolve a field's value from another field's value (§8.4 AliasValue)."""

    def __init__(self, target):
        # type: (str) -> None
        self.target = target


class LazyProvider(object):
    """Compute a value from a callable (spec §8.4 LazyValue). Stateless."""

    def __init__(self, func):
        # type: (Callable[[Any], Any]) -> None
        self.func = func


class PromptProvider(object):
    """Prompt the user for a value (spec §8.4 PromptValue). Stateless."""

    def __init__(self, message=None, default=MISSING, password=False, choices=None):
        # type: (str, Any, bool, Optional[list]) -> None
        self.message = message
        self.default = default
        self.password = password
        self.choices = choices


class ConfirmProvider(object):
    """Ask the user for a yes/no confirmation (spec §8.4 ConfirmValue)."""

    def __init__(self, message=None, default=MISSING):
        # type: (str, Any) -> None
        self.message = message
        self.default = default


class ErrorProvider(object):
    """Always raise ConfigError when resolved (spec §8.4 ErrorValue)."""

    def __init__(self, message):
        # type: (str) -> None
        self.message = message


class ChainProvider(object):
    """Try multiple providers in order until one yields a value (v2 §3.7).

    Replaces the old Config DSL's ``|`` operator: ``Prompt() | Lazy(fn) |
    default`` becomes ``ChainProvider(PromptProvider(...), LazyProvider(fn))``
    with ``default`` on the ConfigField.
    """

    def __init__(self, *providers):
        # type: (*Any) -> None
        self.providers = list(providers)


# --------------------------------------------------------------------------- #
# Field / Schema (§8.3)
# --------------------------------------------------------------------------- #

class ConfigField(object):
    def __init__(self, name, default=MISSING, cast=None, validator=None,
                 aliases=(), required=False, secret=False, description="",
                 deprecated=False, provider=None):
        # type: (str, Any, Optional[Callable], Optional[Callable], Sequence[str], bool, bool, str, bool, Any) -> None
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


class ConfigSchema(object):
    """A set of named ConfigFields with alias indexing."""

    def __init__(self):
        self._fields = {}  # type: dict
        self._alias_to_name = {}  # type: dict

    def define(self, field):
        # type: (ConfigField) -> "ConfigSchema"
        self._fields[field.name] = field
        for alias in field.aliases:
            self._alias_to_name[alias] = field.name
        return self

    def get(self, name):
        # type: (str) -> Optional[ConfigField]
        if name in self._fields:
            return self._fields[name]
        target = self._alias_to_name.get(name)
        if target is not None:
            return self._fields.get(target)
        return None

    def __contains__(self, name):
        # type: (str) -> bool
        return self.get(name) is not None


# --------------------------------------------------------------------------- #
# Sources (§8.2)
# --------------------------------------------------------------------------- #

class ConfigSource(object):
    name = "source"

    def get(self, key):
        # type: (str) -> Tuple[Any, bool]
        raise NotImplementedError


class EnvironmentSource(ConfigSource):
    name = "environment"

    def __init__(self, prefix=""):
        # type: (str) -> None
        self._prefix = prefix

    def get(self, key):
        value = os.environ.get(self._prefix + key, MISSING)
        if value is MISSING:
            return (MISSING, False)
        return (value, True)


class RuntimeOverrideSource(ConfigSource):
    name = "runtime-override"

    def __init__(self):
        self._data = {}  # type: dict

    def set(self, key, value):
        # type: (str, Any) -> None
        self._data[key] = value

    def clear(self, key=None):
        # type: (Optional[str]) -> None
        if key is None:
            self._data.clear()
        else:
            self._data.pop(key, None)

    def get(self, key):
        if key in self._data:
            return (self._data[key], True)
        return (MISSING, False)


class PersistentSource(ConfigSource):
    """Reads/writes from a ConfigStore (§8.5), key-prefixed by namespace."""

    name = "persistent"

    def __init__(self, store, namespace="config"):
        self._store = store
        self._prefix = (namespace + ".") if namespace else ""

    def _full(self, key):
        return self._prefix + key

    def get(self, key):
        full = self._full(key)
        if full in self._store:
            return (self._store.get(full), True)
        return (MISSING, False)

    def set(self, key, value):
        self._store.set(self._full(key), value)

    def delete(self, key):
        return self._store.delete(self._full(key))


class FileSource(ConfigSource):
    name = "file"

    def __init__(self, data):
        # type: (dict) -> None
        self._data = dict(data)

    def get(self, key):
        if key in self._data:
            return (self._data[key], True)
        return (MISSING, False)


class DefaultSource(ConfigSource):
    name = "default"

    def __init__(self, schema):
        # type: (ConfigSchema) -> None
        self._schema = schema

    def get(self, key):
        field = self._schema.get(key)
        if field is not None and field.default is not MISSING:
            return (field.default, True)
        return (MISSING, False)


# --------------------------------------------------------------------------- #
# Resolver (§8.2, §8.9, §8.10)
# --------------------------------------------------------------------------- #

class ResolvedConfig(object):
    def __init__(self, value, field, source_name, raw_value):
        # type: (Any, ConfigField, str, Any) -> None
        self.value = value
        self.field = field
        self.source_name = source_name
        self.raw_value = raw_value


class ConfigResolver(object):
    def __init__(self, schema, sources):
        # type: (ConfigSchema, Sequence[ConfigSource]) -> None
        self._schema = schema
        self._sources = list(sources)
        self._memo = {}  # type: Dict[str, ResolvedConfig]

    def clear_memo(self):
        # type: () -> None
        """Clear the resolution memo (§3.8 reload)."""
        self._memo.clear()

    def get(self, key, type=None, default=MISSING):
        # type: (str, Any, Any) -> Any
        """Convenience: resolve and return the value (for LazyProvider lambdas).

        LazyProvider receives the resolver as its argument; this method lets
        lambdas do ``r.get("OTHER_KEY")`` without needing the full Config wrapper.
        """
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

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _candidates(field):
        # type: (ConfigField) -> Tuple[str, ...]
        return (field.name,) + field.aliases

    def _cast_validate(self, field, raw):
        value = raw
        if field.cast is not None:
            try:
                value = field.cast(raw)
            except Exception as exc:
                raise ConfigCastError(
                    "cannot cast %r for %s: %s" % (raw, field.name, exc))
        if field.validator is not None:
            try:
                ok = field.validator(value)
            except Exception as exc:
                raise ConfigValidationError(
                    "validator for %s raised: %s" % (field.name, exc))
            if not ok:
                raise ConfigValidationError(
                    "value %r failed validation for %s" % (value, field.name))
        return value

    def _first_present(self, field):
        # type: (ConfigField) -> Tuple[Optional[str], Any, bool]
        """Return (source_name, raw, present) for the first source that has any
        candidate key; the resolver's source order encodes §8.2 precedence."""
        for source in self._sources:
            for candidate in self._candidates(field):
                raw, present = source.get(candidate)
                if present:
                    return (source.name, raw, True)
        return (None, MISSING, False)

    # -- public ------------------------------------------------------------

    def resolve(self, key, _stack=None):
        # type: (str, Optional[List[str]]) -> ResolvedConfig
        # Memo: cache top-level resolutions so interactive prompts (PromptProvider)
        # only ask once per session (§3.7/§3.8). Skip memo during recursive
        # resolution (alias/lazy chains with _stack) to avoid partial caching.
        if _stack is None and key in self._memo:
            return self._memo[key]
        result = self._resolve_inner(key, _stack)
        if _stack is None:
            self._memo[key] = result
        return result

    def _resolve_inner(self, key, _stack=None):
        # type: (str, Optional[List[str]]) -> ResolvedConfig
        field = self._schema.get(key)
        if field is None:
            raise ConfigNotFoundError("unknown config key %r" % (key,))

        # §8.4 / §8.10: providers re-target or compute the value.
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
            value = prompt(
                provider.message or field.name,
                default=provider.default,
                password=provider.password,
                choices=provider.choices,
            )
            return ResolvedConfig(self._cast_validate(field, value), field, "prompt", value)

        if isinstance(provider, ConfirmProvider):
            from ..rich import confirm
            value = confirm(
                provider.message or field.name,
                default=provider.default,
            )
            return ResolvedConfig(self._cast_validate(field, value), field, "confirm", value)

        if isinstance(provider, ErrorProvider):
            raise ConfigError(provider.message)

        if isinstance(provider, ChainProvider):
            # Try each sub-provider in order until one yields a value.
            for sub in provider.providers:
                try:
                    return self._try_provider(sub, field, key, _stack)
                except Exception:
                    continue
            # All sub-providers failed; fall through to sources/default.

        source_name, raw, present = self._first_present(field)
        if present:
            return ResolvedConfig(self._cast_validate(field, raw), field, source_name, raw)
        # No source provided -> the field's own default (DefaultSource covers the
        # normal case; this handles a schema whose resolver has no DefaultSource).
        if field.default is not MISSING:
            return ResolvedConfig(self._cast_validate(field, field.default), field, "default", field.default)
        if field.required:
            raise ConfigNotFoundError("required config %r is not set" % (key,))
        raise ConfigNotFoundError("config %r is not set and has no default" % (key,))

    def _try_provider(self, provider, field, key, _stack=None):
        # type: (Any, ConfigField, str, Optional[List[str]]) -> ResolvedConfig
        """Dispatch a single provider (used by ChainProvider and resolve)."""
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
            value = prompt(
                provider.message or field.name,
                default=provider.default,
                password=provider.password,
                choices=provider.choices,
            )
            return ResolvedConfig(self._cast_validate(field, value), field, "prompt", value)

        if isinstance(provider, ConfirmProvider):
            from ..rich import confirm
            value = confirm(provider.message or field.name, default=provider.default)
            return ResolvedConfig(self._cast_validate(field, value), field, "confirm", value)

        if isinstance(provider, ErrorProvider):
            raise ConfigError(provider.message)

        raise ConfigNotFoundError("unknown provider type: %r" % type(provider).__name__)

    def explain(self, key):
        # type: (str) -> dict
        resolved = self.resolve(key)
        field = resolved.field
        candidates = []  # type: List[dict]
        for source in self._sources:
            for candidate in self._candidates(field):
                raw, present = source.get(candidate)
                if present:
                    candidates.append({"source": source.name, "key": candidate, "raw": raw})
        # §8.9: secret fields do not expose the raw value.
        if field.secret:
            shown_value = "***"
            shown_raw = "***"
        else:
            shown_value = resolved.value
            shown_raw = resolved.raw_value
        return {
            "resolved_value": shown_value,
            "selected_source": resolved.source_name,
            "raw_value": shown_raw,
            "all_candidates": candidates,
            "secret": field.secret,
            "deprecated": field.deprecated,
            "description": field.description,
        }


# --------------------------------------------------------------------------- #
# Config: user-facing API wrapping ConfigResolver (v2 §3.5)
# --------------------------------------------------------------------------- #

class Config(object):
    """The user-facing config, backed by ConfigResolver (v2 §3.5).

    Provides get/set/persist/remove/unset/explain/reload — the v2 contract.
    ``set`` writes to RuntimeOverrideSource (current process); ``persist`` writes
    to PersistentSource (user-editable JSON). ``get`` resolves through the full
    source chain with §8.2 precedence.
    """

    def __init__(self, environ, schema, sources):
        # type: (Any, ConfigSchema, Sequence[ConfigSource]) -> None
        self._environ = environ
        self._schema = schema
        self._sources = list(sources)
        self._resolver = ConfigResolver(schema, self._sources)

    @property
    def schema(self):
        # type: () -> ConfigSchema
        return self._schema

    def define(self, field):
        # type: (ConfigField) -> "Config"
        self._schema.define(field)
        return self

    def _find(self, source_class):
        for s in self._sources:
            if isinstance(s, source_class):
                return s
        return None

    def get(self, key, type=None, default=MISSING):
        # type: (str, Optional[type], Any) -> Any
        try:
            result = self._resolver.resolve(key)
            value = result.value
        except ConfigNotFoundError:
            return default
        if type is not None and value is not MISSING:
            try:
                value = type(value)
            except (TypeError, ValueError) as exc:
                if default is not MISSING:
                    return default
                raise ConfigCastError("cannot cast %r for %s: %s" % (value, key, exc))
        return value

    def set(self, key, value):
        # type: (str, Any) -> None
        """Runtime override (v2 §3.5: current process only)."""
        runtime = self._find(RuntimeOverrideSource)
        if runtime is None:
            raise ConfigError("no RuntimeOverrideSource configured")
        runtime.set(key, value)

    def persist(self, key, value):
        # type: (str, Any) -> None
        """Write to the persistent user store (v2 §3.5)."""
        persistent = self._find(PersistentSource)
        if persistent is None:
            raise ConfigError("no PersistentSource configured")
        persistent.set(key, value)

    def unset(self, key):
        # type: (str) -> None
        """Remove a runtime override (v2 §3.5)."""
        runtime = self._find(RuntimeOverrideSource)
        if runtime:
            runtime.clear(key)

    def remove(self, key):
        # type: (str) -> None
        """Remove a persistent value (v2 §3.5)."""
        persistent = self._find(PersistentSource)
        if persistent:
            persistent.delete(key)

    def explain(self, key):
        # type: (str) -> dict
        return self._resolver.explain(key)

    def reload(self, clear_runtime=False):
        # type: (bool) -> None
        """Re-read sources + clear memo (v2 §3.8)."""
        if clear_runtime:
            runtime = self._find(RuntimeOverrideSource)
            if runtime:
                runtime.clear()
        self._resolver.clear_memo()

    def keys(self):
        # type: () -> List[str]
        known = set()
        for name in self._schema._fields:
            known.add(name)
        for source in self._sources:
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

    def update_defaults(self, **kwargs):
        # type: (**Any) -> "Config"
        """Register multiple config defaults at once (cntr compatibility).

        Accepts a mix of ConfigField, ChainProvider/Provider, or plain values.
        Plain values become ConfigField(name=key, default=value).
        Old ConfigProperty chains (Config.Prompt|Lazy|Alias) are also detected
        and wrapped for short-term coexistence during cntr migration.
        """
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
                # It's a new provider (ChainProvider/AliasProvider/Lazy/etc.)
                self._schema.define(ConfigField(name=key, provider=value))
            elif hasattr(value, "get") and hasattr(value, "load") and \
                    hasattr(value, "_default"):
                # Old ConfigProperty chain (Config.Prompt|Lazy|Alias|...).
                # Extract the default from the chain tail and use it.
                default = getattr(value, "_default", MISSING)
                if default is MISSING:
                    default = None
                self._schema.define(ConfigField(name=key, default=default))
            else:
                self._schema.define(ConfigField(name=key, default=value))
        return self

    def cast(self, value, type=None):
        # type: (Any, Any) -> Any
        """Standalone type cast (cntr compatibility)."""
        if type is None or type is MISSING:
            return value
        if type == "path":
            import os
            return os.path.abspath(os.path.expanduser(str(value)))
        if callable(type):
            return type(value)
        return value
