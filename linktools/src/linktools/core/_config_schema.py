#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Config model: Schema / Field / Sources / Resolver / Config."""

import os
from typing import TYPE_CHECKING

from ..errors import (
    ConfigCastError, ConfigCycleError, ConfigError, ConfigNotFoundError,
    ConfigValidationError,
)
from ..types import MISSING

if TYPE_CHECKING:
    from typing import Any, Callable, Sequence

__all__ = [
    "ConfigField", "ConfigSchema",
    "ConfigSource", "EnvironmentSource", "RuntimeOverrideSource",
    "PersistentSource", "FileSource", "DefaultSource",
    "AliasProvider", "LazyProvider", "PromptProvider", "ConfirmProvider",
    "ErrorProvider", "ChainProvider",
    "ConfigResolver", "ResolvedConfig", "Config",
]


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
        resolved = self.resolve(key)
        field = resolved.field
        candidates: "list[dict]" = []
        for source in self._sources:
            for candidate in self._candidates(field):
                raw, present = source.get(candidate)
                if present:
                    candidates.append({"source": source.name, "key": candidate, "raw": raw})
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
            return default
        if type is not None and value is not MISSING:
            try:
                value = type(value)
            except (TypeError, ValueError) as exc:
                if default is not MISSING:
                    return default
                raise ConfigCastError("cannot cast %r for %s: %s" % (value, key, exc))
        return value

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
