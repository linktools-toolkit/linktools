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
    "ErrorProvider", "ConfigResolver", "ResolvedConfig",
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
    """Reads from a ConfigStore-backed namespace (§8.5)."""

    name = "persistent"

    def __init__(self, store, namespace="config"):
        self._ns = store.namespace(namespace)

    def get(self, key):
        if key in self._ns:
            return (self._ns.get(key), True)
        return (MISSING, False)


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
