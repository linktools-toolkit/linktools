#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ChainProvider must only fall through to the next sub-provider when a
sub-provider genuinely has no value to offer (ConfigNotFoundError, e.g. an
AliasProvider whose target is unset). A cycle, a cast/validator failure, an
ErrorProvider, or a bare program error must propagate immediately instead of
being silently swallowed and treated as "try the next provider" -- that
fail-open behavior lets a high-priority provider's real error surface as a
lower-priority provider's/default's value instead.
"""
import os

import pytest

from linktools.core._config_store import ConfigStore
from linktools.core._config import (
    ConfigField, ConfigResolver, ConfigSchema,
    AliasProvider, ChainProvider, ErrorProvider, LazyProvider,
    PersistentSource, RuntimeOverrideSource, EnvironmentSource,
)
from linktools.errors import (
    ConfigCastError, ConfigCycleError, ConfigError, ConfigValidationError,
)


def _make_resolver(tmp_path, schema):
    store = ConfigStore(tmp_path / "settings.json")
    sources = [
        EnvironmentSource((os.environ, "")),
        RuntimeOverrideSource(),
        PersistentSource(store, "test"),
    ]
    return ConfigResolver(schema, sources)


def test_alias_missing_target_falls_through_to_next_provider(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(
        name="FOO", default="fallback",
        provider=ChainProvider(AliasProvider("MISSING_TARGET")),
    ))
    resolver = _make_resolver(tmp_path, schema)

    assert resolver.resolve("FOO").value == "fallback"


def test_alias_cycle_fails_immediately_not_swallowed(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="A", default="unused",
                               provider=ChainProvider(AliasProvider("B"))))
    schema.define(ConfigField(name="B", default="unused",
                               provider=ChainProvider(AliasProvider("A"))))
    resolver = _make_resolver(tmp_path, schema)

    with pytest.raises(ConfigCycleError):
        resolver.resolve("A")


def test_first_provider_cast_failure_fails_immediately(tmp_path):
    calls = []

    def second(_r):
        calls.append(1)
        return "unused"

    schema = ConfigSchema()
    schema.define(ConfigField(
        name="PORT", cast=int, default=1234,
        provider=ChainProvider(
            LazyProvider(lambda r: "not-a-number"),
            LazyProvider(second),
        ),
    ))
    resolver = _make_resolver(tmp_path, schema)

    with pytest.raises(ConfigCastError):
        resolver.resolve("PORT")
    assert calls == []


def test_first_provider_validator_failure_fails_immediately(tmp_path):
    calls = []

    def second(_r):
        calls.append(1)
        return "ok"

    schema = ConfigSchema()
    schema.define(ConfigField(
        name="TOKEN", validator=lambda v: False, default="unused",
        provider=ChainProvider(
            LazyProvider(lambda r: "bad"),
            LazyProvider(second),
        ),
    ))
    resolver = _make_resolver(tmp_path, schema)

    with pytest.raises(ConfigValidationError):
        resolver.resolve("TOKEN")
    assert calls == []


def test_lazy_provider_program_error_fails_immediately(tmp_path):
    calls = []

    def second(_r):
        calls.append(1)
        return "unused"

    def boom(_r):
        raise RuntimeError("boom")

    schema = ConfigSchema()
    schema.define(ConfigField(
        name="FOO", default="unused",
        provider=ChainProvider(LazyProvider(boom), LazyProvider(second)),
    ))
    resolver = _make_resolver(tmp_path, schema)

    with pytest.raises(RuntimeError, match="boom"):
        resolver.resolve("FOO")
    assert calls == []


def test_error_provider_preserves_custom_message(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(
        name="FOO", default="unused",
        provider=ChainProvider(ErrorProvider("explicit failure")),
    ))
    resolver = _make_resolver(tmp_path, schema)

    with pytest.raises(ConfigError, match="explicit failure"):
        resolver.resolve("FOO")


def test_successful_first_provider_does_not_call_later_providers(tmp_path):
    calls = []

    def second(_r):
        calls.append(1)
        return "unused"

    schema = ConfigSchema()
    schema.define(ConfigField(
        name="FOO",
        provider=ChainProvider(LazyProvider(lambda r: "first"), LazyProvider(second)),
    ))
    resolver = _make_resolver(tmp_path, schema)

    assert resolver.resolve("FOO").value == "first"
    assert calls == []
