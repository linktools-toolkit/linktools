#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cached=True on Lazy/Prompt/Confirm providers persists the resolved value.

Before this, "compute/ask once, reuse forever" (the old Config.*(cached=True)
descriptor semantics) had no equivalent in the ConfigField/Provider API: a
non-deterministic LazyProvider (e.g. generating a random secret) or a
PromptProvider re-ran on every process, so callers who needed one-time
generation had to reach into ``environ.config_store`` directly and hardcode
the "<namespace>.<FIELD_NAME>" key format themselves. cached=True routes
through the schema's PersistentSource instead, so it Just Works.
"""
import os

import pytest
from linktools.types import MISSING

from linktools.core._config_store import ConfigStore
from linktools.core._config import (
    ConfigField, ConfigResolver, ConfigSchema, ConfigError, ConfigCastError,
    LazyProvider, PromptProvider, ConfirmProvider,
    PersistentSource, RuntimeOverrideSource, EnvironmentSource,
)
from linktools.errors import ConfigValidationError


def _make_resolver(tmp_path, schema, *, persistent_namespace="test"):
    store = ConfigStore(tmp_path / "settings.json")
    sources = [
        EnvironmentSource((os.environ, "")),
        RuntimeOverrideSource(),
        PersistentSource(store, persistent_namespace),
    ]
    return ConfigResolver(schema, sources), store


def test_cached_lazy_computes_once(tmp_path):
    calls = []
    schema = ConfigSchema()
    schema.define(ConfigField(name="SECRET", provider=LazyProvider(
        lambda r: (calls.append(1), "generated")[1], cached=True)))
    resolver, store = _make_resolver(tmp_path, schema)

    v1 = resolver.resolve("SECRET").value
    resolver.clear_memo()
    v2 = resolver.resolve("SECRET").value

    assert v1 == v2 == "generated"
    assert calls == [1]
    assert store.get("test.SECRET") == "generated"


def test_cached_lazy_reuses_across_resolver_instances(tmp_path):
    """A fresh ConfigResolver (simulating a new process) against the same
    store must reuse the persisted value, not recompute."""
    calls = []
    schema = ConfigSchema()
    schema.define(ConfigField(name="SECRET", provider=LazyProvider(
        lambda r: (calls.append(1), "generated")[1], cached=True)))
    store = ConfigStore(tmp_path / "settings.json")
    sources = [PersistentSource(store, "test")]

    ConfigResolver(schema, sources).resolve("SECRET")
    ConfigResolver(schema, sources).resolve("SECRET")

    assert calls == [1]


def test_cached_prompt_asks_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("linktools.rich.prompt", lambda *a, **k: (calls.append(1), "typed")[1])

    schema = ConfigSchema()
    schema.define(ConfigField(name="TOKEN", provider=PromptProvider(cached=True)))
    resolver, store = _make_resolver(tmp_path, schema)

    v1 = resolver.resolve("TOKEN").value
    resolver.clear_memo()
    v2 = resolver.resolve("TOKEN").value

    assert v1 == v2 == "typed"
    assert calls == [1]
    assert store.get("test.TOKEN") == "typed"


def test_cached_confirm_asks_once(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("linktools.rich.confirm", lambda *a, **k: (calls.append(1), True)[1])

    schema = ConfigSchema()
    schema.define(ConfigField(name="ENABLE", provider=ConfirmProvider(default=False, cached=True)))
    resolver, store = _make_resolver(tmp_path, schema)

    v1 = resolver.resolve("ENABLE").value
    resolver.clear_memo()
    v2 = resolver.resolve("ENABLE").value

    assert v1 == v2 is True
    assert calls == [1]
    assert store.get("test.ENABLE") is True


def test_cached_provider_without_persistent_source_raises(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="SECRET", provider=LazyProvider(lambda r: "x", cached=True)))
    resolver = ConfigResolver(schema, [])

    with pytest.raises(ConfigError, match="PersistentSource"):
        resolver.resolve("SECRET")


def test_cached_provider_does_not_persist_value_that_fails_cast(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(
        name="PORT", cast=int,
        provider=LazyProvider(lambda r: "not-a-port", cached=True),
    ))
    resolver, store = _make_resolver(tmp_path, schema)

    with pytest.raises(ConfigCastError):
        resolver.resolve("PORT")
    assert store.get("test.PORT") is MISSING


def test_cached_lazy_inside_chain_provider(tmp_path):
    """cached=True must also work nested inside a ChainProvider (the common
    real-world shape: alias fallback -> cached generate)."""
    from linktools.core._config import ChainProvider, AliasProvider

    calls = []
    schema = ConfigSchema()
    schema.define(ConfigField(name="SECRET", provider=ChainProvider(
        AliasProvider("OTHER_KEY"),
        LazyProvider(lambda r: (calls.append(1), "generated")[1], cached=True),
    )))
    resolver, store = _make_resolver(tmp_path, schema)

    v1 = resolver.resolve("SECRET").value
    resolver.clear_memo()
    v2 = resolver.resolve("SECRET").value

    assert v1 == v2 == "generated"
    assert calls == [1]
    assert store.get("test.SECRET") == "generated"


def test_cached_lazy_non_idempotent_cast_runs_once_on_first_resolve(tmp_path):
    """A non-idempotent cast (e.g. one that prefixes a string) must run
    exactly once against the freshly-computed raw value -- not once to
    build the value and again when the resolved value is returned."""
    cast_calls = []

    def add_prefix(value):
        cast_calls.append(value)
        return "prefix:" + value

    schema = ConfigSchema()
    schema.define(ConfigField(
        name="SECRET", cast=add_prefix,
        provider=LazyProvider(lambda r: "value", cached=True),
    ))
    resolver, store = _make_resolver(tmp_path, schema)

    result = resolver.resolve("SECRET").value

    assert result == "prefix:value"
    assert cast_calls == ["value"]
    assert store.get("test.SECRET") == "prefix:value"


def test_cached_provider_validator_runs_once_on_first_resolve(tmp_path):
    validator_calls = []

    def counting_validator(value):
        validator_calls.append(value)
        return True

    schema = ConfigSchema()
    schema.define(ConfigField(
        name="SECRET", validator=counting_validator,
        provider=LazyProvider(lambda r: "value", cached=True),
    ))
    resolver, _ = _make_resolver(tmp_path, schema)

    resolver.resolve("SECRET")

    assert validator_calls == ["value"]


def test_cached_provider_reads_persisted_value_cast_once(tmp_path):
    """Once a value is persisted, a fresh resolver reading it back must also
    cast/validate exactly once."""
    cast_calls = []

    def counting_cast(value):
        cast_calls.append(value)
        return value

    schema = ConfigSchema()
    schema.define(ConfigField(
        name="SECRET", cast=counting_cast,
        provider=LazyProvider(lambda r: "value", cached=True),
    ))
    store = ConfigStore(tmp_path / "settings.json")
    ConfigResolver(schema, [PersistentSource(store, "test")]).resolve("SECRET")

    del cast_calls[:]
    fresh_resolver = ConfigResolver(schema, [PersistentSource(store, "test")])
    result = fresh_resolver.resolve("SECRET").value

    assert result == "value"
    assert cast_calls == ["value"]


def test_cached_provider_does_not_persist_value_that_fails_validator(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(
        name="TOKEN", validator=lambda v: False,
        provider=LazyProvider(lambda r: "bad", cached=True),
    ))
    resolver, store = _make_resolver(tmp_path, schema)

    with pytest.raises(ConfigValidationError):
        resolver.resolve("TOKEN")
    assert store.get("test.TOKEN") is MISSING


def test_non_cached_lazy_still_recomputes_every_time(tmp_path):
    """Regression guard: default (cached=False) behavior must be unchanged."""
    calls = []
    schema = ConfigSchema()
    schema.define(ConfigField(name="VALUE", provider=LazyProvider(
        lambda r: (calls.append(1), len(calls))[1])))
    resolver, _ = _make_resolver(tmp_path, schema)

    v1 = resolver.resolve("VALUE").value
    resolver.clear_memo()
    v2 = resolver.resolve("VALUE").value

    assert v1 == 1
    assert v2 == 2
    assert len(calls) == 2
