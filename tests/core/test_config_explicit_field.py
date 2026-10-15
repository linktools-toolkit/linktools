#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit fields use their own contracts without changing named resolution."""

from typing import TYPE_CHECKING

import pytest

from linktools.core import (
    Config, ConfigField, ConfigResolver, ConfigSchema, ConfigStore,
    ConfirmProvider, DefaultSource, EnvironmentSource, FileSource, LazyProvider,
    PersistentSource, PromptProvider, RuntimeOverrideSource,
)
from linktools.errors import ConfigCastError, ConfigNotFoundError, ConfigValidationError
from linktools.types import MISSING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _config(*fields: ConfigField, values: "dict[str, Any] | None" = None) -> Config:
    schema = ConfigSchema()
    for field in fields:
        schema.define(field)
    runtime = RuntimeOverrideSource()
    for key, value in (values or {}).items():
        runtime.set(key, value)
    return Config(None, schema, [runtime, DefaultSource(schema)])


@pytest.mark.parametrize("name", ["VALUE", None, ""])
def test_explicit_fields_with_same_name_resolve_independently(name: "str | None") -> None:
    config = _config()
    calls = []
    first = ConfigField(name, provider=LazyProvider(lambda resolver: calls.append("first") or "first"))
    second = ConfigField(name, provider=LazyProvider(lambda resolver: calls.append("second") or "second"))
    revision = config.schema.revision

    assert config.get(first) == "first"
    assert config.get(second) == "second"
    assert config.get(first) == "first"
    assert calls == ["first", "second", "first"]
    assert config.schema.fields() == []
    assert config.schema.revision == revision
    with pytest.raises(ConfigNotFoundError):
        config.get(name or "")


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("field_first", [False, True])
def test_explicit_field_and_name_reads_do_not_share_casts(registered: bool, field_first: bool) -> None:
    schema_field = ConfigField("VALUE", cast=lambda value: "schema:" + value)
    config = _config(*(schema_field,) if registered else (), values={"VALUE": "7"})
    field = ConfigField("VALUE", cast=int)
    expected_named = "schema:7" if registered else "7"
    reads = [(field, 7), ("VALUE", expected_named)]
    if not field_first:
        reads.reverse()
    revision = config.schema.revision

    for key, expected in reads * 2:
        assert config.get(key) == expected
    assert config.schema.get("VALUE") is (schema_field if registered else None)
    assert config.schema.revision == revision


def test_explicit_field_reads_preserve_existing_named_memo() -> None:
    calls = []
    schema_field = ConfigField("VALUE", provider=LazyProvider(
        lambda resolver: calls.append(True) or len(calls)))
    config = _config(schema_field)
    field = ConfigField("VALUE", provider=LazyProvider(lambda resolver: "explicit"))

    assert config.get("VALUE") == 1
    assert config.get(field) == "explicit"
    assert config.get("VALUE") == 1
    assert calls == [True]


@pytest.mark.parametrize("with_provider", [False, True])
def test_explicit_field_preserves_schema_default_source_priority(with_provider: bool) -> None:
    schema_field = ConfigField("VALUE", default="41", cast=lambda value: "schema:" + value)
    config = _config(schema_field)
    field = ConfigField("VALUE", default="42", cast=int,
                        provider=LazyProvider(lambda resolver: "43") if with_provider else None)

    assert config.get(field) == (43 if with_provider else 41)
    assert config.get("VALUE") == "schema:41"
    assert config.get(field) == (43 if with_provider else 41)
    assert config.schema.get("VALUE") is schema_field


@pytest.mark.parametrize("winner", ["environment", "runtime", "persistent", "file"])
def test_explicit_field_existing_sources_outrank_provider(tmp_path: "Path", winner: str) -> None:
    schema = ConfigSchema().define(ConfigField("VALUE", default="15"))
    store = ConfigStore(tmp_path / "settings.json")
    runtime = RuntimeOverrideSource()
    if winner in ("environment", "runtime"):
        runtime.set("VALUE", "12")
    if winner != "file":
        store.set("test.VALUE", "13")
    environment = {"VALUE": "11"} if winner == "environment" else {}
    config = Config(None, schema, [
        EnvironmentSource((environment, "")), runtime, PersistentSource(store, "test"),
        FileSource({"VALUE": "14"}), DefaultSource(schema),
    ])

    def unexpected_provider(resolver: ConfigResolver) -> str:
        raise AssertionError("existing source value must win")

    field = ConfigField("VALUE", default="16", cast=int, provider=LazyProvider(unexpected_provider))
    assert config.get(field) == {"environment": 11, "runtime": 12, "persistent": 13, "file": 14}[winner]


def test_explicit_field_aliases_do_not_register_schema_names() -> None:
    config = _config(values={"OLD_VALUE": "17"})
    field = ConfigField("VALUE", aliases=("OLD_VALUE",), cast=int)

    assert config.get(field) == 17
    assert config.get("OLD_VALUE") == "17"
    with pytest.raises(ConfigNotFoundError):
        config.get("VALUE")
    assert config.schema.fields() == []


@pytest.mark.parametrize("prime_with_field", [False, True])
@pytest.mark.parametrize("failure", ["cast", "validator_false", "validator_raises"])
def test_explicit_field_errors_cannot_reuse_a_successful_memo(prime_with_field: bool, failure: str) -> None:
    config = _config(values={"VALUE": "invalid"})
    assert config.get(ConfigField("VALUE") if prime_with_field else "VALUE") == "invalid"

    def broken_validator(value: str) -> bool:
        raise ValueError("rejected")

    fields = {
        "cast": ConfigField("VALUE", cast=int),
        "validator_false": ConfigField("VALUE", validator=lambda value: False),
        "validator_raises": ConfigField("VALUE", validator=broken_validator),
    }
    error = ConfigCastError if failure == "cast" else ConfigValidationError
    with pytest.raises(error):
        config.get(fields[failure], default="outer-fallback")
    assert config.get("VALUE") == "invalid"


def test_explicit_field_default_cast_and_validation_precede_outer_type() -> None:
    validated = []
    field = ConfigField("VALUE", default="7", cast=int,
                        validator=lambda value: validated.append(value) or value == 7)
    value = _config().get(field, type=str, default="outer-fallback")

    assert value == "7"
    assert validated == [7]


def test_explicit_field_outer_fallback_preserves_get_semantics() -> None:
    config = _config()
    fallback = object()

    assert config.get(ConfigField("MISSING"), type=int, default=fallback) is fallback
    assert config.get(ConfigField("NONE", default=None), default=fallback) is None
    field = ConfigField("VALUE", default="invalid")
    assert config.get(field, type=int, default=fallback) is fallback
    with pytest.raises(ConfigCastError):
        config.get(field, type=int)
    with pytest.raises(ConfigNotFoundError):
        config.get(ConfigField("MISSING"))


def test_explicit_mutable_field_rechecks_default_cast_provider_and_validator() -> None:
    config = _config()
    field = ConfigField("VALUE", default="7", cast=int)
    assert config.get(field) == 7

    field.default = "8"
    assert config.get(field) == 8
    field.cast = str
    assert config.get(field) == "8"
    field.provider = LazyProvider(lambda resolver: "9")
    assert config.get(field) == "9"
    field.validator = lambda value: False
    with pytest.raises(ConfigValidationError):
        config.get(field)


def test_explicit_cached_provider_persists_across_reads_and_configs(tmp_path: "Path") -> None:
    calls = []
    schema = ConfigSchema()
    path = tmp_path / "settings.json"
    store = ConfigStore(path)
    field = ConfigField("VALUE", cast=int, provider=LazyProvider(
        lambda resolver: calls.append(True) or "19", cached=True))
    config = Config(None, schema, [PersistentSource(store, "test"), DefaultSource(schema)])

    assert config.get(field) == config.get(field) == 19
    assert store.get("test.VALUE") == 19
    saved = path.read_bytes()
    fresh = Config(None, schema, [PersistentSource(ConfigStore(path), "test"), DefaultSource(schema)])
    with Config.read_only_resolution():
        assert fresh.get(field) == 19
    assert path.read_bytes() == saved
    assert calls == [True]
    assert schema.fields() == []


@pytest.mark.parametrize("provider_kind", ["lazy", "prompt", "confirm"])
def test_explicit_cached_provider_read_only_blocks_creation_and_outer_fallback(
        tmp_path: "Path", provider_kind: str) -> None:
    calls = []
    store = ConfigStore(tmp_path / "settings.json")
    schema = ConfigSchema().define(ConfigField("VALUE", default="schema-default"))
    config = Config(None, schema, [PersistentSource(store, "test"), DefaultSource(schema)])
    providers = {
        "lazy": LazyProvider(lambda resolver: calls.append(True) or "generated", cached=True),
        "prompt": PromptProvider(cached=True),
        "confirm": ConfirmProvider(cached=True),
    }
    field = ConfigField("VALUE", provider=providers[provider_kind], default="field-default")
    assert config.get("VALUE") == "schema-default"
    before = list(tmp_path.iterdir())

    with Config.read_only_resolution():
        with pytest.raises(ConfigNotFoundError, match="read-only"):
            config.get(field, default="outer-fallback")
    assert calls == []
    assert store.get("test.VALUE") is MISSING
    assert list(tmp_path.iterdir()) == before
    assert config.get("VALUE") == "schema-default"
