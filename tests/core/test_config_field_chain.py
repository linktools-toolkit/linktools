# -*- coding: utf-8 -*-
"""Tests for the ConfigField ergonomics simplification:

1. ``name`` is optional -- omitted when a field is defined through a
   ``configs`` dict consumed by ``Config.update_defaults`` (the dict key
   supplies the name).
2. ``ConfigField.chain(*providers, **kwargs)`` is shorthand for
   ``ConfigField(provider=ChainProvider(*providers), **kwargs)``, collapsing
   to a bare provider (no ChainProvider wrapper) when only one is given.
"""
from linktools.core import (
    AliasProvider, Config, ConfigField, ConfigSchema, ChainProvider,
    DefaultSource, EnvironmentSource, PersistentSource, RuntimeOverrideSource,
)
from linktools.core._config import ConfigStore
from linktools.core._locks import LockManager


def _make_config(tmp_path):
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema(allow_unknown=True)
    return Config(None, schema, sources=[
        EnvironmentSource(""),
        RuntimeOverrideSource(),
        PersistentSource(store, "test"),
        DefaultSource(schema),
    ])


def test_chain_with_one_provider_still_wraps_in_chain_provider():
    # Not a no-op: ChainProvider lets a failing sub-provider fall through to
    # _first_present/field.default, which a bare provider would not do.
    provider = AliasProvider("OTHER")
    field = ConfigField.chain(provider)
    assert isinstance(field.provider, ChainProvider)
    assert field.provider.providers == [provider]


def test_chain_with_multiple_providers_wraps_in_chain_provider():
    a, b = AliasProvider("A"), AliasProvider("B")
    field = ConfigField.chain(a, b)
    assert isinstance(field.provider, ChainProvider)
    assert field.provider.providers == [a, b]


def test_chain_passes_through_other_kwargs():
    field = ConfigField.chain(AliasProvider("A"), default="d", cast=str)
    assert field.default == "d"
    assert field.cast is str


def test_name_omitted_is_filled_from_update_defaults_key(tmp_path):
    config = _make_config(tmp_path)
    config.update_defaults(FOO=ConfigField.chain(AliasProvider("BAR"), default="d"))
    assert config.schema.get("FOO").name == "FOO"


def test_name_omitted_field_resolves_correctly(tmp_path):
    config = _make_config(tmp_path)
    config.update_defaults(
        BAR="bar-value",
        FOO=ConfigField.chain(AliasProvider("BAR")),
    )
    assert config.get("FOO") == "bar-value"
