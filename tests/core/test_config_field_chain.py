#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ConfigField.chain preserves provider order, fallback, and field options."""
import os

import pytest

from linktools.core import (
    AliasProvider, Config, ConfigField, ConfigSchema,
    DefaultSource, EnvironmentSource, PersistentSource, RuntimeOverrideSource,
)
from linktools.core._config_store import ConfigStore
from linktools.core._locks import LockManager


def _make_config(tmp_path):
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema()
    return Config(None, schema, sources=[
        EnvironmentSource((os.environ, "")),
        RuntimeOverrideSource(),
        PersistentSource(store, "test"),
        DefaultSource(schema),
    ])


def test_chain_with_one_missing_alias_uses_field_default_and_cast(tmp_path):
    config = _make_config(tmp_path)
    config.update_defaults(FOO=ConfigField.chain(AliasProvider("MISSING"), default=7, cast=str))
    assert config.get("FOO") == "7"


@pytest.mark.parametrize("first,expected", (("A", "first"), ("MISSING", "second")))
def test_chain_uses_first_available_provider(tmp_path, first, expected):
    config = _make_config(tmp_path)
    config.update_defaults(
        A="first", B="second",
        FOO=ConfigField.chain(AliasProvider(first), AliasProvider("B")),
    )
    assert config.get("FOO") == expected


def test_name_omitted_field_resolves_correctly(tmp_path):
    config = _make_config(tmp_path)
    config.update_defaults(
        BAR="bar-value",
        FOO=ConfigField.chain(AliasProvider("BAR")),
    )
    assert config.get("FOO") == "bar-value"
