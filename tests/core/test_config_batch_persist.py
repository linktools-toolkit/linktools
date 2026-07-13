#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config.validate_value/persist_many: pure validation (no I/O) plus one
atomic multi-key write, so a caller (cntr's `config set`) can check every
key against every relevant schema before persisting any of them -- a later
key's failure must never leave an earlier key already written (review
P1-05).
"""
import pytest

from linktools.core._config_store import ConfigStore
from linktools.core._config import Config, ConfigField, ConfigSchema, PersistentSource
from linktools.errors import ConfigCastError, ConfigValidationError
from linktools.types import MISSING


def _make_config(tmp_path):
    store = ConfigStore(tmp_path / "settings.json")
    schema = ConfigSchema()
    config = Config(environ=None, schema=schema, sources=[PersistentSource(store, "test")])
    return config, store


def test_validate_value_does_not_write_anything(tmp_path):
    config, store = _make_config(tmp_path)
    config.schema.define(ConfigField(name="PORT", cast=int))

    result = config.validate_value("PORT", "8080")

    assert result == 8080
    assert store.get("test.PORT") is MISSING


def test_validate_value_raises_on_cast_failure(tmp_path):
    config, _ = _make_config(tmp_path)
    config.schema.define(ConfigField(name="PORT", cast=int))
    with pytest.raises(ConfigCastError):
        config.validate_value("PORT", "not-a-number")


def test_validate_value_raises_on_validator_failure(tmp_path):
    config, _ = _make_config(tmp_path)
    config.schema.define(ConfigField(name="PORT", cast=int, validator=lambda v: 0 < v < 100))
    with pytest.raises(ConfigValidationError):
        config.validate_value("PORT", "9999")


def test_validate_value_unknown_key_passes_through(tmp_path):
    config, _ = _make_config(tmp_path)
    assert config.validate_value("SOME_UNKNOWN_KEY", "raw") == "raw"


def test_persist_many_writes_all_keys_in_one_call(tmp_path):
    config, store = _make_config(tmp_path)
    config.persist_many({"A": "1", "B": "2"})
    assert store.get("test.A") == "1"
    assert store.get("test.B") == "2"


def test_persist_many_stores_raw_values_not_cast(tmp_path):
    config, store = _make_config(tmp_path)
    config.schema.define(ConfigField(name="PORT", cast=int))
    config.persist_many({"PORT": "8080"})
    # Raw string, not the cast int -- a different Config/repo may cast the
    # same persisted key differently.
    assert store.get("test.PORT") == "8080"
    assert config.get("PORT") == 8080


def test_persist_many_raises_without_persistent_source():
    schema = ConfigSchema()
    config = Config(environ=None, schema=schema, sources=[])
    from linktools.errors import ConfigError
    with pytest.raises(ConfigError, match="PersistentSource"):
        config.persist_many({"A": "1"})
