#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config.set/persist/unset/remove must invalidate every dependent
memoized value, not just the written key.

Regression: these write paths used to call
``self._resolver.clear_memo(key)`` -- clearing only the exact key just
written. A field resolved indirectly through it (AliasProvider,
LazyProvider reading another field, a ChainProvider mixing either) kept
returning its stale memoized value until something else happened to clear
the whole memo (define()/update_defaults()/reload()). There is no
provider dependency graph, so the fix is to clear the whole memo on every
write -- the field count is small enough that this costs nothing
observable.
"""
import os

from linktools.core._config import (
    Config, ConfigField, ConfigSchema, AliasProvider, LazyProvider, ChainProvider,
    EnvironmentSource, RuntimeOverrideSource, PersistentSource, DefaultSource,
)
from linktools.core._config_store import ConfigStore


def _make_config(tmp_path, schema):
    store = ConfigStore(tmp_path / "settings.json")
    sources = [
        EnvironmentSource((os.environ, "LT_")),
        RuntimeOverrideSource(),
        PersistentSource(store, "test"),
        DefaultSource(schema),
    ]
    return Config(None, schema, sources)


def test_set_invalidates_alias_cache(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="SOURCE", default="old"))
    schema.define(ConfigField(name="ALIAS", provider=AliasProvider("SOURCE")))
    config = _make_config(tmp_path, schema)

    assert config.get("ALIAS") == "old"
    config.set("SOURCE", "new")
    assert config.get("ALIAS") == "new"


def test_persist_invalidates_lazy_cache(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="SOURCE", default="old"))
    schema.define(ConfigField(name="DERIVED", provider=LazyProvider(
        lambda resolver: "value:%s" % resolver.get("SOURCE"))))
    config = _make_config(tmp_path, schema)

    assert config.get("DERIVED") == "value:old"
    config.persist("SOURCE", "new")
    assert config.get("DERIVED") == "value:new"


def test_persist_invalidates_chain_provider_cache(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="SOURCE", default="old"))
    schema.define(ConfigField(name="CHAINED", provider=ChainProvider(
        LazyProvider(lambda resolver: "chain:%s" % resolver.get("SOURCE")))))
    config = _make_config(tmp_path, schema)

    assert config.get("CHAINED") == "chain:old"
    config.persist("SOURCE", "new")
    assert config.get("CHAINED") == "chain:new"


def test_unset_falls_back_to_persistent_for_alias_target(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="SOURCE", default="default-val"))
    schema.define(ConfigField(name="ALIAS", provider=AliasProvider("SOURCE")))
    config = _make_config(tmp_path, schema)

    config.persist("SOURCE", "persisted-val")
    config.set("SOURCE", "runtime-val")
    assert config.get("ALIAS") == "runtime-val"

    config.unset("SOURCE")
    assert config.get("ALIAS") == "persisted-val"


def test_remove_falls_back_to_default_for_lazy_dependency(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="SOURCE", default="default-val"))
    schema.define(ConfigField(name="DERIVED", provider=LazyProvider(
        lambda resolver: "value:%s" % resolver.get("SOURCE"))))
    config = _make_config(tmp_path, schema)

    config.persist("SOURCE", "persisted-val")
    assert config.get("DERIVED") == "value:persisted-val"

    config.remove("SOURCE")
    assert config.get("DERIVED") == "value:default-val"


def test_set_invalidates_a_previously_resolved_unrelated_key_too(tmp_path):
    """Whole-memo clearing is intentionally broader than the single written
    key -- confirm an unrelated already-memoized field is still correctly
    re-resolved (not just left stale-but-harmless) after a write."""
    schema = ConfigSchema()
    schema.define(ConfigField(name="OTHER", default="other-default"))
    schema.define(ConfigField(name="SOURCE", default="old"))
    schema.define(ConfigField(name="ALIAS", provider=AliasProvider("SOURCE")))
    config = _make_config(tmp_path, schema)

    assert config.get("OTHER") == "other-default"  # memoize OTHER first
    config.set("SOURCE", "new")
    assert config.get("ALIAS") == "new"
    assert config.get("OTHER") == "other-default"  # unaffected, still correct
