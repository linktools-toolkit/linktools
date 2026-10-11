#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planning enters a no-new-values scope before any declaration discovery."""
from types import SimpleNamespace

import pytest

from linktools.cntr.execution.planner import ExecutionPlanner
from linktools.core import Config, ConfigField, ConfigSchema, LazyProvider
from linktools.core._config import PersistentSource
from linktools.core._config_store import ConfigStore
from linktools.errors import ConfigNotFoundError


def test_plan_blocks_new_secret_before_selection_can_persist_it(tmp_path):
    schema = ConfigSchema()
    calls = []
    schema.define(ConfigField("TOKEN", provider=LazyProvider(lambda r: calls.append(True) or "secret", cached=True)))
    store = ConfigStore(tmp_path / "values.json")
    config = Config(None, schema, [PersistentSource(store, "test")])
    def select(*args, **kwargs):
        config.get("TOKEN")
        raise AssertionError("planning must stop on unavailable configuration")
    manager = SimpleNamespace(compose_operations=SimpleNamespace(select=select))
    with pytest.raises(ConfigNotFoundError, match="read-only"):
        ExecutionPlanner(manager).plan("up", ["app"])
    assert not calls
    assert store.keys() == []
    assert not (tmp_path / "values.json").exists()
    assert not Config.is_read_only_resolution()


def test_legacy_manager_migration_is_skipped_in_read_only_scope(tmp_path):
    from linktools.cntr._migrate import migrate_legacy_settings
    legacy = tmp_path / "config" / "containers.yml"
    legacy.parent.mkdir()
    legacy.write_text('["nginx"]')
    manager = SimpleNamespace(data_path=tmp_path)
    store = ConfigStore(tmp_path / "container.json")
    with Config.read_only_resolution():
        assert migrate_legacy_settings(manager, store) is store
    assert legacy.read_text() == '["nginx"]'
    assert not (tmp_path / "container.json").exists()


def test_legacy_container_migration_never_opens_cache_in_read_only_scope():
    from linktools.cntr import BaseContainer
    class Environment:
        @property
        def cache(self):
            raise AssertionError("read-only planning must not open legacy cache")
    namespace = SimpleNamespace()
    manager = SimpleNamespace(logger=None, environ=Environment(),
                              settings=SimpleNamespace(namespace=lambda name: namespace))
    container = BaseContainer(manager, "/unused", name="test")
    with Config.read_only_resolution():
        assert container.settings is namespace
