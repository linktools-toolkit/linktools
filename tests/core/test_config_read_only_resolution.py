#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only resolution scopes preserve saved configuration and never create secrets."""
import threading

import pytest

from linktools.core import Config, ConfigField, ConfigResolver, ConfigSchema, LazyProvider, PromptProvider, ConfirmProvider
from linktools.core._config import PersistentSource, RuntimeOverrideSource
from linktools.core._config_store import ConfigStore
from linktools.errors import ConfigError, ConfigNotFoundError


def config(tmp_path, field):
    schema = ConfigSchema()
    schema.define(field)
    store = ConfigStore(tmp_path / "settings.json")
    return Config(None, schema, [RuntimeOverrideSource(), PersistentSource(store, "test")]), store


@pytest.mark.parametrize("provider", ["lazy", "prompt", "confirm"])
def test_read_only_missing_cached_values_never_compute_or_write(tmp_path, monkeypatch, provider):
    def fail(*args, **kwargs):
        raise AssertionError("effectful provider executed")
    monkeypatch.setattr("linktools.rich.prompt", fail)
    monkeypatch.setattr("linktools.rich.confirm", fail)
    values = {"lazy": LazyProvider(fail, cached=True), "prompt": PromptProvider(cached=True),
              "confirm": ConfirmProvider(cached=True)}
    cfg, store = config(tmp_path, ConfigField("SECRET", provider=values[provider], required=True))
    before = list(tmp_path.iterdir())
    with Config.read_only_resolution():
        with pytest.raises(ConfigNotFoundError, match="read-only"):
            cfg.get("SECRET")
    assert store.keys() == []
    assert list(tmp_path.iterdir()) == before


def test_read_only_saved_and_explicit_values_still_resolve(tmp_path):
    def fail(resolver):
        raise AssertionError("provider executed")
    cfg, store = config(tmp_path, ConfigField("SECRET", provider=LazyProvider(fail, cached=True)))
    cfg.persist("SECRET", "saved")
    previous = (tmp_path / "settings.json").read_bytes()
    with Config.read_only_resolution():
        assert cfg.get("SECRET") == "saved"
        cfg.set("SECRET", "explicit")
        assert cfg.get("SECRET") == "explicit"
    assert (tmp_path / "settings.json").read_bytes() == previous


def test_context_covers_new_resolvers_and_nested_scopes(tmp_path):
    calls = []
    with Config.read_only_resolution():
        cfg, store = config(tmp_path, ConfigField("SECRET", provider=LazyProvider(
            lambda r: calls.append(True) or "new", cached=True)))
        with ConfigResolver.read_only_resolution():
            assert Config.is_read_only_resolution()
        assert Config.is_read_only_resolution()
        with pytest.raises(ConfigNotFoundError):
            cfg.get("SECRET")
    assert not Config.is_read_only_resolution()
    assert cfg.get("SECRET") == "new"
    assert calls == [True]


def test_read_only_value_does_not_poison_later_normal_memo(tmp_path):
    calls = []
    cfg, store = config(tmp_path, ConfigField("VALUE", provider=LazyProvider(lambda r: len(calls))))
    with Config.read_only_resolution():
        assert cfg.get("VALUE") == 0
    calls.append(True)
    assert cfg.get("VALUE") == 1


def test_read_only_cached_missing_cannot_fall_back_to_invented_empty(tmp_path):
    cfg, store = config(tmp_path, ConfigField.chain(
        LazyProvider(lambda r: "generated", cached=True), name="SECRET", default=""))
    with Config.read_only_resolution():
        with pytest.raises(ConfigNotFoundError, match="read-only"):
            cfg.get("SECRET", default="")


def test_read_only_scope_is_thread_local():
    states = []
    with Config.read_only_resolution():
        thread = threading.Thread(target=lambda: states.append(Config.is_read_only_resolution()))
        thread.start()
        thread.join()
    assert states == [False]


@pytest.mark.parametrize("operation", ["persist", "persist_many", "remove"])
def test_explicit_persistent_changes_rejected_in_readonly(tmp_path, operation):
    cfg, store = config(tmp_path, ConfigField("VALUE", default="default"))
    with Config.read_only_resolution(), pytest.raises(ConfigError, match="read-only"):
        if operation == "persist":
            cfg.persist("VALUE", "changed")
        elif operation == "persist_many":
            cfg.persist_many({"VALUE": "changed"})
        else:
            cfg.remove("VALUE")
