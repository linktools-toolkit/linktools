# -*- coding: utf-8 -*-
"""FileSource / ConfigSource.before_provider wiring (spec Part III).

Exercises the generic ConfigResolver/Config mechanism with two FileSource
instances (local-file, global-file) inserted between PersistentSource and
DefaultSource, exactly as _environ.py wires them in Commit 3. This file
covers Commit 2 in isolation: the resolver-level priority mechanics, not the
BaseEnviron bootstrap wiring itself.
"""
import os

from linktools.core import (
    Config, ConfigField, ConfigSchema, DefaultSource, EnvironmentSource,
    FileSource, LazyProvider, PersistentSource, PromptProvider, RuntimeOverrideSource,
)
from linktools.core._config_store import ConfigStore
from linktools.core._locks import LockManager


def _make_config(tmp_path, local_data=None, global_data=None, env_prefix=""):
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema()
    local_source = FileSource(local_data or {}, name="local-file")
    global_source = FileSource(global_data or {}, name="global-file")
    return Config(None, schema, sources=[
        EnvironmentSource((os.environ, env_prefix)),
        RuntimeOverrideSource(),
        PersistentSource(store, "test"),
        local_source,
        global_source,
        DefaultSource(schema),
    ]), local_source, global_source


# -- exact priority chain (spec §69) -----------------------------------------

def test_os_environment_outranks_runtime(tmp_path, monkeypatch):
    config, _, _ = _make_config(tmp_path)
    monkeypatch.setenv("KEY", "env")
    config.set("KEY", "runtime")
    assert config.get("KEY") == "env"


def test_runtime_outranks_persistent(tmp_path):
    config, _, _ = _make_config(tmp_path)
    config.persist("KEY", "persistent")
    config.set("KEY", "runtime")
    assert config.get("KEY") == "runtime"


def test_persistent_outranks_local_file(tmp_path):
    config, _, _ = _make_config(tmp_path, local_data={"KEY": "local"})
    config.persist("KEY", "persistent")
    assert config.get("KEY") == "persistent"


def test_local_file_outranks_global_file(tmp_path):
    config, _, _ = _make_config(tmp_path, local_data={"KEY": "local"}, global_data={"KEY": "global"})
    assert config.get("KEY") == "local"


def test_global_file_outranks_provider(tmp_path):
    config, _, _ = _make_config(tmp_path, global_data={"KEY": "global"})
    config.update_defaults(KEY=ConfigField(provider=PromptProvider(default="_")))
    assert config.get("KEY") == "global"


def test_provider_outranks_default(tmp_path):
    config, _, _ = _make_config(tmp_path)
    config.update_defaults(KEY=ConfigField(provider=LazyProvider(lambda r: "provided"), default="default-value"))
    assert config.get("KEY") == "provided"


def test_default_used_when_nothing_else_present(tmp_path):
    config, _, _ = _make_config(tmp_path)
    config.update_defaults(KEY=ConfigField(default="default-value"))
    assert config.get("KEY") == "default-value"


def test_set_config_not_shadowed_by_file_value(tmp_path):
    # A field WITH a provider must still let RuntimeOverride win over the
    # local/global file values (the regression this whole chain protects
    # against: providers used to run before file/persistent/env were ever
    # consulted).
    config, _, _ = _make_config(tmp_path, local_data={"KEY": "local"}, global_data={"KEY": "global"})
    config.update_defaults(KEY=ConfigField(provider=PromptProvider(default="_")))
    config.set("KEY", "runtime")
    assert config.get("KEY") == "runtime"


def test_persist_not_shadowed_by_file_value(tmp_path):
    config, _, _ = _make_config(tmp_path, local_data={"KEY": "local"}, global_data={"KEY": "global"})
    config.update_defaults(KEY=ConfigField(provider=PromptProvider(default="_")))
    config.persist("KEY", "persisted")
    assert config.get("KEY") == "persisted"


# -- explain() shows local-file/global-file source names --------------------

def test_explain_reports_local_file_source(tmp_path):
    config, _, _ = _make_config(tmp_path, local_data={"KEY": "local"}, global_data={"KEY": "global"})
    result = config.explain("KEY")
    assert result["selected_source"] == "local-file"
    sources = {c["source"] for c in result["all_candidates"]}
    assert {"local-file", "global-file"} <= sources


def test_explain_reports_global_file_source_when_local_absent(tmp_path):
    config, _, _ = _make_config(tmp_path, global_data={"KEY": "global"})
    result = config.explain("KEY")
    assert result["selected_source"] == "global-file"


# -- reload() atomically replaces FileSource data ----------------------------

def test_reload_replaces_file_source_data_via_reload_fn(tmp_path):
    state = {"data": {"KEY": "v1"}}
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema()
    local_source = FileSource(state["data"], name="local-file", reload_fn=lambda: (state["data"], None))
    config = Config(None, schema, sources=[
        EnvironmentSource((os.environ, "")), RuntimeOverrideSource(), PersistentSource(store, "test"),
        local_source, DefaultSource(schema),
    ])
    assert config.get("KEY") == "v1"
    state["data"] = {"KEY": "v2"}
    config.reload()
    assert config.get("KEY") == "v2"


def test_reload_does_not_clear_runtime_by_default(tmp_path):
    config, _, _ = _make_config(tmp_path)
    config.set("KEY", "runtime")
    config.reload()
    assert config.get("KEY") == "runtime"


def test_reload_clears_runtime_when_requested(tmp_path):
    config, _, _ = _make_config(tmp_path)
    config.update_defaults(KEY=ConfigField(default="fallback"))
    config.set("KEY", "runtime")
    config.reload(clear_runtime=True)
    assert config.get("KEY") == "fallback"


# -- cast="path" resolves relative to the winning FileSource's base_path ----

def test_relative_path_field_resolves_against_local_file_base_path(tmp_path):
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema()
    local_root = tmp_path / "repo-root"
    local_source = FileSource({"DATA_DIR": "./data"}, name="local-file", base_path=str(local_root))
    config = Config(None, schema, sources=[
        EnvironmentSource((os.environ, "")), RuntimeOverrideSource(), PersistentSource(store, "test"),
        local_source, DefaultSource(schema),
    ])
    config.define(ConfigField(name="DATA_DIR", cast="path"))

    assert config.get("DATA_DIR") == str((local_root / "data").resolve())


def test_relative_path_field_resolves_against_global_file_base_path(tmp_path):
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema()
    global_root = tmp_path / "home" / ".linktools"
    local_source = FileSource({}, name="local-file", base_path=str(tmp_path / "cwd"))
    global_source = FileSource({"DATA_DIR": "./data"}, name="global-file", base_path=str(global_root))
    config = Config(None, schema, sources=[
        EnvironmentSource((os.environ, "")), RuntimeOverrideSource(), PersistentSource(store, "test"),
        local_source, global_source, DefaultSource(schema),
    ])
    config.define(ConfigField(name="DATA_DIR", cast="path"))

    assert config.get("DATA_DIR") == str((global_root / "data").resolve())


def test_absolute_path_field_ignores_base_path(tmp_path):
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema()
    absolute = str(tmp_path / "elsewhere")
    local_source = FileSource({"DATA_DIR": absolute}, name="local-file", base_path=str(tmp_path / "repo"))
    config = Config(None, schema, sources=[
        EnvironmentSource((os.environ, "")), RuntimeOverrideSource(), PersistentSource(store, "test"),
        local_source, DefaultSource(schema),
    ])
    config.define(ConfigField(name="DATA_DIR", cast="path"))

    assert config.get("DATA_DIR") == absolute


def test_file_source_replace_updates_data_and_base_path():
    source = FileSource({"K": "v1"}, name="local-file", base_path="/old")
    source.replace({"K": "v2"}, base_path="/new")
    assert source.get("K") == ("v2", True)
    assert source.base_path == "/new"
