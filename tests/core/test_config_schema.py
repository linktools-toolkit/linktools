# -*- coding: utf-8 -*-
"""Tests for the new Config Schema/Sources/Resolver (spec §8.2-§8.9).

Standalone build (PR 07); the legacy Config stays until cntr's DSL migrates to it
(PR 08). Exercises source precedence, casting/validation, cycle detection,
secret masking in explain, and multi-instance isolation.
"""
import pytest

from linktools.core._config_schema import (
    ConfigField, ConfigSchema, ConfigResolver,
    EnvironmentSource, RuntimeOverrideSource, DefaultSource,
)
from linktools.errors import ConfigCastError, ConfigValidationError, ConfigNotFoundError, ConfigCycleError
from linktools.types import MISSING


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

def test_sources_isolate():
    ro = RuntimeOverrideSource()
    ro.set("K", "rt")
    assert ro.get("K") == ("rt", True)
    other = RuntimeOverrideSource()
    assert other.get("K") == (MISSING, False)  # independent instance


def test_environment_source_reads_os_environ(monkeypatch):
    monkeypatch.setenv("LT_FOO", "bar")
    src = EnvironmentSource("LT_")
    assert src.get("FOO") == ("bar", True)
    assert src.get("NOPE") == (MISSING, False)


# --------------------------------------------------------------------------- #
# Schema / precedence
# --------------------------------------------------------------------------- #

def test_default_when_nothing_else_provides():
    schema = ConfigSchema().define(ConfigField(name="HOST", default="localhost"))
    r = ConfigResolver(schema, sources=[DefaultSource(schema)])
    assert r.resolve("HOST").value == "localhost"


def test_environment_beats_default(monkeypatch):
    monkeypatch.setenv("LT_HOST", "env-host")
    schema = ConfigSchema().define(ConfigField(name="HOST", default="localhost"))
    r = ConfigResolver(schema, sources=[
        EnvironmentSource("LT_"), DefaultSource(schema)])
    res = r.resolve("HOST")
    assert res.value == "env-host"
    assert res.source_name == "environment"


def test_runtime_override_beats_environment(monkeypatch):
    monkeypatch.setenv("LT_HOST", "env")
    schema = ConfigSchema().define(ConfigField(name="HOST", default="def"))
    ro = RuntimeOverrideSource()
    ro.set("HOST", "runtime")
    r = ConfigResolver(schema, sources=[ro, EnvironmentSource("LT_"), DefaultSource(schema)])
    assert r.resolve("HOST").value == "runtime"
    assert r.resolve("HOST").source_name == "runtime-override"


def test_precedence_order_is_first_wins(monkeypatch):
    # §8.2: EnvironmentSource > RuntimeOverride > Persistent > File > Default.
    monkeypatch.setenv("LT_K", "from-env")
    schema = ConfigSchema().define(ConfigField(name="K", default="def"))
    ro = RuntimeOverrideSource(); ro.set("K", "from-runtime")
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_"), ro, DefaultSource(schema)])
    assert r.resolve("K").value == "from-env"  # environment wins over runtime


def test_unknown_key_raises_not_found():
    schema = ConfigSchema()
    r = ConfigResolver(schema, sources=[DefaultSource(schema)])
    with pytest.raises(ConfigNotFoundError):
        r.resolve("NOPE")


def test_aliases_resolve(monkeypatch):
    monkeypatch.setenv("LT_OLD", "v")
    schema = ConfigSchema().define(ConfigField(name="NEW", aliases=("OLD",), default="d"))
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_"), DefaultSource(schema)])
    assert r.resolve("NEW").value == "v"  # found via alias OLD


# --------------------------------------------------------------------------- #
# Cast / validate
# --------------------------------------------------------------------------- #

def test_cast_applied(monkeypatch):
    monkeypatch.setenv("LT_PORT", "8080")
    schema = ConfigSchema().define(ConfigField(name="PORT", default=0, cast=int))
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_"), DefaultSource(schema)])
    assert r.resolve("PORT").value == 8080


def test_cast_failure_raises_config_cast_error(monkeypatch):
    monkeypatch.setenv("LT_PORT", "not-an-int")
    schema = ConfigSchema().define(ConfigField(name="PORT", cast=int))
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_")])
    with pytest.raises(ConfigCastError):
        r.resolve("PORT")


def test_validator_failure_raises(monkeypatch):
    monkeypatch.setenv("LT_PORT", "99999")
    schema = ConfigSchema().define(
        ConfigField(name="PORT", cast=int, validator=lambda v: 0 < v < 65536))
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_")])
    with pytest.raises(ConfigValidationError):
        r.resolve("PORT")


# --------------------------------------------------------------------------- #
# explain (§8.9) -- secret never exposed
# --------------------------------------------------------------------------- #

def test_explain_reports_source_and_candidates(monkeypatch):
    monkeypatch.setenv("LT_HOST", "env-host")
    schema = ConfigSchema().define(ConfigField(name="HOST", default="localhost"))
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_"), DefaultSource(schema)])
    info = r.explain("HOST")
    assert info["resolved_value"] == "env-host"
    assert info["selected_source"] == "environment"
    assert info["secret"] is False
    assert isinstance(info["all_candidates"], list)


def test_explain_masks_secret_value(monkeypatch):
    # §8.9: a secret field must not expose its raw value in explain.
    monkeypatch.setenv("LT_TOKEN", "supersecret")
    schema = ConfigSchema().define(ConfigField(name="TOKEN", secret=True, default="x"))
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_"), DefaultSource(schema)])
    info = r.explain("TOKEN")
    assert info["secret"] is True
    assert info["resolved_value"] != "supersecret"
    assert info["raw_value"] != "supersecret"
    assert "***" in str(info["resolved_value"])


# --------------------------------------------------------------------------- #
# Cycle detection (§8.10) via Alias provider
# --------------------------------------------------------------------------- #

def test_alias_cycle_detected():
    from linktools.core._config_schema import AliasProvider
    schema = ConfigSchema()
    schema.define(ConfigField(name="A", provider=AliasProvider("B")))
    schema.define(ConfigField(name="B", provider=AliasProvider("A")))
    r = ConfigResolver(schema, sources=[DefaultSource(schema)])
    with pytest.raises(ConfigCycleError):
        r.resolve("A")


# --------------------------------------------------------------------------- #
# reload clears runtime override by default
# --------------------------------------------------------------------------- #

def test_runtime_override_clear():
    ro = RuntimeOverrideSource()
    ro.set("K", "v")
    ro.clear()
    assert ro.get("K") == (MISSING, False)
