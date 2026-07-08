# -*- coding: utf-8 -*-
"""Tests for the new Config Schema/Sources/Resolver (spec §8.2-§8.9).

Standalone build (PR 07); the legacy Config stays until cntr's DSL migrates to it
(PR 08). Exercises source precedence, casting/validation, cycle detection,
secret masking in explain, and multi-instance isolation.
"""
import pytest

from linktools.core import (
    Config, ConfigField, ConfigSchema, ConfigResolver,
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
    from linktools.core import AliasProvider
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


# --------------------------------------------------------------------------- #
# PR-4: explain unknown/missing keys + get/require semantics (spec §6)
# --------------------------------------------------------------------------- #

def test_explain_unknown_persistent_key():
    # allow_unknown: key present in a source but not defined in the schema.
    schema = ConfigSchema(allow_unknown=True)
    ro = RuntimeOverrideSource(); ro.set("DYNAMIC", "rt-val")
    r = ConfigResolver(schema, sources=[ro])
    info = r.explain("DYNAMIC")
    assert info["unknown"] is True
    assert info["resolved_value"] == "rt-val"
    assert info["selected_source"] == "runtime-override"
    assert "key is not defined in schema" in info["warnings"]
    assert info["secret"] is False          # field is None -> must not crash


def test_explain_unknown_env_key(monkeypatch):
    monkeypatch.setenv("LT_DYNAMIC", "from-env")
    schema = ConfigSchema(allow_unknown=True)
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_")])
    info = r.explain("DYNAMIC")
    assert info["unknown"] is True
    assert info["resolved_value"] == "from-env"
    assert info["selected_source"] == "environment"


def test_explain_missing_key_reports_not_found():
    schema = ConfigSchema(allow_unknown=True)
    r = ConfigResolver(schema, sources=[DefaultSource(schema)])
    info = r.explain("DOES_NOT_EXIST")
    assert info["found"] is False
    assert info["unknown"] is True
    assert info["resolved_value"] is MISSING
    assert info["all_candidates"] == []


def test_explain_known_key_has_no_unknown_warning(monkeypatch):
    monkeypatch.setenv("LT_HOST", "h")
    schema = ConfigSchema().define(ConfigField(name="HOST", default="d"))
    r = ConfigResolver(schema, sources=[EnvironmentSource("LT_"), DefaultSource(schema)])
    info = r.explain("HOST")
    assert info["unknown"] is False
    assert info["warnings"] == []


def _config(allow_unknown=True):
    schema = ConfigSchema(allow_unknown=allow_unknown)
    return Config(environ=None, schema=schema,
                  sources=[RuntimeOverrideSource(), DefaultSource(schema)])


def test_get_missing_without_default_raises():
    cfg = _config()
    with pytest.raises(ConfigNotFoundError):
        cfg.get("NOPE")


def test_get_missing_with_default_returns_default():
    cfg = _config()
    assert cfg.get("NOPE", default="fallback") == "fallback"


def test_require_missing_raises():
    cfg = _config()
    with pytest.raises(ConfigNotFoundError):
        cfg.require("NOPE")


def test_require_present_returns_value():
    schema = ConfigSchema().define(ConfigField(name="HOST", default="localhost"))
    cfg = Config(environ=None, schema=schema, sources=[DefaultSource(schema)])
    assert cfg.require("HOST") == "localhost"


def test_environ_debug_is_bool():
    from linktools.core import environ
    assert isinstance(environ.debug, bool)


def test_unknown_key_raises_when_allow_unknown_false():
    # With allow_unknown disabled, an unknown key must raise (not resolve to a
    # value); explain() still returns a found=False dict rather than crashing.
    schema = ConfigSchema(allow_unknown=False)
    r = ConfigResolver(schema, sources=[DefaultSource(schema)])
    with pytest.raises(ConfigNotFoundError):
        r.resolve("NOPE")
    info = r.explain("NOPE")
    assert info["found"] is False
    cfg = Config(environ=None, schema=schema, sources=[DefaultSource(schema)])
    with pytest.raises(ConfigNotFoundError):
        cfg.get("NOPE")
