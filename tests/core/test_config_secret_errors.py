#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A ``ConfigField(secret=True)`` value must never appear in a cast-error or
validator-error exception message.

Regression: `_cast_validate()` built `ConfigCastError`/`ConfigValidationError`
messages with the raw/cast-attempted value interpolated directly (``%r``),
with no secret-awareness at all -- a secret field's value that failed to
cast, or failed its validator, leaked straight into the exception text
(and from there, typically, into a log line or a CLI error message).
"""
import pytest

from linktools.core._config import (
    Config, ConfigField, ConfigSchema, AliasProvider,
    EnvironmentSource, RuntimeOverrideSource, PersistentSource, DefaultSource,
)
from linktools.core._config_store import ConfigStore
from linktools.errors import ConfigCastError, ConfigValidationError

_SECRET_VALUE = "very-sensitive-value-78231"


def _make_config(tmp_path, schema):
    store = ConfigStore(tmp_path / "settings.json")
    sources = [
        EnvironmentSource("LT_"),
        RuntimeOverrideSource(),
        PersistentSource(store, "test"),
        DefaultSource(schema),
    ]
    return Config(None, schema, sources)


def test_cast_error_does_not_leak_secret_value(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="CREDENTIAL", cast=int, secret=True))
    config = _make_config(tmp_path, schema)
    config.set("CREDENTIAL", _SECRET_VALUE)  # not a valid int -> cast fails

    with pytest.raises(ConfigCastError) as excinfo:
        config.get("CREDENTIAL")
    assert _SECRET_VALUE not in str(excinfo.value)


def test_validator_false_does_not_leak_secret_value(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="CREDENTIAL", secret=True, validator=lambda v: len(v) > 100))
    config = _make_config(tmp_path, schema)
    config.set("CREDENTIAL", _SECRET_VALUE)  # too short -> validator returns False

    with pytest.raises(ConfigValidationError) as excinfo:
        config.get("CREDENTIAL")
    assert _SECRET_VALUE not in str(excinfo.value)


def test_validator_raising_with_secret_in_its_own_message_is_not_forwarded(tmp_path):
    """Core cannot scrub an arbitrary third-party exception's own text --
    for a secret field it must drop that message entirely rather than risk
    forwarding a value the validator itself embedded."""
    def _validator(value):
        raise ValueError("rejected value: %s" % value)

    schema = ConfigSchema()
    schema.define(ConfigField(name="CREDENTIAL", secret=True, validator=_validator))
    config = _make_config(tmp_path, schema)
    config.set("CREDENTIAL", _SECRET_VALUE)

    with pytest.raises(ConfigValidationError) as excinfo:
        config.get("CREDENTIAL")
    assert _SECRET_VALUE not in str(excinfo.value)


def test_validator_raising_on_non_secret_field_keeps_original_detail(tmp_path):
    """Sanity: the secret-only special-casing must not swallow useful error
    detail for an ordinary, non-secret field."""
    def _validator(value):
        raise ValueError("rejected value: %s" % value)

    schema = ConfigSchema()
    schema.define(ConfigField(name="PLAIN", secret=False, validator=_validator))
    config = _make_config(tmp_path, schema)
    config.set("PLAIN", "not-secret-at-all")

    with pytest.raises(ConfigValidationError) as excinfo:
        config.get("PLAIN")
    assert "not-secret-at-all" in str(excinfo.value)


def test_cast_error_on_non_secret_field_keeps_raw_value(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="PORT", cast=int, secret=False))
    config = _make_config(tmp_path, schema)
    config.set("PORT", "not-a-number")

    with pytest.raises(ConfigCastError) as excinfo:
        config.get("PORT")
    assert "not-a-number" in str(excinfo.value)


def test_alias_of_secret_field_cast_error_does_not_leak_via_alias(tmp_path):
    """Writing through the ALIAS name must be redacted identically to
    writing through the canonical name -- the same underlying ConfigField
    object (and its `secret` flag) governs both."""
    schema = ConfigSchema()
    schema.define(ConfigField(name="CREDENTIAL", cast=int, secret=True, aliases=("OLD_CREDENTIAL",)))
    config = _make_config(tmp_path, schema)
    config.set("OLD_CREDENTIAL", _SECRET_VALUE)  # written via the alias name; not a valid int

    with pytest.raises(ConfigCastError) as excinfo:
        config.get("CREDENTIAL")
    assert _SECRET_VALUE not in str(excinfo.value)


def test_alias_of_secret_field_explain_reports_secret(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="CREDENTIAL", cast=int, secret=True, aliases=("OLD_CREDENTIAL",)))
    config = _make_config(tmp_path, schema)
    config.set("OLD_CREDENTIAL", "12345")  # a validly castable value this time

    info = config.explain("OLD_CREDENTIAL")
    assert info["secret"] is True
    assert info["resolved_value"] == "***"


def test_explain_does_not_leak_secret_raw_or_resolved_value(tmp_path):
    schema = ConfigSchema()
    schema.define(ConfigField(name="CREDENTIAL", secret=True))
    config = _make_config(tmp_path, schema)
    config.set("CREDENTIAL", _SECRET_VALUE)

    import json
    payload = json.dumps(config.explain("CREDENTIAL"), default=str)
    assert _SECRET_VALUE not in payload
