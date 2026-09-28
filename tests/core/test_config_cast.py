#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ConfigField cast handling: the ``"path"``/``"json"`` string casts, and the
``bool``/``str`` builtin special-cases, must all resolve through _cast_value.

ConfigField.cast accepts a callable or one of the ``"path"``/``"json"``
literals (mirroring ``Config.cast(value, type=...)``). A bare
``field.cast(value)`` call raises ``TypeError`` on the string-literal forms,
so resolution routes through ``_cast_value`` instead (covers ``cast="path"``
fields like cntr's DOCKER_APP_PATH). ``bool``/``str`` are further special-cased
there even though they're plain callables: passed straight to the builtin,
``bool("false")`` is ``True`` (any non-empty string is truthy) and
``str(["a"])`` produces a Python repr rather than valid JSON.
"""
from linktools.core._config import (
    ConfigField,
    ConfigResolver,
    ConfigSchema,
)


def _resolve(field):
    schema = ConfigSchema()
    schema.define(field)
    return ConfigResolver(schema, sources=[]).resolve(field.name).value


def test_cast_path_string_resolves_to_abspath(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert _resolve(ConfigField(name="P", cast="path", default="rel/x")) == str(tmp_path / "rel" / "x")


def test_cast_path_string_does_not_raise_on_absolute(tmp_path):
    # The regression: a bare field.cast(value) call would raise TypeError here.
    assert _resolve(ConfigField(name="P", cast="path", default="/srv/app")) == "/srv/app"


def test_callable_cast_still_works():
    assert _resolve(ConfigField(name="N", cast=int, default="5")) == 5


def test_no_cast_passes_value_through():
    assert _resolve(ConfigField(name="R", default="raw")) == "raw"


# --------------------------------------------------------------------------- #
# cast=bool must not fall into Python's bool("false") == True trap.
# --------------------------------------------------------------------------- #

def test_cast_bool_true_strings():
    for value in ("true", "True", "yes", "y", "on", "1"):
        assert _resolve(ConfigField(name="B", cast=bool, default=value)) is True


def test_cast_bool_false_strings():
    for value in ("false", "False", "no", "n", "off", "0"):
        assert _resolve(ConfigField(name="B", cast=bool, default=value)) is False


def test_cast_bool_passes_through_real_bool():
    assert _resolve(ConfigField(name="B", cast=bool, default=True)) is True
    assert _resolve(ConfigField(name="B", cast=bool, default=False)) is False


def test_cast_bool_unparseable_string_raises():
    import pytest
    from linktools.errors import ConfigCastError

    with pytest.raises(ConfigCastError):
        _resolve(ConfigField(name="B", cast=bool, default="maybe"))


# --------------------------------------------------------------------------- #
# cast=str should JSON-dump structured values (not Python repr) and map None
# to "" (not "None").
# --------------------------------------------------------------------------- #

def test_cast_str_passes_through_string():
    assert _resolve(ConfigField(name="S", cast=str, default="already")) == "already"


def test_cast_str_json_dumps_structured_values():
    assert _resolve(ConfigField(name="S", cast=str, default=["a", "b"])) == '["a", "b"]'
    assert _resolve(ConfigField(name="S", cast=str, default={"a": 1})) == '{"a": 1}'


def test_cast_str_none_becomes_empty_string():
    assert _resolve(ConfigField(name="S", cast=str, default=None)) == ""


def test_cast_str_other_values_use_str():
    assert _resolve(ConfigField(name="S", cast=str, default=5)) == "5"


# --------------------------------------------------------------------------- #
# cast="json" parses a raw string (e.g. an env var) into JSON data.
# --------------------------------------------------------------------------- #

def test_cast_json_parses_string():
    assert _resolve(ConfigField(name="J", cast="json", default='["a", 1]')) == ["a", 1]


def test_cast_json_passes_through_structured_value():
    assert _resolve(ConfigField(name="J", cast="json", default={"a": 1})) == {"a": 1}


def test_cast_json_invalid_string_raises():
    import pytest
    from linktools.errors import ConfigCastError

    with pytest.raises(ConfigCastError):
        _resolve(ConfigField(name="J", cast="json", default="not json"))
