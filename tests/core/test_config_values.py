#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit config-dependent values preserve deferred reads and legacy behavior."""

import threading
import traceback
from collections import UserDict
from typing import TYPE_CHECKING

import pytest

from linktools.core import (
    Config, ConfigField, ConfigFormat, ConfigLazy, ConfigRef, ConfigSchema,
    ConfigStore, ConfigValue, DefaultSource, LazyProvider, PersistentSource,
    RuntimeOverrideSource,
)
from linktools.errors import ConfigCastError, ConfigCycleError, ConfigError, ConfigNotFoundError
from linktools.runtime import Proxy
from linktools.types import MISSING

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any


def _config(values: "dict[str, Any] | None" = None,
            *fields: ConfigField) -> Config:
    schema = ConfigSchema()
    for field in fields:
        schema.define(field)
    source = RuntimeOverrideSource()
    for key, value in (values or {}).items():
        source.set(key, value)
    return Config(None, schema, [source, DefaultSource(schema)])


def test_config_values_public_extension_point() -> None:
    with pytest.raises(TypeError):
        ConfigValue()

    class CustomValue(ConfigValue):
        def resolve(self, config: Config) -> "Any":
            return {"nested": [ConfigRef("VALUE")]}

    assert _config({"VALUE": 42}).resolve(CustomValue()) == {"nested": [42]}


def test_config_values_construction_repr_and_bool_do_not_read() -> None:
    calls = []
    secret = "private-format-or-default"

    class Callback:
        def __call__(self, config: Config) -> str:
            calls.append(config)
            return secret

        def __repr__(self) -> str:
            raise AssertionError("callback repr must not be inspected")

    values = [
        ConfigRef("VALUE", default=secret),
        ConfigFormat(secret + "{value}", value=Proxy(lambda: calls.append("proxy"))),
        ConfigLazy(Callback()),
    ]
    for value in values:
        assert secret not in repr(value)
        assert secret not in str(value)
        assert bool(value)
    assert calls == []


@pytest.mark.parametrize("key", [None, 0, ConfigField("VALUE")])
def test_config_values_reference_rejects_non_string_keys(key: "Any") -> None:
    with pytest.raises(TypeError, match="non-empty string"):
        ConfigRef(key)


def test_config_values_reference_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ConfigRef("")


def test_config_values_reference_accepts_string_subclass_without_bool_probe() -> None:
    class Key(str):
        def __bool__(self) -> bool:
            raise AssertionError("key truthiness must not be inspected")

    key = Key("VALUE")
    assert ConfigRef(key).key is key
    with pytest.raises(ValueError):
        ConfigRef(Key(""))


def test_config_values_constructors_do_not_inspect_proxy_class() -> None:
    calls = []
    proxy = Proxy(lambda: calls.append(True) or "VALUE")
    with pytest.raises(TypeError):
        ConfigRef(proxy)
    with pytest.raises(TypeError):
        ConfigFormat(proxy)
    assert calls == []


def test_config_values_format_requires_plain_string() -> None:
    class String(str):
        pass

    for value in (None, 12, String("{}")):
        with pytest.raises(TypeError, match="plain string"):
            ConfigFormat(value)
    with pytest.raises(TypeError, match="callable"):
        ConfigLazy(None)


def test_config_values_context_is_unbound_and_results_are_not_cached() -> None:
    first = _config({"VALUE": "first"})
    second = _config({"VALUE": "second"})
    calls = []
    reference = ConfigRef("VALUE")
    formatted = ConfigFormat("<{value}>", value=reference)
    callback = ConfigLazy(lambda config: calls.append(config) or formatted)

    assert first.resolve(callback) == "<first>"
    assert second.resolve(callback) == "<second>"
    first.set("VALUE", "changed")
    assert first.resolve(callback) == "<changed>"
    assert calls == [first, second, first]
    assert first.schema.fields() == []
    assert second.schema.fields() == []


def test_config_values_reference_preserves_get_missing_default_and_cast() -> None:
    config = _config({"NUMBER": "27", "BAD": "not-a-number", "FLAG": "false"})
    with pytest.raises(ConfigNotFoundError):
        config.resolve(ConfigRef("ABSENT"))
    assert config.resolve(ConfigRef("ABSENT", default=None)) is None
    assert config.resolve(ConfigRef("NUMBER", type=int)) == 27
    assert config.resolve(ConfigRef("FLAG", type=bool)) is False
    assert config.resolve(ConfigRef("BAD", type=int, default=0)) == 0
    with pytest.raises(ConfigCastError):
        config.resolve(ConfigRef("BAD", type=int))

    config.define(ConfigField("BAD", cast=int))
    with pytest.raises(ConfigCastError):
        config.resolve(ConfigRef("BAD", default=0))


def test_config_values_reference_resolves_default_recursively() -> None:
    config = _config({"VALUE": "resolved"})
    fallback = {"value": (ConfigLazy(lambda cfg: ConfigRef("VALUE")),)}
    assert config.resolve(ConfigRef("ABSENT", default=fallback)) == {"value": ("resolved",)}
    assert type(fallback["value"][0]) is ConfigLazy


def test_config_values_reference_result_is_resolved_after_get() -> None:
    declaration = ConfigLazy(lambda config: [ConfigRef("PLAIN")])
    calls = []

    class ReturningConfig(Config):
        def get(self, key: "str | ConfigField", type: "Any" = None,
                default: "Any" = MISSING) -> "Any":
            calls.append((key, type, default))
            return declaration if key == "EXPRESSION" else "resolved"

    config = ReturningConfig(None, ConfigSchema(), [])
    assert config.resolve(ConfigRef("EXPRESSION")) == ["resolved"]
    assert calls == [("EXPRESSION", None, MISSING), ("PLAIN", None, MISSING)]


@pytest.mark.parametrize("value", [None, False, 0, "", MISSING])
def test_config_values_preserve_plain_sentinel_and_falsy_values(value: "Any") -> None:
    config = _config()
    assert config.resolve(value) is value
    assert config.resolve(ConfigLazy(lambda cfg: value)) is value
    if value is not MISSING:
        assert config.resolve(ConfigRef("ABSENT", default=value)) is value


def test_config_values_copy_only_exact_containers_and_leave_keys_untouched() -> None:
    calls = []
    config = _config({"VALUE": 4})
    key = ConfigLazy(lambda cfg: calls.append("key"))
    reference = ConfigRef("VALUE")
    shared = [reference]
    original = {key: (shared, shared), "nested": [{"value": reference}]}
    resolved = config.resolve(original)

    assert type(resolved) is dict
    assert type(resolved[key]) is tuple
    assert type(resolved[key][0]) is list
    assert resolved[key] == ([4], [4])
    assert resolved[key][0] is not resolved[key][1]
    assert resolved["nested"] == [{"value": 4}]
    assert original[key][0] is shared
    assert shared[0] is reference
    assert original["nested"][0]["value"] is reference
    assert calls == []


def test_config_values_leave_nonplain_objects_and_callable_opaque() -> None:
    calls = []

    class Dictionary(dict):
        pass

    class List(list):
        pass

    class Tuple(tuple):
        pass

    class Object:
        value = ConfigLazy(lambda cfg: calls.append("attribute"))

    expression = ConfigLazy(lambda cfg: calls.append("expression"))
    values = [Dictionary(x=expression), List([expression]), Tuple([expression]),
              UserDict(x=expression), (item for item in [expression]),
              lambda: calls.append("callable"), Object(), "{VALUE}"]
    config = _config()
    for value in values:
        assert config.resolve(value) is value
    assert calls == []


def test_config_values_proxy_is_opaque_until_normal_format_consumption() -> None:
    calls = []
    proxy = Proxy(lambda: calls.append(True) or "old")
    config = _config()
    assert config.resolve(proxy) is proxy
    assert config.resolve({"value": [proxy]})["value"][0] is proxy
    assert config.resolve(ConfigLazy(lambda cfg: proxy)) is proxy
    assert calls == []
    assert config.resolve(ConfigFormat("{}", proxy)) == "old"
    assert config.resolve(ConfigFormat("{value}", value=proxy)) == "old"
    assert calls == [True]


def test_config_values_format_standard_fields_conversions_specs_and_escaping() -> None:
    config = _config({"COUNT": "7", "NAME": "name", "BRACES": "{not_config}"})
    value = ConfigFormat("{{literal}} {0:04d} {name!r} {braces}",
                         ConfigRef("COUNT", type=int), name=ConfigRef("NAME"),
                         braces=ConfigRef("BRACES"))
    assert config.resolve(value) == "{literal} 0007 'name' {not_config}"
    assert config.resolve(ConfigFormat("{}", "{x}")) == "{x}"


def test_config_values_format_resolves_all_explicit_arguments_once_per_position() -> None:
    calls = []
    config = _config()
    value = ConfigLazy(lambda cfg: calls.append(cfg) or len(calls))
    assert config.resolve(ConfigFormat("{x}{x}", x=value, unused=value)) == "11"
    assert calls == [config, config]
    assert config.resolve(ConfigFormat("{0}{1}", value, value)) == "34"
    assert config.resolve([value, value]) == [5, 6]


def test_config_values_format_does_not_read_implicit_keys() -> None:
    calls = []
    config = _config({}, ConfigField("KEY", provider=LazyProvider(
        lambda resolver: calls.append(True) or "value")))
    with pytest.raises(ConfigError, match="formatting failed"):
        config.resolve(ConfigFormat("{KEY}"))
    assert calls == []


def test_config_values_lazy_branches_skip_unneeded_dependencies() -> None:
    calls = []
    config = _config({"ENABLED": False}, ConfigField("SECRET", provider=LazyProvider(
        lambda resolver: calls.append(True) or "value")))
    value = ConfigLazy(lambda cfg: ConfigRef("SECRET") if cfg.get("ENABLED") else "")
    assert config.resolve(value) == ""
    assert calls == []


@pytest.mark.parametrize("kind", ["list", "dict", "tuple"])
def test_config_values_container_cycles_are_typed_and_cleaned(kind: str) -> None:
    config = _config()
    if kind == "dict":
        value = {}
        value["self"] = value
    else:
        value = []
        value.append(value if kind == "list" else (value,))
    with pytest.raises(ConfigCycleError, match="cycle"):
        config.resolve(value)
    value.clear()
    assert config.resolve(value) == value
    assert config.resolve([1]) == [1]


def test_config_values_self_return_and_mutual_expression_cycles() -> None:
    config = _config()
    value = ConfigLazy(lambda cfg: value)
    with pytest.raises(ConfigCycleError):
        config.resolve(value)
    first = ConfigLazy(lambda cfg: second)
    second = ConfigLazy(lambda cfg: {"first": first})
    with pytest.raises(ConfigCycleError):
        config.resolve(first)
    assert config.resolve(ConfigRef("ABSENT", default="ok")) == "ok"


def test_config_values_nested_resolve_cycle_is_detected_and_retry_succeeds() -> None:
    config = _config()
    retry = []

    def callback(cfg: Config) -> "Any":
        if not retry:
            retry.append(True)
            return cfg.resolve(value)
        return cfg.resolve({"ok": ConfigLazy(lambda current: 1)})

    value = ConfigLazy(callback)
    with pytest.raises(ConfigCycleError):
        config.resolve(value)
    assert config.resolve(value) == {"ok": 1}


def test_config_values_nested_contexts_share_cycle_detection_without_binding() -> None:
    first = _config({"DONE": False})
    second = _config({"DONE": True})
    value = ConfigLazy(lambda cfg: "done" if cfg.get("DONE") else second.resolve(value))
    assert first.resolve(value) == "done"
    second.set("DONE", False)
    with pytest.raises(ConfigCycleError):
        first.resolve(value)
    second.set("DONE", True)
    assert first.resolve(value) == "done"


def test_config_values_cycle_scope_is_thread_local() -> None:
    config = _config()
    results = []
    started = threading.Event()
    resume = threading.Event()

    def callback(cfg: Config) -> str:
        if threading.current_thread() is worker:
            started.set()
            if not resume.wait(10):
                raise AssertionError("main thread did not resume worker")
        return "resolved"

    value = ConfigLazy(callback)

    def run() -> None:
        try:
            results.append(config.resolve(value))
        except Exception as error:
            results.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert started.wait(10)
        assert config.resolve(value) == "resolved"
    finally:
        resume.set()
        worker.join(10)
    assert not worker.is_alive()
    assert results == ["resolved"]


@pytest.mark.parametrize("error_type", [ValueError, ConfigError, ConfigCycleError])
def test_config_values_callback_errors_hide_secrets_and_allow_retry(error_type: type) -> None:
    secret = "private-callback-value"
    calls = []
    config = _config()

    def callback(cfg: Config) -> str:
        if not calls:
            calls.append(True)
            raise error_type(secret)
        return "ok"

    value = ConfigLazy(callback)
    expected = ConfigCycleError if error_type is ConfigCycleError else ConfigError
    with pytest.raises(expected) as error:
        config.resolve(value)
    assert secret not in str(error.value)
    assert secret not in "".join(traceback.format_exception(
        type(error.value), error.value, error.value.__traceback__))
    assert config.resolve(value) == "ok"


def test_config_values_format_errors_hide_secret_values_and_original_exceptions() -> None:
    secret = "private-format-value"

    class SensitiveValue:
        def __format__(self, spec: str) -> str:
            raise ValueError(secret)

        def __repr__(self) -> str:
            raise AssertionError("repr must not be inspected")

    config = _config()
    for value in (ConfigFormat("{0}", SensitiveValue()), ConfigFormat("{" + secret + "}")):
        with pytest.raises(ConfigError, match="formatting failed") as error:
            config.resolve(value)
        assert secret not in str(error.value)
        assert secret not in "".join(traceback.format_exception(
            type(error.value), error.value, error.value.__traceback__))
    assert config.resolve(ConfigFormat("{}", "ok")) == "ok"


def test_config_values_reference_retains_secret_cast_redaction() -> None:
    secret = "private-not-a-number"
    config = _config({"SECRET": secret}, ConfigField("SECRET", secret=True))
    with pytest.raises(ConfigCastError) as error:
        config.resolve(ConfigRef("SECRET", type=int))
    assert secret not in str(error.value)
    assert "***" in str(error.value)


def test_config_values_readonly_blocks_cached_provider_and_persistent_writes(tmp_path: "Path") -> None:
    calls = []
    schema = ConfigSchema().define(ConfigField("SECRET", provider=LazyProvider(
        lambda resolver: calls.append(True) or "generated", cached=True)))
    store = ConfigStore(tmp_path / "settings.json")
    config = Config(None, schema, [RuntimeOverrideSource(), PersistentSource(store, "test")])
    with Config.read_only_resolution():
        with pytest.raises(ConfigNotFoundError, match="read-only"):
            config.resolve(ConfigRef("SECRET", default="fallback"))
        with pytest.raises(ConfigError):
            config.resolve(ConfigLazy(lambda cfg: cfg.persist("SECRET", "written")))
        assert config.resolve(ConfigLazy(lambda cfg: cfg.is_read_only_resolution())) is True
    assert calls == []
    assert store.keys() == []
    assert not (tmp_path / "settings.json").exists()
    assert config.resolve(ConfigRef("SECRET")) == "generated"
    assert calls == [True]


def test_config_values_do_not_change_existing_get_provider_memo() -> None:
    calls = []
    config = _config({}, ConfigField("VALUE", provider=LazyProvider(
        lambda resolver: calls.append(True) or len(calls))))
    declaration = ConfigRef("VALUE")
    assert config.resolve([declaration, declaration]) == [1, 1]
    assert config.get("VALUE") == 1
    assert calls == [True]


def test_config_values_default_does_not_swallow_callback_or_format_errors() -> None:
    config = _config()

    def fail(cfg: Config) -> str:
        raise ValueError("callback failure")

    for value in (ConfigLazy(fail), ConfigFormat("{missing}")):
        with pytest.raises(ConfigError):
            config.resolve(ConfigRef("ABSENT", default=value))
