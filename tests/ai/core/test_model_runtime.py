#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.model.registry import (
    ModelBundle,
    ModelClientUnavailable,
    ModelRegistry,
    RuntimeModelConfig,
    _bundle_from_config,
)


def _config(**overrides) -> RuntimeModelConfig:
    defaults = dict(
        model_type="standard",
        protocol="openai",
        model="gpt-4o-mini",
        base_url="https://api.example.com/v1",
        api_key="sk-test",
        auth_token=None,
        timeout_seconds=300,
        raw={"max_output_tokens": 4096, "max_turns": 10, "max_retries": 1},
    )
    defaults.update(overrides)
    return RuntimeModelConfig(**defaults)


def test_bundle_from_config_builds_openai_model():
    bundle = _bundle_from_config(_config())
    assert bundle.config.model_type == "standard"
    assert bundle.model.model_name == "gpt-4o-mini"
    assert bundle.settings["max_tokens"] == 4096
    assert bundle.usage_limits.request_limit == 10


def test_bundle_from_config_rejects_unsupported_protocol():
    with pytest.raises(ModelClientUnavailable, match="unsupported protocol"):
        _bundle_from_config(_config(protocol="anthropic"))


def test_bundle_from_config_rejects_missing_base_url():
    with pytest.raises(ModelClientUnavailable, match="requires base_url"):
        _bundle_from_config(_config(base_url=""))


def test_model_registry_register_with_config_returns_bundle():
    registry = ModelRegistry()
    registry.register("standard", config=_config())
    bundle = registry.get("standard")
    assert isinstance(bundle, ModelBundle)
    assert bundle.config.model_type == "standard"
    assert bundle.model.model_name == "gpt-4o-mini"


def test_model_registry_register_with_model_wraps_it_directly():
    registry = ModelRegistry()
    fake_model = FunctionModel(lambda messages, info: None)
    registry.register("custom", model=fake_model)
    bundle = registry.get("custom")
    assert bundle.model is fake_model
    assert bundle.config.model_type == "custom"
    assert bundle.settings["max_tokens"] == 4096
    assert bundle.usage_limits.request_limit == 10


def test_model_registry_register_with_model_accepts_explicit_settings_and_limits():
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import UsageLimits

    registry = ModelRegistry()
    fake_model = FunctionModel(lambda messages, info: None)
    registry.register(
        "custom", model=fake_model,
        settings=ModelSettings(max_tokens=100),
        usage_limits=UsageLimits(request_limit=3),
    )
    bundle = registry.get("custom")
    assert bundle.settings["max_tokens"] == 100
    assert bundle.usage_limits.request_limit == 3


def test_model_registry_register_rejects_both_config_and_model():
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="exactly one"):
        registry.register("standard", config=_config(), model=FunctionModel(lambda m, i: None))


def test_model_registry_register_rejects_neither_config_nor_model():
    registry = ModelRegistry()
    with pytest.raises(ValueError, match="exactly one"):
        registry.register("standard")


def test_model_registry_raises_for_unregistered_model_type():
    registry = ModelRegistry()
    with pytest.raises(ModelClientUnavailable, match="standard"):
        registry.get("standard")


def test_model_registry_register_overwrites_existing_entry():
    registry = ModelRegistry()
    registry.register("standard", config=_config(model="first"))
    registry.register("standard", config=_config(model="second"))
    assert registry.get("standard").config.model == "second"
