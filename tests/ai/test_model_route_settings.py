#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model route configuration coverage."""

from typing import Any

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.model._openai import _RetryingModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def test_openai_operational_settings_do_not_change_durable_identity() -> None:
    first = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
    ).snapshot().resolve("default")
    second = ModelRegistry.openai(
        model="openai:gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
        timeout=60,
        max_retries=3,
        max_tokens=2048,
    ).snapshot().resolve("default")

    assert dict(first.semantic_payload) == dict(second.semantic_payload)
    assert first.fingerprint == second.fingerprint


def test_openai_custom_endpoint_is_operational_configuration() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://gateway.example/v1",
    ).snapshot().resolve("default")

    assert dict(binding.semantic_payload) == {
        "version": 1,
        "provider": "openai",
        "model_identity": "openai:gpt-test",
        "settings": {},
    }


def test_openai_max_tokens_changes_durable_identity() -> None:
    plain = ModelRegistry.openai(model="gpt-test").snapshot().resolve("default")
    configured = ModelRegistry.openai(
        model="gpt-test",
        max_tokens=2048,
    ).snapshot().resolve("default")

    assert dict(plain.semantic_payload)["settings"] == {}
    assert dict(configured.semantic_payload)["settings"] == {"max_tokens": 2048}
    assert plain.fingerprint != configured.fingerprint


def test_model_registry_restore_requires_exact_semantic_settings() -> None:
    historical = ModelRegistry.openai(
        model="gpt-test",
        max_tokens=1024,
    ).snapshot().resolve("default")
    registry = ModelRegistry.openai(
        model="gpt-test",
        max_tokens=2048,
    )

    with pytest.raises(AIError) as raised:
        registry.snapshot().restore(
            dict(historical.semantic_payload),
            route_id="default",
        )

    assert raised.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


def test_openai_route_materializes_settings_and_retries() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        api_key="test-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
    ).snapshot().resolve("default")

    model = binding.materialize()

    assert isinstance(model, _RetryingModel)
    assert isinstance(model.wrapped, OpenAIChatModel)
    assert model.wrapped.settings == {"timeout": 30, "max_tokens": 2048}
    provider = model.wrapped.provider
    assert isinstance(provider, OpenAIProvider)
    assert provider.client.max_retries == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": True},
        {"max_retries": -1},
        {"max_retries": True},
        {"max_tokens": 0},
        {"max_tokens": True},
    ],
)
def test_openai_route_rejects_invalid_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelRegistry.openai(model="gpt-test", **kwargs)
