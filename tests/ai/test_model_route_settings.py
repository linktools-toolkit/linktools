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
        provider_instance="corp-openai-primary",
        base_url="https://first.example/v1",
        api_key="first-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
    ).snapshot().resolve("default")
    second = ModelRegistry.openai(
        model="openai:gpt-test",
        provider_instance="corp-openai-primary",
        base_url="https://second.example/v1",
        api_key="second-key",
        timeout=60,
        max_retries=3,
        max_tokens=2048,
    ).snapshot().resolve("default")

    assert dict(first.semantic_payload) == dict(second.semantic_payload)
    assert first.fingerprint == second.fingerprint


def test_openai_provider_instance_changes_durable_identity() -> None:
    first = ModelRegistry.openai(
        model="gpt-test",
        provider_instance="corp-openai-primary",
        base_url="https://gateway.example/v1",
    ).snapshot().resolve("default")
    second = ModelRegistry.openai(
        model="gpt-test",
        provider_instance="corp-openai-secondary",
        base_url="https://gateway.example/v1",
    ).snapshot().resolve("default")

    assert dict(first.semantic_payload)["provider_instance"] == "corp-openai-primary"
    assert dict(second.semantic_payload)["provider_instance"] == "corp-openai-secondary"
    assert first.fingerprint != second.fingerprint


def test_openai_custom_endpoint_gets_conservative_instance_identity() -> None:
    first = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
    ).snapshot().resolve("default")
    second = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://second.example/v1",
    ).snapshot().resolve("default")

    first_instance = dict(first.semantic_payload)["provider_instance"]
    second_instance = dict(second.semantic_payload)["provider_instance"]
    assert isinstance(first_instance, str) and first_instance.startswith("openai-endpoint-")
    assert isinstance(second_instance, str) and second_instance.startswith("openai-endpoint-")
    assert first_instance != second_instance
    assert first.fingerprint != second.fingerprint


def test_openai_public_provider_has_stable_default_instance() -> None:
    binding = ModelRegistry.openai(model="gpt-test").snapshot().resolve("default")

    assert dict(binding.semantic_payload)["provider_instance"] == "openai-public"


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


def test_model_registry_restore_requires_exact_provider_instance() -> None:
    historical = ModelRegistry.openai(
        model="gpt-test",
        provider_instance="corp-openai-primary",
        base_url="https://old.example/v1",
    ).snapshot().resolve("default")
    registry = ModelRegistry.openai(
        model="gpt-test",
        provider_instance="corp-openai-secondary",
        base_url="https://new.example/v1",
    )

    with pytest.raises(AIError) as raised:
        registry.snapshot().restore(
            dict(historical.semantic_payload),
            route_id="default",
        )

    assert raised.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE


def test_legacy_public_openai_binding_can_restore_without_provider_instance() -> None:
    registry = ModelRegistry.openai(model="gpt-test")
    current = registry.snapshot().resolve("default")
    historical = dict(current.semantic_payload)
    historical.pop("provider_instance")

    restored = registry.snapshot().restore(historical, route_id="default")

    assert restored is current


def test_legacy_openai_binding_cannot_restore_to_custom_endpoint() -> None:
    public = ModelRegistry.openai(model="gpt-test").snapshot().resolve("default")
    historical = dict(public.semantic_payload)
    historical.pop("provider_instance")
    registry = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://gateway.example/v1",
    )

    with pytest.raises(AIError) as raised:
        registry.snapshot().restore(historical, route_id="default")

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
        {"provider_instance": ""},
        {"provider_instance": "bad instance"},
    ],
)
def test_openai_route_rejects_invalid_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelRegistry.openai(model="gpt-test", **kwargs)
