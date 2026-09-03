#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model route configuration coverage."""

from typing import Any

import httpx2
import pytest
from linktools.ai.agent import AgentCompiler
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.model._openai import (
    _AsyncClientTransport,
    _RetryingOpenAIProvider,
    _raise_retryable_status,
    _retry_error_result,
)
from linktools.ai.spec import AgentSpec
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tenacity import RetryCallState, Retrying


def test_openai_operational_settings_do_not_change_durable_identity() -> None:
    first = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://first.example/v1",
        api_key="first-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
        context_window=8192,
    ).snapshot().resolve("default")
    second = ModelRegistry.openai(
        model="openai:gpt-test",
        base_url="https://second.example/v1",
        api_key="second-key",
        timeout=60,
        max_retries=3,
        retry_delay=1,
        max_tokens=2048,
        context_window=8192,
    ).snapshot().resolve("default")

    assert dict(first.semantic_payload) == dict(second.semantic_payload)
    assert first.fingerprint == second.fingerprint


def test_openai_semantic_settings_change_durable_identity() -> None:
    legacy = ModelRegistry.openai(model="gpt-test").snapshot().resolve("default")
    configured = ModelRegistry.openai(
        model="gpt-test",
        max_tokens=2048,
        context_window=8192,
    ).snapshot().resolve("default")

    assert dict(legacy.semantic_payload)["settings"] == {}
    assert dict(configured.semantic_payload)["settings"] == {
        "max_tokens": 2048,
        "context_window": 8192,
    }
    assert legacy.fingerprint != configured.fingerprint


def test_legacy_model_binding_restores_with_current_semantic_settings() -> None:
    legacy = ModelRegistry.openai(model="gpt-test").snapshot().resolve("default")
    registry = ModelRegistry.openai(
        model="gpt-test",
        api_key="test-key",
        max_tokens=2048,
        context_window=8192,
    )

    restored = registry.snapshot().restore(
        dict(legacy.semantic_payload),
        route_id="default",
    )

    assert dict(restored.semantic_payload) == dict(legacy.semantic_payload)
    assert restored.fingerprint == legacy.fingerprint
    model = restored.materialize()
    assert model.settings == {"max_tokens": 2048}
    assert model.profile.get("context_window") == 8192


def test_legacy_agent_binding_restores_with_current_model_settings() -> None:
    spec = AgentSpec("agent")
    legacy_compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=(),
        agents={"agent": spec},
    )
    legacy_binding = legacy_compiler.bind(legacy_compiler.compile(spec))
    current_compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(
            model="gpt-test",
            api_key="test-key",
            max_tokens=2048,
            context_window=8192,
        ).snapshot(),
        candidates=(),
        agents={"agent": spec},
    )

    restored = current_compiler.restore(legacy_binding.snapshot)

    assert restored.digest == legacy_binding.digest
    assert restored.snapshot == legacy_binding.snapshot
    model = restored.definition.model.materialize()
    assert model.settings == {"max_tokens": 2048}
    assert model.profile.get("context_window") == 8192


def test_new_model_binding_requires_exact_semantic_settings() -> None:
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

    assert raised.value.code is ErrorCode.MODEL_CONNECTION_NOT_FOUND


def test_openai_route_materializes_model_settings_and_profile() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        api_key="test-key",
        timeout=30,
        max_retries=1,
        max_tokens=2048,
        context_window=8192,
    ).snapshot().resolve("default")

    model = binding.materialize()

    assert isinstance(model, OpenAIChatModel)
    assert model.settings == {"timeout": 30, "max_tokens": 2048}
    assert model.profile.get("context_window") == 8192
    provider = model.provider
    assert isinstance(provider, OpenAIProvider)
    assert provider.client.max_retries == 1


@pytest.mark.asyncio
async def test_openai_retry_provider_owns_client_and_preserves_default_timeout() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        api_key="test-key",
        max_retries=2,
        retry_delay=0,
    ).snapshot().resolve("default")
    model = binding.materialize()
    provider = model.provider

    assert isinstance(provider, _RetryingOpenAIProvider)
    first_client = provider._http_client
    assert provider.client.max_retries == 0
    assert first_client.timeout.connect == 5
    assert first_client.timeout.read == 600
    assert not first_client.is_closed

    async with model:
        assert not first_client.is_closed
    assert first_client.is_closed

    async with model:
        second_client = provider._http_client
        assert second_client is not first_client
        assert not second_client.is_closed
    assert second_client.is_closed


@pytest.mark.asyncio
async def test_retryable_error_body_is_buffered_before_transport_closes_response() -> None:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, request=request, json={"error": "rate limited"})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    transport = _AsyncClientTransport(client)
    try:
        response = await transport.handle_async_request(
            httpx2.Request("POST", "https://example.test/v1/chat/completions")
        )
        assert response.is_stream_consumed
        assert response.json() == {"error": "rate limited"}
    finally:
        await transport.aclose()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": 0},
        {"timeout": float("inf")},
        {"timeout": True},
        {"max_retries": -1},
        {"max_retries": True},
        {"retry_delay": -0.1, "max_retries": 1},
        {"retry_delay": float("nan"), "max_retries": 1},
        {"retry_delay": 1},
        {"max_tokens": 0},
        {"max_tokens": True},
        {"context_window": 0},
        {"context_window": True},
        {"max_tokens": 8193, "context_window": 8192},
    ],
)
def test_openai_route_rejects_invalid_settings(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelRegistry.openai(model="gpt-test", **kwargs)


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 503])
def test_retryable_http_statuses_raise(status_code: int) -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx2.Response(status_code, request=request)

    with pytest.raises(httpx2.HTTPStatusError):
        _raise_retryable_status(response)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_non_retryable_http_statuses_pass_through(status_code: int) -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx2.Response(status_code, request=request)

    _raise_retryable_status(response)


def test_openai_retry_header_overrides_default_status_policy() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    disabled = httpx2.Response(503, request=request, headers={"x-should-retry": "false"})
    enabled = httpx2.Response(400, request=request, headers={"x-should-retry": "true"})

    _raise_retryable_status(disabled)
    with pytest.raises(httpx2.HTTPStatusError):
        _raise_retryable_status(enabled)


def test_exhausted_http_retry_returns_last_response() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx2.Response(429, request=request)
    error = httpx2.HTTPStatusError(
        "rate limited",
        request=request,
        response=response,
    )

    assert _retry_error_result(_failed_retry_state(error)) is response


def test_exhausted_transport_retry_reraises_last_error() -> None:
    request = httpx2.Request("POST", "https://example.test/v1/chat/completions")
    error = httpx2.ConnectError("connection failed", request=request)

    with pytest.raises(httpx2.ConnectError) as raised:
        _retry_error_result(_failed_retry_state(error))

    assert raised.value is error


def _failed_retry_state(error: BaseException) -> RetryCallState:
    state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    state.set_exception((type(error), error, None))
    return state
