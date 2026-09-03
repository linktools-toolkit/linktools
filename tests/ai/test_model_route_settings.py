#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model route configuration coverage."""

from typing import Any

import httpx2
import pytest
from linktools.ai.model import ModelRegistry
from linktools.ai.model._openai import _raise_retryable_status, _retry_error_result
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tenacity import RetryCallState, Retrying


def test_openai_route_settings_do_not_change_durable_identity() -> None:
    legacy = ModelRegistry.openai(
        model="gpt-test",
        base_url="https://legacy.example/v1",
        api_key="legacy-key",
    ).snapshot().resolve("default")
    registry = ModelRegistry.openai(
        model="openai:gpt-test",
        base_url="https://current.example/v1",
        api_key="current-key",
        timeout_seconds=60,
        max_retries=3,
        retry_delay_seconds=1,
        max_output_tokens=8192,
        context_window_tokens=128000,
    )
    configured = registry.snapshot().resolve("default")

    assert dict(configured.semantic_payload) == dict(legacy.semantic_payload)
    assert configured.fingerprint == legacy.fingerprint
    assert registry.snapshot().restore(
        dict(legacy.semantic_payload),
        route_id="default",
    ) is configured


def test_openai_route_materializes_model_settings_and_profile() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        api_key="test-key",
        timeout_seconds=30,
        max_retries=1,
        max_output_tokens=2048,
        context_window_tokens=8192,
    ).snapshot().resolve("default")

    model = binding.materialize()

    assert isinstance(model, OpenAIChatModel)
    assert model.settings == {"timeout": 30, "max_tokens": 2048}
    assert model.profile.get("context_window") == 8192
    provider = model.provider
    assert isinstance(provider, OpenAIProvider)
    assert provider.client.max_retries == 1


def test_openai_retry_delay_disables_sdk_retry_layer() -> None:
    binding = ModelRegistry.openai(
        model="gpt-test",
        api_key="test-key",
        max_retries=2,
        retry_delay_seconds=0,
    ).snapshot().resolve("default")

    model = binding.materialize()

    provider = model.provider
    assert isinstance(provider, OpenAIProvider)
    assert provider.client.max_retries == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
        {"max_retries": -1},
        {"max_retries": True},
        {"retry_delay_seconds": -0.1, "max_retries": 1},
        {"retry_delay_seconds": float("nan"), "max_retries": 1},
        {"retry_delay_seconds": 1},
        {"max_output_tokens": 0},
        {"max_output_tokens": True},
        {"context_window_tokens": 0},
        {"context_window_tokens": True},
        {"max_output_tokens": 8193, "context_window_tokens": 8192},
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
