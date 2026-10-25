#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider categories and response validation keep finite retry boundaries."""

import json
from collections.abc import AsyncIterable

import httpx2
import pytest
from openai.types.chat import ChatCompletion
from pydantic_ai import Agent, exceptions as sdk_errors
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import create_model, model_binding_error
from linktools.ai.model._binding import _RetryingModel
from linktools.ai.runtime._agent_executor import _execution_error

from .test_model_error_classification import _mock_clients
from .test_model_retry_backoff import ScriptedProvider, request


def _project(error: Exception) -> AIError:
    return _execution_error(error, usage_limits=UsageLimits(), run_usage=RunUsage())


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("status", (None, 400, 429))
async def test_typed_context_window_error_never_retries_unchanged_input(streaming: bool, status: int | None) -> None:
    category = getattr(sdk_errors, "ModelContextWindowExceededError")
    error = (category("fixture", "synthetic overflow") if status is None else
             ModelHTTPError.for_category(category, status_code=status, model_name="fixture"))
    provider = ScriptedProvider([error])
    with pytest.raises(category) as caught:
        await request(_RetryingModel(provider, 2, 0), streaming)
    assert caught.value is error and provider.calls == 1
    projected = model_binding_error(error)
    assert projected.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert projected.retryable is False
    assert projected.safe_details == {
        "model_name": "fixture", "reason": "context_window_exceeded",
        **({"status_code": status} if status is not None else {}),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_opaque_error_message_does_not_invent_permanent_category(streaming: bool) -> None:
    error = ModelAPIError("fixture", "context_length_exceeded: unstructured text")
    provider = ScriptedProvider([error, error, error])
    with pytest.raises(ModelAPIError) as caught:
        await request(_RetryingModel(provider, 2, 0), streaming)
    assert caught.value is error and provider.calls == 3
    projected = model_binding_error(error)
    assert projected.code is ErrorCode.MODEL_API_ERROR
    assert "reason" not in projected.safe_details


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ("context_length_exceeded", "invalid_request_error"))
@pytest.mark.parametrize("streaming", (False, True))
async def test_responses_context_category_uses_available_sdk_structure(
    monkeypatch: pytest.MonkeyPatch, code: str, streaming: bool,
) -> None:
    sent = []
    error = {"code": code, "message": "maximum context length exceeded"}
    body = (b'event: error\ndata: ' + json.dumps({
        "type": "error", **error, "sequence_number": 1,
    }).encode() + b'\n\n' if streaming else json.dumps({
        "id": "fixture-response", "object": "response", "created_at": 1,
        "model": "fixture", "status": "failed", "error": error, "output": [],
    }).encode())

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        sent.append(http_request)
        return httpx2.Response(200, headers={
            "content-type": "text/event-stream" if streaming else "application/json",
        }, content=body)

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(protocol="openai-responses", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0)
    async with binding.materialize() as model:
        with pytest.raises(ModelAPIError) as caught:
            await request(model, streaming)
    projected = model_binding_error(caught.value)
    assert len(sent) == 1
    assert projected.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert projected.safe_details["reason"] == "context_window_exceeded"
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("choices", ([], None))
async def test_nonstream_malformed_chat_choices_are_model_response_errors(
    monkeypatch: pytest.MonkeyPatch, choices: list[object] | None,
) -> None:
    sent = []

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        sent.append(http_request)
        return httpx2.Response(200, json={
            "id": "fixture", "object": "chat.completion", "created": 1,
            "model": "fixture", "choices": choices,
        })

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(protocol="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0)
    async with binding.materialize() as model:
        with pytest.raises(UnexpectedModelBehavior) as caught:
            await model.request([], None, ModelRequestParameters())
    assert len(sent) == 1
    assert _project(caught.value).code is ErrorCode.MODEL_RESPONSE_INVALID
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_internal_index_error_is_not_reclassified_as_provider_response(monkeypatch: pytest.MonkeyPatch) -> None:
    error = IndexError("synthetic implementation failure")

    def invalid_hook(self: OpenAIChatModel, response: ChatCompletion) -> ChatCompletion:
        raise error

    clients = _mock_clients(monkeypatch, lambda _: httpx2.Response(200, json={
        "id": "fixture", "object": "chat.completion", "created": 1, "model": "fixture",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
    }))
    monkeypatch.setattr(OpenAIChatModel, "_validate_completion", invalid_hook)
    binding = create_model(protocol="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0)
    async with binding.materialize() as model:
        with pytest.raises(IndexError) as caught:
            await model.request([], None, ModelRequestParameters())
    assert caught.value is error
    assert _project(error).code is ErrorCode.INTERNAL_ERROR
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_consumer_error_with_provider_cause_is_not_a_transport_retry(streaming: bool) -> None:
    error = ValueError("synthetic consumer failure")
    error.__cause__ = ModelHTTPError(503, "fixture")
    provider = ScriptedProvider([error])
    with pytest.raises(ValueError) as caught:
        await request(_RetryingModel(provider, 2, 0), streaming)
    assert caught.value is error and provider.calls == 1
    assert _project(error).code is ErrorCode.INTERNAL_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("shape,expected_calls", (("zero_eof", 1), ("blank", 2), ("length", 1)))
async def test_empty_output_keeps_sdk_validation_and_correction_budget(
    monkeypatch: pytest.MonkeyPatch, shape: str, expected_calls: int,
) -> None:
    sent = []
    body = b"" if shape == "zero_eof" else b'data: ' + json.dumps({
        "id": "fixture", "object": "chat.completion.chunk", "created": 1, "model": "fixture",
        "choices": [{"index": 0, "delta": {"role": "assistant"},
                     "finish_reason": "length" if shape == "length" else "stop"}],
    }).encode() + b'\n\ndata: [DONE]\n\n'

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        sent.append(http_request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async def consume(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(protocol="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0)
    async with Agent(binding.materialize(), retries=1) as agent:
        with pytest.raises(UnexpectedModelBehavior) as caught:
            await agent.run("fixture", event_stream_handler=consume)
    assert len(sent) == expected_calls
    assert _project(caught.value).code is ErrorCode.MODEL_RESPONSE_INVALID
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("category,expected", (
    ("ModelRateLimitError", ErrorCode.MODEL_RATE_LIMITED),
    ("ModelUnavailableError", ErrorCode.MODEL_UNAVAILABLE),
    ("ModelOverloadedError", ErrorCode.MODEL_UNAVAILABLE),
    ("ModelServerError", ErrorCode.MODEL_UNAVAILABLE),
))
@pytest.mark.parametrize("in_stream", (False, True))
def test_typed_transient_failure_preserves_category_and_retry_hint(
    category: str, expected: ErrorCode, in_stream: bool,
) -> None:
    error_type = getattr(sdk_errors, category)
    error = error_type("fixture", "synthetic failure", retry_after=17, in_stream=in_stream)
    projected = model_binding_error(error)
    assert projected.code is expected
    assert projected.retryable is not in_stream
    assert projected.safe_details == {"model_name": "fixture", "retry_after": 17}


@pytest.mark.asyncio
@pytest.mark.parametrize("code,expected", (
    ("rate_limit_exceeded", ErrorCode.MODEL_RATE_LIMITED),
    ("server_error", ErrorCode.MODEL_UNAVAILABLE),
    ("overloaded_error", ErrorCode.MODEL_UNAVAILABLE),
))
async def test_nonstream_responses_failed_body_keeps_transient_category(
    monkeypatch: pytest.MonkeyPatch, code: str, expected: ErrorCode,
) -> None:
    sent = []

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        sent.append(http_request)
        return httpx2.Response(200, json={
            "id": "fixture-response", "object": "response", "created_at": 1,
            "model": "fixture", "status": "failed", "output": [],
            "error": {"code": code, "message": "synthetic failure"},
        })

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(protocol="openai-responses", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0)
    async with binding.materialize() as model:
        with pytest.raises(ModelAPIError) as caught:
            await model.request([], None, ModelRequestParameters())
    assert len(sent) == 3
    projected = model_binding_error(caught.value)
    assert projected.code is expected
    assert projected.retryable is True
    assert all(client.is_closed for client in clients)
