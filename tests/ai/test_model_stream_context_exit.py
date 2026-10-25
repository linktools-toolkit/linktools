#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The retry wrapper preserves provider context-exit reasons without draining."""

import asyncio
from contextlib import AbstractAsyncContextManager
from types import TracebackType

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from linktools.ai.model import ModelClientRequestRecorder, ModelStreamAttempt
from linktools.ai.model._binding import _RetryingModel
from linktools.ai.model._stream import _StreamContextExit


def _assert_no_control_exception(error: BaseException) -> None:
    pending, seen = [error], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        assert not isinstance(value, _StreamContextExit)
        pending.extend(item for item in (value.__cause__, value.__context__) if item is not None)


class _Scope(AbstractAsyncContextManager[StreamedResponse]):
    def __init__(self, provider: "_Provider", parameters: ModelRequestParameters) -> None:
        self.provider = provider
        self.response = ModelResponse([TextPart("fixture")], model_name="fixture")
        self.stream = CompletedStreamedResponse(self.response, model_request_parameters=parameters, replay_events=True)

    async def __aenter__(self) -> StreamedResponse:
        self.provider.entries += 1
        return self.stream

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                        traceback: TracebackType | None) -> bool:
        self.provider.exits.append((exc_type, exc, traceback))
        if exc is None:
            self.response.provider_response_id = "late-response"
            self.response.usage = RequestUsage(input_tokens=17, output_tokens=2)
        if self.provider.cleanup_error is not None:
            raise self.provider.cleanup_error
        return self.provider.suppress


class _Provider(TestModel):
    def __init__(self, *, cleanup_error: BaseException | None = None, suppress: bool = False) -> None:
        super().__init__()
        self.entries = 0
        self.exits = []
        self.cleanup_error = cleanup_error
        self.suppress = suppress

    def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
    ) -> AbstractAsyncContextManager[StreamedResponse]:
        return _Scope(self, model_request_parameters)


@pytest.mark.asyncio
@pytest.mark.parametrize("consumption", ("none", "first", "all"))
async def test_normal_context_exit_runs_provider_cleanup_once_and_propagates_failure(consumption: str) -> None:
    error = ModelHTTPError(503, "fixture")
    provider = _Provider(cleanup_error=error)
    with pytest.raises(ModelHTTPError) as captured:
        async with _RetryingModel(provider, 2, 0).request_stream([], None, ModelRequestParameters()) as response:
            if consumption == "first":
                await anext(aiter(response))
            elif consumption == "all":
                async for _ in response:
                    pass
    assert captured.value is error
    _assert_no_control_exception(captured.value)
    assert provider.entries == 1
    assert provider.exits == [(None, None, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (ValueError, asyncio.CancelledError, GeneratorExit))
@pytest.mark.parametrize("cleanup_fails", (False, True))
async def test_consumer_exit_reason_reaches_provider_and_keeps_primary_error(
    error_type: type[BaseException], cleanup_fails: bool,
) -> None:
    error = error_type("fixture")
    previous_cause, previous_context = LookupError("prior cause"), RuntimeError("prior context")
    error.__cause__, error.__context__ = previous_cause, previous_context
    provider = _Provider(cleanup_error=ModelHTTPError(503, "fixture") if cleanup_fails else None)
    with pytest.raises(error_type) as captured:
        async with _RetryingModel(provider, 2, 0).request_stream([], None, ModelRequestParameters()):
            raise error
    assert captured.value is error
    assert captured.value.__cause__ is previous_cause
    assert captured.value.__context__ is previous_context
    _assert_no_control_exception(captured.value)
    assert provider.entries == len(provider.exits) == 1
    assert provider.exits[0][:2] == (error_type, error)
    assert provider.exits[0][2] is not None


@pytest.mark.asyncio
async def test_provider_can_suppress_consumer_error_without_draining_or_retry() -> None:
    error = ValueError("fixture")
    provider = _Provider(suppress=True)
    async with _RetryingModel(provider, 2, 0).request_stream([], None, ModelRequestParameters()):
        raise error
    assert provider.entries == len(provider.exits) == 1
    assert provider.exits[0][:2] == (ValueError, error)


@pytest.mark.asyncio
async def test_normal_early_close_keeps_late_metadata_and_usage_without_successful_consumption() -> None:
    provider = _Provider()
    observed = []
    async def observer(attempt: ModelStreamAttempt) -> None:
        observed.append(attempt)
    with ModelClientRequestRecorder(stream_attempt_observer=observer):
        async with _RetryingModel(provider, 2, 0).request_stream([], None, ModelRequestParameters()) as response:
            pass
        assert response.get().provider_response_id == "late-response"
        assert response.usage.input_tokens == 17
        await response.aclose()
    assert provider.entries == len(provider.exits) == 1
    assert [attempt.status for attempt in observed] == ["RUNNING", "CANCELLED"]
    assert observed[-1].usage.input_tokens == 17
