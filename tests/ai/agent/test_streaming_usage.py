#!/usr/bin/env python3
"""Streaming observations remain cumulative across requests and interruption."""

from dataclasses import dataclass
from decimal import Decimal

import pytest

from linktools.ai.agent.engine import AgentEngine
from linktools.ai.execution.snapshots import (
    ModelRequestUsageObservation,
    RequestUsage,
    RunUsageCapture,
)
from linktools.ai.execution.domain import RunUsage


@dataclass(frozen=True, slots=True)
class _RawUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    total_cost: "Decimal | None" = None


@dataclass(frozen=True, slots=True)
class _Response:
    usage: _RawUsage
    provider_name: str = "provider"
    model_name: str = "model"
    provider_response_id: str = "response-1"


class _Stream:
    def __init__(
        self,
        *,
        response: _Response,
        stream_error: "BaseException | None" = None,
        get_error: "BaseException | None" = None,
    ) -> None:
        self._response = response
        self._stream_error = stream_error
        self._get_error = get_error
        self.provider_name = response.provider_name
        self.model_name = response.model_name
        self.provider_response_id = response.provider_response_id

    def __aiter__(self) -> "_Stream":
        return self

    async def __anext__(self) -> object:
        if self._stream_error is not None:
            raise self._stream_error
        raise StopAsyncIteration

    def get(self) -> _Response:
        if self._get_error is not None:
            raise self._get_error
        return self._response

    def usage(self) -> _RawUsage:
        return self._response.usage


class _StreamContext:
    def __init__(self, stream: _Stream) -> None:
        self._stream = stream

    async def __aenter__(self) -> _Stream:
        return self._stream

    async def __aexit__(
        self,
        exc_type: "object | None",
        exc: "object | None",
        traceback: "object | None",
    ) -> bool:
        return False


class _Node:
    def __init__(self, stream: _Stream) -> None:
        self._stream = stream

    def stream(self, run_ctx: object) -> _StreamContext:
        return _StreamContext(self._stream)


class _Cancellation:
    async def raise_if_cancelled(self) -> None:
        return None


class _LiveEvents:
    async def publish(self, event: object) -> None:
        return None


async def _forward_stream(stream: _Stream, observe_usage):
    return await AgentEngine()._forward_model_stream(
        _Node(stream),
        object(),
        cancellation=_Cancellation(),
        live_events=_LiveEvents(),
        request_key="request-1",
        observe_usage=observe_usage,
    )


@pytest.mark.asyncio
async def test_stream_completion_observes_usage_once():
    response = _Response(_RawUsage(input_tokens=3, output_tokens=1, total_tokens=4))
    observed = []

    async def observe(key, usage, observed_response):
        observed.append((key, usage, observed_response))

    assert await _forward_stream(_Stream(response=response), observe) == ""
    assert len(observed) == 1
    assert observed[0][0] == "request-1"
    assert observed[0][1].total_tokens == 4
    assert observed[0][2] is response


@pytest.mark.asyncio
async def test_failed_stream_checkpoints_partial_usage_and_reraises_primary():
    response = _Response(_RawUsage(input_tokens=3, output_tokens=1, total_tokens=4))
    primary = RuntimeError("stream interrupted")
    observed = []

    async def observe(key, usage, observed_response):
        observed.append((key, usage, observed_response))

    stream = _Stream(
        response=response,
        stream_error=primary,
        get_error=RuntimeError("response unavailable"),
    )
    stream.usage = response.usage
    with pytest.raises(RuntimeError) as raised:
        await _forward_stream(stream, observe)

    assert raised.value is primary
    assert len(observed) == 1
    assert observed[0][1].total_tokens == 4
    assert observed[0][2].provider_response_id == "response-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("usage", [None, _RawUsage()])
async def test_failed_stream_without_confirmed_usage_does_not_checkpoint(usage):
    response = _Response(_RawUsage())
    observed = []

    async def observe(key, request_usage, observed_response):
        observed.append((key, request_usage, observed_response))

    stream = _Stream(
        response=response,
        stream_error=RuntimeError("stream interrupted"),
        get_error=RuntimeError("response unavailable"),
    )
    if usage is None:
        stream.usage = lambda: None
    else:
        stream.usage = lambda: usage

    with pytest.raises(RuntimeError, match="stream interrupted"):
        await _forward_stream(stream, observe)
    assert observed == []


@pytest.mark.asyncio
async def test_failed_stream_with_zero_cost_checkpoints_usage():
    response = _Response(_RawUsage(total_cost=Decimal("0")))
    observed = []

    async def observe(key, usage, observed_response):
        observed.append((key, usage, observed_response))

    with pytest.raises(RuntimeError, match="stream interrupted"):
        await _forward_stream(
            _Stream(
                response=response,
                stream_error=RuntimeError("stream interrupted"),
                get_error=RuntimeError("response unavailable"),
            ),
            observe,
        )
    assert len(observed) == 1
    assert observed[0][1].total_cost == Decimal("0")


@pytest.mark.asyncio
async def test_failed_stream_checkpoint_error_replaces_primary_with_cause():
    response = _Response(_RawUsage(input_tokens=1, total_tokens=1))
    primary = RuntimeError("stream interrupted")
    checkpoint_error = RuntimeError("checkpoint failed")

    async def observe(key, usage, observed_response):
        raise checkpoint_error

    with pytest.raises(RuntimeError) as raised:
        await _forward_stream(
            _Stream(
                response=response,
                stream_error=primary,
                get_error=RuntimeError("response unavailable"),
            ),
            observe,
        )
    assert raised.value is checkpoint_error
    assert raised.value.__cause__ is primary


@pytest.mark.asyncio
async def test_completed_stream_checkpoint_error_is_not_retried():
    response = _Response(_RawUsage(input_tokens=1, total_tokens=1))
    checkpoint_error = RuntimeError("checkpoint failed")
    calls = 0

    async def observe(key, usage, observed_response):
        nonlocal calls
        calls += 1
        raise checkpoint_error

    with pytest.raises(RuntimeError) as raised:
        await _forward_stream(_Stream(response=response), observe)
    assert raised.value is checkpoint_error
    assert calls == 1


def test_two_streaming_rounds_do_not_replace_cumulative_usage_with_request_delta():
    capture = RunUsageCapture()
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="stream-1",
            usage=RequestUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            provider_name=None,
            response_model_name=None,
        ),
        pricing=None,
    )
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="stream-2",
            usage=RequestUsage(input_tokens=3, output_tokens=1, total_tokens=4),
            provider_name=None,
            response_model_name=None,
        ),
        pricing=None,
    )

    assert capture.snapshot() == RunUsage(input_tokens=13, output_tokens=3, total_tokens=16)


def test_interrupted_stream_observes_prior_plus_request_local_usage():
    request = RequestUsage(input_tokens=3, output_tokens=1, total_tokens=4)
    capture = RunUsageCapture()
    capture.observe_request(
        ModelRequestUsageObservation(
            request_key="interrupted-stream",
            usage=request,
            provider_name=None,
            response_model_name=None,
        ),
        pricing=None,
    )

    assert capture.snapshot().input_tokens == 3
    assert capture.snapshot().output_tokens == 1
