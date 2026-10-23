#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model transport backoff preserves provider hints and cancellation boundaries."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

import pytest
import httpx2
from pydantic_ai import Agent
from pydantic_ai import exceptions as sdk_errors
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from linktools.ai.model._binding import _RetryingModel
from linktools.ai.model import create_model


class ScriptedProvider(TestModel):
    def __init__(self, errors: list[BaseException]) -> None:
        super().__init__()
        self.errors = list(errors)
        self.calls = 0

    def next_error(self) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.next_error()
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[object] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        self.next_error()
        async with super().request_stream(messages, model_settings,
                                          model_request_parameters, run_context) as stream:
            yield stream


async def request(model: _RetryingModel, streaming: bool) -> None:
    if streaming:
        async with model.request_stream([], None, ModelRequestParameters()) as stream:
            async for _ in stream:
                pass
    else:
        await model.request([], None, ModelRequestParameters())


@pytest.fixture
def recorded_waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits = []

    async def sleep(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr("linktools.ai.model._binding.asyncio.sleep", sleep)
    monkeypatch.setattr("linktools.ai.model._binding.random.uniform", lambda lower, upper: upper)
    return waits


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_retry_waits_grow_with_bounded_jitter_and_cap(
    recorded_waits: list[float], streaming: bool,
) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(9)])
    await request(_RetryingModel(provider, 9, 1), streaming)
    assert provider.calls == 10
    assert len(recorded_waits) == 9
    assert recorded_waits == sorted(recorded_waits)
    for index, delay in enumerate(recorded_waits):
        base = min(60, 2 ** index)
        assert base <= delay <= min(60, base * 1.25)
    assert recorded_waits[-1] == 60


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_retry_after_is_a_minimum_not_a_replacement_for_backoff(
    recorded_waits: list[float], streaming: bool,
) -> None:
    provider = ScriptedProvider([
        ModelHTTPError(429, "fixture", headers={"Retry-After": "20"}),
        ModelHTTPError(503, "fixture"),
    ])
    await request(_RetryingModel(provider, 2, 1), streaming)
    assert provider.calls == 3
    assert recorded_waits[0] >= 20
    assert 2 <= recorded_waits[1] <= 2.5


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_server_delay_beyond_local_cap_is_not_shortened(
    recorded_waits: list[float], streaming: bool,
) -> None:
    error = ModelHTTPError(429, "fixture", headers={"retry-after": "3600"})
    provider = ScriptedProvider([error])
    await request(_RetryingModel(provider, 4, 1), streaming)
    assert provider.calls == 2
    assert recorded_waits == [3600]


@pytest.mark.asyncio
async def test_existing_large_initial_delay_is_not_shortened(recorded_waits: list[float]) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(2)])
    await request(_RetryingModel(provider, 2, 120), False)
    assert recorded_waits == [120, 120]


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ("invalid", "-1", "NaN", "Infinity"))
async def test_invalid_server_delay_falls_back_to_local_backoff(
    recorded_waits: list[float], header: str,
) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture", headers={"retry-after": header})])
    await request(_RetryingModel(provider, 1, 1), False)
    assert len(recorded_waits) == 1
    assert 1 <= recorded_waits[0] <= 1.25


@pytest.mark.asyncio
async def test_retry_after_http_date_uses_sdk_normalization(recorded_waits: list[float]) -> None:
    header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=20), usegmt=True)
    error = ModelHTTPError(429, "fixture", headers={"Retry-After": header})
    expected = error.retry_after
    assert expected is not None and expected > 18
    await request(_RetryingModel(ScriptedProvider([error]), 1, 1), False)
    assert recorded_waits[0] >= expected - 0.1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_zero_delay_and_retry_count_keep_existing_meaning(
    recorded_waits: list[float], streaming: bool,
) -> None:
    provider = ScriptedProvider([ModelHTTPError(503, "fixture") for _ in range(2)])
    await request(_RetryingModel(provider, 2, 0), streaming)
    assert recorded_waits == [0, 0]
    error = ModelHTTPError(503, "fixture")
    provider = ScriptedProvider([error])
    with pytest.raises(ModelHTTPError) as caught:
        await request(_RetryingModel(provider, 0, 1), streaming)
    assert caught.value is error and provider.calls == 1
    assert recorded_waits == [0, 0]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("status", (400, 401, 403, 404, 422))
async def test_permanent_http_failures_do_not_wait_or_retry(
    recorded_waits: list[float], streaming: bool, status: int,
) -> None:
    error = ModelHTTPError(status, "fixture")
    provider = ScriptedProvider([error])
    with pytest.raises(ModelHTTPError) as caught:
        await request(_RetryingModel(provider, 2, 1), streaming)
    assert caught.value is error and provider.calls == 1 and not recorded_waits


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_known_quota_exhaustion_does_not_retry_as_temporary_429(
    recorded_waits: list[float], streaming: bool,
) -> None:
    if not hasattr(sdk_errors, "ModelQuotaExceededError"):
        pytest.skip("SDK has no public typed quota error")
    error = ModelHTTPError.for_category(sdk_errors.ModelQuotaExceededError,
                                        status_code=429, model_name="fixture")
    provider = ScriptedProvider([error])
    with pytest.raises(sdk_errors.ModelQuotaExceededError) as caught:
        await request(_RetryingModel(provider, 2, 1), streaming)
    assert caught.value is error and provider.calls == 1 and not recorded_waits


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_expired_overall_request_deadline_never_restarts(
    recorded_waits: list[float], streaming: bool,
) -> None:
    if not hasattr(sdk_errors, "ModelRequestTimeout"):
        pytest.skip("SDK has no public request_timeout deadline")
    error = sdk_errors.ModelRequestTimeout("fixture", 30)
    provider = ScriptedProvider([error])
    with pytest.raises(sdk_errors.ModelRequestTimeout) as caught:
        await request(_RetryingModel(provider, 2, 1), streaming)
    assert caught.value is error and provider.calls == 1 and not recorded_waits


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_cancel_during_backoff_never_dispatches_another_attempt(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    waiting = asyncio.Event()

    async def sleep(_delay: float) -> None:
        waiting.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("linktools.ai.model._binding.asyncio.sleep", sleep)
    provider = ScriptedProvider([ModelHTTPError(429, "fixture")])
    active = asyncio.create_task(request(_RetryingModel(provider, 2, 1), streaming))
    await waiting.wait()
    active.cancel()
    with pytest.raises(asyncio.CancelledError):
        await active
    assert provider.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_public_request_deadline_covers_all_backoff_waits(streaming: bool) -> None:
    if not hasattr(sdk_errors, "ModelRequestTimeout"):
        pytest.skip("SDK has no public request_timeout deadline")
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(3)])
    model = _RetryingModel(provider, 3, 1)

    async def consume(_ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    with pytest.raises(sdk_errors.ModelRequestTimeout):
        await Agent(model).run("synthetic", model_settings={"request_timeout": 0.01},
                               **({"event_stream_handler": consume} if streaming else {}))
    assert provider.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fraction", (0.0, 1.0))
async def test_jitter_stays_within_additive_bounds(
    monkeypatch: pytest.MonkeyPatch, recorded_waits: list[float], fraction: float,
) -> None:
    monkeypatch.setattr("linktools.ai.model._binding.random.uniform",
                        lambda lower, upper: lower + (upper - lower) * fraction)
    provider = ScriptedProvider([ModelHTTPError(503, "fixture") for _ in range(2)])
    await request(_RetryingModel(provider, 2, 1), False)
    assert recorded_waits == [1 + 0.25 * fraction, 2 + 0.5 * fraction]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), True, -1))
async def test_invalid_normalized_provider_wait_uses_local_delay(
    recorded_waits: list[float], invalid: int | float | bool,
) -> None:
    class InvalidWait(ModelHTTPError):
        @property
        def retry_after(self) -> int | float | bool:
            return invalid

        @retry_after.setter
        def retry_after(self, value: int | float | None) -> None:
            pass

    provider = ScriptedProvider([InvalidWait(429, "fixture")])
    await request(_RetryingModel(provider, 1, 1), False)
    assert recorded_waits == [1.25]


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ("date", "3600", "invalid", "-1"))
async def test_public_binding_uses_sdk_normalized_retry_after(
    monkeypatch: pytest.MonkeyPatch, recorded_waits: list[float], header: str,
) -> None:
    sent = []
    clients = []
    if header == "date":
        header = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=20), usegmt=True)

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        sent.append(http_request)
        if len(sent) == 1:
            return httpx2.Response(429, headers={"Retry-After": header}, json={
                "error": {"message": "synthetic throttled", "code": "rate_limit_exceeded"}})
        return httpx2.Response(200, json={
            "id": "synthetic", "object": "chat.completion", "created": 1,
            "model": "fixture", "choices": [{"index": 0, "message": {
                "role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        })

    def client(**kwargs: object) -> httpx2.AsyncClient:
        result = httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr("linktools.ai.model._openai.DefaultAsyncHttpxClient", client)
    binding = create_model(protocol="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=1, retry_delay=1)
    async with binding.materialize() as model:
        response = await model.request([], None, ModelRequestParameters())
    assert response.parts[0].content == "ok"
    assert len(sent) == 2 and all(item.is_closed for item in clients)
    if header == "3600":
        assert recorded_waits == [3600]
    elif header in {"invalid", "-1"}:
        assert recorded_waits == [1.25]
    else:
        assert 18 < recorded_waits[0] <= 20


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_public_deadline_interrupts_server_wait_beyond_local_cap(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    if not hasattr(sdk_errors, "ModelRequestTimeout"):
        pytest.skip("SDK has no public request_timeout deadline")
    sent = []
    clients = []

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        sent.append(http_request)
        return httpx2.Response(429, headers={"retry-after": "3600"}, json={
            "error": {"message": "synthetic throttled", "code": "rate_limit_exceeded"}})

    def client(**kwargs: object) -> httpx2.AsyncClient:
        result = httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)
        clients.append(result)
        return result

    async def consume(_ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    monkeypatch.setattr("linktools.ai.model._openai.DefaultAsyncHttpxClient", client)
    binding = create_model(protocol="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=2, retry_delay=1)
    async with Agent(binding.materialize()) as agent:
        with pytest.raises(sdk_errors.ModelRequestTimeout):
            await agent.run("synthetic", model_settings={"request_timeout": 0.03},
                            **({"event_stream_handler": consume} if streaming else {}))
    assert len(sent) == 1 and all(item.is_closed for item in clients)


@pytest.mark.asyncio
async def test_many_retries_cannot_overflow_backoff(recorded_waits: list[float]) -> None:
    provider = ScriptedProvider([ModelHTTPError(503, "fixture") for _ in range(1100)])
    await request(_RetryingModel(provider, 1100, 1), False)
    assert len(recorded_waits) == 1100
    assert recorded_waits[-1] == 60
