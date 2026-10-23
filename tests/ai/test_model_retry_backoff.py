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
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelRequest, ModelResponse, UserPromptPart
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
    await request(_RetryingModel(provider, 4, 1, retry_max_delay=120), streaming)
    assert provider.calls == 2
    assert recorded_waits == [3600]


@pytest.mark.asyncio
async def test_existing_large_initial_delay_is_not_shortened(recorded_waits: list[float]) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(2)])
    await request(_RetryingModel(provider, 2, 120), False)
    assert recorded_waits == [120, 120]


@pytest.mark.asyncio
@pytest.mark.parametrize("base,cap,expected", (
    (20, 10, [20, 20, 20]),
    (20, 30, [25, 30, 30]),
    (0, 0, [0, 0, 0]),
    (0.5, 0.75, [0.625, 0.75, 0.75]),
))
async def test_configured_cap_bounds_jitter_without_shortening_initial_base(
    recorded_waits: list[float], base: float, cap: float, expected: list[float],
) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(3)])
    await request(_RetryingModel(provider, 3, base, retry_max_delay=cap), False)
    assert recorded_waits == expected


@pytest.mark.parametrize("invalid", (-1, float("nan"), float("inf"), True, None, "120", 10**400))
def test_public_retry_max_delay_requires_finite_non_negative_number(invalid: object) -> None:
    with pytest.raises(ValueError, match="retry_max_delay must be a finite non-negative number"):
        create_model(protocol="openai", model="fixture", retry_max_delay=invalid)


@pytest.mark.parametrize("invalid", (0, 0.5, float("nan"), float("inf"), True, None, "2", 10**400))
def test_public_retry_backoff_factor_requires_finite_number_at_least_one(invalid: object) -> None:
    with pytest.raises(ValueError, match="retry_backoff_factor"):
        create_model(protocol="openai", model="fixture", retry_backoff_factor=invalid)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("factor,expected", ((1, [1.25, 1.25, 1.25]), (1.5, [1.25, 1.875, 2.8125])))
async def test_backoff_factor_controls_base_growth_and_retains_jitter(
    recorded_waits: list[float], streaming: bool, factor: float, expected: list[float],
) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(3)])
    await request(_RetryingModel(provider, 3, 1, retry_backoff_factor=factor), streaming)
    assert provider.calls == 4
    assert recorded_waits == expected


@pytest.mark.asyncio
async def test_zero_base_remains_zero_with_large_factor(recorded_waits: list[float]) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(3)])
    await request(_RetryingModel(provider, 3, 0, retry_max_delay=5e-324,
                                 retry_backoff_factor=1e308), False)
    assert recorded_waits == [0, 0, 0]


@pytest.mark.asyncio
async def test_subnormal_initial_base_grows_before_reaching_cap(recorded_waits: list[float]) -> None:
    provider = ScriptedProvider([ModelHTTPError(429, "fixture") for _ in range(3)])
    await request(_RetryingModel(provider, 3, 5e-324, retry_backoff_factor=1e308), False)
    assert recorded_waits[0] == 5e-324
    assert 0 < recorded_waits[1] < 1e-15
    assert recorded_waits[2] == 60


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
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("base,cap,factor", ((1, 60, 2), (5, 120, 4)))
async def test_jitter_stays_within_additive_bounds(
    monkeypatch: pytest.MonkeyPatch, recorded_waits: list[float], fraction: float,
    streaming: bool, base: float, cap: float, factor: float,
) -> None:
    monkeypatch.setattr("linktools.ai.model._binding.random.uniform",
                        lambda lower, upper: lower + (upper - lower) * fraction)
    provider = ScriptedProvider([ModelHTTPError(503, "fixture") for _ in range(3)])
    await request(_RetryingModel(provider, 3, base, retry_max_delay=cap,
                                 retry_backoff_factor=factor), streaming)
    assert recorded_waits == [base * multiplier * (1 + 0.25 * fraction)
                              for multiplier in (1, factor, factor * factor)]


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
@pytest.mark.parametrize("protocol", ("openai", "openai-responses", "anthropic"))
async def test_public_binding_passes_custom_backoff_to_each_protocol(
    monkeypatch: pytest.MonkeyPatch, recorded_waits: list[float], protocol: str,
) -> None:
    from .test_model_protocols import response

    sent = []
    clients = []

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        sent.append(http_request)
        if len(sent) <= 3:
            return httpx2.Response(429, json={"type": "error", "error": {
                "type": "rate_limit_error", "code": "rate_limit_exceeded", "message": "synthetic throttled"}})
        return httpx2.Response(200, json=response(protocol))

    def client(**kwargs: object) -> httpx2.AsyncClient:
        result = httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)
        clients.append(result)
        return result

    module = "_anthropic" if protocol == "anthropic" else "_openai"
    monkeypatch.setattr(f"linktools.ai.model.{module}.DefaultAsyncHttpxClient", client)
    binding = create_model(protocol=protocol, model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=3,
                           retry_delay=5, retry_max_delay=120, retry_backoff_factor=4)
    assert binding.contract == create_model(protocol=protocol, model="fixture").contract
    async with binding.materialize() as model:
        result = await model.request([ModelRequest([UserPromptPart("synthetic")])],
                                     None, ModelRequestParameters())
    assert result.parts[0].content == "answer"
    assert len(sent) == 4 and all(item.is_closed for item in clients)
    assert recorded_waits == [6.25, 25, 100]


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
@pytest.mark.parametrize("factor,retries", ((2, 1100), (1.5, 2000), (1e308, 3)))
async def test_many_retries_cannot_overflow_backoff(
    recorded_waits: list[float], factor: float, retries: int,
) -> None:
    provider = ScriptedProvider([ModelHTTPError(503, "fixture") for _ in range(retries)])
    await request(_RetryingModel(provider, retries, 1, retry_backoff_factor=factor), False)
    assert provider.calls == retries + 1
    assert len(recorded_waits) == retries
    assert recorded_waits[0] == 1.25
    assert recorded_waits[-1] == 60
