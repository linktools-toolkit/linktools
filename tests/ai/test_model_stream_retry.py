#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Stream retry commitment follows the first outward SDK event, not headers."""

import asyncio
import collections
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx2
import pytest
from pydantic_ai import Agent, exceptions as sdk_errors
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import TraceCapability
from linktools.ai.model import ModelClientRequestRecorder, create_model
from linktools.ai.model import _binding, _openai
from linktools.ai.model._binding import _RetryingModel
from linktools.ai.model._stream import _RetryingStreamedResponse
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage


def _event(delta: dict[str, object], *, finish: str | None = None) -> bytes:
    return b"data: " + json.dumps({
        "id": "synthetic", "object": "chat.completion.chunk", "created": 1,
        "model": "fixture", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }).encode() + b"\n\n"


_ROLE = _event({"role": "assistant"})
_PARTIAL = _event({"role": "assistant", "content": "discarded"})
_SUCCESS = _event({"role": "assistant", "content": "winner"}, finish="stop") + b"data: [DONE]\n\n"


class _Source(httpx2.AsyncByteStream):
    def __init__(self, chunks: list[bytes | BaseException], *, blocked: asyncio.Event | None = None) -> None:
        self.chunks = iter(chunks)
        self.blocked = blocked
        self.closes = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk
        if self.blocked is not None:
            self.blocked.set()
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closes += 1


def _model(
    monkeypatch: pytest.MonkeyPatch, chunks: Callable[[int], list[bytes | BaseException]],
    *, retries: int = 3, blocked: asyncio.Event | None = None,
    retry_frames: list[int] | None = None,
):
    sources: list[_Source] = []
    depths: list[int] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        frame = sys._getframe()
        depth = 0
        retry_depth = 0
        while frame is not None:
            depth += 1
            retry_depth += frame.f_code is _RetryingStreamedResponse._iterate_attempts.__code__
            frame = frame.f_back
        depths.append(depth)
        if retry_frames is not None:
            retry_frames.append(retry_depth)
        source = _Source(chunks(len(depths)), blocked=blocked)
        sources.append(source)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=source)

    def client(**kwargs: object) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(_openai, "DefaultAsyncHttpxClient", client)
    binding = create_model(protocol="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=retries,
                           retry_delay=5, retry_max_delay=120, retry_backoff_factor=4)
    return binding, sources, depths


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    result: list[float] = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        caller = sys._getframe(1)
        if caller.f_code.co_name == "_wait_for_retry" and caller.f_code.co_filename == _binding.__file__:
            result.append(delay)
            await real_sleep(0)
        else:
            await real_sleep(delay)

    monkeypatch.setattr(_binding.asyncio, "sleep", sleep)
    monkeypatch.setattr(_binding.random, "uniform", lambda lower, upper: lower)
    return result


async def _consume(ctx, events) -> None:
    async for _ in events:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", (b"", b'data: {"incomplete":', _ROLE))
async def test_stream_disconnect_before_first_event_retries_without_accumulating_output(
    monkeypatch: pytest.MonkeyPatch, waits: list[float], prefix: bytes,
) -> None:
    retry_frames: list[int] = []
    binding, sources, depths = _model(monkeypatch, lambda attempt: (
        [prefix, httpx2.RemoteProtocolError("synthetic disconnect")] if attempt <= 3 else [_SUCCESS]
    ), retry_frames=retry_frames)
    with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
        result = await Agent(binding.materialize()).run("synthetic", event_stream_handler=_consume)
    assert result.output == "winner"
    # Await wrappers can unwind after the first suspension; retries must not add frames.
    assert len(depths) == 4 and len(set(depths[1:])) == 1 and max(depths[1:]) <= depths[0]
    assert retry_frames == [1] * 4
    assert waits == [5, 20, 80]
    assert all(source.closes == 1 for source in sources)
    assert [request.response.body for request in recorder.requests] == [prefix] * 3 + [_SUCCESS]


@pytest.mark.asyncio
async def test_runtime_role_only_disconnect_retries_in_one_logical_request(
    monkeypatch: pytest.MonkeyPatch, waits: list[float],
) -> None:
    binding, sources, depths = _model(monkeypatch, lambda attempt: (
        [_ROLE, httpx2.RemoteProtocolError("synthetic disconnect")] if attempt <= 3 else [_SUCCESS]
    ))
    features = RuntimeFeatures("stream-retry")
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.model("default", binding)
    features.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    async with Persistence.open('stream-retry', storage=PersistenceStorage.in_memory()) as persistence_148, Runtime.open(persistence=persistence_148, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == "SUCCEEDED"
        interactions = (await execution.model_interactions(include_client_requests=True)).items
        assert len(interactions) == 1 and len(interactions[0].client_requests) == 4
    assert len(depths) == 4 and waits == [5, 20, 80]
    assert all(source.closes == 1 for source in sources)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", (0, 3))
async def test_retry_limit_still_bounds_role_only_failures(
    monkeypatch: pytest.MonkeyPatch, waits: list[float], limit: int,
) -> None:
    binding, sources, depths = _model(monkeypatch, lambda attempt: [
        _ROLE, httpx2.RemoteProtocolError("synthetic disconnect"),
    ], retries=limit)
    with pytest.raises(ModelAPIError):
        await Agent(binding.materialize()).run("synthetic", event_stream_handler=_consume)
    assert len(depths) == limit + 1 and waits == [5, 20, 80][:limit]
    assert all(source.closes == 1 for source in sources)


@pytest.mark.asyncio
async def test_visible_partial_stream_is_not_silently_replayed(
    monkeypatch: pytest.MonkeyPatch, waits: list[float],
) -> None:
    binding, sources, depths = _model(monkeypatch, lambda attempt: [
        _PARTIAL, httpx2.RemoteProtocolError("synthetic disconnect"),
    ])
    with pytest.raises(ModelAPIError):
        await Agent(binding.materialize()).run("synthetic", event_stream_handler=_consume)
    assert len(depths) == 1 and waits == [] and sources[0].closes == 1


@pytest.mark.asyncio
async def test_consumer_failure_after_handover_never_retries(waits: list[float]) -> None:
    model = _RetryingModel(TestModel(), 3, 5)
    error = ModelHTTPError(503, "fixture")
    with pytest.raises(ModelHTTPError) as caught:
        async with model.request_stream([], None, ModelRequestParameters()):
            raise error
    assert caught.value is error and waits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", (False, True))
async def test_lookahead_preserves_exact_native_event_sequence(fallback: bool) -> None:
    async def stream(messages, info):
        yield "first "
        yield "second"

    events: list[str] = []

    async def consume(ctx, source) -> None:
        async for event in source:
            events.append(type(event).__name__)

    model = _RetryingModel(FunctionModel(stream_function=stream), 3, 5)
    result = await Agent(FallbackModel(model) if fallback else model).run(
        "synthetic", event_stream_handler=consume,
    )
    assert result.output == "first second"
    assert collections.Counter(events) == {
        "PartStartEvent": 1, "PartDeltaEvent": 1, "PartEndEvent": 1, "FinalResultEvent": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", (False, True))
async def test_request_deadline_survives_lookahead_and_native_fallback(fallback: bool) -> None:

    async def stream(messages, info):
        yield "first "
        await asyncio.Event().wait()

    model = _RetryingModel(FunctionModel(stream_function=stream), 3, 5)
    with pytest.raises(sdk_errors.ModelRequestTimeout):
        await asyncio.wait_for(Agent(FallbackModel(model) if fallback else model).run(
            "synthetic", model_settings={"request_timeout": 0.02}, event_stream_handler=_consume,
        ), 1)


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_first_event_closes_without_retry(
    monkeypatch: pytest.MonkeyPatch, waits: list[float],
) -> None:
    blocked = asyncio.Event()
    binding, sources, depths = _model(monkeypatch, lambda attempt: [_ROLE], blocked=blocked)
    task = asyncio.create_task(Agent(binding.materialize()).run("synthetic", event_stream_handler=_consume))
    await asyncio.wait_for(blocked.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(depths) == 1 and waits == [] and sources[0].closes == 1


@pytest.mark.asyncio
async def test_completed_response_without_events_is_not_retried(waits: list[float]) -> None:
    response = ModelResponse(parts=[TextPart("completed")])

    class Completed(TestModel):
        calls = 0

        @asynccontextmanager
        async def request_stream(self, messages, model_settings, parameters, run_context=None):
            self.calls += 1
            yield CompletedStreamedResponse(response, model_request_parameters=parameters)

    provider = Completed()
    model = _RetryingModel(provider, 3, 5)
    async with model.request_stream([], None, ModelRequestParameters()) as streamed:
        assert [event async for event in streamed] == []
        assert streamed.get().parts == response.parts
    assert provider.calls == 1 and waits == []


@pytest.mark.asyncio
async def test_partial_retry_awaits_attempt_boundary_and_keeps_winner_only(
    monkeypatch: pytest.MonkeyPatch, waits: list[float],
) -> None:
    binding, sources, depths = _model(monkeypatch, lambda attempt: (
        [_PARTIAL, httpx2.RemoteProtocolError("synthetic disconnect")] if attempt <= 3 else [_SUCCESS]
    ))
    order: list[tuple[str, int]] = []
    active = 0

    async def observed(attempt) -> None:
        nonlocal active
        await asyncio.sleep(0)
        active = attempt.stream_attempt_seq
        order.append((attempt.status, active))

    async def consume(ctx, events) -> None:
        async for event in events:
            order.append((type(event).__name__, active))

    with ModelClientRequestRecorder(stream_attempt_observer=observed, record_http_bodies=True) as recorder:
        result = await Agent(binding.materialize()).run("synthetic", event_stream_handler=consume)
    assert result.output == "winner"
    assert len(depths) == 4 and len(set(depths[1:])) == 1 and max(depths[1:]) <= depths[0]
    assert waits == [5, 20, 80] and all(source.closes == 1 for source in sources)
    assert [attempt.status for attempt in recorder.stream_attempts] == ["FAILED"] * 3 + ["SUCCEEDED"]
    assert [(attempt.client_request_start, attempt.client_request_end) for attempt in recorder.stream_attempts] == [
        (0, 1), (1, 2), (2, 3), (3, 4),
    ]
    for request in recorder.requests[:-1]:
        assert "sdk_iteration_error_ns" in request.diagnostics
        assert "sdk_exhausted_ns" not in request.diagnostics
    assert "sdk_exhausted_ns" in recorder.requests[-1].diagnostics
    for seq in range(1, 5):
        assert order.index(("RUNNING", seq)) < order.index(("PartStartEvent", seq))
    for seq in range(1, 4):
        assert order.index(("FAILED", seq)) < order.index(("RUNNING", seq + 1))
    assert [part.content for message in result.new_messages() if isinstance(message, ModelResponse)
            for part in message.parts if isinstance(part, TextPart)] == ["winner"]


@pytest.mark.asyncio
async def test_replacement_observer_failure_prevents_another_request(
    monkeypatch: pytest.MonkeyPatch, waits: list[float],
) -> None:
    binding, sources, depths = _model(monkeypatch, lambda attempt: [
        _PARTIAL, httpx2.RemoteProtocolError("synthetic disconnect"),
    ])
    async def observed(attempt) -> None:
        if attempt.status == "FAILED":
            raise RuntimeError("synthetic replacement failure")
    with ModelClientRequestRecorder(stream_attempt_observer=observed):
        with pytest.raises(RuntimeError, match="replacement failure"):
            await Agent(binding.materialize()).run("synthetic", event_stream_handler=_consume)
    assert len(depths) == 1 and waits == [] and sources[0].closes == 1


def _with_usage(chunk: bytes, input_tokens: int, output_tokens: int) -> bytes:
    raw = json.loads(chunk.split(b"\n\n", 1)[0].removeprefix(b"data: "))
    raw["usage"] = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens}
    return b"data: " + json.dumps(raw).encode() + b"\n\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", (False, True))
async def test_failed_attempt_known_usage_is_counted_once_on_success(
    monkeypatch: pytest.MonkeyPatch, waits: list[float], partial: bool,
) -> None:
    binding, sources, depths = _model(monkeypatch, lambda attempt: (
        [_with_usage(_PARTIAL if partial else _ROLE, 7, 1), httpx2.RemoteProtocolError("synthetic")]
        if attempt <= 3 else [_with_usage(_SUCCESS, 11, 2), b"data: [DONE]\n\n"]
    ))
    async def observed(attempt) -> None:
        pass
    with ModelClientRequestRecorder(stream_attempt_observer=observed) as recorder:
        result = await Agent(binding.materialize()).run("synthetic", event_stream_handler=_consume)
    assert result.output == "winner"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (32, 5)
    assert [(attempt.usage.input_tokens, attempt.usage.output_tokens) for attempt in recorder.stream_attempts] == [
        (7, 1), (7, 1), (7, 1), (11, 2),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", (False, True))
async def test_failed_attempt_known_usage_is_counted_once_on_exhaustion(
    monkeypatch: pytest.MonkeyPatch, waits: list[float], partial: bool,
) -> None:
    from pydantic_ai.usage import RunUsage
    binding, sources, depths = _model(monkeypatch, lambda attempt: [
        _with_usage(_PARTIAL if partial else _ROLE, 7, 1), httpx2.RemoteProtocolError("synthetic"),
    ])
    usage = RunUsage()
    async def observed(attempt) -> None:
        pass
    with ModelClientRequestRecorder(stream_attempt_observer=observed):
        with pytest.raises(ModelAPIError):
            await Agent(binding.materialize()).run("synthetic", usage=usage, event_stream_handler=_consume)
    assert (usage.input_tokens, usage.output_tokens) == (28, 4)


class _FixtureBinding:
    provider = "synthetic"
    model_identity = "synthetic"
    contract = {"provider": "synthetic", "model_identity": "synthetic", "input_mime_types": []}

    def __init__(self, model):
        self.model = model

    def materialize(self):
        return self.model


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("text", "tools", "output"))
async def test_runtime_partial_replacement_commits_only_winner_and_its_tools(kind, waits):
    from pydantic import BaseModel
    from pydantic_ai.models.function import DeltaToolCall
    from linktools.ai.capability import AgentContext
    class Answer(BaseModel):
        value: int
    calls = []
    used = []

    async def old_tool(ctx: AgentContext[object]) -> str:
        used.append("old")
        return "old"

    async def new_tool(ctx: AgentContext[object]) -> str:
        used.append("new")
        return "new"

    async def stream(messages, info):
        calls.append(messages)
        if kind == "text":
            yield "discarded" if len(calls) == 1 else "winner"
            if len(calls) == 1:
                raise ModelHTTPError(503, "fixture")
        elif kind == "output":
            yield {0: DeltaToolCall(name=info.output_tools[0].name,
                                   json_args=json.dumps({"value": 1 if len(calls) == 1 else 2}),
                                   tool_call_id=f"fixture-{len(calls)}")}
            if len(calls) == 1:
                raise ModelHTTPError(503, "fixture")
        elif len(calls) <= 2:
            yield {0: DeltaToolCall(name="old_tool" if len(calls) == 1 else "new_tool",
                                   json_args="{}", tool_call_id=f"fixture-{len(calls)}")}
            if len(calls) == 1:
                raise ModelHTTPError(503, "fixture")
        else:
            yield "winner"

    features = RuntimeFeatures("partial-retry")
    features.model("default", _FixtureBinding(_RetryingModel(FunctionModel(stream_function=stream), 3, 5)))
    features.agent("default", model="default", allow_tools=("old_tool", "new_tool"),
                   allow_skills=(), allow_subagents=())
    features.tool(old_tool, effect_policy="none")
    features.tool(new_tool, effect_policy="none")
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('partial-retry', storage=PersistenceStorage.in_memory()) as persistence_426, Runtime.open(persistence=persistence_426, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic", **({"output": Answer} if kind == "output" else {}))
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == "SUCCEEDED", result.error_code
        assert result.output == ({"value": 2} if kind == "output" else {"text": "winner"})
        items = (await execution.model_interactions(include_content=True)).items
        first = items[0]
        assert [item.status for item in first.stream_attempts] == ["FAILED", "SUCCEEDED"]
        assert first.stream_attempts[0].response is not None
        assert first.stream_attempts[1].response is None
        assert all(item.client_request_start == item.client_request_end == 0 for item in first.stream_attempts)
        boundaries = (await execution.list_events(include_content=True, limit=100)).items
        attempts = [(event.event_type, event.payload["stream_attempt_seq"])
                    for event in boundaries if event.event_type.startswith("MODEL_ATTEMPT_")]
        assert attempts[:3] == [("MODEL_ATTEMPT_STARTED", 1), ("MODEL_ATTEMPT_FINISHED", 1), ("MODEL_ATTEMPT_STARTED", 2)]
        completed = [event for event in boundaries if event.event_type == "MODEL_REQUEST_FINISHED"]
        assert completed[0].payload["stream_attempt_seq"] == 2
        assert completed[0].payload["message_seq"] >= 1
        parts = [event for event in boundaries if event.event_type == "ASSISTANT_PART_COMPLETED"]
        assert all("message_seq" not in event.payload and "stream_attempt_seq" in event.payload for event in parts)
        assert "discarded" not in json.dumps(first.response) and "old_tool" not in json.dumps(first.response)
    assert used == (["new"] if kind == "tools" else [])
    assert len(calls) == (3 if kind == "tools" else 2) and waits == [5]


@pytest.mark.asyncio
async def test_runtime_exhausted_partial_is_trace_only_and_keeps_known_usage(monkeypatch, waits):
    binding, sources, depths = _model(monkeypatch, lambda attempt: [
        _with_usage(_PARTIAL, 7, 1), httpx2.RemoteProtocolError("synthetic"),
    ])
    features = RuntimeFeatures("exhausted-partial")
    features.model("default", binding)
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('exhausted-partial', storage=PersistenceStorage.in_memory()) as persistence_460, Runtime.open(persistence=persistence_460, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == "FAILED" and result.error_code == "MODEL_UNAVAILABLE"
        items = (await execution.model_interactions(include_content=True)).items
        assert len(items) == 1 and items[0].response is None
        assert [item.status for item in items[0].stream_attempts] == ["FAILED"] * 4
        assert (items[0].usage.input_tokens, items[0].usage.output_tokens) == (28, 4)
        assert all("discarded" in json.dumps(item.response) for item in items[0].stream_attempts)
        transcript = (await execution.transcript(include_content=True)).items
        assert "discarded" not in str(transcript)
        boundaries = (await execution.list_events(include_content=True, limit=100)).items
        finished = [event for event in boundaries if event.event_type == "MODEL_REQUEST_FINISHED"]
        assert len(finished) == 1 and finished[0].payload["stream_attempt_seq"] == 4
        assert "message_seq" not in finished[0].payload
    assert waits == [5, 20, 80] and all(source.closes == 1 for source in sources)


@pytest.mark.asyncio
async def test_budget_stops_partial_retry_before_another_provider_call(monkeypatch, waits):
    from pydantic_ai.usage import RunUsage, UsageLimits, UsageLimitExceeded
    binding, sources, depths = _model(monkeypatch, lambda attempt: [
        _with_usage(_PARTIAL, 7, 1), httpx2.RemoteProtocolError("synthetic"),
    ])
    usage = RunUsage()
    async def observed(attempt):
        pass
    with ModelClientRequestRecorder(stream_attempt_observer=observed):
        with pytest.raises(UsageLimitExceeded):
            await Agent(binding.materialize()).run("synthetic", usage=usage,
                usage_limits=UsageLimits(input_tokens_limit=10), event_stream_handler=_consume)
    assert len(depths) == 2 and waits == [5]
    assert usage.input_tokens == 14 and usage.output_tokens == 2
    assert all(source.closes == 1 for source in sources)


@pytest.mark.asyncio
async def test_cancellation_during_backoff_counts_failed_usage_once(monkeypatch):
    from pydantic_ai.usage import RunUsage
    binding, sources, depths = _model(monkeypatch, lambda attempt: [
        _with_usage(_PARTIAL, 7, 1), httpx2.RemoteProtocolError("synthetic"),
    ])
    sleeping = asyncio.Event()
    async def wait(self, error, backoff):
        sleeping.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(_RetryingModel, "_wait_for_retry", wait)
    usage = RunUsage()
    async def observed(attempt):
        pass
    with ModelClientRequestRecorder(stream_attempt_observer=observed):
        task = asyncio.create_task(Agent(binding.materialize()).run(
            "synthetic", usage=usage, event_stream_handler=_consume,
        ))
        await asyncio.wait_for(sleeping.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(depths) == 1 and sources[0].closes == 1
    assert (usage.input_tokens, usage.output_tokens) == (7, 1)


def _continuation_retry_features(same_id: bool, *, retry_delay: float = 5, fail_continuation: bool = False):
    from decimal import Decimal
    from pydantic_ai.messages import PartStartEvent
    from pydantic_ai.usage import RequestUsage
    class Broken(CompletedStreamedResponse):
        async def _get_event_iterator(self):
            yield PartStartEvent(index=0, part=TextPart("discarded"))
            raise ModelHTTPError(503, "fixture")
    class Continuing(TestModel):
        calls = 0
        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
            self.calls += 1
            if self.calls == 3 and fail_continuation:
                raise ModelHTTPError(503, "fixture")
            response = ModelResponse(
                [TextPart("discarded" if self.calls == 1 else "segment-one " if self.calls == 2 else "segment-two")],
                model_name="fixture", provider_name="synthetic",
                provider_response_id=None if self.calls == 1 else "first" if self.calls == 2 or same_id else "second",
                state="incomplete" if self.calls == 1 else "suspended" if self.calls == 2 else "complete",
                usage=RequestUsage(input_tokens=3 if self.calls == 1 else 5 if self.calls == 2 else 8,
                                   output_tokens=1 if self.calls == 1 else 2 if self.calls == 2 else 4,
                                   cost=Decimal(".003") if self.calls == 1 else Decimal(".005") if self.calls == 2 else Decimal(".008")),
            )
            response_type = Broken if self.calls == 1 else CompletedStreamedResponse
            yield response_type(response, model_request_parameters=model_request_parameters, replay_events=True)
    model = Continuing()
    features = RuntimeFeatures("continuation-retry")
    features.model("default", _FixtureBinding(_RetryingModel(model, 3, retry_delay)))
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(TraceCapability(), scope="runtime")
    return features, model


async def runtime_continuation_retry_events(same_id: bool = False, *, include_models: bool = False):
    """Real SDK/Runtime watch events for consumer contract replay, no network."""
    features, _ = _continuation_retry_features(same_id, retry_delay=0)
    async with Persistence.open('continuation-retry', storage=PersistenceStorage.in_memory()) as persistence_559, Runtime.open(persistence=persistence_559, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        observed = [item async for item in execution.watch(include_content=True)]
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == "SUCCEEDED", result.error_code
        events = tuple(item.event for item in observed if hasattr(item, "event"))
        if include_models:
            return events, (await execution.model_interactions(include_content=True)).items
        return events


@pytest.mark.asyncio
@pytest.mark.parametrize("same_id", (False, True))
async def test_retry_then_normal_continuation_keeps_segments_and_exact_usage(same_id, waits):
    features, model = _continuation_retry_features(same_id)
    async with Persistence.open('continuation-retry', storage=PersistenceStorage.in_memory()) as persistence_574, Runtime.open(persistence=persistence_574, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == "SUCCEEDED", result.error_code
        assert "discarded" not in str(result.output) and "segment-two" in str(result.output)
        assert ("segment-one" in str(result.output)) is not same_id
        item = (await execution.model_interactions(include_content=True)).items[0]
        assert [attempt.status for attempt in item.stream_attempts] == ["FAILED", "SUCCEEDED", "SUCCEEDED"]
        assert (item.usage.input_tokens, item.usage.output_tokens) == ((11, 5) if same_id else (16, 7))
        assert result.usage.input_tokens == item.usage.input_tokens
        events = (await execution.list_events(include_content=True, limit=100)).items
        starts = [event.payload for event in events if event.event_type == "MODEL_ATTEMPT_STARTED"]
        assert [payload["replaces_stream_attempt_seq"] for payload in starts] == [None, 1, None]
        finishes = [event.payload for event in events if event.event_type == "MODEL_ATTEMPT_FINISHED"]
        assert [payload["status"] for payload in finishes] == ["FAILED", "SUCCEEDED", "SUCCEEDED"]
    assert model.calls == 3 and waits == [5]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("current_native", "current_suspended", "input_suspended"))
async def test_provider_owned_effects_are_not_replayed_before_first_event(kind, waits):
    from pydantic_ai.messages import NativeToolCallPart
    class Broken(CompletedStreamedResponse):
        async def _get_event_iterator(self):
            raise ModelHTTPError(503, "fixture")
            yield
    class Provider(TestModel):
        calls = 0
        @asynccontextmanager
        async def request_stream(self, messages, settings, parameters, run_context=None):
            self.calls += 1
            response = ModelResponse(
                [NativeToolCallPart(tool_name="web_search", args={}, tool_call_id="fixture")]
                if kind == "current_native" else [],
                state="suspended" if kind == "current_suspended" else "incomplete",
            )
            yield Broken(response, model_request_parameters=parameters)
    provider = Provider()
    model = _RetryingModel(provider, 3, 5)
    messages = [ModelResponse([], state="suspended")] if kind == "input_suspended" else []
    with pytest.raises(ModelAPIError):
        async with model.request_stream(messages, None, ModelRequestParameters()):
            pytest.fail("failed preflight cannot enter")
    assert provider.calls == 1 and waits == []


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer_failure", (False, True))
async def test_consumer_close_with_retryable_cleanup_error_never_redispatches(consumer_failure, waits):
    class Provider(TestModel):
        calls = 0
        @asynccontextmanager
        async def request_stream(self, messages, settings, parameters, run_context=None):
            self.calls += 1
            try:
                yield CompletedStreamedResponse(ModelResponse([TextPart("partial")]),
                    model_request_parameters=parameters, replay_events=True)
            finally:
                raise ModelHTTPError(503, "fixture")
    provider = Provider()
    model = _RetryingModel(provider, 3, 5)
    original = ValueError("synthetic consumer")
    async def observed(attempt):
        pass
    with ModelClientRequestRecorder(stream_attempt_observer=observed):
        with pytest.raises(ValueError if consumer_failure else ModelHTTPError) as caught:
            async with model.request_stream([], None, ModelRequestParameters()) as stream:
                await anext(aiter(stream))
                if consumer_failure:
                    raise original
    if consumer_failure:
        assert caught.value is original
    assert provider.calls == 1 and waits == []


@pytest.mark.asyncio
async def test_native_fallback_known_usage_is_not_overwritten_by_stream_attempt_metrics():
    from decimal import Decimal
    from pydantic_ai.usage import RequestUsage
    def rejected(messages, info):
        return ModelResponse([TextPart("rejected")], model_name="rejected",
                             usage=RequestUsage(input_tokens=7, output_tokens=1, cost=Decimal(".007")))
    def accepted(messages, info):
        return ModelResponse([TextPart("winner")], model_name="accepted",
                             usage=RequestUsage(input_tokens=11, output_tokens=2, cost=Decimal(".011")))
    def reject(response: ModelResponse) -> bool:
        return response.model_name == "rejected"
    native = FallbackModel(FunctionModel(rejected, model_name="rejected"),
                           FunctionModel(accepted, model_name="accepted"), fallback_on=reject)
    class Bridge(TestModel):
        @asynccontextmanager
        async def request_stream(self, messages, settings, parameters, run_context=None):
            response = await native.request(messages, settings, parameters)
            yield CompletedStreamedResponse(response, model_request_parameters=parameters, replay_events=True)
    features = RuntimeFeatures("native-fallback-usage")
    features.model("default", _FixtureBinding(_RetryingModel(Bridge(), 3, 0)))
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('native-fallback-usage', storage=PersistenceStorage.in_memory()) as persistence_672, Runtime.open(persistence=persistence_672, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == "SUCCEEDED", result.error_code
        item = (await execution.model_interactions(include_content=True)).items[0]
        assert len(item.stream_attempts) == 1
        assert item.stream_attempts[0].usage.input_tokens == 11
        assert (item.usage.input_tokens, item.usage.output_tokens) == (18, 3)
        assert (result.usage.input_tokens, result.usage.output_tokens) == (18, 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("same_id", (False, True))
async def test_real_watch_preserves_part_start_and_attempt_boundaries(same_id):
    events = await runtime_continuation_retry_events(same_id)
    deltas = [event for event in events if event.event_type == "ASSISTANT_TEXT_DELTA"]
    assert [event.payload["part_index"] for event in deltas] == ([0, 0, 0] if same_id else [0, 0, 1])
    assert all(event.payload["part_start"] is True for event in deltas)
    for seq in (1, 2, 3):
        start = next(index for index, event in enumerate(events) if event.event_type == "MODEL_ATTEMPT_STARTED"
                     and event.payload["stream_attempt_seq"] == seq)
        finish = next(index for index, event in enumerate(events) if event.event_type == "MODEL_ATTEMPT_FINISHED"
                      and event.payload["stream_attempt_seq"] == seq)
        positions = [index for index, event in enumerate(events) if event.event_type == "ASSISTANT_TEXT_DELTA"
                     and event.payload["stream_attempt_seq"] == seq]
        assert positions and start < min(positions) <= max(positions) < finish


@pytest.mark.asyncio
async def test_empty_part_start_has_replay_position_and_never_coalesces_across_attempts():
    from linktools.ai.core import ExecutionDeltaType
    from linktools.ai.execution.events import ExecutionDelta, LiveExecutionEventBroker
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("fixture", 0)
    position = {"agent_run_seq": 1, "model_request_seq": 1, "stream_attempt_seq": 1,
                "part_index": 0, "part_start": True}
    broker.publish(ExecutionDelta("fixture", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "", position=position))
    broker.publish(ExecutionDelta("fixture", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "first", position={**position, "part_start": False}))
    broker.publish(ExecutionDelta("fixture", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "next", position={**position, "stream_attempt_seq": 2}))
    subscription = broker.subscribe("fixture")
    received = [await asyncio.wait_for(anext(subscription), 1) for _ in range(3)]
    assert [item.content for item in received] == ["", "first", "next"]
    assert [item.start_offset for item in received] == [0, 1, 6]
    assert broker.can_resume_preview("fixture", (broker.preview_epoch("fixture"), 1))
    await subscription.close()


@pytest.mark.asyncio
async def test_parallel_runs_and_consecutive_requests_keep_committed_message_positions(waits):
    from pydantic_ai.messages import ToolReturnPart, UserPromptPart
    from pydantic_ai.models.function import DeltaToolCall
    from linktools.ai.capability import AgentContext
    calls = collections.Counter()
    used = []
    async def lookup(ctx: AgentContext[object], key: str) -> str:
        used.append(key)
        return key
    async def stream(messages, info):
        question = next(part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart))
        calls[question] += 1
        if calls[question] == 1:
            yield "discarded"
            raise ModelHTTPError(503, "fixture")
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield {0: DeltaToolCall(name="lookup", json_args=json.dumps({"key": question}), tool_call_id=question)}
        else:
            yield "winner-" + question
    features = RuntimeFeatures("parallel-retry")
    features.model("default", _FixtureBinding(_RetryingModel(FunctionModel(stream_function=stream), 3, 5)))
    features.agent("default", model="default", allow_tools=("lookup",), allow_skills=(), allow_subagents=())
    features.tool(lookup, effect_policy="none")
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('parallel-retry', storage=PersistenceStorage.in_memory()) as persistence_744, Runtime.open(persistence=persistence_744, features=features) as runtime:
        executions = await asyncio.gather(*(runtime.agents.get().start(value) for value in ("a", "b")))
        results = await asyncio.gather(*(execution.wait(timeout_seconds=10) for execution in executions))
        assert [value.result.output for value in results] == [{"text": "winner-a"}, {"text": "winner-b"}]
        for execution in executions:
            events = (await execution.list_events(include_content=True, limit=100)).items
            completed = [event.payload for event in events if event.event_type == "MODEL_REQUEST_FINISHED"]
            assert [(item["model_request_seq"], item["stream_attempt_seq"], item["message_seq"])
                    for item in completed] == [(1, 2, 2), (2, 1, 4)]
    assert sorted(used) == ["a", "b"] and calls == {"a": 3, "b": 3}


async def runtime_empty_start_events(*, include_models: bool = False):
    """Public SDK empty PartStart survives the actual Runtime watch pipeline."""
    class EmptyStart(CompletedStreamedResponse):
        async def _get_event_iterator(self):
            for content in ("", "winner"):
                for event in self._parts_manager.handle_text_delta(vendor_part_id="text", content=content):
                    yield event
    class Streaming(TestModel):
        @asynccontextmanager
        async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
            yield EmptyStart(ModelResponse([TextPart("winner")], model_name="fixture"),
                             model_request_parameters=model_request_parameters)
    features = RuntimeFeatures("empty-start")
    features.model("default", _FixtureBinding(_RetryingModel(Streaming(), 1, 0)))
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    if include_models:
        features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('empty-start', storage=PersistenceStorage.in_memory()) as persistence_773, Runtime.open(persistence=persistence_773, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        observed = [item async for item in execution.watch(include_content=True)]
        assert (await execution.wait(timeout_seconds=10)).result.status == "SUCCEEDED"
        events = tuple(item.event for item in observed if hasattr(item, "event"))
        if include_models:
            return events, (await execution.model_interactions(include_content=True)).items
        return events


@pytest.mark.asyncio
async def test_real_watch_preserves_empty_start_before_text():
    events = await runtime_empty_start_events()
    deltas = [event for event in events if event.event_type == "ASSISTANT_TEXT_DELTA"]
    assert [(event.payload["text"], event.payload["part_start"]) for event in deltas] == [("", True), ("winner", False)]


@pytest.mark.asyncio
async def test_long_part_keeps_active_subscriber_live_and_late_subscriber_sees_gap():
    from linktools.ai.core import ExecutionDeltaType
    from linktools.ai.execution.events import ExecutionDelta, LiveExecutionEventBroker
    broker = LiveExecutionEventBroker(max_bytes=8)
    broker.register_local_producer("fixture", 0)
    subscription = broker.subscribe("fixture")
    position = {"agent_run_seq": 1, "model_request_seq": 1, "stream_attempt_seq": 2,
                "part_index": 0, "part_start": True}
    broker.publish(ExecutionDelta("fixture", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "start", position=position))
    assert (await anext(subscription)).content == "start"
    for index in range(8):
        tail = f"tail{index}"
        broker.publish(ExecutionDelta("fixture", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, tail,
                                      position={**position, "part_start": False}))
        assert (await asyncio.wait_for(anext(subscription), 1)).content == tail
    assert not subscription.replay_required
    resumed = broker.subscribe("fixture")
    tail = await anext(resumed)
    assert tail.position["part_start"] is False and tail.start_offset > 0
    assert tail.stream_truncated
    assert not broker.can_resume_preview("fixture", (broker.preview_epoch("fixture"), 0))
    assert not resumed.replay_required
    await subscription.close()
    await resumed.close()


@pytest.mark.asyncio
async def test_failed_continuation_retains_prior_success_only_as_interrupted_trace(waits):
    features, model = _continuation_retry_features(False, fail_continuation=True)
    async with Persistence.open('continuation-retry', storage=PersistenceStorage.in_memory()) as persistence_820, Runtime.open(persistence=persistence_820, features=features) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == "FAILED"
        item = (await execution.model_interactions(include_content=True)).items[0]
        assert [attempt.status for attempt in item.stream_attempts] == ["FAILED", "SUCCEEDED", "FAILED"]
        assert item.response is not None and "segment-one" in str(item.response)
        assert "interrupted" in str(item.response)
        history = (await execution.transcript(include_content=True)).items
        assert "segment-one" not in str(history) and "discarded" not in str(history)
    assert model.calls == 3 and waits == [5]


@pytest.mark.asyncio
async def test_watch_helper_models_are_actual_canonical_query_results():
    for helper in (runtime_continuation_retry_events, runtime_empty_start_events):
        events, models = await helper(include_models=True)
        assert events and len(models) == 1 and models[0].status == "SUCCEEDED"
        assert models[0].response is not None
        assert "discarded" not in str(models[0].response)
        assert models[0].model_request_seq == 1 and models[0].agent_run_seq == 1
