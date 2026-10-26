#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Native reference capabilities retain model failure and stream contracts."""

import asyncio
import collections
import hashlib
import json
import traceback
from collections.abc import AsyncIterable, AsyncIterator

import httpx2
import pytest
from pydantic_ai import Agent, capture_run_messages, exceptions as sdk_errors
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.tools import RunContext

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import (
    AgentContext, CapabilityContribution, ExecutionCapability, ReferencedInputsCapability, TraceCapability,
)
from linktools.ai.core import ExecutionStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import (
    ModelClientRequestRecorder, ReferencedContent, ReferencedInputModel, create_model, model_request_error,
)
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

from .test_model_error_classification import _mock_clients
from .test_model_stream_context_exit import _Provider as _ExitProvider
from .test_model_stream_retry import _PARTIAL, _ROLE, _SUCCESS, _Source, _with_usage


_BODY = b"synthetic reference body for provider use only"
_SOURCE = CapabilityRef("capability", "reference-source", 1)
_OMISSION = "referenced_content_lifecycle"


def _reference() -> ReferencedContent:
    return ReferencedContent(
        AssetVersionRef(AssetKey("document", "input"), "example", StorageEntryRevision(1),
                        hashlib.sha256(_BODY).hexdigest(), len(_BODY)),
        media_type="text/plain", representation="text", source=_SOURCE,
    )


def _context() -> AgentContext[None]:
    return AgentContext(None, service_principal("tenant", "reader"), "example", None, "run", {})


def _contribution(loaded: list[ReferencedContent]):
    async def load(context: AgentContext[None], reference: ReferencedContent) -> bytes:
        loaded.append(reference)
        return _BODY

    return CapabilityContribution.from_opaque(
        "capability", _SOURCE.id, ReferencedInputsCapability(load), revision=_SOURCE.revision,
    ).materialize_capability()


async def _consume(context: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in events:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_native_reference_context_overflow_preserves_permanent_category_without_http(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    reference, loaded, dispatched = _reference(), [], []
    error = sdk_errors.ModelContextWindowExceededError("fixture", _BODY.decode())

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        dispatched.append(messages)
        assert messages[0].parts[0].content == [_BODY.decode()]
        raise error

    async def fail_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        fail(messages, info)
        yield "unreachable"  # pragma: no cover

    binding = create_model(api="openai", model="fixture", api_key="synthetic",
                           max_retries=3, retry_delay=0)
    provider = FunctionModel(fail, stream_function=fail_stream)
    monkeypatch.setattr(type(binding), "_build_model", lambda self, settings: provider)
    async with Agent(model=binding.materialize(), capabilities=[_contribution(loaded)]) as agent:
        with capture_run_messages() as history, ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            with pytest.raises(AIError) as caught:
                await agent.run([reference], deps=_context(),
                                event_stream_handler=_consume if streaming else None)

    expected = model_request_error(error, content_omission_reason=_OMISSION)
    assert caught.value.code is expected.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert caught.value.retryable is expected.retryable is False
    assert caught.value.safe_details == expected.safe_details == {
        "model_name": "fixture", "reason": "context_window_exceeded", "content_omission_reason": _OMISSION,
    }
    assert loaded == [reference] and len(dispatched) == 1
    assert recorder.requests == () and recorder.body_omission_reason == _OMISSION
    assert any(isinstance(part, UserPromptPart) and part.content == [reference]
               for message in history for part in message.parts)
    assert _BODY.decode() not in repr(history)
    assert _BODY.decode() not in repr(caught.value.diagnostics.to_payload())
    assert _BODY.decode() not in "".join(traceback.format_exception(caught.value))


@pytest.mark.asyncio
async def test_native_reference_stream_retries_reuse_hydration_and_count_known_usage_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, loaded, sent, streams = _reference(), [], [], []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        stream = _Source(
            [_with_usage(_ROLE, 7, 1), httpx2.RemoteProtocolError(_BODY.decode())]
            if len(sent) <= 2 else [_with_usage(_SUCCESS, 11, 2), b"data: [DONE]\n\n"]
        )
        streams.append(stream)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(api="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0)
    async with Agent(model=binding.materialize(), capabilities=[_contribution(loaded)]) as agent:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            result = await agent.run([reference], deps=_context(), event_stream_handler=_consume)

    assert result.output == "winner"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (25, 4)
    assert loaded == [reference] and len(sent) == 3
    assert all(_BODY.decode() in json.dumps(json.loads(body)["messages"]) for body in sent)
    assert all(stream.closes == 1 for stream in streams) and all(client.is_closed for client in clients)
    assert recorder.body_omission_reason == _OMISSION
    assert all(request.body is None and request.response.body is None for request in recorder.requests)
    assert [attempt.status for attempt in recorder.stream_attempts] == ["FAILED", "FAILED", "SUCCEEDED"]
    assert [(attempt.client_request_start, attempt.client_request_end) for attempt in recorder.stream_attempts] == [
        (0, 1), (1, 2), (2, 3),
    ]
    response = result.all_messages()[-1]
    assert len(response.failed_attempts) == 2
    assert all(attempt.error is None for attempt in response.failed_attempts)
    assert _BODY.decode() not in repr(result.all_messages())


@pytest.mark.asyncio
async def test_native_reference_stream_cleanup_failure_is_omitted_and_never_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, loaded = _reference(), []
    provider = _ExitProvider(cleanup_error=ModelHTTPError(503, "fixture", body=_BODY.decode()))
    binding = create_model(api="openai", model="fixture", api_key="synthetic",
                           max_retries=3, retry_delay=0)
    monkeypatch.setattr(type(binding), "_build_model", lambda self, settings: provider)
    async with Agent(model=binding.materialize(), capabilities=[_contribution(loaded)]) as agent:
        with pytest.raises(AIError) as caught:
            async with agent.run_stream([reference], deps=_context()) as result:
                assert await result.get_output() == "fixture"

    assert provider.entries == 1 and provider.exits == [(None, None, None)]
    assert loaded == [reference]
    assert caught.value.code is ErrorCode.MODEL_UNAVAILABLE
    assert caught.value.safe_details["status_code"] == 503
    assert caught.value.diagnostics.content_omission_reason == _OMISSION
    assert _BODY.decode() not in "".join(traceback.format_exception(caught.value))


@pytest.mark.asyncio
async def test_runtime_reference_stream_failure_keeps_attempt_coordinates_and_safe_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, loaded, sent, streams = _reference(), [], [], []

    async def load(context: AgentContext[None], content: ReferencedContent) -> bytes:
        loaded.append(content)
        return _BODY

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        stream = _Source([_with_usage(_PARTIAL, 7, 1), httpx2.RemoteProtocolError(_BODY.decode())])
        streams.append(stream)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(api="openai", model="fixture", api_key="synthetic",
                           base_url="https://offline.invalid/v1", max_retries=1, retry_delay=0)
    features = RuntimeFeatures("reference-stream")
    features.model("default", binding)
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(ReferencedInputsCapability(load), id=_SOURCE.id,
                        revision=_SOURCE.revision, scope="agent")
    features.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('reference-stream', storage=PersistenceStorage.in_memory()) as persistence_200, Runtime.attach(persistence=persistence_200, features=features) as runtime:
        execution = await runtime.agents.get().start([reference])
        result = (await execution.wait(timeout_seconds=10)).result
        interactions = (await execution.model_interactions(include_content=True, include_client_requests=True)).items
        history = (await execution.history(include_content=True)).items
        transcript = (await execution.transcript(include_content=True)).items

    assert result.status is ExecutionStatus.FAILED and result.error_code == "MODEL_UNAVAILABLE"
    expected = model_request_error(
        sdk_errors.ModelConnectionError("fixture", _BODY.decode(), phase="read", in_stream=True),
        content_omission_reason=_OMISSION,
    )
    assert all(result.safe_error_details[key] == value for key, value in expected.safe_details.items())
    assert result.safe_error_details["model_request_seq"] == 1
    assert result.safe_error_details["stream_attempt_seq"] == result.safe_error_details["stream_attempt_count"] == 2
    assert result.error_diagnostics.content_omission_reason == _OMISSION
    assert len(interactions) == 1 and interactions[0].response is None
    assert (interactions[0].usage.input_tokens, interactions[0].usage.output_tokens) == (14, 2)
    assert [attempt.status for attempt in interactions[0].stream_attempts] == ["FAILED", "FAILED"]
    assert [(attempt.client_request_start, attempt.client_request_end) for attempt in interactions[0].stream_attempts] == [
        (0, 1), (1, 2),
    ]
    assert all(request.body is None and request.response.body is None
               for request in interactions[0].client_requests)
    assert _BODY.decode() not in repr((result, interactions, history, transcript))
    assert "discarded" not in repr(transcript)
    assert loaded == [reference] and len(sent) == 2
    assert all(stream.closes == 1 for stream in streams) and all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_native_reference_wrapper_preserves_sdk_deadline_inside_fallback() -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "first "
        await asyncio.Event().wait()

    async def load(reference: ReferencedContent) -> bytes:
        pytest.fail("plain input must not hydrate references")

    model = ReferencedInputModel(FunctionModel(stream_function=stream), load=load)
    with pytest.raises(sdk_errors.ModelRequestTimeout):
        await asyncio.wait_for(Agent(FallbackModel(model)).run(
            "plain input", model_settings={"request_timeout": 0.02}, event_stream_handler=_consume,
        ), timeout=1)


@pytest.mark.asyncio
async def test_native_reference_wrapper_replays_fallback_first_event_exactly_once() -> None:
    events: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield "first "
        yield "second"

    async def load(reference: ReferencedContent) -> bytes:
        pytest.fail("plain input must not hydrate references")

    async def consume(context: RunContext[object], source: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in source:
            events.append(type(event).__name__)

    model = ReferencedInputModel(FunctionModel(stream_function=stream), load=load)
    result = await Agent(FallbackModel(model)).run("plain input", event_stream_handler=consume)
    assert result.output == "first second"
    assert collections.Counter(events) == {
        "PartStartEvent": 1, "PartDeltaEvent": 1, "PartEndEvent": 1, "FinalResultEvent": 1,
    }
