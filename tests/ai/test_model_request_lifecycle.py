#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical model request progress is visible and converges to one result."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import (
    CompletedStreamedResponse,
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionEventType, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import MetricQuery, MetricWindow, Metrics
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._compaction import _ObservedCompactionModel
from linktools.ai.runtime import _event as runtime_event
from linktools.ai.runtime._journal import ModelRequestFact, ModelRequestJournal
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.runtime.service_api import ModelInteractionItem
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._steps import StagingAgentRunStore
from linktools.ai.observe._memory import InMemoryMetricStore


class _LifecycleRecorder:
    def __init__(self) -> None:
        self.values: dict[int, dict[str, object]] = {}
        self.event_phases: list[tuple[str, ModelRequestFact]] = []
        self.finish_calls: dict[int, int] = {}

    def begin_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: object,
        model_settings: object,
        parameters: object,
        streaming: bool,
        model_id: str | None = None,
        source_messages: object = None,
    ) -> None:
        del model, messages, model_settings, parameters, streaming, model_id, source_messages
        self.values[fact.request_sequence] = {
            "status": "RUNNING",
            "started_at": fact.started_at,
            "finished_at": None,
            "error_code": None,
            "usage": None,
        }

    def finish_model_interaction(
        self,
        fact: ModelRequestFact,
        *,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        duration_ns: int,
        usage: object | None,
    ) -> None:
        del model, response, duration_ns
        self.finish_calls[fact.request_sequence] = (
            self.finish_calls.get(fact.request_sequence, 0) + 1
        )
        self.values[fact.request_sequence] = {
            "status": status,
            "started_at": fact.started_at,
            "finished_at": fact.finished_at,
            "error_code": error_code,
            "usage": usage,
        }

    async def record_model_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
        include_observation: bool,
    ) -> None:
        del response, error_code, include_observation
        self.event_phases.append((phase, fact))


def _context(model: Model) -> tuple[RunContext[Any], ModelRequestContext]:
    context = RunContext(
        deps=type("Deps", (), {"correlation": {}})(),
        model=model,
        usage=RunUsage(),
        run_id="run",
        run_step=1,
    )
    request_context = ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    return context, request_context


def _observation_capability(
    recorder: _LifecycleRecorder,
    publish: object,
    *,
    run_sequence: int = 1,
    execution_id: str = "execution",
) -> ModelObservationCapability:
    return ModelObservationCapability(
        None,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id=execution_id,
        agent_run_sequence=run_sequence,
        session_id=None,
        agent_run_id=f"run-{run_sequence}",
        agent_id="agent",
        interaction_recorder=recorder,
        event_sink=publish,  # type: ignore[arg-type]
    )


def test_model_interaction_item_copies_nested_request_and_response_content() -> None:
    started_at = datetime.now(timezone.utc)
    source_request: dict[str, JsonValue] = {
        "messages": [{"parts": [{"text": "prompt"}]}]
    }
    source_response: JsonValue = {"parts": [{"text": "answer"}]}
    item = ModelInteractionItem(
        execution_id="execution",
        agent_run_sequence=1,
        depth=0,
        request_sequence=1,
        purpose="agent",
        step_index=1,
        output_retry_index=None,
        model={"provider": "test", "model_id": "test"},
        request=source_request,
        response=source_response,
        status="SUCCEEDED",
        error_code=None,
        duration_ns=1,
        usage=None,
        started_at=started_at,
        finished_at=started_at + timedelta(milliseconds=1),
    )
    source_request["messages"].clear()  # type: ignore[union-attr]
    source_response.clear()  # type: ignore[union-attr]
    assert item.request["messages"] == [{"parts": [{"text": "prompt"}]}]
    assert item.response == {"parts": [{"text": "answer"}]}


def test_request_events_and_lifecycle_share_the_same_identity_and_times() -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()
        published: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(
            event_type: ExecutionEventType,
            payload: JsonValue,
        ) -> None:
            published.append((event_type, payload))

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, request_context = _context(model)

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            assert [kind for kind, _payload in published] == [
                ExecutionEventType.MODEL_REQUEST_STARTED
            ]
            assert recorder.values[1]["status"] == "RUNNING"
            return ModelResponse(
                parts=[TextPart("answer")],
                usage=RequestUsage(input_tokens=11, output_tokens=7),
            )

        result = await capability.wrap_model_request(
            context,
            request_context=request_context,
            handler=handler,  # type: ignore[arg-type]
        )

        assert result.parts
        assert [kind for kind, _payload in published] == [
            ExecutionEventType.MODEL_REQUEST_STARTED,
            ExecutionEventType.MODEL_REQUEST_FINISHED,
        ]
        started = published[0][1]
        finished = published[1][1]
        assert isinstance(started, Mapping)
        assert isinstance(finished, Mapping)
        assert started["execution_id"] == "execution"
        assert started["agent_run_sequence"] == finished["agent_run_sequence"] == 1
        assert started["request_sequence"] == finished["request_sequence"] == 1
        assert started["status"] == "RUNNING"
        assert started["finished_at"] is None
        assert started["duration_ns"] is None
        assert started["usage"] is None
        assert finished["status"] == "SUCCEEDED"
        assert isinstance(started["started_at"], str)
        assert isinstance(finished["finished_at"], str)
        lifecycle = recorder.values[1]
        assert lifecycle["status"] == "SUCCEEDED"
        assert lifecycle["started_at"].isoformat() == started["started_at"]  # type: ignore[union-attr]
        assert lifecycle["finished_at"].isoformat() == finished["finished_at"]  # type: ignore[union-attr]
        assert finished["usage"] == {
            "model_requests": 1,
            "tool_calls": 0,
            "input_tokens": 11,
            "output_tokens": 7,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        assert recorder.finish_calls == {1: 1}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    (
        (ValueError("provider detail must stay private"), "FAILED", "INTERNAL_ERROR"),
        (asyncio.CancelledError(), "CANCELLED", None),
        (RunCancelled("run cancelled"), "CANCELLED", ErrorCode.EXECUTION_CANCELLED.value),
    ),
)
def test_handler_failure_and_cancellation_finish_once(
    error: BaseException,
    expected_status: str,
    expected_code: str | None,
) -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()
        published: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            published.append((event_type, payload))

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, request_context = _context(model)
        calls = 0

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            nonlocal calls
            calls += 1
            raise error

        with pytest.raises(type(error)):
            await capability.wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )

        assert calls == 1
        assert recorder.values[1]["status"] == expected_status
        assert recorder.values[1]["error_code"] == expected_code
        assert recorder.values[1]["usage"] is None
        assert recorder.finish_calls == {1: 1}
        assert [kind for kind, _payload in published] == [
            ExecutionEventType.MODEL_REQUEST_STARTED,
            ExecutionEventType.MODEL_REQUEST_FINISHED,
        ]
        assert published[-1][1]["status"] == expected_status  # type: ignore[index]
        assert published[-1][1]["error_code"] == expected_code  # type: ignore[index]

    asyncio.run(scenario())


def test_started_publication_failure_closes_request_without_calling_provider() -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()
        published: list[ExecutionEventType] = []

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            del payload
            published.append(event_type)
            if event_type is ExecutionEventType.MODEL_REQUEST_STARTED:
                raise OSError("private transport error")

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, request_context = _context(model)
        calls = 0

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[TextPart("unreachable")])

        with pytest.raises(AIError) as raised:
            await capability.wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )

        assert raised.value.code is ErrorCode.INTERNAL_ERROR
        assert raised.value.safe_details == {"phase": "model_request_event_publication"}
        assert calls == 0
        assert recorder.values[1]["status"] == "FAILED"
        assert recorder.values[1]["error_code"] == ErrorCode.INTERNAL_ERROR.value
        assert recorder.values[1]["usage"] is None
        assert published == [
            ExecutionEventType.MODEL_REQUEST_STARTED,
            ExecutionEventType.MODEL_REQUEST_FINISHED,
        ]

    asyncio.run(scenario())


def test_finished_publication_failure_keeps_successful_request_fact() -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            del payload
            if event_type is ExecutionEventType.MODEL_REQUEST_FINISHED:
                raise OSError("private transport error")

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, request_context = _context(model)

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            return ModelResponse(
                parts=[TextPart("answer")],
                usage=RequestUsage(input_tokens=5, output_tokens=2),
            )

        with pytest.raises(AIError) as raised:
            await capability.wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )

        assert raised.value.safe_details == {"phase": "model_request_event_publication"}
        assert recorder.values[1]["status"] == "SUCCEEDED"
        assert recorder.values[1]["error_code"] is None
        assert recorder.values[1]["usage"] is not None
        assert recorder.finish_calls == {1: 1}

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_location", ("record", "publish"))
def test_started_run_cancelled_is_recorded_as_cancelled(
    failure_location: str,
) -> None:
    async def scenario() -> None:
        class FailingRecorder(_LifecycleRecorder):
            async def record_model_event(
                self,
                fact: ModelRequestFact,
                *,
                phase: str,
                response: ModelResponse | None = None,
                error_code: str | None = None,
                include_observation: bool,
            ) -> None:
                if failure_location == "record" and phase == "started":
                    raise RunCancelled("cancelled while recording start")
                await super().record_model_event(
                    fact,
                    phase=phase,
                    response=response,
                    error_code=error_code,
                    include_observation=include_observation,
                )

        recorder = FailingRecorder()
        published: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            if (
                failure_location == "publish"
                and event_type is ExecutionEventType.MODEL_REQUEST_STARTED
            ):
                raise RunCancelled("cancelled while publishing start")
            published.append((event_type, payload))

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, request_context = _context(model)

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            raise AssertionError("provider must not run after start cancellation")

        with pytest.raises(RunCancelled):
            await capability.wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )

        assert recorder.values[1]["status"] == "CANCELLED"
        assert recorder.values[1]["error_code"] == ErrorCode.EXECUTION_CANCELLED.value
        assert recorder.finish_calls == {1: 1}
        assert published[-1][0] is ExecutionEventType.MODEL_REQUEST_FINISHED
        assert published[-1][1]["status"] == "CANCELLED"  # type: ignore[index]

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_count", (1, 2))
def test_cancellation_during_failed_handoff_preserves_failure_and_propagates_cancel(
    cancel_count: int,
) -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()
        finish_entered = asyncio.Event()
        release_finish = asyncio.Event()

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            del payload
            if event_type is ExecutionEventType.MODEL_REQUEST_FINISHED:
                finish_entered.set()
                await release_finish.wait()

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, request_context = _context(model)

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            raise ValueError("provider failure")

        task = asyncio.create_task(
            capability.wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(finish_entered.wait(), timeout=2)
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        release_finish.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert recorder.values[1]["status"] == "FAILED"
        assert recorder.values[1]["error_code"] == ErrorCode.INTERNAL_ERROR.value
        assert recorder.finish_calls == {1: 1}

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel_count", (1, 2))
def test_cancellation_during_success_handoff_cannot_rewrite_success(
    cancel_count: int,
) -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()
        finish_entered = asyncio.Event()
        release_finish = asyncio.Event()

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            del payload
            if event_type is ExecutionEventType.MODEL_REQUEST_FINISHED:
                finish_entered.set()
                await release_finish.wait()

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, request_context = _context(model)

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            return ModelResponse(
                parts=[TextPart("answer")],
                usage=RequestUsage(input_tokens=3, output_tokens=4),
            )

        task = asyncio.create_task(
            capability.wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )
        )
        await asyncio.wait_for(finish_entered.wait(), timeout=2)
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        release_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert recorder.values[1]["status"] == "SUCCEEDED"
        assert recorder.values[1]["usage"] is not None
        assert recorder.finish_calls == {1: 1}

    asyncio.run(scenario())


def test_compaction_run_cancelled_is_recorded_as_cancelled() -> None:
    async def scenario() -> None:
        async def cancel_model(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del messages, info
            raise RunCancelled("runtime stopped")

        model = FunctionModel(cancel_model)
        context = RunContext(
            deps=None,
            model=model,
            usage=RunUsage(),
            run_id="run",
            run_step=1,
        )
        journal = ModelRequestJournal(
            source_namespace="workspace",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id="run",
        )
        observations: list[tuple[str, ModelRequestFact, BaseException | None]] = []

        async def observe(
            _context: RunContext[Any],
            fact: ModelRequestFact,
            phase: str,
            _model: Model,
            _response: ModelResponse | None,
            error: BaseException | None,
        ) -> None:
            observations.append((phase, fact, error))

        wrapped = _ObservedCompactionModel(
            model,
            ctx=context,  # type: ignore[arg-type]
            journal=journal,
            observer=observe,
            recorder=None,
            source_messages=(),
        )
        with pytest.raises(RunCancelled):
            await wrapped.request([], None, ModelRequestParameters())

        assert [phase for phase, _fact, _error in observations] == [
            "started",
            "cancelled",
        ]
        assert observations[-1][1].status == "CANCELLED"
        with pytest.raises(RuntimeError, match="missing"):
            journal.current(observations[-1][1].request_sequence)

    asyncio.run(scenario())


def test_compaction_started_record_run_cancelled_closes_request() -> None:
    async def scenario() -> None:
        class FailingRecorder(_LifecycleRecorder):
            async def record_model_event(
                self,
                fact: ModelRequestFact,
                *,
                phase: str,
                response: ModelResponse | None = None,
                error_code: str | None = None,
                include_observation: bool,
            ) -> None:
                if phase == "started":
                    raise RunCancelled("cancelled while recording compaction start")
                await super().record_model_event(
                    fact,
                    phase=phase,
                    response=response,
                    error_code=error_code,
                    include_observation=include_observation,
                )

        lifecycle_recorder = FailingRecorder()

        async def publish(_event_type: ExecutionEventType, _payload: JsonValue) -> None:
            return

        capability = _observation_capability(lifecycle_recorder, publish)
        async def respond(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del messages, info
            raise AssertionError("provider must not run after start cancellation")

        model = FunctionModel(respond)
        context = RunContext(
            deps=None,
            model=model,
            usage=RunUsage(),
            run_id="run",
            run_step=1,
        )
        wrapped = _ObservedCompactionModel(
            model,
            ctx=context,  # type: ignore[arg-type]
            journal=capability._journal,
            observer=None,
            recorder=capability.record_external_model_request,
            source_messages=(),
        )
        with pytest.raises(RunCancelled):
            await wrapped.request([], None, ModelRequestParameters())

        assert lifecycle_recorder.values[1]["status"] == "CANCELLED"
        assert lifecycle_recorder.values[1]["error_code"] == ErrorCode.EXECUTION_CANCELLED.value
        assert lifecycle_recorder.finish_calls == {1: 1}

    asyncio.run(scenario())


def test_compaction_started_publish_run_cancelled_closes_external_request() -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            del payload
            if event_type is ExecutionEventType.MODEL_REQUEST_STARTED:
                raise RunCancelled("cancelled while publishing compaction start")

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        context, _request_context = _context(model)
        wrapped = _ObservedCompactionModel(
            model,
            ctx=context,
            journal=capability._journal,
            observer=None,
            recorder=capability.record_external_model_request,
            source_messages=(),
        )
        with pytest.raises(RunCancelled):
            await wrapped.request([], None, ModelRequestParameters())

        assert recorder.values[1]["status"] == "CANCELLED"
        assert recorder.values[1]["error_code"] == ErrorCode.EXECUTION_CANCELLED.value
        assert recorder.finish_calls == {1: 1}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "failure_kind",
    ("preparation", "storage", "async_cancel", "run_cancel"),
)
def test_compaction_preaccept_failures_propagate_without_terminal_record(
    failure_kind: str,
) -> None:
    async def scenario() -> None:
        if failure_kind == "preparation":
            failure: BaseException = ValueError("request preparation failed")
        elif failure_kind == "storage":
            failure = AIError(ErrorCode.STORAGE_CONFLICT)
        elif failure_kind == "async_cancel":
            failure = asyncio.CancelledError()
        else:
            failure = RunCancelled("cancelled during request preparation")

        store = StagingAgentRunStore()
        agent_run_id = "compaction-preaccept"
        run_recorder = AgentRunRecorder(
            store,
            execution_id="execution",
            agent_run_id=agent_run_id,
        )
        await run_recorder.register_agent_run(
            AgentRunRecord(
                agent_run_id,
                agent_conversation_id="conversation",
                agent_id="agent",
            )
        )

        def fail_intern_payload(_agent_run_id: str, _payload: bytes) -> tuple[str, int]:
            raise failure

        store.intern_payload = fail_intern_payload  # type: ignore[method-assign]
        published: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(
            event_type: ExecutionEventType,
            payload: JsonValue,
        ) -> None:
            published.append((event_type, payload))

        capability = ModelObservationCapability(
            None,
            source_namespace="workspace",
            tenant_id="tenant",
            execution_id="execution",
            session_id=None,
            agent_run_id=agent_run_id,
            agent_id="agent",
            interaction_recorder=run_recorder,
            event_sink=publish,
        )

        async def should_not_run(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del messages, info
            raise AssertionError("provider must not run before request acceptance")

        model = FunctionModel(should_not_run)
        context, _request_context = _context(model)
        wrapped = _ObservedCompactionModel(
            model,
            ctx=context,
            journal=capability._journal,
            observer=None,
            recorder=capability.record_external_model_request,
            source_messages=(),
        )

        with pytest.raises(type(failure)) as raised:
            await wrapped.request([], None, ModelRequestParameters())

        assert raised.value is failure
        assert await store.list_model_interactions(
            agent_run_id=agent_run_id
        ) == []
        assert await store.list_events(agent_run_id=agent_run_id) == []
        assert published == []
        with pytest.raises(RuntimeError, match="missing"):
            capability._journal.current(1)

    asyncio.run(scenario())


def test_compaction_after_recorder_acceptance_finishes_provider_failure() -> None:
    async def scenario() -> None:
        store = StagingAgentRunStore()
        agent_run_id = "compaction-accepted"
        run_recorder = AgentRunRecorder(
            store,
            execution_id="execution",
            agent_run_id=agent_run_id,
        )
        await run_recorder.register_agent_run(
            AgentRunRecord(
                agent_run_id,
                agent_conversation_id="conversation",
                agent_id="agent",
            )
        )
        published: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(
            event_type: ExecutionEventType,
            payload: JsonValue,
        ) -> None:
            published.append((event_type, payload))

        capability = ModelObservationCapability(
            None,
            source_namespace="workspace",
            tenant_id="tenant",
            execution_id="execution",
            session_id=None,
            agent_run_id=agent_run_id,
            agent_id="agent",
            interaction_recorder=run_recorder,
            event_sink=publish,
        )
        provider_error = ValueError("provider failed after request acceptance")

        async def fail_provider(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del messages, info
            raise provider_error

        model = FunctionModel(fail_provider)
        context, _request_context = _context(model)
        wrapped = _ObservedCompactionModel(
            model,
            ctx=context,
            journal=capability._journal,
            observer=None,
            recorder=capability.record_external_model_request,
            source_messages=(),
        )

        with pytest.raises(ValueError) as raised:
            await wrapped.request([], None, ModelRequestParameters())

        assert raised.value is provider_error
        values = await store.list_model_interactions(
            agent_run_id=agent_run_id
        )
        assert [(value.request_sequence, value.status) for value in values] == [
            (1, "FAILED")
        ]
        assert [event_type for event_type, _payload in published] == [
            ExecutionEventType.MODEL_REQUEST_STARTED,
            ExecutionEventType.MODEL_REQUEST_FINISHED,
        ]
        assert published[-1][1]["status"] == "FAILED"  # type: ignore[index]

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ("succeeded", "failed"))
@pytest.mark.parametrize("cancel_count", (1, 2))
def test_compaction_terminal_handoff_propagates_single_or_repeated_cancel(
    outcome: str,
    cancel_count: int,
) -> None:
    async def scenario() -> None:
        finish_entered = asyncio.Event()
        release_finish = asyncio.Event()
        facts: list[tuple[str, ModelRequestFact]] = []

        async def respond(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del messages, info
            if outcome == "failed":
                raise ValueError("compaction provider failure")
            return ModelResponse(parts=[TextPart("summary")])

        async def record(
            _context: RunContext[Any],
            fact: ModelRequestFact,
            phase: str,
            _model: Model,
            _response: ModelResponse | None,
            _error: BaseException | None,
            _messages: object,
            _model_settings: object,
            _parameters: object,
            _streaming: bool,
            _source_messages: object,
        ) -> None:
            facts.append((phase, fact))
            if phase in {"completed", "failed"}:
                finish_entered.set()
                await release_finish.wait()

        model = FunctionModel(respond)
        context = RunContext(
            deps=None,
            model=model,
            usage=RunUsage(),
            run_id="run",
            run_step=1,
        )
        journal = ModelRequestJournal(
            source_namespace="workspace",
            tenant_id="tenant",
            execution_id="execution",
            agent_run_id="run",
        )
        wrapped = _ObservedCompactionModel(
            model,
            ctx=context,  # type: ignore[arg-type]
            journal=journal,
            observer=None,
            recorder=record,
            source_messages=(),
        )
        task = asyncio.create_task(
            wrapped.request([], None, ModelRequestParameters())
        )
        await asyncio.wait_for(finish_entered.wait(), timeout=2)
        for _ in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        release_finish.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        expected_status = "SUCCEEDED" if outcome == "succeeded" else "FAILED"
        expected_phase = "completed" if outcome == "succeeded" else "failed"
        assert [phase for phase, _fact in facts] == ["started", expected_phase]
        assert facts[-1][1].status == expected_status

    asyncio.run(scenario())


def test_agent_run_sequence_separates_local_request_sequences() -> None:
    async def scenario() -> None:
        outputs: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            outputs.append((event_type, payload))

        model = TestModel()
        request_context = ModelRequestContext(
            model=model,
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            return ModelResponse(parts=[TextPart("ok")])

        for run_sequence in (1, 2):
            context, _ = _context(model)
            await _observation_capability(
                _LifecycleRecorder(), publish, run_sequence=run_sequence
            ).wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )

        starts = [
            payload
            for event_type, payload in outputs
            if event_type is ExecutionEventType.MODEL_REQUEST_STARTED
        ]
        assert [(item["agent_run_sequence"], item["request_sequence"]) for item in starts] == [  # type: ignore[index]
            (1, 1),
            (2, 1),
        ]

    asyncio.run(scenario())


def test_concurrent_execution_ids_separate_the_same_local_request_sequence() -> None:
    async def scenario() -> None:
        outputs: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            outputs.append((event_type, payload))

        model = TestModel()
        request_context = ModelRequestContext(
            model=model,
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            return ModelResponse(parts=[TextPart("ok")])

        for execution_id in ("execution-a", "execution-b"):
            context, _ = _context(model)
            await _observation_capability(
                _LifecycleRecorder(),
                publish,
                execution_id=execution_id,
            ).wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )

        starts = [
            payload
            for event_type, payload in outputs
            if event_type is ExecutionEventType.MODEL_REQUEST_STARTED
        ]
        assert [
            (item["execution_id"], item["agent_run_sequence"], item["request_sequence"])
            for item in starts
        ] == [
            ("execution-a", 1, 1),
            ("execution-b", 1, 1),
        ]

    asyncio.run(scenario())


def test_output_retry_uses_new_request_and_keeps_prior_success() -> None:
    async def scenario() -> None:
        recorder = _LifecycleRecorder()
        published: list[tuple[ExecutionEventType, JsonValue]] = []

        async def publish(event_type: ExecutionEventType, payload: JsonValue) -> None:
            published.append((event_type, payload))

        capability = _observation_capability(recorder, publish)
        model = TestModel()
        request_context = ModelRequestContext(
            model=model,
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )

        async def handler(_request: ModelRequestContext) -> ModelResponse:
            return ModelResponse(parts=[TextPart("ok")])

        for retry in (0, 1):
            context = RunContext(
                deps=type("Deps", (), {"correlation": {}})(),
                model=model,
                usage=RunUsage(),
                run_id="run",
                run_step=1,
                retry=retry,
            )
            await capability.wrap_model_request(
                context,
                request_context=request_context,
                handler=handler,  # type: ignore[arg-type]
            )

        assert recorder.values[1]["status"] == "SUCCEEDED"
        assert recorder.values[2]["status"] == "SUCCEEDED"
        assert [
            (payload["request_sequence"], payload["output_retry_index"])
            for event_type, payload in published
            if event_type is ExecutionEventType.MODEL_REQUEST_STARTED
        ] == [(1, None), (2, 1)]  # type: ignore[index]
        assert recorder.finish_calls == {1: 1, 2: 1}

    asyncio.run(scenario())


class _BlockingModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:blocking"
    vision = False
    model_digest = "b" * 64
    contract: dict[str, JsonValue] = {"provider": "test", "model": "blocking"}

    def __init__(
        self,
        entered: asyncio.Event,
        release: asyncio.Event,
        *,
        first_tool: bool = False,
    ) -> None:
        self._entered = entered
        self._release = release
        self._first_tool = first_tool

    def materialize(self) -> FunctionModel:
        calls = 0

        async def respond(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del messages, info
            nonlocal calls
            calls += 1
            if self._first_tool and calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "complete",
                            {},
                            tool_call_id="complete-1",
                        )
                    ]
                )
            self._entered.set()
            await self._release.wait()
            return ModelResponse(
                parts=[TextPart("done")],
                usage=RequestUsage(input_tokens=8, output_tokens=3),
            )

        return _BlockingFunctionModel(respond)


class _BlockingFunctionModel(FunctionModel):
    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: object | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        del run_context
        response = await self.request(
            messages,
            model_settings,
            model_request_parameters,
        )
        yield CompletedStreamedResponse(
            response,
            model_request_parameters=model_request_parameters,
            replay_events=True,
        )


class _BlockingModels:
    def __init__(self, *, first_tool: bool = False) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.binding = _BlockingModelBinding(
            self.entered,
            self.release,
            first_tool=first_tool,
        )

    def capture(self) -> _BlockingModels:
        return self

    def resolve(self, route_id: str) -> _BlockingModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return self.binding

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _BlockingModelBinding:
        if route_id not in {None, "default"} or dict(payload) != self.binding.contract:
            raise AssertionError("unexpected model snapshot")
        return self.binding


@pytest.mark.parametrize("persistent", (False, True))
def test_execution_stream_and_history_expose_blocked_request_before_response(
    persistent: bool,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        models = _BlockingModels()
        group = CapabilityGroup[object]("application")
        group.agent("default", model="default", allow_tools=())
        storage_path = tmp_path / "runtime-storage"
        storage = (
            RuntimeStorage.filesystem(storage_path)
            if persistent
            else RuntimeStorage.in_memory()
        )

        async with Runtime.open(
            "default",
            models=models,  # type: ignore[arg-type]
            storage=storage,
            capabilities=(group,),
            metrics=Metrics.in_memory(),
        ) as runtime:
            execution = await runtime.agents.get("default").start("TOP_SECRET_PROMPT")
            wait_task = asyncio.create_task(execution.wait())
            await asyncio.wait_for(models.entered.wait(), timeout=5)

            async def next_started_event() -> object:
                async for tree_event in execution.watch():
                    if (
                        tree_event.event.event_type
                        == ExecutionEventType.MODEL_REQUEST_STARTED.value
                    ):
                        return tree_event
                raise AssertionError("execution stream ended before request start")

            tree_event = await asyncio.wait_for(next_started_event(), timeout=5)
            event = tree_event.event  # type: ignore[attr-defined]
            assert tree_event.execution_id == execution.execution_id  # type: ignore[attr-defined]
            assert isinstance(event.payload, Mapping)
            assert event.payload["status"] == "RUNNING"
            assert event.payload["request_sequence"] == 1
            assert "TOP_SECRET_PROMPT" not in repr(event.payload)
            page = await execution.model_interactions(include_content=False)
            assert len(page.items) == 1
            running = page.items[0]
            assert running.status == "RUNNING"
            assert running.request == {}
            assert running.response is None
            assert running.content_included is False
            assert running.started_at is not None
            assert running.finished_at is None
            assert running.duration_ns is None
            assert running.usage is None
            assert running.error_code is None
            assert running.request_sequence == event.payload["request_sequence"]

            models.release.set()
            result = await wait_task
            assert result.status.value == "SUCCEEDED"
            finished_page = await execution.model_interactions(include_content=True)
            finished = finished_page.items[0]
            assert finished.status == "SUCCEEDED"
            assert finished.request["messages"]
            assert finished.response is not None
            assert finished.finished_at is not None
            assert finished.duration_ns is not None
            assert finished.usage is not None

        if persistent:
            async with Runtime.open(
                "default",
                models=models,  # type: ignore[arg-type]
                storage=RuntimeStorage.filesystem(storage_path),
                capabilities=(group,),
                metrics=Metrics.in_memory(),
            ) as reopened:
                assert reopened.history is not None
                page = await reopened.history.model_interactions(
                    execution.execution_id,
                    principal=reopened.default_principal,
                    include_content=True,
                )
                assert [(item.request_sequence, item.status) for item in page.items] == [
                    (1, "SUCCEEDED")
                ]

    asyncio.run(scenario())


def test_history_cursor_keeps_lifecycle_identity_during_archive_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        models = _BlockingModels(first_tool=True)
        group = CapabilityGroup[object]("application")

        def complete(_context: object) -> str:
            return "tool completed"

        group.tool(complete, effect_policy="none")
        group.agent("default", model="default", allow_tools=("complete",))
        storage = RuntimeStorage.filesystem(tmp_path / "handoff-storage")

        async with Runtime.open(
            "default",
            models=models,  # type: ignore[arg-type]
            storage=storage,
            capabilities=(group,),
            metrics=Metrics.in_memory(),
        ) as runtime:
            execution = await runtime.agents.get("default").start("continue")
            wait_task = asyncio.create_task(execution.wait())
            try:
                await asyncio.wait_for(models.entered.wait(), timeout=5)
                first = await execution.model_interactions(
                    limit=1,
                    include_content=True,
                )
                assert [(item.request_sequence, item.status) for item in first.items] == [
                    (1, "SUCCEEDED")
                ]
                assert first.next_cursor is not None

                archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
                original_list = archive.list_model_interactions
                archive_read_started = asyncio.Event()
                allow_archive_read_to_return = asyncio.Event()
                archive_snapshot: list[object] = []

                async def delayed_archive_read(
                    *,
                    agent_run_id: str,
                    after_request_sequence: int | None = None,
                    limit: int | None = None,
                ) -> list[object]:
                    values = await original_list(
                        agent_run_id=agent_run_id,
                        after_request_sequence=after_request_sequence,
                        limit=limit,
                    )
                    if after_request_sequence == 1:
                        archive_snapshot.extend(values)
                        archive_read_started.set()
                        await allow_archive_read_to_return.wait()
                    return values

                monkeypatch.setattr(archive, "list_model_interactions", delayed_archive_read)
                continuation = asyncio.create_task(
                    execution.model_interactions(
                        cursor=first.next_cursor,
                        limit=1,
                        include_content=True,
                    )
                )
                await asyncio.wait_for(archive_read_started.wait(), timeout=3)
                assert archive_snapshot == []

                models.release.set()
                result = await wait_task
                assert result.status.value == "SUCCEEDED"
                await storage.retention.release_execution_handoff(
                    execution.execution_id,
                    tenant_id=runtime.default_principal.tenant_id,
                )
                allow_archive_read_to_return.set()

                captured_page = await continuation
                assert [
                    (item.request_sequence, item.status)
                    for item in captured_page.items
                ] == [(2, "RUNNING")]
                assert captured_page.items[0].request["messages"]
                assert captured_page.items[0].response is None

                durable_page = await execution.model_interactions(
                    cursor=first.next_cursor,
                    limit=1,
                    include_content=True,
                )
                assert [
                    (item.request_sequence, item.status)
                    for item in durable_page.items
                ] == [(2, "SUCCEEDED")]
            finally:
                models.release.set()
                if not wait_task.done():
                    await asyncio.gather(wait_task, return_exceptions=True)

    asyncio.run(scenario())


def test_public_content_mutation_does_not_change_live_or_archived_history(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        models = _BlockingModels()
        group = CapabilityGroup[object]("application")
        group.agent("default", model="default", allow_tools=())
        storage = RuntimeStorage.filesystem(tmp_path / "content-storage")

        async with Runtime.open(
            "default",
            models=models,  # type: ignore[arg-type]
            storage=storage,
            capabilities=(group,),
            metrics=Metrics.in_memory(),
        ) as runtime:
            execution = await runtime.agents.get("default").start("preserve content")
            wait_task = asyncio.create_task(execution.wait())
            try:
                await asyncio.wait_for(models.entered.wait(), timeout=5)
                live = await execution.model_interactions(include_content=True)
                expected_request = deepcopy(dict(live.items[0].request))
                messages = live.items[0].request["messages"]
                assert isinstance(messages, list) and messages
                messages.clear()

                reread_live = await execution.model_interactions(include_content=True)
                assert dict(reread_live.items[0].request) == expected_request

                models.release.set()
                result = await wait_task
                assert result.status.value == "SUCCEEDED"
                terminal = await execution.model_interactions(include_content=True)
                expected_request = deepcopy(dict(terminal.items[0].request))
                expected_response = deepcopy(terminal.items[0].response)
                assert isinstance(expected_response, dict)
                terminal.items[0].request["messages"].clear()  # type: ignore[union-attr]
                terminal.items[0].response.clear()

                reread_terminal = await execution.model_interactions(include_content=True)
                assert dict(reread_terminal.items[0].request) == expected_request
                assert reread_terminal.items[0].response == expected_response

                await storage.retention.release_execution_handoff(
                    execution.execution_id,
                    tenant_id=runtime.default_principal.tenant_id,
                )
                reread_archive = await execution.model_interactions(include_content=True)
                assert dict(reread_archive.items[0].request) == expected_request
                assert reread_archive.items[0].response == expected_response
            finally:
                models.release.set()
                if not wait_task.done():
                    await asyncio.gather(wait_task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("buffer_limit", ("items", "bytes"))
def test_history_refresh_survives_event_buffer_fallback_for_blocked_request(
    buffer_limit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        if buffer_limit == "items":
            monkeypatch.setattr(runtime_event, "_QUEUE_LIMIT", 1)
        else:
            original_init = runtime_event.LiveExecutionEventBroker.__init__

            def small_buffer(
                broker: runtime_event.LiveExecutionEventBroker,
                *,
                max_bytes: int = runtime_event._DEFAULT_BUFFER_BYTES,
            ) -> None:
                del max_bytes
                original_init(broker, max_bytes=1)

            monkeypatch.setattr(
                runtime_event.LiveExecutionEventBroker,
                "__init__",
                small_buffer,
            )

        replays: list[str] = []
        original_replay = runtime_event.LiveExecutionEventBroker._require_replay

        def note_replay(
            broker: runtime_event.LiveExecutionEventBroker,
            execution_id: str,
        ) -> None:
            replays.append(execution_id)
            original_replay(broker, execution_id)

        monkeypatch.setattr(
            runtime_event.LiveExecutionEventBroker,
            "_require_replay",
            note_replay,
        )
        models = _BlockingModels(first_tool=True)
        group = CapabilityGroup[object]("application")

        def complete(_context: object) -> str:
            return "tool completed"

        group.tool(complete, effect_policy="none")
        group.agent("default", model="default", allow_tools=("complete",))
        metric_store = InMemoryMetricStore()
        metrics = Metrics.from_store(metric_store, namespace="model-lifecycle-fallback")
        metric_start = datetime.now(timezone.utc) - timedelta(seconds=1)

        async with Runtime.open(
            "default",
            models=models,  # type: ignore[arg-type]
            storage=RuntimeStorage.in_memory(),
            capabilities=(group,),
            metrics=metrics,
        ) as runtime:
            execution = await runtime.agents.get("default").start("continue")
            wait_task = asyncio.create_task(execution.wait())
            try:
                await asyncio.wait_for(models.entered.wait(), timeout=5)
                assert execution.execution_id in replays
                running_page = await execution.model_interactions(
                    include_content=False
                )
                assert [
                    (item.request_sequence, item.status)
                    for item in running_page.items
                ] == [(1, "SUCCEEDED"), (2, "RUNNING")]
                assert running_page.items[1].started_at is not None
                assert running_page.items[1].finished_at is None

                models.release.set()
                result = await wait_task
                assert result.status.value == "SUCCEEDED"
                await runtime.metrics.flush()
                metric_end = datetime.now(timezone.utc) + timedelta(seconds=1)
                query = await metrics.query(
                    MetricQuery(
                        "linktools.model.request.count",
                        MetricWindow.between(metric_start, metric_end),
                    )
                )
                assert len(query.points) == 1
                assert query.points[0].value == 2
                assert query.points[0].sample_count == 2
            finally:
                models.release.set()
                if not wait_task.done():
                    await asyncio.gather(wait_task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("buffer_limit", ("items", "bytes"))
def test_active_and_reconnected_streams_follow_broker_replay_fallback(
    buffer_limit: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        if buffer_limit == "items":
            monkeypatch.setattr(runtime_event, "_QUEUE_LIMIT", 1)
            broker = runtime_event.LiveExecutionEventBroker()
        else:
            broker = runtime_event.LiveExecutionEventBroker(max_bytes=1)
        broker.register_local_producer("execution", 1)
        active = broker.subscribe("execution")
        broker.publish_event(
            "execution",
            ExecutionEventType.MODEL_REQUEST_STARTED.value,
            {"request_sequence": 1, "status": "RUNNING"},
            durable_sequence=None,
        )
        if buffer_limit == "items":
            broker.publish_event(
                "execution",
                ExecutionEventType.MODEL_REQUEST_FINISHED.value,
                {"request_sequence": 1, "status": "SUCCEEDED"},
                durable_sequence=None,
            )

        active_marker = await anext(active)
        assert isinstance(active_marker, runtime_event._LiveReplayRequired)
        assert active.replay_required

        reconnected = broker.subscribe("execution")
        reconnect_marker = await anext(reconnected)
        assert isinstance(reconnect_marker, runtime_event._LiveReplayRequired)
        assert reconnected.replay_required
        await active.close()
        await reconnected.close()

    asyncio.run(scenario())
