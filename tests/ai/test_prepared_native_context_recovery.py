#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current native message contexts survive the prepared-before-admission boundary."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import fields
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import TaskStatus
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.inputs import AgentTaskInput, AgentTaskInputContext
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task import _local as task_local

from ._task_test_helpers import task_engine_scope
from .test_model_interaction_direct_result import _Binding


_RESPONSE_FIELDS = frozenset(field.name for field in fields(ModelResponse))


@pytest.mark.asyncio
@pytest.mark.parametrize("direct", (False, True))
async def test_prepared_native_context_reopens_without_rebuilding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, direct: bool,
) -> None:
    monkeypatch.setattr(task_local, "_LEASE_SECONDS", 1)
    monkeypatch.setattr(task_local, "_HEARTBEAT_SECONDS", 0.1)
    prior = ModelResponse(
        [TextPart("previous answer")], model_name="previous-model",
        provider_name="openai", provider_response_id="previous-response",
        usage=RequestUsage(input_tokens=7, output_tokens=3),
        **{name: value for name, value in (("model_id", "previous-route"), ("canonical_system", "openai"))
           if name in _RESPONSE_FIELDS},
    )
    context = ExecutionInputContext.from_messages((
        ModelRequest([UserPromptPart("previous question")]), prior,
    ))
    builds: list[str] = []
    requests: list[str] = []
    observations: list[str] = []
    entered = asyncio.Event()

    def check_history(messages: list[ModelMessage]) -> None:
        historical = [message for message in messages if isinstance(message, ModelResponse)]
        assert len(historical) == 1
        value = historical[0]
        assert (value.model_name, value.provider_name, value.provider_response_id) == (
            "previous-model", "openai", "previous-response",
        )
        if "model_id" in _RESPONSE_FIELDS:
            assert value.model_id == "previous-route"
        if "canonical_system" in _RESPONSE_FIELDS:
            assert value.canonical_system == "openai"
        assert (value.usage.input_tokens, value.usage.output_tokens) == (7, 3)
        assert value.parts == prior.parts

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        requests.append("provider")
        check_history(messages)
        yield "resumed answer"

    class ObserveDirect(AbstractCapability):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position="innermost")

        async def wrap_model_request(
            self, ctx: RunContext[object], *, request_context: ModelRequestContext,
            handler: WrapModelRequestHandler,
        ) -> ModelResponse:
            check_history(request_context.messages)
            observations.append("restored native context")
            if direct:
                return ModelResponse(
                    [TextPart("resumed answer")], model_name="returned-model",
                    provider_name="openai", provider_response_id="returned-response",
                    usage=RequestUsage(input_tokens=4, output_tokens=2),
                    **{name: value for name, value in (("model_id", "returned-route"), ("canonical_system", "openai"))
                       if name in _RESPONSE_FIELDS},
                )
            return await handler(request_context)

    def features() -> RuntimeFeatures:
        value = RuntimeFeatures("prepared-native")
        value.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
        value.model("default", _Binding(FunctionModel(stream_function=stream, model_name="live-model")))
        value.capability(ExecutionCapability(retain_history=True))
        value.capability(TraceCapability())
        value.capability(ObserveDirect(), id="native-context-observer")
        return value

    async def build_input(value: AgentTaskInputContext) -> str:
        builds.append(value.node_id)
        return "current question"

    storage = PersistenceStorage.sqlite(tmp_path / "prepared.sqlite")
    graph_id = "prepared-native-graph"
    async with Persistence.open("prepared-native", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features(),
    ) as runtime, task_engine_scope(runtime, persistence) as engine:
        worker = Task.from_agent("prepared-native-worker", runtime.agents.get(), build_input=build_input)
        graph = TaskGraph(graph_id, (TaskNode(
            "worker", task=worker, input=AgentTaskInput("base question", input_context=context),
        ),))
        original_start = type(worker.agent_execution).start

        async def pause_before_start(*args: object, **kwargs: object) -> object:
            entered.set()
            await asyncio.Event().wait()
            return await original_start(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(type(worker.agent_execution), "start", pause_before_start)
            await engine.with_definitions(worker).start(
                graph, idempotency_key="prepared-native-start", principal=runtime.default_principal,
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            prepared = await persistence.task.tasks.get_prepared_input(
                graph_id, "worker", tenant_id=runtime.tenant_id,
            )
            assert prepared is not None
            assert builds == ["worker"] and requests == [] and observations == []
            await runtime.close()

    await asyncio.sleep(1.1)
    async with Persistence.open("prepared-native", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features(),
    ) as runtime, task_engine_scope(runtime, persistence) as engine:
        worker = Task.from_agent("prepared-native-worker", runtime.agents.get(), build_input=build_input)
        recovered_engine = engine.with_definitions(worker)
        await recovered_engine.recover_pending()
        recovered = await recovered_engine.get(graph_id)
        result = (await recovered.wait(timeout_seconds=10)).result
        assert result.wait_status is TaskStatus.SUCCEEDED, result.node_states
        assert builds == ["worker"]
        assert requests == ([] if direct else ["provider"])
        assert observations == ["restored native context"]
        state, = result.node_states
        assert state.execution_id is not None
        stored_prepared = await persistence.task.tasks.get_prepared_input(
            graph_id, "worker", tenant_id=runtime.tenant_id,
        )
        assert stored_prepared == prepared
        items = (await runtime.history.model_interactions(
            state.execution_id, principal=runtime.default_principal, include_content=True,
        )).items
        assert len(items) == 1
        assert items[0].status == "SUCCEEDED"
        execution = await runtime.executions.get(state.execution_id)
        events = (await execution.list_events(include_content=True, limit=100)).items
        assert len([event for event in events if event.event_type == "MODEL_REQUEST_FINISHED"]) == 1
        assert all("is_direct_result" not in event.payload for event in events)
        if direct:
            assert items[0].request == {}
            if "model_id" in _RESPONSE_FIELDS:
                assert items[0].response["model_id"] == "returned-route"
            else:
                assert "model_id" not in items[0].response
            if "canonical_system" in _RESPONSE_FIELDS:
                assert items[0].response["canonical_system"] == "openai"
            else:
                assert "canonical_system" not in items[0].response
            assert items[0].response["provider_response_id"] == "returned-response"
        else:
            if "model_id" in _RESPONSE_FIELDS:
                assert "previous-route" in str(items[0].request)
            assert "previous-response" in str(items[0].request)

    async with Persistence.open("prepared-native", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=features(),
    ) as runtime:
        persisted = (await runtime.history.model_interactions(
            state.execution_id, principal=runtime.default_principal, include_content=True,
        )).items
        assert persisted == items
        assert await persistence.task.tasks.get_prepared_input(
            graph_id, "worker", tenant_id=runtime.tenant_id,
        ) == prepared
    assert builds == ["worker"]
    assert requests == ([] if direct else ["provider"])
    assert observations == ["restored native context"]
