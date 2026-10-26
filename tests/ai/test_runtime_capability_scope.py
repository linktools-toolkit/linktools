#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped group capabilities share native hooks across Agent runs."""

from __future__ import annotations

from linktools.ai.persistence import Persistence
from linktools.ai.task_engine import TaskEngine
from ._runtime_test_helpers import model_features

import asyncio
from dataclasses import replace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    CombinedCapability,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from linktools.ai.capability import ExecutionCapability
from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model._binding import _RetryingModel
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

from ._runtime_test_helpers import _Models, _UsageFunctionModel

pytestmark = pytest.mark.asyncio


class _Backend:
    def __init__(self) -> None:
        self.semaphore = asyncio.Semaphore(1)
        self.requests: list[str] = []
        self.entered: list[str] = []
        self.released: list[str] = []
        self.runs: list[AgentContext[None]] = []
        self.tools: list[tuple[str, str]] = []
        self.active = 0
        self.peak = 0
        self.changed = asyncio.Event()

    @asynccontextmanager
    async def slot(self, owner: str) -> AsyncIterator[None]:
        self.requests.append(owner)
        self.changed.set()
        await self.semaphore.acquire()
        self.entered.append(owner)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            self.active -= 1
            self.released.append(owner)
            self.semaphore.release()

    async def requested(self, count: int) -> None:
        async def wait() -> None:
            while len(self.requests) < count:
                self.changed.clear()
                if len(self.requests) < count:
                    await self.changed.wait()
        await asyncio.wait_for(wait(), 3)


class _GlobalPolicy(AbstractCapability[AgentContext[None]]):
    id = "host.runtime_capability"

    def __init__(self, backend: _Backend) -> None:
        self.backend = backend

    async def for_run(self, ctx: RunContext[AgentContext[None]]) -> _GlobalPolicy:
        self.backend.runs.append(ctx.deps)
        instance = _GlobalPolicy(self.backend)
        instance.id = self.id
        return instance

    async def wrap_model_request(
        self, ctx: RunContext[AgentContext[None]], *, request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        async with self.backend.slot(ctx.deps.execution_id):
            return await handler(request_context)

    async def wrap_tool_execute(
        self, ctx: RunContext[AgentContext[None]], *, call: ToolCallPart,
        tool_def: ToolDefinition, args: dict[str, Any], handler: WrapToolExecuteHandler,
    ) -> Any:
        self.backend.tools.append((ctx.deps.execution_id, tool_def.name))
        return await handler(args)


class _Unselected(AbstractCapability[AgentContext[None]]):
    id = "host.unselected"

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        raise AssertionError("Agent-selectable capability must remain excluded")


def _group() -> RuntimeFeatures[None]:
    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    for name in ("default", "worker"):
        group.agent(name, model="default", allow_tools=(), allow_skills=(),
                    allow_subagents=(), allow_capabilities=())
    group.capability(_Unselected(), scope="agent")
    return group


def _policies(group: RuntimeFeatures[None], *capabilities: AbstractCapability[AgentContext[None]]) -> RuntimeFeatures[None]:
    for index, capability in enumerate(capabilities):
        group.capability(capability, id=getattr(capability, "id", None) or f"host.policy.{index}", scope="runtime")
    return group


async def _immediate(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart("done")])


async def test_runtime_capability_shares_backend_across_agents_and_agent_tasks_without_selection() -> None:
    backend = _Backend()
    finish, streamed, ordinary_ran = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def chunks(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "first"
        streamed.set()
        await finish.wait()
        yield "last"

    async def ordinary(_context: TaskNodeContext[None]) -> JsonValue:
        ordinary_ran.set()
        return "ordinary"

    policies = _policies(_group(), _GlobalPolicy(backend))
    policies.model("default", _Models(FunctionModel(stream_function=chunks)))
    async with Persistence.open('runtime_capability', storage=PersistenceStorage.in_memory()) as persistence_143, Runtime.open(persistence=persistence_143, features=replace(await policies.capture(), filter=lambda _agent, item: item.kind != 'capability')) as runtime, TaskEngine.open(persistence=persistence_143) as engine:
        late = _Unselected()
        late.id = "host.late"
        policies.capability(late, scope="runtime")
        first = await runtime.agents.get().start("first")
        await asyncio.wait_for(streamed.wait(), 3)
        agent_task = Task.from_agent("test.worker", runtime.agents.get("worker"))
        plain_task = Task("test.ordinary", ordinary, effect_policy="none")
        graph = TaskGraph("runtime_capability-graph", (
            TaskNode("worker", task=agent_task, input=AgentTaskInput("second")),
            TaskNode("ordinary", task=plain_task),
        ))
        run = await engine.with_definitions(agent_task, plain_task).start(graph, idempotency_key="start")
        await backend.requested(2)
        await asyncio.wait_for(ordinary_ran.wait(), 3)
        assert backend.entered == [first.execution_id]
        assert backend.active == 1
        finish.set()
        assert (await first.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        second = await run.execution("worker")
        assert backend.entered == [first.execution_id, second.execution_id]
        assert backend.entered == backend.released and backend.peak == 1
        assert {context.execution_id for context in backend.runs} == set(backend.entered)


@pytest.mark.parametrize("outcome", ("complete", "cancel", "error"))
async def test_stream_permit_spans_consumption_and_releases_on_every_exit(outcome: str) -> None:
    backend = _Backend()
    streamed, finish, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def chunks(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        try:
            yield "first"
            streamed.set()
            await finish.wait()
            if outcome == "error":
                raise RuntimeError("stream failed")
            yield "last"
        finally:
            closed.set()

    async with Persistence.open('runtime_capability-stream', storage=PersistenceStorage.in_memory()) as persistence_190, Runtime.open(persistence=persistence_190, features=model_features(_Models(FunctionModel(stream_function=chunks)), _policies(_group(), _GlobalPolicy(backend)))) as runtime:
        execution = await runtime.agents.get().start("go")
        await asyncio.wait_for(streamed.wait(), 3)
        assert backend.active == 1 and backend.released == [] and not closed.is_set()
        if outcome == "cancel":
            await execution.cancel()
        else:
            finish.set()
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is {
            "complete": ExecutionStatus.SUCCEEDED,
            "cancel": ExecutionStatus.CANCELLED,
            "error": ExecutionStatus.FAILED,
        }[outcome]
        assert closed.is_set() and backend.active == 0
        assert backend.entered == backend.released == [execution.execution_id]


async def test_queued_cancellation_does_not_release_another_executions_permit() -> None:
    backend = _Backend()
    finish = asyncio.Event()

    async def chunks(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        yield "first"
        await finish.wait()
        yield "last"

    async with Persistence.open('runtime_capability-queue', storage=PersistenceStorage.in_memory()) as persistence_220, Runtime.open(persistence=persistence_220, features=model_features(_Models(FunctionModel(stream_function=chunks)), _policies(_group(), _GlobalPolicy(backend)))) as runtime:
        first = await runtime.agents.get().start("first")
        await backend.requested(1)
        second = await runtime.agents.get("worker").start("second")
        await backend.requested(2)
        await second.cancel()
        assert (await second.wait(timeout_seconds=5)).result.status is ExecutionStatus.CANCELLED
        assert backend.active == 1 and backend.released == []
        finish.set()
        assert (await first.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert backend.entered == backend.released == [first.execution_id]


async def test_shared_host_backend_spans_runtime_instances() -> None:
    backend = _Backend()
    runtime_capability = _GlobalPolicy(backend)
    finish = asyncio.Event()

    async def blocked(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        await finish.wait()
        return ModelResponse(parts=[TextPart("done")])

    async with Persistence.open('first', storage=PersistenceStorage.in_memory()) as persistence_245, Runtime.open(persistence=persistence_245, features=model_features(_Models(_UsageFunctionModel(blocked)), _policies(_group(), runtime_capability))) as first_runtime, Persistence.open('second', storage=PersistenceStorage.in_memory()) as persistence_248, Runtime.open(persistence=persistence_248, features=model_features(_Models(_UsageFunctionModel(_immediate)), _policies(_group(), runtime_capability))) as second_runtime:
        first = await first_runtime.agents.get().start("first")
        await backend.requested(1)
        second = await second_runtime.agents.get().start("second")
        await backend.requested(2)
        assert backend.entered == [first.execution_id]
        finish.set()
        for execution in (first, second):
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert {context.namespace for context in backend.runs} == {"first", "second"}
        assert backend.peak == 1 and backend.entered == backend.released


async def test_runtime_capability_sees_logical_model_call_not_each_transport_retry() -> None:
    backend = _Backend()
    attempts = 0

    async def transient(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ModelHTTPError(429, "fake")
        return ModelResponse(parts=[TextPart("done")])

    model = _RetryingModel(_UsageFunctionModel(transient), max_retries=2, retry_delay=0)
    async with Persistence.open('runtime_capability-retry', storage=PersistenceStorage.in_memory()) as persistence_276, Runtime.open(persistence=persistence_276, features=model_features(_Models(model), _policies(_group(), _GlobalPolicy(backend)))) as runtime:
        execution = await runtime.agents.get().start("go")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert attempts == 3
        assert backend.entered == backend.released == [execution.execution_id]


async def test_runtime_capability_covers_delegated_agents_and_tool_hooks() -> None:
    backend = _Backend()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for msg in messages for part in msg.parts):
            return ModelResponse(parts=[TextPart("done")])
        if any(tool.name == "delegate_task" for tool in info.function_tools):
            return ModelResponse(parts=[ToolCallPart("delegate_task", {"subagent_id": "child", "task": "work"})])
        return ModelResponse(parts=[ToolCallPart("ping", {})])

    async def ping(_context: AgentContext[None]) -> str:
        return "pong"

    group: RuntimeFeatures[None] = RuntimeFeatures("delegation")
    group.tool(ping, effect_policy="none")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=("child",))
    group.agent("child", model="default", allow_tools=("ping",), allow_skills=(), allow_subagents=())
    async with Persistence.open('runtime_capability-delegation', storage=PersistenceStorage.in_memory()) as persistence_303, Runtime.open(persistence=persistence_303, features=model_features(_Models(_UsageFunctionModel(model)), _policies(group, _GlobalPolicy(backend)))) as runtime:
        execution = await runtime.agents.get().start("go")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert {name for _owner, name in backend.tools} == {"delegate_task", "ping"}
        assert len({context.execution_id for context in backend.runs}) == 2
        assert backend.active == 0 and backend.entered == backend.released


@pytest.mark.parametrize("kind", ("not_capability", "deferred", "reserved", "duplicate", "combined", "container_id"))
async def test_runtime_capability_rejects_invalid_or_not_always_on_entries(kind: str) -> None:
    runtime_capability: object = _GlobalPolicy(_Backend())
    if kind == "not_capability":
        runtime_capability = object()
    elif kind == "deferred":
        runtime_capability.defer_loading = True
    elif kind == "reserved":
        runtime_capability.id = "linktools.ai.model-observation"
    elif kind == "combined":
        runtime_capability.defer_loading = True
        runtime_capability = CombinedCapability([runtime_capability])
    elif kind == "container_id":
        runtime_capability = CombinedCapability([runtime_capability], id="linktools.reserved")
    entries = (runtime_capability, runtime_capability) if kind == "duplicate" else (runtime_capability,)
    with pytest.raises((TypeError, AIError)):
        async with Persistence.open('runtime-policy-invalid', storage=PersistenceStorage.in_memory()) as persistence_330, Runtime.open(persistence=persistence_330, features=model_features(_Models(_UsageFunctionModel(_immediate)), _policies(_group(), *entries))):
            raise AssertionError("invalid runtime capability was accepted")


async def test_selected_capability_cannot_silently_replace_runtime_capability_with_the_same_id() -> None:
    backend = _Backend()
    group: RuntimeFeatures[None] = RuntimeFeatures("collision")
    group.capability(_GlobalPolicy(_Backend()), scope="agent")
    group.agent("default", model="default", allow_tools=(), allow_skills=(),
                allow_subagents=(), allow_capabilities=("host.runtime_capability",))
    with pytest.raises(AIError) as raised:
        async with Persistence.open('runtime-policy-collision', storage=PersistenceStorage.in_memory()) as persistence_344, Runtime.open(persistence=persistence_344, features=model_features(_Models(_UsageFunctionModel(_immediate)), _policies(group, _GlobalPolicy(backend)))):
            raise AssertionError("conflicting runtime capability was accepted")
    assert raised.value.code is ErrorCode.CAPABILITY_CONFLICT
    assert backend.requests == []


async def test_anonymous_native_capabilities_remain_distinct_in_combined_runtime_capability() -> None:
    first_backend, second_backend = _Backend(), _Backend()
    first, second = _GlobalPolicy(first_backend), _GlobalPolicy(second_backend)
    first.id = second.id = None
    async with Persistence.open('runtime_capability-anonymous', storage=PersistenceStorage.in_memory()) as persistence_357, Runtime.open(persistence=persistence_357, features=model_features(_Models(_UsageFunctionModel(_immediate)), _policies(_group(), CombinedCapability([first, second])))) as runtime:
        execution = await runtime.agents.get().start("go")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert first_backend.entered == second_backend.entered == [execution.execution_id]


async def test_reopened_session_and_historical_retry_use_current_host_runtime_capability(tmp_path: Path) -> None:
    first_backend, next_backend = _Backend(), _Backend()
    models = _Models(_UsageFunctionModel(_immediate))
    first_features = _policies(_group(), _GlobalPolicy(first_backend), ExecutionCapability(retain_history=True))
    async with Persistence.open('runtime_capability-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_370, Runtime.open(persistence=persistence_370, features=model_features(models, first_features)) as runtime:
        first = await runtime.agents.get().start("standalone")
        assert (await first.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        session_first = await runtime.agents.get().session("conversation").start("first")
        assert (await session_first.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
    current = _GlobalPolicy(next_backend)
    current.id = "host.replacement"
    async with Persistence.open('runtime_capability-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_380, Runtime.open(persistence=persistence_380, features=model_features(models, _policies(_group(), current, ExecutionCapability(retain_history=True)))) as runtime:
        historical = await runtime.executions.get(first.execution_id)
        retry = await historical.retry("retry")
        assert (await retry.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        second = await runtime.agents.get().session("conversation").start("second")
        assert (await second.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert first_backend.entered == [first.execution_id, session_first.execution_id]
        assert next_backend.entered == [retry.execution_id, second.execution_id]
        assert next_backend.entered == next_backend.released


async def test_runtime_capability_acquisition_failure_does_not_call_provider() -> None:
    provider_calls = 0

    async def model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal provider_calls
        provider_calls += 1
        return ModelResponse(parts=[TextPart("unexpected")])

    class Unavailable(AbstractCapability[AgentContext[None]]):
        async def wrap_model_request(
            self, ctx: RunContext[AgentContext[None]], *, request_context: ModelRequestContext,
            handler: WrapModelRequestHandler,
        ) -> ModelResponse:
            raise RuntimeError("host backend unavailable")

    async with Persistence.open('runtime_capability-unavailable', storage=PersistenceStorage.in_memory()) as persistence_409, Runtime.open(persistence=persistence_409, features=model_features(_Models(_UsageFunctionModel(model)), _policies(_group(), Unavailable()))) as runtime:
        execution = await runtime.agents.get().start("go")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.FAILED
        assert provider_calls == 0


async def test_nested_native_id_conflict_cannot_replace_selected_capability() -> None:
    backend = _Backend()
    group = _group()
    group.capability(_GlobalPolicy(_Backend()), scope="agent")
    group.agent("selected", allow_tools=(), allow_skills=(), allow_subagents=(),
                allow_capabilities=("host.runtime_capability",))
    policies = _policies(group, CombinedCapability([_GlobalPolicy(backend)]))
    async with Persistence.open('runtime-policy-nested-collision', storage=PersistenceStorage.in_memory()) as persistence_425, Runtime.open(persistence=persistence_425, features=model_features(_Models(_UsageFunctionModel(_immediate)), policies)) as runtime:
        execution = await runtime.agents.get("selected").start("go")
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.CAPABILITY_CONFLICT.value
        assert backend.requests == []


async def test_runtime_native_ids_are_unique_across_named_combined_registrations() -> None:
    policies = _group()
    for name in ("first", "second"):
        policies.capability(
            CombinedCapability([_GlobalPolicy(_Backend())]), id=name, scope="runtime",
        )
    with pytest.raises(AIError) as raised:
        async with Persistence.open('runtime-policy-native-duplicates', storage=PersistenceStorage.in_memory()) as persistence_443, Runtime.open(persistence=persistence_443, features=model_features(_Models(_UsageFunctionModel(_immediate)), policies)):
            raise AssertionError("duplicate native runtime IDs were accepted")
    assert raised.value.code is ErrorCode.CAPABILITY_CONFLICT


@pytest.mark.parametrize("native_id", ("", " ", 7))
async def test_runtime_combined_capability_rejects_malformed_native_leaf_ids(native_id: object) -> None:
    policy = _GlobalPolicy(_Backend())
    policy.id = native_id
    with pytest.raises(AIError) as raised:
        _policies(_group(), CombinedCapability([policy]))
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
