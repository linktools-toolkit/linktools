#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Agent node deadlines cancel execution trees and survive durable recovery."""
from ._runtime_test_helpers import model_features

import asyncio
import multiprocessing
from contextlib import AsyncExitStack
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, RunBudget, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import CheckpointCapability
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import Persistence
from linktools.ai.task_engine._agent import AgentTaskRunner
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task_engine import TaskEngine

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("model", "tool", "unknown_tool", "subagent"))
async def test_agent_node_deadline_cancels_work_and_preserves_unknown_effects(
    phase: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    finished = asyncio.Event()
    initial_time = datetime.now(timezone.utc)
    loop = asyncio.get_running_loop()
    clock_resumed_at: float | None = None

    class Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            elapsed = 0 if clock_resumed_at is None else loop.time() - clock_resumed_at
            current_time = initial_time + timedelta(seconds=elapsed)
            return current_time.replace(tzinfo=None) if tz is None else current_time.astimezone(tz)

    wait_execution = AgentTaskRunner._wait_execution

    async def wait_after_work_started(self, *args, **kwargs):
        nonlocal clock_resumed_at
        await entered.wait()
        clock_resumed_at = loop.time()
        return await wait_execution(self, *args, **kwargs)

    # Exclude setup from the deadline, then let real elapsed time drive the
    # original wait and cancellation after the intended work has started.
    for module in ("task_engine._agent", "execution._service", "runtime._execution", "runtime._local"):
        monkeypatch.setattr(f"linktools.ai.{module}.datetime", Clock)
    monkeypatch.setattr(AgentTaskRunner, "_wait_execution", wait_after_work_started)

    async def work() -> str:
        entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        finished.set()
        return "late result"

    async def slow(context: AgentContext[None]) -> str:
        del context
        return await work()

    async def model(messages, info):
        del messages
        if phase == "subagent":
            if any(tool.name == "delegate_task" for tool in info.function_tools):
                return ModelResponse(parts=[ToolCallPart("delegate_task", {"subagent_id": "child", "task": "work"})])
            return ModelResponse(parts=[TextPart(await work())])
        if phase == "model":
            return ModelResponse(parts=[TextPart(await work())])
        return ModelResponse(parts=[ToolCallPart("slow", {})])

    group = RuntimeFeatures("deadline")
    group.tool(slow, effect_policy="non_replay_safe" if phase == "unknown_tool" else "none")
    group.agent("default", model="default", allow_tools=() if phase in {"model", "subagent"} else ("slow",),
                allow_skills=(), allow_subagents=("child",) if phase == "subagent" else ())
    group.agent("child", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('deadline', storage=PersistenceStorage.in_memory()) as persistence_89, Runtime.open(persistence=persistence_89, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime, TaskEngine.open(persistence=persistence_89) as engine:
        task = Task.from_agent("deadline.agent", runtime.agents.get())
        run = await engine.with_definitions(task).start(
            TaskGraph("deadline", (TaskNode("agent", task=task, input=AgentTaskInput("work"), timeout_seconds=0.3),)),
            idempotency_key="start",
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=3)).result
        node, = (await run.state()).node_states
        execution = await runtime.executions.inspect(node.execution_id, principal=runtime.default_principal)
        assert cancelled.is_set()
        assert not finished.is_set()
        if phase == "unknown_tool":
            assert result.status is TaskStatus.RECOVERY_REQUIRED
            assert node.status is TaskStatus.RECOVERY_REQUIRED
            assert execution.status is ExecutionStatus.RECOVERY_REQUIRED
        else:
            assert result.status is TaskStatus.FAILED
            assert node.status is TaskStatus.FAILED
            assert node.error_code == ErrorCode.EXECUTION_WAIT_TIMEOUT.value
            assert node.error_origin == "node"
            assert node.safe_error_details["reason"] == "node_deadline_exceeded"
            assert execution.status is ExecutionStatus.CANCELLED
            assert execution.started_at is not None
            assert execution.terminal_at is not None
            assert datetime.fromisoformat(node.safe_error_details["deadline_at"]) == execution.started_at + timedelta(seconds=0.3)


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier", ("node", "graph", "graph_only", "none"))
async def test_agent_node_deadline_uses_tighter_limit_without_replacing_shared_budget(
    earlier: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    current_time = datetime.now(timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return current_time.replace(tzinfo=None) if tz is None else current_time.astimezone(tz)

    wait_execution = AgentTaskRunner._wait_execution

    async def wait_after_model_started(self, *args, **kwargs):
        nonlocal current_time
        await entered.wait()
        if earlier != "none":
            current_time += timedelta(seconds=1)
        return await wait_execution(self, *args, **kwargs)

    # First establish a real shared-budget reservation, then expire only the
    # tighter limit. Startup speed must not decide whether a request exists.
    for module in ("task_engine._agent", "execution._service", "runtime._execution", "runtime._local", "persistence._budget"):
        monkeypatch.setattr(f"linktools.ai.{module}.datetime", Clock)
    monkeypatch.setattr(AgentTaskRunner, "_wait_execution", wait_after_model_started)

    async def model(messages, info):
        del messages, info
        entered.set()
        if earlier != "none":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return ModelResponse(parts=[TextPart("done")])

    group = RuntimeFeatures("deadline-budget")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('deadline-budget', storage=PersistenceStorage.in_memory()) as persistence_161, Runtime.open(persistence=persistence_161, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime, TaskEngine.open(persistence=persistence_161) as engine:
        task = Task.from_agent("deadline.agent", runtime.agents.get())
        budget = RunBudget(model_requests=4, deadline_at=(
            None if earlier == "none" else current_time + timedelta(seconds=3 if earlier == "node" else 0.3)
        ))
        run = await engine.with_definitions(task).start(
            TaskGraph("deadline-budget", (TaskNode(
                "agent", task=task, input=AgentTaskInput("work"),
                timeout_seconds=None if earlier in {"none", "graph_only"} else 0.3 if earlier == "node" else 3,
            ),)), idempotency_key="start", budget=budget,
         principal=runtime.default_principal)
        result = (await run.wait(timeout_seconds=4)).result
        node, = (await run.state()).node_states
        usage = await run.budget_usage()
        assert usage.limits == budget
        assert usage.scope_id == "graph:deadline-budget"
        assert usage.model_requests == 1
        assert await (await run.execution("agent")).budget_usage() == usage
        if earlier == "none":
            assert result.status is TaskStatus.SUCCEEDED
        else:
            assert cancelled.is_set()
            assert result.status is TaskStatus.FAILED
            assert node.error_code == (ErrorCode.EXECUTION_WAIT_TIMEOUT if earlier == "node" else ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED).value


def _recovery_group(subagent: bool, change: str = "none") -> RuntimeFeatures:
    async def marker(value: str) -> str:
        return value

    group = RuntimeFeatures("deadline-recovery")
    group.capability(CheckpointCapability(), scope="runtime")
    group.tool(marker, effect_policy="non_replay_safe" if change == "tool" else "none")
    group.agent("default", model="default", allow_tools=(), allow_skills=(),
                allow_subagents=("child",) if subagent else ())
    group.agent("child", model="default", allow_tools=("marker",), allow_skills=(), allow_subagents=(),
                system_prompt="changed" if change == "prompt" else "",
                metadata={"label": "new"} if change == "metadata" else {},
                description="new description" if change == "metadata" else None,
                revision=2 if change == "revision" else 1)
    return group


def _start_interrupted_graph(
    path: str, ready: object, backend: str, timeout: float | None = 0.5, subagent: bool = False,
    interrupt_child: bool = True, missing_capture: bool = False,
) -> None:
    observer_entered = asyncio.Event()
    model_entered = asyncio.Event()

    async def pause_before_deadline_observation(*args, **kwargs):
        del args, kwargs
        observer_entered.set()
        await asyncio.Event().wait()

    async def start() -> None:
        entered = model_entered

        async def model(messages, info):
            del messages
            if subagent and interrupt_child and any(tool.name == "delegate_task" for tool in info.function_tools):
                return ModelResponse(parts=[ToolCallPart("delegate_task", {"subagent_id": "child", "task": "work"})])
            entered.set()
            await asyncio.sleep(30)
            return ModelResponse(parts=[TextPart("late")])

        group = _recovery_group(subagent)
        async with Persistence.open('deadline-recovery', storage=PersistenceStorage.sqlite(Path(path)) if backend == 'sqlite' else PersistenceStorage.filesystem(Path(path))) as persistence_230, Runtime.open(persistence=persistence_230, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime, TaskEngine.open(persistence=persistence_230) as engine:
            task = Task.from_agent("deadline.agent", runtime.agents.get())
            run = await engine.with_definitions(task).start(
                TaskGraph("deadline-recovery", (TaskNode("agent", task=task, input=AgentTaskInput("work"), timeout_seconds=timeout),)),
                idempotency_key="start",
             principal=runtime.default_principal)
            # The parent bounds the complete startup handshake, including
            # filesystem persistence and child execution admission.
            await entered.wait()
            await observer_entered.wait()
            execution = await run.execution("agent")
            view = await runtime.executions.inspect(execution.execution_id, principal=runtime.default_principal)
            assert view.status is ExecutionStatus.STARTED
            assert view.terminal_at is None
            ready.send((execution.execution_id, view.started_at, runtime.default_principal.tenant_id))
            await asyncio.Event().wait()

    # Crash at the established/captured execution boundary. Suspending only
    # the writer's observer prevents machine speed from letting its 0.5s timer
    # settle cancellation before the parent can kill it. Model/child work and
    # all recovery-side deadline handling remain real.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(AgentTaskRunner, "_wait_execution", pause_before_deadline_observation)
        if missing_capture:
            from linktools.ai.inputs import InputCaptures

            async def pause_before_capture(self, execution_id, invocation):
                await model_entered.wait()
                view = await self._execution.inspect(execution_id, principal=invocation.principal)
                ready.send((execution_id, view.started_at, invocation.principal.tenant_id))
                await asyncio.Event().wait()

            patch.setattr(InputCaptures, "record_invocation", pause_before_capture)
        asyncio.run(start())


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,subagent,change", (
    ("sqlite", False, "none"), ("filesystem", False, "none"),
    ("sqlite", True, "none"), ("filesystem", True, "none"),
    ("sqlite", True, "metadata"), ("sqlite", True, "prompt"),
    ("sqlite", True, "tool"), ("sqlite", True, "revision"),
    ("sqlite", False, "missing_capture"), ("filesystem", False, "missing_capture"),
))
async def test_agent_node_recovery_keeps_first_start_deadline(
    tmp_path: Path, backend: str, subagent: bool, change: str,
) -> None:
    path = tmp_path / "state.db"
    context = multiprocessing.get_context("spawn")
    read, write = context.Pipe(duplex=False)
    process = context.Process(target=_start_interrupted_graph, args=(str(path), write, backend, None if change == "missing_capture" else 0.5, subagent, True, change == "missing_capture"))
    process.start()
    try:
        assert await asyncio.to_thread(read.poll, 30), "Interrupted execution did not reach its ready boundary"
        execution_id, started_at, tenant_id = read.recv()
        assert started_at is not None
    finally:
        process.terminate()
        await asyncio.to_thread(process.join, 5)
        assert not process.is_alive()
        assert process.exitcode is not None
        read.close()
        write.close()
    # Read without a Runtime: startup reconciliation must not run before the
    # crash premise (including the real in-flight child) has been verified.
    snapshot = Persistence(PersistenceStorage.sqlite(path) if backend == "sqlite" else PersistenceStorage.filesystem(path))
    await snapshot.initialize(namespace="deadline-recovery", tenant_id=tenant_id, read_only=True)
    try:
        original = await snapshot.execution.executions.get(execution_id, tenant_id=tenant_id)
        assert original is not None
        assert original.status is ExecutionStatus.STARTED
        assert original.started_at == started_at
        assert original.result is None
        children = await snapshot.execution.executions.list_children(execution_id, tenant_id=tenant_id)
        assert len(children) == int(subagent)
        assert all(child.status is ExecutionStatus.STARTED and child.result is None for child in children)
    finally:
        await snapshot.close()
    await asyncio.sleep(max(0, (started_at + timedelta(seconds=0.6) - datetime.now(timezone.utc)).total_seconds()))
    calls = 0

    async def model(messages, info):
        nonlocal calls
        del messages, info
        calls += 1
        await asyncio.sleep(30)
        return ModelResponse(parts=[TextPart("late")])

    group = _recovery_group(subagent, change)
    owners = AsyncExitStack()
    persistence = await owners.enter_async_context(Persistence.open(
        "deadline-recovery",
        storage=PersistenceStorage.sqlite(path) if backend == "sqlite" else PersistenceStorage.filesystem(path),
    ))
    runtime = await owners.enter_async_context(Runtime.open(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), group),
    ))
    engine = await owners.enter_async_context(TaskEngine.open(persistence=persistence))
    try:
        await asyncio.sleep(0.15)
        assert calls == 0
        current = await runtime.executions.inspect(execution_id, principal=runtime.default_principal)
        assert current.started_at == started_at
        if change == "missing_capture":
            return
        task = Task.from_agent("deadline.agent", runtime.agents.get())
        run = await engine.with_definitions(task).get("deadline-recovery", principal=runtime.default_principal)
        if change in {"prompt", "tool", "revision"}:
            with pytest.raises(AIError) as caught:
                await run.recover(idempotency_key="recover")
            assert caught.value.code in {ErrorCode.STORAGE_INTEGRITY_ERROR, ErrorCode.AGENT_BINDING_UNAVAILABLE}
            assert calls == 0
            return
        await run.recover(idempotency_key="recover")
        result = (await run.wait(timeout_seconds=3)).result
        current = await runtime.executions.inspect(execution_id, principal=runtime.default_principal)
        assert current.started_at == started_at
        assert calls == 0
        # The old process stopped without a durable outcome. Do not restart it
        # after the deadline or manufacture proof that its effects were settled.
        assert result.status is TaskStatus.RECOVERY_REQUIRED
        node, = (await run.state()).node_states
        assert node.status is TaskStatus.RECOVERY_REQUIRED
        assert node.error_code == ErrorCode.STORAGE_RECOVERY_REQUIRED.value
    finally:
        if subagent and change in {"none", "metadata"}:
            # The interrupted delegated effect remains unknown on close too.
            with pytest.raises(AIError) as caught:
                await owners.aclose()
            assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        else:
            await owners.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("subagent", (False, True))
async def test_agent_recovery_relaunches_with_original_binding(tmp_path: Path, subagent: bool) -> None:
    path = tmp_path / "state"
    context = multiprocessing.get_context("spawn")
    read, write = context.Pipe(duplex=False)
    process = context.Process(target=_start_interrupted_graph, args=(str(path), write, "sqlite" if subagent else "filesystem", 120 if subagent else None, subagent, False))
    process.start()
    try:
        assert await asyncio.to_thread(read.poll, 30), "Interrupted execution did not reach its ready boundary"
        execution_id, started_at, _tenant_id = read.recv()
    finally:
        process.terminate()
        await asyncio.to_thread(process.join, 5)
        assert not process.is_alive()
        assert process.exitcode is not None
        read.close()
        write.close()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        del messages
        calls += 1
        if subagent and calls == 1 and any(tool.name == "delegate_task" for tool in info.function_tools):
            return ModelResponse(parts=[ToolCallPart("delegate_task", {"subagent_id": "child", "task": "work"})])
        return ModelResponse(parts=[TextPart("recovered")])

    group = _recovery_group(subagent)
    storage = PersistenceStorage.sqlite(path) if subagent else PersistenceStorage.filesystem(path)
    async with Persistence.open('deadline-recovery', storage=storage) as persistence_380, Runtime.open(persistence=persistence_380, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime, TaskEngine.open(persistence=persistence_380) as engine:
        original_binding = (await runtime._state.execution.executions.get(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )).binding
        task = Task.from_agent("deadline.agent", runtime.agents.get())
        run = await engine.with_definitions(task).get("deadline-recovery", principal=runtime.default_principal)
        if subagent:
            await run.recover(idempotency_key="recover")
        execution = await runtime.executions.get(execution_id)
        # Durable parent/child recovery needs an I/O observation guard, separate
        # from the node deadline whose original start time must be preserved.
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert calls == (3 if subagent else 1)
        current = await runtime.executions.inspect(execution_id, principal=runtime.default_principal)
        assert current.started_at == started_at
        if subagent:
            assert (await run.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert (await runtime._state.execution.executions.get(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )).binding == original_binding
