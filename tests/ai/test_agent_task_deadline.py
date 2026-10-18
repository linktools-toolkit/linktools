#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent node deadlines cancel execution trees and survive durable recovery."""

import asyncio
import multiprocessing
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext, RuntimeFeatures
from linktools.ai.core import ExecutionStatus, RunBudget, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import AgentTaskInput, CheckpointCapability, Runtime, RuntimeStorage
from linktools.ai.runtime._agent_task import RuntimeAgentTaskRunner
from linktools.ai.task import TaskGraph, TaskNode

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

    wait_execution = RuntimeAgentTaskRunner._wait_execution

    async def wait_after_work_started(self, *args, **kwargs):
        nonlocal clock_resumed_at
        await entered.wait()
        clock_resumed_at = loop.time()
        return await wait_execution(self, *args, **kwargs)

    # Exclude setup from the deadline, then let real elapsed time drive the
    # original wait and cancellation after the intended work has started.
    for module in ("_agent_task", "_execution", "_local"):
        monkeypatch.setattr(f"linktools.ai.runtime.{module}.datetime", Clock)
    monkeypatch.setattr(RuntimeAgentTaskRunner, "_wait_execution", wait_after_work_started)

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
    async with Runtime.open(
        "deadline", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), runtime_features=group,
    ) as runtime:
        task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(
            TaskGraph("deadline", (TaskNode("agent", task=task, input=AgentTaskInput("work"), timeout_seconds=0.3),)),
            idempotency_key="start",
        )
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

    wait_execution = RuntimeAgentTaskRunner._wait_execution

    async def wait_after_model_started(self, *args, **kwargs):
        nonlocal current_time
        await entered.wait()
        if earlier != "none":
            current_time += timedelta(seconds=1)
        return await wait_execution(self, *args, **kwargs)

    # First establish a real shared-budget reservation, then expire only the
    # tighter limit. Startup speed must not decide whether a request exists.
    for module in ("_agent_task", "_execution", "_local", "state._budget"):
        monkeypatch.setattr(f"linktools.ai.runtime.{module}.datetime", Clock)
    monkeypatch.setattr(RuntimeAgentTaskRunner, "_wait_execution", wait_after_model_started)

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
    async with Runtime.open(
        "deadline-budget", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), runtime_features=group,
    ) as runtime:
        task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
        budget = RunBudget(model_requests=4, deadline_at=(
            None if earlier == "none" else current_time + timedelta(seconds=3 if earlier == "node" else 0.3)
        ))
        run = await runtime.tasks.bind(task).start(
            TaskGraph("deadline-budget", (TaskNode(
                "agent", task=task, input=AgentTaskInput("work"),
                timeout_seconds=None if earlier in {"none", "graph_only"} else 0.3 if earlier == "node" else 3,
            ),)), idempotency_key="start", budget=budget,
        )
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
    interrupt_child: bool = True,
) -> None:
    observer_entered = asyncio.Event()

    async def pause_before_deadline_observation(*args, **kwargs):
        del args, kwargs
        observer_entered.set()
        await asyncio.Event().wait()

    async def start() -> None:
        entered = asyncio.Event()

        async def model(messages, info):
            del messages
            if subagent and interrupt_child and any(tool.name == "delegate_task" for tool in info.function_tools):
                return ModelResponse(parts=[ToolCallPart("delegate_task", {"subagent_id": "child", "task": "work"})])
            entered.set()
            await asyncio.sleep(30)
            return ModelResponse(parts=[TextPart("late")])

        group = _recovery_group(subagent)
        async with Runtime.open(
            "deadline-recovery", models=_Models(_UsageFunctionModel(model)),
            storage=(RuntimeStorage.sqlite(Path(path)) if backend == "sqlite" else RuntimeStorage.filesystem(Path(path))), runtime_features=group,
        ) as runtime:
            task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
            run = await runtime.tasks.bind(task).start(
                TaskGraph("deadline-recovery", (TaskNode("agent", task=task, input=AgentTaskInput("work"), timeout_seconds=timeout),)),
                idempotency_key="start",
            )
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
        patch.setattr(RuntimeAgentTaskRunner, "_wait_execution", pause_before_deadline_observation)
        asyncio.run(start())


@pytest.mark.asyncio
@pytest.mark.parametrize("backend,subagent,change", (
    ("sqlite", False, "none"), ("filesystem", False, "none"),
    ("sqlite", True, "none"), ("filesystem", True, "none"),
    ("sqlite", True, "metadata"), ("sqlite", True, "prompt"),
    ("sqlite", True, "tool"), ("sqlite", True, "revision"),
))
async def test_agent_node_recovery_keeps_first_start_deadline(
    tmp_path: Path, backend: str, subagent: bool, change: str,
) -> None:
    path = tmp_path / "state.db"
    context = multiprocessing.get_context("spawn")
    read, write = context.Pipe(duplex=False)
    process = context.Process(target=_start_interrupted_graph, args=(str(path), write, backend, 0.5, subagent))
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
    snapshot = RuntimeStorage.sqlite(path) if backend == "sqlite" else RuntimeStorage.filesystem(path)
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
    runtime_context = Runtime.open(
        "deadline-recovery", models=_Models(_UsageFunctionModel(model)),
        storage=(RuntimeStorage.sqlite(path) if backend == "sqlite" else RuntimeStorage.filesystem(path)), runtime_features=group,
    )
    runtime = await runtime_context.__aenter__()
    try:
        await asyncio.sleep(0.15)
        assert calls == 0
        current = await runtime.executions.inspect(execution_id, principal=runtime.default_principal)
        assert current.started_at == started_at
        task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).get("deadline-recovery")
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
                await runtime_context.__aexit__(None, None, None)
            assert caught.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        else:
            await runtime_context.__aexit__(None, None, None)


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
    storage = RuntimeStorage.sqlite(path) if subagent else RuntimeStorage.filesystem(path)
    async with Runtime.open(
        "deadline-recovery", models=_Models(_UsageFunctionModel(model)),
        storage=storage, runtime_features=group,
    ) as runtime:
        original_binding = (await storage.execution.executions.get(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )).binding
        task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).get("deadline-recovery")
        if subagent:
            await run.recover(idempotency_key="recover")
        execution = await runtime.executions.get(execution_id)
        assert (await execution.wait(timeout_seconds=3)).result.status is ExecutionStatus.SUCCEEDED
        assert calls == (3 if subagent else 1)
        current = await runtime.executions.inspect(execution_id, principal=runtime.default_principal)
        assert current.started_at == started_at
        if subagent:
            assert (await run.wait(timeout_seconds=3)).result.status is TaskStatus.SUCCEEDED
        assert (await storage.execution.executions.get(
            execution_id, tenant_id=runtime.default_principal.tenant_id,
        )).binding == original_binding
