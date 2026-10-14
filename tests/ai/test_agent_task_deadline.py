#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent node deadlines cancel execution trees and survive durable recovery."""

import asyncio
import multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionStatus, RunBudget, TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import AgentTaskInput, Runtime, RuntimeStorage
from linktools.ai.task import TaskGraph, TaskNode

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("model", "tool", "unknown_tool"))
async def test_agent_node_deadline_cancels_work_and_preserves_unknown_effects(phase: str) -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    finished = asyncio.Event()

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
        del messages, info
        if phase == "model":
            return ModelResponse(parts=[TextPart(await work())])
        return ModelResponse(parts=[ToolCallPart("slow", {})])

    group = CapabilityGroup("deadline")
    group.tool(slow, effect_policy="non_replay_safe" if phase == "unknown_tool" else "none")
    group.agent("default", model="default", allow_tools=() if phase == "model" else ("slow",),
                allow_skills=(), allow_subagents=())
    async with Runtime.open(
        "deadline", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), capabilities=(group,),
    ) as runtime:
        task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).start(
            TaskGraph("deadline", (TaskNode("agent", task=task, input=AgentTaskInput("work"), timeout_seconds=0.3),)),
            idempotency_key="start",
        )
        await asyncio.wait_for(entered.wait(), 2)
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
async def test_agent_node_deadline_uses_tighter_limit_without_replacing_shared_budget(earlier: str) -> None:
    cancelled = asyncio.Event()

    async def model(messages, info):
        del messages, info
        if earlier != "none":
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return ModelResponse(parts=[TextPart("done")])

    group = CapabilityGroup("deadline-budget")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open(
        "deadline-budget", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), capabilities=(group,),
    ) as runtime:
        task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
        budget = RunBudget(model_requests=4, deadline_at=(
            None if earlier == "none" else datetime.now(timezone.utc) + timedelta(seconds=3 if earlier == "node" else 0.3)
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


def _start_interrupted_graph(path: str, ready: object) -> None:
    async def start() -> None:
        entered = asyncio.Event()

        async def model(messages, info):
            del messages, info
            entered.set()
            await asyncio.sleep(30)
            return ModelResponse(parts=[TextPart("late")])

        group = CapabilityGroup("deadline-recovery")
        group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
        async with Runtime.open(
            "deadline-recovery", models=_Models(_UsageFunctionModel(model)),
            storage=RuntimeStorage.sqlite(Path(path)), capabilities=(group,),
        ) as runtime:
            task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
            run = await runtime.tasks.bind(task).start(
                TaskGraph("deadline-recovery", (TaskNode("agent", task=task, input=AgentTaskInput("work"), timeout_seconds=0.5),)),
                idempotency_key="start",
            )
            await asyncio.wait_for(entered.wait(), 2)
            execution = await run.execution("agent")
            view = await runtime.executions.inspect(execution.execution_id, principal=runtime.default_principal)
            ready.send((execution.execution_id, view.started_at))
            await asyncio.Event().wait()

    asyncio.run(start())


@pytest.mark.asyncio
async def test_agent_node_recovery_keeps_first_start_deadline(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    context = multiprocessing.get_context("spawn")
    read, write = context.Pipe(duplex=False)
    process = context.Process(target=_start_interrupted_graph, args=(str(path), write))
    process.start()
    try:
        assert await asyncio.to_thread(read.poll, 10)
        execution_id, started_at = read.recv()
        assert started_at is not None
    finally:
        process.terminate()
        await asyncio.to_thread(process.join, 5)
        read.close()
        write.close()
    await asyncio.sleep(max(0, (started_at + timedelta(seconds=0.6) - datetime.now(timezone.utc)).total_seconds()))
    calls = 0

    async def model(messages, info):
        nonlocal calls
        del messages, info
        calls += 1
        await asyncio.sleep(30)
        return ModelResponse(parts=[TextPart("late")])

    group = CapabilityGroup("deadline-recovery")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open(
        "deadline-recovery", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.sqlite(path), capabilities=(group,),
    ) as runtime:
        task = runtime.tasks.from_agent("deadline.agent", runtime.agents.get())
        run = await runtime.tasks.bind(task).get("deadline-recovery")
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
