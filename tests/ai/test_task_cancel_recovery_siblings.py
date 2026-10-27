#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Whole-graph cancellation still reaches live siblings of unknown effects."""
from ._runtime_test_helpers import model_features

import asyncio
from pathlib import Path

import pytest

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, TaskStatus, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import Task, TaskGraph, TaskNode
from linktools.ai.task.engine import TaskEngine

from .test_task_mixed_node_reliability import _TaskTestModels


@pytest.mark.asyncio
async def test_graph_cancel_quiesces_active_agent_sibling_when_an_effect_is_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINKTOOLS_PATH", str(tmp_path / "linktools"))
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled = asyncio.Event()
    effects: list[str] = []

    async def slow(context: AgentContext[None]) -> str:
        del context
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        effects.append("late-write")
        return "done"

    async def unknown(context: AgentContext[None]) -> str:
        del context
        await started.wait()
        effects.append("unknown-write")
        raise RuntimeError("external outcome is uncertain")

    group = RuntimeFeatures[None]("application")
    group.tool(slow, effect_policy="non_replay_safe")
    group.tool(unknown, effect_policy="non_replay_safe")
    for name, tools in (
        ("default", ()),
        ("slow", ("slow",)),
        ("unknown", ("unknown",)),
    ):
        group.agent(name, allow_tools=tools, allow_subagents=(), allow_skills=())
    storage = PersistenceStorage.in_memory()
    try:
        async with Persistence.open('cancel-unknown-sibling', storage=storage) as persistence_60, Runtime.attach(persistence=persistence_60, features=model_features(_TaskTestModels(), group)) as runtime, TaskEngine.open(persistence=persistence_60) as engine:
            unknown_task = Task.from_agent(
                "review.unknown", runtime.agents.get("unknown")
            )
            slow_task = Task.from_agent(
                "review.slow", runtime.agents.get("slow")
            )
            graph = TaskGraph(
                "cancel-unknown-sibling",
                (
                    TaskNode(
                        "unknown", task=unknown_task, input=AgentTaskInput("write")
                    ),
                    TaskNode("slow", task=slow_task, input=AgentTaskInput("write")),
                ),
            )
            run = await engine.with_definitions(unknown_task, slow_task).start(
                graph,
                principal=runtime.default_principal, idempotency_key="start",
            )
            await asyncio.wait_for(started.wait(), timeout=10)
            initial = (await run.wait(timeout_seconds=10)).result
            assert initial.status is TaskStatus.RECOVERY_REQUIRED
            slow_execution = await run.execution("slow")
            before = await runtime.executions.inspect(
                slow_execution.execution_id,
                principal=runtime.default_principal,
            )
            assert before.status is ExecutionStatus.STARTED
            assert not cancelled.is_set()

            result = await asyncio.wait_for(
                run.cancel(idempotency_key="cancel"), timeout=10
            )
            assert result.status is TaskStatus.RECOVERY_REQUIRED
            assert cancelled.is_set()
            release.set()
            assert effects == ["unknown-write"]

            for node_id in ("unknown", "slow"):
                execution = await run.execution(node_id)
                with pytest.raises(AIError) as raised:
                    await execution.wait(timeout_seconds=10)
                assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
                operations = await runtime._state.recovery.tools.list_by_execution(
                    execution.execution_id,
                    tenant_id=runtime.default_principal.tenant_id,
                )
                assert len(operations) == 1
                assert operations[0].status is ToolOperationStatus.EFFECT_UNKNOWN
            repeated = await run.cancel(idempotency_key="cancel")
            assert repeated.status is TaskStatus.RECOVERY_REQUIRED
            assert effects == ["unknown-write"]
    finally:
        release.set()
