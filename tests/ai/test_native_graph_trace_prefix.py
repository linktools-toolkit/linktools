#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A session-bound Graph archives only its own Trace message occurrences."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai_harness.step_persistence import StepPersistence

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskStatus
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _HostPolicy(TenantAuthorizationPolicy):
    pass


@pytest.mark.asyncio
async def test_graph_trace_settles_without_copying_accepted_session_prefix(tmp_path: Path) -> None:
    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    policy = _HostPolicy()
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    async with Persistence.open("native-graph-trace", storage=storage) as state:
        step = await state.create_step_store()
        try:
            features = RuntimeFeatures("native-graph-trace")
            features.capability(ExecutionCapability(retain_history=True), scope="runtime")
            features.capability(StepPersistence(store=step, capture_frontier=True), scope="runtime")
            features.capability(TraceCapability(), scope="runtime")
            features.agent("default", model="default", allow_tools=())
            async with (
                Runtime.open(persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
                             authorization=policy) as runtime,
                TaskEngine.open(persistence=state, context=runtime.context, authorization=policy) as engine,
            ):
                principal = runtime.default_principal
                await runtime.sessions.create_turn_session("chat", principal=principal)
                first = await runtime.sessions.start("chat", "accepted prefix", agent="default", principal=principal,
                                                     idempotency_key="first")
                assert (await first.wait_final()).status is ExecutionStatus.SUCCEEDED
                task = Task.from_agent("trace.agent", runtime.agents.get())
                run = await engine.with_definitions(task).start(TaskGraph("trace-graph", (
                    TaskNode("agent", task=task, input=AgentTaskInput("graph-only input", session_id="chat")),
                )), principal=principal, idempotency_key="graph")
                assert (await run.wait()).result.wait_status is TaskStatus.SUCCEEDED
                execution = await run.execution("agent")
                page = await runtime.history.turn_session("chat", principal=principal)
                assert [item["turn_id"] for item in page["timeline"]["items"]] == [
                    first.key.turn_id, execution.execution_id,
                ]
                interaction, = (await execution.model_interactions(include_content=True)).items
                assert "accepted prefix" in str(interaction.request)
                assert "graph-only input" in str(interaction.request)
        finally:
            await step.close()
