#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-owned Agents retain durable observation without Runtime opt-ins."""

import asyncio

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import TaskStatus
from linktools.ai.capability import TraceCapability, CheckpointCapability
from linktools.ai.runtime import AgentTaskInput, Runtime, RuntimeStorage
from linktools.ai.task import TaskGraph, TaskNode

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed
from .test_optional_runtime_persistence import _start_delegated_execution


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", (False, True), ids=("agent", "delegated-agent"))
async def test_task_agent_trace_survives_terminal_release_and_reopen(tmp_path, delegated: bool) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    requests: list[str] = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(str(messages))
        if len(requests) == 1:
            entered.set()
            await release.wait()
            if delegated:
                assert "delegate_task" in {tool.name for tool in info.function_tools}
                return ModelResponse(parts=[ToolCallPart(
                    "delegate_task", {"subagent_id": "child", "task": "child answer"},
                    tool_call_id="delegate-child",
                )])
        if delegated and any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("root answer")])
        return ModelResponse(parts=[TextPart("answer")])

    group: RuntimeFeatures[None] = RuntimeFeatures("application")
    group.agent("default", model="default", allow_tools=(), allow_subagents=("child",) if delegated else ())
    group.agent("child", model="default", allow_tools=(), allow_subagents=())
    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "task-agent.db"
    identities: list[str] = []
    storage = RuntimeStorage.sqlite(database)
    async with Runtime.open(
        "task-agent-persistence", models=models, features=group, storage=storage,
    ) as runtime:
        task = runtime.tasks.from_agent("answer", runtime.agents.get())
        graph_definition = TaskGraph("durable-agent", (TaskNode("answer", task=task, input=AgentTaskInput("question")),))
        graph = await runtime.tasks.bind(task).start(graph_definition, idempotency_key="once")
        observed = []

        async def observe(event) -> None:
            observed.append(event)

        waiting = asyncio.create_task(graph.wait(on_event=observe, timeout_seconds=10))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            state = await _wait_for_committed(graph.state, lambda value: value.node_states[0].execution_id is not None)
            execution_id = state.node_states[0].execution_id
            assert execution_id is not None
            identities.append(execution_id)
            record = await storage.execution.executions.get(execution_id, tenant_id=runtime.tenant_id)
            assert record is not None and record.requires_task_invocation_capture
            assert record.restart_checkpoints and record.durable_trace
            boundary = await _wait_for_committed(
                lambda: runtime.history.capture_model_interaction_cutoffs(execution_id, principal=runtime.default_principal),
                lambda value: bool(value.cutoffs),
            )
            release.set()
            outcome = await waiting
            assert outcome.result.status is TaskStatus.SUCCEEDED
            assert outcome.observation_error is None
            boundary, _ = await runtime.history.read_model_interaction_updates(
                execution_id, principal=runtime.default_principal, after_cutoffs=boundary.cutoffs,
            )
            assert boundary.durable_history_available
            if delegated:
                child, = await runtime.executions.list_children(execution_id, principal=runtime.default_principal)
                identities.append(child.execution_id)
                child_record = await storage.execution.executions.get(child.execution_id, tenant_id=runtime.tenant_id)
                assert child_record is not None and not child_record.requires_task_invocation_capture
                assert child_record.restart_checkpoints and child_record.durable_trace
            assert observed
        finally:
            release.set()
            await waiting
    count = len(requests)
    async with Runtime.open(
        "task-agent-persistence", models=models, features=group, storage=RuntimeStorage.sqlite(database),
    ) as reopened:
        for execution_id in identities:
            page = await reopened.history.model_interactions(execution_id, principal=reopened.default_principal)
            assert page.items
        task = reopened.tasks.from_agent("answer", reopened.agents.get())
        graph = await reopened.tasks.bind(task).start(graph_definition, idempotency_key="once")
        assert (await graph.wait(timeout_seconds=10)).result.status is TaskStatus.SUCCEEDED
        assert len(requests) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", (False, True))
@pytest.mark.parametrize("trace", (False, True))
async def test_delegated_guarantees_follow_frozen_parent_across_runtime(tmp_path, restart: bool, trace: bool) -> None:
    entered = {prompt: asyncio.Event() for prompt in ("root", "child", "grandchild")}
    release = asyncio.Event()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        prompt = next(part.content for message in reversed(messages) for part in message.parts
                      if isinstance(part, UserPromptPart))
        entered[prompt].set()
        await release.wait()
        return ModelResponse(parts=[TextPart(prompt)])

    def capabilities(with_guarantees: bool) -> RuntimeFeatures[None]:
        group: RuntimeFeatures[None] = RuntimeFeatures("application")
        group.agent("default", model="default", allow_tools=())
        if with_guarantees and restart:
            group.capability(CheckpointCapability(), scope="runtime")
        if with_guarantees and trace:
            group.capability(TraceCapability(), scope="runtime")
        return group

    models = _Models(_UsageFunctionModel(model))
    database = tmp_path / "inherited-guarantees.db"
    original, delegated = RuntimeStorage.sqlite(database), RuntimeStorage.sqlite(database)
    async with Runtime.open("inheritance", models=models, storage=original, features=capabilities(True)) as owner:
        root = await owner.agents.get().start("root", idempotency_key="root")
        try:
            await asyncio.wait_for(entered["root"].wait(), 5)
            async with Runtime.open("inheritance", models=models, storage=delegated, features=capabilities(False)) as borrower:
                child_id = await _start_delegated_execution(borrower, delegated, root.execution_id, "child")
                await asyncio.wait_for(entered["child"].wait(), 5)
                grandchild_id = await _start_delegated_execution(borrower, delegated, child_id, "grandchild")
                await asyncio.wait_for(entered["grandchild"].wait(), 5)
                identities = (root.execution_id, child_id, grandchild_id)
                for execution_id in identities:
                    record = await delegated.execution.executions.get(execution_id, tenant_id=borrower.tenant_id)
                    assert record is not None
                    assert (record.restart_checkpoints, record.durable_trace) == (restart, trace)
                release.set()
                for execution_id in (child_id, grandchild_id):
                    execution = await borrower.executions.get(execution_id)
                    await execution.wait(timeout_seconds=10)
                await root.wait(timeout_seconds=10)
        finally:
            release.set()
    reopened_storage = RuntimeStorage.sqlite(database)
    async with Runtime.open("inheritance", models=models, storage=reopened_storage, features=capabilities(False)) as reopened:
        for execution_id in identities:
            record = await reopened_storage.execution.executions.get(execution_id, tenant_id=reopened.tenant_id)
            assert record is not None
            assert (record.restart_checkpoints, record.durable_trace) == (restart, trace)
