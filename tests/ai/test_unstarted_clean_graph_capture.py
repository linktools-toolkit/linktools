#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task.engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Clean graph replay strips historical context from unstarted captured inputs."""
from ._runtime_test_helpers import model_features

from ._runtime_test_helpers import workspace_features

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput, AgentTaskInputContext, CaptureGraphRequest, CaptureInputRequest
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import Task, TaskNodeContext, TaskGraph, TaskNode
from linktools.ai.workspace import Workspace

from .test_captured_execution_context import _NoToolsModels


@pytest.mark.asyncio
@pytest.mark.parametrize(("task_mode", "input_mode", "mode"), (
    ("literal", "fixed_input", "declaration_graph"),
    ("literal", "reproject_input", "declaration_graph"),
    ("projected", "fixed_input", "declaration_graph"),
    ("projected", "fixed_input", "materialized_graph"),
    ("projected", "reproject_input", "declaration_graph"),
))
async def test_clean_unstarted_graph_preserves_captured_input_semantics(
    tmp_path: Path, task_mode: str, input_mode: str, mode: str,
) -> None:
    work = tmp_path / "workspace"
    work.mkdir()
    attachment = work / "input.txt"
    attachment.write_text("accepted attachment", encoding="utf-8")
    group = workspace_features("clean-input", workspace=Workspace.load(work))
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    context = ExecutionInputContext.from_messages(
        (ModelRequest(parts=[UserPromptPart("historical secret")]),
         ModelResponse(parts=[TextPart("historical answer")])),
        session_metadata={"private": "historical session metadata"},
    )
    ready = False
    projected_graphs = []

    async def project(context: AgentTaskInputContext) -> str:
        projected_graphs.append(context.graph_id)
        return context.input["question"]

    async def gate(context: TaskNodeContext) -> str:
        if not ready:
            raise AIError(ErrorCode.TASK_NODE_FAILED)
        return "ready"

    async with Persistence.open('clean-input', storage=PersistenceStorage.filesystem(tmp_path / 'state')) as persistence_60, Runtime.attach(persistence=persistence_60, features=model_features(_NoToolsModels(), group)) as runtime, TaskEngine.open(persistence=persistence_60, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("owner", runtime.tenant_id)
        agent_task = Task.from_agent("clean.agent", runtime.agents.get(),
            build_input=project if task_mode == "projected" else None)
        gate_task = Task("clean.gate", gate, effect_policy="none")
        engine = runtime_engine.with_definitions(agent_task, gate_task)
        source = await engine.start(TaskGraph("source", (
            TaskNode("agent", task=agent_task, input=AgentTaskInput(
                "current prompt" if task_mode == "literal" else "",
                parameters={"question": "current prompt"} if task_mode == "projected" else {},
                files=("input.txt",),
                input_context=context)),
        )), principal=principal, idempotency_key="source")
        assert ((await source.wait()).result).status is TaskStatus.SUCCEEDED
        execution = await source.execution("agent")
        capture = await runtime_engine.inputs.capture_input(
            execution.execution_id, CaptureInputRequest(principal, "source-input"))
        task_capture = await runtime_engine.inputs.task_input(
            capture, principal=principal, input_mode=input_mode)
        original = await runtime_engine.inputs.read_task(task_capture, principal=principal)
        blocked = await engine.start(TaskGraph("blocked", (
            TaskNode("gate", task=gate_task),
            TaskNode("agent", ("gate",), task=agent_task, input_capture=task_capture),
        )), principal=principal, idempotency_key="blocked")
        assert ((await blocked.wait()).result).status is TaskStatus.FAILED
        with pytest.raises(AIError) as unavailable:
            await blocked.execution("agent")
        assert unavailable.value.code is ErrorCode.EXECUTION_NOT_READY
        with pytest.raises(AIError) as unavailable:
            await runtime_engine.inputs.capture_graph("blocked", CaptureGraphRequest(
                principal, "captured-graph", mode=mode, context_policy="captured"))
        assert unavailable.value.code is ErrorCode.INPUT_CONTEXT_UNAVAILABLE
        assert unavailable.value.safe_details["reason"] == "graph_node_never_started"
        if task_mode == "projected" and input_mode == "reproject_input":
            with pytest.raises(AIError) as unavailable:
                await runtime_engine.inputs.capture_graph("blocked", CaptureGraphRequest(
                    principal, "clean-graph", mode=mode, context_policy="clean"))
            assert unavailable.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
            assert unavailable.value.safe_details["reason"] == "graph_node_never_started"
            assert projected_graphs == ["source"]
            assert (await runtime_engine.inputs.read_task(task_capture, principal=principal)) == original
            return
        graph_capture = await runtime_engine.inputs.capture_graph(
            "blocked", CaptureGraphRequest(principal, "clean-graph", mode=mode, context_policy="clean"))
        assert await runtime_engine.inputs.capture_graph(
            "blocked", CaptureGraphRequest(principal, "clean-graph", mode=mode, context_policy="clean")) == graph_capture
        template = await runtime_engine.inputs.read_graph(graph_capture, principal=principal)
        agent = next(node for node in template.nodes if node.node_id == "agent")
        clean = await runtime_engine.inputs.read_task(agent.input_capture, principal=principal)
        assert clean.input_mode == original.input_mode
        for before, after in ((original.input, clean.input), (original.original_input, clean.original_input)):
            expected = dict(before)
            expected.pop("capture_context", None)
            expected.update(session_id=None, memory_scope=None)
            assert dict(after) == expected
        assert (await runtime_engine.inputs.read_task(task_capture, principal=principal)) == original

        attachment.write_text("changed live attachment", encoding="utf-8")
        ready = True
        replay = await engine.start(TaskGraph("replay", template.nodes), principal=principal,
                                    idempotency_key="replay")
        assert ((await replay.wait()).result).status is TaskStatus.SUCCEEDED
        assert projected_graphs == (["source"] if task_mode == "projected" else [])
        replay_execution = await replay.execution("agent")
        interactions = await replay_execution.model_interactions(include_content=True)
        request = str(interactions.items[0].request)
        assert "current prompt" in request
        assert "historical secret" not in request
        assert "historical answer" not in request
        replay_capture = await runtime_engine.inputs.capture_input(
            replay_execution.execution_id, CaptureInputRequest(principal, "replay-input"))
        accepted = await runtime_engine.inputs.read_agent(replay_capture, principal=principal)
        assert [item.data for item in accepted.prompt if isinstance(item, BinaryContent)] == [b"accepted attachment"]
        assert accepted.input_context.model_messages() == ()
        assert dict(accepted.input_context.session_metadata) == {}
        assert (await runtime.executions.inspect(replay_execution.execution_id, principal=principal)).session_id is None
