#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task import Task
from linktools.ai.task_engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Captured Task executions retain child bindings independently of live declarations."""

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.agent import AgentBindingContract
from linktools.ai.capability import AgentContext
from linktools.ai.core import ExecutionStatus, JsonValue, Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import AgentTaskInput, CaptureInputRequest
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import TaskGraph, TaskNode

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _HistoricalOutput(BaseModel):
    historical_count: int


class _CurrentOutput(BaseModel):
    current_label: str


class _RootOutput(BaseModel):
    summary: str


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if "delegate_task" in {tool.name for tool in info.function_tools}:
        returned = any(isinstance(part, ToolReturnPart) and part.tool_name == "delegate_task"
                       for message in messages for part in message.parts)
        if not returned:
            return ModelResponse(parts=[ToolCallPart("delegate_task", {
                "subagent_id": "child", "task": "inspect the synthetic sample",
            })])
        if info.output_tools:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"summary": "complete"})])
        return ModelResponse(parts=[TextPart("complete")])
    if not info.output_tools:
        return ModelResponse(parts=[TextPart("historical text")])
    output = info.output_tools[0]
    properties = output.parameters_json_schema["properties"]
    value = {"historical_count": 7} if "historical_count" in properties else {"current_label": "current"}
    return ModelResponse(parts=[ToolCallPart(output.name, value)])


def _features(
    revision: int, schema: dict[str, JsonValue] | None, *, tool_revision: int | None = None,
    respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse] = _respond,
) -> RuntimeFeatures:
    features = RuntimeFeatures("captured-child-binding")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=("child",))
    features.agent("child", revision=revision, output_schema=schema,
                   system_prompt=f"Synthetic child revision {revision}",
                   allow_tools=() if tool_revision is None else ("lookup",),
                   allow_skills=(), allow_subagents=())
    if tool_revision is not None:
        def lookup(context: AgentContext[None]) -> str:
            return "synthetic lookup"
        features.tool(lookup, revision=tool_revision, effect_policy="none")
    return model_features(_Models(_UsageFunctionModel(respond)), features)


@pytest.mark.asyncio
@pytest.mark.parametrize("structured_child", (False, True), ids=("text-child", "structured-child"))
@pytest.mark.parametrize("override_root", (False, True), ids=("default-root", "explicit-root"))
async def test_captured_task_preserves_child_contract_after_reopen(
    tmp_path: Path, structured_child: bool, override_root: bool,
) -> None:
    historical_schema = _HistoricalOutput.model_json_schema() if structured_child else None
    historical_output = {"historical_count": 7} if structured_child else {"text": "historical text"}
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('captured-child', storage=storage) as persistence_80, Runtime.attach(persistence=persistence_80, features=_features(1, historical_schema)) as runtime, TaskEngine.open(persistence=persistence_80, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("capture-owner", runtime.tenant_id)
        source = await runtime.agents.get().start("delegate sample inspection", principal=principal)
        assert (await source.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        source_children = await runtime.executions.list_children(source.execution_id, principal=principal)
        assert len(source_children) == 1
        source_child = source_children[0]
        original_info = await runtime.history.inspect_execution(source_child.execution_id, principal=principal)
        assert (await runtime.executions.result(source_child.execution_id, principal=principal)).output == historical_output
        capture = await runtime_engine.inputs.capture_input(
            source.execution_id, CaptureInputRequest(principal, "capture-parent", "clean"),
        )

    async with Persistence.open('captured-child', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_93, Runtime.attach(persistence=persistence_93, features=_features(2, _CurrentOutput.model_json_schema())) as runtime, TaskEngine.open(persistence=persistence_93, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        captured_capture = await runtime_engine.inputs.read_agent(capture, principal=principal)
        captured_capture.require_importable()
        captured = Task.from_agent("historical-task", runtime.agents.restore(captured_capture.binding))
        frozen = AgentBindingContract.from_payload(captured.contract["config"]["binding_contract"])
        child_contract = frozen.subagent_binding_map["child"]
        assert child_contract.agent_spec.revision == 1
        assert child_contract.output_mode == ("structured" if structured_child else "text")
        assert child_contract.binding_digest == original_info.binding_digest
        fresh = Task.from_agent("current-task", runtime.agents.get())
        engine = runtime_engine.with_definitions(captured, fresh)
        run = await engine.start(TaskGraph("compare-captured-child", (
            TaskNode("historical", task=captured, input=AgentTaskInput("delegate sample inspection"),
                     output_type=_RootOutput if override_root else None),
            TaskNode("current", task=fresh, input=AgentTaskInput("delegate sample inspection")),
        )), principal=principal, idempotency_key="compare-captured-child")
        assert (await run.wait(timeout_seconds=20)).result.status is TaskStatus.SUCCEEDED
        graph_id = run.graph_id
        historical_root = await run.execution("historical")
        current_root = await run.execution("current")
        historical_children = await runtime.executions.list_children(historical_root.execution_id, principal=principal)
        current_children = await runtime.executions.list_children(current_root.execution_id, principal=principal)
        assert len(historical_children) == len(current_children) == 1
        historical_child_id = historical_children[0].execution_id
        current_child_id = current_children[0].execution_id
        # The ordinary Task must keep using the newly declared child, not the capture.
        assert (await runtime.executions.result(current_child_id, principal=principal)).output == {"current_label": "current"}
        assert (await runtime.executions.result(historical_child_id, principal=principal)).output == historical_output
        historical_info = await runtime.history.inspect_execution(historical_child_id, principal=principal)
        assert historical_info.binding_digest == original_info.binding_digest
        assert (await runtime.history.inspect_execution(current_child_id, principal=principal)).binding_digest != original_info.binding_digest
        root_output = {"summary": "complete"} if override_root else {"text": "complete"}
        assert await run.result("historical") == root_output
        historical_root_info = await runtime.history.inspect_execution(historical_root.execution_id, principal=principal)
        assert any(ref.kind == "agent" and ref.id == "child" and ref.revision == 1
                   for ref in historical_root_info.capability_refs)

    async with Persistence.open('captured-child', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_129, Runtime.attach(persistence=persistence_129, features=_features(2, _CurrentOutput.model_json_schema())) as runtime, TaskEngine.open(persistence=persistence_129, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        readback = await runtime_engine.with_definitions().get(graph_id, principal=principal)
        assert await readback.result("historical") == root_output
        assert (await runtime.executions.result(historical_child_id, principal=principal)).output == historical_output
        assert await runtime.history.inspect_execution(historical_child_id, principal=principal) == historical_info
        assert (await runtime.executions.result(current_child_id, principal=principal)).output == {"current_label": "current"}
        assert (await runtime.executions.result(source_child.execution_id, principal=principal)).output == historical_output


@pytest.mark.asyncio
async def test_captured_task_rejects_unavailable_child_tool_revision(tmp_path: Path) -> None:
    async with Persistence.open('captured-tool', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_141, Runtime.attach(persistence=persistence_141, features=_features(1, None, tool_revision=1)) as runtime, TaskEngine.open(persistence=persistence_141, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        principal = Principal("capture-owner", runtime.tenant_id)
        source = await runtime.agents.get().start("delegate sample inspection", principal=principal)
        assert (await source.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        capture = await runtime_engine.inputs.capture_input(
            source.execution_id, CaptureInputRequest(principal, "capture-tool-parent", "clean"),
        )

    child_model_requests: list[list[ModelMessage]] = []

    def observe(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if "delegate_task" not in {tool.name for tool in info.function_tools}:
            child_model_requests.append(messages)
        return _respond(messages, info)

    with pytest.raises(AIError) as close_error:
        async with Persistence.open('captured-tool', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_158, Runtime.attach(persistence=persistence_158, features=_features(2, _CurrentOutput.model_json_schema(), tool_revision=2, respond=observe)) as runtime, TaskEngine.open(persistence=persistence_158, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
            captured_capture = await runtime_engine.inputs.read_agent(capture, principal=principal)
            captured_capture.require_importable()
            captured = Task.from_agent("unavailable-child", runtime.agents.restore(captured_capture.binding))
            run = await runtime_engine.with_definitions(captured).start(TaskGraph("unavailable-child", (
                TaskNode("historical", task=captured, input=AgentTaskInput("delegate sample inspection")),
            )), principal=principal, idempotency_key="unavailable-child")
            assert (await run.wait(timeout_seconds=20)).result.status is TaskStatus.FAILED
            execution = await run.execution("historical")
            with pytest.raises(AIError) as wait_error:
                await execution.wait(timeout_seconds=10)
            assert wait_error.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
    assert close_error.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
    assert child_model_requests == []
