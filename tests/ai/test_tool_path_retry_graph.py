#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
from linktools.ai.task.engine import TaskEngine
from linktools.ai.task import Task
"""Workspace argument correction stays inside the agent's bounded tool loop."""
from ._runtime_test_helpers import is_tool_retry, model_features

from ._runtime_test_helpers import workspace_features

from pathlib import Path

import pytest
from pydantic_ai.messages import (
    ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart,
    ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.capability import TraceCapability
from linktools.ai.inputs import AgentTaskInput
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.task import TaskGraph, TaskNode
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_limit", (None, 0, 2))
async def test_workspace_path_correction_respects_graph_and_effect_boundaries(
    tmp_path: Path, retry_limit: int | None,
) -> None:
    (tmp_path / "AGENTS.md").write_text("Read workspace files carefully.\n", encoding="utf-8")
    (tmp_path / "report[1].txt").write_text("bracket report", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("plain notes", encoding="utf-8")
    requests = 0
    effects = 0
    downstream_requests = 0
    retries: list[RetryPromptPart | ToolReturnPart] = []
    returns: list[ToolReturnPart] = []

    async def record_effect(_context: AgentContext[None]) -> str:
        nonlocal effects
        effects += 1
        return "effect recorded"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests, downstream_requests
        del info
        if any(isinstance(part, UserPromptPart) and part.content == "downstream"
               for message in messages for part in message.parts):
            downstream_requests += 1
            return ModelResponse(parts=[TextPart("downstream complete")])
        requests += 1
        retries.extend(part for part in messages[-1].parts if is_tool_retry(part))
        returns.extend(part for part in messages[-1].parts
                       if isinstance(part, ToolReturnPart) and not is_tool_retry(part))
        if requests == 1:
            return ModelResponse(parts=[
                ToolCallPart("read_file", {"path": '["report[1].txt", "notes.txt"]'},
                             tool_call_id="read-invalid-1"),
                ToolCallPart("record_effect", {}, tool_call_id="effect-once"),
            ])
        if retry_limit is not None:
            return ModelResponse(parts=[ToolCallPart(
                "read_file", {"path": '["report[1].txt", "notes.txt"]'},
                tool_call_id=f"read-invalid-{requests}",
            )])
        if requests == 2:
            return ModelResponse(parts=[ToolCallPart(
                "read_file", {"path": ["report[1].txt", "notes.txt"]},
                tool_call_id="read-corrected",
            )])
        return ModelResponse(parts=[TextPart("files read")])

    group = workspace_features("path-correction", workspace=Workspace.load(tmp_path))
    group.capability(TraceCapability(), scope="runtime")
    group.tool(record_effect, effect_policy="non_replay_safe")
    group.agent("default", model="default", allow_tools=("read_file", "record_effect"),
                tool_retries=2 if retry_limit is None else retry_limit)
    async with Persistence.open('path-correction', storage=PersistenceStorage.in_memory()) as persistence_82, Runtime.attach(persistence=persistence_82, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime, TaskEngine.open(persistence=persistence_82) as engine:
        task = Task.from_agent("path.agent", runtime.agents.get())
        run = await engine.with_definitions(task).start(TaskGraph("path-graph", (
            TaskNode("reader", task=task, input=AgentTaskInput("read the files")),
            TaskNode("downstream", task=task, dependencies=("reader",),
                     input=AgentTaskInput("downstream")),
        )), idempotency_key="path-run")
        result = (await run.wait(timeout_seconds=10)).result
        nodes = {node.node_id: node for node in (await run.state()).node_states}
        execution = await run.execution("reader")
        assert effects == 1
        if retry_limit is None:
            assert result.status is TaskStatus.SUCCEEDED
            assert all(node.status is TaskStatus.SUCCEEDED for node in nodes.values())
            assert requests == 3
            assert downstream_requests == 1
            assert [part.tool_call_id for part in retries] == ["read-invalid-1"]
            assert "does not exist" in str(retries[0].content)
            corrected = [part for part in returns if part.tool_call_id == "read-corrected"]
            assert len(corrected) == 1
            assert corrected[0].tool_name == "read_file"
            assert "bracket report" in str(corrected[0].content)
            assert "plain notes" in str(corrected[0].content)
        else:
            assert result.status is TaskStatus.FAILED
            assert nodes["reader"].status is TaskStatus.FAILED
            assert nodes["reader"].error_code == ErrorCode.MODEL_RESPONSE_INVALID.value
            assert nodes["downstream"].status is TaskStatus.BLOCKED
            assert nodes["downstream"].execution_id is None
            assert downstream_requests == 0
            assert requests == retry_limit + 1
            assert [part.tool_call_id for part in retries] == [
                f"read-invalid-{attempt}" for attempt in range(1, retry_limit + 1)
            ]
            assert (await execution.wait(timeout_seconds=1)).result.status is ExecutionStatus.FAILED
        invalid_trace = (await execution.trace(tool_call_id="read-invalid-1")).items
        errors = [item for item in invalid_trace if item.payload["kind"] == "TOOL_ERROR"]
        assert len(errors) == 1
        assert errors[0].payload["status"] == "FAILED"
        assert errors[0].payload["model_request_seq"] == 1
        effect_trace = (await execution.trace(tool_call_id="effect-once")).items
        assert not any(item.payload["kind"] == "TOOL_ERROR" for item in effect_trace)
        effect_history = (await execution.history(tool_call_id="effect-once", include_content=True)).items
        assert len([item for item in effect_history if item.item_kind == "tool_call"]) == 1
        assert any("effect recorded" in str(item.content) for item in effect_history)
