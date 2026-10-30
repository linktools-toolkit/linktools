#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instruction synchronization is feedback, not an ordinary tool retry."""

from pathlib import Path
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.capability import AgentContext, CapabilityContribution, ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import (
    _Models,
    _UsageFunctionModel,
    _wait_for_committed,
    is_tool_retry,
    model_features,
    workspace_features,
)


def _scope(root: Path, name: str) -> None:
    directory = root / name
    directory.mkdir()
    (directory / "AGENTS.md").write_text(f"RULE-{name}", encoding="utf-8")
    (directory / "file.txt").write_text(f"CONTENTS-{name}", encoding="utf-8")


@pytest.mark.asyncio
async def test_instruction_discovery_and_sibling_fences_work_with_zero_retry_budgets(tmp_path: Path) -> None:
    scopes = tuple(f"pkg{index}" for index in range(5))
    for scope in scopes:
        _scope(tmp_path, scope)
    requests = 0

    async def model(messages, info):
        nonlocal requests
        requests += 1
        if 2 <= requests <= len(scopes) + 1:
            feedback = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
            assert len(feedback) == (2 if requests == 2 else 1)
            assert all(part.outcome == "failed" for part in feedback)
            assert all("Re-check the current instructions" in part.content for part in feedback)
            assert not any(is_tool_retry(part) for part in feedback)
            for scope in scopes[:requests - 1]:
                assert f"RULE-{scope}" in (info.instructions or "")
        if requests <= len(scopes):
            path = f"{scopes[requests - 1]}/file.txt"
            calls = [ToolCallPart("read_file", {"path": path}, f"discover-{requests}")]
            if requests == 1:
                calls.append(ToolCallPart("read_file", {"path": path}, "same-step-sibling"))
            return ModelResponse(parts=calls)
        if requests == len(scopes) + 1:
            return ModelResponse(parts=[ToolCallPart(
                "read_file", {"path": [f"{scope}/file.txt" for scope in scopes]}, "read-known",
            )])
        assert requests == len(scopes) + 2
        for scope in scopes:
            assert f"CONTENTS-{scope}" in str(messages[-1])
        return ModelResponse(parts=[TextPart("done")])

    features = workspace_features("instruction-feedback", workspace=Workspace.load(tmp_path))
    features.agent("reader", allow_tools=("read_file",), tool_retries=0, output_retries=0)
    async with Persistence.open("instruction-feedback", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            result = (await runtime.agents.get("reader").run("read", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert requests == len(scopes) + 2


@pytest.mark.asyncio
async def test_instruction_feedback_preserves_previously_consumed_ordinary_retry_budget(tmp_path: Path) -> None:
    _scope(tmp_path, "pkg")
    requests = 0

    async def model(messages, info):
        nonlocal requests
        requests += 1
        if requests == 2:
            assert any(is_tool_retry(part) for part in messages[-1].parts)
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "pkg/file.txt"}, "discover")])
        if requests == 3:
            assert "RULE-pkg" in (info.instructions or "")
            assert any(isinstance(part, ToolReturnPart) and part.outcome == "failed"
                       for part in messages[-1].parts)
        assert requests in (1, 3)
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": "missing.txt"}, f"invalid-{requests}")])

    features = workspace_features("preserve-retry-budget", workspace=Workspace.load(tmp_path))
    features.agent("reader", allow_tools=("read_file",), tool_retries=1, output_retries=0)
    async with Persistence.open("preserve-retry-budget", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            result = (await runtime.agents.get("reader").run("read", timeout_seconds=10)).result
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == ErrorCode.MODEL_RESPONSE_INVALID.value
            assert requests == 3


@pytest.mark.asyncio
async def test_instruction_feedback_does_not_override_same_response_final_answer(tmp_path: Path) -> None:
    _scope(tmp_path, "pkg")
    requests = 0

    async def model(messages, info):
        nonlocal requests
        del messages
        requests += 1
        assert requests == 1
        assert "RULE-pkg" not in (info.instructions or "")
        return ModelResponse(parts=[
            ToolCallPart("read_file", {"path": "pkg/file.txt"}, "blocked"),
            ToolCallPart(info.output_tools[0].name, {"answer": "same-response answer"}, "final"),
        ])

    features = workspace_features("instruction-final", workspace=Workspace.load(tmp_path))
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.agent("reader", allow_tools=("read_file",), tool_retries=0, output_retries=0,
                   output_schema={"type": "object", "properties": {"answer": {"type": "string"}},
                                  "required": ["answer"], "additionalProperties": False})
    async with Persistence.open("instruction-final", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            session = await runtime.agents.get("reader").create_session("session")
            execution = await session.start("read")
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"answer": "same-response answer"}
            assert requests == 1
            stored = await persistence.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert stored is not None and stored.repository_instruction_scopes == ("pkg",)
            trace = (await execution.trace(tool_call_id="blocked")).items
            errors = [item for item in trace if item.payload["kind"] == "TOOL_ERROR"]
            assert len(errors) == 1 and errors[0].payload["status"] == "FAILED"
            assert not any("CONTENTS-pkg" in str(item.payload) for item in trace)


@pytest.mark.asyncio
async def test_instruction_feedback_does_not_swallow_a_sibling_approval_pause(tmp_path: Path) -> None:
    _scope(tmp_path, "pkg")
    requests = 0

    async def checked(context: AgentContext[object]) -> str:
        del context
        pytest.fail("unapproved tool must not execute")

    async def model(messages, info):
        nonlocal requests
        del messages, info
        requests += 1
        assert requests == 1
        return ModelResponse(parts=[
            ToolCallPart("read_file", {"path": "pkg/file.txt"}, "discover"),
            ToolCallPart("checked", {}, "approval"),
        ])

    features = workspace_features("instruction-pause", workspace=Workspace.load(tmp_path))
    source = features.tool(checked, effect_policy="none")
    source.requires_approval = True
    features.agent("reader", allow_tools=("read_file", "checked"), tool_retries=0, output_retries=0)
    capture = await model_features(_Models(_UsageFunctionModel(model)), features).capture()
    features = replace(capture, contributions=tuple(
        CapabilityContribution.from_opaque("tool", "checked", source)
        if item.kind == "tool" and item.id == "checked" else item
        for item in capture.contributions
    ))
    async with Persistence.open("instruction-pause", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get("reader").start("read and ask")
            await _wait_for_committed(
                lambda: persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id),
                lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
            )
            approvals = await persistence.recovery.approvals.list_pending(execution.execution_id, tenant_id=runtime.tenant_id)
            assert len(approvals) == 1 and requests == 1
            checkpoint = await persistence.recovery.checkpoints.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert checkpoint is not None and checkpoint.repository_instruction_overlay is not None
            assert checkpoint.pending_tools is not None
            assert [call.tool_call_id for call in checkpoint.pending_tools.approvals] == ["approval"]
