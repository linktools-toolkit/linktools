#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current-turn selection does not discard the current run's durable tool state."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.capability import SessionCapability
from linktools.ai.core import ApprovalDecision, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.workspace import ToolPermissionPolicy, Workspace, WorkspacePolicy

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, workspace_features


class _HostAuthorization(TenantAuthorizationPolicy):
    """Authorize this synthetic host to manage its own Sessions."""


@pytest.mark.asyncio
async def test_current_turn_keeps_tool_results_across_runtime_recovery(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("retained tool result", encoding="utf-8")
    requests = []

    async def model(messages, info):
        requests.append(tuple(messages))
        if len(requests) == 1:
            return ModelResponse(parts=[TextPart("old answer")])
        assert "old question" not in str(messages)
        assert "old answer" not in str(messages)
        if len(requests) == 2:
            assert len(messages) == 1
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "file.txt"}, "read")])
        assert len(requests) == 3
        prompts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        assert prompts == ["read the file"]
        returned, = (part for part in messages[-1].parts if isinstance(part, ToolReturnPart))
        assert returned.tool_call_id == "read" and "retained tool result" in str(returned.content)
        assert len(messages) >= 3
        return ModelResponse(parts=[TextPart("done")])

    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(default="ask")))
    features = workspace_features("session-context-recovery", workspace=workspace)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file",), tool_retries=0, output_retries=0)
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    def open_host():
        return Runtime.open("session-context-recovery", storage=storage, features=features,
                            metrics=None, authorization=_HostAuthorization())

    async with open_host() as runtime:
        principal = Principal("owner", runtime.tenant_id)
        await runtime.sessions.create_turn_session("session", principal=principal)
        old = await runtime.sessions.start(
            "session", "old question", agent="reader", principal=principal, idempotency_key="old",
        )
        assert (await old.wait_final(timeout_seconds=5)).view["output"] == {"text": "old answer"}
        turn = await runtime.agents.get("reader").start(
            "read the file", session_id="session", principal=principal, idempotency_key="read",
            session_context="current_turn",
        )
        assert (await turn.wait()).status is ExecutionStatus.WAITING_DEFERRED
        approval, = await runtime.approvals.list(turn.key, principal=principal)
        await runtime.handoff()
    async with open_host() as runtime:
        persisted, = await runtime.approvals.list(turn.key, principal=principal)
        assert persisted.approval_id == approval.approval_id
        await runtime.approvals.decide(turn.key, ApprovalDecisionRequest(
            principal, persisted.approval_id, "approve", ApprovalDecision.APPROVE,
        ))
        resumed = await runtime.agents.get("reader").start(
            "read the file", session_id="session", principal=principal, idempotency_key="read",
            session_context="current_turn",
        )
        assert resumed.key == turn.key
        finished = await resumed.wait_final(timeout_seconds=10)
        assert finished.status is ExecutionStatus.SUCCEEDED
        assert finished.view["output"] == {"text": "done"}
        assert len(requests) == 3
        timeline = await runtime.sessions.timeline("session", principal=principal)
        assert [item["turn_id"] for item in timeline["items"]] == [old.key.turn_id, turn.key.turn_id]
