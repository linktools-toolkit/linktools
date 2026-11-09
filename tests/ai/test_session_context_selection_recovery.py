#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current-turn selection does not discard the current run's durable tool state."""

from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.workspace import ToolPermissionPolicy, Workspace, WorkspacePolicy

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed, model_features, workspace_features


@pytest.mark.asyncio
async def test_current_turn_keeps_tool_results_across_runtime_recovery(tmp_path):
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
        assert "retained tool result" in str(messages[-1])
        assert len(messages) >= 3
        return ModelResponse(parts=[TextPart("done")])

    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(default="ask")))
    features = workspace_features("session-context-recovery", workspace=workspace)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file",), tool_retries=0, output_retries=0)
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("session-context-recovery", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=features, metrics=None) as runtime:
            session = runtime.agents.get("reader").session("session")
            old = await session.start("old question")
            assert (await old.wait(timeout_seconds=5)).result.output == {"text": "old answer"}
            execution = await session.start("read the file", session_context="current_turn")
            tenant = runtime.tenant_id
            await _wait_for_committed(
                lambda: state.execution.executions.get(execution.execution_id, tenant_id=tenant),
                lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED,
            )
            approvals = await state.checkpoint.approvals.list_pending(execution.execution_id, tenant_id=tenant)
            assert len(approvals) == 1
            await state.checkpoint.approvals.decide(
                approvals[0].approval_id, tenant_id=tenant, expected_status=ApprovalStatus.PENDING,
                idempotency_key_digest="a" * 64, decision=ApprovalDecision.APPROVE,
                principal_id=runtime.default_principal.principal_id, decision_digest="b" * 64,
                decided_at=datetime.now(timezone.utc),
            )
    async with Persistence.open("session-context-recovery", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=features, metrics=None) as runtime:
            finished = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=10)
            assert finished.result.status is ExecutionStatus.SUCCEEDED
            assert finished.result.output == {"text": "done"}
            assert len(requests) == 3
            timeline = await runtime.agents.get("reader").session("session").timeline()
            assert {turn.execution_id for turn in timeline.items} == {old.execution_id, execution.execution_id}
