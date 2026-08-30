#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composed Runtime approval-wait regression."""

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime.state import RecoveryCheckpointState
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.workspace import (
    Workspace,
    WorkspacePolicy,
    WorkspaceToolPermissionPolicy,
)
from pydantic_ai.models.test import TestModel


class _ToolModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(call_tools=["read_file"])


class _ToolModels:
    def snapshot(self) -> "_ToolModels":
        return self

    def resolve(self, route_id: str) -> _ToolModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _ToolModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _ToolModelBinding:
        if route_id not in {None, "default"} or dict(payload) != _ToolModelBinding.semantic_payload:
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _ToolModelBinding()


@pytest.mark.asyncio
async def test_composed_runtime_ask_enters_durable_approval_wait(tmp_path: Path) -> None:
    agent_path = tmp_path / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=("read_file",))
        )
    )
    workspace = Workspace.load(
        tmp_path,
        policy=WorkspacePolicy(
            tool_permissions=WorkspaceToolPermissionPolicy(default="ask")
        ),
    )
    state = RuntimeState.filesystem(tmp_path / "runtime-state")

    async with Runtime.open(
        workspace,
        models=_ToolModels(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        execution = await runtime.agent("default").start("read a file")
        record = None
        for _ in range(200):
            record = await state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.tenant_id,
            )
            if record is not None and record.status is ExecutionStatus.WAITING_APPROVAL:
                break
            await asyncio.sleep(0.01)

        assert record is not None
        assert record.status is ExecutionStatus.WAITING_APPROVAL
        checkpoint = await state.recovery.checkpoints.get(
            execution.execution_id,
            tenant_id=runtime.tenant_id,
        )
        assert checkpoint is not None
        assert checkpoint.state is RecoveryCheckpointState.WAITING
        assert checkpoint.pending_approval is not None
        approvals = await state.recovery.approvals.list_pending(
            execution.execution_id,
            tenant_id=runtime.tenant_id,
        )
        assert len(approvals) == 1
        assert approvals[0].execution_id == execution.execution_id
