#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Composed Runtime approval-wait regression."""
from ._runtime_test_helpers import model_features

from ._runtime_test_helpers import workspace_features

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import ApprovalStatus, ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.contracts import RecoveryCheckpointState
from linktools.ai.workspace import (
    Workspace,
    WorkspacePolicy,
    ToolPermissionPolicy,
)
from pydantic_ai.models.test import TestModel


class _ToolModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(call_tools=["read_file"])


class _ToolModels(_ToolModelBinding):
    pass


@pytest.mark.parametrize("backend", ("memory", "filesystem"))
@pytest.mark.asyncio
async def test_composed_runtime_ask_enters_approval_wait(
    tmp_path: Path,
    backend: str,
) -> None:
    workspace = Workspace.load(
        tmp_path,

        policy=WorkspacePolicy(
            tool_permissions=ToolPermissionPolicy(default="ask")
        ),
    )
    state = (
        PersistenceStorage.in_memory()
        if backend == "memory"
        else PersistenceStorage.filesystem(tmp_path / "runtime-state")
    )
    application = workspace_features("application", workspace=workspace)
    application.agent(
        "default",
        model="default",
        allow_tools=("read_file",),
    )

    async with Persistence.open('default', storage=state) as persistence_65, Runtime.open(persistence=persistence_65, features=model_features(_ToolModels(), application)) as runtime:
        execution = await runtime.agents.get("default").start("read a file")
        record = None
        for _ in range(200):
            record = await runtime._state.execution.executions.get(
                execution.execution_id,
                tenant_id=runtime.tenant_id,
            )
            if record is not None and record.status is ExecutionStatus.WAITING_DEFERRED:
                break
            await asyncio.sleep(0.01)

        assert record is not None
        assert record.status is ExecutionStatus.WAITING_DEFERRED
        checkpoint = await runtime._state.recovery.checkpoints.get(
            execution.execution_id,
            tenant_id=runtime.tenant_id,
        )
        assert checkpoint is not None
        assert checkpoint.state is RecoveryCheckpointState.WAITING
        assert checkpoint.pending_tools is not None
        approvals = await runtime._state.recovery.approvals.list_pending(
            execution.execution_id,
            tenant_id=runtime.tenant_id,
        )
        assert len(approvals) == 1
        assert approvals[0].execution_id == execution.execution_id
        assert approvals[0].status is ApprovalStatus.PENDING
