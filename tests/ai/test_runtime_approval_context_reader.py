#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval context-reader contract regressions."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state import RecoveryCheckpointState


def _backend(
    *,
    execution: object,
    checkpoint: object = None,
) -> tuple[LocalExecutionBackend, AsyncMock, AsyncMock]:
    execution_get = AsyncMock(return_value=execution)
    checkpoint_get = AsyncMock(return_value=checkpoint)
    backend = object.__new__(LocalExecutionBackend)
    backend._tenant_id = "tenant"
    backend._execution = SimpleNamespace(
        executions=SimpleNamespace(get=execution_get)
    )
    backend._recovery = SimpleNamespace(
        checkpoints=SimpleNamespace(get=checkpoint_get)
    )
    return backend, execution_get, checkpoint_get


@pytest.mark.asyncio
async def test_tool_approvals_rejects_duplicate_ids_before_storage_reads() -> None:
    backend, execution_get, checkpoint_get = _backend(execution=None)

    with pytest.raises(AIError) as error:
        await backend.tool_approvals(
            ("approval", "approval"),
            execution_id="execution",
            tenant_id="tenant",
        )

    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    execution_get.assert_not_awaited()
    checkpoint_get.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_approvals_non_waiting_execution_skips_recovery_read() -> None:
    backend, execution_get, checkpoint_get = _backend(
        execution=SimpleNamespace(status=ExecutionStatus.SUCCEEDED),
    )

    result = await backend.tool_approvals(
        ("approval",),
        execution_id="execution",
        tenant_id="tenant",
    )

    assert result == {}
    execution_get.assert_awaited_once_with("execution", tenant_id="tenant")
    checkpoint_get.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_approvals_waiting_execution_fails_closed_on_recovery_mismatch() -> None:
    backend, execution_get, checkpoint_get = _backend(
        execution=SimpleNamespace(status=ExecutionStatus.WAITING_APPROVAL),
        checkpoint=SimpleNamespace(
            state=RecoveryCheckpointState.ACTIVE,
            pending_approval=object(),
        ),
    )

    with pytest.raises(AIError) as error:
        await backend.tool_approvals(
            ("approval",),
            execution_id="execution",
            tenant_id="tenant",
        )

    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    execution_get.assert_awaited_once_with("execution", tenant_id="tenant")
    checkpoint_get.assert_awaited_once_with("execution", tenant_id="tenant")
