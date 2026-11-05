#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery checkpoint integrity and recoverable-frontier regressions."""

from linktools.ai.persistence import Persistence
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.core import ExecutionStatus
from .test_execution_recovery_commands import _execution

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.contracts import (
    ExecutionRecord,
    ExecutionRecoveryDetail,
    ExecutionRecoveryState,
    RecoveryHandoffPhase,
)


def _checkpoint(
    execution_id: str,
    state: ExecutionRecoveryState,
    revision: int = 0,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return replace(_execution(now), execution_id=execution_id, root_execution_id=execution_id,
                   revision=revision, status=ExecutionStatus.FAILED if state is ExecutionRecoveryState.COMPLETED else ExecutionStatus.STARTED,
                   recovery=ExecutionRecoveryDetail(agent_run_id="run-1" if state is ExecutionRecoveryState.ACTIVE else None))


@pytest.mark.asyncio
async def test_recoverable_checkpoint_is_visible_from_first_write() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="recovery-integrity", tenant_id="tenant")
    try:
        repository = state.execution.executions
        await repository.create_execution(_checkpoint("e1", ExecutionRecoveryState.ACTIVE))
        await repository.create_execution(_checkpoint("e2", ExecutionRecoveryState.COMPLETED))

        page = await repository.list_recoverable_page(
            tenant_id="tenant",
            cursor=None,
            limit=10,
        )
        assert tuple(item.execution_id for item in page.items) == ("e1",)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_completed_transition_removes_recoverable_checkpoint() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="recovery-complete", tenant_id="tenant")
    try:
        repository = state.execution.executions
        created = await repository.create_execution(
            _checkpoint("e1", ExecutionRecoveryState.ACTIVE)
        )
        await repository.compare_and_swap(
            "e1",
            tenant_id="tenant",
            expected_revision=created.revision,
            next_record=replace(
                created,
                status=ExecutionStatus.FAILED,
                recovery=ExecutionRecoveryDetail(),
                revision=created.revision + 1,
                updated_at=datetime.now(timezone.utc),
            ),
        )
        page = await repository.list_recoverable_page(
            tenant_id="tenant",
            cursor=None,
            limit=10,
        )
        assert page.items == ()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_missing_checkpoint_cannot_be_updated_as_recoverable() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="recovery-missing", tenant_id="tenant")
    try:
        with pytest.raises(AIError) as raised:
            await state.execution.executions.compare_and_swap(
                "missing",
                tenant_id="tenant",
                expected_revision=0,
                next_record=_checkpoint(
                    "missing",
                    ExecutionRecoveryState.COMPLETED,
                ),
            )
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await state.close()
