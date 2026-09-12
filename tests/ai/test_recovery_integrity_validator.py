#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery checkpoint integrity and recoverable-frontier regressions."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
)


def _checkpoint(
    execution_id: str,
    state: RecoveryCheckpointState,
    revision: int = 0,
) -> RecoveryCheckpoint:
    now = datetime.now(timezone.utc)
    active = state is RecoveryCheckpointState.ACTIVE
    return RecoveryCheckpoint(
        execution_id=execution_id,
        tenant_id="tenant",
        step_run_id="run-1" if active else None,
        state=state,
        revision=revision,
        created_at=now,
        updated_at=now,
        handoff_phase=RecoveryHandoffPhase.NONE,
    )


@pytest.mark.asyncio
async def test_recoverable_checkpoint_is_visible_from_first_write() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-integrity", tenant_id="tenant")
    try:
        repository = state.recovery.checkpoints
        await repository.create(_checkpoint("e1", RecoveryCheckpointState.ACTIVE))
        await repository.create(_checkpoint("e2", RecoveryCheckpointState.COMPLETED))

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
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-complete", tenant_id="tenant")
    try:
        repository = state.recovery.checkpoints
        created = await repository.create(
            _checkpoint("e1", RecoveryCheckpointState.ACTIVE)
        )
        await repository.compare_and_swap(
            "e1",
            tenant_id="tenant",
            expected_revision=created.revision,
            next_record=replace(
                created,
                state=RecoveryCheckpointState.COMPLETED,
                step_run_id=None,
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
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-missing", tenant_id="tenant")
    try:
        with pytest.raises(AIError) as raised:
            await state.recovery.checkpoints.compare_and_swap(
                "missing",
                tenant_id="tenant",
                expected_revision=0,
                next_record=_checkpoint(
                    "missing",
                    RecoveryCheckpointState.COMPLETED,
                ),
            )
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await state.close()
