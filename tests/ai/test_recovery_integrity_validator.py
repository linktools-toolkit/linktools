#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery active-index consistency without bootstrap markers."""

from datetime import datetime, timezone

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._codec import encode_domain, encode_envelope
from linktools.ai.runtime.state._contracts import (
    RecoveryActiveRecord,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryExecutionInput,
    RecoveryHandoffPhase,
    RecoveryIdempotencyInput,
)
from linktools.ai.runtime.state._store import (
    StoredRecord,
    partition_digest,
    record_key_digest,
)


def _checkpoint(
    execution_id: str,
    state: RecoveryCheckpointState,
    revision: int = 0,
) -> RecoveryCheckpoint:
    now = datetime.now(timezone.utc)
    return RecoveryCheckpoint(
        execution_id,
        "tenant",
        RecoveryExecutionInput(
            user_prompt="prompt",
            principal_id="owner",
            principal_kind="user",
            session_id=None,
            memory_scope=None,
            agent_id="default",
            binding_digest="binding",
            lineage_kind="run",
            parent_execution_id=None,
            root_execution_id=execution_id,
            source_execution_id=None,
            base_execution_id=None,
            conversation_step_run_id=None,
            idempotency=RecoveryIdempotencyInput("scope", "key", "digest"),
        ),
        "run-1" if state is RecoveryCheckpointState.ACTIVE else None,
        1 if state is RecoveryCheckpointState.ACTIVE else 0,
        state,
        RecoveryHandoffPhase.NONE,
        None,
        None,
        None,
        revision,
        now,
        now,
    )


@pytest.mark.asyncio
async def test_active_index_is_consistent_from_first_write() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-integrity", tenant_id="tenant")
    try:
        repository = state.recovery.checkpoints
        page = await repository.list_recoverable_page(
            tenant_id="tenant",
            cursor=None,
            limit=10,
        )
        assert page.items == ()
        await repository.create(_checkpoint("e1", RecoveryCheckpointState.ACTIVE))
        await repository.create(_checkpoint("e2", RecoveryCheckpointState.COMPLETED))

        report = await repository.validate_recovery_active_index(tenant_id="tenant")
        assert report.active_count == 1
        assert report.admission_count == 2
        assert report.inconsistent_execution_ids == ()

        page = await repository.list_recoverable_page(
            tenant_id="tenant",
            cursor=None,
            limit=10,
        )
        assert tuple(item.execution_id for item in page.items) == ("e1",)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_completed_transition_removes_active_entry() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-complete", tenant_id="tenant")
    try:
        repository = state.recovery.checkpoints
        created = await repository.create(
            _checkpoint("e1", RecoveryCheckpointState.ACTIVE)
        )
        from dataclasses import replace

        await repository.compare_and_swap(
            "e1",
            tenant_id="tenant",
            expected_revision=created.revision,
            next_record=replace(
                created,
                state=RecoveryCheckpointState.COMPLETED,
                step_run_id=None,
                agent_run_sequence=0,
                revision=created.revision + 1,
                updated_at=datetime.now(timezone.utc),
            ),
        )
        report = await repository.validate_recovery_active_index(tenant_id="tenant")
        assert report.active_count == 0
        assert report.inconsistent_execution_ids == ()
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_validator_reports_tampered_active_entry() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-tamper", tenant_id="tenant")
    try:
        repository = state.recovery.checkpoints
        await repository.create(_checkpoint("e1", RecoveryCheckpointState.COMPLETED))
        store = repository.state_store
        key = record_key_digest(
            "recovery-tamper",
            "tenant",
            "recovery",
            "recovery_active",
            "e1",
        )

        async def inject(transaction: object) -> None:
            await transaction.insert_record(
                StoredRecord(
                    key,
                    partition_digest(
                        "recovery-tamper",
                        "tenant",
                        "recovery",
                        "recovery_active",
                    ),
                    None,
                    None,
                    "recovery_active",
                    "e1",
                    "completed",
                    0,
                    None,
                    0,
                    None,
                    encode_envelope(
                        {
                            "type": "RecoveryActiveRecord",
                            "payload": encode_domain(
                                RecoveryActiveRecord("e1", "tenant")
                            ),
                        }
                    ),
                )
            )

        await store.mutate(inject)
        report = await repository.validate_recovery_active_index(tenant_id="tenant")
        assert report.inconsistent_execution_ids == ("e1",)
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_missing_admission_is_not_recoverable() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="recovery-missing", tenant_id="tenant")
    try:
        page = await state.recovery.checkpoints.list_recoverable_page(
            tenant_id="tenant",
            cursor=None,
            limit=10,
        )
        assert page.items == ()
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
