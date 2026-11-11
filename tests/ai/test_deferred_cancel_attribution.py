#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred cancellation readback attributes the actual durable operation."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionEventType, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.contracts import ExecutionCancelRequestCommit
from .test_deferred_command_readback import _prepare, _storage


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_deferred_readback_does_not_claim_competing_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, commands, _, continuation, _, _ = await _prepare(state, "cancel")
        before = await state.execution.executions.get("execution", tenant_id="tenant")
        group = state.execution.executions.state_store.storage_group
        original = group.mutate
        failure = AIError(ErrorCode.STORAGE_UNAVAILABLE)

        async def competing_commit_then_failure(stores, callback):
            # The requested callback is not called. A different authorized
            # cancellation wins after this request's original frontier was read.
            monkeypatch.setattr(group, "mutate", original)
            winner = await commands.commit_deferred_cancel_checkpoint(
                ExecutionCancelRequestCommit(
                    "execution", before.revision, before.event_seq,
                    "competing-cancel", datetime.now(timezone.utc),
                ),
                expected_pending_tools=continuation,
            )
            assert winner.status is ExecutionStatus.CANCELLING
            raise failure

        monkeypatch.setattr(group, "mutate", competing_commit_then_failure)
        result = None
        caught = None
        try:
            result = await operation()
        except AIError as error:
            caught = error
        page = await state.execution.events.list(
            "execution", tenant_id="tenant", after_event_seq=before.event_seq, limit=10,
        )
        assert len(page.items) == 1
        assert page.items[0].event_type == ExecutionEventType.CANCEL_REQUESTED
        assert page.items[0].payload == {"operation_id": "competing-cancel"}
        assert caught is not None, (
            "The command returned success for operation_id='cancel', but its mutation callback never ran "
            "and the sole durable CANCEL_REQUESTED belongs to 'competing-cancel'. "
            f"Returned status={result.status}; durable payload={page.items[0].payload}"
        )
        assert caught.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_deferred_readback_does_not_claim_resumed_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from dataclasses import replace
    from linktools.ai.core import ApprovalDecision, ApprovalStatus, ExternalCallStatus, StopReason, UsageMetrics
    from linktools.ai.persistence.contracts import ExecutionTerminalCommit, ResultRecord
    from linktools.ai.storage import StoredPayload

    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, commands, _, continuation, approvals, calls = await _prepare(state, "cancel")
        before = await state.execution.executions.get("execution", tenant_id="tenant")
        group = state.execution.executions.state_store.storage_group
        original = group.mutate

        async def normal_completion_then_failure(stores, callback):
            monkeypatch.setattr(group, "mutate", original)
            now = datetime.now(timezone.utc)
            for approval in approvals:
                await state.checkpoint.approvals.decide(
                    approval.approval_id, tenant_id="tenant", expected_status=ApprovalStatus.PENDING,
                    idempotency_key="a" * 64, decision=ApprovalDecision.APPROVE,
                    principal_id="owner", decision_digest="b" * 64, decided_at=now,
                )
            for call in calls:
                await state.checkpoint.external_calls.supply(
                    call.call_id, tenant_id="tenant", expected_status=ExternalCallStatus.PENDING,
                    idempotency_key="c" * 64, resolution_kind="succeeded",
                    result_payload=StoredPayload.inline_text("result"), resolution_metadata={}, supplied_at=now,
                )
            resumed = await commands.claim_deferred_resume_checkpoint(
                execution_id="execution", tenant_id="tenant",
                expected_execution_revision=before.revision, expected_event_seq=before.event_seq,
                expected_agent_run_seq=before.agent_run_seq, expected_pending_tools=continuation,
            )
            final = await commands.commit_terminal_checkpoint(
                ExecutionTerminalCommit(
                    resumed.revision, resumed.event_seq,
                    replace(resumed, status=ExecutionStatus.FAILED,
                            error_code=ErrorCode.EXECUTION_FAILED.value,
                            recovery=replace(resumed.recovery, pending_tools=None)),
                    ResultRecord(None, StopReason.ERROR, UsageMetrics(), now),
                    ExecutionEventType.EXECUTION_FAILED,
                    {"error_code": ErrorCode.EXECUTION_FAILED.value, "safe_error_details": {}},
                    producer_generation=resumed.producer_generation,
                ),
                expected_execution=resumed,
            )
            assert final.execution.status is ExecutionStatus.FAILED
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        monkeypatch.setattr(group, "mutate", normal_completion_then_failure)
        result = None
        caught = None
        try:
            result = await operation()
        except AIError as error:
            caught = error
        page = await state.execution.events.list(
            "execution", tenant_id="tenant", after_event_seq=before.event_seq, limit=10,
        )
        assert len(page.items) == 1
        assert page.items[0].event_type == ExecutionEventType.EXECUTION_FAILED
        assert all(event.event_type != ExecutionEventType.CANCEL_REQUESTED for event in page.items)
        assert caught is not None, (
            "The cancel callback never ran and no CANCEL_REQUESTED exists; an independent "
            f"resume and ordinary terminal commit were misreported as cancel success: {result.status}"
        )
        assert caught.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_COMMIT_UNKNOWN}


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_deferred_readback_accepts_own_cancel_before_later_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    from dataclasses import replace
    from linktools.ai.core import StopReason, UsageMetrics
    from linktools.ai.persistence.contracts import ExecutionTerminalCommit, ResultRecord

    async with Persistence.open("readback", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        operation, commands, _, _, _, _ = await _prepare(state, "cancel")
        before = await state.execution.executions.get("execution", tenant_id="tenant")
        group = state.execution.executions.state_store.storage_group
        original = group.mutate

        async def complete_then_lose_ack(stores, callback):
            monkeypatch.setattr(group, "mutate", original)
            cancelled = await original(stores, callback)
            now = datetime.now(timezone.utc)
            final = await commands.commit_terminal_checkpoint(
                ExecutionTerminalCommit(
                    cancelled.revision, cancelled.event_seq,
                    replace(cancelled, status=ExecutionStatus.CANCELLED,
                            error_code=ErrorCode.EXECUTION_CANCELLED.value),
                    ResultRecord(None, StopReason.CANCELLED, UsageMetrics(), now),
                    ExecutionEventType.EXECUTION_CANCELLED,
                    {"error_code": ErrorCode.EXECUTION_CANCELLED.value, "safe_error_details": {}},
                ),
                expected_execution=cancelled,
            )
            assert final.execution.status is ExecutionStatus.CANCELLED
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", complete_then_lose_ack)
        result = await operation()
        assert result.status is ExecutionStatus.CANCELLED
        page = await state.execution.events.list(
            "execution", tenant_id="tenant", after_event_seq=before.event_seq, limit=10,
        )
        assert [(event.event_type, event.payload) for event in page.items] == [
            (ExecutionEventType.CANCEL_REQUESTED, {"operation_id": "cancel"}),
            (ExecutionEventType.EXECUTION_CANCELLED,
             {"error_code": ErrorCode.EXECUTION_CANCELLED.value, "safe_error_details": {}}),
        ]
