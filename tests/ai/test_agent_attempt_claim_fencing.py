#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ambiguous admission acknowledgements retain exact producer ownership."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime.state._contracts import (
    AgentAttemptClaim,
    PendingDeferredCall,
    PendingToolContinuation,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
)
from linktools.ai.storage import StoredPayload

from .test_execution_recovery_commands import _execution
from .test_runtime_approval_recovery import _commands


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("initial", "deferred"))
@pytest.mark.parametrize("outcome", ("own", "own_then_unrelated_revision", "rival"))
async def test_admission_readback_requires_own_receipt(
    monkeypatch: pytest.MonkeyPatch, kind: str, outcome: str,
) -> None:
    namespace = "attempt-claim"
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace=namespace, tenant_id="tenant")
    try:
        now = datetime.now(timezone.utc)
        pending = PendingToolContinuation(
            "prior-run", calls=(PendingDeferredCall("call", "tool", StoredPayload.inline_json({})),),
        ) if kind == "deferred" else None
        execution = replace(
            _execution(now), revision=1, agent_run_seq=0 if pending is None else 1,
            status=ExecutionStatus.STARTED if pending is None else ExecutionStatus.WAITING_DEFERRED,
        )
        checkpoint = RecoveryCheckpoint(
            execution.execution_id, None if pending is None else pending.source_agent_run_id,
            RecoveryCheckpointState.ADMITTED if pending is None else RecoveryCheckpointState.WAITING,
            0, now, now, pending_tools=pending,
        )
        repository = state.execution.executions
        await repository.create_with_history_head(execution)
        await state.recovery.checkpoints.create(checkpoint)
        commands = _commands(state, namespace)

        async def claim():
            if pending is None:
                return await commands.commit_agent_attempt_checkpoint(AgentAttemptClaim(
                    execution.execution_id, execution.revision, execution.agent_run_seq,
                    checkpoint.revision, checkpoint.state,
                ))
            return await commands.claim_deferred_resume_checkpoint(
                execution_id=execution.execution_id, tenant_id="tenant",
                expected_execution_revision=execution.revision,
                expected_event_seq=execution.event_seq,
                expected_recovery_revision=checkpoint.revision,
                expected_agent_run_seq=execution.agent_run_seq,
                expected_pending_tools=pending,
            )

        group = repository.state_store.storage_group
        original_mutate = group.mutate
        intercepted = False
        rival = None

        async def lose_acknowledgement(stores, callback):
            nonlocal intercepted, rival
            if intercepted:
                return await original_mutate(stores, callback)
            intercepted = True
            if outcome == "rival":
                rival = await claim()
            else:
                await original_mutate(stores, callback)
                if outcome == "own_then_unrelated_revision":
                    await repository.acquire_dependency_hold(
                        execution.execution_id, tenant_id="tenant", hold_id="dependent",
                    )
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", lose_acknowledgement)
        if outcome == "rival":
            with pytest.raises(AIError) as caught:
                await claim()
            assert caught.value.code is ErrorCode.STORAGE_CONFLICT
            assert caught.value.retryable is False
            assert rival is not None
            current, active = rival
        else:
            current, active = await claim()
        assert current.agent_run_seq == execution.agent_run_seq + 1
        assert active.state is RecoveryCheckpointState.ACTIVE
        head = await repository.get_history_head(execution.execution_id, tenant_id="tenant")
        assert head.producer_generation == execution.revision + 1
        assert isinstance(head.producer_claim_id, str) and head.producer_claim_id
        if outcome == "own_then_unrelated_revision":
            assert current.revision > head.producer_generation
        assert await repository.get(execution.execution_id, tenant_id="tenant") == current
    finally:
        await state.close()
