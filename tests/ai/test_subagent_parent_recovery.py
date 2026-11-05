#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Child recovery commands preserve their own fencing and commit receipts."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionRecoveryDetail, ExecutionRecoveryState, PendingDeferredCall, PendingToolContinuation
from linktools.ai.storage import StoredPayload
from .test_execution_recovery_commands import _commands as _recovery_commands
from .test_keyed_recovery_storage import _operation
from .test_runtime_approval_recovery import _commands
from .test_subagent_parent_admission import _BACKENDS, _parent, _reservation, _storage


pytestmark = pytest.mark.asyncio


async def _child_recovery(state: Persistence, kind: str):
    repository = state.execution.executions
    reservation = _reservation()
    await repository.reserve_start(reservation)
    deferred = kind == "deferred"
    pending = PendingToolContinuation("prior-run", calls=(
        PendingDeferredCall("call", "tool", StoredPayload.inline_json({})),
    )) if deferred else None
    child = replace(reservation.execution, revision=1, agent_run_seq=1,
                    status=ExecutionStatus.WAITING_DEFERRED if deferred else ExecutionStatus.STARTED,
                    recovery=ExecutionRecoveryDetail(agent_run_id="prior-run", pending_tools=pending))
    await repository.compare_and_swap("child", tenant_id="tenant", expected_revision=0, next_record=child)

    async def resume(*, operation=None):
        if deferred:
            current = await _commands(state, "child-recovery").claim_deferred_resume_checkpoint(
                execution_id="child", tenant_id="tenant", expected_execution_revision=child.revision,
                expected_event_seq=child.event_seq,
                expected_agent_run_seq=child.agent_run_seq, expected_pending_tools=pending,
            )
            return current
        return await _recovery_commands(state).commit_resumed(child, recovery_operation=operation)

    return child, resume


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("kind", ("active", "deferred"))
async def test_child_recovery_preserves_its_own_revision_fence(tmp_path: Path, backend: str, kind: str) -> None:
    async with Persistence.open("child-recovery", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        await _parent(state)
        repository = state.execution.executions
        _, resume = await _child_recovery(state, kind)
        resumed = await resume()
        assert resumed.status is ExecutionStatus.STARTED and resumed.revision == 2
        assert (await repository.get("child", tenant_id="tenant")).producer_generation == 2
        with pytest.raises(AIError) as stale:
            await resume()
        assert stale.value.code is ErrorCode.STORAGE_CONFLICT
        assert await repository.get("child", tenant_id="tenant") == resumed


@pytest.mark.parametrize("kind", ("active", "deferred"))
async def test_child_recovery_readback_preserves_commit_after_parent_stops(monkeypatch, kind: str) -> None:
    async with Persistence.open("child-recovery", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        parent = await _parent(state)
        repository = state.execution.executions
        _, resume = await _child_recovery(state, kind)
        group = repository.state_store.storage_group
        mutate = group.mutate
        committed = False

        async def lose_acknowledgement(stores, callback):
            nonlocal committed
            if committed:
                return await mutate(stores, callback)
            committed = True
            await mutate(stores, callback)
            await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=parent.revision,
                next_record=replace(parent, status=ExecutionStatus.CANCELLED, revision=2))
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        monkeypatch.setattr(group, "mutate", lose_acknowledgement)
        resumed = await resume()
        assert resumed.revision == 2
        assert (await repository.get("parent", tenant_id="tenant")).status is ExecutionStatus.CANCELLED
        assert await repository.get("child", tenant_id="tenant") == resumed


async def test_child_recovery_receipt_replay_preserves_existing_commit() -> None:
    async with Persistence.open("child-recovery", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        parent = await _parent(state)
        repository = state.execution.executions
        _, resume = await _child_recovery(state, "active")
        operation = replace(_operation(), execution_id="child", resource_id="child")
        resumed = await resume(operation=operation)
        await repository.compare_and_swap("parent", tenant_id="tenant", expected_revision=parent.revision,
            next_record=replace(parent, status=ExecutionStatus.CANCELLED, revision=2))
        assert await resume(operation=operation) == resumed
        assert (await repository.get("parent", tenant_id="tenant")).status is ExecutionStatus.CANCELLED
        assert await repository.get("child", tenant_id="tenant") == resumed
