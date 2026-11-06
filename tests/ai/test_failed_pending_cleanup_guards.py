#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""No-source cleanup requires stopped local authority, not just source absence."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.runs import RuntimeAgentRunStore
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime._pending_turn import _PendingExecutionTurns


def _backend():
    backend = object.__new__(LocalExecutionBackend)
    backend._tenant_id = "tenant"
    backend._pending_turns = _PendingExecutionTurns()
    backend._tasks = {}
    backend._turn_monitors = {}
    backend._turn_owners = {}
    backend._execution_durable_tasks = {}
    backend._recovery_reconcile_tasks = {}
    backend._worker_failures = {"old": object()}
    backend._pending_audit_events = {"old": ["old"], "new": ["new"]}
    backend._pending_audit_locks = {}
    backend._execution_buffer = SimpleNamespace(discard=lambda _id: None, flush_cleanup=AsyncMock(),
                                                require_unsubmitted=lambda _id: None)
    backend._run_store = object.__new__(RuntimeAgentRunStore)
    backend._run_store.discard_unsubmitted_execution_staging = AsyncMock()
    coordinator = SimpleNamespace(read_owner=AsyncMock(return_value=None))
    backend._pending_turns.coordination = coordinator
    turn = SimpleNamespace(
        tenant_id="tenant", execution=SimpleNamespace(execution_id="old"),
        failure=AIError(ErrorCode.STORAGE_INTEGRITY_ERROR), uncertain=False,
        committing=False, committed=None, reconcile=None, coordination=object(),
        boundary=asyncio.Lock(), session=None,
        reservation=SimpleNamespace(idempotency=SimpleNamespace(scope="scope", idempotency_key_digest="request")),
    )
    turn.reservation.execution = turn.execution
    backend._pending_turns._executions["tenant", "old"] = turn
    return backend, turn


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", (
    "uncertain", "committing", "committed", "reconcile", "healthy", "replaced", "no_failure",
    "worker", "recovery",
))
async def test_pending_cleanup_retains_unsettled_authority(guard: str) -> None:
    backend, turn = _backend()
    pending = None
    if guard in {"uncertain", "committing"}:
        setattr(turn, guard, True)
    elif guard in {"committed", "reconcile"}:
        setattr(turn, guard, object())
    elif guard == "healthy":
        turn.failure = None
    elif guard == "replaced":
        backend._pending_turns._executions["tenant", "old"] = object()
    elif guard == "no_failure":
        backend._worker_failures.clear()
    else:
        pending = asyncio.create_task(asyncio.Event().wait())
        target = backend._tasks if guard == "worker" else backend._recovery_reconcile_tasks
        target["old"] = pending
    try:
        await backend._cleanup_failed_pending_turn(turn, forget=True)
        backend._run_store.discard_unsubmitted_execution_staging.assert_not_awaited()
        backend._execution_buffer.flush_cleanup.assert_not_awaited()
        assert backend._pending_turns.get("old", tenant_id="tenant") is not None
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_error", (False, True))
async def test_pending_cleanup_requires_coordination_absence(owner_error: bool) -> None:
    backend, turn = _backend()
    reader = backend._pending_turns.coordination.read_owner
    if owner_error:
        reader.side_effect = OSError("coordination unavailable")
    else:
        reader.return_value = object()
    with pytest.raises(OSError if owner_error else AIError):
        await backend._cleanup_failed_pending_turn(turn, forget=True)
    reader.assert_awaited_once_with("old", tenant_id="tenant")
    backend._run_store.discard_unsubmitted_execution_staging.assert_not_awaited()
    assert "old" in backend._worker_failures


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ("renewal", "source_task", "owner_read"))
async def test_pending_cleanup_rechecks_submission_after_await(race: str) -> None:
    backend, turn = _backend()
    entered, release = asyncio.Event(), asyncio.Event()

    async def competing() -> None:
        entered.set()
        try:
            await release.wait()
        finally:
            # Promotion already in progress must settle before cleanup can decide.
            async with turn.boundary:
                turn.reconcile = object()

    if race == "owner_read":
        async def read_owner(*args, **kwargs):
            turn.reconcile = object()
            return None
        backend._pending_turns.coordination.read_owner.side_effect = read_owner
        await backend._cleanup_failed_pending_turn(turn, forget=True)
    else:
        competitor = asyncio.create_task(competing())
        await entered.wait()
        if race == "renewal":
            backend._turn_monitors["old"] = competitor
        else:
            backend._execution_durable_tasks["old"] = {competitor}
        cleanup = asyncio.create_task(backend._cleanup_failed_pending_turn(turn, forget=True))
        try:
            if race == "source_task":
                await asyncio.sleep(0)
                assert not cleanup.done()
                release.set()
            await asyncio.wait_for(cleanup, 2)
        finally:
            release.set()
            await asyncio.gather(competitor, return_exceptions=True)
    backend._run_store.discard_unsubmitted_execution_staging.assert_not_awaited()
    assert "old" in backend._worker_failures


@pytest.mark.asyncio
async def test_cancelled_cleanup_keeps_renewal_identity_until_it_stops() -> None:
    backend, turn = _backend()
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def renew() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    renewal = asyncio.create_task(renew())
    backend._turn_monitors["old"] = renewal
    await entered.wait()
    cleanup = asyncio.create_task(backend._cleanup_failed_pending_turn(turn))
    await cancelled.wait()
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    assert backend._turn_monitors["old"] is renewal and not renewal.done()
    backend._run_store.discard_unsubmitted_execution_staging.assert_not_awaited()
    release.set()
    await renewal
    await backend._cleanup_failed_pending_turn(turn, forget=True)
    assert backend._pending_turns.get("old", tenant_id="tenant") is None
    assert backend._pending_audit_events == {"new": ["new"]}


@pytest.mark.asyncio
async def test_failed_pending_promotion_cannot_restart_after_staging_release() -> None:
    backend, turn = _backend()
    backend._check_pending_lease = AsyncMock()
    await backend._cleanup_failed_pending_turn(turn)
    with pytest.raises(AIError) as failure:
        await backend.promote_pending_execution("old")
    assert failure.value is turn.failure
    backend._check_pending_lease.assert_not_awaited()


class _Lock:
    @asynccontextmanager
    async def hold(self, _key):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ("execution", "absent", "read_error", "flight", "seal", "generation"))
async def test_run_owner_requires_fresh_absence_and_unsealed_generation(guard: str) -> None:
    store = object.__new__(RuntimeAgentRunStore)
    store._tenant_id = "tenant"
    store._ensure_business = AsyncMock()
    store._execution_producers = {"run": ("old", 3 if guard == "generation" else 2)}
    store._terminal_seals = {"run": object()} if guard == "seal" else {}
    store._durability_flights = {"run": object()} if guard == "flight" else {}
    store._history_lock = _Lock()
    store.trace_persistence = SimpleNamespace(stop=AsyncMock(), discard=AsyncMock(),
        settle_execution_cleanup=AsyncMock(), dirty={"run"})
    store.wait_projection_flight = AsyncMock()
    store._release_staging_many = AsyncMock()
    in_read = False

    async def read(callback):
        nonlocal in_read
        in_read = True
        if guard == "read_error":
            raise OSError("source unavailable")
        try:
            return await callback(object())
        finally:
            in_read = False

    async def get(*args, **kwargs):
        assert in_read, "cleanup proof must bypass hot cache through a source transaction"
        return object() if guard == "execution" else None

    store._execution_repository = SimpleNamespace(state_store=SimpleNamespace(read=read),
                                                  get=get)
    if guard == "absent":
        await store._discard_unsubmitted_execution_staging("old")
        store.trace_persistence.discard.assert_awaited_once_with("run")
        store._release_staging_many.assert_awaited_once_with(
            candidate_agent_run_ids=("run",), execution_id="old",
        )
        assert store.trace_persistence.dirty == set()
        return
    with pytest.raises(OSError if guard == "read_error" else AIError):
        await store._discard_unsubmitted_execution_staging("old")
    store.trace_persistence.discard.assert_not_awaited()
    store._release_staging_many.assert_not_awaited()
    assert store.trace_persistence.dirty == {"run"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("recovery", "renewal", "source_task"))
async def test_pending_cleanup_rechecks_quiescence_after_owner_read(kind: str) -> None:
    backend, turn = _backend()
    task = asyncio.create_task(asyncio.Event().wait())

    async def read_owner(*args, **kwargs):
        if kind == "source_task":
            backend._execution_durable_tasks["old"] = {task}
        else:
            target = backend._recovery_reconcile_tasks if kind == "recovery" else backend._turn_monitors
            target["old"] = task
        return None

    backend._pending_turns.coordination.read_owner.side_effect = read_owner
    try:
        await backend._cleanup_failed_pending_turn(turn, forget=True)
        backend._run_store.discard_unsubmitted_execution_staging.assert_not_awaited()
        assert "old" in backend._worker_failures
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", ("sealed", "successor"))
async def test_execution_source_plan_blocks_sibling_trace_discard(guard: str) -> None:
    from linktools.ai.runtime._execution_buffer import _ExecutionPayloadBuffer
    from linktools.ai.storage import InMemoryCacheBackend

    backend, turn = _backend()
    buffer = _ExecutionPayloadBuffer(InMemoryCacheBackend(max_bytes=2**20))
    buffer.bind_producer("old", 3 if guard == "successor" else 2)
    if guard == "sealed":
        buffer._scopes["old"].sealed = True
    backend._execution_buffer = buffer
    with pytest.raises(AIError) as failure:
        await backend._cleanup_failed_pending_turn(turn, forget=True)
    assert failure.value.code is ErrorCode.STORAGE_CONFLICT
    backend._run_store.discard_unsubmitted_execution_staging.assert_not_awaited()
    assert "old" in backend._worker_failures


@pytest.mark.asyncio
async def test_unsubmitted_cache_append_uncertainty_keeps_exact_ack_identity() -> None:
    from linktools.ai.runtime._execution_buffer import _Entry, _ExecutionPayloadBuffer
    from linktools.ai.storage import InMemoryCacheBackend

    cache = InMemoryCacheBackend(max_bytes=2**20)
    buffer = _ExecutionPayloadBuffer(cache)
    buffer.bind_producer("old", 2)
    scope = buffer._scopes["old"]
    content = b"original uncertain cache admission"
    assert await cache.append(scope.id, "original-id", "old", content, wait=False)
    assert await cache.append("other", "other-id", "other", b"other", wait=False)
    scope.entries.append(_Entry("original-id", "unused"))
    scope.uncertain = ("original-id", "old", content)
    # This is cache admission uncertainty, with no sealed/source submission.
    # The retained original ID can be acknowledged without inventing a new ID.
    buffer.require_unsubmitted("old")
    buffer.discard("old")
    await buffer.flush_cleanup("old")
    assert await cache.read_batch(scope.id, limit=10) == ()
    assert await cache.read_batch("other", limit=10) == (("other-id", "other", b"other"),)
    assert scope.uncertain is None


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", (False, True))
async def test_pending_cleanup_retains_submitted_plan_after_owner_stops(unknown: bool) -> None:
    backend, turn = _backend()
    turn.committing = True
    turn.reconcile = object()
    turn.uncertain = unknown
    closed = asyncio.Event()
    closed.set()
    backend._turn_owners["old"] = SimpleNamespace(lifetime=None, closed=closed)
    await backend._cleanup_failed_pending_turn(turn, forget=True)
    backend._pending_turns.coordination.read_owner.assert_not_awaited()
    backend._run_store.discard_unsubmitted_execution_staging.assert_not_awaited()
    backend._execution_buffer.flush_cleanup.assert_not_awaited()
    assert backend._pending_turns.get("old", tenant_id="tenant") is turn
    assert "old" in backend._worker_failures


@pytest.mark.asyncio
@pytest.mark.parametrize("guard", (
    "PENDING", "CLAIMED", "EFFECT_UNKNOWN", "source_commit_unknown", "worker", "owner", "recovery",
    "new_source_task", "new_owner_during_proof",
))
async def test_revoked_recovery_cleanup_retains_unsettled_work(guard: str) -> None:
    from linktools.ai.core import ExecutionStatus, ToolOperationStatus
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
    from linktools.ai.runtime._execution_buffer import _Entry, _ExecutionPayloadBuffer
    from linktools.ai.storage import InMemoryCacheBackend

    backend, turn = _backend()
    backend._pending_turns.release(turn)
    backend.load_execution = AsyncMock(return_value=SimpleNamespace(status=ExecutionStatus.RECOVERY_REQUIRED))
    archive = object.__new__(ModelInteractionStateStepArchive)
    archive.require_revoked_execution_producer = AsyncMock()
    backend._run_stores = {PersistenceDomain.TRACE: archive}
    backend._agent_run_lifecycle = SimpleNamespace(discard_revoked_producer_staging=AsyncMock())
    cache = InMemoryCacheBackend(max_bytes=2**20)
    buffer = backend._execution_buffer = _ExecutionPayloadBuffer(cache)
    buffer.bind_producer("old", 2)
    scope = buffer._scopes["old"]
    content = b"capture whose source outcome is not settled"
    assert await cache.append(scope.id, "capture-id", "old", content, wait=False)
    scope.entries.append(_Entry("capture-id", "unused"))
    scope.sealed = guard == "source_commit_unknown"
    status = ToolOperationStatus(guard) if guard in {"PENDING", "CLAIMED", "EFFECT_UNKNOWN"} else ToolOperationStatus.COMPLETED
    pending = None
    if guard in {"worker", "recovery", "new_source_task"}:
        pending = asyncio.create_task(asyncio.Event().wait())
        if guard != "new_source_task":
            target = backend._tasks if guard == "worker" else backend._recovery_reconcile_tasks
            target["old"] = pending
    elif guard == "owner":
        backend._turn_owners["old"] = SimpleNamespace(closed=asyncio.Event())

    async def operations(*args, **kwargs):
        if guard == "new_source_task":
            backend._execution_durable_tasks["old"] = {pending}
        return (SimpleNamespace(status=status),)

    backend._tool_operations = SimpleNamespace(list_by_execution=operations)
    if guard == "new_owner_during_proof":
        async def revoked(*args, **kwargs):
            backend._turn_owners["old"] = SimpleNamespace(closed=asyncio.Event())
        archive.require_revoked_execution_producer.side_effect = revoked
    try:
        await backend._release_revoked_failed_producers()
        assert await cache.read_batch(scope.id, limit=10) == (("capture-id", "old", content),)
        assert not scope.released
        assert "old" in backend._worker_failures
        assert backend._pending_audit_events["old"] == ["old"]
        backend._agent_run_lifecycle.discard_revoked_producer_staging.assert_not_awaited()
    finally:
        if pending is not None:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
