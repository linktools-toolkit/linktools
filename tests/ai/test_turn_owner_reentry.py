#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Existing-turn actions remain subordinate to their captured parent owner."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

from linktools.ai.core import CommitPolicy, ExecutionLineageKind, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionHandle
from linktools.ai.runtime._local import LocalExecutionBackend, _OwnedExecutionTurn
from linktools.ai.runtime._pending_turn import _PendingExecutionTurns

from .test_subagent_parent_admission import _reservation


class _Backend(LocalExecutionBackend):
    def __init__(self):
        self._tenant_id = "tenant"
        self._commit = CommitPolicy.sync()
        self._turn_owners = {}
        self._tasks = {}
        self._handoff_scopes = {}
        self._turn_monitors = {}
        self._recovery_reconcile_tasks = {}
        self._pending_turns = _PendingExecutionTurns()
        self.lifetimes = []
        self.failures = []
        self.reservations = {}
        self.worker_stopped = asyncio.Event()

    def _track_checkpoint_task(self, task, label, execution_id=None):
        self.lifetimes.append(task)

    def _record_worker_failure(self, execution_id, error, *, phase=None):
        self.failures.append(error)

    def _execution_task_map(self):
        return {}

    def _request_worker_shutdown(self, execution_id, worker):
        worker.cancel()

    async def _drain_worker_task(self, execution_id, worker):
        await asyncio.gather(worker, return_exceptions=True)
        self._tasks.pop(execution_id, None)
        self.worker_stopped.set()

    async def load_execution(self, execution_id, *, tenant_id):
        return self.reservations[execution_id].execution

    async def _recovery_idempotency(self, execution):
        return self.reservations[execution.execution_id].idempotency


@asynccontextmanager
async def _child_owner(generation=1):
    backend = _Backend()
    principal = Principal("owner", "tenant")
    reservation = _reservation(generation=generation)
    reservation = replace(reservation, execution=replace(reservation.execution, status=ExecutionStatus.STARTED))
    parent_reservation = replace(reservation, execution=replace(
        reservation.execution, execution_id="parent", parent_execution_id=None,
        parent_invocation_id=None, lineage_kind=ExecutionLineageKind.RUN,
    ), parent_producer_generation=None)
    parent = _OwnedExecutionTurn(parent_reservation, None, principal, admitting=False,
                                 producer_generation=generation)
    backend._turn_owners["parent"] = parent
    backend.reservations["child"] = reservation

    async def admit(coordination):
        backend._tasks["child"] = asyncio.create_task(asyncio.Event().wait())
        return ExecutionHandle("child")

    await backend.run_turn(reservation, principal, admit)
    owner = backend._turn_owners["child"]
    try:
        yield backend, parent, owner
    finally:
        backend._stop_owned_turn(parent)
        await asyncio.gather(*backend.lifetimes)
        assert backend.failures == []


@pytest.mark.asyncio
async def test_parent_stop_keeps_owner_until_existing_action_rejects_dispatch() -> None:
    async with _child_owner() as (backend, parent, owner):
        entered, release = asyncio.Event(), asyncio.Event()
        installed = []

        async def install():
            entered.set()
            await release.wait()
            backend._require_turn_dispatch(owner)
            installed.append("child")

        action = asyncio.create_task(backend.run_existing_turn(owner.reservation.execution, 1, install))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            backend._stop_owned_turn(parent)
            await asyncio.wait_for(backend.worker_stopped.wait(), 2)
            # Give the lifetime its chance to drain. The admitted action, not
            # the already-quiescent worker, must keep this owner alive.
            await asyncio.sleep(0)
            assert not owner.closed.is_set()
            assert backend._turn_owners.get("child") is owner
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(action, 2)
            await asyncio.wait_for(owner.closed.wait(), 2)
            assert installed == []
            assert "child" not in backend._turn_owners
            assert "child" not in backend._tasks
        finally:
            release.set()
            await asyncio.gather(action, return_exceptions=True)


@pytest.mark.asyncio
async def test_stale_parent_replay_cannot_stop_current_child_owner() -> None:
    async with _child_owner(generation=2) as (backend, parent, owner):
        with pytest.raises(AIError) as raised:
            await asyncio.wait_for(backend.resume_subagent(owner.reservation.execution, 1), 2)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert not parent.stopping
        assert not owner.stopping
        assert not owner.closed.is_set()
        assert backend._turn_owners.get("child") is owner
        assert not backend._tasks["child"].done()


@pytest.mark.asyncio
async def test_failed_admission_unblocks_existing_action_before_owner_drain() -> None:
    backend = _Backend()
    principal = Principal("owner", "tenant")
    reservation = _reservation()
    reservation = replace(reservation, execution=replace(
        reservation.execution, parent_execution_id=None, parent_invocation_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
    ), parent_producer_generation=None)
    entered, release = asyncio.Event(), asyncio.Event()
    failure = AIError(ErrorCode.SESSION_BUSY)
    installed = []

    async def admit(coordination):
        entered.set()
        await release.wait()
        raise failure

    async def install():
        installed.append("child")

    starting = asyncio.create_task(backend.run_turn(reservation, principal, admit))
    following = None
    settling = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        owner = backend._turn_owners["child"]
        following = asyncio.create_task(backend.run_existing_turn(reservation.execution, None, install))
        async with asyncio.timeout(2):
            while not owner.admissions:
                await asyncio.sleep(0)
        release.set()
        settling = asyncio.gather(starting, following, return_exceptions=True)
        results = await asyncio.wait_for(asyncio.shield(settling), 2)
        assert results == [failure, failure]
        await asyncio.wait_for(owner.closed.wait(), 2)
        assert installed == []
        assert "child" not in backend._turn_owners
        assert backend.failures == []
    finally:
        release.set()
        for task in (starting, following):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (starting, following, settling) if task is not None),
                             return_exceptions=True)
        await asyncio.gather(*backend.lifetimes, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_signal_waits_for_admission_publication(monkeypatch) -> None:
    backend = _Backend()
    reservation = _reservation()
    signalled, cancel_called = asyncio.Event(), asyncio.Event()
    published = False

    class Scope:
        async def wait_signal(self):
            if not signalled.is_set():
                signalled.set()
                return "cancel_requested"
            await asyncio.Event().wait()

    owner = _OwnedExecutionTurn(
        reservation, None, Principal("owner", "tenant"), scope=Scope(),
        admission=asyncio.get_running_loop().create_future(),
    )
    backend._turn_owners["child"] = owner

    async def watch_promotion(owner):
        await asyncio.Event().wait()

    async def cancel(execution_id, principal):
        cancel_called.set()
        assert published, "cancellation reached a turn before its admission was published"
        assert execution_id == "child"

    monkeypatch.setattr(backend, "_watch_pending_promotion", watch_promotion)
    backend._pending_cancel = cancel
    monitor = asyncio.create_task(backend._watch_turn(owner))
    try:
        await asyncio.wait_for(signalled.wait(), 2)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(cancel_called.wait(), .02)
        published = True
        owner.admitting = False
        owner.admission.set_result(ExecutionHandle("child"))
        await asyncio.wait_for(cancel_called.wait(), 2)
        await asyncio.gather(*backend.lifetimes)
        assert backend.failures == []
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, *backend.lifetimes, return_exceptions=True)
