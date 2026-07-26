#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm persisted fencing contract.

The run-level owner is ``SwarmRun.execution_token``. Every fresh commit (other
than ``start``, which establishes the token) must supply a fence whose token
EQUALS the persisted value; a mismatch (stale worker / rotated token) is
rejected. An already-completed commit still replays after the token rotates
(replay precedes the fence check). Step commits keep their secondary
``expected_version`` / ``active_run_id`` fence on top of the run-level token."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import (
    SwarmCompleted,
    SwarmStarted,
    SwarmStepCompleted,
)
from linktools.ai.run.commit import RunCommitId
from linktools.ai.run.models import RunResult
from linktools.ai.swarm.commit import (
    CompleteSwarmCommand,
    CompleteSwarmStepCommand,
    CompleteSwarmPayload,
    CompleteSwarmStepPayload,
    StartSwarmCommand,
    StartSwarmPayload,
    SwarmCommitId,
    SwarmCommitPolicy,
    SwarmExecutionFence,
    SwarmFenceLostError,
    SwarmFenceRequiredError,
    SwarmFenceStateError,
)
from linktools.ai.swarm.models import (
    SwarmRun,
    SwarmStatus,
    SwarmStep,
    SwarmStepStatus,
    TaskInput,
    TokenUsage,
)
from linktools.ai.swarm.persistence.codec import SwarmCommitCodec
from linktools.ai.swarm.persistence.filesystem import FilesystemSwarmStore
from linktools.ai.swarm.persistence.filesystem_commit import (
    FilesystemSwarmCommitCoordinator,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ctx(run_id: str = "driving-1") -> EventStreamContext:
    return EventStreamContext(
        stream_id=run_id, run_id=run_id, root_run_id=run_id,
        parent_run_id=None, session_id="sess", runnable_id="swarm",
    )


def _run(
    swarm_run_id: str = "swarm-1", *, execution_token: str | None = None
) -> SwarmRun:
    return SwarmRun(
        id=swarm_run_id,
        run_id="driving-1",
        round=0,
        status=SwarmStatus.RUNNING,
        version=1,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=_now(),
        updated_at=_now(),
        execution_token=execution_token,
        execution_owner_id=None if execution_token is None else "test-owner",
        execution_generation=0 if execution_token is None else 1,
    )


def _make_coordinator(tmp_path):
    swarm_store = FilesystemSwarmStore(root=tmp_path / "swarm")
    coordinator = FilesystemSwarmCommitCoordinator(
        swarm_store,
        event_store=_NullEventStore(),
        transactions_root=tmp_path / "tx",
        policy=SwarmCommitPolicy(fencing_required=True),
        codec=SwarmCommitCodec(),
    )
    return coordinator, swarm_store


class _NullEventStore:
    """A minimal event store that records appends so a fence failure's
    'zero event writes' can be asserted."""

    def __init__(self) -> None:
        self.appended: list = []

    async def append(self, *args, **kwargs):
        self.appended.append((args, kwargs))

    async def append_once(self, *args, **kwargs):
        self.appended.append((args, kwargs))


def _start_command(swarm_run_id: str, token: str, *, run=None) -> StartSwarmCommand:
    return StartSwarmCommand(
        commit_id=SwarmCommitId(f"start:{swarm_run_id}"),
        swarm_run_id=swarm_run_id,
        expected_version=1,
        payload=StartSwarmPayload(
            run=run or _run(swarm_run_id, execution_token=token),
            started_event=SwarmStarted(swarm_run_id="swarm-1", swarm_id="swarm-1"),
            event_context=_ctx(),
        ),
        fence=SwarmExecutionFence(token),
    )


def _complete_command(swarm_run_id: str, token: str, *, commit_id: str | None = None, expected_version: int = 1) -> CompleteSwarmCommand:
    return CompleteSwarmCommand(
        commit_id=SwarmCommitId(commit_id or f"complete:{swarm_run_id}"),
        swarm_run_id=swarm_run_id,
        expected_version=expected_version,
        payload=CompleteSwarmPayload(
            result=RunResult(output={"done": True}),
            completed_event=SwarmCompleted(swarm_run_id="swarm-1"),
            event_context=_ctx(),
        ),
        fence=SwarmExecutionFence(token),
    )


async def _rotate_token(
    swarm_store: FilesystemSwarmStore, swarm_run_id: str, new_token: str
) -> str:
    """Rotate ownership through the production atomic store operation."""
    current = await swarm_store.get_run(swarm_run_id)
    assert current is not None
    lease = await swarm_store.claim_execution(
        swarm_run_id,
        owner_id=new_token,
        expected_generation=current.execution_generation,
    )
    return lease.token


# --- real token validation --------------------------------------------------


def test_random_nonempty_fence_is_rejected(tmp_path):
    """A fence whose token was never persisted for this run is rejected."""
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "owner-token"))
        # A random token (never the persisted one) must be rejected.
        with pytest.raises(SwarmFenceLostError):
            await coordinator.complete(_complete_command("swarm-1", "random-other-token"))

    asyncio.run(_run_async())


def test_previous_worker_fence_is_rejected(tmp_path):
    """After a reclaim rotates the token, the previous worker's fence fails."""
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "worker-A"))
        await _rotate_token(swarm_store, "swarm-1", "worker-B")
        with pytest.raises(SwarmFenceLostError):
            await coordinator.complete(_complete_command("swarm-1", "worker-A"))

    asyncio.run(_run_async())


def test_current_worker_fence_commits(tmp_path):
    """The current owner's token commits successfully."""
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "worker-A"))
        result = await coordinator.complete(_complete_command("swarm-1", "worker-A"))
        assert result["swarm_run_id"] == "swarm-1"

    asyncio.run(_run_async())


# --- replay vs fence ordering ------------------------------------------------


def test_replay_succeeds_after_fence_rotates(tmp_path):
    """An already-completed commit replays even after the token rotates: the
    replay check precedes the fence check, so token rotation does not break
    idempotent replays of finished commits."""
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "worker-A"))
        # First complete (worker-A owns).
        first = await coordinator.complete(
            _complete_command("swarm-1", "worker-A", commit_id="complete:swarm-1")
        )
        # Token rotates (reclaim). The SAME complete commit_id now replays --
        # it must return the FIRST result, NOT raise SwarmFenceLostError.
        await _rotate_token(swarm_store, "swarm-1", "worker-B")
        replayed = await coordinator.complete(
            _complete_command("swarm-1", "worker-A", commit_id="complete:swarm-1")
        )
        assert replayed == first

    asyncio.run(_run_async())


def test_new_commit_fails_after_fence_rotates(tmp_path):
    """A FRESH commit (new commit_id) under a stale token fails after the
    token rotates -- only already-completed commits are protected by replay."""
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "worker-A"))
        await _rotate_token(swarm_store, "swarm-1", "worker-B")
        with pytest.raises(SwarmFenceLostError):
            await coordinator.complete(
                _complete_command("swarm-1", "worker-A", commit_id="complete:NEW")
            )

    asyncio.run(_run_async())


# --- fence state errors -----------------------------------------------------


def test_missing_fence_is_rejected(tmp_path):
    """fencing_required=True with no fence supplied is SwarmFenceRequiredError
    (the policy layer surfaces this before the store is even reached)."""
    policy = SwarmCommitPolicy(fencing_required=True)
    with pytest.raises(SwarmFenceRequiredError):
        policy.validate(supplied=None, stored_token="anything")


def test_run_with_no_token_is_state_error(tmp_path):
    """A run created without an execution_token (e.g. directly via the store,
    bypassing start) cannot satisfy a fenced commit -- SwarmFenceStateError."""
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        # Create the run directly with no token (bypassing the coordinator).
        await swarm_store.create_run(_run("swarm-1"))  # execution_token=None
        with pytest.raises(SwarmFenceStateError):
            await coordinator.complete(_complete_command("swarm-1", "any-token"))

    asyncio.run(_run_async())


# --- single codec + filesystem fence inside lock -----------------------------


def test_filesystem_fence_check_happens_inside_lock(tmp_path):
    """FilesystemSwarmStore.assert_execution_fence acquires the per-store lock
    (the same lock mutations hold), so concurrent in-process fence checks
    SERIALISE: at most one is ever inside the critical section at a time. We
    prove this (not just that the method runs) by instrumenting the lock
    itself to track peak concurrency across many overlapping coroutine
    calls -- a non-lock-guarded impl would let them all run simultaneously
    and peak > 1."""
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await swarm_store.create_run(_run("swarm-1", execution_token="tok-0"))
        current_token = await _rotate_token(swarm_store, "swarm-1", "worker-1")

        # Instrument the ACTUAL lock object the store uses: wrap its
        # acquire/release so we count holders strictly inside the critical
        # section (after acquire returns, before release).
        real_lock = swarm_store._lock
        peak: list[int] = [0]
        holders: list[int] = [0]

        class _TrackingLock:
            def locked(self):
                return real_lock.locked()

            async def acquire(self):
                await real_lock.acquire()
                holders[0] += 1
                peak[0] = max(peak[0], holders[0])
                # Yield so the scheduler can try to start other waiters --
                # if they were not blocked on this lock they would run here
                # and inflate peak.
                await asyncio.sleep(0)

            def release(self):
                holders[0] -= 1
                real_lock.release()

            async def __aenter__(self):
                await self.acquire()

            async def __aexit__(self, exc_type, exc, tb):
                self.release()

        swarm_store._lock = _TrackingLock()

        # Many concurrent fence checks against the same run. Under the real
        # lock they queue one-at-a-time (peak == 1); without a lock they
        # overlap (peak >> 1).
        await asyncio.gather(*[
            swarm_store.assert_execution_fence(
                "swarm-1", expected_token=current_token
            )
            for _ in range(20)
        ])
        assert peak[0] == 1, (
            f"fence checks did NOT serialize under the store lock "
            f"(peak concurrency {peak[0]} > 1)"
        )

    asyncio.run(_run_async())


def test_fence_failure_has_zero_event_and_log_writes(tmp_path):
    """A fence failure performs zero state/event/log writes: the fence is
    checked before the inflight journal or any business write."""
    events = _NullEventStore()
    swarm_store = FilesystemSwarmStore(root=tmp_path / "swarm")
    coordinator = FilesystemSwarmCommitCoordinator(
        swarm_store,
        event_store=events,
        transactions_root=tmp_path / "tx",
        policy=SwarmCommitPolicy(fencing_required=True),
        codec=SwarmCommitCodec(),
    )

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "worker-A"))
        events_before = len(events.appended)
        completed_before = len(list(coordinator._completed_dir.glob("*.json")))
        await _rotate_token(swarm_store, "swarm-1", "worker-B")
        with pytest.raises(SwarmFenceLostError):
            await coordinator.complete(
                _complete_command("swarm-1", "worker-A", commit_id="complete:fresh")
            )
        # The fence failure happened BEFORE the (complete) event append (zero
        # NEW events) and BEFORE the completion log was written (no NEW entry).
        assert len(events.appended) == events_before
        assert len(list(coordinator._completed_dir.glob("*.json"))) == completed_before
        # And the run is still RUNNING (the business write never ran).
        run = await swarm_store.get_run("swarm-1")
        assert run.status is SwarmStatus.RUNNING

    asyncio.run(_run_async())


# --- step secondary fencing -------------------------------------------------


def _step(swarm_run_id: str = "swarm-1") -> SwarmStep:
    return SwarmStep(
        id="task-1",
        swarm_run_id=swarm_run_id,
        parent_task_id=None,
        assigned_agent_id="agent-1",
        description="do thing",
        status=SwarmStepStatus.CLAIMED,
        dependencies=(),
        input=TaskInput(prompt="x", metadata={}),
        result=None,
        error=None,
        attempts=0,
        version=5,
        claimed_at=_now(),
        lease_expires_at=None,
        created_at=_now(),
        updated_at=_now(),
        active_run_id="child-run-1",
    )


def _complete_step_command(swarm_run_id: str, token: str, *, expected_version: int, active_run_id: "str | None", commit_id: str = "complete_step:1") -> CompleteSwarmStepCommand:
    return CompleteSwarmStepCommand(
        commit_id=SwarmCommitId(commit_id),
        swarm_run_id=swarm_run_id,
        step_attempt_id="attempt-1",
        expected_version=expected_version,
        payload=CompleteSwarmStepPayload(
            task_id="task-1",
            result=RunResult(output={"step": "done"}),
            active_run_id=active_run_id,
            completed_event=SwarmStepCompleted(swarm_run_id="swarm-1", task_id="task-1"),
            event_context=_ctx(),
        ),
        fence=SwarmExecutionFence(token),
    )


def test_step_version_is_secondary_fence(tmp_path):
    """A step commit with a stale expected_version fails the secondary
    (step-level) fence even when the run-level token matches."""
    from linktools.ai.errors import SwarmConflictError

    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "worker-A"))
        await swarm_store.create_task(_step("swarm-1"))
        # Wrong expected_version (task is at v5, supply v4).
        with pytest.raises(SwarmConflictError):
            await coordinator.complete_step(
                _complete_step_command(
                    "swarm-1", "worker-A", expected_version=4, active_run_id="child-run-1",
                )
            )

    asyncio.run(_run_async())


def test_step_active_run_id_is_secondary_fence(tmp_path):
    """A step commit whose active_run_id does not match the task's current
    child run fails the secondary fence."""
    from linktools.ai.errors import SwarmConflictError

    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        await coordinator.start(_start_command("swarm-1", "worker-A"))
        await swarm_store.create_task(_step("swarm-1"))  # active_run_id="child-run-1"
        with pytest.raises(SwarmConflictError):
            await coordinator.complete_step(
                _complete_step_command(
                    "swarm-1", "worker-A",
                    expected_version=5, active_run_id="stale-child-run",
                )
            )

    asyncio.run(_run_async())


# --- SQL fence uses the UoW session + FOR UPDATE ----------------------------


def test_sql_fence_check_uses_uow_session(tmp_path):
    """The SQL fence check runs inside the UoW's shared session (NOT a fresh
    session the store opens itself) AND issues SELECT ... FOR UPDATE so the
    row is locked for the duration of the surrounding transaction. Both are
    required for the run-level owner fence to be race-free against a
    concurrent reclaim: a fence that opened its own session / skipped FOR
    UPDATE would release the row before the mutation writes, letting a
    reclaim sneak in between check and write."""
    pytest.importorskip("aiosqlite")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.swarm.models import TokenUsage
    from linktools.ai.swarm.persistence.sqlalchemy import SqlAlchemySwarmStore

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/fence-uow.db")

    async def _setup():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        store = SqlAlchemySwarmStore(session_factory=session_factory)
        # Seed a run with a known token through the store's own write path.
        await store.create_run(
            SwarmRun(
                id="swarm-uow",
                run_id="driving-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_now(),
                updated_at=_now(),
                execution_token="uow-tok",
            )
        )
        return store, session_factory

    store, session_factory = asyncio.run(_setup())

    async def _run_async():
        # Open a UoW session by hand and bind the store to it (mirrors what
        # Storage.transaction() does -- sets store._session so every method
        # uses THIS session, not its own freshly-opened one).
        async with session_factory() as session:
            captured: "list[str]" = []
            real_execute = session.execute

            async def _spy(statement, *args, **kwargs):
                try:
                    captured.append(
                        str(statement.compile(compile_kwargs={"literal_binds": True}))
                    )
                except Exception:
                    captured.append(str(statement))
                return await real_execute(statement, *args, **kwargs)

            session.execute = _spy  # type: ignore[method-assign]
            store._session = session  # UoW mode: use THIS session, not own

            run = await store.assert_execution_fence(
                "swarm-uow", expected_token="uow-tok"
            )
            assert run.execution_token == "uow-tok"
            # The fence SELECT was issued on the UoW session (the spy fired).
            assert any("'swarm-uow'" in s for s in captured), (
                f"fence check did not run on the UoW session; captured: {captured}"
            )
            # And it locked the row (FOR UPDATE) for the tx's duration.
            assert any("FOR UPDATE" in s.upper() for s in captured), (
                f"fence SELECT did not acquire a row lock; captured: {captured}"
            )

    try:
        asyncio.run(_run_async())
    finally:
        asyncio.run(engine.dispose())
