#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemySwarmCommitCoordinator contract: each operation opens one
Storage UoW that the swarm-state write + the swarm_commit_log row share, so
a retried call with the SAME (commit_id, request_hash) returns the recorded
result and the SAME commit_id with a DIFFERENT request_hash raises
SwarmCommitConflictError."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import SwarmCompleted, SwarmStarted
from linktools.ai.run.models import RunResult
from linktools.ai.swarm.commit import (
    CompleteSwarmCommand,
    CompleteSwarmPayload,
    StartSwarmCommand,
    StartSwarmPayload,
    SwarmCommitConflictError,
    SwarmCommitId,
    SwarmCommitPolicy,
    SwarmExecutionFence,
)
from linktools.ai.swarm.models import SwarmRun, SwarmStatus, TokenUsage
from linktools.ai.swarm.persistence.codec import SwarmCommitCodec
from linktools.ai.swarm.persistence.sqlalchemy import SqlAlchemySwarmStore
from linktools.ai.swarm.persistence.sqlalchemy_commit import (
    SqlAlchemySwarmCommitCoordinator,
)
from linktools.ai.storage.sqlalchemy.models import Base


# A fixed fence token shared by start (which stamps it as the run-level
# execution_token) and complete (which must supply a matching token). The
# coordinator is configured with fencing_required=True because start always
# establishes a token and terminal commits must prove ownership against it.
_FENCE_TOKEN = "sql-test-token"
_FENCE = SwarmExecutionFence(_FENCE_TOKEN)


def _now() -> "datetime":
    return datetime.now(timezone.utc)


def _ctx(run_id: str = "driving-1") -> EventStreamContext:
    return EventStreamContext(
        stream_id=run_id,
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id="sess",
        runnable_id="swarm",
    )


def _swarm_run(swarm_run_id: str, run_id: str) -> SwarmRun:
    return SwarmRun(
        id=swarm_run_id,
        run_id=run_id,
        round=0,
        status=SwarmStatus.PENDING,
        version=1,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=_now(),
        updated_at=_now(),
    )


def _start_command(
    swarm_run_id: str,
    run_id: str,
    *,
    commit_id: "str | None" = None,
) -> StartSwarmCommand:
    return StartSwarmCommand(
        commit_id=SwarmCommitId(commit_id or f"start:{swarm_run_id}"),
        swarm_run_id=swarm_run_id,
        expected_version=1,
        payload=StartSwarmPayload(
            run=_swarm_run(swarm_run_id, run_id),
            started_event=SwarmStarted(swarm_run_id=swarm_run_id, swarm_id="swarm"),
            event_context=_ctx(run_id),
        ),
        fence=_FENCE,
    )


def _complete_command(
    swarm_run_id: str,
    *,
    commit_id: "str | None" = None,
    expected_version: int = 2,
) -> CompleteSwarmCommand:
    return CompleteSwarmCommand(
        commit_id=SwarmCommitId(commit_id or f"complete:{swarm_run_id}"),
        swarm_run_id=swarm_run_id,
        expected_version=expected_version,
        payload=CompleteSwarmPayload(
            result=RunResult(output={"done": True}),
            completed_event=SwarmCompleted(swarm_run_id=swarm_run_id),
            event_context=_ctx(),
        ),
        fence=_FENCE,
    )


def _run(coro):
    return asyncio.run(coro)


class _NullEventStore:
    """EventStore stub: the SQL coordinator appends lifecycle events via
    ``tx.events`` inside the UoW; these tests assert on swarm_run state, not
    on emitted events, so a null sink is sufficient."""

    async def append(self, *args, **kwargs):
        pass

    async def append_once(self, *args, **kwargs):
        pass


def _make_storage(tmp_path):
    """Build a SqlAlchemyStorage-equivalent that exposes .transaction().

    The coordinator's transaction usage needs ``tx.session`` (the SQLAlchemy
    AsyncSession shared with the commit-log recorder) AND ``tx.swarms`` (a
    SqlAlchemySwarmStore bound to that session in UoW mode so swarm-state
    writes join the same transaction). We construct a minimal shim that
    wires both."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/swarm-commit.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _run(_create())
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # The standalone store (used for the between-start-and-complete
    # update_run seed call, which runs outside a coordinator UoW).
    swarm_store = SqlAlchemySwarmStore(session_factory=session_factory)

    class _Tx:
        def __init__(self, session):
            self.session = session
            # UoW-mode swarm store bound to THIS session so swarm-state writes
            # commit/roll back with the commit-log row in one transaction.
            self.swarms = SqlAlchemySwarmStore(
                session_factory=session_factory, session=session
            )
            # EventStore stub: the coordinator appends lifecycle events via
            # tx.events inside the same UoW.
            self.events = _NullEventStore()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            if exc is None:
                await self.session.commit()
            else:
                await self.session.rollback()
            await self.session.close()

    class _Storage:
        def transaction(self):
            return _Tx(session_factory())

    return _Storage(), swarm_store


def _make_coordinator(tmp_path):
    storage, swarm_store = _make_storage(tmp_path)
    coordinator = SqlAlchemySwarmCommitCoordinator(
        storage,
        # fencing_required=True: start stamps the supplied fence token as the
        # run-level execution_token, and terminal commits prove ownership
        # against it (fencing_required=False would reject the supplied fence
        # under SwarmCommitPolicy.validate()).
        policy=SwarmCommitPolicy(fencing_required=True),
        codec=SwarmCommitCodec(),
    )
    return coordinator, swarm_store


def test_start_is_idempotent_by_commit_id(tmp_path):
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        command = _start_command("swarm-run-1", "driving-run-1")
        first = await coordinator.start(command)
        second = await coordinator.start(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second
    assert first["swarm_run_id"] == "swarm-run-1"


def test_start_same_commit_id_different_payload_conflicts(tmp_path):
    coordinator, swarm_store = _make_coordinator(tmp_path)

    async def _run_async():
        # Same commit_id, different driving run_id -> different request_hash
        # -> SwarmCommitConflictError on the second start.
        await coordinator.start(
            _start_command("swarm-run-x", "driving-a", commit_id="start:swarm-x")
        )
        with pytest.raises(SwarmCommitConflictError):
            await coordinator.start(
                _start_command("swarm-run-x", "driving-b", commit_id="start:swarm-x")
            )

    _run(_run_async())


def test_terminal_complete_is_idempotent(tmp_path):
    coordinator, swarm_store = _make_coordinator(tmp_path)
    # Seed a swarm run via start, then move it to RUNNING (the legal pre-
    # terminal state) the way the engine would between start and complete.
    _run(coordinator.start(_start_command("swarm-run-c", "driving-c")))
    _run(
        swarm_store.update_run(
            "swarm-run-c",
            expected_version=1,
            expected_token=_FENCE_TOKEN,
            status=SwarmStatus.RUNNING,
        )
    )

    async def _run_async():
        command = _complete_command("swarm-run-c")
        first = await coordinator.complete(command)
        second = await coordinator.complete(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second
    run = _run(swarm_store.get_run("swarm-run-c"))
    assert run.status is SwarmStatus.SUCCEEDED
