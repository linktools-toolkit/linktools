#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemySwarmCommitCoordinator contract: each operation opens one
Storage UoW that the swarm-state write + the swarm_commit_log row share, so
a retried call with the SAME (commit_id, request_hash) returns the recorded
result and the SAME commit_id with a DIFFERENT request_hash raises
SwarmCommitConflictError."""

import asyncio
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc)

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.run.persistence._replay import canonical_request_hash
from linktools.ai.storage.sqlalchemy.models import Base
from linktools.ai.swarm.commit import (
    CancelSwarmCommand,
    CompleteSwarmCommand,
    CompleteSwarmStepCommand,
    FailSwarmCommand,
    FailSwarmStepCommand,
    StartSwarmCommand,
    StartSwarmStepCommand,
    SwarmCommitConflictError,
)
from linktools.ai.swarm.models import (
    SwarmRun,
    TokenUsage,
    SwarmStatus,
    SwarmStep,
    SwarmStepStatus,
)
from linktools.ai.swarm.persistence.sqlalchemy import SqlAlchemySwarmStore
from linktools.ai.swarm.persistence.sqlalchemy_commit import (
    SqlAlchemySwarmCommitCoordinator,
)


def _run(coro):
    return asyncio.run(coro)


def _make_storage(tmp_path):
    """Build a SqlAlchemyStorage-equivalent that exposes .transaction().

    The coordinator only needs .transaction() (yielding a tx with .session)
    + a SqlAlchemySwarmStore. We construct a minimal shim that opens a real
    session-only transaction (no cross-store semantics needed for these
    tests)."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/swarm-commit.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    _run(_create())
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    swarm_store = SqlAlchemySwarmStore(session_factory=session_factory)

    class _Tx:
        def __init__(self, session):
            self.session = session

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


def test_start_is_idempotent_by_commit_id(tmp_path):
    storage, swarm_store = _make_storage(tmp_path)
    coordinator = SqlAlchemySwarmCommitCoordinator(storage, swarm_store)
    payload = {
        "id": "swarm-run-1",
        "run_id": "driving-run-1",
        "round": 0,
        "status": SwarmStatus.PENDING,
        "version": 1,
        "token_usage": TokenUsage(),
        "cost": "0",
        "created_at": _now(),
        "updated_at": _now(),
    }

    async def _run_async():
        command = StartSwarmCommand(
            commit_id="start:swarm-1",
            swarm_run_id="swarm-run-1",
            expected_version=1,
            payload=payload,
            event_context=None,
        )
        first = await coordinator.start(command)
        second = await coordinator.start(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second
    assert first["swarm_run_id"] == "swarm-run-1"


def test_start_same_commit_id_different_payload_conflicts(tmp_path):
    storage, swarm_store = _make_storage(tmp_path)
    coordinator = SqlAlchemySwarmCommitCoordinator(storage, swarm_store)

    async def _run_async():
        payload_a = {
            "id": "swarm-run-x",
            "run_id": "driving-a",
            "round": 0,
            "status": SwarmStatus.PENDING,
            "version": 1,
            "token_usage": TokenUsage(),
            "cost": "0",
            "created_at": _now(),
            "updated_at": _now(),
        }
        payload_b = {
            "id": "swarm-run-x",
            "run_id": "driving-b",  # different driving run -> different hash
            "round": 0,
            "status": SwarmStatus.PENDING,
            "version": 1,
            "token_usage": TokenUsage(),
            "cost": "0",
            "created_at": _now(),
            "updated_at": _now(),
        }
        await coordinator.start(
            StartSwarmCommand(
                commit_id="start:swarm-x",
                swarm_run_id="swarm-run-x",
                expected_version=1,
                payload=payload_a,
                event_context=None,
            )
        )
        with pytest.raises(SwarmCommitConflictError):
            await coordinator.start(
                StartSwarmCommand(
                    commit_id="start:swarm-x",
                    swarm_run_id="swarm-run-x",
                    expected_version=1,
                    payload=payload_b,
                    event_context=None,
                )
            )

    _run(_run_async())


def test_terminal_complete_is_idempotent(tmp_path):
    storage, swarm_store = _make_storage(tmp_path)
    coordinator = SqlAlchemySwarmCommitCoordinator(storage, swarm_store)
    # Seed a swarm run via start, then move it to RUNNING (the legal pre-
    # terminal state) the way the engine would between start and complete.
    _run(
        coordinator.start(
            StartSwarmCommand(
                commit_id="start:swarm-c",
                swarm_run_id="swarm-run-c",
                expected_version=1,
                payload={
                    "id": "swarm-run-c",
                    "run_id": "driving-c",
                    "round": 0,
                    "status": SwarmStatus.PENDING,
                    "version": 1,
                    "token_usage": TokenUsage(),
                    "cost": "0",
                    "created_at": _now(),
                    "updated_at": _now(),
                },
                event_context=None,
            )
        )
    )
    _run(
        swarm_store.update_run(
            "swarm-run-c", expected_version=1, status=SwarmStatus.RUNNING
        )
    )

    async def _run_async():
        command = CompleteSwarmCommand(
            commit_id="complete:swarm-c",
            swarm_run_id="swarm-run-c",
            expected_version=2,
            payload={},
            event_context=None,
        )
        first = await coordinator.complete(command)
        second = await coordinator.complete(command)
        return first, second

    first, second = _run(_run_async())
    assert first == second
    run = _run(swarm_store.get_run("swarm-run-c"))
    assert run.status is SwarmStatus.SUCCEEDED
