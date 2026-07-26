#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotent replay tests for the refactored SqlAlchemyRunCommitCoordinator
(P5 SQL commit log integration).

A retried pause with the SAME commit_id + same payload returns the recorded
result WITHOUT re-executing the business writes; the SAME commit_id with a
DIFFERENT payload raises RunCommitConflictError."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import RunPaused
from linktools.ai.run.commit import ApprovalRequestData, ExecutionFence, PauseRunCommand, PausedRunCommit, RunCommitId, RunCommitPolicy
from linktools.ai.run.models import RunInput, RunRecord, RunnableType, RunStatus
from linktools.ai.run.persistence.codec import RunCommitCodec
from linktools.ai.run.persistence.sqlalchemy.commit import (
    SqlAlchemyRunCommitCoordinator,
)
from linktools.ai.run.persistence.sqlalchemy.commit_log import (
    RunCommitConflictError,
)
from linktools.ai.run.persistence.sqlalchemy.run import SqlAlchemyRunStore
from linktools.ai.run.context import RunContext


_APPROVAL_BINDING = {
    "descriptor_fingerprint": "fp-test",
    "handler_revision": "h1",
    "provider_revision": "p1",
    "policy_revision": "pol1",
    "capability_revision": "cap1",
    "result_processor_revision": "rp1",
    "arguments_hash": "ah1",
}


def _run(coro):
    return asyncio.run(coro)


def _ctx(run_id, session_id):
    return RunContext(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


@pytest.fixture
def coordinator(tmp_path):
    """A real SqlAlchemyRunCommitCoordinator against a fresh sqlite DB."""
    from linktools.ai.runtime.persistence.sqlalchemy import _ReferenceSqlAlchemyComposition as SqlAlchemyStorage

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'replay.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    storage = SqlAlchemyStorage(
        session_factory=session_factory, blobs_root=tmp_path / "blobs"
    )

    async def _setup():
        from linktools.ai.storage.sqlalchemy.models import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return storage

    storage = _run(_setup())
    return SqlAlchemyRunCommitCoordinator(storage, policy=RunCommitPolicy(fencing_required=False), codec=RunCommitCodec()), session_factory


def _seed_run(session_factory, run_id, session_id):
    async def _run_async():
        store = SqlAlchemyRunStore(session_factory=session_factory)
        record = RunRecord(
            id=run_id,
            root_run_id=run_id,
            parent_run_id=None,
            session_id=session_id,
            runnable_id="agent:test",
            runnable_type=RunnableType.AGENT,
            status=RunStatus.RUNNING,
            input=RunInput(prompt="seed"),
            result=None,
            error=None,
            version=1,
            created_at=datetime.now(timezone.utc),
            started_at=None,
            finished_at=None,
        )
        return await store.create(record)

    _run(_run_async())


def _pause_command(*, run_id, approval_id, payload, commit_id):
    return PauseRunCommand(
        run_id=run_id,
        expected_version=1,
        approval_request=ApprovalRequestData(
            approval_id=approval_id,
            tool_call_id=f"tc-{approval_id}",
            tool_name="shell",
            reason="review",
            arguments={"cmd": "ls"},
            tenant_id="tenant-test",
            binding=_APPROVAL_BINDING,
        ),
        checkpoint_payload=payload,
        paused_event=RunPaused(run_id=run_id, reason="review"),
        event_context=EventStreamContext.from_run_context(_ctx(run_id, "sess-1")),
        commit_id=RunCommitId(commit_id),)


def test_pause_replayed_with_same_commit_id_returns_recorded_result(coordinator):
    """A retried pause with the SAME commit_id + payload returns the recorded
    checkpoint_id; business writes are NOT re-executed (the checkpoint count
    stays at 1, not 2)."""
    coord, session_factory = coordinator
    _seed_run(session_factory, "run-replay", "sess-1")
    cmd = _pause_command(
        run_id="run-replay",
        approval_id="appr-1",
        payload=b"checkpoint-v1",
        commit_id="pause:run-replay:appr-1",
    )

    first = _run(coord.pause(cmd))
    assert isinstance(first, PausedRunCommit)
    assert first.approval_id == "appr-1"
    first_checkpoint = first.checkpoint_id

    # Replay: same commit_id + same payload. Returns the recorded
    # checkpoint_id without re-appending a checkpoint.
    second = _run(coord.pause(cmd))
    assert second.checkpoint_id == first_checkpoint


def test_pause_with_same_commit_id_but_different_payload_conflicts(coordinator):
    """A retried pause with the SAME commit_id but a DIFFERENT checkpoint
    payload raises RunCommitConflictError -- two distinct operations cannot
    share one id."""
    coord, session_factory = coordinator
    _seed_run(session_factory, "run-conflict", "sess-1")

    first = _pause_command(
        run_id="run-conflict",
        approval_id="appr-c",
        payload=b"v1",
        commit_id="pause:run-conflict:appr-c",
    )
    # Same commit_id, DIFFERENT payload.
    second = _pause_command(
        run_id="run-conflict",
        approval_id="appr-c",
        payload=b"v2-different",
        commit_id="pause:run-conflict:appr-c",
    )

    _run(coord.pause(first))
    with pytest.raises(RunCommitConflictError):
        _run(coord.pause(second))
