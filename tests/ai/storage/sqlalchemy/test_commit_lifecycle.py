#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/sqlalchemy/test_commit_lifecycle.py —
SqlAlchemyRunCommitCoordinator's start/resume/fail/request_cancel/
acknowledge_cancel -- the 5 commands added alongside the pre-existing pause/
complete to close out the full RunCommitCoordinator Protocol. Each opens one
transaction across every store involved, same as pause/complete."""

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import RunConflictError
from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import (
    RunCancelled as RunCancelledEvent,
    RunFailed as RunFailedEvent,
    RunResumed as RunResumedEvent,
    RunStarted as RunStartedEvent,
)
from linktools.ai.run.commit import (
    AcknowledgeCancelRunCommand,
    FailRunCommand,
    RequestCancelRunCommand,
    ResumeRunCommand,
    StartRunCommand,
    ExecutionFence,
    RunCommitId,
)
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import (
    RunErrorInfo,
    RunInput,
    RunnableType,
    RunRecord,
    RunStatus,
)
from linktools.ai.runtime.persistence import SqlAlchemyStorage
from linktools.ai.run.persistence.sqlalchemy.commit import SqlAlchemyRunCommitCoordinator
from linktools.ai.storage.sqlalchemy.models import Base


def _storage(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/commit.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/commit.db")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyStorage(
        session_factory=session_factory, blobs_root=tmp_path / "blobs"
    )


def _record(run_id, session_id, status, version=1, **kw) -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        status=status,
        input=RunInput(prompt="hi"),
        result=None,
        error=None,
        version=version,
        created_at=now,
        started_at=now,
        finished_at=None,
        **kw,
    )


def _seed(storage, run_id, session_id, status=RunStatus.RUNNING, **kw):
    asyncio.run(storage.runs.create(_record(run_id, session_id, status, **kw)))


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


def test_start_creates_running_record_and_started_event(tmp_path):
    storage = _storage(tmp_path)

    async def _run():
        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        record = _record("run-s1", "sess-s1", RunStatus.RUNNING)
        commit = await coordinator.start(
            StartRunCommand(
                record=record,
                started_event=RunStartedEvent(run_id="run-s1", runnable_id="agent-1"),
                event_context=EventStreamContext.from_run_context(
                    _ctx("run-s1", "sess-s1")
                ),
                commit_id=RunCommitId(f"start:rec"),)
        )
        assert commit.record.status is RunStatus.RUNNING
        persisted = await storage.runs.get("run-s1")
        assert persisted is not None and persisted.status is RunStatus.RUNNING
        page = await storage.events.list("run-s1", limit=100)
        types = [type(e.payload).__name__ for e in page.items]
        assert types.count("RunStarted") == 1

    asyncio.run(_run())


def test_resume_commits_atomically_running(tmp_path):
    storage = _storage(tmp_path)
    _seed(storage, "run-r1", "sess-r1", status=RunStatus.WAITING_APPROVAL)

    async def _run():
        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        commit = await coordinator.resume(
            ResumeRunCommand(
                run_id="run-r1",
                expected_version=1,
                approval_id="appr-1",
                resumed_event=RunResumedEvent(run_id="run-r1"),
                event_context=EventStreamContext.from_run_context(
                    _ctx("run-r1", "sess-r1")
                ),
                commit_id=RunCommitId(f"resume:run-r1"),)
        )
        assert commit.run_id == "run-r1"
        record = await storage.runs.get("run-r1")
        assert record.status is RunStatus.RUNNING
        page = await storage.events.list("run-r1", limit=100)
        types = [type(e.payload).__name__ for e in page.items]
        assert types.count("RunResumed") == 1

    asyncio.run(_run())


def test_fail_commits_atomically_failed_with_error_and_event(tmp_path):
    storage = _storage(tmp_path)
    _seed(storage, "run-f1", "sess-f1")

    async def _run():
        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        commit = await coordinator.fail(
            FailRunCommand(
                run_id="run-f1",
                expected_version=1,
                execution_fence=None,
                error=RunErrorInfo(error_type="RuntimeError", message="boom"),
                failed_event=RunFailedEvent(
                    run_id="run-f1", error_type="RuntimeError", message="boom"
                ),
                event_context=EventStreamContext.from_run_context(
                    _ctx("run-f1", "sess-f1")
                ),
                commit_id=RunCommitId(f"fail:run-f1"),)
        )
        assert commit.run_id == "run-f1"
        record = await storage.runs.get("run-f1")
        assert record.status is RunStatus.FAILED
        assert record.error is not None and record.error.error_type == "RuntimeError"
        page = await storage.events.list("run-f1", limit=100)
        types = [type(e.payload).__name__ for e in page.items]
        assert types.count("RunFailed") == 1

    asyncio.run(_run())


def test_fail_rejects_stale_execution_token(tmp_path):
    """A worker whose lease was reclaimed (a different execution_token is now
    the run's current claim) must not be able to commit a terminal fail on
    the run's behalf -- the fencing check rejects it, and nothing in the
    same transaction (the FAILED transition, the RunFailed event) commits."""
    storage = _storage(tmp_path)
    _seed(storage, "run-f2", "sess-f2", execution_token="current-token")

    async def _run():
        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        with pytest.raises(RunConflictError):
            await coordinator.fail(
                FailRunCommand(
                    run_id="run-f2",
                    expected_version=1,
                    execution_fence=ExecutionFence("stale-token"),
                    error=RunErrorInfo(error_type="RuntimeError", message="boom"),
                    failed_event=RunFailedEvent(
                        run_id="run-f2", error_type="RuntimeError", message="boom"
                    ),
                    event_context=EventStreamContext.from_run_context(
                        _ctx("run-f2", "sess-f2")
                    ),
                    commit_id=RunCommitId(f"fail:run-f2"),)
            )
        record = await storage.runs.get("run-f2")
        assert record.status is RunStatus.RUNNING
        page = await storage.events.list("run-f2", limit=100)
        assert not page.items

    asyncio.run(_run())


def test_request_cancel_commits_atomically_cancelling(tmp_path):
    storage = _storage(tmp_path)
    _seed(storage, "run-c1", "sess-c1")

    async def _run():
        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        commit = await coordinator.request_cancel(
            RequestCancelRunCommand(
                run_id="run-c1",
                expected_version=1,
                requested_by="user-1",
                reason="no longer needed",
                event_context=EventStreamContext.from_run_context(
                    _ctx("run-c1", "sess-c1")
                ),
                commit_id=RunCommitId(f"request-cancel:run-c1"),)
        )
        assert commit.run_id == "run-c1"
        record = await storage.runs.get("run-c1")
        assert record.status is RunStatus.CANCELLING
        assert record.cancel_requested_by == "user-1"
        assert record.cancel_reason == "no longer needed"

    asyncio.run(_run())


def test_acknowledge_cancel_commits_atomically_cancelled(tmp_path):
    storage = _storage(tmp_path)
    _seed(storage, "run-a1", "sess-a1", status=RunStatus.CANCELLING)

    async def _run():
        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        commit = await coordinator.acknowledge_cancel(
            AcknowledgeCancelRunCommand(
                run_id="run-a1",
                expected_version=1,
                execution_fence=None,
                cancelled_event=RunCancelledEvent(run_id="run-a1", reason="stopped"),
                event_context=EventStreamContext.from_run_context(
                    _ctx("run-a1", "sess-a1")
                ),
                commit_id=RunCommitId(f"ack-cancel:run-a1"),)
        )
        assert commit.run_id == "run-a1"
        record = await storage.runs.get("run-a1")
        assert record.status is RunStatus.CANCELLED
        page = await storage.events.list("run-a1", limit=100)
        types = [type(e.payload).__name__ for e in page.items]
        assert types.count("RunCancelled") == 1

    asyncio.run(_run())
