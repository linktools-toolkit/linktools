#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/filesystem/test_commit_lifecycle.py

FilesystemRunCommitCoordinator's start/resume/fail/request_cancel/
acknowledge_cancel -- the 5 commands added alongside the pre-existing pause/
complete to close out the full RunCommitCoordinator Protocol."""

import asyncio
from datetime import datetime, timezone

from linktools.ai.run.commit import RunFenceLostError
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
    RunCommitPolicy,
)
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunErrorInfo, RunInput, RunRecord, RunStatus
from linktools.ai.run.models import RunnableType
from linktools.ai.run.persistence.codec import RunCommitCodec
from linktools.ai.runtime.persistence.facade import FilesystemStorage
from linktools.ai.run.persistence.commit import FilesystemRunCommitCoordinator


def _record(run_id: str, session_id: str, status: RunStatus, version: int, **kw) -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        status=status,
        input=RunInput(prompt="hello"),
        result=None,
        error=None,
        version=version,
        created_at=now,
        started_at=now if status is not RunStatus.PENDING else None,
        finished_at=None,
        **kw,
    )


def _context(run_id: str, session_id: str) -> RunContext:
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


def _coordinator(storage: FilesystemStorage) -> FilesystemRunCommitCoordinator:
    return FilesystemRunCommitCoordinator(
        approval_store=storage.approvals,
        checkpoint_store=storage.checkpoints,
        run_store=storage.runs,
        session_store=storage.sessions,
        event_store=storage.events,
        policy=RunCommitPolicy(fencing_required=False),
        codec=RunCommitCodec(),
    )


def test_start_creates_running_record_and_started_event(tmp_path):
    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        coordinator = _coordinator(storage)
        record = _record("run-s1", "sess-s1", RunStatus.RUNNING, 1)
        ctx = EventStreamContext.from_run_context(_context("run-s1", "sess-s1"))

        commit = await coordinator.start(
            StartRunCommand(
                record=record,
                started_event=RunStartedEvent(run_id="run-s1", runnable_id="agent-1"),
                event_context=ctx,
                commit_id=RunCommitId(f"start:rec"),)
        )
        assert commit.record.status is RunStatus.RUNNING

        persisted = await storage.runs.get("run-s1")
        assert persisted is not None
        assert persisted.status is RunStatus.RUNNING

        events = await storage.events.list("run-s1")
        assert any(
            type(e.payload).__name__ == "RunStarted" for e in events.items
        )

    asyncio.run(_run())


def test_start_is_idempotent_on_retry(tmp_path):
    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        coordinator = _coordinator(storage)
        record = _record("run-s2", "sess-s2", RunStatus.RUNNING, 1)
        ctx = EventStreamContext.from_run_context(_context("run-s2", "sess-s2"))
        command = StartRunCommand(
            record=record,
            started_event=RunStartedEvent(run_id="run-s2", runnable_id="agent-1"),
            event_context=ctx,
            commit_id=RunCommitId("start:run-s2"),)

        first = await coordinator.start(command)
        second = await coordinator.start(command)
        assert first.record.id == second.record.id

        events = await storage.events.list("run-s2")
        started = [e for e in events.items if type(e.payload).__name__ == "RunStarted"]
        assert len(started) == 1

    asyncio.run(_run())


def test_resume_transitions_waiting_approval_to_running(tmp_path):
    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        coordinator = _coordinator(storage)
        await storage.runs.create(
            _record("run-r1", "sess-r1", RunStatus.WAITING_APPROVAL, 1)
        )
        ctx = EventStreamContext.from_run_context(_context("run-r1", "sess-r1"))

        commit = await coordinator.resume(
            ResumeRunCommand(
                run_id="run-r1",
                expected_version=1,
                approval_id="appr-1",
                resumed_event=RunResumedEvent(run_id="run-r1"),
                event_context=ctx,
                commit_id=RunCommitId(f"resume:run-r1"),)
        )
        assert commit.run_id == "run-r1"

        record = await storage.runs.get("run-r1")
        assert record.status is RunStatus.RUNNING

        events = await storage.events.list("run-r1")
        assert any(type(e.payload).__name__ == "RunResumed" for e in events.items)

    asyncio.run(_run())


def test_fail_transitions_to_failed_with_error_and_event(tmp_path):
    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        coordinator = _coordinator(storage)
        await storage.runs.create(_record("run-f1", "sess-f1", RunStatus.RUNNING, 1))
        ctx = EventStreamContext.from_run_context(_context("run-f1", "sess-f1"))
        error = RunErrorInfo(error_type="RuntimeError", message="boom")

        commit = await coordinator.fail(
            FailRunCommand(
                run_id="run-f1",
                expected_version=1,
                execution_fence=None,
                error=error,
                failed_event=RunFailedEvent(
                    run_id="run-f1", error_type="RuntimeError", message="boom"
                ),
                event_context=ctx,
                commit_id=RunCommitId(f"fail:run-f1"),)
        )
        assert commit.run_id == "run-f1"

        record = await storage.runs.get("run-f1")
        assert record.status is RunStatus.FAILED
        assert record.error is not None
        assert record.error.error_type == "RuntimeError"

        events = await storage.events.list("run-f1")
        assert any(type(e.payload).__name__ == "RunFailed" for e in events.items)

    asyncio.run(_run())


def test_fail_rejects_stale_execution_token(tmp_path):
    """A worker whose lease was reclaimed (a different execution_token is now
    the run's current claim) must not be able to commit a terminal fail on
    the run's behalf -- the fencing check rejects it."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            policy=RunCommitPolicy(fencing_required=True),
            codec=RunCommitCodec(),
        )
        await storage.runs.create(
            _record(
                "run-f2", "sess-f2", RunStatus.RUNNING, 1, execution_token="current-token"
            )
        )
        ctx = EventStreamContext.from_run_context(_context("run-f2", "sess-f2"))

        try:
            await coordinator.fail(
                FailRunCommand(
                    run_id="run-f2",
                    expected_version=1,
                    execution_fence=ExecutionFence("stale-token"),
                    error=RunErrorInfo(error_type="RuntimeError", message="boom"),
                    failed_event=RunFailedEvent(
                        run_id="run-f2", error_type="RuntimeError", message="boom"
                    ),
                    event_context=ctx,
                    commit_id=RunCommitId(f"fail:run-f2"),)
            )
            raised = False
        except RunFenceLostError:
            raised = True
        assert raised

        # The run was NOT transitioned by the rejected commit.
        record = await storage.runs.get("run-f2")
        assert record.status is RunStatus.RUNNING

    asyncio.run(_run())


def test_request_cancel_transitions_to_cancelling(tmp_path):
    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        coordinator = _coordinator(storage)
        await storage.runs.create(_record("run-c1", "sess-c1", RunStatus.RUNNING, 1))
        ctx = EventStreamContext.from_run_context(_context("run-c1", "sess-c1"))

        commit = await coordinator.request_cancel(
            RequestCancelRunCommand(
                run_id="run-c1",
                expected_version=1,
                requested_by="user-1",
                reason="no longer needed",
                event_context=ctx,
                commit_id=RunCommitId(f"request-cancel:run-c1"),)
        )
        assert commit.run_id == "run-c1"

        record = await storage.runs.get("run-c1")
        assert record.status is RunStatus.CANCELLING
        assert record.cancel_requested_by == "user-1"
        assert record.cancel_reason == "no longer needed"

    asyncio.run(_run())


def test_request_cancel_recovery_does_not_fail_the_run(tmp_path):
    """Unlike every other commit kind, an INCOMPLETE request_cancel must NOT
    be fail-closed on recovery -- the run's own execution is undisturbed by a
    crashed cancel request, so recovery discards the journal and leaves the
    run's actual RUNNING state alone."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        from linktools.ai.run.persistence.journal import (
            TransactionJournal,
            TransactionKind,
        )

        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            policy=RunCommitPolicy(fencing_required=False),
            codec=RunCommitCodec(),
            transactions_root=tmp_path / "transactions",
        )
        await storage.runs.create(_record("run-c2", "sess-c2", RunStatus.RUNNING, 1))
        journal = TransactionJournal(tmp_path / "transactions")
        journal.begin(
            kind=TransactionKind.REQUEST_CANCEL,
            run_id="run-c2",
            target_run_status="cancelling",
            command={},
            commit_id="recovery:tx:run-c2",
            request_hash="",
            command_payload=b"{}",
        )

        await coordinator.recover_incomplete_commits()

        record = await storage.runs.get("run-c2")
        # NOT failed -- the run's own execution was never disturbed.
        assert record.status is RunStatus.RUNNING
        assert not journal.list_incomplete()

    asyncio.run(_run())


def test_acknowledge_cancel_transitions_cancelling_to_cancelled(tmp_path):
    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        coordinator = _coordinator(storage)
        await storage.runs.create(_record("run-a1", "sess-a1", RunStatus.CANCELLING, 1))
        ctx = EventStreamContext.from_run_context(_context("run-a1", "sess-a1"))

        commit = await coordinator.acknowledge_cancel(
            AcknowledgeCancelRunCommand(
                run_id="run-a1",
                expected_version=1,
                execution_fence=None,
                cancelled_event=RunCancelledEvent(run_id="run-a1", reason="stopped"),
                event_context=ctx,
                commit_id=RunCommitId(f"ack-cancel:run-a1"),)
        )
        assert commit.run_id == "run-a1"

        record = await storage.runs.get("run-a1")
        assert record.status is RunStatus.CANCELLED

        events = await storage.events.list("run-a1")
        assert any(type(e.payload).__name__ == "RunCancelled" for e in events.items)

    asyncio.run(_run())


def test_recovery_reappends_started_event_when_start_reached_commit_point(tmp_path):
    """A crash leaves an incomplete START journal but the run record DID
    land (the commit point). Recovery re-publishes the RunStarted event and
    discards the journal, without re-creating the record."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        from linktools.ai.run.persistence.journal import (
            TransactionJournal,
            TransactionKind,
        )

        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            policy=RunCommitPolicy(fencing_required=False),
            codec=RunCommitCodec(),
            transactions_root=tmp_path / "transactions",
        )
        await storage.runs.create(_record("run-s3", "sess-s3", RunStatus.RUNNING, 1))
        journal = TransactionJournal(tmp_path / "transactions")
        ctx = EventStreamContext.from_run_context(_context("run-s3", "sess-s3"))
        recovery_command = StartRunCommand(
            record=_record("run-s3", "sess-s3", RunStatus.RUNNING, 1),
            started_event=RunStartedEvent(run_id="run-s3", runnable_id="agent-1"),
            event_context=ctx,
            commit_id=RunCommitId("recovery:tx:run-s3"),
        )
        journal.begin(
            kind=TransactionKind.START,
            run_id="run-s3",
            target_run_status="running",
            command={},
            commit_id="recovery:tx:run-s3",
            request_hash="",
            command_payload=RunCommitCodec().encode_request("start", recovery_command),
        )

        await coordinator.recover_incomplete_commits()

        events = await storage.events.list("run-s3")
        assert any(type(e.payload).__name__ == "RunStarted" for e in events.items)
        assert not journal.list_incomplete()

    asyncio.run(_run())
