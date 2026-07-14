#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Issue 3 (v4 guide §6): the FileRunCommitCoordinator must dedup critical
events by ``commit_id``, not just by event type.

A run may legitimately pause more than once (one approval per pause), each
pause carrying its own commit_id. Deduping by ``(run_id, event_type)`` alone
either drops a later legitimate pause's ApprovalRequested/RunPaused (the second
looks like a duplicate of the first) or, on recovery, duplicates the event
(recovery bypassed the dedup entirely and re-appended unconditionally).

These tests pin the three behaviors the spec states in §6.10-§6.12:

1. Recovery does not duplicate RunCompleted for a complete commit whose event
   is already written (§6.10).
2. Two distinct approvals on the same run each persist their own
   ApprovalRequested + RunPaused, tagged with their own commit_id (§6.11).
3. A retried identical pause command does not add a second event (§6.12).
"""

import asyncio
from datetime import datetime, timezone

from linktools.ai.events.context import EventContext
from linktools.ai.run.commit import CompleteRunCommand, PauseRunCommand
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import (
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
    RunnableType,
)
from linktools.ai.session.models import MessageRole, NewSessionMessage
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.facade import FileStorage


def _record(run_id: str, session_id: str, status: RunStatus, version: int) -> RunRecord:
    return RunRecord(
        id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        status=status,
        input=RunInput(prompt="x"),
        result=None,
        error=None,
        version=version,
        created_at=datetime.now(timezone.utc),
        started_at=None,
        finished_at=None,
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


def _make_coordinator(storage, tmp_path):
    from linktools.ai.storage.file.commit import FileRunCommitCoordinator

    return FileRunCommitCoordinator(
        approval_store=storage.approvals,
        checkpoint_store=storage.checkpoints,
        run_store=storage.runs,
        session_store=storage.sessions,
        event_store=storage.events,
        transactions_root=tmp_path / "transactions",
    )


def _event_ctx(run_id: str, session_id: str) -> EventContext:
    return EventContext.from_run_context(_context(run_id, session_id))


async def _count(storage, run_id: str, payload_type: str) -> int:
    page = await storage.events.list(run_id, after_sequence=0, limit=10000)
    return sum(
        1
        for e in page.items
        if type(e.payload).__name__ == payload_type  # noqa: E721
    )


async def _commit_ids_for(storage, run_id: str, payload_type: str) -> "list[str]":
    page = await storage.events.list(run_id, after_sequence=0, limit=10000)
    return [
        e.metadata.get("commit_id")
        for e in page.items
        if type(e.payload).__name__ == payload_type  # noqa: E721
    ]


def test_recovery_does_not_duplicate_run_completed(tmp_path):
    """§6.10: complete() already wrote RunCompleted; a crash leaves the journal
    at RUN_TRANSITIONED and recovery re-runs. RunCompleted count must stay 1."""

    async def _run():
        storage = FileStorage(root=tmp_path)
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="sess-c",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        await storage.runs.create(_record("run-c", "sess-c", RunStatus.RUNNING, 1))
        coordinator = _make_coordinator(storage, tmp_path)
        await coordinator.complete(
            CompleteRunCommand(
                run_id="run-c",
                session_id="sess-c",
                expected_version=1,
                messages=(
                    NewSessionMessage(
                        role=MessageRole.USER, content="hi", run_id="run-c"
                    ),
                ),
                checkpoint_payload=b'{"m":[]}',
                result=RunResult(output="ok"),
                event_context=_event_ctx("run-c", "sess-c"),
            )
        )
        # RunCompleted was written and tagged with the complete commit_id.
        assert await _count(storage, "run-c", "RunCompleted") == 1
        assert await _commit_ids_for(storage, "run-c", "RunCompleted") == [
            "complete:run-c:1"
        ]

        # Simulate a crash: hand-write an incomplete COMPLETE journal whose
        # commit_id matches the one that already wrote the event. The run is
        # already SUCCEEDED (the commit point), so recovery re-appends critical
        # events best-effort -- and must NOT duplicate RunCompleted.
        from linktools.ai.storage.file.journal import (
            TransactionJournal,
            TransactionKind,
        )

        journal = TransactionJournal(tmp_path / "transactions")
        journal.begin(
            kind=TransactionKind.COMPLETE,
            run_id="run-c",
            target_run_status="succeeded",
            commit_id="complete:run-c:1",
            command={
                "event_context": {
                    "stream_id": "run-c",
                    "run_id": "run-c",
                    "root_run_id": "run-c",
                    "parent_run_id": None,
                    "session_id": "sess-c",
                    "runnable_id": "agent-1",
                }
            },
        )
        await coordinator.recover_incomplete_commits()

        assert await _count(storage, "run-c", "RunCompleted") == 1
        assert not journal.list_incomplete()

    asyncio.run(_run())


def test_two_distinct_approvals_each_persist_their_events(tmp_path):
    """§6.11: a run pauses for approval, is resumed, then pauses again for a
    second approval. Each pause has its own commit_id and must keep its own
    ApprovalRequested + RunPaused -- dedup by event type alone would drop the
    second pause's events."""

    async def _run():
        storage = FileStorage(root=tmp_path)
        await storage.runs.create(_record("run-d", "sess-d", RunStatus.RUNNING, 1))
        coordinator = _make_coordinator(storage, tmp_path)

        async def _pause(commit_id, approval_id, tool_call_id, expected_version):
            return await coordinator.pause(
                PauseRunCommand(
                    run_id="run-d",
                    expected_version=expected_version,
                    approval_request={
                        "approval_id": approval_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": "shell",
                        "reason": "needs review",
                        "arguments": {"cmd": "ls"},
                    },
                    checkpoint_payload=b'{"m":[]}',
                    event_context=_event_ctx("run-d", "sess-d"),
                    commit_id=commit_id,
                )
            )

        # First pause -> commit-a.
        await _pause("commit-a", "appr-1", "call-1", expected_version=1)
        # Simulate the resume transition (WAITING_APPROVAL -> RUNNING).
        await storage.runs.transition("run-d", RunStatus.RUNNING, expected_version=2)
        # Second pause -> commit-b.
        await _pause("commit-b", "appr-2", "call-2", expected_version=3)

        assert await _count(storage, "run-d", "ApprovalRequested") == 2
        assert await _count(storage, "run-d", "RunPaused") == 2
        # Each event carries its own commit_id.
        assert sorted(await _commit_ids_for(storage, "run-d", "ApprovalRequested")) == [
            "commit-a",
            "commit-b",
        ]
        assert sorted(await _commit_ids_for(storage, "run-d", "RunPaused")) == [
            "commit-a",
            "commit-b",
        ]

    asyncio.run(_run())


def test_retried_identical_pause_command_does_not_duplicate(tmp_path):
    """§6.12: the caller retries the SAME pause command (same commit_id) because
    it missed the first response. Only one ApprovalRequested + one RunPaused may
    exist for that commit."""

    async def _run():
        storage = FileStorage(root=tmp_path)
        await storage.runs.create(_record("run-e", "sess-e", RunStatus.RUNNING, 1))
        coordinator = _make_coordinator(storage, tmp_path)

        async def _pause():
            return await coordinator.pause(
                PauseRunCommand(
                    run_id="run-e",
                    expected_version=1,
                    approval_request={
                        "approval_id": "appr-1",
                        "tool_call_id": "call-1",
                        "tool_name": "shell",
                        "reason": "needs review",
                        "arguments": {"cmd": "ls"},
                    },
                    checkpoint_payload=b'{"m":[]}',
                    event_context=_event_ctx("run-e", "sess-e"),
                    commit_id="commit-same",
                )
            )

        first = await _pause()
        # The run is now WAITING_APPROVAL; the caller retries the identical
        # command. The coordinator must return the prior result without writing
        # a second event.
        second = await _pause()
        assert second.approval_id == first.approval_id

        assert await _count(storage, "run-e", "ApprovalRequested") == 1
        assert await _count(storage, "run-e", "RunPaused") == 1

    asyncio.run(_run())
