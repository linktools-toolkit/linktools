#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/filesystem/test_commit.py — FilesystemRunCommitCoordinator contract.

: the coordinator's ``complete()`` path imports ``mark_completed`` from
``...run.lifecycle``. Before the fix the relative import had the wrong depth
(``..run.lifecycle`` resolves to ``storage.run.lifecycle``, which does not
exist), so the ImportError only surfaced at run-completion time -- after the
Session and Checkpoint had already been written. These tests drive ``pause()``
and ``complete()`` directly so the import is exercised at test time."""

import asyncio
from datetime import datetime, timezone

from linktools.ai.events.context import EventContext
from linktools.ai.run.commit import CompleteRunCommand, PauseRunCommand
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunRecord, RunResult, RunStatus
from linktools.ai.run.models import RunnableType
from linktools.ai.session.models import MessageRole, NewSessionMessage
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.facade import FilesystemStorage


# The approval store enforces a complete execution binding as a security
# invariant: every persisted approval must bind the tool
# descriptor + handler/provider/policy/capability/result-processor revisions
# and the arguments hash. Production code fills these via the governed tool
# executor; storage-level commit tests synthesize the request directly, so
# they carry a fixed binding to satisfy the store-layer check.
_APPROVAL_BINDING = {
    "descriptor_fingerprint": "fp-test",
    "handler_revision": "h1",
    "provider_revision": "p1",
    "policy_revision": "pol1",
    "capability_revision": "cap1",
    "result_processor_revision": "rp1",
    "arguments_hash": "ah1",
}


def _record(run_id: str, session_id: str, status: RunStatus, version: int) -> RunRecord:
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
        started_at=now,
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


def test_complete_imports_mark_completed_and_transitions_succeeded(tmp_path):
    """complete() must import mark_completed (the bug) and transition the
    run to SUCCEEDED, persisting exactly one session turn + one checkpoint."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="sess-1",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        created = await storage.runs.create(
            _record("run-1", "sess-1", RunStatus.RUNNING, 1)
        )
        from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
        )
        result = RunResult(output={"response": {"message": "done"}})
        commit = await coordinator.complete(
            CompleteRunCommand(
                run_id="run-1",
                session_id="sess-1",
                expected_version=created.version,
                messages=(
                    NewSessionMessage(
                        role=MessageRole.USER,
                        content="hello",
                        run_id="run-1",
                    ),
                    NewSessionMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"response": {"message": "done"}}',
                        run_id="run-1",
                    ),
                ),
                checkpoint_payload=b'{"messages": []}',
                result=result,
                event_context=EventContext.from_run_context(
                    _context("run-1", "sess-1")
                ),
            )
        )
        assert commit.result is result
        final = await storage.runs.get("run-1")
        assert final is not None
        assert final.status is RunStatus.SUCCEEDED
        # Exactly one USER + one ASSISTANT + one checkpoint for a single complete.
        messages = await storage.sessions.list_messages("sess-1")
        assert sum(1 for m in messages if m.role is MessageRole.USER) == 1
        assert sum(1 for m in messages if m.role is MessageRole.ASSISTANT) == 1
        # Exactly one checkpoint written by this complete() (sequence starts at 1).
        latest = await storage.checkpoints.latest("run-1")
        assert latest is not None
        assert latest.sequence == 1

    asyncio.run(_run())


def test_pause_persists_approval_checkpoint_and_transition(tmp_path):
    """pause() persists the approval request, a checkpoint, and transitions the
    run to WAITING_APPROVAL."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        created = await storage.runs.create(
            _record("run-2", "sess-2", RunStatus.RUNNING, 1)
        )
        from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
        )
        commit = await coordinator.pause(
            PauseRunCommand(
                run_id="run-2",
                expected_version=created.version,
                approval_request={
                    "approval_id": "appr-1",
                    "tool_call_id": "call-1",
                    "tool_name": "shell",
                    "reason": "needs review",
                    "arguments": {"cmd": "ls"},
                    **_APPROVAL_BINDING,
                },
                checkpoint_payload=b'{"messages": []}',
                event_context=EventContext.from_run_context(
                    _context("run-2", "sess-2")
                ),
            )
        )
        assert commit.checkpoint_id
        final = await storage.runs.get("run-2")
        assert final is not None
        assert final.status is RunStatus.WAITING_APPROVAL
        approval = await storage.approvals.get(commit.approval_id)
        assert approval is not None
        assert approval.run_id == "run-2"

    asyncio.run(_run())


def test_complete_journal_is_discarded_on_success(tmp_path):
    """A successful complete() leaves no incomplete journal (it advanced to
    COMMITTED and was deleted), so recovery has nothing to do."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="sess-j",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        await storage.runs.create(_record("run-j", "sess-j", RunStatus.RUNNING, 1))
        from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            transactions_root=tmp_path / "transactions",
        )
        await coordinator.complete(
            CompleteRunCommand(
                run_id="run-j",
                session_id="sess-j",
                expected_version=1,
                messages=(
                    NewSessionMessage(
                        role=MessageRole.USER, content="hi", run_id="run-j"
                    ),
                    NewSessionMessage(
                        role=MessageRole.ASSISTANT, content="ok", run_id="run-j"
                    ),
                ),
                checkpoint_payload=b'{"m":[]}',
                result=RunResult(output="ok"),
                event_context=EventContext.from_run_context(
                    _context("run-j", "sess-j")
                ),
            )
        )
        # No leftover journal files (COMMITTED journals are deleted).
        assert not list((tmp_path / "transactions").glob("*.json"))

    asyncio.run(_run())


def test_recovery_marks_run_failed_when_complete_did_not_reach_commit_point(tmp_path):
    """failure injection: a crash leaves an incomplete COMPLETE journal and
    the run never reached SUCCEEDED. Recovery must mark the run FAILED
    (fail-closed) so the orphan session/checkpoint writes are not surfaced as a
    successful run, then discard the journal."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
        from linktools.ai.storage.filesystem.journal import (
            TransactionJournal,
            TransactionKind,
        )

        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            transactions_root=tmp_path / "transactions",
        )
        await storage.runs.create(_record("run-c", "sess-c", RunStatus.RUNNING, 1))
        # Simulate a crash AFTER SESSION_WRITTEN but BEFORE the SUCCEEDED
        # transition: hand-write an incomplete journal at that state.
        journal = TransactionJournal(tmp_path / "transactions")
        journal.begin(
            kind=TransactionKind.COMPLETE,
            run_id="run-c",
            target_run_status="succeeded",
            command={"event_context": {}},
        )
        # (the journal file is now PREPARED -- the lowest state.)
        assert journal.list_incomplete()

        await coordinator.recover_incomplete_commits()

        # Run was NOT SUCCEEDED -> recovery marked it FAILED.
        record = await storage.runs.get("run-c")
        assert record.status is RunStatus.FAILED
        # The journal was resolved (discarded).
        assert not journal.list_incomplete()

    asyncio.run(_run())


def test_recovery_completes_when_pause_reached_commit_point(tmp_path):
    """a crash leaves an incomplete PAUSE journal but the run DID reach
    WAITING_APPROVAL (the commit point). Recovery treats it as durable: the run
    stays WAITING_APPROVAL and the journal is discarded (best-effort events
    re-appended)."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
        from linktools.ai.storage.filesystem.journal import (
            TransactionJournal,
            TransactionKind,
        )

        coordinator = FilesystemRunCommitCoordinator(
            approval_store=storage.approvals,
            checkpoint_store=storage.checkpoints,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            transactions_root=tmp_path / "transactions",
        )
        # Run is already WAITING_APPROVAL (the transition landed before crash).
        await storage.runs.create(
            _record("run-p", "sess-p", RunStatus.WAITING_APPROVAL, 1)
        )
        journal = TransactionJournal(tmp_path / "transactions")
        journal.begin(
            kind=TransactionKind.PAUSE,
            run_id="run-p",
            target_run_status="waiting_approval",
            command={"event_context": {}},
        )

        await coordinator.recover_incomplete_commits()

        record = await storage.runs.get("run-p")
        # Stays at the commit point -- recovery did not FAILED it.
        assert record.status is RunStatus.WAITING_APPROVAL
        assert not journal.list_incomplete()

    asyncio.run(_run())

