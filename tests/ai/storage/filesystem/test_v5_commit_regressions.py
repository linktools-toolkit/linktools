#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""BUG-02 (v5 guide §9): a File complete commit must publish session messages
only AFTER the run reaches SUCCEEDED (the commit point).

Before the reorder the session messages were written first, so a crash before
the run transition left a FAILED run whose answer already polluted the
conversation history. Now the order is checkpoint -> SUCCEEDED -> publish
messages -> event; a pre-commit crash publishes nothing, and a post-commit
crash is closed by recovery republishing from the journal (idempotent via
commit_id)."""

import asyncio
from datetime import datetime, timezone

from linktools.ai.events.context import EventContext
from linktools.ai.run.commit import CompleteRunCommand
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import (
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
    RunnableType,
)
from linktools.ai.session.models import (
    MessageRole,
    NewSessionMessage,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.journal import TransactionJournal, TransactionKind


def _record(run_id, session_id, status, version, result=None):
    return RunRecord(
        id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        status=status,
        input=RunInput(prompt="x"),
        result=result,
        error=None,
        version=version,
        created_at=datetime.now(timezone.utc),
        started_at=None,
        finished_at=None,
    )


def _ctx(run_id, session_id):
    return EventContext.from_run_context(
        RunContext(
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
    )


def _coordinator(storage, tmp_path):
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    return FilesystemRunCommitCoordinator(
        approval_store=storage.approvals,
        checkpoint_store=storage.checkpoints,
        run_store=storage.runs,
        session_store=storage.sessions,
        event_store=storage.events,
        transactions_root=tmp_path / "transactions",
    )


def _msg(role, content, run_id):
    return {"role": role.value, "content": content, "run_id": run_id, "metadata": {}}


def _seed_session(storage, session_id):
    asyncio.run(
        storage.sessions.create(
            SessionRecord(
                id=session_id,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
    )


def test_normal_complete_publishes_session_messages(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    asyncio.run(storage.runs.create(_record("run-h", "sess-h", RunStatus.RUNNING, 1)))
    _seed_session(storage, "sess-h")
    coordinator = _coordinator(storage, tmp_path)

    asyncio.run(
        coordinator.complete(
            CompleteRunCommand(
                run_id="run-h",
                session_id="sess-h",
                expected_version=1,
                messages=(
                    NewSessionMessage(
                        role=MessageRole.USER, content="hi", run_id="run-h"
                    ),
                    NewSessionMessage(
                        role=MessageRole.ASSISTANT, content="answer", run_id="run-h"
                    ),
                ),
                checkpoint_payload=b'{"m":[]}',
                result=RunResult(output="answer"),
                event_context=_ctx("run-h", "sess-h"),
            )
        )
    )

    run = asyncio.run(storage.runs.get("run-h"))
    assert run.status is RunStatus.SUCCEEDED
    msgs = asyncio.run(storage.sessions.list_messages("sess-h"))
    assert [m.role for m in msgs] == [MessageRole.USER, MessageRole.ASSISTANT]


def test_pre_commit_crash_publishes_no_session_messages(tmp_path):
    """A crash before the SUCCEEDED transition (journal at PREPARED, run still
    RUNNING) -> recovery marks the run FAILED and the session history stays
    clean -- the failed answer never lands in the conversation."""
    storage = FilesystemStorage(root=tmp_path)
    asyncio.run(storage.runs.create(_record("run-p", "sess-p", RunStatus.RUNNING, 1)))
    _seed_session(storage, "sess-p")
    coordinator = _coordinator(storage, tmp_path)

    journal = TransactionJournal(tmp_path / "transactions")
    journal.begin(
        kind=TransactionKind.COMPLETE,
        run_id="run-p",
        target_run_status="succeeded",
        commit_id="complete:run-p:1",
        command={
            "session_id": "sess-p",
            "messages": [
                _msg(MessageRole.USER, "hi", "run-p"),
                _msg(MessageRole.ASSISTANT, "should-not-leak", "run-p"),
            ],
            "event_context": {},
        },
    )

    asyncio.run(coordinator.recover_incomplete_commits())

    run = asyncio.run(storage.runs.get("run-p"))
    assert run.status is RunStatus.FAILED
    msgs = asyncio.run(storage.sessions.list_messages("sess-p"))
    assert not msgs, "a pre-commit crash must publish no session messages"


def test_post_commit_crash_recovery_republishes_messages(tmp_path):
    """A crash AFTER the SUCCEEDED transition (run already SUCCEEDED, messages
    not yet written) -> recovery republishes exactly USER + ASSISTANT from the
    journal, no duplicates."""
    storage = FilesystemStorage(root=tmp_path)
    asyncio.run(
        storage.runs.create(
            _record(
                "run-c", "sess-c", RunStatus.SUCCEEDED, 1, result=RunResult(output="ok")
            )
        )
    )
    _seed_session(storage, "sess-c")
    coordinator = _coordinator(storage, tmp_path)

    journal = TransactionJournal(tmp_path / "transactions")
    journal.begin(
        kind=TransactionKind.COMPLETE,
        run_id="run-c",
        target_run_status="succeeded",
        commit_id="complete:run-c:1",
        command={
            "session_id": "sess-c",
            "messages": [
                _msg(MessageRole.USER, "hi", "run-c"),
                _msg(MessageRole.ASSISTANT, "answer", "run-c"),
            ],
            "event_context": {
                "stream_id": "run-c",
                "run_id": "run-c",
                "root_run_id": "run-c",
                "parent_run_id": None,
                "session_id": "sess-c",
                "runnable_id": "agent-1",
            },
        },
    )

    asyncio.run(coordinator.recover_incomplete_commits())

    run = asyncio.run(storage.runs.get("run-c"))
    assert run.status is RunStatus.SUCCEEDED
    msgs = asyncio.run(storage.sessions.list_messages("sess-c"))
    assert [m.role for m in msgs] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert [m.content for m in msgs] == ["hi", "answer"]
    # Recovering again (journal already discarded) adds no duplicates.
    asyncio.run(coordinator.recover_incomplete_commits())
    msgs2 = asyncio.run(storage.sessions.list_messages("sess-c"))
    assert len(msgs2) == 2


class _FlakyEvents:
    """Wraps a real EventStore but makes the first ``fail_times`` append calls
    raise, so a critical-event write failure (and its retry) is deterministic."""

    def __init__(self, real, fail_times: int) -> None:
        self._real = real
        self._fail_times = fail_times
        self.append_calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    async def append(self, **kwargs):
        self.append_calls += 1
        if self.append_calls <= self._fail_times:
            raise RuntimeError("injected event store failure")
        return await self._real.append(**kwargs)


def _count_events(storage, run_id, payload_type) -> int:
    page = asyncio.run(storage.events.list(run_id, after_sequence=0, limit=10000))
    return sum(1 for e in page.items if type(e.payload).__name__ == payload_type)


def test_complete_event_failure_retains_journal_then_retry_writes_once(tmp_path):
    """BUG-03: a critical-event write failure during complete() must NOT be
    swallowed -- the journal stays un-committed, the caller sees the error, and
    a retry writes the RunCompleted event exactly once."""
    import pytest

    from linktools.ai.run.commit import CompleteRunCommand

    storage = FilesystemStorage(root=tmp_path)
    asyncio.run(storage.runs.create(_record("run-e", "sess-e", RunStatus.RUNNING, 1)))
    _seed_session(storage, "sess-e")
    coordinator = _coordinator(storage, tmp_path)
    coordinator._events = _FlakyEvents(storage.events, fail_times=1)

    cmd = CompleteRunCommand(
        run_id="run-e",
        session_id="sess-e",
        expected_version=1,
        messages=(
            NewSessionMessage(role=MessageRole.USER, content="hi", run_id="run-e"),
            NewSessionMessage(
                role=MessageRole.ASSISTANT, content="answer", run_id="run-e"
            ),
        ),
        checkpoint_payload=b'{"m":[]}',
        result=RunResult(output="answer"),
        event_context=_ctx("run-e", "sess-e"),
    )

    with pytest.raises(RuntimeError):
        asyncio.run(coordinator.complete(cmd))
    journal = TransactionJournal(tmp_path / "transactions")
    assert journal.list_incomplete(), "journal must be retained on event failure"

    # Retry against the healthy store: event lands exactly once.
    coordinator._events = storage.events
    asyncio.run(coordinator.complete(cmd))
    assert _count_events(storage, "run-e", "RunCompleted") == 1
    assert not journal.list_incomplete()


def test_recovery_event_failure_retains_journal_for_retry(tmp_path):
    """BUG-03: when recovery cannot re-append the critical event, the journal is
    retained (not discarded) so the next recovery reatries -- the event is not
    permanently lost."""
    storage = FilesystemStorage(root=tmp_path)
    asyncio.run(
        storage.runs.create(
            _record(
                "run-r", "sess-r", RunStatus.SUCCEEDED, 1, result=RunResult(output="ok")
            )
        )
    )
    _seed_session(storage, "sess-r")
    coordinator = _coordinator(storage, tmp_path)

    journal = TransactionJournal(tmp_path / "transactions")
    journal.begin(
        kind=TransactionKind.COMPLETE,
        run_id="run-r",
        target_run_status="succeeded",
        commit_id="complete:run-r:1",
        command={
            "session_id": "sess-r",
            "messages": [_msg(MessageRole.ASSISTANT, "answer", "run-r")],
            "event_context": {
                "stream_id": "run-r",
                "run_id": "run-r",
                "root_run_id": "run-r",
                "parent_run_id": None,
                "session_id": "sess-r",
                "runnable_id": "agent-1",
            },
        },
    )

    coordinator._events = _FlakyEvents(storage.events, fail_times=1)
    asyncio.run(coordinator.recover_incomplete_commits())
    assert journal.list_incomplete(), "journal retained when event recovery fails"

    coordinator._events = storage.events
    asyncio.run(coordinator.recover_incomplete_commits())
    assert not journal.list_incomplete(), "journal discarded once the event lands"
    assert _count_events(storage, "run-r", "RunCompleted") == 1


def test_recovery_pause_event_failure_retains_journal(tmp_path):
    """BUG-03 covers the PAUSE critical events (ApprovalRequested / RunPaused),
    not only RunCompleted. A failure re-appending them retains the journal."""
    storage = FilesystemStorage(root=tmp_path)
    asyncio.run(
        storage.runs.create(_record("run-pa", "sess-pa", RunStatus.WAITING_APPROVAL, 1))
    )
    _seed_session(storage, "sess-pa")
    coordinator = _coordinator(storage, tmp_path)

    journal = TransactionJournal(tmp_path / "transactions")
    journal.begin(
        kind=TransactionKind.PAUSE,
        run_id="run-pa",
        target_run_status="waiting_approval",
        commit_id="pause:run-pa:appr-1",
        command={
            "approval_id": "appr-1",
            "tool_call_id": "c1",
            "tool_name": "shell",
            "reason": "review",
            "arguments": {},
            "checkpoint_payload_b64": "",
            "event_context": {
                "stream_id": "run-pa",
                "run_id": "run-pa",
                "root_run_id": "run-pa",
                "parent_run_id": None,
                "session_id": "sess-pa",
                "runnable_id": "agent-1",
            },
        },
    )

    coordinator._events = _FlakyEvents(storage.events, fail_times=1)
    asyncio.run(coordinator.recover_incomplete_commits())
    assert journal.list_incomplete(), "pause journal retained on event failure"

    coordinator._events = storage.events
    asyncio.run(coordinator.recover_incomplete_commits())
    assert not journal.list_incomplete()
    assert _count_events(storage, "run-pa", "ApprovalRequested") == 1
    assert _count_events(storage, "run-pa", "RunPaused") == 1
