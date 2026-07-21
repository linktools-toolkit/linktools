#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/sqlalchemy/test_commit.py — SqlAlchemyRunCommitCoordinator
contract (WP-04).

pause() and complete() each run inside one SqlAlchemyStorage.transaction():
every store writes through the same AsyncSession + same transaction, so the
whole commit lands together or rolls back together. The failure-injection
tests prove the rollback: when a late step (the SUCCEEDED transition) raises,
the session messages and checkpoint written earlier in the same txn are rolled
back too -- no orphan artifacts, run stays in its prior status."""

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.events.context import EventContext
from linktools.ai.run.commit import CompleteRunCommand, PauseRunCommand
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import (
    RunInput,
    RunnableType,
    RunRecord,
    RunResult,
    RunStatus,
)
from linktools.ai.session.models import (
    MessageRole,
    NewSessionMessage,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.storage import SqlAlchemyStorage
from linktools.ai.storage.sqlalchemy.models import Base


_APPROVAL_BINDING = {
    "descriptor_fingerprint": "fp-test",
    "handler_revision": "h1",
    "provider_revision": "p1",
    "policy_revision": "pol1",
    "capability_revision": "cap1",
    "result_processor_revision": "rp1",
    "arguments_hash": "ah1",
}


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


def _seed(storage, run_id, session_id, status=RunStatus.RUNNING):
    now = datetime.now(timezone.utc)

    async def _seed_async():
        await storage.sessions.create(
            SessionRecord(
                id=session_id,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        await storage.runs.create(
            RunRecord(
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
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )

    asyncio.run(_seed_async())


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


def _messages(run_id):
    return (
        NewSessionMessage(role=MessageRole.USER, content="hi", run_id=run_id),
        NewSessionMessage(
            role=MessageRole.ASSISTANT, content='{"response": "ok"}', run_id=run_id
        ),
    )


def test_complete_commits_atomically_succeeded(tmp_path):
    storage = _storage(tmp_path)
    _seed(storage, "run-1", "sess-1")

    async def _run():
        from linktools.ai.storage.sqlalchemy.commit import (
            SqlAlchemyRunCommitCoordinator,
        )

        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        result = RunResult(output={"response": "ok"})
        commit = await coordinator.complete(
            CompleteRunCommand(
                run_id="run-1",
                session_id="sess-1",
                expected_version=1,
                messages=_messages("run-1"),
                checkpoint_payload=b'{"messages": []}',
                result=result,
                event_context=EventContext.from_run_context(_ctx("run-1", "sess-1")),
            )
        )
        assert commit.result is result
        record = await storage.runs.get("run-1")
        assert record.status is RunStatus.SUCCEEDED
        # §8.3 step 5: the SUCCEEDED transition persists the RunResult.
        assert record.result is not None
        assert record.result.output == result.output
        messages = await storage.sessions.list_messages("sess-1")
        assert sum(1 for m in messages if m.role is MessageRole.USER) == 1
        assert sum(1 for m in messages if m.role is MessageRole.ASSISTANT) == 1
        checkpoint = await storage.checkpoints.latest("run-1")
        assert checkpoint is not None and checkpoint.sequence == 1
        page = await storage.events.list("run-1", limit=100)
        types = [type(e.payload).__name__ for e in page.items]
        assert types.count("RunCompleted") == 1

    asyncio.run(_run())


def test_pause_commits_atomically_waiting_approval(tmp_path):
    storage = _storage(tmp_path)
    _seed(storage, "run-2", "sess-2")

    async def _run():
        from linktools.ai.storage.sqlalchemy.commit import (
            SqlAlchemyRunCommitCoordinator,
        )

        coordinator = SqlAlchemyRunCommitCoordinator(storage)
        commit = await coordinator.pause(
            PauseRunCommand(
                run_id="run-2",
                expected_version=1,
                approval_request={
                    "approval_id": "appr-2",
                    "tool_call_id": "call-2",
                    "tool_name": "shell",
                    "reason": "review",
                    "arguments": {"cmd": "ls"},
                    **_APPROVAL_BINDING,
                },
                checkpoint_payload=b'{"messages": []}',
                event_context=EventContext.from_run_context(_ctx("run-2", "sess-2")),
            )
        )
        record = await storage.runs.get("run-2")
        assert record.status is RunStatus.WAITING_APPROVAL
        approval = await storage.approvals.get(commit.approval_id)
        assert approval is not None and approval.run_id == "run-2"
        checkpoint = await storage.checkpoints.latest("run-2")
        assert checkpoint is not None
        assert checkpoint.sequence == 1
        # §6.7: exactly one of each critical artifact for a single pause.
        approvals_for_run = await storage.approvals.list_for_run("run-2")
        assert len(approvals_for_run) == 1
        page = await storage.events.list("run-2", limit=100)
        types = [type(e.payload).__name__ for e in page.items]
        assert types.count("ApprovalRequested") == 1, types
        assert types.count("RunPaused") == 1, types
        assert "RunFailed" not in types

    asyncio.run(_run())


def test_complete_rolls_back_when_transition_fails(tmp_path):
    """Failure injection (WP-04 §8.5): if the SUCCEEDED transition raises, the
    session messages + checkpoint written earlier in the SAME txn must roll
    back -- the run stays RUNNING and no partial turn is persisted."""
    storage = _storage(tmp_path)
    _seed(storage, "run-3", "sess-3")

    async def _run():
        from linktools.ai.storage.sqlalchemy.commit import (
            SqlAlchemyRunCommitCoordinator,
        )

        coordinator = SqlAlchemyRunCommitCoordinator(storage)

        # Wrap the UoW runs store so the SUCCEEDED transition raises while
        # leaving the real store usable for the post-condition reads. We
        # patch the coordinator's per-transaction runs transition by
        # intercepting transaction().
        original_transaction = storage.transaction

        class _FailingRuns:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            async def transition(self, *args, **kwargs):
                if kwargs.get("expected_version") is not None and (
                    args and args[-1] is RunStatus.SUCCEEDED
                ):
                    raise RuntimeError("injected transition failure")
                return await self._real.transition(*args, **kwargs)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _patched_transaction():
            async with original_transaction() as tx:
                yield type(tx)(
                    session=tx.session,
                    assets=tx.assets,
                    artifact_records=tx.artifact_records,
                    runs=_FailingRuns(tx.runs),
                    events=tx.events,
                    checkpoints=tx.checkpoints,
                    approvals=tx.approvals,
                    sessions=tx.sessions,
                    swarms=tx.swarms,
                    memories=tx.memories,
                    idempotency=tx.idempotency,
                    jobs=tx.jobs,
                    evaluations=tx.evaluations,
                )

        object.__setattr__(storage, "transaction", _patched_transaction)

        with pytest.raises(RuntimeError, match="injected"):
            await coordinator.complete(
                CompleteRunCommand(
                    run_id="run-3",
                    session_id="sess-3",
                    expected_version=1,
                    messages=_messages("run-3"),
                    checkpoint_payload=b'{"messages": []}',
                    result=RunResult(output="x"),
                    event_context=EventContext.from_run_context(
                        _ctx("run-3", "sess-3")
                    ),
                )
            )

        # Post-condition: full rollback. Run stayed RUNNING; no session turn,
        # no checkpoint, no RunCompleted event were persisted.
        record = await storage.runs.get("run-3")
        assert record.status is RunStatus.RUNNING
        messages = await storage.sessions.list_messages("sess-3")
        assert len(messages) == 0
        assert await storage.checkpoints.latest("run-3") is None
        page = await storage.events.list("run-3", limit=100)
        types = [type(e.payload).__name__ for e in page.items]
        assert "RunCompleted" not in types

    asyncio.run(_run())


def test_pause_rolls_back_when_checkpoint_append_fails(tmp_path):
    """Failure injection (WP-04 §8.2/§8.5): the pause path writes approval ->
    checkpoint -> transition -> events in one txn. If the checkpoint append
    (step 2) raises, the approval written in step 1 must roll back too -- no
    orphan approval, no WAITING_APPROVAL transition, run stays RUNNING."""
    from contextlib import asynccontextmanager

    storage = _storage(tmp_path)
    _seed(storage, "run-4", "sess-4")

    async def _run():
        from linktools.ai.storage.sqlalchemy.commit import (
            SqlAlchemyRunCommitCoordinator,
        )

        coordinator = SqlAlchemyRunCommitCoordinator(storage)

        class _FailingCheckpoints:
            def __init__(self, real):
                self._real = real

            def __getattr__(self, name):
                return getattr(self._real, name)

            async def append(self, *args, **kwargs):
                raise RuntimeError("injected checkpoint append failure")

        original_transaction = storage.transaction

        @asynccontextmanager
        async def _patched_transaction():
            async with original_transaction() as tx:
                yield type(tx)(
                    session=tx.session,
                    assets=tx.assets,
                    artifact_records=tx.artifact_records,
                    runs=tx.runs,
                    events=tx.events,
                    checkpoints=_FailingCheckpoints(tx.checkpoints),
                    approvals=tx.approvals,
                    sessions=tx.sessions,
                    swarms=tx.swarms,
                    memories=tx.memories,
                    idempotency=tx.idempotency,
                    jobs=tx.jobs,
                    evaluations=tx.evaluations,
                )

        object.__setattr__(storage, "transaction", _patched_transaction)

        with pytest.raises(RuntimeError, match="injected checkpoint"):
            await coordinator.pause(
                PauseRunCommand(
                    run_id="run-4",
                    expected_version=1,
                    approval_request={
                        "approval_id": "appr-4",
                        "tool_call_id": "call-4",
                        "tool_name": "shell",
                        "reason": "review",
                        "arguments": {"cmd": "ls"},
                        **_APPROVAL_BINDING,
                    },
                    checkpoint_payload=b'{"messages": []}',
                    event_context=EventContext.from_run_context(
                        _ctx("run-4", "sess-4")
                    ),
                )
            )

        # Post-condition: full rollback. Run stayed RUNNING; no orphan approval,
        # no checkpoint, no WAITING_APPROVAL transition persisted.
        record = await storage.runs.get("run-4")
        assert record.status is RunStatus.RUNNING
        approvals = await storage.approvals.list_for_run("run-4")
        assert len(approvals) == 0, "approval must roll back with the checkpoint"
        assert await storage.checkpoints.latest("run-4") is None

    asyncio.run(_run())

