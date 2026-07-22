#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture locks for the v4 fixed-scope closure (guide ).

One representative test per fixed issue, so a future change -- or a deleted
per-area test file -- cannot silently re-introduce the gap each fix closed:

1. SQL idempotency: a concurrent first-time claim never leaks a raw
   IntegrityError.
2. RunDefinitionStore is a required Storage capability; Runtime.build fails
   fast without it.
3. File commit events dedup by commit_id, so recovery does not duplicate and a
   second legitimate approval keeps its events.
4. Swarm resume rejects a terminal driving Run before strategy.resume.
"""

import asyncio
import dataclasses
from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

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
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.sqlalchemy.idempotency import SqlAlchemyIdempotencyStore
from linktools.ai.storage.sqlalchemy.models import Base, ToolIdempotencyRow
from linktools.ai.tool.idempotency import ClaimDisposition


# --------------------------------------------------------------------------- #
# SQL concurrent first-time claim
# --------------------------------------------------------------------------- #


async def _make_sql_store(tmp_path, db_name: str = "v4.db"):
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / db_name}",
        connect_args={"timeout": 30.0},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return (
        engine,
        session_factory,
        SqlAlchemyIdempotencyStore(session_factory=session_factory),
    )


def _force_fresh_insert_collision(monkeypatch) -> None:
    """Force both fresh-INSERT flushes to rendezvous so the loser deterministically
    hits the UNIQUE(scope, key) constraint instead of seeing the winner's row."""
    let_first_proceed = asyncio.Event()
    flushes = {"n": 0}
    real_flush = AsyncSession.flush

    async def coordinated(self, *args, **kwargs):
        flushes["n"] += 1
        if flushes["n"] == 1:
            await asyncio.wait_for(let_first_proceed.wait(), timeout=5.0)
        elif flushes["n"] == 2:
            let_first_proceed.set()
        return await real_flush(self, *args, **kwargs)

    monkeypatch.setattr(AsyncSession, "flush", coordinated)


@pytest.mark.asyncio
async def test_v4_sql_concurrent_claim_never_leaks_integrity_error(
    tmp_path, monkeypatch
):
    """two concurrent first-time claims on the same (scope, key) yield one
    ACQUIRED and one stable IN_PROGRESS -- never a propagated IntegrityError."""
    _force_fresh_insert_collision(monkeypatch)
    engine, session_factory, store = await _make_sql_store(tmp_path)

    async def claim(owner: str):
        return await store.claim(
            scope="tool:v4", key="k", request_hash="h", owner_id=owner
        )

    first, second = await asyncio.gather(claim("a"), claim("b"))
    dispositions = {first.disposition, second.disposition}

    assert ClaimDisposition.ACQUIRED in dispositions, dispositions
    assert ClaimDisposition.IN_PROGRESS in dispositions, dispositions

    async with session_factory() as session:
        count = int(
            (
                await session.execute(
                    select(func.count()).select_from(ToolIdempotencyRow)
                )
            ).scalar_one()
        )
    assert count == 1, "the loser must never persist a duplicate row"
    await engine.dispose()


# --------------------------------------------------------------------------- #
# RunDefinitionStore is required
# --------------------------------------------------------------------------- #


def test_v4_storage_requires_run_definition_store_and_runtime_fails_fast(tmp_path):
    """run_definitions is a required Storage field (no default) and
    Runtime.build raises RuntimeInitializationError when it is None."""
    from linktools.ai.errors import RuntimeInitializationError
    from linktools.ai.runtime import Runtime
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    fields = {f.name: f for f in dataclasses.fields(FilesystemStorage)}
    assert fields["run_definitions"].default is dataclasses.MISSING, (
        "run_definitions must be a required field (no default)"
    )

    storage = FilesystemStorage(root=tmp_path)
    object.__setattr__(storage, "run_definitions", None)
    with pytest.raises(RuntimeInitializationError):
        Runtime.build(
            storage=storage,
            commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
        )


# --------------------------------------------------------------------------- #
# File commit events dedup by commit_id
# --------------------------------------------------------------------------- #


def _record(run_id, session_id, status, version):
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


async def _count(storage, run_id, payload_type):
    page = await storage.events.list(run_id, after_sequence=0, limit=10000)
    return sum(1 for e in page.items if type(e.payload).__name__ == payload_type)


def test_v4_file_commit_events_dedup_by_commit_id(tmp_path):
    """a run that pauses for two distinct approvals keeps both events
    (one per commit_id), and recovery does not duplicate RunCompleted."""

    async def _run():
        storage = FilesystemStorage(root=tmp_path)
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id="sess",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        coordinator = _coordinator(storage, tmp_path)

        # --- two distinct approvals on the same run each keep their events ---
        await storage.runs.create(_record("run", "sess", RunStatus.RUNNING, 1))

        async def _pause(commit_id, approval_id, tool_call_id, expected_version):
            await coordinator.pause(
                PauseRunCommand(
                    run_id="run",
                    expected_version=expected_version,
                        approval_request={
                        "approval_id": approval_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": "shell",
                        "reason": "review",
                            "arguments": {"cmd": "ls"},
                            "descriptor_fingerprint": "descriptor-v1",
                            "handler_revision": "handler-v1",
                            "provider_revision": "provider-v1",
                            "policy_revision": "policy-v1",
                            "capability_revision": "capability-v1",
                            "result_processor_revision": "processor-v1",
                            "arguments_hash": __import__(
                                "linktools.ai.agent.approval", fromlist=["compute_arguments_hash"]
                            ).compute_arguments_hash("shell", {"cmd": "ls"}),
                        },
                    checkpoint_payload=b'{"m":[]}',
                    event_context=_ctx("run", "sess"),
                    commit_id=commit_id,
                )
            )

        await _pause("commit-a", "appr-1", "call-1", expected_version=1)
        await storage.runs.transition("run", RunStatus.RUNNING, expected_version=2)
        await _pause("commit-b", "appr-2", "call-2", expected_version=3)
        assert await _count(storage, "run", "ApprovalRequested") == 2
        assert await _count(storage, "run", "RunPaused") == 2

        # --- recovery does not duplicate RunCompleted for a complete commit ---
        await storage.runs.create(_record("run2", "sess", RunStatus.RUNNING, 1))
        await coordinator.complete(
            CompleteRunCommand(
                run_id="run2",
                session_id="sess",
                expected_version=1,
                messages=(
                    NewSessionMessage(
                        role=MessageRole.USER, content="hi", run_id="run2"
                    ),
                ),
                checkpoint_payload=b'{"m":[]}',
                result=RunResult(output="ok"),
                event_context=_ctx("run2", "sess"),
            )
        )
        assert await _count(storage, "run2", "RunCompleted") == 1

        from linktools.ai.storage.filesystem.journal import (
            TransactionJournal,
            TransactionKind,
        )

        journal = TransactionJournal(tmp_path / "transactions")
        journal.begin(
            kind=TransactionKind.COMPLETE,
            run_id="run2",
            target_run_status="succeeded",
            commit_id="complete:run2:1",
            command={
                "event_context": {
                    "stream_id": "run2",
                    "run_id": "run2",
                    "root_run_id": "run2",
                    "parent_run_id": None,
                    "session_id": "sess",
                    "runnable_id": "agent-1",
                }
            },
        )
        await coordinator.recover_incomplete_commits()
        assert await _count(storage, "run2", "RunCompleted") == 1

    asyncio.run(_run())


# --------------------------------------------------------------------------- #
# Swarm resume rejects a terminal driving Run
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "driving_status",
    [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED],
    ids=["succeeded", "failed", "cancelled"],
)
def test_v4_swarm_resume_rejects_terminal_driving_run(tmp_path, driving_status):
    """a PAUSED swarm whose driving Run is terminal is rejected before
    strategy.resume runs (the swarm and driving Run are left untouched)."""
    from decimal import Decimal

    from linktools.ai.agent.engine import AgentEngine
    from linktools.ai.errors import InvalidRunTransitionError
    from linktools.ai.run.controller import RunController
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
    from linktools.ai.storage.filesystem.definition import FilesystemRunDefinitionStore
    from linktools.ai.storage.filesystem.event import FilesystemEventStore
    from linktools.ai.storage.filesystem.run import FilesystemRunStore
    from linktools.ai.storage.filesystem.session import FilesystemSessionStore
    from linktools.ai.storage.filesystem.swarm import FilesystemSwarmStore
    from linktools.ai.swarm.models import SwarmRun, SwarmStatus, TokenUsage
    from linktools.ai.swarm.runner import SwarmRunner

    now = datetime.now(timezone.utc)

    class _Stores:
        def __init__(self, root):
            self.run_store = FilesystemRunStore(root=root / "runs")
            self.session_store = FilesystemSessionStore(root=root / "sessions")
            self.event_store = FilesystemEventStore(root=root / "events")
            self.checkpoint_store = FilesystemCheckpointStore(root=root / "checkpoints")
            self.swarm_store = FilesystemSwarmStore(root=root / "swarm")
            self.run_definitions = FilesystemRunDefinitionStore(root=root / "definitions")
            self.run_controller = RunController()
            self.agent_runner = AgentEngine(
                run_store=self.run_store,
                session_store=self.session_store,
                event_store=self.event_store,
                checkpoint_store=self.checkpoint_store,
                run_controller=self.run_controller,
                commit_coordinator=FilesystemRunCommitCoordinator(
                    approval_store=FilesystemApprovalStore(root=root / "approvals"),
                    checkpoint_store=self.checkpoint_store,
                    run_store=self.run_store,
                    session_store=self.session_store,
                    event_store=self.event_store,
                ),
            )

    stores = _Stores(tmp_path)
    # The compiler is never reached -- the terminal driving Run is rejected
    # before snapshot/compile/strategy.resume -- so a dummy suffices and keeps
    # this lock independent of AgentCompiler/GovernedToolInvoker wiring.
    compiler = object()

    async def _seed():
        await stores.run_store.create(
            RunRecord(
                id="drive",
                root_run_id="drive",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=driving_status,
                input=RunInput(prompt="done"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm",
                run_id="drive",
                round=0,
                status=SwarmStatus.PAUSED,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )

    asyncio.run(_seed())
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
        run_controller=stores.run_controller,
        run_definitions=stores.run_definitions,
    )
    with pytest.raises(InvalidRunTransitionError):
        asyncio.run(runner.resume("swarm"))

    async def _verify():
        driving = await stores.run_store.get("drive")
        swarm = await stores.swarm_store.get_run("swarm")
        return driving, swarm

    driving, swarm = asyncio.run(_verify())
    assert driving.status is driving_status
    assert swarm.status is SwarmStatus.PAUSED
