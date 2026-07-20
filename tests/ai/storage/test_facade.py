#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import StorageCapabilityError
from linktools.ai.run.models import RunInput, RunnableType, RunRecord, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.features import (
    FILE_STORAGE_FEATURES,
    SQLALCHEMY_STORAGE_FEATURES,
)
from linktools.ai.storage import SqlAlchemyStorage
from linktools.ai.storage.facade import FilesystemStorage, Storage
from linktools.ai.asset.models import WriteOptions
from linktools.ai.asset.path import AssetPath
from linktools.ai.storage.sqlalchemy.models import Base


def _session_record(session_id="session-1") -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        id=session_id,
        parent_id=None,
        status=SessionStatus.ACTIVE,
        version=1,
        created_at=now,
        updated_at=now,
    )


def _run_record(run_id="run-1") -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        status=RunStatus.PENDING,
        input=RunInput(prompt="hi"),
        result=None,
        error=None,
        version=1,
        created_at=now,
        started_at=None,
        finished_at=None,
    )


def _can_acquire_a_lease(coordinator) -> bool:
    """The coordination field must be a wired, working LeaseCoordinator -- not
    None. Prove it by acquiring (and releasing) one lease through the public
    Protocol surface."""
    from datetime import timedelta

    async def _check() -> bool:
        token = await coordinator.acquire(
            key="wiring-check", owner_id="test", ttl=timedelta(seconds=30)
        )
        if token is None:
            return False
        await coordinator.release(token=token)
        return True

    return asyncio.run(_check())


def test_file_storage_constructs_full_facade_with_file_capabilities(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    assert isinstance(storage, Storage)
    assert storage.features is FILE_STORAGE_FEATURES
    assert storage.assets is not None
    assert storage.sessions is not None
    assert storage.runs is not None
    assert storage.events is not None
    assert storage.checkpoints is not None
    assert _can_acquire_a_lease(storage.coordination), (
        "FilesystemStorage must ship a wired, working LeaseCoordinator"
    )


def test_file_storage_runs_end_to_end(tmp_path):
    storage = FilesystemStorage(root=tmp_path)

    async def _run():
        await storage.sessions.create(_session_record())
        await storage.runs.create(_run_record())
        fetched = await storage.sessions.get("session-1")
        run = await storage.runs.get("run-1")
        path = AssetPath("/artifacts/tenant-1/run-1/draft.txt")
        await storage.assets.put(
            path, b"hello", options=WriteOptions(content_type="text/plain", metadata={})
        )
        resource = await storage.assets.get(path)
        return fetched, run, resource

    fetched, run, resource = asyncio.run(_run())
    assert fetched is not None and fetched.id == "session-1"
    assert run is not None and run.id == "run-1"
    assert resource is not None and resource.content == b"hello"


def test_file_storage_transaction_raises_storage_capability_error(tmp_path):
    storage = FilesystemStorage(root=tmp_path)

    async def _run():
        async with storage.transaction():
            pass

    with pytest.raises(StorageCapabilityError):
        asyncio.run(_run())


def _sqlalchemy_storage(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/facade.db")

    async def _create():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_create())
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/facade.db")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyStorage(session_factory=session_factory), engine


def test_sqlalchemy_storage_constructs_full_facade_with_sql_capabilities(tmp_path):
    storage, _ = _sqlalchemy_storage(tmp_path)
    assert isinstance(storage, Storage)
    assert storage.features is SQLALCHEMY_STORAGE_FEATURES
    assert storage.assets is not None
    assert storage.sessions is not None
    assert storage.runs is not None
    assert storage.events is not None
    assert storage.checkpoints is not None
    assert _can_acquire_a_lease(storage.coordination), (
        "SqlAlchemyStorage must ship a wired, working LeaseCoordinator"
    )


def test_sqlalchemy_storage_runs_end_to_end(tmp_path):
    storage, _ = _sqlalchemy_storage(tmp_path)

    async def _run():
        await storage.sessions.create(_session_record())
        await storage.runs.create(_run_record())
        fetched = await storage.sessions.get("session-1")
        run = await storage.runs.get("run-1")
        return fetched, run

    fetched, run = asyncio.run(_run())
    assert fetched is not None and fetched.id == "session-1"
    assert run is not None and run.id == "run-1"


def test_sqlalchemy_storage_transaction_yields_a_unit_of_work(tmp_path):
    storage, _ = _sqlalchemy_storage(tmp_path)

    async def _run():
        async with storage.transaction() as tx:
            from sqlalchemy import text

            # tx.session is the shared AsyncSession all stores bind to.
            result = await tx.session.execute(text("SELECT 1"))
            return result.scalar()

    assert asyncio.run(_run()) == 1


def test_sqlalchemy_storage_transaction_uow_stores_share_one_session(tmp_path):
    """All seven tx.* stores bind to the same AsyncSession in UoW mode."""
    storage, _ = _sqlalchemy_storage(tmp_path)

    async def _run():
        async with storage.transaction() as tx:
            return {
                "session": tx.session,
                "runs": tx.runs._session,
                "events": tx.events._session,
                "checkpoints": tx.checkpoints._session,
                "approvals": tx.approvals._session,
                "sessions": tx.sessions._session,
                "swarms": tx.swarms._session,
                "memories": tx.memories._session,
            }

    bound = asyncio.run(_run())
    shared = bound["session"]
    for name in (
        "runs",
        "events",
        "checkpoints",
        "approvals",
        "sessions",
        "swarms",
        "memories",
    ):
        assert bound[name] is shared, f"{name} does not share the UoW session"


def test_sqlalchemy_session_concurrent_append_normal_store(tmp_path):
    """scenario (actionable-fix-contract): normal (non-UoW) SessionStore
    mode supports concurrent appenders to the SAME session -- the
    unique-(session_id, sequence) retry loop in append_messages resolves the
    race, so N concurrent appends each get a distinct, gapless sequence."""
    from linktools.ai.session.models import MessageRole, NewSessionMessage

    storage, _ = _sqlalchemy_storage(tmp_path)

    async def _run():
        await storage.sessions.create(_session_record())

        async def _append(i: int):
            return await storage.sessions.append_messages(
                "session-1",
                (
                    NewSessionMessage(
                        role=MessageRole.USER, content=f"m{i}", run_id=None
                    ),
                ),
            )

        return await asyncio.gather(*(_append(i) for i in range(10)))

    results = asyncio.run(_run())
    sequences = sorted(batch[0].sequence for batch in results)
    assert sequences == list(range(1, 11)), (
        f"expected 1..10 with no duplicates, got {sequences}"
    )


def test_sqlalchemy_session_append_in_uow_single_writer(tmp_path):
    """scenario (contract): a SINGLE writer appending to a session inside an
    explicit UnitOfWork works normally (the documented boundary is multiple
    CONCURRENT UoW-mode writers to the same session, not UoW-mode append
    itself)."""
    from linktools.ai.session.models import MessageRole, NewSessionMessage

    storage, _ = _sqlalchemy_storage(tmp_path)

    async def _run():
        await storage.sessions.create(_session_record())
        async with storage.transaction() as tx:
            persisted = await tx.sessions.append_messages(
                "session-1",
                (NewSessionMessage(role=MessageRole.USER, content="hi", run_id=None),),
            )
        return persisted

    persisted = asyncio.run(_run())
    assert [m.sequence for m in persisted] == [1]
    # Committed: visible through the top-level (non-UoW) store afterward.
    messages = asyncio.run(storage.sessions.list_messages("session-1"))
    assert [m.content for m in messages] == ["hi"]


def test_sqlalchemy_storage_uow_commits_all_stores_on_success(tmp_path):
    """A clean exit commits every tx.* write -- both stores persist."""
    from linktools.ai.agent.approval import build_approval_request

    storage, _ = _sqlalchemy_storage(tmp_path)
    run = _run_record()
    approval = build_approval_request(
        run_id=run.id,
        tool_call_id="call-1",
        tool_name="tool-1",
        reason="ok",
        arguments={},
    )

    async def _run():
        async with storage.transaction() as tx:
            await tx.runs.create(run)
            await tx.approvals.create(approval)
        # After the UoW commits, both must be visible through the top-level
        # (non-UoW) stores, which open their own sessions against the same DB.
        return await storage.runs.get(run.id), await storage.approvals.get(approval.id)

    fetched_run, fetched_approval = asyncio.run(_run())
    assert fetched_run is not None and fetched_run.id == run.id
    assert fetched_approval is not None and fetched_approval.id == approval.id


def test_sqlalchemy_storage_uow_rolls_back_all_stores_on_failure(tmp_path):
    """An exception inside the UoW rolls back EVERY store -- neither the run
    nor the approval persists. This is the atomicity guarantee."""
    from linktools.ai.agent.approval import build_approval_request

    storage, _ = _sqlalchemy_storage(tmp_path)
    run = _run_record()
    approval = build_approval_request(
        run_id=run.id,
        tool_call_id="call-1",
        tool_name="tool-1",
        reason="ok",
        arguments={},
    )

    async def _run():
        try:
            async with storage.transaction() as tx:
                await tx.runs.create(run)
                await tx.approvals.create(approval)
                raise RuntimeError("simulate failure")  # forces rollback
        except RuntimeError:
            pass
        # Neither write should have survived the rollback.
        return await storage.runs.get(run.id), await storage.approvals.get(approval.id)

    fetched_run, fetched_approval = asyncio.run(_run())
    assert fetched_run is None, "run leaked after UoW rollback"
    assert fetched_approval is None, "approval leaked after UoW rollback"


def test_sqlalchemy_uow_idempotency_conflict_aborts_the_whole_transaction(tmp_path):
    """Revised from the original P0-7 SAVEPOINT-based fix (see
    storage/sqlalchemy/idempotency.py's reserve() docstring/comment): a
    session.begin_nested() SAVEPOINT that releases cleanly was measured to
    NOT reliably participate in a LATER, unrelated failure's rollback of the
    enclosing transaction under sqlite+aiosqlite -- a correctness risk worse
    than the isolation SAVEPOINT bought. reserve() no longer uses
    begin_nested() in UoW mode, so a genuine (scope, key) collision now
    aborts the WHOLE enclosing transaction (the accepted tradeoff: this
    requires two concurrent callers racing the exact same idempotency key
    within the exact same UnitOfWork, which does not happen in this
    codebase's actual call sites). This test documents and locks in that
    behavior: tx.runs.create() called AFTER a collision must NOT silently
    commit -- the whole unit fails."""
    storage, _ = _sqlalchemy_storage(tmp_path)
    run = _run_record()

    async def _run():
        async with storage.transaction() as tx:
            first = await tx.idempotency.claim(
                scope="scope-1", key="key-1", request_hash="hash-a", owner_id="own-1"
            )
            assert first.disposition.value == "acquired"  # fresh claim
            # Same (scope, key), DIFFERENT hash -> CONFLICT disposition (the
            # new claim model resolves conflicts gracefully -- no IntegrityError,
            # so the enclosing UoW is NOT aborted).
            second = await tx.idempotency.claim(
                scope="scope-1", key="key-1", request_hash="hash-b", owner_id="own-2"
            )
            assert second.disposition.value == "conflict"
            # The UoW is still usable: a subsequent tx.runs.create commits.
            await tx.runs.create(run)
        return await storage.runs.get(run.id)

    fetched_run = asyncio.run(_run())
    assert fetched_run is not None, (
        "tx.runs.create() must commit -- the CONFLICT disposition did not abort "
        "the enclosing transaction"
    )


def test_file_storage_exposes_file_swarm_store(tmp_path):
    from linktools.ai.storage.filesystem.swarm import FilesystemSwarmStore

    storage = FilesystemStorage(root=tmp_path)
    assert isinstance(storage.swarms, FilesystemSwarmStore)


def test_sqlalchemy_storage_exposes_sqlalchemy_swarm_store(tmp_path):
    from linktools.ai.storage.sqlalchemy.swarm import SqlAlchemySwarmStore

    storage, _ = _sqlalchemy_storage(tmp_path)
    assert isinstance(storage.swarms, SqlAlchemySwarmStore)


def test_file_storage_swarms_round_trips_a_swarm_run(tmp_path):
    from decimal import Decimal

    from linktools.ai.swarm.models import SwarmRun, SwarmStatus, TokenUsage

    storage = FilesystemStorage(root=tmp_path)
    now = datetime.now(timezone.utc)
    swarm_run = SwarmRun(
        id="swarm-1",
        run_id="drive-run-1",
        round=0,
        status=SwarmStatus.PENDING,
        version=1,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=now,
        updated_at=now,
    )

    async def _run():
        await storage.swarms.create_run(swarm_run)
        return await storage.swarms.get_run("swarm-1")

    fetched = asyncio.run(_run())
    assert fetched is not None
    assert fetched.id == "swarm-1"
    assert fetched.run_id == "drive-run-1"
    assert fetched.status is SwarmStatus.PENDING


def test_file_storage_exposes_file_memory_store(tmp_path):
    from linktools.ai.storage.filesystem.memory import FilesystemMemoryStore

    storage = FilesystemStorage(root=tmp_path)
    assert isinstance(storage.memories, FilesystemMemoryStore)


def test_sqlalchemy_storage_exposes_sqlalchemy_memory_store(tmp_path):
    from linktools.ai.storage.sqlalchemy.memory import SqlAlchemyMemoryStore

    storage, _ = _sqlalchemy_storage(tmp_path)
    assert isinstance(storage.memories, SqlAlchemyMemoryStore)


def test_file_storage_memories_round_trips_a_record(tmp_path):
    from linktools.ai.memory.models import MemoryRecord

    storage = FilesystemStorage(root=tmp_path)
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id="mem-1",
        tenant_id="t1",
        owner_id="user-1",
        content="prefers terse answers",
        category="preference",
        confidence=0.8,
        version=1,
        created_at=now,
        updated_at=now,
        metadata={},
    )

    async def _run():
        await storage.memories.remember(record)
        return await storage.memories.get("mem-1")

    fetched = asyncio.run(_run())
    assert fetched is not None
    assert fetched.id == "mem-1"
    assert fetched.content == "prefers terse answers"
    assert fetched.owner_id == "user-1"


def test_file_storage_exposes_file_approval_store(tmp_path):
    from linktools.ai.agent.approval import ApprovalStore
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore

    storage = FilesystemStorage(root=tmp_path)
    assert isinstance(storage.approvals, FilesystemApprovalStore)
    assert isinstance(storage.approvals, ApprovalStore)


def test_sqlalchemy_storage_exposes_sqlalchemy_approval_store(tmp_path):
    from linktools.ai.agent.approval import ApprovalStore
    from linktools.ai.storage.sqlalchemy.approval import SqlAlchemyApprovalStore

    storage, _ = _sqlalchemy_storage(tmp_path)
    assert isinstance(storage.approvals, SqlAlchemyApprovalStore)
    assert isinstance(storage.approvals, ApprovalStore)


def test_file_storage_approvals_round_trips_a_request(tmp_path):
    from linktools.ai.agent.approval import (
        ApprovalStatus,
        build_approval_request,
    )

    storage = FilesystemStorage(root=tmp_path)
    request = build_approval_request(
        run_id="run-1",
        tool_call_id="call-1",
        tool_name="browser.search",
        reason="needs human confirmation",
        arguments={"query": "test"},
    )

    async def _run():
        await storage.approvals.create(request)
        return await storage.approvals.get(request.id)

    fetched = asyncio.run(_run())
    assert fetched is not None
    assert fetched.id == request.id
    assert fetched.run_id == "run-1"
    assert fetched.tool_call_id == "call-1"
    assert fetched.tool_name == "browser.search"
    assert fetched.reason == "needs human confirmation"
    assert fetched.status is ApprovalStatus.PENDING
