#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import StorageCapabilityError
from linktools.ai.run.models import RunInput, RunnableType, RunRecord, RunStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.capabilities import FILE_STORAGE_CAPABILITIES, SQLALCHEMY_STORAGE_CAPABILITIES
from linktools.ai.storage.facade import FileStorage, SqlAlchemyStorage, Storage
from linktools.ai.storage.resource.models import WriteOptions
from linktools.ai.storage.resource.path import ResourcePath
from linktools.ai.storage.sqlalchemy.models import Base


def _session_record(session_id="session-1") -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        id=session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1,
        created_at=now, updated_at=now,
    )


def _run_record(run_id="run-1") -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id, root_run_id=run_id, parent_run_id=None, session_id="session-1",
        runnable_id="agent-1", runnable_type=RunnableType.AGENT, status=RunStatus.PENDING,
        input=RunInput(prompt="hi"), result=None, error=None, version=1,
        created_at=now, started_at=None, finished_at=None,
    )


def test_file_storage_constructs_full_facade_with_file_capabilities(tmp_path):
    storage = FileStorage(root=tmp_path)
    assert isinstance(storage, Storage)
    assert storage.capabilities is FILE_STORAGE_CAPABILITIES
    assert storage.resources is not None
    assert storage.sessions is not None
    assert storage.runs is not None
    assert storage.events is not None
    assert storage.checkpoints is not None


def test_file_storage_runs_end_to_end(tmp_path):
    storage = FileStorage(root=tmp_path)

    async def _run():
        await storage.sessions.create(_session_record())
        await storage.runs.create(_run_record())
        fetched = await storage.sessions.get("session-1")
        run = await storage.runs.get("run-1")
        path = ResourcePath("/artifacts/tenant-1/run-1/draft.txt")
        await storage.resources.put(path, b"hello", options=WriteOptions(content_type="text/plain", metadata={}))
        resource = await storage.resources.get(path)
        return fetched, run, resource

    fetched, run, resource = asyncio.run(_run())
    assert fetched is not None and fetched.id == "session-1"
    assert run is not None and run.id == "run-1"
    assert resource is not None and resource.content == b"hello"


def test_file_storage_transaction_raises_storage_capability_error(tmp_path):
    storage = FileStorage(root=tmp_path)

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
    assert storage.capabilities is SQLALCHEMY_STORAGE_CAPABILITIES
    assert storage.resources is not None
    assert storage.sessions is not None
    assert storage.runs is not None
    assert storage.events is not None
    assert storage.checkpoints is not None


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
    for name in ("runs", "events", "checkpoints", "approvals", "sessions", "swarms", "memories"):
        assert bound[name] is shared, f"{name} does not share the UoW session"


def test_sqlalchemy_storage_uow_commits_all_stores_on_success(tmp_path):
    """A clean exit commits every tx.* write -- both stores persist."""
    from linktools.ai.agent.approval import build_approval_request
    storage, _ = _sqlalchemy_storage(tmp_path)
    run = _run_record()
    approval = build_approval_request(
        run_id=run.id, tool_call_id="call-1", tool_name="tool-1", reason="ok", arguments={},
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
        run_id=run.id, tool_call_id="call-1", tool_name="tool-1", reason="ok", arguments={},
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


def test_sqlalchemy_uow_idempotency_conflict_does_not_poison_transaction(tmp_path):
    """P0-7 (review doc §7.6): a unique-(scope,key) collision on
    tx.idempotency.reserve() must not poison the surrounding UnitOfWork. Before
    the SAVEPOINT fix, the IntegrityError raised on flush() left the shared
    AsyncSession's transaction unusable, so a subsequent tx.runs.create() in
    the same unit would itself fail (or silently be dropped) instead of
    committing normally."""
    storage, _ = _sqlalchemy_storage(tmp_path)
    run = _run_record()

    async def _run():
        async with storage.transaction() as tx:
            first = await tx.idempotency.reserve("scope-1", "key-1", "hash-a")
            assert first is None  # fresh reservation
            # Same (scope, key) but the SAME hash: reserve() returns the
            # existing RESERVED record rather than raising -- exercise the
            # actual conflict path (different hash) instead.
            with pytest.raises(Exception):
                await tx.idempotency.reserve("scope-1", "key-1", "hash-b")
            # The collision above must NOT have poisoned tx's shared session --
            # this write must still commit when the `async with` exits cleanly.
            await tx.runs.create(run)
        return await storage.runs.get(run.id)

    fetched_run = asyncio.run(_run())
    assert fetched_run is not None, (
        "tx.runs.create() did not survive the UoW after an idempotency "
        "conflict -- the conflict poisoned the shared transaction"
    )


def test_file_storage_exposes_file_swarm_store(tmp_path):
    from linktools.ai.storage.file.swarm import FileSwarmStore
    storage = FileStorage(root=tmp_path)
    assert isinstance(storage.swarms, FileSwarmStore)


def test_sqlalchemy_storage_exposes_sqlalchemy_swarm_store(tmp_path):
    from linktools.ai.storage.sqlalchemy.swarm import SqlAlchemySwarmStore
    storage, _ = _sqlalchemy_storage(tmp_path)
    assert isinstance(storage.swarms, SqlAlchemySwarmStore)


def test_file_storage_swarms_round_trips_a_swarm_run(tmp_path):
    from decimal import Decimal

    from linktools.ai.swarm.models import SwarmRun, SwarmStatus, TokenUsage

    storage = FileStorage(root=tmp_path)
    now = datetime.now(timezone.utc)
    swarm_run = SwarmRun(
        id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.PENDING,
        version=1, token_usage=TokenUsage(), cost=Decimal("0"),
        created_at=now, updated_at=now,
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
    from linktools.ai.storage.file.memory import FileMemoryStore
    storage = FileStorage(root=tmp_path)
    assert isinstance(storage.memories, FileMemoryStore)


def test_sqlalchemy_storage_exposes_sqlalchemy_memory_store(tmp_path):
    from linktools.ai.storage.sqlalchemy.memory import SqlAlchemyMemoryStore
    storage, _ = _sqlalchemy_storage(tmp_path)
    assert isinstance(storage.memories, SqlAlchemyMemoryStore)


def test_file_storage_memories_round_trips_a_record(tmp_path):
    from linktools.ai.memory.models import MemoryRecord
    storage = FileStorage(root=tmp_path)
    now = datetime.now(timezone.utc)
    record = MemoryRecord(
        id="mem-1", owner_id="user-1", content="prefers terse answers",
        category="preference", confidence=0.8, version=1,
        created_at=now, updated_at=now, metadata={},
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
    from linktools.ai.storage.file.approval import FileApprovalStore
    storage = FileStorage(root=tmp_path)
    assert isinstance(storage.approvals, FileApprovalStore)
    assert isinstance(storage.approvals, ApprovalStore)


def test_sqlalchemy_storage_exposes_sqlalchemy_approval_store(tmp_path):
    from linktools.ai.agent.approval import ApprovalStore
    from linktools.ai.storage.sqlalchemy.approval import SqlAlchemyApprovalStore
    storage, _ = _sqlalchemy_storage(tmp_path)
    assert isinstance(storage.approvals, SqlAlchemyApprovalStore)
    assert isinstance(storage.approvals, ApprovalStore)


def test_file_storage_approvals_round_trips_a_request(tmp_path):
    from linktools.ai.agent.approval import ApprovalRequest, ApprovalStatus, build_approval_request
    storage = FileStorage(root=tmp_path)
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

