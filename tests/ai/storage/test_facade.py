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


def test_sqlalchemy_storage_transaction_yields_a_shared_session(tmp_path):
    storage, _ = _sqlalchemy_storage(tmp_path)

    async def _run():
        async with storage.transaction() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            return result.scalar()

    assert asyncio.run(_run()) == 1
