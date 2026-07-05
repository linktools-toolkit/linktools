#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_run_store_contract.py — runs the same RunStore
contract against both FileRunStore and SqlAlchemyRunStore."""
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import InvalidRunTransitionError, RunConflictError, RunNotFoundError
from linktools.ai.run.models import RunErrorInfo, RunInput, RunnableType, RunRecord, RunResult, RunStatus
from linktools.ai.storage.file.run import FileRunStore


def _record(run_id="run-1", parent_run_id=None, status=RunStatus.PENDING, version=1) -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id, root_run_id=run_id if parent_run_id is None else "run-1", parent_run_id=parent_run_id,
        session_id="session-1", runnable_id="agent-1", runnable_type=RunnableType.AGENT, status=status,
        input=RunInput(prompt="hi"), result=None, error=None, version=version,
        created_at=now, started_at=None, finished_at=None,
    )


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FileRunStore(root=tmp_path / f"runs-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.run import SqlAlchemyRunStore

    counter = {"n": 0}

    def sqlalchemy_factory():
        counter["n"] += 1
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/runs-db-{counter['n']}.db")

        async def _create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        import asyncio
        import threading

        result = {}

        def _run():
            result["exc"] = None
            try:
                asyncio.run(_create())
            except Exception as exc:
                result["exc"] = exc

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()
        if result["exc"] is not None:
            raise result["exc"]
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyRunStore(session_factory=session_factory)

    return sqlalchemy_factory


@pytest.mark.asyncio
async def test_create_then_get_roundtrip(store_factory):
    store = store_factory()
    created = await store.create(_record())
    fetched = await store.get("run-1")
    assert fetched is not None
    assert fetched.id == "run-1"
    assert fetched.status == RunStatus.PENDING
    assert created == fetched


@pytest.mark.asyncio
async def test_get_missing_returns_none(store_factory):
    store = store_factory()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_transition_pending_to_running_succeeds(store_factory):
    store = store_factory()
    await store.create(_record())
    updated = await store.transition("run-1", RunStatus.RUNNING, expected_version=1)
    assert updated.status == RunStatus.RUNNING
    assert updated.version == 2


@pytest.mark.asyncio
async def test_transition_invalid_target_raises(store_factory):
    store = store_factory()
    await store.create(_record())
    with pytest.raises(InvalidRunTransitionError):
        await store.transition("run-1", RunStatus.SUCCEEDED, expected_version=1)


@pytest.mark.asyncio
async def test_transition_wrong_expected_version_raises_conflict(store_factory):
    store = store_factory()
    await store.create(_record())
    with pytest.raises(RunConflictError):
        await store.transition("run-1", RunStatus.RUNNING, expected_version=99)


@pytest.mark.asyncio
async def test_transition_missing_run_raises_not_found(store_factory):
    store = store_factory()
    with pytest.raises(RunNotFoundError):
        await store.transition("nope", RunStatus.RUNNING, expected_version=1)


@pytest.mark.asyncio
async def test_transition_to_succeeded_stores_result(store_factory):
    store = store_factory()
    await store.create(_record())
    await store.transition("run-1", RunStatus.RUNNING, expected_version=1)
    done = await store.transition(
        "run-1", RunStatus.SUCCEEDED, expected_version=2, result=RunResult(output={"ok": True}),
    )
    assert done.status == RunStatus.SUCCEEDED
    assert done.result.output == {"ok": True}


@pytest.mark.asyncio
async def test_transition_to_failed_stores_error(store_factory):
    store = store_factory()
    await store.create(_record())
    await store.transition("run-1", RunStatus.RUNNING, expected_version=1)
    failed = await store.transition(
        "run-1", RunStatus.FAILED, expected_version=2, error=RunErrorInfo(error_type="X", message="boom"),
    )
    assert failed.status == RunStatus.FAILED
    assert failed.error.message == "boom"


@pytest.mark.asyncio
async def test_list_children_returns_only_direct_children(store_factory):
    store = store_factory()
    await store.create(_record(run_id="parent", status=RunStatus.PENDING))
    await store.create(_record(run_id="child-1", parent_run_id="parent"))
    await store.create(_record(run_id="child-2", parent_run_id="parent"))
    await store.create(_record(run_id="unrelated", parent_run_id=None))
    children = await store.list_children("parent")
    assert {c.id for c in children} == {"child-1", "child-2"}
