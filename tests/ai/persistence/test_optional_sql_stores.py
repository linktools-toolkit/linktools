import asyncio
from datetime import timedelta

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import pytest

from linktools.ai.errors import StorageConflictError
from linktools.ai.tasks.models import TaskExecution, TaskPlan
from linktools.ai.tasks.persistence.sqlalchemy import SqlAlchemyTaskBackend
from linktools.ai.tasks.store import TaskStore
from linktools.ai.tool.persistence.sqlalchemy import SqlAlchemyToolStateBackend
from linktools.ai.tool.state import ToolOperation, ToolOperationStatus
from linktools.ai.tool.store import ToolStateStore


@pytest.mark.asyncio
async def test_sql_task_store_fences_claims(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = TaskStore(SqlAlchemyTaskBackend(factory))
    await store.initialize_storage(engine)
    await store.save_plan(TaskPlan("p", ()))
    await store.add_execution(TaskExecution("e", "p", "n", "ready"))
    claimed = await store.claim("e", owner="worker")
    assert claimed.lease.expires_at is not None
    renewed = await store.renew("e", owner="worker", fence=claimed.fence)
    assert renewed.lease.expires_at >= claimed.lease.expires_at
    with pytest.raises(StorageConflictError):
        await store.complete("e", owner="other", fence=claimed.fence, result=None)
    await store.complete("e", owner="worker", fence=claimed.fence, result="done")
    with pytest.raises(StorageConflictError):
        await store.complete("e", owner="worker", fence=claimed.fence, result="overwrite")
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_task_store_allows_only_one_concurrent_claim(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'task-claim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = TaskStore(SqlAlchemyTaskBackend(factory))
    await store.initialize_storage(engine)
    await store.save_plan(TaskPlan("p", ()))
    await store.add_execution(TaskExecution("e", "p", "n", "ready"))
    results = await asyncio.gather(
        store.claim("e", owner="worker-a"),
        store.claim("e", owner="worker-b"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, TaskExecution) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_task_store_commits_terminal_result_exactly_once(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'task-result.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = TaskStore(SqlAlchemyTaskBackend(factory))
    await store.initialize_storage(engine)
    await store.save_plan(TaskPlan("p", ()))
    await store.add_execution(TaskExecution("e", "p", "n", "ready"))
    claimed = await store.claim("e", owner="worker")
    results = await asyncio.gather(
        store.complete("e", owner="worker", fence=claimed.fence, result="first"),
        store.complete("e", owner="worker", fence=claimed.fence, result="second"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, TaskExecution) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_task_store_reclaims_expired_lease_and_rejects_stale_fence(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'task-reclaim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = TaskStore(SqlAlchemyTaskBackend(factory))
    await store.initialize_storage(engine)
    await store.save_plan(TaskPlan("p", ()))
    await store.add_execution(TaskExecution("e", "p", "n", "ready"))
    stale = await store.claim("e", owner="worker-a", duration=timedelta(seconds=-1))
    current = await store.claim("e", owner="worker-b")
    assert current.fence == stale.fence + 1
    with pytest.raises(StorageConflictError):
        await store.complete("e", owner="worker-a", fence=stale.fence, result=None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_replays_completed_operation(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tools.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ToolStateStore(SqlAlchemyToolStateBackend(factory))
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    claimed = await store.claim("o", owner="worker", duration=timedelta(minutes=1))
    assert claimed.lease.expires_at is not None
    await store.complete("o", owner="worker", fence=claimed.fence, result={"ok": True})
    assert (await store.prepare(operation)).result == {"ok": True}
    with pytest.raises(StorageConflictError):
        await store.complete("o", owner="worker", fence=claimed.fence, result={"ok": False})
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_concurrent_prepare_is_idempotent(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-prepare.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ToolStateStore(SqlAlchemyToolStateBackend(factory))
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", ToolOperationStatus.PREPARED)
    results = await asyncio.gather(
        store.prepare(operation),
        store.prepare(operation),
    )
    assert results[0].id == results[1].id == "o"
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_allows_only_one_concurrent_claim(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-claim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ToolStateStore(SqlAlchemyToolStateBackend(factory))
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    results = await asyncio.gather(
        store.claim("o", owner="worker-a"),
        store.claim("o", owner="worker-b"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ToolOperation) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_commits_terminal_result_exactly_once(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-result.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ToolStateStore(SqlAlchemyToolStateBackend(factory))
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    claimed = await store.claim("o", owner="worker")
    results = await asyncio.gather(
        store.complete("o", owner="worker", fence=claimed.fence, result="first"),
        store.complete("o", owner="worker", fence=claimed.fence, result="second"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ToolOperation) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_reclaims_expired_lease_and_rejects_stale_fence(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-reclaim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ToolStateStore(SqlAlchemyToolStateBackend(factory))
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    stale = await store.claim("o", owner="worker-a", duration=timedelta(seconds=-1))
    current = await store.claim("o", owner="worker-b")
    assert current.fence == stale.fence + 1
    with pytest.raises(StorageConflictError):
        await store.complete("o", owner="worker-a", fence=stale.fence, result=None)
    await engine.dispose()
