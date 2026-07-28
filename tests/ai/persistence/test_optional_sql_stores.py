from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import pytest

from linktools.ai.tasks.models import TaskExecution, TaskPlan
from linktools.ai.tasks.persistence.sqlalchemy import SqlAlchemyTaskStore
from linktools.ai.tool.persistence.sqlalchemy import SqlAlchemyToolStateStore
from linktools.ai.tool.state import ToolOperation, ToolOperationStatus


@pytest.mark.asyncio
async def test_sql_task_store_fences_claims(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyTaskStore(factory)
    await store.initialize_storage(engine)
    await store.save_plan(TaskPlan("p", ()))
    await store.add_execution(TaskExecution("e", "p", "n", "ready"))
    claimed = await store.claim("e", owner="worker")
    with pytest.raises(ValueError):
        await store.complete("e", owner="other", fence=claimed.fence, result=None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_replays_completed_operation(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tools.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyToolStateStore(factory)
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    claimed = await store.claim("o", owner="worker")
    await store.complete("o", owner="worker", fence=claimed.fence, result={"ok": True})
    assert (await store.prepare(operation)).result == {"ok": True}
    await engine.dispose()
