import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.capability.persistence.sqlalchemy import SqlAlchemyCapabilityStore
from linktools.ai.capability.entries import CapabilityEntry, CapabilityEntryInfo


@pytest.mark.asyncio
async def test_sqlalchemy_capability_store_round_trip_and_revision():
    engine = create_async_engine("sqlite+aiosqlite://")
    store = SqlAlchemyCapabilityStore(async_sessionmaker(engine, expire_on_commit=False))
    await store.initialize_storage(engine)
    entry = CapabilityEntry(CapabilityEntryInfo("agent/a", "agent", 1, "e1"), b"body")
    await store.put(entry)
    assert await store.get("agent/a") == entry
    assert await store.list_info(kind="agent") == (entry.info,)
    assert await store.current_revision() == 1
    changes = await store.list_changes(after_revision=0, through_revision=1)
    assert changes[0].info == entry.info
    await store.delete("agent/a")
    assert await store.get("agent/a") is None
    assert await store.current_revision() == 2
    await engine.dispose()
