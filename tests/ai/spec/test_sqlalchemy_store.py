import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.spec.document import SpecDocument, SpecDocumentInfo
from linktools.ai.spec.persistence.local import LocalSpecBackend
from linktools.ai.spec.persistence.sqlalchemy import SqlAlchemySpecBackend
from linktools.ai.spec.store import SpecStore
from linktools.ai.storage.multi import StorageLayer


@pytest.mark.asyncio
async def test_sqlalchemy_capability_store_round_trip_and_revision():
    engine = create_async_engine("sqlite+aiosqlite://")
    store = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await store.initialize_storage(engine)
    assert await store.current_revision() == 0
    entry = SpecDocument(SpecDocumentInfo("agent/a", "agent", 1, "e1"), b"body")
    await store.put(entry)
    assert await store.get("agent/a") == entry
    assert await store.get_many(("agent/a", "missing")) == {
        "agent/a": entry,
    }
    assert await store.list_info(kind="agent") == (entry.info,)
    assert await store.current_revision() == 1
    changes = await store.list_changes(after_revision=0, through_revision=1)
    assert changes[0].info == entry.info
    await store.delete("agent/a")
    assert await store.get("agent/a") is None
    assert await store.current_revision() == 2
    await engine.dispose()


@pytest.mark.asyncio
async def test_spec_store_explicitly_assembles_sql_backend_capabilities():
    engine = create_async_engine("sqlite+aiosqlite://")
    backend = SqlAlchemySpecBackend(
        async_sessionmaker(engine, expire_on_commit=False)
    )
    store = SpecStore(
        backend,
        writer=backend,
        revision=backend,
        changes=backend,
    )
    await store.initialize_storage(engine)
    entry = SpecDocument(SpecDocumentInfo("agent/a", "agent", 1, "e1"), b"body")
    await store.put(entry)
    assert await store.get("agent/a") == entry
    assert await store.current_revision() == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_versioned_sql_primary_with_unversioned_local_overlay(tmp_path):
    engine = create_async_engine("sqlite+aiosqlite://")
    primary = SqlAlchemySpecBackend(
        async_sessionmaker(engine, expire_on_commit=False)
    )
    overlay = LocalSpecBackend(tmp_path)
    store = SpecStore(
        primary,
        writer=primary,
        revision=primary,
        changes=primary,
        overlays=(
            StorageLayer(
                overlay,
                initializer=lambda *_: overlay.initialize_storage(),
            ),
        ),
    )
    await store.initialize_storage(engine)
    primary_entry = SpecDocument(
        SpecDocumentInfo("agent/shared", "agent", 1, "primary"),
        b"primary",
    )
    overlay_shared = SpecDocument(
        SpecDocumentInfo("agent/shared", "agent", 1, "overlay"),
        b"overlay",
    )
    overlay_only = SpecDocument(
        SpecDocumentInfo("agent/local", "agent", 1, "local"),
        b"local",
    )
    await primary.put(primary_entry)
    await overlay.reset((overlay_shared, overlay_only))
    assert await store.get("agent/shared") == primary_entry
    assert await store.get("agent/local") == overlay_only
    await store.delete("agent/shared")
    assert await store.get("agent/shared") == overlay_shared
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlalchemy_capability_concurrent_puts_preserve_revision_order(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'capability.db'}")
    store = SqlAlchemySpecBackend(async_sessionmaker(engine, expire_on_commit=False))
    await store.initialize_storage(engine)
    first = SpecDocument(SpecDocumentInfo("agent/a", "agent", 1, "e1"), b"first")
    second = SpecDocument(SpecDocumentInfo("agent/a", "agent", 2, "e2"), b"second")
    await asyncio.gather(store.put(first), store.put(second))
    revision = await store.current_revision()
    changes = await store.list_changes(after_revision=0, through_revision=revision)
    assert revision == 2
    assert [change.revision for change in changes] == [1, 2]
    assert (await store.get("agent/a")).info == changes[-1].info
    await engine.dispose()
