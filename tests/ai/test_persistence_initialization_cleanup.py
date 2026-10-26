#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failed initialization retains cleanup ownership without masking its cause."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence import _resource_scope
from linktools.ai.persistence._memory import InMemoryStateStorageGroup, InMemoryStateStore
from linktools.ai.persistence.runs import RuntimeAgentRunStore


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("resource", "store", "run_store"))
async def test_initialization_failure_retains_resources_for_close_retry(
    monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    persistence = Persistence(PersistenceStorage.in_memory())
    primary = ValueError("initialization failed")
    cleanup = OSError("resource still busy")
    blocked = True
    released = asyncio.Event()
    original_close = InMemoryStateStorageGroup.close

    async def close(group: InMemoryStateStorageGroup) -> None:
        if blocked:
            raise cleanup
        await original_close(group)
        released.set()

    async def fail_initialize(*args: object, **kwargs: object) -> None:
        raise primary

    monkeypatch.setattr(InMemoryStateStorageGroup, "close", close)
    if phase == "resource":
        original_initialize = _resource_scope.StorageResourceScope.initialize

        async def initialize(scope: _resource_scope.StorageResourceScope, *args: object, **kwargs: object) -> None:
            await original_initialize(scope, *args, **kwargs)
            raise primary

        monkeypatch.setattr(_resource_scope.StorageResourceScope, "initialize", initialize)
    elif phase == "store":
        monkeypatch.setattr(InMemoryStateStore, "initialize", fail_initialize)
    else:
        monkeypatch.setattr(RuntimeAgentRunStore, "initialize", fail_initialize)
    try:
        with pytest.raises(ValueError) as caught:
            await persistence.initialize(namespace="initialization", tenant_id="tenant")
        assert caught.value is primary
        assert caught.value.__cause__ is cleanup
        assert not persistence.ready
        assert not released.is_set()
        blocked = False
        await persistence.close()
        assert released.is_set()
        await persistence.close()
    finally:
        blocked = False
        await persistence.close()


@pytest.mark.asyncio
async def test_partial_store_close_failure_preserves_its_underlying_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = Persistence(PersistenceStorage.in_memory())
    primary = ValueError("store initialization failed")
    blocked = True
    group_released = asyncio.Event()
    original_store_close = InMemoryStateStore.close
    original_group_close = InMemoryStateStorageGroup.close

    async def initialize(store: InMemoryStateStore) -> None:
        raise primary

    async def close_store(store: InMemoryStateStore) -> None:
        if blocked:
            raise OSError("store close failed")
        await original_store_close(store)

    async def close_group(group: InMemoryStateStorageGroup) -> None:
        await original_group_close(group)
        group_released.set()

    monkeypatch.setattr(InMemoryStateStore, "initialize", initialize)
    monkeypatch.setattr(InMemoryStateStore, "close", close_store)
    monkeypatch.setattr(InMemoryStateStorageGroup, "close", close_group)
    try:
        with pytest.raises(ValueError) as caught:
            await persistence.initialize(namespace="initialization", tenant_id="tenant")
        assert caught.value is primary
        assert not group_released.is_set()
        blocked = False
        await persistence.close()
        assert group_released.is_set()
    finally:
        blocked = False
        await persistence.close()


@pytest.mark.asyncio
async def test_cancelled_initialization_settles_cleanup_and_retains_failed_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = Persistence(PersistenceStorage.in_memory())
    initializing = asyncio.Event()
    closing = asyncio.Event()
    finish_close = asyncio.Event()
    blocked = True
    released = asyncio.Event()
    cleanup = OSError("cleanup failed")
    original_close = InMemoryStateStorageGroup.close

    async def initialize(store: InMemoryStateStore) -> None:
        initializing.set()
        await asyncio.Event().wait()

    async def close(group: InMemoryStateStorageGroup) -> None:
        closing.set()
        await finish_close.wait()
        if blocked:
            raise cleanup
        await original_close(group)
        released.set()

    monkeypatch.setattr(InMemoryStateStore, "initialize", initialize)
    monkeypatch.setattr(InMemoryStateStorageGroup, "close", close)
    task = asyncio.create_task(persistence.initialize(namespace="initialization", tenant_id="tenant"))
    try:
        await asyncio.wait_for(initializing.wait(), 2)
        task.cancel("startup cancelled")
        await asyncio.wait_for(closing.wait(), 2)
        task.cancel("cleanup waiter cancelled")
        await asyncio.sleep(0)
        assert not task.done()
        finish_close.set()
        with pytest.raises(asyncio.CancelledError, match="startup cancelled"):
            await task
        assert not released.is_set()
        assert not persistence.ready
        blocked = False
        await persistence.close()
        assert released.is_set()
    finally:
        blocked = False
        finish_close.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await persistence.close()


@pytest.mark.asyncio
async def test_sqlite_provisioning_failure_retains_temporary_engine_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncEngine
    from linktools.ai.storage import SqlStorageContext

    persistence = Persistence(PersistenceStorage.sqlite(tmp_path / "state.sqlite"))
    primary = ValueError("database initialization failed")
    blocked = True
    contexts: list[SqlStorageContext] = []
    original_dispose = AsyncEngine.dispose
    original_create = _resource_scope.create_sql_storage_context

    def create_context(engine: AsyncEngine, *, owns_engine: bool = False) -> SqlStorageContext:
        context = original_create(engine, owns_engine=owns_engine)
        contexts.append(context)
        return context

    async def initialize(context: SqlStorageContext) -> None:
        raise primary

    async def dispose(engine: AsyncEngine, close: bool = True) -> None:
        if blocked:
            raise OSError("engine cleanup failed")
        await original_dispose(engine, close=close)

    monkeypatch.setattr(_resource_scope, "create_sql_storage_context", create_context)
    monkeypatch.setattr(SqlStorageContext, "initialize", initialize)
    monkeypatch.setattr(AsyncEngine, "dispose", dispose)
    try:
        with pytest.raises(ValueError) as caught:
            await persistence.initialize(namespace="initialization", tenant_id="tenant")
        assert caught.value is primary
        assert contexts and not contexts[0].closed
        assert tuple(tmp_path.rglob("*.init-*"))
        blocked = False
        await persistence.close()
        assert all(context.closed for context in contexts)
        assert not tuple(tmp_path.rglob("*.init-*"))
    finally:
        blocked = False
        await persistence.close()


@pytest.mark.asyncio
async def test_sqlite_engine_construction_failure_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlalchemy.ext.asyncio

    persistence = Persistence(PersistenceStorage.sqlite(tmp_path / "state.sqlite"))
    primary = ImportError("database driver unavailable")

    def create_engine(*args: object, **kwargs: object) -> None:
        raise primary

    monkeypatch.setattr(sqlalchemy.ext.asyncio, "create_async_engine", create_engine)
    with pytest.raises(ImportError) as caught:
        await persistence.initialize(namespace="initialization", tenant_id="tenant")
    assert caught.value is primary
    assert not tuple(tmp_path.rglob("*.init-*"))
    assert not persistence.ready
    await persistence.close()
