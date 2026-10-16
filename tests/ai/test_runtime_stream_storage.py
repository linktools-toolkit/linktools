#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional stream backend assembly and borrowed lifetime."""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.runtime import RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.storage import InMemoryStreamBackend


@pytest.mark.asyncio
async def test_storage_constructors_preserve_borrowed_stream_backend(tmp_path) -> None:
    backend = InMemoryStreamBackend(max_bytes=1024)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'external.db'}")
    plan = RuntimeStoragePlan(execution=RuntimeStorageRoute.sqlite(tmp_path / "plan.db"))
    stores = (
        RuntimeStorage(plan, stream_backend=backend),
        RuntimeStorage.filesystem(tmp_path / "files", stream_backend=backend),
        RuntimeStorage.sqlite(tmp_path / "state.db", stream_backend=backend),
        RuntimeStorage.from_root(tmp_path / "root", stream_backend=backend),
        RuntimeStorage.sql(engine, stream_backend=backend),
        RuntimeStorage.from_plan(plan, stream_backend=backend),
    )
    try:
        for storage in stores:
            assert storage.stream_backend is backend
            await storage.close()
        writer = await backend.open(b"scope", b"producer")
        await writer.append(1, b"still usable")
        batch = await backend.read(b"scope", 0, limit=10, wait_seconds=0)
        assert batch.records == ((1, b"still usable"),)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ready_storage_close_preserves_shared_stream_records(tmp_path) -> None:
    backend = InMemoryStreamBackend(max_bytes=1024)
    storage = RuntimeStorage.sqlite(tmp_path / "state.db", stream_backend=backend)
    await storage.initialize(namespace="runtime", tenant_id="tenant")
    writer = await backend.open(b"scope", b"producer")
    await writer.append(1, b"pending")
    await storage.close()
    await writer.seal(2, b"committed")
    batch = await backend.read(b"scope", 0, limit=10, wait_seconds=0)
    assert batch.records == ((1, b"pending"), (2, b"committed"))
    assert batch.sealed


def test_default_storage_has_no_shared_stream_backend() -> None:
    assert RuntimeStorage.in_memory().stream_backend is None


@pytest.mark.parametrize("route", (RuntimeStorageRoute.memory(), RuntimeStorageRoute.transient()))
def test_nondurable_execution_rejects_shared_stream_backend(route) -> None:
    backend = InMemoryStreamBackend(max_bytes=1024)
    with pytest.raises(ValueError, match="requires durable execution"):
        RuntimeStorage.from_plan(RuntimeStoragePlan(execution=route), stream_backend=backend)
