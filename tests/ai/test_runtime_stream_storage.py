#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional stream backend assembly and borrowed lifetime."""

from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.runtime import RuntimeStorage, RuntimeStoragePlan, RuntimeDomain
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.storage import StorageRoute
from linktools.ai.storage import InMemoryStreamBackend


@pytest.mark.asyncio
async def test_storage_constructors_preserve_borrowed_stream_backend(tmp_path) -> None:
    backend = InMemoryStreamBackend(max_bytes=1024)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'external.db'}")
    plan = RuntimeStoragePlan(execution=StorageRoute.sqlite(tmp_path / "plan.db", cache=backend))
    stores = (
        RuntimeStorage(plan),
        RuntimeStorage.in_memory(cache=backend),
        RuntimeStorage.filesystem(tmp_path / "files", cache=backend),
        RuntimeStorage.sqlite(tmp_path / "state.db", cache=backend),
        RuntimeStorage.from_root(tmp_path / "root", cache=backend),
        RuntimeStorage.sql(engine, cache=backend),
        RuntimeStorage.from_plan(plan),
    )
    try:
        for storage in stores:
            assert storage.plan.route(RuntimeDomain.EXECUTION).cache is backend
            await RuntimeState(storage).close()
        writer = await backend.open_stream(b"scope", b"producer")
        await writer.append(1, b"still usable")
        batch = await backend.read_stream(b"scope", 0, limit=10, wait_seconds=0)
        assert batch.records == ((1, b"still usable"),)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ready_storage_close_preserves_shared_stream_records(tmp_path) -> None:
    backend = InMemoryStreamBackend(max_bytes=1024)
    storage = RuntimeState(RuntimeStorage.sqlite(tmp_path / "state.db", cache=backend))
    await storage.initialize(namespace="runtime", tenant_id="tenant")
    writer = await backend.open_stream(b"scope", b"producer")
    await writer.append(1, b"pending")
    await storage.close()
    await writer.seal(2, b"committed")
    batch = await backend.read_stream(b"scope", 0, limit=10, wait_seconds=0)
    assert batch.records == ((1, b"pending"), (2, b"committed"))
    assert batch.sealed


def test_default_storage_has_no_shared_stream_backend() -> None:
    assert all(RuntimeStorage.in_memory().plan.route(domain).cache is None for domain in RuntimeDomain)


@pytest.mark.parametrize("route_factory", (StorageRoute.memory, StorageRoute.transient))
def test_nondurable_execution_accepts_unused_cache_capabilities(route_factory) -> None:
    backend = InMemoryStreamBackend(max_bytes=1024)
    storage = RuntimeStorage.from_plan(RuntimeStoragePlan(execution=route_factory(cache=backend)))
    assert storage.plan.route(RuntimeDomain.EXECUTION).cache is backend


@pytest.mark.parametrize("methods", (
    ("get", "put", "delete"),
    ("append", "read_batch", "ack"),
    ("open_stream", "read_stream"),
))
@pytest.mark.parametrize("invalid", (None, "not callable"))
def test_partial_cache_capabilities_fail_at_storage_construction(
    methods: tuple[str, ...], invalid: str | None,
) -> None:
    async def operation(*args: object, **kwargs: object) -> None:
        raise AssertionError("construction must not call cache methods")

    cache = SimpleNamespace(**{method: operation for method in methods})
    setattr(cache, methods[-1], invalid)
    with pytest.raises(TypeError, match="cache must implement all methods"):
        RuntimeStorage.in_memory(cache=cache)


@pytest.mark.parametrize("methods", (
    ("get", "put", "delete"),
    ("append", "read_batch", "ack"),
    ("open_stream", "read_stream"),
))
def test_noncallable_cache_capabilities_are_rejected(methods: tuple[str, ...]) -> None:
    cache = SimpleNamespace(**dict.fromkeys(methods, "not callable"))
    with pytest.raises(TypeError, match="cache must implement all methods"):
        RuntimeStorage.in_memory(cache=cache)
