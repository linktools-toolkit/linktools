#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hot caches and execution coordination have independent borrowed lifetimes."""

from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.persistence import PersistenceStorage, PersistencePlan, PersistenceDomain
from linktools.ai.persistence import Persistence
from linktools.ai.storage import StorageRoute
from linktools.ai.storage import InMemoryCacheBackend, InMemoryStreamBackend
from linktools.ai.capability import ExecutionCapability


@pytest.mark.asyncio
async def test_storage_constructors_preserve_borrowed_hot_cache(tmp_path) -> None:
    backend = InMemoryCacheBackend(max_bytes=1024)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'external.db'}")
    plan = PersistencePlan(execution=StorageRoute.sqlite(tmp_path / "plan.db", cache=backend))
    stores = (
        PersistenceStorage(plan),
        PersistenceStorage.in_memory(cache=backend),
        PersistenceStorage.filesystem(tmp_path / "files", cache=backend),
        PersistenceStorage.sqlite(tmp_path / "state.db", cache=backend),
        PersistenceStorage.from_root(tmp_path / "root", cache=backend),
        PersistenceStorage.sql(engine, cache=backend),
        PersistenceStorage.from_plan(plan),
    )
    try:
        for storage in stores:
            assert storage.plan.route(PersistenceDomain.EXECUTION).cache is backend
            await Persistence(storage).close()
        await backend.put("scope", b"still usable")
        assert await backend.get("scope") == b"still usable"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ready_storage_close_preserves_borrowed_hot_cache(tmp_path) -> None:
    backend = InMemoryCacheBackend(max_bytes=1024)
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "state.db", cache=backend))
    await storage.initialize(namespace="runtime", tenant_id="tenant")
    await backend.put("scope", b"pending")
    await storage.close()
    assert await backend.get("scope") == b"pending"
    await backend.put("scope", b"committed")
    assert await backend.get("scope") == b"committed"


def test_default_storage_has_no_hot_cache() -> None:
    assert all(PersistenceStorage.in_memory().plan.route(domain).cache is None for domain in PersistenceDomain)


@pytest.mark.parametrize("route_factory", (StorageRoute.memory, StorageRoute.transient))
def test_stream_only_backend_belongs_to_execution_coordination(route_factory) -> None:
    backend = InMemoryStreamBackend(max_bytes=1024)
    storage = PersistenceStorage.in_memory(coordination=backend)
    assert storage.coordination is backend
    with pytest.raises(TypeError, match="cache must provide get, put and delete"):
        PersistenceStorage.from_plan(PersistencePlan(execution=route_factory(cache=backend)))


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
    with pytest.raises(TypeError, match="cache must provide get, put and delete"):
        PersistenceStorage.in_memory(cache=cache)


@pytest.mark.parametrize("methods", (
    ("get", "put", "delete"),
    ("append", "read_batch", "ack"),
    ("open_stream", "read_stream"),
))
def test_noncallable_cache_capabilities_are_rejected(methods: tuple[str, ...]) -> None:
    cache = SimpleNamespace(**dict.fromkeys(methods, "not callable"))
    with pytest.raises(TypeError, match="cache must provide get, put and delete"):
        PersistenceStorage.in_memory(cache=cache)


@pytest.mark.parametrize("methods", (
    ("get", "put", "delete"),
    ("claim_turn", "read_turn", "renew_turn", "release_turn", "cancel_turn"),
    ("open_stream", "read_stream"),
))
def test_storage_accepts_each_complete_coordination_capability(methods: tuple[str, ...]) -> None:
    async def operation(*args: object, **kwargs: object) -> None:
        raise AssertionError("construction must not call coordination methods")

    backend = SimpleNamespace(**{method: operation for method in methods})
    assert PersistenceStorage.in_memory(coordination=backend).coordination is backend


@pytest.mark.parametrize("methods", (
    ("get", "put", "delete"),
    ("claim_turn", "read_turn", "renew_turn", "release_turn", "cancel_turn"),
    ("open_stream", "read_stream"),
))
@pytest.mark.parametrize("invalid", (None, "not callable"))
def test_storage_rejects_partial_coordination_capabilities(
    methods: tuple[str, ...], invalid: str | None,
) -> None:
    async def operation(*args: object, **kwargs: object) -> None:
        raise AssertionError("construction must not call coordination methods")

    backend = SimpleNamespace(**{method: operation for method in methods})
    setattr(backend, methods[-1], invalid)
    with pytest.raises(TypeError, match="coordination must implement"):
        PersistenceStorage.in_memory(coordination=backend)
