#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A local writer request survives declarative routing without becoming authority."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistencePlan, PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore, StorageRoute


@pytest.mark.asyncio
async def test_sqlite_route_request_survives_plan_rebuild_and_cache_selection(tmp_path: Path) -> None:
    path = tmp_path / "owned.db"
    cache = InMemoryCacheBackend(max_bytes=1024)
    requested = StorageRoute.sqlite(path, exclusive_writer=True)
    selected = requested.with_cache(cache)
    plan = replace(PersistencePlan.from_default(selected), conversation=StorageRoute.sqlite(tmp_path / "conversation.db"))
    objects = InMemoryObjectStore("borrowed")
    storage = PersistenceStorage.from_plan(plan, object_store=objects)
    async with Persistence.open("route-owner", storage=storage) as state:
        assert state.exclusive_execution_writer
        assert state.object_store(PersistenceDomain.EXECUTION) is objects
        assert state.plan.execution.exclusive_writer
        assert state.plan.execution.cache is cache
        assert state.execution.executions.state_store.storage_group is not state.conversation.sessions.state_store.storage_group
        contender = Persistence(PersistenceStorage(plan, object_store=objects))
        try:
            with pytest.raises(AIError) as rejected:
                await contender.initialize(namespace="route-owner", tenant_id="default")
            assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        finally:
            await contender.close()
    async with Persistence.open("route-owner", storage=PersistenceStorage(plan, object_store=objects)) as reopened:
        assert reopened.exclusive_execution_writer


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", (False, True))
async def test_conflicting_sqlite_route_requests_fail_before_creating_any_source(tmp_path: Path, read_only: bool) -> None:
    path = tmp_path / "shared.db"
    plan = PersistencePlan(execution=StorageRoute.sqlite(path, exclusive_writer=True), conversation=StorageRoute.sqlite(path))
    state = Persistence(PersistenceStorage(plan))
    try:
        with pytest.raises(ValueError, match="exclusive_writer"):
            await state.initialize(namespace="conflict", tenant_id="default", read_only=read_only)
        assert not list(tmp_path.iterdir())
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("exclusive", (False, True))
async def test_sqlite_route_read_only_never_obtains_writer_permission(tmp_path: Path, exclusive: bool) -> None:
    route = StorageRoute.sqlite(tmp_path / "source.db", exclusive_writer=exclusive)
    storage = PersistenceStorage(PersistencePlan.from_default(route))
    async with Persistence.open("readonly", storage=storage) as writer:
        assert writer.exclusive_execution_writer is exclusive
        before = route.path.read_bytes()
        async with Persistence.open("readonly", storage=storage, read_only=True) as observer:
            assert not observer.exclusive_execution_writer
            assert route.path.read_bytes() == before
        if not exclusive:
            async with Persistence.open("readonly", storage=storage) as peer:
                assert not peer.exclusive_execution_writer


def test_sqlite_route_rejects_non_boolean_writer_requests(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exclusive_writer"):
        StorageRoute.sqlite(tmp_path / "source.db", exclusive_writer=1)
