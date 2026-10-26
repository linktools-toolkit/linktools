#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Unused Runtime services perform no business IO or background execution."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TypeVar

import pytest

from linktools.ai.core import TaskStatus

from linktools.ai.runtime import Runtime
from linktools.ai.task_engine import TaskEngine
from linktools.ai.evaluation_engine import Evaluations
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence import Persistence
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence.state import StateStore
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext

ValueT = TypeVar("ValueT")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_idle_services_do_not_read_write_or_launch_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "runtime.db"))
    initialize = Persistence.initialize

    async def initialize_and_guard(
        state: Persistence, *, namespace: str, tenant_id: str, read_only: bool = False,
    ) -> None:
        # Shared schema provisioning is permitted; domain business IO is not.
        await initialize(
            state, namespace=namespace, tenant_id=tenant_id, read_only=read_only,
        )
        service_stores = {
            state.task.tasks.state_store,
            state.evaluation.records.state_store,
            state.artifact.records.state_store,
        }

        async def unused_service_io(*args: object, **kwargs: object) -> object:
            raise AssertionError("an unused Runtime service accessed business state")

        for store in service_stores:
            for operation in ("read_record", "read_records", "read", "mutate"):
                monkeypatch.setattr(store, operation, unused_service_io)

        def guard_group_operation(
            operation: Callable[..., Awaitable[ValueT]],
        ) -> Callable[..., Awaitable[ValueT]]:
            async def guarded(
                stores: Sequence[StateStore], *args: object, **kwargs: object,
            ) -> ValueT:
                assert not service_stores.intersection(stores), (
                    "an unused Runtime service joined a business transaction"
                )
                return await operation(stores, *args, **kwargs)
            return guarded

        for group in {store.storage_group for store in service_stores}:
            for name in ("read_many", "mutate"):
                monkeypatch.setattr(group, name, guard_group_operation(getattr(group, name)))

    monkeypatch.setattr(Persistence, "initialize", initialize_and_guard)
    async with Persistence.open('idle-services', storage=storage) as persistence_73, Runtime.open(persistence=persistence_73) as runtime, TaskEngine.open(persistence=persistence_73) as engine, Evaluations.open(engine) as evaluations:
        assert engine.persistence is persistence_73
        assert evaluations is not None
        assert persistence_73.artifact is not None
        await asyncio.sleep(0)
        launcher = engine._graph_service._launcher
        assert launcher is not None
        assert not launcher._graphs
        assert not evaluations._watchers
    assert not runtime._state.ready


@pytest.mark.asyncio
async def test_task_service_activates_on_call_without_agent_capabilities() -> None:
    calls: list[str] = []

    async def run(context: TaskNodeContext[None]) -> str:
        calls.append(context.node_id)
        return "task result"

    task = Task("service.activation", run, effect_policy="none")
    graph = TaskGraph("service-activation", (TaskNode("run", task=task),))
    async with Persistence.open('service-activation', storage=PersistenceStorage.in_memory()) as persistence_95, TaskEngine.open(task, persistence=persistence_95) as engine, Evaluations.open(engine) as evaluations:
        assert calls == []
        execution = await engine.start(graph, idempotency_key="activate")
        assert (await execution.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        assert await execution.result("run") == "task result"
        assert calls == ["run"]
        assert not evaluations._watchers
