#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal aggregation remains transactional and bounded by the target node."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._task_summary import _TaskSummary
from linktools.ai.task import TaskNode

from .test_task_io_bounded_reads import _admit
from .test_task_io_measurements import _measure


@pytest.fixture(params=("memory", "local_sqlite", "sqlite"))
def storage(tmp_path: Path, request: pytest.FixtureRequest) -> PersistenceStorage:
    if request.param == "memory":
        return PersistenceStorage.in_memory()
    if request.param == "local_sqlite":
        return PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    return PersistenceStorage.sqlite(tmp_path / "state.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("width", (2, 8, 200))
async def test_terminal_summary_completion_rows_are_linear(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    async with Persistence.open("summary-budget", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        await _admit(state, tuple(TaskNode(f"node-{index}") for index in range(width)))
        repository = state.task.tasks
        totals = Counter()
        for index in range(width):
            lease = await repository.claim("graph", f"node-{index}", tenant_id="tenant", owner="worker", lease_seconds=300)
            with _measure(monkeypatch) as counts:
                await repository.complete(lease, tenant_id="tenant", execution_id=f"execution-{index}", result_digest="a" * 64)
            totals.update(counts)
        assert totals["record_rows"] <= 4 * width
        assert totals["write_commits"] == width
        assert (await repository.graph_state("graph", tenant_id="tenant")).status is TaskStatus.SUCCEEDED
        print("TERMINAL_SUMMARY_IO", width, dict(totals))


@pytest.mark.asyncio
async def test_terminal_summary_missing_projection_rebuilds_from_states(storage: PersistenceStorage) -> None:
    async with Persistence.open("summary-rebuild", storage=storage, tenant_id="tenant") as state:
        await _admit(state, (TaskNode("first"), TaskNode("second")))
        repository = state.task.tasks
        lease = await repository.claim("graph", "first", tenant_id="tenant", owner="worker", lease_seconds=300)

        async def remove(transaction):
            record = await transaction.get_record(repository._graph_key("graph"))
            data = dict(record.data)
            data.pop("task_summary")
            assert await transaction.replace_record(replace(record, data=data, storage_revision=record.storage_revision + 1), expected_storage_revision=record.storage_revision)

        await repository.state_store.mutate(remove)
        await repository.complete(lease, tenant_id="tenant", execution_id="first-execution", result_digest="a" * 64)
        full = await repository.graph_state("graph", tenant_id="tenant")
        assert full.status is TaskStatus.PENDING
        record = await repository._record(repository._graph_key("graph"))
        _TaskSummary.read(record).validate(full.node_states, full.nodes)
        second = await repository.claim("graph", "second", tenant_id="tenant", owner="worker", lease_seconds=300)
        await repository.handoff_execution(second, tenant_id="tenant", execution_id="second-execution")
        await repository.complete(None, graph_id="graph", node_id="second", tenant_id="tenant", execution_id="second-execution", result_digest="b" * 64)
        assert (await repository.graph_state("graph", tenant_id="tenant")).status is TaskStatus.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("null", "mismatch"))
async def test_terminal_summary_public_read_rejects_corruption(storage: PersistenceStorage, damage: str) -> None:
    async with Persistence.open("summary-corruption", storage=storage, tenant_id="tenant") as state:
        await _admit(state, (TaskNode("node"),))
        repository = state.task.tasks

        async def corrupt(transaction):
            record = await transaction.get_record(repository._graph_key("graph"))
            data = dict(record.data)
            if damage == "null":
                data["task_summary"] = None
            else:
                summary = _TaskSummary.read(record)
                counts = dict(summary.counts)
                counts[TaskStatus.SUCCEEDED] += 1
                data = dict(replace(summary, counts=counts).attach(record).data)
            assert await transaction.replace_record(replace(record, data=data, storage_revision=record.storage_revision + 1), expected_storage_revision=record.storage_revision)

        await repository.state_store.mutate(corrupt)
        with pytest.raises(AIError) as caught:
            await repository.graph_state("graph", tenant_id="tenant")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_header", (False, True))
async def test_terminal_summary_scheduler_retries_mixed_generation(monkeypatch: pytest.MonkeyPatch, cached_header: bool) -> None:
    async with Persistence.open("summary-review", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        repository = state.task.tasks
        await _admit(state, (TaskNode("first"), TaskNode("second")))
        old = await repository._record(repository._graph_key("graph"))
        await repository.claim("graph", "first", tenant_id="tenant", owner="worker", lease_seconds=300)
        store = repository.state_store
        original = type(store).mutate
        injected = False

        class CachedHeaderReadCommitted:
            def __init__(self, transaction):
                self.transaction = transaction
                self.header_reads = 0

            def __getattr__(self, name):
                return getattr(self.transaction, name)

            async def get_record(self, key):
                if key == old.key:
                    self.header_reads += 1
                    if cached_header or self.header_reads == 1:
                        return old
                return await self.transaction.get_record(key)

        async def mutate(current, fn):
            nonlocal injected
            if current is not store or injected:
                return await original(current, fn)
            injected = True
            async def wrapped(transaction):
                return await fn(CachedHeaderReadCommitted(transaction))
            return await original(current, wrapped)

        monkeypatch.setattr(type(store), "mutate", mutate)
        view = await repository.scheduler_state("graph", tenant_id="tenant")
        assert injected
        assert view.status is TaskStatus.RUNNING


@pytest.mark.asyncio
async def test_terminal_summary_sparse_and_full_counts_are_equivalent() -> None:
    async with Persistence.open("summary-counts", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await _admit(state, (TaskNode("first"), TaskNode("second")))
        repository = state.task.tasks
        record = await repository._record(repository._graph_key("graph"))
        summary = _TaskSummary.read(record)
        sparse = record.data["task_summary"]
        assert all(count > 0 for count in sparse["counts"].values())
        assert len(sparse["counts"]) < len(TaskStatus)
        full = {**sparse, "counts": {status.value: summary.counts[status] for status in TaskStatus}}
        assert _TaskSummary.read(replace(record, data={**record.data, "task_summary": full})) == summary
        for invalid in (None, True, -1):
            counts = {**sparse["counts"], TaskStatus.RUNNING.value: invalid}
            with pytest.raises(AIError) as caught:
                _TaskSummary.read(replace(record, data={**record.data, "task_summary": {**sparse, "counts": counts}}))
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("complete", "handoff"))
async def test_terminal_summary_delta_retries_old_header_new_target(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    async with Persistence.open("summary-delta-race", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await _admit(state, (TaskNode("first"), TaskNode("second")))
        repository = state.task.tasks
        old = await repository._record(repository._graph_key("graph"))
        lease = await repository.claim("graph", "first", tenant_id="tenant", owner="worker", lease_seconds=300)
        store = repository.state_store
        original = type(store).mutate
        injected = False

        class OldHeaderNewTarget:
            def __init__(self, transaction):
                self.transaction = transaction

            def __getattr__(self, name):
                return getattr(self.transaction, name)

            async def get_records(self, keys):
                records = dict(await self.transaction.get_records(keys))
                if old.key in records:
                    records[old.key] = old
                return records

        async def mutate(current, fn):
            nonlocal injected
            if current is not store or injected:
                return await original(current, fn)
            injected = True

            async def wrapped(transaction):
                return await fn(OldHeaderNewTarget(transaction))

            return await original(current, wrapped)

        monkeypatch.setattr(type(store), "mutate", mutate)
        if operation == "complete":
            result = await repository.complete(lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64)
            assert result.status is TaskStatus.SUCCEEDED
        else:
            result = await repository.handoff_execution(lease, tenant_id="tenant", execution_id="execution")
            assert result.status is TaskStatus.WAITING
        assert injected
        full = await repository.graph_state("graph", tenant_id="tenant")
        assert next(node for node in full.node_states if node.node_id == "first").status is result.status
