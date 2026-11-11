#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency metadata shares a bounded read without changing result semantics."""

import asyncio
from collections import Counter
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.task import (
    LocalTaskGraphLauncher, TaskGraph, TaskGraphAdmission, TaskGraphLimits,
    TaskGraphRequest, TaskNode, TaskResultRecord,
)

from .test_task_io_bounded_reads import _admit


def _storage(backend: str, path: Path) -> PersistenceStorage:
    if backend == "memory":
        return PersistenceStorage.in_memory()
    if backend == "local_sqlite":
        return PersistenceStorage.from_root(path, exclusive_writer=True)
    return PersistenceStorage.sqlite(path / "state.sqlite")


async def _prepare(state: Persistence, *, width: int = 2) -> TaskNode:
    target = TaskNode("target", ("source",))
    await _admit(state, (TaskNode("source"), target) + tuple(
        TaskNode(f"unrelated-{i}") for i in range(width - 2)
    ))
    repository = state.task.tasks
    lease = await repository.claim("graph", "source", tenant_id="tenant", owner="worker", lease_seconds=30)
    await repository.complete(lease, tenant_id="tenant", execution_id="source-execution", result_digest="a" * 64)
    return target


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_dependency_records_reuse_one_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    async with Persistence.open("dependency-read", storage=_storage(backend, tmp_path), tenant_id="tenant") as state:
        await _prepare(state)
        repository = state.task.tasks
        expected_states = await repository.get_node_states("graph", ("target", "source"), tenant_id="tenant")
        expected_results = await repository.get_results("graph", ("source",), tenant_id="tenant")
        read = repository.state_store.read
        seen = []
        kinds = set()

        async def observed_read(callback):
            async def observed_callback(transaction):
                seen.append(transaction)
                get_records = transaction.get_records

                async def observed_records(keys):
                    records = await get_records(keys)
                    kinds.update(record.kind for record in records.values())
                    return records

                with monkeypatch.context() as patch:
                    patch.setattr(transaction, "get_records", observed_records)
                    return await callback(transaction)
            return await read(observed_callback)

        monkeypatch.setattr(repository.state_store, "read", observed_read)
        assert await repository.get_dependency_records("graph", ("target", "source"), tenant_id="tenant") == (
            expected_states, expected_results,
        )
        assert len(seen) == 1
        assert kinds == {"task_node_state", "task_node_definition", "task_result"}


@pytest.mark.asyncio
async def test_dependency_context_uses_one_read_scope(tmp_path: Path) -> None:
    width = 3
    async with Persistence.open("dependency-budget", storage=_storage("sqlite", tmp_path), tenant_id="tenant") as state:
        target = await _prepare(state, width=width)
        repository = state.task.tasks
        engine = repository.state_store.storage_group.context.engine.sync_engine
        counts = Counter()

        def sql(_connection, _cursor, statement, _parameters, _context, _many):
            counts[statement.lstrip().split()[0].upper()] += 1

        def begin(_connection):
            counts["transactions"] += 1

        def commit(_connection):
            counts["commits"] += 1

        def checkout(*_args):
            counts["checkouts"] += 1

        listeners = (("before_cursor_execute", sql), ("begin", begin), ("commit", commit), ("checkout", checkout))
        for name, callback in listeners:
            event.listen(engine, name, callback)
        try:
            results, statuses = await LocalTaskGraphLauncher(repository, None, owner="worker")._dependency_context(
                "graph", target, tenant_id="tenant",
            )
        finally:
            for name, callback in listeners:
                event.remove(engine, name, callback)
        assert set(results) == set(statuses) == {"source"}
        assert results["source"].execution_id == "source-execution"
        assert results["source"].result_digest == "a" * 64
        print("DEPENDENCY_READ_SCOPE", width, dict(counts))
        assert counts["transactions"] == counts["commits"] == counts["checkouts"] == 1
        assert 1 <= counts["SELECT"] <= 2
        assert not counts["INSERT"] and not counts["UPDATE"] and not counts["DELETE"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
@pytest.mark.parametrize("fault", ("error", "cancel"))
async def test_dependency_read_failure_releases_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, fault: str,
) -> None:
    async with Persistence.open("dependency-failure", storage=_storage(backend, tmp_path), tenant_id="tenant") as state:
        await _prepare(state)
        repository = state.task.tasks
        read = repository.state_store.read
        entered = asyncio.Event()

        async def interrupted_read(callback):
            async def interrupted_callback(transaction):
                get_records = transaction.get_records

                async def interrupted_records(keys):
                    records = await get_records(keys)
                    if any(record.kind == "task_result" for record in records.values()):
                        entered.set()
                        if fault == "error":
                            raise RuntimeError("result read unavailable")
                        await asyncio.Event().wait()
                    return records

                with monkeypatch.context() as patch:
                    patch.setattr(transaction, "get_records", interrupted_records)
                    return await callback(transaction)
            return await read(interrupted_callback)

        monkeypatch.setattr(repository.state_store, "read", interrupted_read)
        pending = asyncio.create_task(repository.get_dependency_records("graph", ("source",), tenant_id="tenant"))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if fault == "cancel":
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(RuntimeError, match="result read unavailable"):
                    await pending
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        monkeypatch.setattr(repository.state_store, "read", read)
        views, records = await asyncio.wait_for(
            repository.get_dependency_records("graph", ("source",), tenant_id="tenant"), 2,
        )
        assert [view.node_id for view in views] == ["source"]
        assert records["source"].result_digest == "a" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_dependency_read_rejects_corrupt_result_metadata(tmp_path: Path, backend: str) -> None:
    async with Persistence.open("dependency-corrupt", storage=_storage(backend, tmp_path), tenant_id="tenant") as state:
        target = await _prepare(state)
        repository = state.task.tasks

        async def corrupt(transaction):
            records = await transaction.list_records(RecordQuery(kind="task_result"))
            assert len(records) == 1
            record = records[0]
            assert await transaction.delete_record(
                record.key_digest,
                expected_storage_revision=record.storage_revision,
            )
            await transaction.insert_record(replace(record, parent_digest=b"x" * 32))

        await repository.state_store.mutate(corrupt)
        with pytest.raises(AIError) as raised:
            await LocalTaskGraphLauncher(repository, None, owner="worker")._dependency_context("graph", target, tenant_id="tenant")
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_successful_dependency_batches_bound_sql_parameters(tmp_path: Path) -> None:
    async with Persistence.open("dependency-fan-in", storage=_storage("sqlite", tmp_path), tenant_id="tenant") as state:
        source_ids = tuple(f"source-{index}" for index in range(257))
        target = TaskNode("target", source_ids)
        graph = TaskGraph("graph", tuple(TaskNode(node_id) for node_id in source_ids) + (target,))
        request = TaskGraphRequest(
            graph, Principal("tester", "tenant"), "submit:graph",
            limits=TaskGraphLimits(max_nodes=len(graph.nodes), max_concurrency=len(source_ids)),
        )
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
        repository = state.task.tasks
        sources = frozenset(source_ids)

        # Keep the real read-batch boundary without a claim/complete transaction
        # for every dependency. Canonical writers keep states, summaries and
        # events consistent across the bulk running and terminal transitions.
        async def terminal_fixture(transaction):
            before = await repository._event_state_in_transaction(transaction, "graph")
            graph_record = await transaction.get_record(repository._graph_key("graph"))
            expires_at = (await transaction.now()) + timedelta(seconds=30)
            running = tuple(
                replace(node, status=TaskStatus.RUNNING, owner="worker", fence=1, lease_expires_at=expires_at)
                if node.node_id in sources else node
                for node in before.node_states
            )
            await repository._apply_graph_transition(
                transaction, before, graph_record, running, TaskStatus.RUNNING,
            )
            before = await repository._event_state_in_transaction(transaction, "graph")
            graph_record = await transaction.get_record(repository._graph_key("graph"))
            completed = tuple(
                replace(
                    node, status=TaskStatus.SUCCEEDED, owner=None, lease_expires_at=None,
                    execution_id=f"execution-{node.node_id}", result_digest="a" * 64,
                )
                if node.node_id in sources else node
                for node in before.node_states
            )
            await transaction.insert_records(tuple(
                repository._stored(
                    "task_result", ["graph", node_id],
                    TaskResultRecord("graph", node_id, "a" * 64, execution_id=f"execution-{node_id}"),
                    scope=repository._result_scope("graph"), parent=repository._result_parent("graph"),
                )
                for node_id in source_ids
            ))
            await repository._apply_graph_transition(
                transaction, before, graph_record, completed, TaskStatus.PENDING,
            )

        await repository.state_store.mutate(terminal_fixture)
        prepared = await repository.graph_state("graph", tenant_id="tenant")
        assert prepared.status is TaskStatus.PENDING
        assert sum(node.status is TaskStatus.SUCCEEDED for node in prepared.node_states) == len(source_ids)
        assert next(node for node in prepared.node_states if node.node_id == "target").status is TaskStatus.PENDING
        engine = repository.state_store.storage_group.context.engine.sync_engine
        parameter_counts = []

        def sql(_connection, _cursor, statement, parameters, _context, many):
            if statement.lstrip().upper().startswith("SELECT") and "FROM ai_state_records" in statement:
                assert not many
                parameter_counts.append(len(parameters))

        event.listen(engine, "before_cursor_execute", sql)
        try:
            views, results = await repository.get_dependency_records("graph", source_ids, tenant_id="tenant")
        finally:
            event.remove(engine, "before_cursor_execute", sql)
        assert [view.node_id for view in views] == list(source_ids)
        assert all(view.status is TaskStatus.SUCCEEDED and view.fence == 1 for view in views)
        assert set(results) == set(source_ids)
        assert all(results[node_id].execution_id == f"execution-{node_id}" for node_id in source_ids)
        assert all(results[view.node_id].result_digest == view.result_digest == "a" * 64 for view in views)
        assert parameter_counts and max(parameter_counts) <= 999


@pytest.mark.asyncio
async def test_dependency_read_preserves_input_and_tenant_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("dependency-input", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        repository = state.task.tasks

        async def no_read(_callback):
            raise AssertionError("an empty or rejected request must not open a read scope")

        monkeypatch.setattr(repository.state_store, "read", no_read)
        assert await repository.get_dependency_records("graph", (), tenant_id="tenant") == ((), {})
        assert await repository.get_dependency_records("graph", ("source",), tenant_id="other") == ((), {})
        for invalid, error in ((["source"], TypeError), (("",), ValueError), ((1,), ValueError), (("source", "source"), ValueError)):
            with pytest.raises(error):
                await repository.get_dependency_records("graph", invalid, tenant_id="tenant")
