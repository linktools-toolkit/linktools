#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task terminal publication keeps graph, node, result and event facts atomic."""

from collections import Counter
from dataclasses import replace

import pytest
from sqlalchemy import event

from linktools.ai.core import TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.task import TaskExpanderRef, TaskGraph, TaskNode
from linktools.ai.storage import SQLiteDialect

from ._task_test_helpers import admit_graph


class _RowwiseDialect(SQLiteDialect):
    @property
    def prefer_update_executemany(self) -> bool:
        return True


async def _prepare(state, expanded):
    graph = TaskGraph("graph", (TaskNode("source", expander=TaskExpanderRef("expand", 1) if expanded else None),))
    await admit_graph(state, graph)
    lease = await state.task.tasks.claim("graph", "source", tenant_id="tenant", owner="worker", lease_seconds=30)
    return lease


async def _finish(repository, lease, outcome):
    if outcome == "fail":
        return await repository.fail(lease, tenant_id="tenant", error_code=ErrorCode.TASK_NODE_FAILED.value, error_digest="b" * 64)
    return await repository.complete(lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
        expanded_nodes=(TaskNode("child", ("source",)),) if outcome == "expand" else ())


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite", "sqlite_rowwise"))
@pytest.mark.parametrize("outcome", ("complete", "fail", "expand"))
async def test_terminal_publication_retains_atomic_facts(tmp_path, backend, outcome):
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("terminal-batch", storage=storage, tenant_id="tenant") as state:
        repository = state.task.tasks
        if backend == "sqlite_rowwise":
            repository.state_store.storage_group.context.dialect = _RowwiseDialect()
        lease = await _prepare(state, outcome == "expand")
        counts = Counter()
        def sql(_connection, _cursor, statement, _params, _context, many):
            counts[statement.lstrip().split()[0].upper()] += 1
            counts["executemany"] += bool(many)
        def begin(_connection):
            counts["begin"] += 1
        def commit(_connection):
            counts["commit"] += 1
        engine = None if backend == "memory" else repository.state_store.storage_group.context.engine.sync_engine
        listeners = (("before_cursor_execute", sql), ("begin", begin), ("commit", commit))
        if engine is not None:
            for name, callback in listeners:
                event.listen(engine, name, callback)
        try:
            terminal = await _finish(repository, lease, outcome)
        finally:
            if engine is not None:
                for name, callback in listeners:
                    event.remove(engine, name, callback)
        view = await repository.graph_state("graph", tenant_id="tenant")
        source = next(node for node in view.node_states if node.node_id == "source")
        expected = TaskStatus.FAILED if outcome == "fail" else TaskStatus.SUCCEEDED
        assert terminal.status is source.status is expected
        assert source.owner is None and source.lease_expires_at is None
        assert source.fence == lease.fence
        assert source.result_digest == (None if outcome == "fail" else "a" * 64)
        assert {node.node_id for node in view.nodes} == ({"source", "child"} if outcome == "expand" else {"source"})
        events = await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=100)
        assert events.items[-1].event_seq == view.event_seq
        if backend == "sqlite":
            print("TASK_TERMINAL_IO", outcome, dict(counts))
            assert counts["commit"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome,target", (("complete", "task_node_state"), ("complete", "task_graph"),
    ("fail", "task_node_state"), ("fail", "task_graph"), ("expand", "task_node_state")))
@pytest.mark.parametrize("rowwise", (False, True))
async def test_terminal_cas_failure_rolls_back_result_nodes_and_events(tmp_path, monkeypatch, outcome, target, rowwise):
    async with Persistence.open("terminal-rollback", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant") as state:
        repository = state.task.tasks
        lease = await _prepare(state, outcome == "expand")
        if rowwise:
            repository.state_store.storage_group.context.dialect = _RowwiseDialect()
        before = await repository.graph_state("graph", tenant_id="tenant")
        before_events = await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=100)
        original = _SqlTransaction.replace_records
        injected = False
        async def stale_node(transaction, replacements):
            nonlocal injected
            values = []
            for item in replacements:
                if item.record.kind == target and item.record.state in {"SUCCEEDED", "FAILED"}:
                    injected = True
                    item = replace(item, expected_storage_revision=item.expected_storage_revision + 1,
                        record=replace(item.record, storage_revision=item.record.storage_revision + 1))
                values.append(item)
            return await original(transaction, tuple(values))
        monkeypatch.setattr(_SqlTransaction, "replace_records", stale_node)
        with pytest.raises(AIError) as raised:
            await _finish(repository, lease, outcome)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert injected
        assert await repository.graph_state("graph", tenant_id="tenant") == before
        assert await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=100) == before_events
        assert await repository.get_results("graph", ("source",), tenant_id="tenant") == {}
