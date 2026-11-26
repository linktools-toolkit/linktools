#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical TaskGraph headers have one physical owner without hiding opaque rows."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._record_owner import record_owner_edge, record_owner_names, record_owner_table
from linktools.ai.persistence.state import RecordQuery, record_key
from tests.ai.test_task_recovery_index import _request
from linktools.ai.task import TaskGraph, TaskGraphAdmission


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_graph_owner_preserves_generic_queries_and_rollback(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("graph-owner", tenant_id="tenant", storage=storage) as state:
        request = _request("graph")
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), request.graph)
        store = state.task.admissions.state_store
        key = record_key("graph-owner", "tenant", "task", "task_graph", ("graph",))
        assert record_owner_table(key) == "ai_task_graphs"
        assert record_owner_names(keys=(key,)) == ("ai_task_graphs",)
        assert record_owner_edge(key) != record_owner_edge(b"opaque")
        raw = await store.read_record(key)
        assert raw is not None
        opaque = replace(raw, key=b"opaque", sort_key=b"opaque")
        await store.mutate(lambda tx: tx.insert_record(opaque))
        selected = await store.read(lambda tx: tx.list_records(RecordQuery(kind="task_graph")))
        assert {row.key for row in selected} == {key, opaque.key}
        assert {row.key for row in await store.read(lambda tx: tx.scan_records())} >= {key, opaque.key}
        if backend == "sqlite":
            async with store.context.engine.connect() as connection:
                assert await connection.scalar(text("SELECT count(*) FROM ai_task_graphs")) == 1
                assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='task_graph'")) == 1
        async def rollback(tx):
            await tx.delete_records((key,))
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        with pytest.raises(AIError):
            await store.mutate(rollback)
        assert await store.read_record(key) == raw
        await store.mutate(lambda tx: tx.delete_records((key, opaque.key)))
        assert await store.read_record(key) is None


@pytest.mark.parametrize("dialect_name", ("mysql", "postgresql", "sqlite"))
def test_graph_owner_index_budget_and_constant_projection(dialect_name: str) -> None:
    from sqlalchemy import create_mock_engine, select
    from linktools.ai.persistence.schema import build_runtime_sql_metadata, PersistenceDomain
    from linktools.ai.persistence._record_owner import record_owner_rows
    metadata = build_runtime_sql_metadata(frozenset(PersistenceDomain))
    table = metadata.tables["ai_task_graphs"]
    statements = []
    engine = create_mock_engine(f"{dialect_name}://", lambda sql, *a, **kw: statements.append(str(sql.compile(dialect=engine.dialect))))
    table.create(engine, checkfirst=False)
    indexes = [sql for sql in statements if sql.startswith("CREATE INDEX") or sql.startswith("CREATE UNIQUE INDEX")]
    assert len(table.c) == 16
    assert table.c.key.type.length == 352
    assert {"scope_key", "parent_key", "scope_locator", "parent_locator"}.isdisjoint(table.c.keys())
    import re
    constraints = re.findall(r"\b(?:PRIMARY KEY|UNIQUE) \([^()]+\)", " ".join(statements))
    assert len(indexes) + len(constraints) == 5
    assert (len(indexes) + len(constraints)) * 3 <= len(table.c)
    assert all(len(index.columns) <= 3 for index in table.indexes)
    key = record_key("n", "t", "task", "task_graph", ("g",))
    sql = str(select(record_owner_rows(metadata, keys=(key,))).compile(dialect=engine.dialect))
    assert "ai_task_graphs" in sql and "ai_state_records" not in sql
    maximum = record_key("n" * 128, "t" * 128, "task", "task_graph", ("界" * 21 + "a",))
    assert len(maximum) == 352 and record_owner_table(maximum) == "ai_task_graphs"
    assert record_owner_table(record_key("n", "t", "task", "task_graph", ("界" * 21 + "ab",))) == "ai_state_records"
    assert record_owner_table(record_key("n", "t", "task", "task_graph", ("graph", "child"))) == "ai_state_records"


def test_task_graph_id_byte_limit() -> None:
    graph_id = "界" * 21 + "a"
    assert TaskGraph(graph_id, ()).graph_id == graph_id
    with pytest.raises(ValueError, match="task graph identity"):
        TaskGraph(graph_id + "b", ())
