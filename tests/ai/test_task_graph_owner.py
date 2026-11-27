#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical TaskGraph headers have one physical owner without hiding opaque rows."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
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
        raw = await store.read_record(key)
        assert raw is not None
        opaque = replace(raw, key=b"opaque", sort_key=b"opaque")
        await store.mutate(lambda tx: tx.insert_record(opaque))
        selected = await store.read(lambda tx: tx.list_records(RecordQuery(kind="task_graph")))
        assert {row.key for row in selected} == {key, opaque.key}
        assert {row.key for row in await store.read(lambda tx: tx.scan_records())} >= {key, opaque.key}
        if backend == "sqlite":
            async with store.context.engine.connect() as connection:
                assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='task_graph'")) == 2
        async def rollback(tx):
            await tx.delete_records((key,))
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        with pytest.raises(AIError):
            await store.mutate(rollback)
        assert await store.read_record(key) == raw
        await store.mutate(lambda tx: tx.delete_records((key, opaque.key)))
        assert await store.read_record(key) is None


def test_task_graph_id_byte_limit() -> None:
    graph_id = "界" * 21 + "a"
    assert TaskGraph(graph_id, ()).graph_id == graph_id
    with pytest.raises(ValueError, match="task graph identity"):
        TaskGraph(graph_id + "b", ())
