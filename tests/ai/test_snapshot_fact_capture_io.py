#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detached fact capture uses storage order without changing portable order."""

from pathlib import Path

import pytest
from sqlalchemy import event, text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.snapshots import (
    SnapshotLimits, _SnapshotCaptureBudget, _StateSnapshotReader, _capture_snapshot_reader,
)
from linktools.ai.persistence.state import StateTransaction, StoredFact

from .test_state_batch_delete_backends import _open_store, _record


@pytest.mark.asyncio
async def test_sql_capture_seeks_fact_index_and_restores_raw_order(tmp_path: Path) -> None:
    owner = _record(1)
    keys = (b"common-" * 30 + b"z", b"common-" * 30 + b"a", b"z", b"A", b"a")
    facts = tuple(StoredFact(key, sequence, owner.key, "probe", None, None, {"body": "x" * 1000})
                  for key in keys for sequence in (1, 2, 3))
    async with _open_store("sql", tmp_path) as store:
        async def seed(tx: StateTransaction) -> None:
            await tx.insert_record(owner)
            await tx.insert_facts(facts)
        await store.mutate(seed)
        other = await store.storage_group.open_partition(b"other-partition", create=True)
        assert other is not None
        await other.mutate(seed)
        statements = []

        def capture(_conn, _cursor, sql, params, _context, _many):
            if sql.startswith("SELECT") and "ai_state_facts" in sql:
                statements.append((sql, params))

        event.listen(store.context.engine.sync_engine, "before_cursor_execute", capture)
        try:
            async def read(_group):
                return await _capture_snapshot_reader(_StateSnapshotReader(store),
                    budget=_SnapshotCaptureBudget(SnapshotLimits(100, 100000)), page_size=2)
            result = await store.storage_group.read_many((store,), read)
        finally:
            event.remove(store.context.engine.sync_engine, "before_cursor_execute", capture)
        assert result.facts == tuple(sorted(facts, key=lambda value: (value.stream_key, value.sequence)))
        assert len(statements) == 8
        async with store.context.engine.connect() as connection:
            for sql, params in statements:
                plan = (await connection.exec_driver_sql("EXPLAIN QUERY PLAN " + sql, params)).fetchall()
                assert not any("TEMP B-TREE" in row[3] for row in plan)


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", (SnapshotLimits(2, 100000), SnapshotLimits(100, 50)))
async def test_sql_fact_capture_stops_at_shared_budget(tmp_path: Path, limits: SnapshotLimits) -> None:
    owner = _record(1)
    facts = tuple(StoredFact(f"stream-{i}".encode(), 1, owner.key, "probe", None, None,
                             {"body": "x" * 100}) for i in range(8))
    async with _open_store("sql", tmp_path) as store:
        async def seed(tx: StateTransaction) -> None:
            await tx.insert_record(owner)
            await tx.insert_facts(facts)
        await store.mutate(seed)
        budget = _SnapshotCaptureBudget(limits)
        async def read(_group):
            return await _StateSnapshotReader(store).capture_facts(budget=budget, page_size=2)
        with pytest.raises(AIError) as rejected:
            await store.storage_group.read_many((store,), read)
        assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
        assert budget.entries <= limits.max_entries and budget.byte_size <= limits.max_bytes
        assert await store.read(lambda tx: tx.get_record(owner.key)) == owner


@pytest.mark.asyncio
async def test_sql_fact_capture_rejects_corrupt_locator(tmp_path: Path) -> None:
    owner = _record(1)
    fact = StoredFact(b"stream", 1, owner.key, "probe", None, None, {})
    async with _open_store("sql", tmp_path) as store:
        async def seed(tx: StateTransaction) -> None:
            await tx.insert_record(owner)
            await tx.insert_fact(fact)
        await store.mutate(seed)
        async with store.context.engine.begin() as connection:
            await connection.execute(text("UPDATE ai_state_facts SET stream_locator = :locator"),
                                     {"locator": "0" * 64})
        async def read(_group):
            return await _StateSnapshotReader(store).capture_facts(
                budget=_SnapshotCaptureBudget(SnapshotLimits(100, 100000)), page_size=2)
        with pytest.raises(AIError) as rejected:
            await store.storage_group.read_many((store,), read)
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_sql_fact_capture_cancellation_releases_snapshot(tmp_path: Path) -> None:
    import asyncio

    owner = _record(1)
    facts = tuple(StoredFact(f"stream-{i}".encode(), 1, owner.key, "probe", None, None, {})
                  for i in range(8))
    async with _open_store("sql", tmp_path) as store:
        async def seed(tx: StateTransaction) -> None:
            await tx.insert_record(owner)
            await tx.insert_facts(facts)
        await store.mutate(seed)
        pages = 0
        task = None

        def cancel(_conn, _cursor, sql, _params, _context, _many):
            nonlocal pages
            if sql.startswith("SELECT") and "ai_state_facts" in sql:
                pages += 1
                if pages == 2:
                    asyncio.get_running_loop().call_soon(task.cancel)

        async def read(_group):
            return await _StateSnapshotReader(store).capture_facts(
                budget=_SnapshotCaptureBudget(SnapshotLimits(100, 100000)), page_size=2)
        event.listen(store.context.engine.sync_engine, "before_cursor_execute", cancel)
        try:
            task = asyncio.create_task(store.storage_group.read_many((store,), read))
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            event.remove(store.context.engine.sync_engine, "before_cursor_execute", cancel)
        assert pages == 2
        assert await store.read(lambda tx: tx.get_record(owner.key)) == owner
