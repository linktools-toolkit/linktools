#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate physical pooling and optional publication from transaction counts."""

import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from linktools.ai.persistence import PersistenceDomain
from linktools.ai.storage import InMemoryCacheBackend, create_sql_storage_context

from .test_observation_coalescing import _observe, _setup


@pytest.mark.asyncio
async def test_sql_context_reuses_physical_connections_across_transactions(tmp_path) -> None:
    counts = {}
    for pooled in (False, True):
        engine = create_async_engine(
            "sqlite+aiosqlite:///" + str(tmp_path / ("pooled.db" if pooled else "unpooled.db")),
            **({} if pooled else {"poolclass": NullPool}),
        )
        context = create_sql_storage_context(engine)
        current = {"connect": 0, "checkout": 0, "transaction": 0}

        def connect(*args):
            current["connect"] += 1

        def checkout(*args):
            current["checkout"] += 1

        def begin(*args):
            current["transaction"] += 1

        await context.initialize()
        for name, callback in (("connect", connect), ("checkout", checkout), ("begin", begin)):
            event.listen(engine.sync_engine, name, callback)
        try:
            for _ in range(10):
                async with context.sessions() as session, session.begin():
                    assert await session.scalar(text("SELECT 1")) == 1
            counts[pooled] = current.copy()
        finally:
            await context.close()
            await engine.dispose()
    print("10 independent SQL transactions, pooled versus unpooled:", counts)
    assert counts[True]["connect"] < counts[False]["connect"]
    assert counts[True]["transaction"] == counts[False]["transaction"] == 10
    assert counts[True]["checkout"] == counts[False]["checkout"] == 10


@pytest.mark.asyncio
async def test_trace_buffer_batches_publication_without_losing_the_final_tail(tmp_path) -> None:
    counts = {}
    for buffered in (False, True):
        directory = tmp_path / str(buffered)
        directory.mkdir()
        state = await _setup(directory, "sqlite", cache=None)
        store = state.run_store
        store.trace_persistence.configure(
            "run", buffer=InMemoryCacheBackend(max_bytes=65536) if buffered else None,
            batch_size=128, flush_interval=60,
        )
        engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
        commits = 0

        def commit(_connection):
            nonlocal commits
            commits += 1

        event.listen(engine, "commit", commit)
        try:
            for index in range(8):
                await _observe(store, index)
            archive = store.read_store(PersistenceDomain.TRACE)
            assert len(await archive.list_events(agent_run_id="run")) == (0 if buffered else 8)
            await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
            assert len(await archive.list_events(agent_run_id="run")) == 8
            counts[buffered] = commits
        finally:
            event.remove(engine, "commit", commit)
            await state.close()
    print("8 trace observations including final flush, source commits:", counts)
    assert counts[True] < counts[False]
