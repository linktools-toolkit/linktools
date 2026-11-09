#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent transaction and capacity checks for SQL message receipts."""

import asyncio
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore
from sqlalchemy import event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.persistence import SqlStepStore
from linktools.ai.storage import create_sql_storage_context

from .test_step_message_cache import _SqlTraffic


def _snapshot(key: str, text: str) -> ContinuableSnapshot:
    return ContinuableSnapshot(run_id="run", step_index=0, idempotency_key=key,
                               messages=[ModelResponse(parts=[TextPart(text)])])


@pytest.mark.asyncio
async def test_same_key_race_does_not_cache_losing_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'same-key.db'}")
    context = create_sql_storage_context(engine)
    root = SqlStepStore(context, namespace="steps", tenant_id="one")
    await root.initialize(local=True)
    clone = root.with_tool_effect_store(InMemoryStepStore())
    inserted, release = asyncio.Event(), asyncio.Event()
    original = context.dialect.insert_ignore_conflict_many
    held = False

    async def pause_after_insert(session, **kwargs):
        nonlocal held
        result = await original(session, **kwargs)
        if not held:
            held = True
            inserted.set()
            await release.wait()
        return result

    monkeypatch.setattr(context.dialect, "insert_ignore_conflict_many", pause_after_insert)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    winner, loser = _snapshot("same-key", "winner"), _snapshot("same-key", "unpublished loser")
    try:
        first = asyncio.create_task(root.save_snapshot(winner))
        await inserted.wait()
        second = asyncio.create_task(clone.save_snapshot(loser))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)
        assert traffic.message_rows == 1
        assert await root.latest_snapshot(run_id="run") == winner
        published = replace(loser, idempotency_key="new-key")
        await clone.save_snapshot(published)
        assert traffic.message_rows == 2
        assert await root.latest_snapshot(run_id="run") == published
        await root.save_snapshot(replace(published, idempotency_key="warm-key"))
        assert traffic.message_rows == 2
    finally:
        release.set()
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await clone.close()
        await root.close()
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_aborted_retry_resubmits_before_confirming(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'retry.db'}")
    context = create_sql_storage_context(engine)
    store = SqlStepStore(context, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    original = context.dialect.insert_ignore_conflict_many
    fail_once = True

    async def abort_first_attempt(session, **kwargs):
        nonlocal fail_once
        result = await original(session, **kwargs)
        if fail_once:
            fail_once = False
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise OperationalError("injected abort", {}, error)
        return result

    monkeypatch.setattr(context.dialect, "insert_ignore_conflict_many", abort_first_attempt)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    snapshot = _snapshot("first", "retry body")
    try:
        await store.save_snapshot(snapshot)
        assert traffic.message_rows == 2
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 1
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots")) == 1
        await store.save_snapshot(replace(snapshot, idempotency_key="warm"))
        assert traffic.message_rows == 2
        assert await store.latest_snapshot(run_id="run") == replace(snapshot, idempotency_key="warm")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_history_larger_than_cache_keeps_all_references(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'oversized.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_cached_messages=3, max_snapshots_per_run=1)
    await store.initialize(local=True)
    snapshot = ContinuableSnapshot(run_id="run", step_index=0,
        messages=[ModelResponse(parts=[TextPart(str(index))]) for index in range(17)])
    try:
        for index in range(4):
            current = replace(snapshot, idempotency_key=str(index), step_index=index)
            await store.save_snapshot(current)
            # The private check observes the required resident memory bound, not an eviction order.
            assert len(store._confirmed_messages) <= 3
            assert len(store._confirmed_sequences) <= 3
            assert await store.latest_snapshot(run_id="run") == current
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 17
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_sequences")) == 17
    finally:
        await store.close()
        await engine.dispose()
