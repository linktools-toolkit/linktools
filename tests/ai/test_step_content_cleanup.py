#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline native cleanup preserves live roots and discards stale write caches."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.steps import native_store_locator
from linktools.ai.persistence import SqlStepStore


@pytest.fixture(autouse=True)
def _exercise_explicit_offline_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence import _native_content_index

    async def defer(_context, **kwargs) -> None:
        pass

    # These tests deliberately accumulate orphans for the explicit offline
    # collector. Automatic collection has its own owner/concurrency coverage.
    monkeypatch.setattr(_native_content_index, "maintain_native_content", defer)


class _Offline:
    def __init__(self) -> None:
        self.entered = False

    @asynccontextmanager
    async def offline_exclusivity(self):
        assert not self.entered
        self.entered = True
        try:
            yield
        finally:
            self.entered = False


def _snapshot(run: str, step: int, *messages: str) -> ContinuableSnapshot:
    return ContinuableSnapshot(run_id=run, step_index=step, idempotency_key=str(step),
        messages=[ModelRequest(parts=[UserPromptPart(value, timestamp=datetime(2026, 11, 9, tzinfo=timezone.utc))]) for value in messages])


@pytest.mark.asyncio
async def test_offline_cleanup_preserves_shared_roots_namespaces_and_reinserts_retired_content(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'cleanup.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    other = SqlStepStore(engine, namespace="steps", tenant_id="two")
    await store.initialize(local=True)
    retired = _snapshot("run", 1, "shared", "retired")
    latest = _snapshot("run", 2, "shared", "current", "shared", "current", "shared", "shared", "current")
    repeated = _snapshot("branch", 1, "shared", "shared")
    foreign = _snapshot("foreign", 1, "retired")
    guard = _Offline()
    reads = []

    def observe(connection, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith(("SELECT", "WITH")):
            assert guard.entered
            reads.append(statement)

    try:
        for snapshot in (retired, latest, repeated, _snapshot("empty", 1)):
            await store.save_snapshot(snapshot)
        await other.save_snapshot(foreign)
        await other.close()
        event.listen(engine.sync_engine, "before_cursor_execute", observe)
        result = await store.purge_unreferenced_content(exclusive=guard)
        event.remove(engine.sync_engine, "before_cursor_execute", observe)
        assert reads, "cleanup must inspect retained content under offline exclusion"
        assert result.deleted_sequences > 0
        assert result.deleted_messages == 1
        assert result.retained_messages == 2
        assert not guard.entered
        assert await store.latest_snapshot(run_id="run") == latest
        assert await store.latest_snapshot(run_id="branch") == repeated
        assert len(await store.list_snapshots(run_id="run")) == 1
        # Replay tombstones survive cleanup. A new occurrence can reuse the
        # deleted content because this adapter's old body knowledge was cleared.
        await store.save_snapshot(retired)
        assert len(await store.list_snapshots(run_id="run")) == 1
        revived = replace(retired, run_id="revived")
        await store.save_snapshot(revived)
        assert await store.latest_snapshot(run_id="revived") == revived
        foreign_reader = SqlStepStore(engine, namespace="steps", tenant_id="two")
        try:
            assert await foreign_reader.latest_snapshot(run_id="foreign") == foreign
        finally:
            await foreign_reader.close()
        again = await store.purge_unreferenced_content(exclusive=guard)
        assert again.deleted_messages == again.deleted_sequences == 0
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", observe):
            event.remove(engine.sync_engine, "before_cursor_execute", observe)
        await store.close()
        await other.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("missing-body", "wrong-node", "wrong-root-count"))
async def test_cleanup_rejects_broken_live_references_before_deleting_orphans(tmp_path: Path, damage: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'broken.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)
    try:
        await store.save_snapshot(_snapshot("run", 1, "retired"))
        await store.save_snapshot(_snapshot("run", 2, "live", "another"))
        async with engine.begin() as connection:
            root = await connection.scalar(text("SELECT sequence_root FROM ai_step_snapshots WHERE pruned = 0"))
            if damage == "missing-body":
                await connection.execute(text(
                    "DELETE FROM ai_step_messages WHERE sha256 = "
                    "(SELECT message_sha256 FROM ai_step_sequences WHERE sha256 = :root)"
                ), {"root": root})
            elif damage == "wrong-node":
                await connection.execute(text(
                    "UPDATE ai_step_sequences SET tree_count = tree_count + 2 WHERE sha256 = :root"
                ), {"root": root})
            else:
                await connection.execute(text(
                    "UPDATE ai_step_snapshots SET messages = :value WHERE pruned = 0"
                ), {"value": '{"stored_messages":0,"checkpoint":null}'})
            before = tuple([await connection.scalar(text(f"SELECT count(*) FROM {table}"))
                            for table in ("ai_step_sequences", "ai_step_messages")])
        with pytest.raises(AIError) as failure:
            await store.purge_unreferenced_content(exclusive=_Offline())
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        async with engine.connect() as connection:
            after = tuple([await connection.scalar(text(f"SELECT count(*) FROM {table}"))
                           for table in ("ai_step_sequences", "ai_step_messages")])
        assert after == before
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_cleanup_rolls_back_sequence_deletion_when_body_deletion_fails(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rollback.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)

    def fail_body_delete(connection, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith("DELETE FROM AI_STEP_MESSAGES"):
            raise RuntimeError("body deletion rejected")

    try:
        await store.save_snapshot(_snapshot("run", 1, "retired"))
        current = _snapshot("run", 2, "retained")
        await store.save_snapshot(current)
        event.listen(engine.sync_engine, "before_cursor_execute", fail_body_delete)
        with pytest.raises(RuntimeError, match="body deletion rejected"):
            await store.purge_unreferenced_content(exclusive=_Offline())
        event.remove(engine.sync_engine, "before_cursor_execute", fail_body_delete)
        async with engine.connect() as connection:
            for table in ("ai_step_sequences", "ai_step_messages"):
                assert await connection.scalar(text(f"SELECT count(*) FROM {table}")) == 2
        assert await store.latest_snapshot(run_id="run") == current
        result = await store.purge_unreferenced_content(exclusive=_Offline())
        assert result.deleted_sequences == result.deleted_messages == 1
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", fail_body_delete):
            event.remove(engine.sync_engine, "before_cursor_execute", fail_body_delete)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("unknown", "cancelled"))
async def test_cleanup_lost_acknowledgement_clears_body_cache_before_new_writes(tmp_path: Path, monkeypatch, outcome: str) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'unacknowledged.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)
    retired = _snapshot("run", 1, "retired")
    original = type(store.storage_context).run_mutation

    async def unacknowledged(context, callback, **kwargs):
        result = await original(context, callback, **kwargs)
        if kwargs.get("domain") == "steps.content_cleanup":
            if outcome == "unknown":
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            raise asyncio.CancelledError()
        return result

    try:
        await store.save_snapshot(retired)
        await store.save_snapshot(_snapshot("run", 2, "retained"))
        monkeypatch.setattr(type(store.storage_context), "run_mutation", unacknowledged)
        with pytest.raises(AIError if outcome == "unknown" else asyncio.CancelledError):
            await store.purge_unreferenced_content(exclusive=_Offline())
        monkeypatch.setattr(type(store.storage_context), "run_mutation", original)
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 1
        revived = replace(retired, run_id="revived")
        await store.save_snapshot(revived)
        assert await store.latest_snapshot(run_id="revived") == revived
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_cleanup_reads_only_retained_bodies_in_bounded_batches(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'bounded.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)
    body_batches = []

    def observe(connection, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith("SELECT") and "ai_step_messages.message" in statement:
            # Observe the SQL boundary: no query may materialize orphan bodies
            # or the entire retained message corpus in one result.
            assert "ai_step_messages.sha256 IN" in statement
            body_batches.append(set(parameters) - {store.store_key, native_store_locator(store.store_key)})

    try:
        await store.save_snapshot(_snapshot("run", 1, "orphan"))
        async with engine.connect() as connection:
            orphan_id = await connection.scalar(text("SELECT sha256 FROM ai_step_messages"))
        # One full validation batch and one tail message exercise bounded reads.
        current = _snapshot("run", 2, *(f"retained-{index}" for index in range(129)))
        await store.save_snapshot(current)
        async with engine.connect() as connection:
            all_ids = set((await connection.execute(text("SELECT sha256 FROM ai_step_messages"))).scalars())
        event.listen(engine.sync_engine, "before_cursor_execute", observe)
        result = await store.purge_unreferenced_content(exclusive=_Offline())
        event.remove(engine.sync_engine, "before_cursor_execute", observe)
        assert result.deleted_messages == 1
        assert result.retained_messages == 129
        assert body_batches and max(map(len, body_batches)) <= 128
        assert set.union(*body_batches) == all_ids - {orphan_id}
        assert sum(map(len, body_batches)) == 129
        assert await store.latest_snapshot(run_id="run") == current
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", observe):
            event.remove(engine.sync_engine, "before_cursor_execute", observe)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_cleanup_validates_late_retained_batch_before_deleting(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'late-damage.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await store.initialize(local=True)
    deletions = []
    body_batches = []

    def observe(connection, cursor, statement, parameters, context, many):
        if statement.lstrip().upper().startswith("DELETE"):
            deletions.append(statement)
        if statement.lstrip().upper().startswith("SELECT") and "ai_step_messages.message" in statement:
            body_batches.append(set(parameters) - {store.store_key, native_store_locator(store.store_key)})

    try:
        current = _snapshot("run", 1, *(f"retained-{index}" for index in range(129)))
        await store.save_snapshot(current)
        async with engine.begin() as connection:
            # The highest digest belongs to the last bounded validation batch.
            damaged = await connection.scalar(text("SELECT max(sha256) FROM ai_step_messages"))
            await connection.execute(text("UPDATE ai_step_messages SET message = '{}' WHERE sha256 = :id"),
                                     {"id": damaged})
        await store.save_snapshot(_snapshot("orphan", 1, "orphan"))
        await store.save_snapshot(_snapshot("orphan", 2))
        event.listen(engine.sync_engine, "before_cursor_execute", observe)
        with pytest.raises(AIError) as failure:
            await store.purge_unreferenced_content(exclusive=_Offline())
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert len(body_batches) > 1 and body_batches[-1] == {damaged}
        assert sum(map(len, body_batches)) == 129
        assert not deletions
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 130
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", observe):
            event.remove(engine.sync_engine, "before_cursor_execute", observe)
        await store.close()
        await engine.dispose()
