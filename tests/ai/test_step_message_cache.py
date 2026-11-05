#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Committed immutable message receipts avoid repeated SQL body submissions."""

import asyncio
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.storage import create_sql_storage_context


class _SqlTraffic:
    def __init__(self) -> None:
        self.statements: Counter[str] = Counter()
        self.message_rows = 0
        self.message_bytes = 0
        self.bound_bytes = 0

    def capture(self, connection, cursor, statement, parameters, context, many) -> None:
        words = statement.split()
        self.statements[words[0]] += 1
        self.bound_bytes += self._bytes(parameters)
        if statement.startswith("INSERT INTO ai_step_messages "):
            self.statements["message_insert"] += 1
            assert not many
            self.message_rows += len(parameters) // 3
            self.message_bytes += sum(len(value.encode()) for value in parameters[2::3])
        elif statement.startswith("INSERT INTO ai_step_snapshots "):
            self.statements["snapshot_insert"] += 1
        elif statement.startswith("INSERT INTO ai_step_message_refs "):
            self.statements["reference_insert"] += 1

    @classmethod
    def _bytes(cls, value: object) -> int:
        if isinstance(value, str):
            return len(value.encode())
        if isinstance(value, bytes):
            return len(value)
        if isinstance(value, (tuple, list)):
            return sum(cls._bytes(item) for item in value)
        return 0


def _snapshot(key: str, *messages: str) -> ContinuableSnapshot:
    return ContinuableSnapshot(run_id="run", step_index=0, idempotency_key=key,
        messages=[ModelRequest(parts=[UserPromptPart(message)]) for message in messages])


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity, submitted", [(0, 2), (4096, 1)])
async def test_repeated_snapshots_skip_only_confirmed_message_body_submissions(
    tmp_path: Path, capacity: int, submitted: int,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'messages.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1,
                         max_cached_messages=capacity)
    await store.initialize(local=True)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        first = _snapshot("first", "body")
        await store.save_snapshot(first)
        second = replace(first, idempotency_key="second", step_index=1)
        await store.save_snapshot(second)
        assert traffic.message_rows == submitted
        assert traffic.statements["snapshot_insert"] == 2
        assert traffic.statements["reference_insert"] == 2
        # Retention reads belong to the existing pruning contract, not the cache.
        assert traffic.statements["SELECT"] == 2
        assert await store.latest_snapshot(run_id="run") == second
        await store.save_snapshot(first)
        assert traffic.message_rows == submitted
        assert await store.latest_snapshot(run_id="run") == second
        cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
        await cold.save_snapshot(replace(first, idempotency_key="cold"))
        assert traffic.message_rows == submitted + 1
        await cold.close()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_failed_publication_does_not_cache_uncommitted_messages(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rollback.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)

    def fail(connection, cursor, statement, parameters, context, many) -> None:
        if statement.startswith("INSERT INTO ai_step_message_refs "):
            raise RuntimeError("publication failed")

    try:
        first = _snapshot("first", "uncommitted")
        event.listen(engine.sync_engine, "before_cursor_execute", fail)
        with pytest.raises(RuntimeError, match="publication failed"):
            await store.save_snapshot(first)
        event.remove(engine.sync_engine, "before_cursor_execute", fail)
        assert traffic.message_rows == 1
        await store.save_snapshot(first)
        assert traffic.message_rows == 2
        await store.save_snapshot(replace(first, idempotency_key="next"))
        assert traffic.message_rows == 2
        assert await store.latest_snapshot(run_id="run") == replace(first, idempotency_key="next")
    finally:
        if event.contains(engine.sync_engine, "before_cursor_execute", fail):
            event.remove(engine.sync_engine, "before_cursor_execute", fail)
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_lost_commit_ack_does_not_promote_or_promote_on_idempotent_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'lost-ack.db'}")
    context = create_sql_storage_context(engine)
    store = SqlStepStore(context, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    original = type(context).run_mutation

    async def lose_ack(self, callback, **kwargs):
        await original(self, callback, **kwargs)
        if cancelled:
            raise asyncio.CancelledError()
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    first = _snapshot("first", "committed-but-unconfirmed")
    try:
        monkeypatch.setattr(type(context), "run_mutation", lose_ack)
        with pytest.raises(asyncio.CancelledError if cancelled else AIError):
            await store.save_snapshot(first)
        assert traffic.message_rows == 1
        monkeypatch.setattr(type(context), "run_mutation", original)
        assert await store.latest_snapshot(run_id="run") == first
        await store.save_snapshot(first)
        assert traffic.message_rows == 1
        await store.save_snapshot(replace(first, idempotency_key="new-boundary"))
        assert traffic.message_rows == 2
        await store.save_snapshot(replace(first, idempotency_key="warm-boundary"))
        assert traffic.message_rows == 2
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_clone_cache_is_scoped_and_cleared_on_close(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'scope.db'}")
    context = create_sql_storage_context(engine)
    root = SqlStepStore(context, namespace="steps", tenant_id="one")
    await root.initialize(local=True)
    first, second = (root.with_tool_effect_store(InMemoryStepStore()) for _ in range(2))
    other_tenant = SqlStepStore(context, namespace="steps", tenant_id="two")
    other_namespace = SqlStepStore(context, namespace="other", tenant_id="one")
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    snapshot = _snapshot("first", "shared-content")
    try:
        await first.save_snapshot(snapshot)
        await second.save_snapshot(replace(snapshot, idempotency_key="second"))
        assert traffic.message_rows == 1
        await other_tenant.save_snapshot(snapshot)
        await other_namespace.save_snapshot(snapshot)
        assert traffic.message_rows == 3
        assert await other_tenant.latest_snapshot(run_id="run") == snapshot
        assert await other_namespace.latest_snapshot(run_id="run") == snapshot
        await first.close()
        await second.save_snapshot(replace(snapshot, idempotency_key="after-close"))
        assert traffic.message_rows == 4
        await root.close()
        # Maintenance occurs only after all clones have stopped and closed.
        await second.close()
        await other_tenant.close()
        await other_namespace.close()
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_messages"))
        reopened = SqlStepStore(context, namespace="steps", tenant_id="one")
        await reopened.save_snapshot(replace(snapshot, idempotency_key="after-maintenance"))
        assert traffic.message_rows == 5
        assert await reopened.latest_snapshot(run_id="run") == replace(snapshot, idempotency_key="after-maintenance")
        await reopened.close()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await root.close()
        await other_tenant.close()
        await other_namespace.close()
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_cache_is_bounded_and_misses_resubmit_without_source_reads(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'bound.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one", max_cached_messages=2)
    await store.initialize(local=True)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    snapshots = [_snapshot(str(index), body) for index, body in enumerate(("a", "b", "c"))]
    try:
        for snapshot in snapshots:
            await store.save_snapshot(snapshot)
        assert traffic.message_rows == 3
        await store.save_snapshot(replace(snapshots[0], idempotency_key="evicted"))
        assert traffic.message_rows == 4
        await store.save_snapshot(replace(snapshots[2], idempotency_key="retained"))
        assert traffic.message_rows == 4
        assert traffic.statements["SELECT"] == 0
        assert await store.latest_snapshot(run_id="run") == replace(snapshots[2], idempotency_key="retained")
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_clone_cannot_use_another_transactions_uncommitted_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'concurrent-cache.db'}")
    context = create_sql_storage_context(engine)
    root = SqlStepStore(context, namespace="steps", tenant_id="one")
    await root.initialize(local=True)
    clone = root.with_tool_effect_store(InMemoryStepStore())
    pending, release = asyncio.Event(), asyncio.Event()
    original = context.dialect.insert_ignore_conflict_many
    held = False

    async def pause_then_fail(session, **kwargs):
        nonlocal held
        result = await original(session, **kwargs)
        if not held:
            held = True
            pending.set()
            await release.wait()
            raise RuntimeError("first transaction rolled back")
        return result

    monkeypatch.setattr(context.dialect, "insert_ignore_conflict_many", pause_then_fail)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    first = _snapshot("first", "pending-body")
    try:
        failed = asyncio.create_task(root.save_snapshot(first))
        await pending.wait()
        second = replace(first, idempotency_key="second")
        successful = asyncio.create_task(clone.save_snapshot(second))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(failed, successful, return_exceptions=True)
        assert isinstance(results[0], RuntimeError) and results[1] is None
        assert traffic.message_rows == 2
        assert await root.latest_snapshot(run_id="run") == second
        await root.save_snapshot(replace(first, idempotency_key="third"))
        assert traffic.message_rows == 2
    finally:
        release.set()
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await clone.close()
        await root.close()
        await context.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_ignored_snapshot_retry_does_not_confirm_its_unpublished_message_body(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ignored-body.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        initial = _snapshot("same-key", "initial-body")
        ignored = _snapshot("same-key", "unpublished-body")
        await store.save_snapshot(initial)
        await store.save_snapshot(ignored)
        assert traffic.message_rows == 1
        published = replace(ignored, idempotency_key="new-key")
        await store.save_snapshot(published)
        assert traffic.message_rows == 2
        assert traffic.statements["SELECT"] == 0
        assert await store.latest_snapshot(run_id="run") == published
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await store.close()
        await engine.dispose()
