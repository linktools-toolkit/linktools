#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native safety points publish one atomic, confirmed SQL transaction."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSessionTransaction, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.capability.steps import AtomicStepStore

from .test_step_message_cache import _SqlTraffic


@pytest_asyncio.fixture
async def atomic_store(tmp_path: Path) -> AsyncIterator[tuple[SqlStepStore, AsyncEngine]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'atomic.db'}")
    store = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await store.initialize(local=True)
    try:
        yield store, engine
    finally:
        await store.close()
        await engine.dispose()


def _records() -> tuple[RunRecord, list[StepEvent], ContinuableSnapshot]:
    run = RunRecord(run_id="run", registration_id="registration")
    events = [StepEvent(run_id="run", kind=kind, step_index=0, idempotency_key=kind)
              for kind in ("run_started", "model_request_started")]
    snapshot = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="snapshot",
        messages=[ModelRequest(parts=[UserPromptPart("body")])])
    return run, events, snapshot


@pytest.mark.asyncio
async def test_compound_native_records_use_one_physical_commit_and_warm_bodies(atomic_store: tuple[SqlStepStore, AsyncEngine]) -> None:
    store, engine = atomic_store
    assert isinstance(store, AtomicStepStore)
    run, events, snapshot = _records()
    commits = []
    record_commit = commits.append
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "commit", record_commit)
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    await store.commit_step(run=run, events=events, snapshot=snapshot)
    assert len(commits) == 1
    event.remove(engine.sync_engine, "commit", record_commit)
    assert traffic.message_rows == 1
    assert await store.get_run(run_id="run") == run
    assert await store.list_events(run_id="run") == events
    assert await store.latest_snapshot(run_id="run") == snapshot
    commits.clear()
    event.listen(engine.sync_engine, "commit", record_commit)
    await store.commit_step(events=events, snapshot=replace(snapshot, idempotency_key="next"))
    assert len(commits) == 1
    assert traffic.message_rows == 1
    assert await store.list_events(run_id="run") == events
    await store.commit_step()
    assert len(commits) == 1


@pytest.mark.asyncio
async def test_compound_failure_rolls_back_every_record_and_body_receipt(atomic_store: tuple[SqlStepStore, AsyncEngine]) -> None:
    store, engine = atomic_store
    run, events, snapshot = _records()
    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)

    def fail(connection, cursor, statement, parameters, context, many):
        if statement.startswith("INSERT INTO ai_step_message_refs "):
            raise RuntimeError("reference failed")

    event.listen(engine.sync_engine, "before_cursor_execute", fail)
    with pytest.raises(RuntimeError, match="reference failed"):
        await store.commit_step(run=run, events=events, snapshot=snapshot)
    event.remove(engine.sync_engine, "before_cursor_execute", fail)
    assert not store._confirmed_messages
    async with engine.connect() as connection:
        for table in ("runs", "events", "snapshots", "message_refs", "messages"):
            assert await connection.scalar(text(f"SELECT count(*) FROM ai_step_{table}")) == 0
    await store.commit_step(run=run, events=events, snapshot=snapshot)
    assert traffic.message_rows == 2
    assert await store.latest_snapshot(run_id="run") == snapshot


@pytest.mark.asyncio
async def test_compound_lost_commit_ack_requires_readback_without_cache_promotion(
    atomic_store: tuple[SqlStepStore, AsyncEngine], monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, engine = atomic_store
    run, events, snapshot = _records()
    original = AsyncSessionTransaction.__aexit__
    lost = False

    async def lose_ack(self, error_type, error, traceback):
        nonlocal lost
        result = await original(self, error_type, error, traceback)
        if error_type is None and not lost:
            lost = True
            raise OperationalError("commit acknowledgement lost", {}, RuntimeError("lost"),
                                   connection_invalidated=True)
        return result

    traffic = _SqlTraffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    with monkeypatch.context() as patch:
        patch.setattr(AsyncSessionTransaction, "__aexit__", lose_ack)
        with pytest.raises(AIError) as raised:
            await store.commit_step(run=run, events=events, snapshot=snapshot)
        assert raised.value.code == ErrorCode.STORAGE_COMMIT_UNKNOWN
    assert not store._confirmed_messages
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        assert await cold.get_run(run_id="run") == run
        assert await cold.list_events(run_id="run") == events
        assert await cold.latest_snapshot(run_id="run") == snapshot
    finally:
        await cold.close()
    await store.commit_step(events=events, snapshot=snapshot)
    assert not store._confirmed_messages
    await store.commit_step(snapshot=replace(snapshot, idempotency_key="next"))
    assert traffic.message_rows == 2
    await store.commit_step(snapshot=replace(snapshot, idempotency_key="warm"))
    assert traffic.message_rows == 2


@pytest.mark.asyncio
async def test_duplicate_run_rejects_compound_group_without_publishing_tail(atomic_store: tuple[SqlStepStore, AsyncEngine]) -> None:
    store, engine = atomic_store
    run, events, snapshot = _records()
    await store.register_run(run)
    with pytest.raises(ValueError, match="already in the store"):
        await store.commit_step(run=replace(run, registration_id="other"), events=events, snapshot=snapshot)
    assert await store.list_events(run_id="run") == []
    assert await store.latest_snapshot(run_id="run") is None
    assert not store._confirmed_messages


@pytest.mark.asyncio
async def test_parallel_compound_calls_preserve_native_keys_and_each_event_order(atomic_store: tuple[SqlStepStore, AsyncEngine]) -> None:
    store, engine = atomic_store
    run, events, snapshot = _records()
    results = await asyncio.gather(
        *(store.commit_step(run=run, events=events, snapshot=snapshot) for _ in range(2)),
        return_exceptions=True,
    )
    assert all(result is None for result in results)
    await asyncio.gather(*(store.commit_step(events=events, snapshot=snapshot) for _ in range(4)))
    assert await store.list_events(run_id="run") == events
    assert await store.list_snapshots(run_id="run") == [snapshot]
    await asyncio.gather(*(store.commit_step(
        events=[replace(item, idempotency_key=f"{index}:{item.kind}") for item in events],
        snapshot=replace(snapshot, idempotency_key=str(index)),
    ) for index in range(4)))
    found = await store.list_events(run_id="run")
    for index in range(4):
        assert [item.kind for item in found if item.idempotency_key.startswith(f"{index}:")] == [item.kind for item in events]
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 1
        assert await connection.scalar(text("SELECT count(*) FROM ai_step_message_refs")) == 5


@pytest.mark.asyncio
async def test_compound_events_preserve_unkeyed_occurrences_and_first_keyed_value(
    atomic_store: tuple[SqlStepStore, AsyncEngine],
) -> None:
    store, _ = atomic_store
    first = StepEvent(run_id="run", kind="run_started", step_index=0, idempotency_key="first")
    unkeyed = replace(first, kind="model_request_started", idempotency_key=None)
    tail = [replace(first, kind="model_request_completed", step_index=index, idempotency_key=str(index))
            for index in range(129)]
    await store.commit_step(events=(first, replace(first, error="ignored"), unkeyed, unkeyed, *tail))
    assert await store.list_events(run_id="run") == [first, unkeyed, unkeyed, *tail]
    await store.commit_step(events=(replace(first, error="still ignored"), unkeyed, *tail))
    assert await store.list_events(run_id="run") == [first, unkeyed, unkeyed, *tail, unkeyed]


@pytest.mark.asyncio
async def test_inflight_compound_group_is_invisible_to_cold_reader_and_cache(
    atomic_store: tuple[SqlStepStore, AsyncEngine], monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, engine = atomic_store
    run, events, snapshot = _records()
    inserted, release = asyncio.Event(), asyncio.Event()
    original = store._context.dialect.insert_ignore_conflict_many

    async def pause_after_bodies(session, **kwargs):
        result = await original(session, **kwargs)
        if kwargs["table"].name == "ai_step_messages":
            inserted.set()
            await release.wait()
        return result

    monkeypatch.setattr(store._context.dialect, "insert_ignore_conflict_many", pause_after_bodies)
    writer = asyncio.create_task(store.commit_step(run=run, events=events, snapshot=snapshot))
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        await inserted.wait()
        assert not writer.done()
        assert not store._confirmed_messages
        assert await cold.get_run(run_id="run") is None
        assert await cold.list_events(run_id="run") == []
        assert await cold.latest_snapshot(run_id="run") is None
        release.set()
        await writer
        assert store._confirmed_messages
        assert await cold.get_run(run_id="run") == run
        assert await cold.list_events(run_id="run") == events
        assert await cold.latest_snapshot(run_id="run") == snapshot
    finally:
        release.set()
        await writer
        await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_context", (False, True))
async def test_native_boundary_joins_control_transaction_without_reads_or_early_receipts(tmp_path: Path, shared_context: bool) -> None:
    from .test_state_group_record_batch import _database, _record

    async with _database(tmp_path / "joined.db") as (engine, state, _):
        native = SqlStepStore(state.context if shared_context else engine, namespace="steps", tenant_id="one")
        run, events, snapshot = _records()
        prepared = await native.prepare_step(run=run, events=events, snapshot=snapshot)
        control = _record("control")
        statements, commits = [], []

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append(statement)

        async def commit(transaction):
            await transaction.insert_record(control)
            receipt = await native.write_prepared_step(transaction, prepared)
            assert not native._confirmed_messages
            return receipt

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        event.listen(engine.sync_engine, "commit", commits.append)
        receipt = await state.mutate(commit)
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len(commits) == 1
        assert not any(statement.startswith("SELECT") for statement in statements)
        assert not native._confirmed_messages
        native.confirm_prepared_step(receipt)
        assert native._confirmed_messages
        assert await state.read(lambda transaction: transaction.get_record(control.key_digest)) == control
        assert await native.get_run(run_id="run") == run
        assert await native.list_events(run_id="run") == events
        assert await native.latest_snapshot(run_id="run") == snapshot
        await native.close()


@pytest.mark.asyncio
async def test_native_boundary_failure_rolls_back_control_and_native_rows(tmp_path: Path) -> None:
    from .test_state_group_record_batch import _database, _record

    async with _database(tmp_path / "joined-rollback.db") as (engine, state, _):
        native = SqlStepStore(state.context, namespace="steps", tenant_id="one")
        run, events, snapshot = _records()
        prepared = await native.prepare_step(run=run, events=events, snapshot=snapshot)
        control = _record("control")

        async def commit(transaction):
            await transaction.insert_record(control)
            await native.write_prepared_step(transaction, prepared)
            raise RuntimeError("later boundary write failed")

        with pytest.raises(AIError) as raised:
            await state.mutate(commit)
        assert raised.value.code is ErrorCode.INTERNAL_ERROR
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert await state.read(lambda transaction: transaction.get_record(control.key_digest)) is None
        assert await native.get_run(run_id="run") is None
        assert await native.latest_snapshot(run_id="run") is None
        assert not native._confirmed_messages
        await native.close()


@pytest.mark.asyncio
async def test_unkeyed_prepared_boundary_appends_on_each_publication(tmp_path: Path) -> None:
    from .test_state_group_record_batch import _database

    async with _database(tmp_path / "append-prepared.db") as (_, state, _):
        native = SqlStepStore(state.context, namespace="steps", tenant_id="one")
        snapshot = ContinuableSnapshot(run_id="run", step_index=0, messages=[])
        event_record = StepEvent(run_id="run", kind="model_request_started", step_index=0)
        prepared = await native.prepare_step(events=(event_record,), snapshot=snapshot)
        try:
            for _ in range(2):
                receipt = await state.mutate(lambda transaction: native.write_prepared_step(transaction, prepared))
                native.confirm_prepared_step(receipt)
            assert await native.list_events(run_id="run") == [event_record, event_record]
            assert await native.list_snapshots(run_id="run") == [snapshot, snapshot]
        finally:
            await native.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("event", "snapshot"))
async def test_unkeyed_physical_identity_collision_fails_without_losing_append(
    atomic_store: tuple[SqlStepStore, AsyncEngine], monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    from uuid import UUID
    from linktools.ai.persistence import _step_store as step_module

    store, _ = atomic_store
    monkeypatch.setattr(step_module, "uuid4", lambda: UUID(int=1))
    item = (StepEvent(run_id="run", kind="model_request_started", step_index=0) if kind == "event"
            else ContinuableSnapshot(run_id="run", step_index=0, messages=[]))
    write = store.append_event if kind == "event" else store.save_snapshot
    await write(item)
    with pytest.raises(IntegrityError):
        await write(item)
    items = await store.list_events(run_id="run") if kind == "event" else await store.list_snapshots(run_id="run")
    assert items == [item]


@pytest.mark.asyncio
@pytest.mark.parametrize("field, changed", (("run_id", "other"), ("idempotency_key", "other-key")))
async def test_snapshot_publication_uses_captured_run_and_replay_identity(
    atomic_store: tuple[SqlStepStore, AsyncEngine], monkeypatch: pytest.MonkeyPatch,
    field: str, changed: str,
) -> None:
    store, _ = atomic_store
    snapshot = ContinuableSnapshot(run_id="run", step_index=0, messages=[], idempotency_key="key")
    captured = replace(snapshot)
    prepared, release = asyncio.Event(), asyncio.Event()
    context_type = type(store.storage_context)
    original = context_type.run_mutation

    async def pause(context, callback, **kwargs):
        if context is store.storage_context:
            prepared.set()
            await release.wait()
        return await original(context, callback, **kwargs)

    monkeypatch.setattr(context_type, "run_mutation", pause)
    task = asyncio.create_task(store.save_snapshot(snapshot))
    try:
        await prepared.wait()
        setattr(snapshot, field, changed)
        release.set()
        await task
        assert await store.latest_snapshot(run_id="run") == captured
        await store.save_snapshot(replace(captured, step_index=1))
        assert await store.list_snapshots(run_id="run") == [captured]
        other = replace(captured, idempotency_key="other-key", step_index=2)
        await store.save_snapshot(other)
        assert await store.list_snapshots(run_id="run") == [captured, other]
        assert await store.latest_snapshot(run_id="other") is None
    finally:
        release.set()
        await task


@pytest.mark.asyncio
async def test_native_transaction_rejects_separate_context_and_expired_scope(tmp_path: Path) -> None:
    from linktools.ai.persistence.state import StateTransactionNestingError
    from .test_state_group_record_batch import _database

    async with _database(tmp_path / "scope.db") as (engine, state, _):
        native = SqlStepStore(state.context, namespace="steps", tenant_id="one")
        separate_engine = create_async_engine(engine.url)
        separate = SqlStepStore(separate_engine, namespace="steps", tenant_id="one")
        run, events, snapshot = _records()
        prepared = await native.prepare_step(run=run, events=events, snapshot=snapshot)
        captured = []

        async def commit(transaction):
            captured.append(transaction)
            with pytest.raises(StateTransactionNestingError, match="active writable"):
                await separate.write_prepared_step(transaction, prepared)

        await state.mutate(commit)
        with pytest.raises(StateTransactionNestingError, match="active writable"):
            await native.write_prepared_step(captured[0], prepared)
        assert await native.get_run(run_id="run") is None
        await native.close()
        await separate.close()
        await separate_engine.dispose()


@pytest.mark.asyncio
async def test_pruned_headers_keep_replay_identity_without_body_retention_roots(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tombstones.db'}")
    native = SqlStepStore(engine, namespace="steps", tenant_id="one", max_snapshots_per_run=1)
    await native.initialize(local=True)
    first = ContinuableSnapshot(run_id="run", step_index=0, idempotency_key="first",
        messages=[ModelRequest(parts=[UserPromptPart(value)]) for value in ("a", "b")])
    next_snapshot = replace(first, step_index=1, idempotency_key="next",
        messages=[*first.messages, ModelRequest(parts=[UserPromptPart("c")])])
    try:
        await native.save_snapshot(first)
        await native.save_snapshot(next_snapshot)
        cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
        assert await cold.latest_snapshot(run_id="run") == next_snapshot
        await cold.close()
        last = replace(first, step_index=2, idempotency_key="last",
            messages=[ModelRequest(parts=[UserPromptPart("retained")])])
        await native.save_snapshot(last)
        await native.close()
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_messages WHERE sha256 NOT IN (SELECT sha256 FROM ai_step_message_refs)"))
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 1
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_snapshots WHERE pruned = 1 AND messages = '{}'")) == 2
        cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
        await cold.save_snapshot(replace(first, messages=[ModelRequest(parts=[UserPromptPart("different replay")])]))
        assert await cold.list_snapshots(run_id="run", include_interrupted=True) == [last]
        async with engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 1
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_message_refs")) == 1
        await cold.close()
    finally:
        await native.close()
        await engine.dispose()
