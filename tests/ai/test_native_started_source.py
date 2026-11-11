#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native run-start receipts bind one immutable source without extra reads."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent
from sqlalchemy import event, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.persistence import SqlStepStore, StepWriteReceipt


@pytest_asyncio.fixture
async def native(tmp_path: Path) -> AsyncIterator[tuple[SqlStepStore, AsyncEngine]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'started.db'}")
    store = SqlStepStore(engine, namespace="native", tenant_id="one")
    await store.initialize(local=True)
    try:
        yield store, engine
    finally:
        await store.close()
        await engine.dispose()


def records() -> tuple[RunRecord, StepEvent]:
    run = RunRecord(run_id="run", registration_id="registration", conversation_id="conversation",
                    parent_run_id="parent", agent_name="agent", metadata={"caller": "原样"})
    started = StepEvent(run_id=run.run_id, kind="run_started", step_index=0,
                        timestamp=run.started_at, conversation_id=run.conversation_id,
                        parent_run_id=run.parent_run_id, agent_name=run.agent_name,
                        metadata=dict(run.metadata), idempotency_key="start")
    return run, started


async def managed_commit(
    store: SqlStepStore, *, run: RunRecord, events: tuple[StepEvent, ...],
) -> StepWriteReceipt:
    prepared = await store.prepare_step(run=run, events=events, managed_start=True)
    receipt = await store.storage_context.run_mutation(
        lambda session: store._write_prepared_step(session, prepared), domain="steps.boundary",
    )
    store.confirm_prepared_step(receipt)
    return receipt


@pytest.mark.asyncio
async def test_fresh_and_confirmed_start_skip_redundant_writes(native) -> None:
    store, engine = native
    run, started = records()
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        receipt = await managed_commit(store, run=run, events=(started,))
        assert not any(sql.startswith("SELECT") for sql in statements)
        assert store.started_from_receipt(receipt) == started
        assert store.confirmed_started(run.run_id) == started
        statements.clear()
        await managed_commit(store, run=run, events=(started,))
        assert not any(sql.startswith(("SELECT", "INSERT", "UPDATE")) for sql in statements)
        with pytest.raises(ValueError, match="body differs"):
            await managed_commit(store, run=run, events=(replace(started, metadata={"caller": "changed"}),))
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert await store.list_events(run_id=run.run_id) == [started]


@pytest.mark.asyncio
async def test_cold_replay_uses_existing_registration_select(native) -> None:
    store, engine = native
    run, started = records()
    await managed_commit(store, run=run, events=(started,))
    cold = SqlStepStore(engine, namespace="native", tenant_id="one")
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        receipt = await managed_commit(cold, run=run, events=(started,))
        selects = [sql for sql in statements if sql.startswith("SELECT")]
        assert len(selects) == 1 and "JOIN ai_step_events" in selects[0]
        selected_columns = selects[0].split("FROM", 1)[0]
        for excluded in ("created_at", "updated_at"):
            assert excluded not in selected_columns
        assert "ai_step_events.id AS" not in selected_columns
        assert "ai_step_runs.id," not in selected_columns
        assert not any(sql.startswith("INSERT INTO ai_step_events") for sql in statements)
        assert cold.started_from_receipt(receipt) == started
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await cold.close()


@pytest.mark.asyncio
async def test_start_receipt_is_provisional_until_outer_commit(native) -> None:
    store, _ = native
    run, started = records()
    prepared = await store.prepare_step(run=run, events=(started,), managed_start=True)
    async with store.storage_context.sessions() as session:
        async with session.begin():
            receipt = await store._write_prepared_step(session, prepared)
            assert store.started_from_receipt(receipt) == started
            assert store.confirmed_started(run.run_id) is None
        assert store.confirmed_started(run.run_id) is None
        store.confirm_prepared_step(receipt)
    assert store.confirmed_started(run.run_id) == started


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["body", "missing", "scope", "hash"])
async def test_cold_source_damage_fails_closed(native, damage: str) -> None:
    store, engine = native
    run, started = records()
    await managed_commit(store, run=run, events=(started,))
    mutations = {
        "body": "UPDATE ai_step_events SET error = 'changed'",
        "missing": "DELETE FROM ai_step_events",
        "scope": "UPDATE ai_step_events SET store_key = 'wrong'",
        "hash": "UPDATE ai_step_events SET body_sha256 = NULL",
    }
    async with engine.begin() as connection:
        await connection.execute(text(mutations[damage]))
    cold = SqlStepStore(engine, namespace="native", tenant_id="one")
    try:
        with pytest.raises(ValueError):
            await managed_commit(cold, run=run, events=(started,))
        assert cold.confirmed_started(run.run_id) is None
    finally:
        await cold.close()


@pytest.mark.asyncio
async def test_public_decoder_checks_inherited_fields_and_scope(native) -> None:
    store, engine = native
    run, started = records()
    await managed_commit(store, run=run, events=(started,))
    async with engine.connect() as connection:
        source = dict((await connection.execute(select(store._table("events")))).mappings().one())
        registration = dict((await connection.execute(select(store._table("runs")))).mappings().one())
    assert source["run_fields"] == 15
    assert SqlStepStore.decode_started_row(source, expected_store_key=store.store_key, run=registration) == started
    for corrupted in (source | {"idempotency_key": b"wrong"}, source | {"run_id": "wrong"},
                      source | {"run_fields": 16}, source | {"body_sha256": None},
                      source | {"step_index": "0"}, source | {"step_index": False}):
        with pytest.raises(ValueError):
            store.decode_started_source(corrupted, run=registration)
    with pytest.raises(ValueError):
        store.decode_started_source(source, run=registration | {"metadata": '{"caller":"changed"}'})
    with pytest.raises(ValueError):
        SqlStepStore.decode_started_row(source, expected_store_key="another", run=registration)


@pytest.mark.asyncio
async def test_key_collision_cannot_mint_source_proof(native) -> None:
    store, _ = native
    run, started = records()
    await store.append_event(replace(started, error="unmanaged collision"))
    with pytest.raises(Exception):
        await managed_commit(store, run=run, events=(started,))
    assert store.confirmed_started(run.run_id) is None
    assert await store.get_run(run_id=run.run_id) is None


@pytest.mark.asyncio
async def test_start_proof_counts_against_run_cache_byte_limit(native, monkeypatch) -> None:
    from linktools.ai.persistence import _step_store

    store, _ = native
    run, started = records()
    monkeypatch.setattr(_step_store, "_RUN_CACHE_BYTES", 1)
    receipt = await managed_commit(store, run=run, events=(started,))
    assert store.confirmed_started(run.run_id) is None
    assert store.started_from_receipt(receipt) == started


@pytest.mark.asyncio
async def test_caller_receipt_survives_cache_eviction_without_sql_reads(native, monkeypatch) -> None:
    from linktools.ai.persistence import _step_store

    store, engine = native
    run, started = records()
    monkeypatch.setattr(_step_store, "_RUN_CACHE_BYTES", 1)
    receipt = await managed_commit(store, run=run, events=(started,))
    prepared = await store.prepare_step(run=run, events=(started,), managed_start=True)
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        async with store.storage_context.sessions() as session, session.begin():
            replay = await store._write_prepared_step(session, prepared, confirmed=receipt)
        assert store.started_from_receipt(replay) == started
        assert not any(sql.startswith(("SELECT", "INSERT", "UPDATE")) for sql in statements)
        changed = await store.prepare_step(run=run, events=(replace(started, error="changed"),), managed_start=True)
        async with store.storage_context.sessions() as session, session.begin():
            with pytest.raises(ValueError, match="body differs"):
                await store._write_prepared_step(session, changed, confirmed=receipt)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.asyncio
async def test_reading_same_run_preserves_confirmed_start(native) -> None:
    store, _ = native
    run, started = records()
    await managed_commit(store, run=run, events=(started,))
    assert await store.get_run(run_id=run.run_id) == run
    assert store.confirmed_started(run.run_id) == started
    assert await store.list_runs() == [run]
    assert store.confirmed_started(run.run_id) == started


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["hash_null", "locator", "missing", "key"])
async def test_unknown_readback_requires_native_source_proof(native, damage: str) -> None:
    store, engine = native
    run, started = records()
    prepared = await store.prepare_step(run=run, events=(started,), managed_start=True)
    async with store.storage_context.sessions() as session, session.begin():
        receipt = await store._write_prepared_step(session, prepared)
    assert store.confirmed_started(run.run_id) is None
    assert await store.list_events(run_id=run.run_id, started_receipt=receipt) == [started]
    assert store.confirmed_started(run.run_id) is None
    mutations = {
        "hash_null": "UPDATE ai_step_events SET body_sha256 = NULL",
        "locator": "UPDATE ai_step_events SET idempotency_key = X'77726F6E67'",
        "missing": "DELETE FROM ai_step_events",
        "key": "UPDATE ai_step_events SET idempotency_key = 'wrong'",
    }
    async with engine.begin() as connection:
        await connection.execute(text(mutations[damage]))
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        with pytest.raises(ValueError):
            await store.list_events(run_id=run.run_id, started_receipt=receipt)
        assert len([sql for sql in statements if sql.startswith("SELECT")]) == 1
        assert store.confirmed_started(run.run_id) is None
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.asyncio
async def test_unmanaged_registration_then_compound_start_remains_valid(native) -> None:
    store, engine = native
    run, started = records()
    await store.register_run(run)
    await store.commit_step(run=run, events=(started,))
    assert await store.list_events(run_id=run.run_id) == [started]
    assert store.confirmed_started(run.run_id) is None
    async with engine.connect() as connection:
        assert (await connection.execute(text("SELECT body_sha256 FROM ai_step_events"))).scalar_one() is None


@pytest.mark.asyncio
async def test_managed_source_requires_explicit_complete_admission(native) -> None:
    store, _ = native
    run, started = records()
    for registration, events in ((None, (started,)), (run, ()),
                                 (run, (replace(started, idempotency_key=None),)),
                                 (run, (started, started))):
        with pytest.raises(ValueError, match="managed start"):
            await store.prepare_step(run=registration, events=events, managed_start=True)
    receipt = await managed_commit(store, run=run, events=(started,))
    unmanaged = await store.prepare_step(run=run, events=(started,))
    async with store.storage_context.sessions() as session, session.begin():
        with pytest.raises(ValueError, match="explicitly managed"):
            await store._write_prepared_step(session, unmanaged, confirmed=receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("replay_mode", ["warm", "cold", "confirmed"])
async def test_registration_linked_metadata_drift_reuses_authoritative_source(native, replay_mode: str) -> None:
    store, engine = native
    run, started = records()
    receipt = await managed_commit(store, run=run, events=(started,))
    candidate_run = replace(run, metadata={"caller": "new host value"})
    candidate_start = replace(started, metadata=dict(candidate_run.metadata))
    target = SqlStepStore(engine, namespace="native", tenant_id="one") if replay_mode == "cold" else store
    try:
        if replay_mode == "confirmed":
            prepared = await target.prepare_step(run=candidate_run, events=(candidate_start,), managed_start=True)
            async with target.storage_context.sessions() as session, session.begin():
                replay = await target._write_prepared_step(session, prepared, confirmed=receipt)
        else:
            replay = await managed_commit(target, run=candidate_run, events=(candidate_start,))
        assert target.run_from_receipt(replay) == run
        assert target.started_from_receipt(replay) == started
        with pytest.raises(ValueError, match="body differs"):
            await managed_commit(target, run=candidate_run, events=(replace(candidate_start, metadata={"independent": "change"}),))
        with pytest.raises(ValueError, match="body differs"):
            await managed_commit(target, run=replace(candidate_run, conversation_id="different"), events=(candidate_start,))
    finally:
        if target is not store:
            await target.close()


@pytest.mark.asyncio
async def test_narrow_start_fields_proof_does_not_claim_metadata_integrity(native) -> None:
    store, engine = native
    run, started = records()
    await managed_commit(store, run=run, events=(started,))
    async with engine.connect() as connection:
        source = dict((await connection.execute(select(store._table("events")))).mappings().one())
        registration = dict((await connection.execute(select(store._table("runs")))).mappings().one())
    full = store.decode_started_source(source, run=registration)
    from dataclasses import fields

    projected = {field.name: getattr(full, field.name) for field in fields(full) if field.name != "metadata"}
    projected["timestamp"] = full.timestamp.isoformat()
    projected.update({name: source[name] for name in ("outcome", "store_key", "idempotency_key", "fields_sha256")})
    assert "metadata" not in projected and "body_sha256" not in projected
    assert SqlStepStore.decode_started_fields(projected, expected_store_key=store.store_key) == replace(started, metadata={})
    for damage in ({"error": "changed"}, {"fields_sha256": None}, {"fields_sha256": "wrong"}, {"step_index": "0"}):
        with pytest.raises(ValueError):
            SqlStepStore.decode_started_fields(projected | damage, expected_store_key=store.store_key)
    with pytest.raises(ValueError):
        store.decode_started_source(source | {"fields_sha256": None}, run=registration)
    changed_registration = registration | {"metadata": '{"caller":"tampered"}'}
    with pytest.raises(ValueError):
        store.decode_started_source(source, run=changed_registration)
    # Narrow projection authenticates only the fields it actually consumes.
    assert SqlStepStore.decode_started_fields(projected, expected_store_key=store.store_key).metadata == {}


@pytest.mark.asyncio
async def test_unmanaged_model_tool_writes_omit_start_proof_columns(native, monkeypatch) -> None:
    store, engine = native
    run, _ = records()

    def unexpected_hash(cls, value):
        raise AssertionError("ordinary model/tool events must not compute managed start proofs")

    monkeypatch.setattr(SqlStepStore, "_started_body", classmethod(unexpected_hash))
    monkeypatch.setattr(SqlStepStore, "_started_fields_body", classmethod(unexpected_hash))
    events = tuple(StepEvent(run_id=run.run_id, kind=kind, step_index=index,
                            idempotency_key=f"event-{index}", metadata={"caller": "unchanged"})
                   for index, kind in enumerate(("model_request_started", "tool_call_started")))
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        await store.commit_step(run=run, events=events)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    inserts = [sql for sql in statements if sql.startswith("INSERT INTO ai_step_events")]
    assert inserts
    assert all("body_sha256" not in sql and "fields_sha256" not in sql for sql in inserts)
    async with engine.connect() as connection:
        rows = (await connection.execute(text("SELECT body_sha256, fields_sha256 FROM ai_step_events"))).all()
    assert rows == [(None, None), (None, None)]


@pytest.mark.asyncio
async def test_managed_start_write_records_proof_encoding_cost(native, monkeypatch, record_property) -> None:
    import time

    store, engine = native
    run, started = records()
    run = replace(run, metadata={"caller": "原样" * (64 * 1024)})
    started = replace(started, metadata=dict(run.metadata))
    measurements = {"body": {"calls": 0, "encoded_bytes": 0, "cpu_ns": 0},
                    "fields": {"calls": 0, "encoded_bytes": 0, "cpu_ns": 0}}
    original_body = SqlStepStore._started_body
    original_fields = SqlStepStore._started_fields_body

    def measure(label, operation, value):
        before = time.process_time_ns()
        result = operation(value)
        measurements[label]["cpu_ns"] += time.process_time_ns() - before
        measurements[label]["calls"] += 1
        measurements[label]["encoded_bytes"] += len(result.encode("utf-8"))
        return result

    def body(cls, value):
        return measure("body", original_body, value)

    def fields(cls, value):
        return measure("fields", original_fields, value)

    monkeypatch.setattr(SqlStepStore, "_started_body", classmethod(body))
    monkeypatch.setattr(SqlStepStore, "_started_fields_body", classmethod(fields))
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    before = time.process_time_ns()
    try:
        receipt = await managed_commit(store, run=run, events=(started,))
    finally:
        write_cpu_ns = time.process_time_ns() - before
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    for label, values in measurements.items():
        for name, value in values.items():
            record_property(f"managed_start_{label}_{name}", value)
    record_property("managed_start_write_cpu_ns", write_cpu_ns)
    record_property("managed_start_metadata_utf8_bytes", len(run.metadata["caller"].encode("utf-8")))
    record_property("managed_start_selects_sqlite", sum(sql.startswith("SELECT") for sql in statements))
    inserts = [sql for sql in statements if sql.startswith("INSERT INTO ai_step_events")]
    assert len(inserts) == 1
    assert "body_sha256" in inserts[0] and "fields_sha256" in inserts[0]
    assert measurements["body"]["calls"] >= 1 and measurements["fields"]["calls"] >= 1
    assert measurements["body"]["encoded_bytes"] > measurements["fields"]["encoded_bytes"]
    assert store.started_from_receipt(receipt) == started
    async with engine.connect() as connection:
        row = (await connection.execute(text("SELECT body_sha256, fields_sha256 FROM ai_step_events"))).one()
    assert len(row.body_sha256) == len(row.fields_sha256) == 64
