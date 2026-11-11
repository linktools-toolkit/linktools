#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact native run references reduce event bytes without extra SQL requests."""

import asyncio
import json
import time
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import replace
from itertools import groupby
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai_harness.step_persistence import RunRecord, StepEvent
from sqlalchemy import Text, event, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence import _step_store as module
from linktools.ai.storage import create_sql_storage_context


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[tuple[SqlStepStore, AsyncEngine]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'references.db'}")
    adapter = SqlStepStore(engine, namespace="steps", tenant_id="one")
    await adapter.initialize(local=True)
    try:
        yield adapter, engine
    finally:
        await adapter.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_public_native_scope_identity_matches_physical_store(store) -> None:
    adapter, _ = store
    assert adapter.store_key == SqlStepStore.scope_key(namespace="steps", tenant_id="one")
    assert adapter.store_key != SqlStepStore.scope_key(namespace="steps", tenant_id="other")
    assert adapter.store_key != SqlStepStore.scope_key(namespace="other", tenant_id="one")


@pytest.mark.parametrize("namespace, tenant_id", [("", "one"), ("steps", ""), (None, "one"), ("steps", 1)])
def test_native_scope_identity_rejects_invalid_partition(namespace, tenant_id) -> None:
    with pytest.raises(AIError) as rejected:
        SqlStepStore.scope_key(namespace=namespace, tenant_id=tenant_id)
    assert rejected.value.code is ErrorCode.REQUEST_FIELD_INVALID


def _records() -> tuple[RunRecord, StepEvent]:
    run = RunRecord(run_id="run", registration_id="registration", conversation_id="conversation",
                    parent_run_id="parent", agent_name="agent", metadata={"caller": "原样 " * 1024})
    boundary = StepEvent(run_id=run.run_id, kind="tool_call_failed", step_index=3,
                         conversation_id=run.conversation_id, parent_run_id=run.parent_run_id,
                         agent_name=run.agent_name, metadata=dict(run.metadata),
                         tool_call_id="call", tool_name="tool", error="original error", idempotency_key="first")
    return run, boundary


async def _rows(engine: AsyncEngine) -> list[dict[str, object]]:
    async with engine.connect() as connection:
        return [dict(row) for row in (await connection.execute(text(
            "SELECT conversation_id, parent_run_id, agent_name, metadata, run_fields "
            "FROM ai_step_events ORDER BY id"
        ))).mappings()]


@pytest.mark.asyncio
async def test_exact_fields_reference_native_run_and_custom_null_values_stay_inline(store) -> None:
    adapter, engine = store
    run, first = _records()
    custom = replace(first, conversation_id=None, parent_run_id="different parent", agent_name="agent ",
                     metadata={"caller": "different", "run_fields": "8", "custom": "\u0000原样"}, idempotency_key="custom")
    partial = replace(first, metadata={}, idempotency_key="partial")
    await adapter.commit_step(run=run, events=(first, custom, partial))
    rows = await _rows(engine)
    assert rows[0] == dict(conversation_id=None, parent_run_id=None, agent_name=None, metadata=None, run_fields=15)
    assert rows[1] == dict(conversation_id=None, parent_run_id="different parent", agent_name="agent ",
                           metadata=json.dumps(custom.metadata), run_fields=0)
    assert rows[2]["metadata"] == "{}" and rows[2]["run_fields"] == 7
    statements = []

    def capture(connection, cursor, statement, parameters, context, many):
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert await adapter.list_events(run_id=run.run_id) == [first, custom, partial]
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
    assert len(statements) == 1 and statements[0].startswith("SELECT")
    assert "LEFT OUTER JOIN ai_step_runs" in statements[0]
    # The join reads the native owner inside this query; it is not a zero-read cache.
    assert statements[0].count("JOIN ai_step_runs") == 1


@pytest.mark.asyncio
async def test_no_source_cold_append_and_registration_replay_never_guess_run_fields(store) -> None:
    adapter, engine = store
    run, first = _records()
    await adapter.append_event(first)
    await adapter.register_run(run)
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        second = replace(first, idempotency_key="cold")
        await cold.append_event(second)
        supplied = replace(run, metadata={"caller": "changed"})
        replayed = replace(first, metadata=dict(supplied.metadata), idempotency_key="replay")
        statements = []

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await cold.commit_step(run=supplied, events=(replayed,))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len([sql for sql in statements if sql.startswith("SELECT")]) == 1
        assert all("metadata" in sql for sql in statements if sql.startswith("SELECT"))
        assert [row["run_fields"] for row in await _rows(engine)] == [0, 0, 7]
        assert cold.confirmed_run(run.run_id) == run
        assert await cold.list_events(run_id=run.run_id) == [first, second, replayed]
        # The replay's existing SELECT supplies the original source, not caller metadata.
        assert cold.confirmed_run(run.run_id) == run
        await cold.append_event(replace(first, idempotency_key="after-read"))
        assert (await _rows(engine))[-1]["run_fields"] == 15
    finally:
        await cold.close()


@pytest.mark.asyncio
async def test_native_refs_keep_first_keyed_value_and_every_unkeyed_occurrence(store) -> None:
    adapter, _ = store
    run, first = _records()
    unkeyed = replace(first, idempotency_key=None)
    await adapter.commit_step(run=run, events=(first, replace(first, metadata={"changed": "ignored"}), unkeyed, unkeyed))
    await adapter.commit_step(events=(replace(first, agent_name="ignored"), unkeyed))
    assert await adapter.list_events(run_id=run.run_id) == [first, unkeyed, unkeyed, unkeyed]


@pytest.mark.asyncio
async def test_event_source_cache_is_partitioned_detached_bounded_and_closed(store, monkeypatch) -> None:
    adapter, engine = store
    run, first = _records()
    await adapter.register_run(run)
    run.metadata["caller"] = "mutated caller object"
    await adapter.append_event(first)
    await adapter.append_event(replace(first, metadata=dict(run.metadata), idempotency_key="mutated"))
    other = SqlStepStore(engine, namespace="steps", tenant_id="other")
    try:
        await other.append_event(replace(first, idempotency_key="other"))
        rows = await _rows(engine)
        assert [row["run_fields"] for row in rows] == [15, 7, 0]
        assert await other.list_events(run_id=run.run_id) == [replace(first, idempotency_key="other")]
    finally:
        await other.close()
    monkeypatch.setattr(module, "_RUN_CACHE_SIZE", 1)
    await adapter.register_run(replace(run, run_id="second"))
    assert list(adapter._confirmed_runs) == ["second"]
    monkeypatch.setattr(module, "_RUN_CACHE_BYTES", 256)
    # Oversized sources still work in the original transaction, but are not retained.
    await adapter.close()
    assert not adapter._confirmed_runs
    large = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        await large.commit_step(run=replace(run, run_id="oversized", metadata=dict(first.metadata)),
                                events=(replace(first, run_id="oversized"),))
        assert not large._confirmed_runs
        await large.append_event(replace(first, run_id="oversized", idempotency_key="inline"))
        assert [row["run_fields"] for row in (await _rows(engine))[-2:]] == [15, 0]
        assert sum(source.byte_size for source in large._confirmed_runs.values()) <= 256
    finally:
        await large.close()


@pytest.mark.asyncio
async def test_event_source_cache_evicts_by_encoded_bytes_not_only_count(store, monkeypatch) -> None:
    adapter, _ = store
    monkeypatch.setattr(module, "_RUN_CACHE_BYTES", 512)
    for index in range(5):
        await adapter.register_run(RunRecord(run_id=str(index), metadata={"payload": "x" * 180}))
        assert sum(source.byte_size for source in adapter._confirmed_runs.values()) <= 512
    assert len(adapter._confirmed_runs) < 5


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rollback", "lost_ack", "cancelled_ack"])
async def test_failed_or_uncertain_commit_does_not_confirm_event_source(store, monkeypatch, failure) -> None:
    adapter, engine = store
    run, first = _records()
    original = type(adapter.storage_context).run_mutation

    async def fail(context, callback, **kwargs):
        if failure == "rollback":
            async def rollback(session):
                await callback(session)
                raise RuntimeError("rollback source")
            return await original(context, rollback, **kwargs)
        await original(context, callback, **kwargs)
        if failure == "cancelled_ack":
            raise asyncio.CancelledError()
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    with monkeypatch.context() as patch:
        patch.setattr(type(adapter.storage_context), "run_mutation", fail)
        expected = RuntimeError if failure == "rollback" else asyncio.CancelledError if failure == "cancelled_ack" else AIError
        with pytest.raises(expected):
            await adapter.commit_step(run=run, events=(first,))
    assert not adapter._confirmed_runs
    if failure == "rollback":
        assert await adapter.get_run(run_id=run.run_id) is None
        assert await _rows(engine) == []
    else:
        cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
        try:
            assert await cold.list_events(run_id=run.run_id) == [first]
        finally:
            await cold.close()
    tail = replace(first, idempotency_key="after-failure")
    await adapter.append_event(tail)
    assert (await _rows(engine))[-1]["run_fields"] == 0


@pytest.mark.asyncio
async def test_joined_transaction_uses_source_before_commit_but_confirms_only_afterward(tmp_path) -> None:
    from .test_state_group_record_batch import _database

    async with _database(tmp_path / "joined.db") as (engine, state, _):
        adapter = SqlStepStore(state.context, namespace="steps", tenant_id="one")
        run, first = _records()
        prepared = await adapter.prepare_step(run=run, events=(first,))
        statements, commits = [], []
        record_commit = commits.append

        def capture(connection, cursor, statement, parameters, context, many):
            statements.append(statement)

        async def write(transaction):
            receipt = await adapter.write_prepared_step(transaction, prepared)
            assert not adapter._confirmed_runs
            return receipt

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        event.listen(engine.sync_engine, "commit", record_commit)
        try:
            receipt = await state.mutate(write)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
            event.remove(engine.sync_engine, "commit", record_commit)
        assert len(commits) == 1
        assert Counter(sql.split()[0] for sql in statements) == {"BEGIN": 1, "INSERT": 2}
        assert not adapter._confirmed_runs
        adapter.confirm_prepared_step(receipt)
        assert (await _rows(engine))[0]["run_fields"] == 15
        assert await adapter.list_events(run_id=run.run_id) == [first]
        assert list(adapter._confirmed_runs) == [run.run_id]
        await adapter.close()


@pytest.mark.asyncio
async def test_missing_native_reference_fails_instead_of_changing_event_values(store) -> None:
    adapter, engine = store
    run, first = _records()
    await adapter.commit_step(run=run, events=(first,))
    await adapter.close()
    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM ai_step_runs"))
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        with pytest.raises(ValueError, match="event run reference is missing"):
            await cold.list_events(run_id=run.run_id)
    finally:
        await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["unknown_bit", "negative_mask", "fractional_mask", "inline_conflict", "null_owner"])
async def test_invalid_native_event_reference_fails_without_returning_invented_values(store, corruption) -> None:
    adapter, engine = store
    run, first = _records()
    if corruption == "fractional_mask":
        await adapter.append_event(first)
        await adapter.register_run(run)
    else:
        await adapter.commit_step(run=run, events=(first,))
    mutations = {
        "unknown_bit": "UPDATE ai_step_events SET run_fields = 16",
        "negative_mask": "UPDATE ai_step_events SET run_fields = -1",
        "fractional_mask": "UPDATE ai_step_events SET run_fields = 0.5",
        "inline_conflict": "UPDATE ai_step_events SET conversation_id = 'conflicting inline value'",
        "null_owner": "UPDATE ai_step_runs SET conversation_id = NULL",
    }
    async with engine.begin() as connection:
        await connection.execute(text(mutations[corruption]))
    with pytest.raises(ValueError, match="event run reference is missing or corrupt"):
        await adapter.list_events(run_id=run.run_id)


@pytest.mark.asyncio
async def test_confirmed_run_is_detached_actual_registration_proof_not_replay_input(store) -> None:
    adapter, engine = store
    run, first = _records()
    assert adapter.confirmed_run(run.run_id) is None
    await adapter.commit_step(run=run, events=(first,))
    proof = adapter.confirmed_run(run.run_id)
    assert proof == run
    proof.metadata["caller"] = "changed copy"
    assert adapter.confirmed_run(run.run_id) == run
    await adapter.commit_step(run=replace(run, metadata={"caller": "changed replay"}))
    assert adapter.confirmed_run(run.run_id) == run
    cold = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        await cold.commit_step(run=replace(run, metadata={"caller": "changed replay"}))
        assert cold.confirmed_run(run.run_id) == run
        assert await cold.get_run(run_id=run.run_id) == run
        assert cold.confirmed_run(run.run_id) == run
    finally:
        await cold.close()
    assert cold.confirmed_run(run.run_id) is None


@pytest.mark.asyncio
async def test_run_receipt_cannot_confirm_another_physical_store(tmp_path) -> None:
    engines = [create_async_engine(f"sqlite+aiosqlite:///{tmp_path / (name + '.db')}") for name in ("one", "two")]
    stores = [SqlStepStore(engine, namespace="same", tenant_id="same") for engine in engines]
    try:
        for adapter in stores:
            await adapter.initialize(local=True)
        run, _ = _records()
        await stores[0].register_run(run)
        receipt = module.StepWriteReceipt(("a" * 64,), ("b" * 64,), stores[0]._confirmed_runs[run.run_id])
        with pytest.raises(ValueError, match="another native store"):
            stores[1].confirm_prepared_step(receipt)
        assert stores[1].confirmed_run(run.run_id) is None
        assert not stores[1]._confirmed_messages
        assert not stores[1]._confirmed_sequences
    finally:
        for adapter in stores:
            await adapter.close()
        for engine in engines:
            await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["cold", "matching", "different"])
async def test_event_metadata_json_encodes_only_for_comparison_or_binding(tmp_path, monkeypatch, source) -> None:
    counts = Counter()
    active = False
    original_dumps, original_loads = json.dumps, json.loads

    def encoded(value, *args, **kwargs):
        if active and isinstance(value, dict):
            counts["comparison"] += 1
        return original_dumps(value, *args, **kwargs)

    def decoded(value, *args, **kwargs):
        if active:
            counts["decode"] += 1
        return original_loads(value, *args, **kwargs)

    def bind_json(value):
        if active:
            counts["bind"] += 1
        return original_dumps(value)

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'encoding.db'}", json_serializer=bind_json)
    adapter = SqlStepStore(engine, namespace="steps", tenant_id="one")
    try:
        await adapter.initialize(local=True)
        run, boundary = _records()
        if source != "cold":
            await adapter.register_run(run)
        if source == "different":
            boundary = replace(boundary, metadata={"caller": "different", "arbitrary.key": "原样"})
        monkeypatch.setattr(module.json, "dumps", encoded)
        monkeypatch.setattr(module.json, "loads", decoded)
        active = True
        await adapter.append_event(boundary)
        active = False
        assert counts["decode"] == 0
        assert counts["comparison"] == (0 if source == "cold" else 1)
        assert counts["bind"] == (0 if source == "matching" else 1)
        assert await adapter.list_events(run_id=run.run_id) == [boundary]
    finally:
        active = False
        await adapter.close()
        await engine.dispose()


def test_event_metadata_values_are_detached_without_json_round_trip() -> None:
    _, boundary = _records()
    values = SqlStepStore._event_values(boundary)
    assert values["metadata"] == boundary.metadata
    assert values["metadata"] is not boundary.metadata
    boundary.metadata["caller"] = "changed after capture"
    assert values["metadata"] != boundary.metadata


class _InlineStepStore(SqlStepStore):
    """Current JSON inline policy, not the historical TEXT encoding path."""

    async def _write_events(self, session, events, *, source=None, started=None, replay_keys=None):
        assert started is None, "the inline comparison does not own managed starts"
        for keyed, group in groupby(events, key=lambda item: item.idempotency_key is not None):
            rows = [self._run_values(item.run_id) | self._event_values(item) for item in group]
            for row in rows:
                row["idempotency_key"] = module.encode_native_replay_key(row["idempotency_key"])
            batch_size = module._BATCH_SIZE if keyed else 1
            for offset in range(0, len(rows), batch_size):
                if not keyed:
                    await session.execute(self._table("events").insert().values(rows[offset]))
                else:
                    await self._context.dialect.insert_ignore_conflict_many(
                        session, table=self._table("events"), rows=rows[offset:offset + batch_size],
                        index_elements=("store_locator", "run_id", "idempotency_key"),
                    )

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self.initialize()
        table = self._table("events")
        async with self._context.sessions() as session:
            rows = (await self._context.run_statement(session.execute(select(table).where(
                *self._run_predicates(table, run_id),
            ).order_by(table.c.id)))).mappings().all()
        return [module._EVENT_ADAPTER.validate_python(dict(row) | {
            "idempotency_key": module.decode_native_replay_key(row["idempotency_key"]),
        }) for row in rows]


class _TextInlineStepStore(_InlineStepStore):
    """Former TEXT event codec only; all modes share the current Run lifecycle."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._table("events").c.metadata.type = Text()

    @classmethod
    def _event_values(cls, event):
        values = super()._event_values(event)
        values["metadata"] = json.dumps(values["metadata"])
        return values

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        await self.initialize()
        table = self._table("events")
        async with self._context.sessions() as session:
            rows = (await self._context.run_statement(session.execute(select(table).where(
                *self._run_predicates(table, run_id),
            ).order_by(table.c.id)))).mappings().all()
        return [module._EVENT_ADAPTER.validate_python(self._record_data(row)) for row in rows]


def _bytes(value: object) -> int:
    if isinstance(value, (tuple, list)):
        return sum(_bytes(item) for item in value)
    if value is None:
        return 0
    return len(value) if isinstance(value, bytes) else len(str(value).encode("utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_matches_run", [True, False])
async def test_matched_event_lifecycle_dml_bound_returned_bytes_and_cpu(
    tmp_path, monkeypatch, record_property, metadata_matches_run,
) -> None:
    samples = {}
    active = None
    original_dumps, original_loads = json.dumps, json.loads

    def encoded(value, *args, **kwargs):
        result = original_dumps(value, *args, **kwargs)
        if active is not None and isinstance(value, dict):
            active["application_metadata_encodes"] += 1
            active["application_metadata_encoded_bytes"] += len(result.encode("utf-8"))
        return result

    def decoded(value, *args, **kwargs):
        result = original_loads(value, *args, **kwargs)
        if active is not None and isinstance(result, dict):
            active["application_metadata_decodes"] += 1
        return result

    def bind_json(value):
        result = original_dumps(value)
        if active is not None:
            active["json_bind_encodes"] += 1
            active["json_bind_encoded_bytes"] += len(result.encode("utf-8"))
        return result

    def read_json(value):
        if active is not None:
            active["json_result_decodes"] += 1
        return original_loads(value)

    monkeypatch.setattr(module.json, "dumps", encoded)
    monkeypatch.setattr(module.json, "loads", decoded)
    run, first = _records()
    run = replace(run, metadata={"payload": "x" * 65536})
    first = replace(first, metadata=dict(run.metadata) if metadata_matches_run else {"payload": "y" * 65536})
    for name, constructor in (("text_inline", _TextInlineStepStore), ("json_inline", _InlineStepStore),
                              ("referenced", SqlStepStore)):
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / (name + '.db')}",
                                     json_serializer=bind_json, json_deserializer=read_json)
        context = create_sql_storage_context(engine)
        adapter = constructor(context, namespace="steps", tenant_id="one")
        await adapter.initialize(local=True)
        metrics = {"statements": Counter(), "bound_bytes": 0, "returned_bytes": 0, "commits": 0,
                   "sql_text_bytes": 0, "returned_rows": 0,
                   "application_metadata_encodes": 0, "application_metadata_encoded_bytes": 0,
                   "application_metadata_decodes": 0, "json_bind_encodes": 0,
                   "json_bind_encoded_bytes": 0, "json_result_decodes": 0}

        def capture(connection, cursor, statement, parameters, execution, many):
            metrics["statements"][statement.split()[0].upper()] += 1
            metrics["sql_text_bytes"] += len(statement.encode("utf-8"))
            metrics["bound_bytes"] += _bytes(parameters)

        def returned(connection, cursor, statement, parameters, execution, many):
            if cursor.description:
                metrics["returned_rows"] += len(cursor._rows)
                metrics["returned_bytes"] += _bytes(tuple(cursor._rows))

        def commit(connection):
            metrics["commits"] += 1

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        event.listen(engine.sync_engine, "after_cursor_execute", returned)
        event.listen(engine.sync_engine, "commit", commit)
        active = metrics
        start = time.process_time()
        expected = [replace(first, step_index=index, idempotency_key=str(index)) for index in range(32)]
        try:
            await adapter.commit_step(run=run, events=expected[:1])
            for boundary in expected[1:]:
                await adapter.append_event(boundary)
            assert metrics["statements"]["SELECT"] == 0
            await adapter.commit_step(run=run, events=(replace(expected[0], metadata={"retry": "ignored"}),))
            assert metrics["statements"]["SELECT"] == (2 if name == "referenced" else 1)
            await adapter.close()
            cold = constructor(context, namespace="steps", tenant_id="one")
            try:
                assert await cold.get_run(run_id=run.run_id) == run
                assert await cold.list_events(run_id=run.run_id) == expected
            finally:
                await cold.close()
            metrics["cpu_seconds"] = time.process_time() - start
        finally:
            active = None
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
            event.remove(engine.sync_engine, "after_cursor_execute", returned)
            event.remove(engine.sync_engine, "commit", commit)
            await adapter.close()
            await context.close()
        rows = await _rows(engine)
        metrics["stored_field_bytes"] = sum(_bytes(tuple(row.values())) for row in rows)
        samples[name] = metrics
        await engine.dispose()
    record_property("native_event_lifecycle_cost", json.dumps({
        "metadata_matches_native_run": metadata_matches_run, "samples": samples,
    }, sort_keys=True))
    text_baseline, baseline, candidate = samples["text_inline"], samples["json_inline"], samples["referenced"]
    assert baseline["statements"] == text_baseline["statements"] == {"BEGIN": 33, "INSERT": 35, "SELECT": 3}
    # The one duplicate replay verifies raw scope after its indexed conflict.
    assert candidate["statements"] == {"BEGIN": 33, "INSERT": 35, "SELECT": 4}
    assert candidate["commits"] == baseline["commits"] == text_baseline["commits"] == 33
    assert text_baseline["application_metadata_encodes"] == candidate["application_metadata_encodes"] == 35
    assert baseline["application_metadata_encodes"] == 2
    assert text_baseline["json_bind_encodes"] == 0
    assert baseline["json_bind_encodes"] == 33
    # Even a matching workload includes one differing replay payload; it still binds once.
    assert candidate["json_bind_encodes"] == (1 if metadata_matches_run else 33)
    assert text_baseline["application_metadata_decodes"] == candidate["application_metadata_decodes"] == 33
    assert baseline["application_metadata_decodes"] == 1
    assert baseline["json_result_decodes"] == 32
    assert text_baseline["json_result_decodes"] == candidate["json_result_decodes"] == 0
    assert (text_baseline["application_metadata_encoded_bytes"]
            == baseline["application_metadata_encoded_bytes"] + baseline["json_bind_encoded_bytes"])
    assert candidate["application_metadata_encoded_bytes"] == text_baseline["application_metadata_encoded_bytes"]
    assert candidate["bound_bytes"] < baseline["bound_bytes"] / (8 if metadata_matches_run else 1)
    assert 0 < candidate["returned_bytes"] <= baseline["returned_bytes"]
    assert candidate["stored_field_bytes"] < baseline["stored_field_bytes"] / (100 if metadata_matches_run else 1)
    # Single CPU samples are diagnostic; deterministic storage bounds determine pass/fail.
