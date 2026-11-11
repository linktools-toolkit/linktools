#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native direct identities preserve exact replay, scope and nullable occurrences."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent
from sqlalchemy import LargeBinary, String, delete, select
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateIndex, CreateTable

from linktools.ai.errors import AIError
from linktools.ai.persistence import NativeMessageRoot, SqlStepStore
from linktools.ai.persistence._native_event_reference import NativeEventReferences
from linktools.ai.persistence._native_run_reference import NativeRunReferences
from linktools.ai.persistence._step_backup import NativeStepBackup
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import decode_domain, encode_domain, encode_envelope
from linktools.ai.persistence.state import FactQuery, StoredFact
from linktools.ai.steps import (
    NativeStepSource, decode_native_store_key, encode_native_store_key, native_store_key,
    rekey_native_step_sources, validate_native_store_key, validate_native_replay_key,
)


def test_native_scope_and_replay_boundaries_preserve_original_values() -> None:
    scope = native_store_key("界" * 85 + "x", "🙂" * 32)
    assert len(scope) == 402
    assert validate_native_store_key(scope) == scope
    assert decode_native_store_key(encode_native_store_key(scope)) == scope
    assert native_store_key("a:b", "c") != native_store_key("a", "b:c")
    for value in (None, "", "k", "k ", "é", "e\u0301", "🙂" * 128):
        assert validate_native_replay_key(value) == value
    for namespace, tenant in (("x" * 257, "t"), ("n", "🙂" * 33)):
        with pytest.raises(AIError):
            native_store_key(namespace, tenant)
    for value in ("🙂" * 128 + "x", "\ud800", "\0", 1):
        with pytest.raises(ValueError):
            validate_native_replay_key(value)
    for value in (scope + b"\0", b"", b"x" * 403, b"\0\x01\xff"):
        with pytest.raises(ValueError):
            validate_native_store_key(value)
    with pytest.raises(ValueError):
        decode_native_store_key(encode_native_store_key(scope) + "=")


def test_native_wire_scope_relocation_and_repr_hide_raw_identity() -> None:
    scope = native_store_key("private-namespace", "private-tenant")
    source = NativeStepSource("a" * 32, scope)
    root = NativeMessageRoot("a" * 32, scope)
    assert NativeMessageRoot.from_json(root.to_json()) == root
    wire = encode_domain(source)
    assert decode_domain(wire, NativeStepSource) == source
    moved = rekey_native_step_sources(wire, {source.group_id: "b" * 32})
    assert decode_domain(moved, NativeStepSource) == replace(source, group_id="b" * 32)
    reference = encode_envelope({"type": "native_event_reference", "payload": {
        "run": "run", "replay": "private-replay", "index": 0,
    }})
    fact = StoredFact(b"stream", 1, b"owner", "step_event", None, None, reference)
    for value in (source, root, fact):
        assert "private-" not in repr(value)
    assert NativeEventReferences.decode_reference(reference)["replay"] == "private-replay"
    for invalid in ({"run": "run", "replay": None, "index": 0},
                    {"run": "run", "replay": "x" * 513, "index": 0},
                    {"run": "run", "replay": "x\0", "index": 0},
                    {"run": "r" * 129, "replay": "key", "index": 0}):
        with pytest.raises(AIError):
            NativeEventReferences.decode_reference(encode_envelope({"type": "native_event_reference", "payload": invalid}))


@pytest.mark.parametrize("name,dialect", [("mysql", mysql.dialect()), ("postgresql", postgresql.dialect()), ("sqlite", sqlite.dialect())])
def test_native_schema_and_joins_preserve_binary_keys_within_index_and_bind_budgets(name, dialect) -> None:
    metadata = build_step_sql_metadata()
    for table_name in ("events", "snapshots"):
        table = metadata.tables["ai_step_" + table_name]
        assert isinstance(table.c.store_key.type, LargeBinary) and table.c.store_key.type.length == 402
        assert isinstance(table.c.idempotency_key.type, LargeBinary) and table.c.idempotency_key.type.length == 512
        assert table.c.idempotency_key.nullable
        keys = [index for index in table.indexes if index.unique]
        assert {tuple(index.columns.keys()) for index in keys} == {("store_key", "run_id", "idempotency_key")}
        for index in table.indexes:
            assert len(index.columns) <= 3
            str(CreateIndex(index).compile(dialect=dialect))
        size = sum(column.type.length * (4 if isinstance(column.type, String) else 1) for column in keys[0].columns)
        assert size == 1426 < 3072
        str(CreateTable(table).compile(dialect=dialect))
    owner = SimpleNamespace(context=SimpleNamespace(dialect=dialect), store_key=b"state")
    query = NativeEventReferences(owner, namespace="界" * 85 + "x", tenant_id="🙂" * 32).statement(query=FactQuery(b"events"))
    compiled = query.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
    sql = str(compiled).upper()
    assert "CONVERT_TO(" in sql if name == "postgresql" else " AS BINARY)" in sql if name == "mysql" else " AS BLOB)" in sql
    assert "\0" not in str(compiled)
    runs = NativeRunReferences(owner, namespace="scope", tenant_id="tenant")
    compiled = runs.statement(keys=tuple(index.to_bytes(32, "big") for index in range(998))).compile(
        dialect=dialect, compile_kwargs={"render_postcompile": True})
    assert len(compiled.params if compiled.positiontup is None else compiled.positiontup) == 999


class _Reader:
    def __init__(self, session):
        self.session = session

    async def read_sql_rows(self, context, statement):
        return (await self.session.execute(statement)).mappings().all()


@pytest.mark.asyncio
async def test_exact_keyed_replays_and_null_occurrences_survive_reopen_and_backup(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}")
    store = SqlStepStore(engine, namespace="scope", tenant_id="tenant")
    other = SqlStepStore(engine, namespace="scope", tenant_id="other")
    await store.initialize(local=True)
    keys = ("", "k", "k ", "é", "e\u0301", "🙂" * 128, None, None)
    try:
        await store.register_run(RunRecord(run_id="run"))
        for index, key in enumerate(keys):
            event = StepEvent(run_id="run", kind="model_request_started", step_index=index, idempotency_key=key)
            snapshot = ContinuableSnapshot(run_id="run", step_index=index, messages=[], idempotency_key=key)
            await store.append_event(event)
            await store.save_snapshot(snapshot)
            if key is not None:
                await store.append_event(event)
                await store.save_snapshot(snapshot)
        assert await other.list_events(run_id="run") == []
        assert await other.list_snapshots(run_id="run", include_interrupted=True) == []
        await store.close()
        store = SqlStepStore(engine, namespace="scope", tenant_id="tenant")
        assert tuple(item.idempotency_key for item in await store.list_events(run_id="run")) == keys
        assert tuple(item.idempotency_key for item in await store.list_snapshots(run_id="run", include_interrupted=True)) == keys
        async with store.storage_context.sessions() as session, session.begin():
            backup = await store.capture_backup(_Reader(session))
        wire = backup.to_json()
        assert tuple(row["idempotency_key"] for row in wire["events"]) == keys
        backup = NativeStepBackup.from_json(wire)
        async with store.storage_context.sessions() as session, session.begin():
            for name in ("runs", "events", "snapshots"):
                await session.execute(delete(store._table(name)))
            await store.restore_backup(session, backup)
        assert tuple(item.idempotency_key for item in await store.list_events(run_id="run")) == keys
        assert tuple(item.idempotency_key for item in await store.list_snapshots(run_id="run", include_interrupted=True)) == keys
        for invalid in ("x" * 513, "x\0"):
            with pytest.raises(ValueError):
                await store.append_event(StepEvent(run_id="run", kind="run_started", step_index=0, idempotency_key=invalid))
            with pytest.raises(ValueError):
                await store.save_snapshot(ContinuableSnapshot(run_id="run", step_index=0, messages=[], idempotency_key=invalid))
        events = store._table("events")
        for invalid in (b"x" * 513, b"x\0", b"\xff"):
            async with engine.begin() as connection:
                await connection.execute(events.update().where(events.c.step_index == 1).values(idempotency_key=invalid))
            with pytest.raises(ValueError):
                await store.list_events(run_id="run")
    finally:
        await store.close()
        await other.close()
        await engine.dispose()
