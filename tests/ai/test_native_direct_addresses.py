#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native direct identities preserve exact replay, scope and unkeyed occurrences."""

import base64
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
    decode_native_replay_key, encode_native_replay_key, native_store_locator, rekey_native_step_sources, validate_native_store_key, validate_native_replay_key,
)


def test_native_scope_and_replay_boundaries_preserve_original_values() -> None:
    scope = native_store_key("界" * 42 + "xx", "🙂" * 32)
    assert len(scope) == 274
    assert validate_native_store_key(scope) == scope
    assert decode_native_store_key(encode_native_store_key(scope)) == scope
    assert native_store_key("a:b", "c") != native_store_key("a", "b:c")
    for value in (None, "", "k", "k ", "é", "e\u0301", "🙂" * 32):
        assert validate_native_replay_key(value) == value
    for namespace, tenant in (("x" * 129, "t"), ("n", "🙂" * 33)):
        with pytest.raises(AIError):
            native_store_key(namespace, tenant)
    for value in ("🙂" * 32 + "x", "\ud800", "\0", 1):
        with pytest.raises(ValueError):
            validate_native_replay_key(value)
    for value in (scope + b"\0", b"", b"x" * 275, b"\0\x01\xff"):
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
                    {"run": "run", "replay": "x" * 129, "index": 0},
                    {"run": "run", "replay": "x\0", "index": 0},
                    {"run": "r" * 129, "replay": "key", "index": 0}):
        with pytest.raises(AIError):
            NativeEventReferences.decode_reference(encode_envelope({"type": "native_event_reference", "payload": invalid}))


@pytest.mark.parametrize("name,dialect", [("mysql", mysql.dialect()), ("postgresql", postgresql.dialect()), ("sqlite", sqlite.dialect())])
def test_native_schema_and_joins_preserve_binary_keys_within_index_and_bind_budgets(name, dialect) -> None:
    metadata = build_step_sql_metadata()
    for table_name in ("events", "snapshots"):
        table = metadata.tables["ai_step_" + table_name]
        assert isinstance(table.c.store_key.type, LargeBinary) and table.c.store_key.type.length == 274
        assert isinstance(table.c.idempotency_key.type, LargeBinary) and table.c.idempotency_key.type.length == 128
        assert not table.c.idempotency_key.nullable
        keys = [index for index in table.indexes if index.unique]
        assert {tuple(index.columns.keys()) for index in keys} == {("store_locator", "run_id", "idempotency_key")}
        for index in table.indexes:
            assert len(index.columns) <= 3
            str(CreateIndex(index).compile(dialect=dialect))
        size = sum(column.type.length * (4 if isinstance(column.type, String) else 1) for column in keys[0].columns)
        assert size == 896 < 3072
        str(CreateTable(table).compile(dialect=dialect))
    owner = SimpleNamespace(context=SimpleNamespace(dialect=dialect), store_key=b"state")
    query = NativeEventReferences(owner, namespace="界" * 42 + "xx", tenant_id="🙂" * 32).statement(query=FactQuery(b"events"))
    compiled = query.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
    sql = str(compiled).upper()
    assert "CONVERT_TO(" in sql if name == "postgresql" else " AS BINARY)" in sql if name == "mysql" else " AS BLOB)" in sql
    assert "\0" not in str(compiled)
    runs = NativeRunReferences(owner, namespace="scope", tenant_id="tenant")
    compiled = runs.statement(keys=tuple(index.to_bytes(32, "big") for index in range(499))).compile(
        dialect=dialect, compile_kwargs={"render_postcompile": True})
    assert len(compiled.params if compiled.positiontup is None else compiled.positiontup) == 999


class _Reader:
    def __init__(self, session):
        self.session = session

    async def read_sql_rows(self, context, statement):
        return (await self.session.execute(statement)).mappings().all()


@pytest.mark.asyncio
async def test_exact_keyed_replays_and_unkeyed_occurrences_survive_reopen_and_backup(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}")
    store = SqlStepStore(engine, namespace="scope", tenant_id="tenant")
    other = SqlStepStore(engine, namespace="scope", tenant_id="other")
    await store.initialize(local=True)
    keys = ("", "k", "k ", "é", "e\u0301", "🙂" * 32, None, None)
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
        assert tuple(decode_native_replay_key(base64.b64decode(row["idempotency_key"]))
                     for row in wire["events"]) == keys
        assert NativeStepBackup.from_json(wire).events == backup.events
        backup = NativeStepBackup.from_json(wire)
        async with store.storage_context.sessions() as session, session.begin():
            for name in ("runs", "events", "snapshots"):
                await session.execute(delete(store._table(name)))
            await store.restore_backup(session, backup)
        assert tuple(item.idempotency_key for item in await store.list_events(run_id="run")) == keys
        assert tuple(item.idempotency_key for item in await store.list_snapshots(run_id="run", include_interrupted=True)) == keys
        for invalid in ("x" * 129, "x\0"):
            with pytest.raises(ValueError):
                await store.append_event(StepEvent(run_id="run", kind="run_started", step_index=0, idempotency_key=invalid))
            with pytest.raises(ValueError):
                await store.save_snapshot(ContinuableSnapshot(run_id="run", step_index=0, messages=[], idempotency_key=invalid))
        events = store._table("events")
        for invalid in (b"x" * 129, b"x\0", b"\xff"):
            async with engine.begin() as connection:
                await connection.execute(events.update().where(events.c.step_index == 1).values(idempotency_key=invalid))
            with pytest.raises(ValueError):
                await store.list_events(run_id="run")
    finally:
        await store.close()
        await other.close()
        await engine.dispose()


def test_native_occurrence_wire_rejects_null_and_malformed_physical_keys() -> None:
    first, second = encode_native_replay_key(None), encode_native_replay_key(None)
    assert len(first) == 17 and first.startswith(b"\x00") and first != second
    assert decode_native_replay_key(first) is None
    assert encode_native_replay_key("") == b""
    assert decode_native_replay_key(b"") == ""
    for value in (None, b"\x00", b"\x00" + b"x" * 15, b"\x00" + b"x" * 17, b"x\x00"):
        with pytest.raises(ValueError):
            decode_native_replay_key(value)


@pytest.mark.asyncio
async def test_prepared_occurrences_keep_physical_identity_across_transaction_retry(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'prepared.db'}")
    store = SqlStepStore(engine, namespace="scope", tenant_id="tenant")
    await store.initialize(local=True)
    event = StepEvent(run_id="run", kind="model_request_started", step_index=0)
    snapshot = ContinuableSnapshot(run_id="run", step_index=0, messages=[])
    try:
        prepared = await store.prepare_step(events=(event,), snapshot=snapshot)
        async with store.storage_context.sessions() as session:
            async with session.begin():
                await store._write_prepared_step(session, prepared)
                first = (await session.execute(select(store._table("events").c.idempotency_key))).scalar_one()
                await session.rollback()
        async with store.storage_context.sessions() as session, session.begin():
            await store._write_prepared_step(session, prepared)
            assert (await session.execute(select(store._table("events").c.idempotency_key))).scalar_one() == first
            assert (await session.execute(select(store._table("snapshots").c.idempotency_key))).scalar_one() == prepared.snapshot_values["idempotency_key"]
        await store.append_event(event)
        assert [item.idempotency_key for item in await store.list_events(run_id="run")] == [None, None]
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("runs", "events", "snapshots", "messages", "sequences"))
async def test_native_locator_collision_never_confirms_or_overwrites_another_scope(tmp_path, kind) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'collision.db'}")
    source = SqlStepStore(engine, namespace="source", tenant_id="tenant")
    other = SqlStepStore(engine, namespace="other", tenant_id="tenant")
    await source.initialize(local=True)
    event = StepEvent(run_id="run", kind="model_request_started", step_index=0, idempotency_key="key")
    messages = [ModelRequest(parts=[UserPromptPart("same content")])]
    snapshot = ContinuableSnapshot(run_id="run", step_index=0, messages=messages, idempotency_key="key")
    try:
        if kind == "runs":
            await source.register_run(RunRecord(run_id="run"))
        elif kind == "events":
            await source.append_event(event)
        else:
            await source.save_snapshot(snapshot)
        table = source._table(kind)
        async with engine.begin() as connection:
            await connection.execute(table.update().values(store_locator=native_store_locator(other.store_key)))
            before = (await connection.execute(select(table))).mappings().all()
        operation = (lambda: other.register_run(RunRecord(run_id="run"))) if kind == "runs" else (
            (lambda: other.append_event(event)) if kind == "events" else lambda: other.save_snapshot(snapshot))
        with pytest.raises((AIError, ValueError)):
            await operation()
        async with engine.connect() as connection:
            assert (await connection.execute(select(table))).mappings().all() == before
        assert await other.list_events(run_id="run") == []
        assert await other.list_snapshots(run_id="run") == []
    finally:
        await source.close()
        await other.close()
        await engine.dispose()


def test_native_event_child_keys_bound_complete_identity_dimensions() -> None:
    from pydantic_ai_harness.step_persistence import InMemoryStepStore
    from linktools.ai.capability.steps import AtomicStepPersistence

    def event_key(*, index=0, step=0, kind="tool_call_started", tool="🙂" * 32):
        capability = AtomicStepPersistence(store=InMemoryStepStore(), run_id="run")
        capability._atomic_event_sequence = index
        ctx = SimpleNamespace(run_id="run", run_step=step, conversation_id="conversation")
        return capability._atomic_prepare_event(ctx, kind=kind, tool_call_id=tool).idempotency_key

    baseline = event_key()
    assert len(baseline) == 36 and event_key() == baseline
    assert len({baseline, event_key(index=1), event_key(step=1),
                event_key(kind="tool_call_completed"), event_key(tool="other")}) == 5
    with pytest.raises(AIError):
        event_key(tool="🙂" * 32 + "x")


@pytest.mark.asyncio
async def test_event_conflict_readback_rejects_partial_identity_visibility(tmp_path, monkeypatch) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'visibility.db'}")
    store = SqlStepStore(engine, namespace="scope", tenant_id="tenant")
    await store.initialize(local=True)
    events = tuple(StepEvent(run_id="run", kind="model_request_started", step_index=index,
                             idempotency_key=str(index)) for index in range(2))

    async def conflict(*args, **kwargs):
        return None

    class Session:
        async def execute(self, statement):
            return SimpleNamespace(all=lambda: [SimpleNamespace(run_id="run", idempotency_key=b"0",
                                                                 store_key=store.store_key)])

    monkeypatch.setattr(store.storage_context.dialect, "insert_ignore_conflict_many", conflict)
    try:
        with pytest.raises(AIError):
            await store._write_events(Session(), events)
    finally:
        await store.close()
        await engine.dispose()


def test_native_tool_terminal_keys_bound_operation_step_fence_and_outcome() -> None:
    from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationDecision
    from linktools.ai.storage import InMemoryObjectStore

    bridge = RuntimeToolOperationBridge(None, InMemoryObjectStore(), namespace="scope", tenant_id="tenant",
        execution_id="execution", agent_run_id="run", binding_digest="a" * 64, owner="owner", background_tasks=set())
    operation = "🙂" * 32
    started = StepEvent(run_id="run", kind="tool_call_started", step_index=2, tool_call_id="call")
    bridge._tool_events[operation] = started
    decision = ToolOperationDecision(operation, "owner", 3, False)
    baseline = bridge._terminal_event(decision).idempotency_key
    assert len(baseline) == 36 and bridge._terminal_event(decision).idempotency_key == baseline
    changed = {bridge._terminal_event(replace(decision, fence=4)).idempotency_key,
               bridge._terminal_event(decision, failed=True).idempotency_key,
               bridge._unknown_event(operation, 3).idempotency_key}
    bridge._tool_events[operation] = replace(started, step_index=3)
    changed.add(bridge._terminal_event(decision).idempotency_key)
    other = "x" * 128
    bridge._tool_events[other] = started
    changed.add(bridge._terminal_event(replace(decision, operation_id=other)).idempotency_key)
    assert len({baseline, *changed}) == 6


@pytest.mark.asyncio
async def test_recovery_requests_bound_maximum_execution_identity_in_both_launch_paths() -> None:
    from unittest.mock import AsyncMock
    from linktools.ai.execution import ExecutionRecoveryDetail
    from linktools.ai.runtime._local import LocalExecutionBackend
    from linktools.ai.runtime._recovery_coordinator import _RecoveryCoordinator
    from .test_local_execution_recovery import _record

    keys = []
    for identity in ("🙂" * 32, "x" * 128):
        execution = replace(_record(), execution_id=identity, recovery=ExecutionRecoveryDetail(agent_run_id="run"))
        port = SimpleNamespace(tenant_id="tenant", validate_binding=AsyncMock(return_value=None),
            _reconcile_session_recovery=AsyncMock(return_value=True), _recovery_idempotency=AsyncMock(return_value=None),
            _ensure_recovery_idempotency=AsyncMock(), restore_user_input=AsyncMock(return_value="prompt"),
            _prepare_recovery_relaunch=lambda value: True, launch=AsyncMock())
        coordinator = _RecoveryCoordinator(port, None)
        assert await coordinator._relaunch_execution(execution, producer_generation=1,
            expected_revision=None, recovery_operation=None, resolved_binding=None)
        first = port.launch.call_args.args[0].idempotency_key
        decision = SimpleNamespace(execution=execution, binding=None)
        local = SimpleNamespace(_tenant_id="tenant", _recovery_coordinator=SimpleNamespace(
            reconcile_deferred=AsyncMock(return_value=decision)), restore_user_input=port.restore_user_input,
            _prepare_recovery_relaunch=lambda value: True, launch=AsyncMock())
        await LocalExecutionBackend._reconcile_deferred_owned(local, identity, tenant_id="tenant")
        second = local.launch.call_args.args[0].idempotency_key
        assert first == second and len(first) == 36
        keys.append(first)
    assert keys[0] != keys[1]
