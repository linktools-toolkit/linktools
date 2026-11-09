#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native registration ownership preserves Trace reads, fences and portability."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai_harness.step_persistence import RunRecord
from sqlalchemy import delete, event, inspect, update
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.persistence import Persistence, PersistenceDomain, SqlStepStore
from linktools.ai.persistence._native_run_reference import is_native_run_reference
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.persistence._snapshot import SnapshotLimits
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import _decode_step_envelope, _encode_step_envelope, decode_record
from linktools.ai.persistence.snapshots import RuntimeSnapshotOperations, SnapshotResources
from linktools.ai.storage import InMemoryObjectStore, read_object

from .test_incremental_projection_boundaries import _storage
from .test_step_projection import _execution


@asynccontextmanager
async def _source(tmp_path: Path, *, confirmed: bool = True, restart: bool = True, large: bool = True, default_metadata: bool = False) -> AsyncIterator[tuple]:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="native-run-facts", tenant_id="tenant")
    native = await state.create_step_store()
    store = state.run_store
    source = RunRecord(
        run_id="run", conversation_id="conversation", parent_run_id="parent", agent_name="agent",
        registration_id="registration",
        metadata=({"capability_scope": "parent", "history_id": "6ae1ec90-9238-43ec-a423-30db52344212",
                   "agent_run_seq": "1", "linktools.transcript_start": "0", "linktools.transcript_run_id": "run"}
                  if default_metadata else
                  {"shared": "shared-native-payload" * (200 if large else 1), "different": "native",
                   "native-only": "retained", "linktools.transcript_start": "0"}),
    )
    run = replace(OfficialStepReader.project_run(source),
        started_at=source.started_at + timedelta(microseconds=1),
        metadata=(OfficialStepReader.project_run(source).metadata if default_metadata else
                  {"shared": source.metadata["shared"], "different": "runtime", "runtime-only": "retained"}))
    try:
        await state.execution.executions.create_execution(replace(
            _execution(), revision=1, producer_generation=1, restart_checkpoints=restart,
        ))
        store.bind_step_store(native)
        store.bind_execution_producer("run", execution_id="execution", producer_generation=1,
                                     restart_checkpoints=restart)
        if confirmed:
            await native.register_run(source)
            store.confirm_step_run(source)
        await store.register_agent_run(run, execution_id="execution")
        yield state, native, source, run
    finally:
        await state.close()
        await native.close()


async def _flush(state: Persistence, *, step: int = 0) -> None:
    await state.run_store.append_event(StepEvent("run", "MODEL_REQUEST_CANCELLED", step,
        metadata={"linktools.ai.model_request_seq": str(step + 1)}, idempotency_key=f"cancel:{step}"))
    await state.run_store.flush_execution_projection("run", execution_id="execution", producer_generation=1)


async def _raw_run(archive):
    records = await archive.state_store.read(lambda tx: tx.scan_records())
    return next(record for record in records if record.kind == "agent_run")


@pytest.mark.asyncio
async def test_confirmed_native_run_owns_lineage_and_common_metadata(tmp_path: Path) -> None:
    async with _source(tmp_path) as (state, native, source, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        raw = await _raw_run(archive)
        assert is_native_run_reference(raw.data)
        encoded = canonical_json_bytes(dict(raw.data))
        assert source.metadata["shared"].encode() not in encoded
        assert b'agent_conversation_id' not in encoded and b'parent_agent_run_id' not in encoded
        assert len(encoded) < len(canonical_json_bytes(_encode_step_envelope(run)))
        assert await archive.get_agent_run(agent_run_id="run") == run
        assert await archive.list_agent_runs(parent_agent_run_id="parent") == [run]
        events = await archive.list_events(agent_run_id="run")
        assert [(item.event_type, item.metadata) for item in events] == [
            ("MODEL_REQUEST_CANCELLED", {"linktools.ai.model_request_seq": "1"})]
        assert await native.get_run(run_id="run") == source
        assert await native.list_events(run_id="run") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmed,restart", ((False, True), (True, False)))
async def test_unconfirmed_or_independent_trace_run_keeps_its_own_fact(
    tmp_path: Path, confirmed: bool, restart: bool,
) -> None:
    async with _source(tmp_path, confirmed=confirmed, restart=restart) as (state, _, _, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        raw = await _raw_run(archive)
        assert not is_native_run_reference(raw.data)
        assert _decode_step_envelope(raw.data) == run


@pytest.mark.asyncio
async def test_later_confirmation_removes_inline_duplicate_without_reordering_events(tmp_path: Path) -> None:
    async with _source(tmp_path, confirmed=False) as (state, native, source, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        before = await _raw_run(archive)
        assert not is_native_run_reference(before.data)
        await native.register_run(source)
        state.run_store.confirm_step_run(source)
        await _flush(state, step=1)
        after = await _raw_run(archive)
        assert is_native_run_reference(after.data)
        assert after.key_digest == before.key_digest
        assert await archive.get_agent_run(agent_run_id="run") == run
        assert [item.step_index for item in await archive.list_events(agent_run_id="run")] == [0, 1]
        assert (await archive.capture_history(("run",)))["run"].event_count == 2


@pytest.mark.asyncio
async def test_cold_history_reads_share_one_native_join_and_keep_raw_cache(tmp_path: Path) -> None:
    async with _source(tmp_path) as (state, _, _, run):
        await _flush(state)
        cold = await state.create_agent_run_store()
        archive = cold.read_store(PersistenceDomain.TRACE)
        statements = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        engine = archive.state_store.context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            assert await archive.get_agent_run(agent_run_id="run") == run
            assert len(statements) == 1
            statements.clear()
            assert (await archive.capture_history(("run",)))["run"].run == run
            assert len(statements) == 2
            statements.clear()
            assert await archive.read_model_interactions_with_run(agent_run_id="run") == (run, [])
            assert len(statements) == 2
            statements.clear()
            key = archive._agent_run_key("run")

            async def read(tx):
                resolved = await archive.read_run_records(tx, (key,))
                assert _decode_step_envelope(resolved[key].data) == run
                assert is_native_run_reference((await tx.get_record(key)).data)

            await archive.state_store.read(read)
            assert len(statements) == 1
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
            await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_native_run_reference_fails_closed_when_source_is_missing_or_changed(tmp_path: Path, missing: bool) -> None:
    async with _source(tmp_path) as (state, native, _, _):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        table = build_step_sql_metadata().tables["ai_step_runs"]
        async with native.storage_context.engine.begin() as connection:
            statement = delete(table) if missing else update(table).values(metadata='{"changed":"source"}')
            await connection.execute(statement.where(table.c.store_digest == native.store_digest))
        with pytest.raises(AIError) as caught:
            await archive.get_agent_run(agent_run_id="run")
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_source_confirmation_uses_original_registration_after_changed_replay(tmp_path: Path) -> None:
    async with _source(tmp_path, confirmed=False) as (state, native, source, run):
        await native.register_run(source)
        changed = replace(source, metadata={**source.metadata, "shared": "different"})
        await native.commit_step(run=changed)
        state.run_store.confirm_step_run(changed)
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        assert is_native_run_reference((await _raw_run(archive)).data)
        assert await state.run_store.checkpoint_persistence.reader.get_run(run_id="run") == source
        assert await archive.get_agent_run(agent_run_id="run") == run


@pytest.mark.asyncio
async def test_portable_trace_subset_projects_native_source_without_changing_live_record(tmp_path: Path) -> None:
    async with _source(tmp_path) as (state, _, _, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        objects = InMemoryObjectStore()
        resources = SnapshotResources(
            namespace="native-run-facts", tenant_id="tenant", durable_domains=frozenset({PersistenceDomain.TRACE}),
            stores={PersistenceDomain.TRACE: archive.state_store},
            object_stores={domain: objects for domain in PersistenceDomain}, read_only=True,
        )
        target = InMemoryObjectStore()
        ref = await RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
            object_store=target, limits=SnapshotLimits(max_entries=1000, max_bytes=1000000))
        manifest = json.loads(await read_object(target, ref.key, expected_digest=ref.digest, expected_size=ref.size))
        records = [decode_record(value) for value in manifest["domains"]["trace"]["records"]]
        exported = next(record for record in records if record.kind == "agent_run")
        assert _decode_step_envelope(exported.data) == run
        assert is_native_run_reference((await _raw_run(archive)).data)


@pytest.mark.asyncio
async def test_selected_trace_schema_provisions_its_native_run_read_dependency(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'selected.db'}")
    try:
        await provision_runtime_database(engine, domains=(PersistenceDomain.EXECUTION, PersistenceDomain.TRACE))
        async with engine.connect() as connection:
            names = await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        assert {"ai_step_runs", "ai_step_events"}.issubset(names)
        assert "ai_step_snapshots" not in names
        from linktools.ai.persistence._sql import SqlStateStore
        from linktools.ai.persistence.runs import StateStepArchive

        store = SqlStateStore(engine, runtime_domain=PersistenceDomain.TRACE)
        archive = StateStepArchive(store, object_store=None, namespace="selected", tenant_id="one",
                                   runtime_domain=PersistenceDomain.TRACE)
        archive.enable_native_run_reads()
        await store.initialize()
        try:
            assert await archive.list_events(agent_run_id="absent") == []
        finally:
            await archive.close()
            await store.close()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_small_run_keeps_smaller_inline_representation(tmp_path: Path) -> None:
    async with _source(tmp_path, large=False) as (state, _, _, run):
        await _flush(state)
        raw = await _raw_run(state.run_store.read_store(PersistenceDomain.TRACE))
        assert not is_native_run_reference(raw.data)
        assert _decode_step_envelope(raw.data) == run


@pytest.mark.asyncio
async def test_uncertain_native_source_invalidates_reference_admission(tmp_path: Path) -> None:
    async with _source(tmp_path) as (state, _, _, run):
        state.run_store.invalidate_step_run("run")
        await _flush(state)
        raw = await _raw_run(state.run_store.read_store(PersistenceDomain.TRACE))
        assert not is_native_run_reference(raw.data)
        assert _decode_step_envelope(raw.data) == run


@pytest.mark.asyncio
async def test_reference_promotion_rolls_back_with_original_projection_transaction(tmp_path: Path) -> None:
    async with _source(tmp_path, confirmed=False) as (state, native, source, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        before = await _raw_run(archive)
        await native.register_run(source)
        state.run_store.confirm_step_run(source)

        def fail(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.startswith("INSERT INTO ai_state_facts "):
                raise RuntimeError("publication failed")

        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", fail)
        try:
            with pytest.raises(RuntimeError, match="publication failed"):
                await archive.sync_projection(run, events=(StepEvent("run", "AGENT_RUN_INTERRUPTED", 1),),
                                              checkpoints=(), execution_id="execution")
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", fail)
        assert await _raw_run(archive) == before
        assert len(await archive.list_events(agent_run_id="run")) == 1


@pytest.mark.asyncio
async def test_native_reference_cannot_redirect_trace_to_another_tenant(tmp_path: Path) -> None:
    async with _source(tmp_path) as (state, native, source, _):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        foreign = SqlStepStore(native.storage_context, namespace="native-run-facts", tenant_id="other")
        try:
            await foreign.register_run(source)
            table = build_step_sql_metadata().tables["ai_step_runs"]
            async with native.storage_context.engine.begin() as connection:
                await connection.execute(delete(table).where(table.c.store_digest == native.store_digest))
            with pytest.raises(AIError) as caught:
                await archive.get_agent_run(agent_run_id="run")
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        finally:
            await foreign.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("shared", "different", "native-only"))
async def test_native_projection_checks_reused_fields_but_retains_runtime_overrides(tmp_path: Path, key: str) -> None:
    async with _source(tmp_path) as (state, native, source, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        table = build_step_sql_metadata().tables["ai_step_runs"]
        metadata = {**source.metadata, key: "changed"}
        async with native.storage_context.engine.begin() as connection:
            await connection.execute(update(table).where(table.c.store_digest == native.store_digest)
                                     .values(metadata=json.dumps(metadata)))
        if key == "shared":
            with pytest.raises(AIError) as caught:
                await archive.get_agent_run(agent_run_id="run")
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            assert await archive.get_agent_run(agent_run_id="run") == run


@pytest.mark.asyncio
async def test_default_runtime_metadata_uses_the_smaller_run_representation(tmp_path: Path) -> None:
    async with _source(tmp_path, default_metadata=True) as (state, _, _, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        raw = await _raw_run(archive)
        assert len(canonical_json_bytes(dict(raw.data))) <= len(canonical_json_bytes(_encode_step_envelope(run)))
        assert await archive.get_agent_run(agent_run_id="run") == run


@pytest.mark.asyncio
async def test_execution_only_provisioning_supports_default_nonrestart_trace_reads(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.capability import ExecutionCapability, TraceCapability
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'execution-only.db'}")
    await provision_runtime_database(engine, domains=(PersistenceDomain.EXECUTION,))
    features = RuntimeFeatures("selected-schema")
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(TraceCapability())
    features.agent("default", model="default", allow_tools=())

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("retained")])

    try:
        async with Persistence.open("selected-schema", storage=PersistenceStorage.sql(engine)) as state:
            async with Runtime.attach(persistence=state,
                    features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                execution = await runtime.agents.get().run("question")
                assert execution.result.status is ExecutionStatus.SUCCEEDED
            archive = state.run_store.read_store(PersistenceDomain.TRACE)
            runs = await archive.list_agent_runs()
            assert runs
            for run in runs:
                assert await archive.get_agent_run(agent_run_id=run.agent_run_id) == run
                assert (await archive.capture_history((run.agent_run_id,)))[run.agent_run_id].run == run
    finally:
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("referenced", (False, True))
async def test_terminal_history_prefetch_reuses_present_and_missing_run_reads(tmp_path: Path, referenced: bool) -> None:
    async with _source(tmp_path, confirmed=referenced) as (state, _, _, run):
        await _flush(state)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        statements = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        async def read(_group):
            await archive.execution_history_heads(("run", "missing"))
            count = len(statements)
            assert await archive.get_agent_run(agent_run_id="run") == run
            assert await archive.get_agent_run(agent_run_id="missing") is None
            assert len(statements) == count

        engine = archive.state_store.context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await archive.state_store.storage_group.read_many((archive.state_store,), read)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("inline", "insert", "promote", "existing"))
async def test_checkpoint_projection_reuses_its_run_read_for_terminal_head(tmp_path: Path, mode: str) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from linktools.ai.persistence._step_contracts import AgentRunCheckpoint

    async with _source(tmp_path, confirmed=mode in {"insert", "existing"}) as (state, native, source, run):
        if mode in {"promote", "existing"}:
            await _flush(state)
        if mode == "promote":
            await native.register_run(source)
            state.run_store.confirm_step_run(source)
        if mode == "existing":
            state.run_store.invalidate_step_run("run")
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        prepared = await archive.prepare_checkpoints(run, (AgentRunCheckpoint(
            "run", 2, [ModelRequest(parts=[UserPromptPart("retained")], run_id="run")],
            transcript_message_count_before=0,
        ),))
        statements = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        async def commit(_group):
            await archive.sync_prepared_projection(
                run, events=(StepEvent("run", "AGENT_RUN_INTERRUPTED", 2),),
                checkpoints=prepared.checkpoints, execution_id="execution", producer_generation=1,
            )
            statements.clear()
            head = await archive.execution_history_head_record("run")
            assert head.checkpoint_count == 1
            assert head.transcript_message_count == 1
            assert await archive.get_agent_run(agent_run_id="run") == run
            # The untouched interaction sequence is the only remaining read;
            # run/head/projection records must all reuse the write transaction.
            assert len(statements) == 1 and "ai_state_sequences" in statements[0]

        engine = archive.state_store.context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            await archive.state_store.storage_group.mutate(archive.transaction_stores, commit)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.asyncio
async def test_prepared_run_reference_survives_source_cache_eviction_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _source(tmp_path) as (state, native, _, run):
        def evict(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.startswith("INSERT INTO ai_state_records "):
                monkeypatch.setattr(native, "confirmed_run", lambda _run_id: None)

        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", evict)
        try:
            await _flush(state)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", evict)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        assert is_native_run_reference((await _raw_run(archive)).data)
        assert await archive.get_agent_run(agent_run_id="run") == run


@pytest.mark.asyncio
@pytest.mark.parametrize("locked", (False, True))
async def test_joined_record_reader_rejects_writing_or_locking_select_plans(tmp_path: Path, locked: bool) -> None:
    from sqlalchemy import select
    from linktools.ai.persistence._schema import build_runtime_sql_metadata
    from linktools.ai.persistence.state import StateTransactionNestingError
    from .test_state_group_record_batch import _database

    async with _database(tmp_path / 'readonly.db') as (_, state, _):
        records = build_runtime_sql_metadata(frozenset({PersistenceDomain.TRACE})).tables["ai_state_records"]
        query = select(records)
        query = query.with_for_update() if locked else query.add_cte(delete(records).cte())

        async def read(tx):
            with pytest.raises(StateTransactionNestingError, match="read-only SQL plan"):
                await tx.read_joined_records(state.context, query)

        await state.read(read)


@pytest.mark.parametrize("dialect_name", ("sqlite", "postgresql", "mysql"))
def test_native_join_preserves_full_record_key_bind_budget(dialect_name: str) -> None:
    from sqlalchemy.dialects import mysql, postgresql, sqlite
    from linktools.ai.persistence._native_run_reference import NativeRunReferences

    dialect = {"sqlite": sqlite, "postgresql": postgresql, "mysql": mysql}[dialect_name].dialect()
    state = SimpleNamespace(context=SimpleNamespace(dialect=dialect), store_digest=b"a" * 32)
    reader = NativeRunReferences(state, namespace="native-run-facts", tenant_id="tenant")
    keys = tuple(index.to_bytes(32, "big") for index in range(998))
    compiled = reader.statement(keys=keys).compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
    parameters = compiled.params if compiled.positiontup is None else compiled.positiontup
    assert len(parameters) == 999
