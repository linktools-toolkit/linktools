#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native start ownership, cold Trace readers, and SQL budgets (CI only)."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepEvent as NativeEvent, StepPersistence
from sqlalchemy import delete, event, update
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import TraceCapability
from linktools.ai.capability.steps import AtomicStepPersistence, StepRunRegistration
from linktools.ai.core import CommitPolicy, ExecutionStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._native_event_reference import NativeEventReferences
from linktools.ai.persistence._snapshot import SnapshotLimits
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import _decode_step_envelope, _encode_step_envelope, decode_fact
from linktools.ai.persistence.snapshots import RuntimeSnapshotOperations, SnapshotResources
from linktools.ai.persistence.state import FactQuery
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryObjectStore, read_object

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_native_run_references import _source
from .test_step_message_cache import _SqlTraffic


class _Traffic(_SqlTraffic):
    def __init__(self) -> None:
        super().__init__()
        self.sql: list[str] = []

    def capture(self, connection, cursor, statement, parameters, context, many) -> None:
        self.sql.append(statement)
        super().capture(connection, cursor, statement, parameters, context, many)

    @property
    def selects(self) -> list[str]:
        return [statement for statement in self.sql if statement.lstrip().upper().startswith("SELECT")]

    def report(self, record_property, prefix: str) -> None:
        for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"):
            record_property(f"{prefix}_{verb.lower()}", self.statements[verb])
        record_property(f"{prefix}_sql_text_bytes", sum(len(sql.encode()) for sql in self.sql))
        record_property(f"{prefix}_bound_text_binary_bytes", self.bound_bytes)
        # DBAPI does not expose wire response bytes. DTO bytes below are explicitly
        # not claimed to measure driver packets, numeric binds, or disk allocation.


@asynccontextmanager
async def _runtime_start(tmp_path: Path, *, outcome: str = "success", backend: str = "owned", metadata_bytes: int = 0, trace_sync: bool = False,
                         extra_capabilities: tuple[AbstractCapability, ...] = (),
                         yield_while_attached: bool = False, observe_without_wait: bool = False) -> AsyncIterator[tuple]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native-start.db'}")
    await provision_database(engine)
    class CustomSqlStore(SqlStepStore):
        pass

    native_class = CustomSqlStore if backend == "subclass" else SqlStepStore
    native = (InMemoryStepStore() if backend == "split" else
              native_class(engine, namespace="native-start", tenant_id="default"))

    class CustomPersistence(StepPersistence):
        pass

    capability = CustomPersistence if backend == "custom" else StepPersistence
    features = RuntimeFeatures("native-start")
    features.capability(capability(store=native, capture_frontier=True,
        metadata={"large": "x" * metadata_bytes} if metadata_bytes else {}), scope="runtime")
    features.capability(TraceCapability(commit=CommitPolicy.sync()) if trace_sync else TraceCapability(), scope="runtime")
    if outcome == "shortcircuit":
        class ShortCircuit(AbstractCapability):
            async def wrap_model_request(self, ctx, *, request_context, handler):
                return ModelResponse(parts=[TextPart("short circuit")])

        features.capability(ShortCircuit(), id="short-circuit", scope="runtime")
    for capability in extra_capabilities:
        features.capability(capability, scope="runtime")
    if outcome == "deferred":
        from pydantic_ai.exceptions import ApprovalRequired
        from linktools.ai.capability import AgentContext

        async def approval(_context: AgentContext[None]) -> str:
            raise ApprovalRequired()

        features.tool(approval, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("approval",) if outcome == "deferred" else ())
    entered = asyncio.Event()

    async def model(_messages, _info):
        entered.set()
        if outcome == "failure":
            raise RuntimeError("early model failure")
        if outcome == "cancel":
            await asyncio.Event().wait()
        if outcome == "deferred":
            return ModelResponse(parts=[ToolCallPart("approval", {}, tool_call_id="approval-1")])
        return ModelResponse(parts=[TextPart("done")])

    traffic = _Traffic()
    try:
        async with Persistence.open("native-start", storage=PersistenceStorage.sql(engine)) as state:
            event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
            try:
                async with Runtime.attach(persistence=state,
                        features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                    execution = await runtime.agents.get().start("work", idempotency_key="work")
                    if outcome == "cancel":
                        await asyncio.wait_for(entered.wait(), 5)
                        await execution.cancel()
                    if observe_without_wait:
                        from ._runtime_test_helpers import _wait_for_committed
                        record = await _wait_for_committed(
                            lambda: state.execution.executions.get(execution.execution_id, tenant_id="default"),
                            lambda value: value is not None and value.status in {
                                ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
                                ExecutionStatus.WAITING_DEFERRED,
                            },
                        )
                        async def worker_finished() -> None:
                            while runtime._execution_service.runtime_backend().worker_installed(execution.execution_id):
                                await asyncio.sleep(0.01)
                        await asyncio.wait_for(worker_finished(), 5)
                        status = record.status
                    else:
                        status = (await execution.wait(timeout_seconds=5)).result.status
                    if outcome in {"success", "shortcircuit"}:
                        assert status is ExecutionStatus.SUCCEEDED
                    else:
                        assert status is not ExecutionStatus.SUCCEEDED
                    if yield_while_attached:
                        runs = await native.list_runs()
                        assert len(runs) == 1
                        yield state, native, runs[0], traffic
                        return
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
            runs = await native.list_runs()
            assert len(runs) == 1
            if backend == "owned":
                head = await state.execution.executions.get(execution.execution_id, tenant_id="default")
                seal = await state.execution.executions.get_history_seal(execution.execution_id, tenant_id="default")
                assert head is not None and seal is not None
                assert head.history_seal_digest == seal.seal_digest
                sealed_run, = [item for item in seal.run_heads if item.agent_run_id == runs[0].run_id]
                trace_events = await state.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id=runs[0].run_id)
                assert sealed_run.event_count == len(trace_events)
            yield state, native, runs[0], traffic
    finally:
        if isinstance(native, SqlStepStore):
            await native.close()
        await engine.dispose()


async def _raw_events(archive, run_id: str):
    return await archive.state_store.read(lambda tx: tx.list_facts(FactQuery(archive._stream(run_id, "event"))))


@pytest.mark.asyncio
@pytest.mark.parametrize("native_first", (False, True))
async def test_shared_start_preparation_has_one_timestamp_and_identity(native_first: bool) -> None:
    registration = StepRunRegistration(run_id="run", parent_run_id="parent", agent_name="agent",
        metadata={"custom": "value"}, share_started=True)
    seed = await AtomicStepPersistence()._atomic_registration_seed()
    tokens = []

    async def token() -> tuple[str, datetime]:
        tokens.append(seed[0])
        return seed

    registration.bind_registration(token)
    if native_first:
        registration.prepare(conversation_id="conversation", registration_id=seed[0], started_at=seed[1])
    run, source = await registration.prepare_started(conversation_id="conversation", step_index=0)
    source.metadata["custom"] = "caller mutation"
    repeated_run, repeated = await registration.prepare_started(conversation_id="conversation", step_index=0)
    assert repeated_run == run
    assert repeated.timestamp == source.timestamp
    assert repeated.idempotency_key == source.idempotency_key
    assert repeated.metadata == {"custom": "value"}
    assert tokens == [seed[0]]
    projected = NativeEventReferences.project(repeated, event_index=0)
    assert projected.timestamp == repeated.timestamp
    assert projected.agent_run_id == repeated.run_id
    with pytest.raises(ValueError, match="boundary changed"):
        await registration.prepare_started(conversation_id="conversation", step_index=1)

    # A cold consumer derives identical complete bodies from the original seed.
    cold = StepRunRegistration(run_id="run", parent_run_id="parent", agent_name="agent",
        metadata={"custom": "value"}, share_started=True)
    cold.bind_registration(token)
    cold_run, cold_source = await cold.prepare_started(conversation_id="conversation", step_index=0)
    assert cold_run == run
    assert cold_source == repeated
    assert NativeEventReferences.project(cold_source, event_index=0) == projected
    assert tokens == [seed[0], seed[0]]
    assert cold_run.started_at == cold_source.timestamp == seed[1]
    cold_source.metadata["custom"] = "another caller mutation"
    assert cold.prepared_started() == (run, repeated)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancel", "shortcircuit"))
async def test_runtime_start_remains_first_and_single_after_terminal_flush(
    tmp_path: Path, outcome: str, record_property,
) -> None:
    async with _runtime_start(tmp_path, outcome=outcome) as (state, native, run, traffic):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        events = await archive.list_events(agent_run_id=run.run_id)
        starts = [item for item in await native.list_events(run_id=run.run_id) if item.kind == "run_started"]
        assert len(starts) == 1
        assert events[0] == NativeEventReferences.project(starts[0], event_index=0)
        assert sum(item.event_type == "AGENT_RUN_STARTED" for item in events) == 1
        assert [item.event_index for item in events] == list(range(len(events)))
        if outcome != "shortcircuit":
            assert next(index for index, item in enumerate(events) if item.event_type == "MODEL_REQUEST_STARTED") > 0
        raw = await _raw_events(archive, run.run_id)
        assert [item.sequence for item in raw] == list(range(1, len(events) + 1))
        assert raw[0].data["value"]["type"] == "native_event_reference"
        assert raw[0].data["value"]["payload"] == {
            "key": native.event_key_digest(starts[0]), "index": 0,
        }
        capture = (await archive.capture_history((run.run_id,)))[run.run_id]
        assert capture.event_count == len(events)
        traffic.report(record_property, outcome)
        record_property("start_sidecar_json_bytes", len(canonical_json_bytes(dict(raw[0].data))))
        record_property("start_inline_json_bytes", len(canonical_json_bytes(_encode_step_envelope(events[0]))))
        record_property("trace_fact_json_bytes", sum(len(canonical_json_bytes(dict(item.data))) for item in raw))
        record_property("trace_returned_dto_json_bytes", sum(len(canonical_json_bytes(_encode_step_envelope(item))) for item in events))
        sidecar_bytes = len(canonical_json_bytes(dict(raw[0].data)))
        old_inline_bytes = len(canonical_json_bytes(_encode_step_envelope(events[0])))
        record_property("native_added_proof_logical_ascii_bytes", 128)
        record_property("start_signed_logical_byte_delta", sidecar_bytes + 128 - old_inline_bytes)


@pytest.mark.asyncio
async def test_cold_start_readers_join_source_in_the_existing_select(tmp_path: Path, record_property) -> None:
    async with _runtime_start(tmp_path) as (state, native, run, _):
        cold = await state.create_agent_run_store()
        archive = cold.read_store(PersistenceDomain.TRACE)
        expected = await state.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id=run.run_id)
        engine = native.storage_context.engine

        async def measured(name, operation):
            traffic = _Traffic()
            event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
            try:
                result = await operation()
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
            traffic.report(record_property, name)
            assert traffic.statements["SELECT"] == 1
            assert not any(traffic.statements[verb] for verb in ("INSERT", "UPDATE", "DELETE"))
            assert "JOIN ai_step_events" in traffic.selects[0]
            assert "ai_step_events.metadata" not in traffic.selects[0]
            assert "ai_step_runs.metadata" not in traffic.selects[0]
            return result

        try:
            assert await measured("list", lambda: archive.list_events(agent_run_id=run.run_id)) == expected
            assert await measured("range", lambda: archive.list_event_range(agent_run_id=run.run_id, start=0, end=1)) == expected[:1]
            captured = await measured("captured", lambda: archive.state_store.read(
                lambda tx: archive._captured_history_events(tx, run.run_id, len(expected))))
            assert captured == list(enumerate(expected, 1))
            selected = await measured("selected", lambda: archive.state_store.read(
                lambda tx: archive._read_selected_history_facts(tx, run.run_id, (1,))))
            assert _decode_step_envelope(selected[1].data) == expected[0]
            assert (await _raw_events(archive, run.run_id))[0].data["value"]["type"] == "native_event_reference"
        finally:
            await cold.preflight_close()
            await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("missing", "scope", "body", "lineage", "metadata"))
async def test_cold_start_reference_fails_closed(tmp_path: Path, corruption: str) -> None:
    async with _runtime_start(tmp_path) as (state, native, run, _):
        expected = await state.run_store.read_store(PersistenceDomain.TRACE).list_events(agent_run_id=run.run_id)
        table = build_step_sql_metadata().tables["ai_step_events"]
        statement = delete(table) if corruption == "missing" else update(table).values(**{
            "scope": {"store_digest": "a" * 64},
            "body": {"step_index": 999},
            "lineage": {"run_id": "foreign-run"},
            "metadata": {"metadata": '{"tampered":"body"}'},
        }[corruption])
        async with native.storage_context.engine.begin() as connection:
            await connection.execute(statement.where(table.c.store_digest == native.store_digest, table.c.kind == "run_started"))
        cold = await state.create_agent_run_store()
        try:
            archive = cold.read_store(PersistenceDomain.TRACE)
            if corruption == "metadata":
                assert await archive.list_events(agent_run_id=run.run_id) == expected
                with pytest.raises((AIError, ValueError)):
                    await native.list_events(run_id=run.run_id)
                with pytest.raises(AIError) as caught:
                    await archive.list_event_range(agent_run_id=run.run_id, start=0, end=1,
                        verify_native_source=True)
                assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            else:
                for read in (lambda: archive.list_events(agent_run_id=run.run_id),
                             lambda: archive.list_event_range(agent_run_id=run.run_id, start=0, end=1)):
                    with pytest.raises(AIError) as caught:
                        await read()
                    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        finally:
            await cold.preflight_close()
            await cold.close()


@pytest.mark.asyncio
async def test_portable_trace_exports_complete_start_without_rewriting_sidecar(tmp_path: Path, record_property) -> None:
    async with _runtime_start(tmp_path) as (state, native, run, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        expected_events = await archive.list_events(agent_run_id=run.run_id)
        expected = expected_events[0]
        objects = InMemoryObjectStore()
        resources = SnapshotResources(namespace="native-start", tenant_id="default",
            durable_domains=frozenset({PersistenceDomain.TRACE}),
            stores={PersistenceDomain.TRACE: archive.state_store},
            object_stores={domain: objects for domain in PersistenceDomain}, read_only=True)
        target = InMemoryObjectStore()
        traffic = _Traffic()
        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            ref = await RuntimeSnapshotOperations(resources, require_ready=lambda: None).export_snapshot(
                object_store=target, limits=SnapshotLimits(max_entries=10000, max_bytes=10000000))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        fact_reads = [sql for sql in traffic.sql if sql.lstrip().upper().startswith("SELECT") and "FROM ai_state_facts" in sql]
        traffic.report(record_property, "portable_export")
        assert len(fact_reads) == 1
        assert "JOIN ai_step_events" in fact_reads[0]
        manifest = json.loads(await read_object(target, ref.key, expected_digest=ref.digest, expected_size=ref.size))
        facts = [decode_fact(value) for value in manifest["domains"]["trace"]["facts"]]
        exported = next(fact for fact in facts if fact.kind == "step_event" and fact.state == "AGENT_RUN_STARTED")
        assert _decode_step_envelope(exported.data) == expected
        assert [_decode_step_envelope(fact.data) for fact in facts if fact.kind == "step_event"] == expected_events
        restored_root = tmp_path / "restored-start"
        await Persistence.restore_snapshot(ref, object_store=target, root=restored_root,
            limits=SnapshotLimits(max_entries=10000, max_bytes=10000000))
        async with Persistence.open("native-start", storage=PersistenceStorage.from_root(restored_root)) as restored:
            restored_archive = restored.run_store.read_store(PersistenceDomain.TRACE)
            restored_events = await restored_archive.list_events(agent_run_id=run.run_id)
            assert restored_events == expected_events
            assert (await restored_archive.list_event_range(agent_run_id=run.run_id, start=0, end=1)) == [expected]
        assert (await _raw_events(archive, run.run_id))[0].data["value"]["type"] == "native_event_reference"


@pytest.mark.asyncio
@pytest.mark.parametrize("native_first", (False, True))
async def test_native_and_trace_publication_order_does_not_reinsert_source(tmp_path: Path, native_first: bool, record_property) -> None:
    async with _source(tmp_path, confirmed=False) as (state, native, run, projected_run):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        source = NativeEvent(run_id=run.run_id, kind="run_started", step_index=0,
            conversation_id=run.conversation_id, parent_run_id=run.parent_run_id,
            agent_name=run.agent_name, metadata=dict(run.metadata), idempotency_key="0:0:run_started:")
        projected = NativeEventReferences.project(source, event_index=0)
        await archive.stage_native_event(run, source, projected)
        if native_first:
            prepared = await native.prepare_step(run=run, events=(source,), managed_start=True)
            receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, prepared))
            native.confirm_prepared_step(receipt)
            archive.confirm_native_event_source(receipt)
        traffic = _Traffic()
        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            await archive.sync_projection(projected_run, events=(projected,), checkpoints=(), execution_id="execution")
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        if not native_first:
            assert native.confirmed_run(run.run_id) is None
            raw_run = await archive.state_store.read(lambda tx: tx.get_record(archive._agent_run_key(run.run_id)))
            assert raw_run.data["value"]["type"] == "native_agent_run_reference"
            assert native.confirmed_run(run.run_id) is None
        receipts = archive.confirm_native_event_projection((projected,))
        inserts = [sql for sql in traffic.sql if sql.startswith("INSERT INTO ai_step_events ")]
        traffic.report(record_property, "native_first" if native_first else "trace_first")
        assert len(inserts) == (0 if native_first else 1)
        if not native_first:
            prepared = await native.prepare_step(run=run, events=(source,), managed_start=True)
            replay = _Traffic()
            event.listen(engine.sync_engine, "before_cursor_execute", replay.capture)
            try:
                receipt = await archive.state_store.mutate(
                    lambda tx: native.write_prepared_step(tx, prepared, confirmed=receipts[0]))
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", replay.capture)
            archive.confirm_native_event_source(receipt)
            replay.report(record_property, "native_after_trace")
            assert not any(replay.statements[verb] for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"))
        assert native.started_from_receipt(receipts[0]) == source
        assert await native.list_events(run_id=run.run_id) == [source]
        assert await archive.list_events(agent_run_id=run.run_id) == [projected]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("custom", "split", "subclass"))
async def test_nonshared_runtime_keeps_independent_inline_start(tmp_path: Path, backend: str) -> None:
    async with _runtime_start(tmp_path, backend=backend) as (state, native, run, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        raw = await _raw_events(archive, run.run_id)
        starts = [fact for fact in raw if fact.state == "AGENT_RUN_STARTED"]
        assert len(starts) == 1
        assert starts[0].data["value"]["type"] != "native_event_reference"
        assert _decode_step_envelope(starts[0].data).event_type == "AGENT_RUN_STARTED"
        assert len([item for item in await native.list_events(run_id=run.run_id) if item.kind == "run_started"]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("commit", (CommitPolicy.sync(), CommitPolicy.terminal()), ids=("sync", "terminal"))
async def test_trace_without_native_checkpoints_publishes_and_seals_inline_history(
    tmp_path: Path, backend: str, commit: CommitPolicy,
) -> None:
    features = RuntimeFeatures("trace-without-checkpoints")
    features.capability(TraceCapability(commit=commit), scope="runtime")
    features.agent("default", model="default", allow_tools=())

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("done")])

    storage = (PersistenceStorage.in_memory() if backend == "memory" else
               PersistenceStorage.sqlite(tmp_path / "trace-only.db"))
    async with Persistence.open("trace-without-checkpoints", storage=storage) as state:
        async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            execution = await runtime.agents.get().start("work", idempotency_key="work")
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED

        head = await state.execution.executions.get(execution.execution_id, tenant_id="default")
        seal = await state.execution.executions.get_history_seal(execution.execution_id, tenant_id="default")
        assert head is not None and not head.restart_checkpoints and head.durable_trace
        assert seal is not None and head.history_seal_digest == seal.seal_digest
        sealed_run, = seal.run_heads
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        run = await archive.get_agent_run(agent_run_id=sealed_run.agent_run_id)
        assert run is not None
        events = await archive.list_events(agent_run_id=run.agent_run_id)
        assert sealed_run.event_count == len(events)
        assert [event.event_index for event in events] == list(range(len(events)))
        assert [event.event_type for event in events].count("AGENT_RUN_STARTED") == 1
        assert events[0].event_type == "AGENT_RUN_STARTED"
        assert events[-1].event_type == "AGENT_RUN_SUCCEEDED"
        if backend == "sqlite":
            raw = await _raw_events(archive, run.agent_run_id)
            assert _decode_step_envelope(raw[0].data) == events[0]


@pytest.mark.asyncio
async def test_unstaged_archive_start_stays_inline(tmp_path: Path) -> None:
    async with _source(tmp_path, confirmed=False) as (state, native, run, projected_run):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        source = NativeEvent(run_id=run.run_id, kind="run_started", step_index=0,
            idempotency_key="0:0:run_started:")
        projected = NativeEventReferences.project(source, event_index=0)
        await archive.sync_projection(projected_run, events=(projected,), checkpoints=(), execution_id="execution")
        raw = await _raw_events(archive, run.run_id)
        assert _decode_step_envelope(raw[0].data) == projected
        assert raw[0].data["value"]["type"] != "native_event_reference"


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ("native", "trace"))
@pytest.mark.parametrize("large", (False, True))
async def test_concurrent_first_publishers_keep_receipt_proof_after_cache_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: str, large: bool, record_property,
) -> None:
    from dataclasses import replace
    from linktools.ai.persistence import _step_store

    async with _source(tmp_path, confirmed=False) as (state, native, original, _):
        run = replace(original, metadata={"large": "x" * (1024 * 1024 + 1)}) if large else original
        # This forces actual confirmation to evict rather than mocking the proof.
        monkeypatch.setattr(_step_store, "_RUN_CACHE_BYTES", 1)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        references = NativeEventReferences(archive.state_store, namespace="native-run-facts", tenant_id="tenant")
        references.bind(native)
        source = NativeEvent(run_id=run.run_id, kind="run_started", step_index=0,
            conversation_id=run.conversation_id, parent_run_id=run.parent_run_id,
            agent_name=run.agent_name, metadata=dict(run.metadata), idempotency_key="0:0:run_started:")
        projected = references.project(source, event_index=0)
        await references.stage(run, source, projected)
        prepared = await native.prepare_step(run=run, events=(source,), managed_start=True)
        committed, attempted, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        entered = []

        async def publish(kind: str) -> None:
            if kind != first:
                await committed.wait()
                attempted.set()
            async with references.publication(run.run_id):
                entered.append(kind)
                if kind == "trace":
                    await archive.state_store.mutate(lambda tx: references.publish(tx, (projected,)))
                else:
                    receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(
                        tx, prepared, confirmed=references.confirmed_receipt(run.run_id)))
                if kind == first:
                    # Hold exactly the committed-but-not-confirmed gap while the
                    # other publisher attempts admission on the same source.
                    committed.set()
                    await release.wait()
                if kind == "trace":
                    assert references.encode(projected)["value"]["payload"] == {
                        "key": native.event_key_digest(source), "index": 0,
                    }
                    references.confirm((projected,))
                else:
                    native.confirm_prepared_step(receipt)
                    references.confirm_native(receipt)
                assert native.confirmed_run(run.run_id) is None

        traffic = _Traffic()
        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        tasks = [asyncio.create_task(publish(kind)) for kind in ("native", "trace")]
        try:
            await asyncio.wait_for(attempted.wait(), 5)
            await asyncio.sleep(0)
            assert entered == [first]
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 5)
        finally:
            release.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        traffic.report(record_property, f"concurrent_{first}_{'large' if large else 'small'}")
        assert traffic.statements["SELECT"] == 0
        assert len([sql for sql in traffic.sql if sql.startswith("INSERT INTO ai_step_events ")]) == 1
        assert await native.list_events(run_id=run.run_id) == [source]
        retained = references.confirmed_receipt(run.run_id)
        assert retained is not None
        assert native.started_from_receipt(retained) == source
        assert not references.has_pending(run.run_id)
        references.release_settled(run.run_id)
        assert references.confirmed_receipt(run.run_id) is None


@pytest.mark.asyncio
async def test_native_source_write_failure_prevents_provider_dispatch(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rejected-source.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="rejected-source", tenant_id="default")
    calls = []
    rejected = []

    async def model(_messages, _info):
        calls.append("provider")
        return ModelResponse(parts=[TextPart("must not dispatch")])

    def reject(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith("INSERT INTO ai_step_events "):
            rejected.append(statement)
            raise AIError(ErrorCode.STORAGE_CONFLICT, "native source unavailable")

    features = RuntimeFeatures("rejected-source")
    features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    try:
        async with Persistence.open("rejected-source", storage=PersistenceStorage.sql(engine)) as state:
            event.listen(engine.sync_engine, "before_cursor_execute", reject)
            try:
                async with Runtime.attach(persistence=state,
                        features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
                    execution = await runtime.agents.get().start("work", idempotency_key="work")
                    try:
                        result = await execution.wait(timeout_seconds=5)
                    except AIError as error:
                        assert error.code in {ErrorCode.STORAGE_COMMIT_UNKNOWN, ErrorCode.STORAGE_CONFLICT,
                                              ErrorCode.STORAGE_INTEGRITY_ERROR}
                    else:
                        assert result.result.status is not ExecutionStatus.SUCCEEDED
                    assert rejected
                    assert calls == []
                    assert await native.list_runs() == []
                    archive = state.run_store.read_store(PersistenceDomain.TRACE)
                    for run in await archive.list_agent_runs():
                        raw = await _raw_events(archive, run.agent_run_id)
                        assert all(fact.data["value"]["type"] != "native_event_reference" for fact in raw)
                    event.remove(engine.sync_engine, "before_cursor_execute", reject)
                    cancellation = await execution.cancel(idempotency_key="cancel-rejected-source")
                    if not cancellation.cancelled:
                        await execution.recover(idempotency_key="recover-rejected-source-cancel")
                    assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.CANCELLED
                    assert calls == []
            finally:
                if event.contains(engine.sync_engine, "before_cursor_execute", reject):
                    event.remove(engine.sync_engine, "before_cursor_execute", reject)
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("error", "cancel", "invalid_token", "invalid_clock"))
async def test_early_start_preparation_failure_preserves_original_error(failure: str) -> None:
    from linktools.ai.runtime._step_store import RuntimeStepStore
    from .test_atomic_step_capability import _context

    source = InMemoryStepStore()
    registration = StepRunRegistration(run_id="run", parent_run_id=None, agent_name=None,
        metadata={}, share_started=True)
    capability = AtomicStepPersistence(store=RuntimeStepStore(source, InMemoryStepStore()),
        run_id="run", run_registration=registration)
    original = asyncio.CancelledError("preparation cancelled") if failure == "cancel" else RuntimeError("preparation failed")

    async def seed() -> tuple[str, datetime]:
        if failure in {"error", "cancel"}:
            raise original
        return ("" if failure == "invalid_token" else "token",
                datetime.now() if failure == "invalid_clock" else datetime.now(timezone.utc))

    registration.bind_registration(seed)
    expected = type(original) if failure in {"error", "cancel"} else ValueError
    with pytest.raises(expected) as caught:
        await registration.prepare_started(conversation_id="conversation", step_index=1)
    if failure in {"error", "cancel"}:
        assert caught.value is original
    else:
        original = caught.value
        assert "seed is invalid" in str(original)
    assert registration.prepared_started() is None

    async def emit(_event) -> None:
        raise AssertionError("no snapshot was prepared")

    with pytest.raises(type(original)) as propagated:
        await capability.on_run_error(_context([], emit), error=original)
    assert propagated.value is original
    assert registration.prepared_started() is None
    assert await source.list_runs() == []
    events = await source.list_events(run_id="run")
    assert len(events) == 1
    assert events[0].kind == "run_failed"
    assert events[0].error == repr(original)
    assert all(item.kind != "run_started" for item in events)


@pytest.mark.asyncio
async def test_large_metadata_is_not_returned_by_cold_trace_select(tmp_path: Path, record_property) -> None:
    returned_sizes = []
    for size in (0, 65537):
        directory = tmp_path / str(size)
        directory.mkdir()
        async with _runtime_start(directory, metadata_bytes=size) as (state, native, run, _):
            cold = await state.create_agent_run_store()
            try:
                archive = cold.read_store(PersistenceDomain.TRACE)
                references = NativeEventReferences(archive.state_store, namespace="native-start", tenant_id="default")
                query = references.statement(query=FactQuery(archive._stream(run.run_id, "event"), limit=1))
                traffic = _Traffic()
                engine = native.storage_context.engine
                event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
                try:
                    rows = await archive.state_store.read(lambda tx: tx.read_joined_facts(archive.state_store.context, query))
                finally:
                    event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
                # Values come from the actual returned SQL mapping, not a DTO.
                # Numeric/driver framing/disk bytes are deliberately not included.
                returned_bytes = sum(_SqlTraffic._bytes(value) for _, row in rows for value in row.values())
                returned_sizes.append(returned_bytes)
                traffic.report(record_property, f"metadata_{size}")
                record_property(f"metadata_{size}_returned_sql_text_binary_bytes", returned_bytes)
                assert traffic.statements["SELECT"] == 1
                assert "ai_step_events.metadata" not in traffic.selects[0]
                assert "ai_step_runs.metadata" not in traffic.selects[0]
                assert all("native_metadata" not in row for _, row in rows)
            finally:
                await cold.preflight_close()
                await cold.close()
    # Independently generated IDs and timestamp text can vary slightly; the
    # returned fixed-field row must not scale with the 64 KiB source metadata.
    assert abs(returned_sizes[1] - returned_sizes[0]) < 128


@pytest.mark.asyncio
@pytest.mark.parametrize("native_first", (False, True))
@pytest.mark.parametrize("cold", (False, True))
async def test_replay_metadata_adopts_original_source_in_both_publication_orders(
    tmp_path: Path, native_first: bool, cold: bool, record_property,
) -> None:
    from dataclasses import replace
    from linktools.ai.persistence._official_step_reader import OfficialStepReader

    async with _source(tmp_path, confirmed=False) as (state, native, original_run, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        original_start = NativeEvent(run_id=original_run.run_id, kind="run_started", step_index=0,
            timestamp=original_run.started_at, conversation_id=original_run.conversation_id,
            parent_run_id=original_run.parent_run_id, agent_name=original_run.agent_name,
            metadata=dict(original_run.metadata), idempotency_key="0:0:run_started:")
        prepared = await native.prepare_step(run=original_run, events=(original_start,), managed_start=True)
        original_receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, prepared))
        native.confirm_prepared_step(original_receipt)
        if cold:
            await native.close()
        candidate_run = replace(original_run, metadata={"new": "candidate metadata"})
        candidate_start = replace(original_start, metadata=dict(candidate_run.metadata))
        candidate = await native.prepare_step(run=candidate_run, events=(candidate_start,), managed_start=True)
        references = NativeEventReferences(archive.state_store, namespace="native-run-facts", tenant_id="tenant")
        references.bind(native)
        projected = references.project(candidate_start, event_index=0)
        await references.stage(candidate_run, candidate_start, projected)
        traffic = _Traffic()
        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            for publisher in (("native", "trace") if native_first else ("trace", "native")):
                async with references.publication(original_run.run_id):
                    if publisher == "native":
                        receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(
                            tx, candidate, confirmed=references.confirmed_receipt(original_run.run_id)))
                        native.confirm_prepared_step(receipt)
                        references.confirm_native(receipt)
                    else:
                        await archive.state_store.mutate(lambda tx: references.publish(tx, (projected,)))
                        assert references.resolve_run(OfficialStepReader.project_run(candidate_run)) == OfficialStepReader.project_run(original_run)
                        receipt, = references.confirm((projected,))
                    assert native.run_from_receipt(receipt) == original_run
                    assert native.started_from_receipt(receipt) == original_start
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        traffic.report(record_property, f"replay_{'cold' if cold else 'warm'}_{'native' if native_first else 'trace'}")
        assert traffic.statements["SELECT"] == (1 if cold else 0)
        assert not [sql for sql in traffic.sql if sql.startswith("INSERT INTO ai_step_events ")]
        assert await native.list_events(run_id=original_run.run_id) == [original_start]


@pytest.mark.asyncio
async def test_runtime_recovers_trace_start_commit_ack_loss_with_joined_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, record_property,
) -> None:
    from sqlalchemy.engine import Engine
    from sqlalchemy.ext.asyncio import AsyncSessionTransaction
    from linktools.ai.persistence._step_archive import StateStepArchive

    original_exit = AsyncSessionTransaction.__aexit__
    original_range = StateStepArchive.list_event_range
    lost = []
    checked_ranges = []
    marker = "native_start_test_ack_loss"

    def mark_start(connection, _cursor, statement, parameters, _context, _many):
        if (not lost and statement.startswith("INSERT INTO ai_state_facts ")
                and "native_event_reference" in str(parameters)):
            connection.info[marker] = True

    async def lose_ack(transaction, error_type, error, traceback):
        selected = False
        if error_type is None and not transaction.nested and not lost:
            connection = await transaction.session.connection()
            selected = connection.info.pop(marker, False)
        result = await original_exit(transaction, error_type, error, traceback)
        if selected:
            lost.append("committed")
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN, "start transaction committed; acknowledgement lost")
        return result

    async def observed_range(archive, *, agent_run_id, start, end, verify_native_source=False):
        if not verify_native_source:
            return await original_range(archive, agent_run_id=agent_run_id, start=start, end=end)
        traffic = _Traffic()
        engine = archive.state_store.context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            result = await original_range(archive, agent_run_id=agent_run_id, start=start, end=end,
                verify_native_source=True)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        traffic.report(record_property, f"ack_readback_{len(checked_ranges)}")
        assert traffic.statements["SELECT"] == 1
        assert "JOIN ai_step_events" in traffic.selects[0]
        assert "ai_step_events.metadata" in traffic.selects[0]
        checked_ranges.append(result)
        return result

    monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", lose_ack)
    monkeypatch.setattr(StateStepArchive, "list_event_range", observed_range)
    event.listen(Engine, "before_cursor_execute", mark_start)
    try:
        async with _runtime_start(tmp_path, trace_sync=True) as (state, native, run, traffic):
            traffic.report(record_property, "runtime_ack_loss")
            assert lost == ["committed"]
            assert any(item.event_type == "AGENT_RUN_STARTED" for page in checked_ranges for item in page)
            starts = [item for item in await native.list_events(run_id=run.run_id) if item.kind == "run_started"]
            assert len(starts) == 1
            archive = state.run_store.read_store(PersistenceDomain.TRACE)
            events = await archive.list_events(agent_run_id=run.run_id)
            assert events[0] == NativeEventReferences.project(starts[0], event_index=0)
            assert len([fact for fact in await _raw_events(archive, run.run_id)
                        if fact.state == "AGENT_RUN_STARTED"]) == 1
    finally:
        event.remove(Engine, "before_cursor_execute", mark_start)


@pytest.mark.asyncio
async def test_admitted_candidate_capture_uses_canonical_run_after_native_tail_invalidation(
    tmp_path: Path, record_property,
) -> None:
    from dataclasses import replace
    from linktools.ai.persistence._official_step_reader import OfficialStepReader
    from linktools.ai.runtime._step_store import RuntimeStepStore

    async with _shared_candidate_source(tmp_path) as (state, native, original, candidate_trace):
        store = state.run_store
        archive = store.read_store(PersistenceDomain.TRACE)
        failure = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN, "native tail acknowledgement unavailable")

        async def reject_tail(**_kwargs):
            raise failure

        store.bind_step_boundary(original.run_id, reject_tail, require_active=lambda: None)
        source = NativeEvent(run_id=original.run_id, kind="run_started", step_index=0,
            timestamp=original.started_at, conversation_id=original.conversation_id,
            parent_run_id=original.parent_run_id, agent_name=original.agent_name,
            metadata=dict(original.metadata), idempotency_key="0:0:run_started:")
        initial = await native.prepare_step(run=original, events=(source,), managed_start=True)
        original_receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, initial))
        native.confirm_prepared_step(original_receipt)
        candidate = replace(original, metadata=dict(candidate_trace.metadata))
        candidate_source = replace(source, metadata=dict(candidate.metadata))
        projected = store.project_native_event(candidate_source, event_index=0)
        await store.stage_native_event(candidate, candidate_source, projected)
        await store.append_event(projected)
        capture = await store.capture_execution_projection(original.run_id)
        assert capture is not None
        captured, flight = capture
        assert captured.run == candidate_trace

        replay = await native.prepare_step(run=candidate, events=(candidate_source,), managed_start=True)
        receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, replay))
        native.confirm_prepared_step(receipt)
        store.confirm_step_run(candidate, receipt=receipt)
        canonical = OfficialStepReader.project_run(original)
        assert store._canonical_trace_run(captured.run) == canonical

        adapter = RuntimeStepStore(native, object(), working_set=store, commit_boundary=reject_tail)
        with pytest.raises(AIError) as rejected:
            await adapter.commit_step(events=(replace(source, kind="run_failed", idempotency_key="tail", error="failure"),))
        assert rejected.value is failure
        # The real adapter invalidated OfficialStepReader; the admitted capture
        # still has candidate bytes and must resolve via canonical staging.
        assert captured.run == candidate_trace
        traffic = _Traffic()
        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            assert store._canonical_trace_run(captured.run) == canonical
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        traffic.report(record_property, "canonical_after_invalidation")
        assert traffic.statements["SELECT"] == 0
        await store.commit_captured_execution_projection(captured, flight, execution_id="execution")
        assert await archive.get_agent_run(agent_run_id=original.run_id) == canonical
        plan = await store.prepare_execution_terminal_seal(
            execution_id="execution", agent_run_ids=(original.run_id,), binding_digest="binding")
        try:
            assert plan.projections
            assert all(projection.run == canonical for projection in plan.projections)
        finally:
            await store.discard_execution_terminal_seal(plan)


@asynccontextmanager
async def _shared_candidate_source(tmp_path: Path, *, large: bool = True) -> AsyncIterator[tuple]:
    from dataclasses import replace
    from pydantic_ai_harness.step_persistence import RunRecord
    from linktools.ai.persistence._official_step_reader import OfficialStepReader
    from .test_incremental_projection_boundaries import _storage
    from .test_step_projection import _execution

    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="native-run-facts", tenant_id="tenant")
    native = await state.create_step_store()
    # The small case deliberately makes a native reference larger than the inline run.
    original = RunRecord(run_id="run", conversation_id="conversation", parent_run_id="parent",
        agent_name="agent", registration_id="registration" if large else "registration" * 128,
        metadata={"shared": "original" * (1024 if large else 1)})
    candidate = replace(OfficialStepReader.project_run(original), metadata={"shared": "candidate" * (1024 if large else 1)})
    try:
        await state.execution.executions.create_execution(replace(_execution(), revision=1,
            producer_generation=1, restart_checkpoints=True))
        state.run_store.bind_step_store(native)
        state.run_store.bind_execution_producer("run", execution_id="execution", producer_generation=1,
            restart_checkpoints=True)
        await state.run_store.register_agent_run(candidate, execution_id="execution")
        yield state, native, original, candidate
    finally:
        await state.close()
        await native.close()


@pytest.mark.asyncio
async def test_two_admitted_buffers_reproject_canonical_metadata_after_reader_invalidation(tmp_path: Path) -> None:
    from dataclasses import replace
    from linktools.ai.persistence._official_step_reader import OfficialStepReader
    from linktools.ai.persistence._projection_payload import decode_projection_payload
    from linktools.ai.persistence._step_contracts import StepEvent
    from linktools.ai.runtime._step_store import RuntimeStepStore
    from linktools.ai.storage import InMemoryCacheBackend

    async with _shared_candidate_source(tmp_path) as (state, native, original, candidate_trace):
        store = state.run_store
        archive = store.read_store(PersistenceDomain.TRACE)
        cache = InMemoryCacheBackend(max_bytes=4 * 1024 * 1024)
        owner = store.trace_persistence
        owner.configure("run", buffer=cache, terminal=True, batch_size=128, flush_interval=60)
        marker = AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def fail_tail(**_kwargs):
            raise marker

        store.bind_step_boundary("run", fail_tail, require_active=lambda: None)
        source = NativeEvent(run_id="run", kind="run_started", step_index=0, timestamp=original.started_at,
            conversation_id=original.conversation_id, parent_run_id=original.parent_run_id,
            agent_name=original.agent_name, metadata=dict(original.metadata), idempotency_key="0:0:run_started:")
        prepared = await native.prepare_step(run=original, events=(source,), managed_start=True)
        receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, prepared))
        native.confirm_prepared_step(receipt)
        candidate = replace(original, metadata=dict(candidate_trace.metadata))
        candidate_source = replace(source, metadata=dict(candidate.metadata))
        projected = store.project_native_event(candidate_source, event_index=0)
        await store.stage_native_event(candidate, candidate_source, projected)
        await store.append_event(projected)
        await owner.defer("run", execution_id="execution", producer_generation=1)
        buffer = owner._buffers["run"]
        first = await cache.read_batch(buffer.id, limit=128)
        assert len(first) == 1
        assert decode_projection_payload(first[0][2]).run == candidate_trace

        store.confirm_step_run(candidate, receipt=receipt)
        canonical = OfficialStepReader.project_run(original)
        tail = StepEvent("run", "MODEL_REQUEST_CANCELLED", 1, event_index=1,
            idempotency_key="1:cancel", metadata={"linktools.ai.model_request_seq": "1"})
        await store.append_event(tail)
        await owner.defer("run", execution_id="execution", producer_generation=1)
        queued = await cache.read_batch(buffer.id, limit=128)
        assert len(queued) == 2
        assert decode_projection_payload(queued[0][2]).run == candidate_trace
        assert decode_projection_payload(queued[1][2]).run == canonical
        adapter = RuntimeStepStore(native, object(), working_set=store, commit_boundary=fail_tail)
        with pytest.raises(AIError) as caught:
            await adapter.commit_step(events=(replace(source, kind="run_failed", idempotency_key="failed"),))
        assert caught.value is marker
        async with buffer.lock:
            selected = await owner._read_pending_locked(buffer)
        assert selected is not None
        drained, admitted_ids = selected
        assert drained.run == canonical
        assert drained.events == (projected, tail)
        assert admitted_ids == tuple(item[0] for item in queued)
        # Reading/reprojection must leave both admitted bytes and IDs untouched.
        assert await cache.read_batch(buffer.id, limit=128) == queued
        await store.flush_execution_projection("run", execution_id="execution", producer_generation=1)
        assert await archive.get_agent_run(agent_run_id="run") == canonical
        assert await archive.list_events(agent_run_id="run") == [projected, tail]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("none", "metadata", "key", "step", "registration"))
@pytest.mark.parametrize("large", (False, True))
async def test_terminal_unknown_verification_checks_pending_source_in_original_run_select(
    tmp_path: Path, damage: str, large: bool, record_property,
) -> None:
    from linktools.ai.persistence._official_step_reader import OfficialStepReader
    from linktools.ai.persistence._step_archive import PreparedExecutionProjection

    async with _shared_candidate_source(tmp_path, large=large) as (state, native, original, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        source = NativeEvent(run_id="run", kind="run_started", step_index=0, timestamp=original.started_at,
            conversation_id=original.conversation_id, parent_run_id=original.parent_run_id,
            agent_name=original.agent_name, metadata=dict(original.metadata), idempotency_key="0:0:run_started:")
        projected = NativeEventReferences.project(source, event_index=0)
        await archive.stage_native_event(original, source, projected)
        canonical = OfficialStepReader.project_run(original)
        await archive.sync_projection(canonical, events=(projected,), checkpoints=(), execution_id="execution")
        # Deliberately leave the receipt provisional, as at terminal ACK loss.
        assert native.confirmed_run("run") is None
        raw_run = await archive.state_store.read(lambda tx: tx.get_record(archive._agent_run_key("run")))
        if not large:
            assert raw_run.data["value"]["type"] != "native_agent_run_reference"
        head = await archive.execution_history_head_record("run")
        projection = PreparedExecutionProjection(run=canonical, events=(projected,), checkpoints=(),
            base_event_offset=0, base_checkpoint_offset=0, target_event_offset=head.event_count,
            target_checkpoint_offset=head.checkpoint_count, target_transcript_message_count=head.transcript_message_count,
            durable_projection_digest=head.projection_digest, target_interaction_offset=head.interaction_count)
        cold = await state.create_agent_run_store()
        engine = native.storage_context.engine
        try:
            baseline = _Traffic()
            event.listen(engine.sync_engine, "before_cursor_execute", baseline.capture)
            try:
                assert await cold.read_store(PersistenceDomain.TRACE).verify_execution_projection_head(projection)
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", baseline.capture)
            baseline.report(record_property, "terminal_without_pending_source")
            assert baseline.statements["SELECT"] == 3
            if damage != "none":
                tables = build_step_sql_metadata().tables
                table = tables["ai_step_runs" if damage == "registration" else "ai_step_events"]
                values = {"metadata": {"metadata": '{"corrupted":"source metadata"}'},
                          "key": {"key_digest": "f" * 64}, "step": {"step_index": 99},
                          "registration": {"registration_id": "foreign-registration"}}[damage]
                statement = update(table).where(table.c.store_digest == native.store_digest)
                if damage != "registration":
                    statement = statement.where(table.c.kind == "run_started")
                async with engine.begin() as connection:
                    await connection.execute(statement.values(**values))
            if damage == "registration":
                assert await archive.list_events(agent_run_id="run") == [projected]
                with pytest.raises(AIError) as rejected:
                    await archive.list_event_range(agent_run_id="run", start=0, end=1, verify_native_source=True)
                assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                assert native.confirmed_run("run") is None
            checked = _Traffic()
            event.listen(engine.sync_engine, "before_cursor_execute", checked.capture)
            try:
                try:
                    accepted = await archive.verify_execution_projection_head(projection)
                except AIError as error:
                    assert damage != "none"
                    assert error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                    accepted = False
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", checked.capture)
            checked.report(record_property, "terminal_with_pending_source")
            assert accepted is (damage == "none")
            assert "JOIN ai_step_events" in checked.selects[0]
            assert checked.statements["SELECT"] <= baseline.statements["SELECT"]
            if damage == "none":
                assert checked.statements["SELECT"] == baseline.statements["SELECT"]
        finally:
            await cold.preflight_close()
            await cold.close()
        if damage == "none":
            archive.confirm_native_event_projection((projected,))


@pytest.mark.parametrize("dialect_name", ("sqlite", "postgresql", "mysql"))
def test_terminal_source_join_keeps_explicit_locator_bind_budget(dialect_name: str) -> None:
    from types import SimpleNamespace
    from sqlalchemy.dialects import mysql, postgresql, sqlite
    from linktools.ai.persistence._native_run_reference import NativeRunReferences

    dialect = {"sqlite": sqlite, "postgresql": postgresql, "mysql": mysql}[dialect_name].dialect()
    # Compile only: the engine never opens a connection or provisions a schema.
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    native = SqlStepStore(engine, namespace="native-start", tenant_id="tenant")
    state = SimpleNamespace(context=SimpleNamespace(engine=engine, dialect=dialect), store_digest=b"a" * 32)
    references = NativeRunReferences(state, namespace="native-start", tenant_id="tenant")
    references.bind(native)
    source = NativeEvent(run_id="run", kind="run_started", step_index=0, idempotency_key="0:0:run_started:")
    compiled = references.statement(keys=(b"b" * 32,), expected_start=source).compile(
        dialect=dialect, compile_kwargs={"render_postcompile": True})
    locators = {"source_run_id", "source_event_key"}
    assert locators <= compiled.params.keys()
    assert compiled.params["source_run_id"] == "run"
    assert compiled.params["source_event_key"] == native.event_key_digest(source)
    assert len(compiled.params if compiled.positiontup is None else compiled.positiontup) == 4
    ordinary = references.statement(keys=tuple(index.to_bytes(32, "big") for index in range(998))).compile(
        dialect=dialect, compile_kwargs={"render_postcompile": True})
    assert not locators.intersection(ordinary.params)
    assert len(ordinary.params if ordinary.positiontup is None else ordinary.positiontup) == 999


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "shortcircuit"))
@pytest.mark.parametrize("metadata_bytes", (0, 65537))
async def test_start_sharing_matches_inline_workload_sql_budget(
    tmp_path: Path, outcome: str, metadata_bytes: int, record_property,
) -> None:
    import time
    import tracemalloc

    # The unchanged SqlStepStore subclass disables only owned-start sharing.
    # This is a current-algorithm control, not a historical-commit benchmark.
    results = {}
    for backend in ("owned", "subclass"):
        directory = tmp_path / backend
        directory.mkdir()
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        started_wall, started_cpu = time.perf_counter_ns(), time.process_time_ns()
        try:
            async with _runtime_start(directory, outcome=outcome, backend=backend,
                    metadata_bytes=metadata_bytes, trace_sync=True) as (state, native, run, traffic):
                archive = state.run_store.read_store(PersistenceDomain.TRACE)
                raw = await _raw_events(archive, run.run_id)
                projected = await archive.list_events(agent_run_id=run.run_id)
                start = next(fact for fact in raw if fact.state == "AGENT_RUN_STARTED")
                source_starts = [item for item in await native.list_events(run_id=run.run_id) if item.kind == "run_started"]
                assert len(source_starts) == 1
                traffic.report(record_property, backend)
                results[backend] = traffic
                if backend == "owned":
                    assert start.data["value"]["type"] == "native_event_reference"
                    sidecar_bytes = len(canonical_json_bytes(dict(start.data)))
                    former_inline = len(canonical_json_bytes(_encode_step_envelope(projected[0])))
                    record_property("shared_start_sidecar_logical_json_bytes", sidecar_bytes)
                    record_property("shared_start_added_proof_ascii_bytes", 128)
                    record_property("shared_start_former_inline_logical_json_bytes", former_inline)
                    record_property("shared_start_signed_logical_byte_delta", sidecar_bytes + 128 - former_inline)
                else:
                    assert start.data["value"]["type"] != "native_event_reference"
        finally:
            elapsed_wall = time.perf_counter_ns() - started_wall
            elapsed_cpu = time.process_time_ns() - started_cpu
            _, peak = tracemalloc.get_traced_memory()
            if not was_tracing:
                tracemalloc.stop()
            record_property(f"{backend}_scenario_wall_ns", elapsed_wall)
            record_property(f"{backend}_scenario_cpu_ns", elapsed_cpu)
            record_property(f"{backend}_tracemalloc_session_peak_bytes", peak)
            record_property(f"{backend}_tracemalloc_inherited_session", was_tracing)
    shared, inline = results["owned"], results["subclass"]
    shared_dml = sum(shared.statements[verb] for verb in ("INSERT", "UPDATE", "DELETE"))
    inline_dml = sum(inline.statements[verb] for verb in ("INSERT", "UPDATE", "DELETE"))
    shared_native_inserts = sum(sql.startswith("INSERT INTO ai_step_events ") for sql in shared.sql)
    inline_native_inserts = sum(sql.startswith("INSERT INTO ai_step_events ") for sql in inline.sql)
    record_property("shared_total_dml", shared_dml)
    record_property("inline_total_dml", inline_dml)
    record_property("shared_native_event_insert_statements", shared_native_inserts)
    record_property("inline_native_event_insert_statements", inline_native_inserts)
    assert shared.statements["SELECT"] <= inline.statements["SELECT"]
    assert shared_native_inserts <= inline_native_inserts + 1
    assert shared_dml <= inline_dml + 1
