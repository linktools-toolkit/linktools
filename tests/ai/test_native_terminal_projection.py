#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Owned terminal source proofs, ordered Trace projection and SQL budgets (CI)."""

from dataclasses import replace
from pathlib import Path

from pydantic_ai_harness.step_persistence import RunRecord

import pytest
from sqlalchemy import delete, event, select, update

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain, SqlStepStore
from linktools.ai.persistence._native_event_reference import NativeEventReferences
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.persistence._step_archive import PreparedExecutionProjection
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import _encode_step_envelope
from linktools.ai.steps import NativeLifecycleEvent

from .test_native_started_source import native, records, managed_commit
from .test_native_start_projection import _Traffic, _raw_events, _runtime_start, _shared_candidate_source


def terminal(run: RunRecord, *, outcome: str = "succeeded") -> NativeLifecycleEvent:
    return NativeLifecycleEvent(
        run_id=run.run_id, kind="run_completed" if outcome in {"succeeded", "interrupted"} else "run_failed",
        step_index=2, timestamp=run.started_at, conversation_id=run.conversation_id,
        parent_run_id=run.parent_run_id, agent_name=run.agent_name, metadata=dict(run.metadata),
        outcome=outcome, error="original diagnostic" if outcome in {"failed", "cancelled"} else None,
        idempotency_key=NativeLifecycleEvent.source_key(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("succeeded", "interrupted", "failed", "cancelled"))
async def test_terminal_outcome_and_full_source_survive_cold_replay(native, outcome: str, record_property) -> None:
    store, engine = native
    run, started = records()
    ended = terminal(run, outcome=outcome)
    receipt = await managed_commit(store, run=run, events=(started, ended))
    assert store.events_from_receipt(receipt) == (started, ended)
    assert await store.list_events(run_id=run.run_id) == [started, ended]
    table = build_step_sql_metadata().tables["ai_step_events"]
    async with engine.connect() as connection:
        row = (await connection.execute(select(table).where(table.c.kind != "run_started"))).mappings().one()
    assert row["outcome"] == outcome and row["metadata"] is None and row["run_fields"] == 15
    assert NativeEventReferences.project(ended, event_index=3).event_type == {
        "succeeded": "AGENT_RUN_SUCCEEDED", "interrupted": "AGENT_RUN_INTERRUPTED",
        "failed": "AGENT_RUN_FAILED", "cancelled": "AGENT_RUN_FAILED",
    }[outcome]
    cold = SqlStepStore(engine, namespace="native", tenant_id="one")
    traffic = _Traffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        replay = await managed_commit(cold, run=run, events=(started, ended))
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        assert cold.events_from_receipt(replay) == (started, ended)
        assert traffic.statements["SELECT"] == 1
        assert not any(sql.startswith("INSERT INTO ai_step_events ") for sql in traffic.sql)
        source_select, = [sql for sql in traffic.sql if sql.startswith("SELECT")]
        assert "JOIN ai_step_events AS terminal" in source_select
        traffic.report(record_property, "cold_terminal_replay")
        record_property("source_row_upper_bound", 1)
        with pytest.raises(ValueError, match="body differs"):
            await managed_commit(cold, run=run, events=(started, replace(ended, metadata={"changed": "body"})))
        with pytest.raises(ValueError, match="body differs"):
            await managed_commit(cold, run=run, events=(started, replace(ended, step_index=3)))
    finally:
        await cold.close()


@pytest.mark.asyncio
async def test_registration_reads_preserve_confirmed_terminal_and_start_proof(native) -> None:
    store, _ = native
    run, started = records()
    ended = terminal(run)
    await managed_commit(store, run=run, events=(started, ended))
    assert await store.get_run(run_id=run.run_id) == run
    assert await store.list_runs() == [run]
    receipt = await managed_commit(store, run=run, events=(started, ended))
    assert store.events_from_receipt(receipt) == (started, ended)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("outcome", "metadata", "hash", "both_hashes", "start_outcome"))
async def test_cold_terminal_replay_rejects_corrupt_full_source(native, damage: str) -> None:
    store, engine = native
    run, started = records()
    ended = terminal(run)
    await managed_commit(store, run=run, events=(started, ended))
    table = build_step_sql_metadata().tables["ai_step_events"]
    values = {"outcome": {"outcome": "interrupted"}, "metadata": {"metadata": {"corrupt": "yes"}, "run_fields": 7},
              "hash": {"body_sha256": None}, "both_hashes": {"body_sha256": None, "fields_sha256": None},
              "start_outcome": {"outcome": "succeeded"}}[damage]
    async with engine.begin() as connection:
        await connection.execute(update(table).where(table.c.kind == ("run_started" if damage == "start_outcome" else "run_completed")).values(**values))
    cold = SqlStepStore(engine, namespace="native", tenant_id="one")
    try:
        with pytest.raises(ValueError):
            await cold.list_events(run_id=run.run_id)
        with pytest.raises(ValueError):
            await managed_commit(cold, run=run, events=(started, ended))
    finally:
        await cold.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancel", "shortcircuit"))
async def test_owned_runtime_terminal_is_native_with_only_trace_position(
    tmp_path: Path, outcome: str, record_property,
) -> None:
    async with _runtime_start(tmp_path, outcome=outcome, metadata_bytes=65537, trace_sync=True) as (state, native, run, traffic):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        sources = await native.list_events(run_id=run.run_id)
        ended, = [item for item in sources if isinstance(item, NativeLifecycleEvent)]
        assert ended.outcome == {"success": "succeeded", "shortcircuit": "succeeded",
                                 "failure": "failed", "cancel": "cancelled"}[outcome]
        trace = await archive.list_events(agent_run_id=run.run_id)
        projected, = [item for item in trace if item.event_type in {"AGENT_RUN_SUCCEEDED", "AGENT_RUN_INTERRUPTED", "AGENT_RUN_FAILED"}]
        assert projected == NativeEventReferences.project(ended, event_index=projected.event_index)
        raw = await _raw_events(archive, run.run_id)
        fact, = [item for item in raw if item.state == projected.event_type]
        assert fact.data["value"] == {"type": "native_event_reference", "payload": {
            "run": ended.run_id, "replay": ended.idempotency_key, "index": projected.event_index,
        }}
        assert [item.event_index for item in trace] == list(range(len(trace)))
        assert [item.sequence for item in raw] == list(range(1, len(raw) + 1))
        traffic.report(record_property, outcome)
        record_property("terminal_sidecar_logical_json_bytes", len(canonical_json_bytes(dict(fact.data))))
        record_property("terminal_former_inline_logical_json_bytes", len(canonical_json_bytes(_encode_step_envelope(projected))))
        record_property("terminal_native_proof_ascii_bytes", 128)


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("none", "missing", "outcome", "metadata", "hash"))
async def test_terminal_seal_unknown_readback_checks_source_in_existing_run_read(
    tmp_path: Path, damage: str, record_property,
) -> None:
    async with _shared_candidate_source(tmp_path) as (state, native, run, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        from pydantic_ai_harness.step_persistence import StepEvent

        started = StepEvent(run_id=run.run_id, kind="run_started", step_index=0,
            timestamp=run.started_at, conversation_id=run.conversation_id, parent_run_id=run.parent_run_id,
            agent_name=run.agent_name, metadata=dict(run.metadata), idempotency_key="0:0:run_started:")
        start_projection = NativeEventReferences.project(started, event_index=0)
        await archive.stage_native_event(run, started, start_projection)
        prepared = await native.prepare_step(run=run, events=(started,), managed_start=True)
        receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, prepared))
        native.confirm_prepared_step(receipt)
        archive.confirm_native_event_source(receipt)
        projected_run = OfficialStepReader.project_run(run)
        await archive.sync_projection(projected_run, events=(start_projection,), checkpoints=(), execution_id="execution")
        archive.confirm_native_event_projection((start_projection,))
        ended = terminal(run)
        projected = NativeEventReferences.project(ended, event_index=1)
        await archive.stage_native_event(run, ended, projected)
        await archive.sync_projection(projected_run, events=(projected,), checkpoints=(), execution_id="execution")
        head = await archive.execution_history_head_record(run.run_id)
        projection = PreparedExecutionProjection(run=projected_run, events=(projected,), checkpoints=(),
            base_event_offset=1, base_checkpoint_offset=0, target_event_offset=head.event_count,
            target_checkpoint_offset=head.checkpoint_count, target_transcript_message_count=head.transcript_message_count,
            durable_projection_digest=head.projection_digest, target_interaction_offset=head.interaction_count)
        engine = native.storage_context.engine
        baseline = _Traffic()
        event.listen(engine.sync_engine, "before_cursor_execute", baseline.capture)
        try:
            assert await archive.verify_execution_projection_head(projection)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", baseline.capture)
        if damage != "none":
            table = build_step_sql_metadata().tables["ai_step_events"]
            statement = delete(table) if damage == "missing" else update(table).values(**{
                "outcome": {"outcome": "interrupted"}, "metadata": {"run_fields": 7, "metadata": {"corrupt": "yes"}},
                "hash": {"body_sha256": None},
            }[damage])
            async with engine.begin() as connection:
                await connection.execute(statement.where(table.c.kind == "run_completed"))
        checked = _Traffic()
        event.listen(engine.sync_engine, "before_cursor_execute", checked.capture)
        try:
            if damage == "none":
                assert await archive.verify_execution_projection_head(projection)
            else:
                with pytest.raises(AIError) as rejected:
                    await archive.verify_execution_projection_head(projection)
                assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", checked.capture)
        checked.report(record_property, "terminal_seal_unknown")
        assert checked.statements["SELECT"] <= baseline.statements["SELECT"]
        assert "JOIN ai_step_events AS terminal_source" in checked.selects[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("dynamic", (False, True))
async def test_resolved_error_transform_keeps_original_native_and_trace_observations(
    tmp_path: Path, dynamic: bool,
) -> None:
    from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
    from linktools.ai.capability import TraceCapability
    from linktools.ai.capability.steps import AtomicStepPersistence

    observed = []

    class TransformError(AbstractCapability):
        id = "terminal-error-transform"

        def get_ordering(self):
            return CapabilityOrdering(position="innermost", wraps=(TraceCapability,), wrapped_by=(AtomicStepPersistence,))

        async def on_run_error(self, ctx, *, error):
            observed.append(repr(error))
            raise RuntimeError("transformed terminal diagnostic") from error

    class PerRunReplacement(AbstractCapability):
        id = "terminal-error-transform"

        async def for_run(self, ctx):
            return TransformError()

    extra = PerRunReplacement() if dynamic else TransformError()
    async with _runtime_start(tmp_path, outcome="failure", trace_sync=True,
                              extra_capabilities=(extra,)) as (state, native, run, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        trace = await archive.list_events(agent_run_id=run.run_id)
        sources = await native.list_events(run_id=run.run_id)
        failures = [item for item in sources if item.kind == "run_failed"]
        trace_failures = [item for item in trace if item.event_type == "AGENT_RUN_FAILED"]
        assert len(failures) == len(trace_failures) == len(observed) == 1
        assert trace_failures[0].error == observed[0]
        assert failures[0].error == repr(RuntimeError("transformed terminal diagnostic"))
        assert not any(isinstance(item, NativeLifecycleEvent) for item in sources)
        raw = await _raw_events(archive, run.run_id)
        assert next(item for item in raw if item.state == "AGENT_RUN_FAILED").data["value"]["type"] != "native_event_reference"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("succeeded", "interrupted", "failed", "cancelled"))
async def test_native_terminal_source_server_read_contract(server_metrics, outcome: str, record_property) -> None:
    from linktools.ai.migrate import provision_database

    _, engine, _ = server_metrics
    await provision_database(engine)
    namespace = "terminal-source-" + outcome
    store = SqlStepStore(engine, namespace=namespace, tenant_id="one")
    cold = SqlStepStore(engine, namespace=namespace, tenant_id="one")
    run, started = records()
    ended = terminal(run, outcome=outcome)
    try:
        await managed_commit(store, run=run, events=(started, ended))
        traffic = _Traffic()
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            receipt = await managed_commit(cold, run=run, events=(started, ended))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        assert cold.events_from_receipt(receipt) == (started, ended)
        assert await cold.list_events(run_id=run.run_id) == [started, ended]
        assert traffic.statements["SELECT"] == (2 if engine.dialect.name == "mysql" else 1)
        assert not any(sql.startswith("INSERT INTO ai_step_events ") for sql in traffic.sql)
        traffic.report(record_property, engine.dialect.name + "_cold_terminal")
        record_property("joined_source_rows", 1)
    finally:
        await cold.close()
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancel", "shortcircuit", "deferred"))
async def test_runtime_terminal_release_drops_active_registration_receipt(tmp_path: Path, outcome: str) -> None:
    async with _runtime_start(tmp_path, outcome=outcome, yield_while_attached=True,
                              observe_without_wait=True) as (state, native, run, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        assert archive.native_event_receipt(run.run_id) is None
        if outcome == "deferred":
            ended, = [item for item in await native.list_events(run_id=run.run_id)
                      if isinstance(item, NativeLifecycleEvent)]
            assert ended.outcome == "interrupted"
            assert any(item.event_type == "AGENT_RUN_INTERRUPTED"
                       for item in await archive.list_events(agent_run_id=run.run_id))


@pytest.mark.asyncio
async def test_managed_terminal_keeps_mixed_native_event_order(native) -> None:
    from pydantic_ai_harness.step_persistence import StepEvent

    store, _ = native
    run, started = records()
    ended = terminal(run)
    later = StepEvent(run_id=run.run_id, kind="run_failed", step_index=3,
                      error="later independent observation", idempotency_key="later")
    await managed_commit(store, run=run, events=(started, ended, later))
    assert await store.list_events(run_id=run.run_id) == [started, ended, later]


@pytest.mark.asyncio
async def test_native_terminal_unknown_rejects_changed_registration_in_one_select(native, record_property) -> None:
    store, engine = native
    run, started = records()
    ended = terminal(run)
    receipt = await managed_commit(store, run=run, events=(started, ended))
    table = build_step_sql_metadata().tables["ai_step_runs"]
    async with engine.begin() as connection:
        await connection.execute(update(table).values(registration_id="changed-registration"))
    traffic = _Traffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        with pytest.raises(ValueError, match="registration differs"):
            await store.list_events(run_id=run.run_id, started_receipt=receipt)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
    assert traffic.statements["SELECT"] == 1
    traffic.report(record_property, "native_terminal_registration_unknown")


@pytest.mark.asyncio
@pytest.mark.parametrize(("native_first", "failed_echo"), ((False, False), (True, False), (False, True)))
async def test_terminal_publishers_reuse_one_source_without_second_write(
    tmp_path: Path, native_first: bool, failed_echo: bool, record_property,
) -> None:
    from pydantic_ai_harness.step_persistence import StepEvent

    async with _shared_candidate_source(tmp_path) as (state, native, run, _):
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        started = StepEvent(run_id=run.run_id, kind="run_started", step_index=0,
            timestamp=run.started_at, conversation_id=run.conversation_id, parent_run_id=run.parent_run_id,
            agent_name=run.agent_name, metadata=dict(run.metadata), idempotency_key="0:0:run_started:")
        first = NativeEventReferences.project(started, event_index=0)
        await archive.stage_native_event(run, started, first)
        prepared = await native.prepare_step(run=run, events=(started,), managed_start=True)
        receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, prepared))
        native.confirm_prepared_step(receipt)
        archive.confirm_native_event_source(receipt)
        projected_run = OfficialStepReader.project_run(run)
        await archive.sync_projection(projected_run, events=(first,), checkpoints=(), execution_id="execution")
        archive.confirm_native_event_projection((first,))
        # The active source receipt must work independently of the global LRU.
        native._confirmed_runs.clear()
        ended = terminal(run, outcome="interrupted")
        projected = NativeEventReferences.project(ended, event_index=1)
        prepared = await native.prepare_step(events=(ended,))
        if native_first:
            receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(
                tx, prepared, confirmed=archive.native_event_receipt(run.run_id)))
            native.confirm_prepared_step(receipt)
            archive.confirm_native_event_source(receipt)
        await archive.stage_native_event(run, ended, projected)
        archive.release_native_event_receipt(run.run_id)
        assert archive.native_event_receipt(run.run_id) is not None
        traffic = _Traffic()
        engine = native.storage_context.engine
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            await archive.sync_projection(projected_run, events=(projected,), checkpoints=(), execution_id="execution")
            receipts = archive.confirm_native_event_projection((projected,))
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        assert sum(sql.startswith("INSERT INTO ai_step_events ") for sql in traffic.sql) == (0 if native_first else 1)
        # Trace retains its ordinary owner/head reads, but no separate native
        # event proof read is added when its bounded registration cache is cold.
        assert not any(sql.lstrip().startswith("SELECT") and "FROM ai_step_events" in sql for sql in traffic.sql)
        traffic.report(record_property, "terminal_trace_publication")
        if failed_echo:
            # Only the Trace transaction committed; the native publisher ended
            # in a definite failure and cannot acknowledge after quiescence.
            archive.release_native_event_receipt(run.run_id)
            assert archive.native_event_receipt(run.run_id) is None
        if not native_first and not failed_echo:
            # A start replay may carry inherited terminal proof without having
            # observed that terminal. It must not consume the native echo slot.
            observed_b = StepEvent(run_id=run.run_id, kind="run_failed", step_index=3,
                error="distinct native observation", idempotency_key="native-b")
            replayed = await native.prepare_step(run=run, events=(started, observed_b), managed_start=True)
            echo = await archive.state_store.mutate(lambda tx: native.write_prepared_step(
                tx, replayed, confirmed=receipts[0]))
            assert native.events_from_receipt(echo) == (started, ended)
            assert echo.lifecycle_keys == (started.idempotency_key,)
            archive.confirm_native_event_source(echo)
            assert archive._native_events.pending_source(run.run_id)[2] == ended
            second = _Traffic()
            event.listen(engine.sync_engine, "before_cursor_execute", second.capture)
            try:
                receipt = await archive.state_store.mutate(lambda tx: native.write_prepared_step(tx, prepared, confirmed=receipts[0]))
                native.confirm_prepared_step(receipt)
                archive.confirm_native_event_source(receipt)
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", second.capture)
            assert all(second.statements[verb] == 0 for verb in ("SELECT", "INSERT", "UPDATE", "DELETE"))
        expected = [started, ended] if native_first or failed_echo else [started, ended, observed_b]
        assert await native.list_events(run_id=run.run_id) == expected
        assert await archive.list_events(agent_run_id=run.run_id) == [first, projected]
        archive.release_native_event_receipt(run.run_id)
        assert archive.native_event_receipt(run.run_id) is None


@pytest.mark.asyncio
async def test_shared_diagnostic_is_produced_once_and_later_native_key_keeps_its_sequence() -> None:
    from datetime import datetime, timezone
    from pydantic_ai.capabilities import CombinedCapability
    from pydantic_ai_harness.step_persistence import InMemoryStepStore
    from linktools.ai.capability import TraceCapability
    from linktools.ai.capability.steps import AtomicStepPersistence, StepRunRegistration
    from linktools.ai.runtime._step_store import RuntimeStepStore
    from .test_atomic_step_capability import _context

    class DiagnosticError(RuntimeError):
        calls = 0

        def __repr__(self) -> str:
            self.calls += 1
            return f"diagnostic-{self.calls}"

    original = DiagnosticError()
    registration = StepRunRegistration(run_id="run", parent_run_id=None, agent_name=None, metadata={}, share_started=True,
        lifecycle_participant_types=(AtomicStepPersistence, TraceCapability),
        lifecycle_observer_types=(AtomicStepPersistence, TraceCapability))

    async def seed() -> tuple[str, datetime]:
        return "registration", datetime.now(timezone.utc)

    registration.bind_registration(seed)
    source = InMemoryStepStore()
    capability = AtomicStepPersistence(store=RuntimeStepStore(source, InMemoryStepStore()), run_id="run", run_registration=registration)

    async def emit(_event) -> None:
        raise AssertionError("no snapshot is expected")

    ctx = _context([], emit)
    ctx.in_durable_context = False
    ctx.root_capability = CombinedCapability([capability, TraceCapability()])
    await registration.prepare_started(conversation_id=ctx.conversation_id, step_index=ctx.run_step)
    registration.observe_trace_start()
    await registration.prepare_terminal(context=ctx, conversation_id=ctx.conversation_id, step_index=ctx.run_step,
        publisher="trace", outcome="failed", error=original)
    for _ in range(2):
        with pytest.raises(DiagnosticError):
            await capability.on_run_error(ctx, error=original)
    events = await source.list_events(run_id="run")
    assert [item.kind for item in events] == ["run_started", "run_failed", "run_failed"]
    assert events[1].error == "diagnostic-1" and events[2].error == "diagnostic-2"
    assert events[2].idempotency_key == f"2:{ctx.run_step}:run_failed:"
    assert original.calls == 2
