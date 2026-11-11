#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One native registration supplies the owned Runtime run projection."""

import asyncio
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import time
import tracemalloc

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, StepEvent, StepPersistence
from sqlalchemy import event

from linktools.ai.capability import TraceCapability
from linktools.ai.capability.steps import StepRunRegistration
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
from linktools.ai.persistence.runs import OfficialStepReader
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._step_store import RuntimeStepStore
from linktools.ai.migrate import provision_database

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_native_run_references import _source
from .test_step_message_cache import _SqlTraffic


@pytest.mark.parametrize("native_first", (False, True))
def test_owned_registration_is_identical_regardless_of_hook_order(native_first: bool) -> None:
    registration = StepRunRegistration(
        run_id="run", parent_run_id="parent", agent_name="agent",
        metadata={"scope": "value", "linktools.transcript_start": "0"},
    )
    first = registration.prepare(conversation_id="conversation", registration_id="token" if native_first else None)
    native = registration.prepare(conversation_id="conversation", registration_id="token")
    runtime = OfficialStepReader.project_run(first)
    assert runtime == OfficialStepReader.project_run(native)
    assert native.registration_id == "token"
    first.metadata["scope"] = "mutated"
    assert registration.prepare(conversation_id="conversation") == native
    with pytest.raises(ValueError, match="conversation changed"):
        registration.prepare(conversation_id="different")
    with pytest.raises(ValueError, match="replay identity changed"):
        registration.prepare(conversation_id="conversation", registration_id="different")


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_size", (0, 4096))
async def test_runtime_native_producer_and_cold_trace_share_exact_registration(
    tmp_path: Path, metadata_size: int,
) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.persistence._native_run_reference import is_native_run_reference
    from linktools.ai.persistence._run_registration import RunRegistrationReference

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await provision_database(engine)
    native = SqlStepStore(engine, namespace="native-producer", tenant_id="default")
    observed = []
    features = RuntimeFeatures("native-producer")
    features.capability(StepPersistence(store=native, capture_frontier=True,
        metadata={} if not metadata_size else {"custom": "x" * metadata_size}), scope="runtime")
    features.capability(TraceCapability(commit=CommitPolicy.sync()), scope="runtime")
    features.agent("default", model="default", allow_tools=())

    async def provider(messages, info):
        source, = await native.list_runs()
        # Synchronous Trace publishes the shared registration before provider dispatch.
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        projected = await archive.get_agent_run(agent_run_id=source.run_id)
        assert projected == OfficialStepReader.project_run(source)
        assert projected.started_at.isoformat() == source.started_at.isoformat()
        assert set(source.metadata) - set(projected.metadata) == {
            "linktools.transcript_start", "linktools.transcript_run_id",
        }
        observed.append((source, projected))
        return ModelResponse(parts=[TextPart("done")])

    try:
        async with Persistence.open("native-producer", storage=PersistenceStorage.sql(engine)) as state:
            async with Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(provider)), features)) as runtime:
                execution = await runtime.agents.get().start("work", idempotency_key="work")
                result = await execution.wait(timeout_seconds=5)
                assert result.result.status is ExecutionStatus.SUCCEEDED, result.result
            source, projected = observed[0]
            cold = await state.create_agent_run_store()
            try:
                archive = cold.read_store(PersistenceDomain.TRACE)
                assert await archive.get_agent_run(agent_run_id=source.run_id) == projected
                rows = await archive.state_store.read(lambda tx: tx.scan_records())
                run = next(row for row in rows if row.kind == "agent_run")
                assert is_native_run_reference(run.data)
                reference = RunRegistrationReference.from_data(run.data)
                assert reference.metadata == {}
                assert reference.removed_metadata == ()
                assert reference.fields == {}
                assert reference.project(source) == projected
            finally:
                await cold.preflight_close()
                await cold.close()
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_cold_registration_replay_confirms_original_without_another_select(tmp_path: Path, record_property) -> None:
    async with _source(tmp_path, confirmed=False) as (state, native, source, _):
        await native.close()
        supplied = replace(source, metadata={"changed": "caller"}, agent_name="different",
                           started_at=source.started_at + timedelta(seconds=1))
        traffic = _SqlTraffic()
        engine = native.storage_context.engine
        adapter = RuntimeStepStore(native, object(), working_set=state.run_store)
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        started = time.process_time_ns()
        tracemalloc.start()
        try:
            await adapter.commit_step(run=supplied)
            elapsed = time.process_time_ns() - started
            _, peak = tracemalloc.get_traced_memory()
            assert native.confirmed_run(source.run_id) == source
            assert await state.run_store.checkpoint_persistence.reader.get_run(run_id=source.run_id) == source
            record_property("replay_selects", traffic.statements["SELECT"])
            record_property("replay_inserts", traffic.statements["INSERT"])
            record_property("replay_parameter_bytes", traffic.bound_bytes)
            returned_source_bytes = native._confirmed_runs[source.run_id].byte_size
            registration_only_bytes = len(source.registration_id.encode("utf-8"))
            record_property("replay_returned_source_bytes", returned_source_bytes)
            record_property("former_registration_only_payload_bytes", registration_only_bytes)
            record_property("replay_additional_source_payload_bytes", returned_source_bytes - registration_only_bytes)
            record_property("replay_cpu_ns", elapsed)
            record_property("replay_peak_traced_bytes", peak)
            assert traffic.statements["SELECT"] == 1
            assert traffic.statements["INSERT"] == 1
            assert traffic.statements["UPDATE"] == 0
            assert traffic.statements["DELETE"] == 0
            assert traffic.bound_bytes > 0
        finally:
            tracemalloc.stop()
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        supplied.metadata["changed"] = "mutated after commit"
        assert await state.run_store.checkpoint_persistence.reader.get_run(run_id=source.run_id) == source


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("rollback", "unknown"))
async def test_replay_source_is_not_confirmed_before_transaction_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    from sqlalchemy.ext.asyncio import AsyncSessionTransaction

    async with _source(tmp_path, confirmed=False) as (state, native, source, _):
        await native.close()
        original = AsyncSessionTransaction.__aexit__

        async def reject(self, error_type, error, traceback):
            if error_type is None:
                if failure == "rollback":
                    rejected = RuntimeError("replay rollback")
                    await original(self, RuntimeError, rejected, rejected.__traceback__)
                else:
                    await original(self, error_type, error, traceback)
                raise RuntimeError("replay acknowledgement unavailable")
            return await original(self, error_type, error, traceback)

        monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", reject)
        adapter = RuntimeStepStore(native, object(), working_set=state.run_store)
        with pytest.raises((RuntimeError, AIError)):
            await adapter.commit_step(run=replace(source, metadata={"caller": "different"}))
        assert native.confirmed_run(source.run_id) is None
        monkeypatch.setattr(AsyncSessionTransaction, "__aexit__", original)
        # The failed confirmation did not seed the Runtime projection with caller bytes.
        assert await state.run_store.checkpoint_persistence.reader.get_run(run_id=source.run_id) == source


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("split", "sql_oversized"))
async def test_committed_run_lineage_and_cached_snapshot_do_not_depend_on_shared_cache(
    tmp_path: Path, backend: str,
) -> None:
    from pydantic_ai.tools import ToolDefinition
    from ._runtime_test_helpers import tool_run_context
    from .test_native_tool_effect_owner import _bridge

    async with _source(
        tmp_path, confirmed=False, extra_metadata={"large": "x" * (1024 * 1024 + 1)},
    ) as (state, sql, source, _):
        native = InMemoryStepStore() if backend == "split" else sql
        if backend == "split":
            await native.register_run(source)
        if backend == "sql_oversized":
            assert sql.confirmed_run(source.run_id) is None
        bridge = _bridge(state)
        events = []

        async def admission(request, *, observed, event, known_new):
            await state.checkpoint.tools.admit(request)
            events.append(event)
            return await state.checkpoint.tools.observe_operation(request.tool_operation_id, tenant_id="tenant")

        async def terminal(*args, **kwargs):
            raise AssertionError("the test does not finish a tool")

        bridge.bind_boundaries(admission=admission, terminal=terminal)
        adapter = RuntimeStepStore(native, bridge, native_boundary=True,
            working_set=state.run_store if backend == "sql_oversized" else None)
        snapshot = ContinuableSnapshot(run_id=source.run_id, step_index=0, messages=[],
            idempotency_key="snapshot", state="complete")
        await adapter.commit_step(run=replace(source, parent_run_id="wrong", metadata={"caller": "changed"}),
                                  snapshot=snapshot)
        if backend == "sql_oversized":
            traffic = _SqlTraffic()
            engine = sql.storage_context.engine
            event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
            try:
                reader = state.run_store.checkpoint_persistence.reader
                assert await reader.get_run(run_id=source.run_id) == source
                assert await reader.latest_snapshot(run_id=source.run_id) == snapshot
                assert traffic.statements["SELECT"] == 0
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        ctx = replace(tool_run_context(), conversation_id=source.conversation_id)
        await bridge.begin(ctx, ToolCallPart("inspect", {}, tool_call_id="call"),
                           ToolDefinition(name="inspect"), {}, False)
        assert len(events) == 1
        assert events[0].parent_run_id == source.parent_run_id
        assert events[0].agent_name == source.agent_name
        assert events[0].metadata == source.metadata


@pytest.mark.asyncio
async def test_receipt_calls_preserve_public_overrides_and_isolate_nested_concurrent_writes(tmp_path: Path) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    calls = []

    class ObservedStore(SqlStepStore):
        async def register_run(self, record):
            calls.append(("register", record.run_id))
            if record.run_id == "rejected-register":
                raise RuntimeError("registration rejected")
            await super().register_run(record)

        async def commit_step(self, *, run=None, events=(), snapshot=None):
            calls.append(("boundary", None if run is None else run.run_id))
            if run is not None and run.run_id == "rejected-boundary":
                raise RuntimeError("boundary rejected")
            await asyncio.sleep(0)
            await super().commit_step(run=run, events=events, snapshot=snapshot)
            if run is not None:
                await self.register_run_receipt(replace(run, run_id=run.run_id + "-nested"))
            elif events:
                await self.register_run_receipt(RunRecord(run_id="nested-event-registration"))
            await super().commit_step()

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'overrides.db'}")
    native = ObservedStore(engine, namespace="scoped", tenant_id="one")
    other = SqlStepStore(engine, namespace="scoped", tenant_id="other")
    await native.initialize(local=True)
    try:
        adapter = RuntimeStepStore(native, object())
        with pytest.raises(RuntimeError, match="registration rejected"):
            await adapter.register_run(RunRecord(run_id="rejected-register"))
        with pytest.raises(RuntimeError, match="boundary rejected"):
            await adapter.commit_step(run=RunRecord(run_id="rejected-boundary"))
        assert await native.get_run(run_id="rejected-register") is None
        assert await native.get_run(run_id="rejected-boundary") is None
        records = [RunRecord(run_id=str(index), registration_id=str(index)) for index in range(2)]
        receipts = await asyncio.gather(*(native.commit_step_receipt(run=run) for run in records))
        assert [native.run_from_receipt(receipt) for receipt in receipts] == records
        event_receipt = await native.commit_step_receipt(events=(
            StepEvent(run_id=records[0].run_id, kind="run_started", step_index=0),
        ))
        assert native.run_from_receipt(event_receipt) is None
        assert await native.get_run(run_id="nested-event-registration") is not None
        for receipt in receipts:
            with pytest.raises(ValueError, match="another native store"):
                other.run_from_receipt(receipt)
        assert native._receipt_sink.get() is None
        await native.register_run(RunRecord(run_id="ordinary"))
        assert native._receipt_sink.get() is None
        assert ("register", "ordinary") in calls
        assert ("boundary", "rejected-boundary") in calls
    finally:
        await native.close()
        await other.close()
        await engine.dispose()
