#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native writers, readers and portable state share the resource ID bound."""

from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, StepEvent
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import RESOURCE_ID_MAX_BYTES, agent_conversation_id, agent_run_id, validate_resource_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.persistence._step_backup import NativeStepBackup
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord, StepEvent as TraceEvent


@pytest_asyncio.fixture
async def native(tmp_path: Path) -> AsyncIterator[SqlStepStore]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ids.db'}")
    store = SqlStepStore(engine, namespace="ids", tenant_id="tenant")
    await store.initialize(local=True)
    try:
        yield store
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.parametrize("value", ("x" * 129, "😀" * 32 + "x", "", " leading", "trailing\u3000", "a\x00b", "\ud800"))
def test_resource_id_rejects_invalid_bytes_without_normalizing(value: str) -> None:
    with pytest.raises(AIError) as caught:
        validate_resource_id(value)
    assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_native_id_boundaries_and_unicode_round_trip_exactly(native: SqlStepStore) -> None:
    identifiers = ("x" * RESOURCE_ID_MAX_BYTES, "😀" * 32, "café", "cafe\u0301", "a b", "A B")
    for value in identifiers:
        assert validate_resource_id(value) == value
        run = RunRecord(run_id=value, conversation_id=value, parent_run_id=value)
        event = StepEvent(run_id=value, conversation_id=value, parent_run_id=value,
                          kind="run_started", step_index=0, idempotency_key="same-key")
        snapshot = ContinuableSnapshot(run_id=value, conversation_id=value, parent_run_id=value,
                                        step_index=0, messages=[], idempotency_key="same-key")
        await native.commit_step(run=run, events=(event,), snapshot=snapshot)
        assert await native.get_run(run_id=value) == run
        assert await native.list_runs(conversation_id=value, parent_run_id=value) == [run]
        assert await native.list_events(run_id=value) == [event]
        assert await native.latest_snapshot(run_id=value) == snapshot
    assert {run.run_id for run in await native.list_runs()} == set(identifiers)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("run_id", "conversation_id", "parent_run_id"))
async def test_native_writers_reject_oversized_identity_before_publishing(native: SqlStepStore, field: str) -> None:
    invalid = {field: "😀" * 32 + "x"}
    run = replace(RunRecord(run_id="run"), **invalid)
    event = replace(StepEvent(run_id="run", kind="run_started", step_index=0), **invalid)
    snapshot = replace(ContinuableSnapshot(run_id="run", step_index=0, messages=[]), **invalid)
    for operation in (
        lambda: native.register_run(run),
        lambda: native.append_event(event),
        lambda: native.save_snapshot(snapshot),
        lambda: native.commit_step(run=run, events=(event,), snapshot=snapshot),
    ):
        with pytest.raises(AIError):
            await operation()
    assert await native.list_runs() == []
    assert await native.list_events(run_id="run") == []
    assert await native.list_snapshots(run_id="run") == []


@pytest.mark.asyncio
async def test_native_queries_and_official_reader_reject_oversized_ids(native: SqlStepStore) -> None:
    value = "x" * (RESOURCE_ID_MAX_BYTES + 1)
    reader = OfficialStepReader(InMemoryStepStore())
    for operation in (
        lambda: native.get_run(run_id=value),
        lambda: native.list_runs(conversation_id=value),
        lambda: native.list_runs(parent_run_id=value),
        lambda: native.list_events(run_id=value),
        lambda: native.latest_snapshot(run_id=value),
        lambda: native.list_snapshots(run_id=value),
        lambda: reader.get_run(run_id=value),
        lambda: reader.list_agent_runs(agent_conversation_id=value),
        lambda: reader.list_agent_runs(parent_agent_run_id=value),
        lambda: reader.list_events(agent_run_id=value),
        lambda: reader.latest_snapshot(run_id=value),
    ):
        with pytest.raises(AIError):
            await operation()


@pytest.mark.parametrize("field", ("run_id", "conversation_id", "parent_run_id"))
def test_portable_runs_and_runtime_projections_reject_oversized_ids(field: str) -> None:
    invalid = "x" * (RESOURCE_ID_MAX_BYTES + 1)
    record = replace(RunRecord(run_id="run"), **{field: invalid})
    with pytest.raises(AIError):
        OfficialStepReader.project_run(record)
    with pytest.raises(AIError):
        OfficialStepReader.restore_snapshot(replace(ContinuableSnapshot(run_id="run", step_index=0, messages=[]),
                                                   **{field: invalid}))
    projected_field = {"run_id": "agent_run_id", "conversation_id": "agent_conversation_id",
                       "parent_run_id": "parent_agent_run_id"}[field]
    for constructor, values in (
        (AgentRunRecord, {"agent_run_id": "run"}),
        (TraceEvent, {"agent_run_id": "run", "event_type": "AGENT_RUN_STARTED", "step_index": 0}),
        (AgentRunCheckpoint, {"agent_run_id": "run", "step_index": 0, "messages": []}),
    ):
        with pytest.raises(AIError):
            constructor(**(values | {projected_field: invalid}))
    scope = SqlStepStore.scope_digest(namespace="ids", tenant_id="tenant")
    row = SqlStepStore._record_values(RunRecord(run_id="run")) | {"store_digest": scope, field: invalid}
    with pytest.raises(AIError):
        SqlStepStore.validate_backup(NativeStepBackup(scope, (row,), (), ()))


def test_runtime_id_generators_fit_the_native_resource_contract() -> None:
    execution = "😀" * 32
    values = dict(namespace="runtime", tenant_id="tenant", execution_id=execution)
    conversation = agent_conversation_id(**values)
    run = agent_run_id(**values, agent_run_seq=1)
    for generated in (conversation, run):
        assert len(generated.encode("utf-8")) == 66
        assert validate_resource_id(generated) == generated
    assert conversation == agent_conversation_id(**values)
    assert run != agent_run_id(**values, agent_run_seq=2)
    with pytest.raises(AIError):
        agent_conversation_id(**(values | {"execution_id": execution + "x"}))
    with pytest.raises(AIError):
        agent_run_id(**(values | {"execution_id": execution + "x"}), agent_run_seq=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("table", ("runs", "events", "snapshots"))
async def test_native_sql_readers_reject_ids_beyond_the_byte_contract(native: SqlStepStore, table: str) -> None:
    await native.register_run(RunRecord(run_id="run"))
    await native.append_event(StepEvent(run_id="run", conversation_id="source", kind="run_started", step_index=0))
    await native.save_snapshot(ContinuableSnapshot(run_id="run", conversation_id="source", step_index=0, messages=[]))
    async with native.storage_context.engine.begin() as connection:
        await connection.execute(text(f"UPDATE ai_step_{table} SET conversation_id = :value"),
                                 {"value": "😀" * 32 + "x"})
    operation = {"runs": native.get_run, "events": native.list_events, "snapshots": native.latest_snapshot}[table]
    with pytest.raises(AIError):
        await operation(run_id="run")
