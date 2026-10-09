#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small history pages do not materialize an entire run's event facts."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event as sql_event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import ExecutionStatus, agent_conversation_id, agent_run_id
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.runtime._journal import MESSAGE_SEQ_METADATA_KEY, MODEL_REQUEST_SEQ_METADATA_KEY
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._memory import InMemoryStateStore
from linktools.ai.runtime.state._memory_transaction import _MemoryOverlay, _MemoryTransaction
from linktools.ai.runtime.state._step_contracts import AgentRunRecord, StepEvent
from linktools.ai.runtime.state._store import FactQuery, StoredFact, StoredRecord

from .test_history_request_association import _history
from .test_history_projection_conformance import _record


async def _seed_trace(storage: RuntimeStorage, count: int):
    await storage.initialize(namespace="history", tenant_id="tenant")
    await storage.execution.executions.create_with_history_head(_record(ExecutionStatus.STARTED, 1))
    run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="execution", agent_run_seq=1)
    run = AgentRunRecord(
        run_id, agent_id="default", metadata={"agent_run_seq": "1"},
        agent_conversation_id=agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="execution"),
    )
    await storage.run_store.register_agent_run(run, execution_id="execution")
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
    for index in range(count):
        await storage.run_store.append_event(StepEvent(
            run_id, "TOOL_CALL_STARTED", index + 1, timestamp=timestamp,
            tool_call_id=f"call-{index}", tool_name="tool", event_index=index,
            idempotency_key=f"event-{index}",
        ), execution_id="execution")
    await storage.run_store.flush_execution_projection(run_id, execution_id="execution")
    return storage.run_store.read_store(RuntimeDomain.EXECUTION), run_id


async def _append_backdated_trace(storage: RuntimeStorage, run_id: str, count: int) -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        await storage.run_store.append_event(StepEvent(
            run_id, "TOOL_CALL_STARTED", index + 2,
            timestamp=timestamp + timedelta(microseconds=index),
            tool_call_id=f"late-{index}", tool_name="tool", event_index=index + 1,
            idempotency_key=f"late-{index}",
        ), execution_id="execution")
    await storage.run_store.flush_execution_projection(run_id, execution_id="execution")


@pytest.mark.asyncio
@pytest.mark.parametrize("message_count", (32, 256))
async def test_small_history_and_trace_pages_bound_materialized_rows(
    monkeypatch: pytest.MonkeyPatch,
    message_count: int,
) -> None:
    async with _history() as history:
        recorder, = history.recorders
        for index in range(message_count):
            recorder.append_transcript_message(ModelResponse(parts=[TextPart(f"response {index}")]))
            await recorder.record_event(
                "MODEL_REQUEST_SUCCEEDED", index + 1,
                metadata={
                    MESSAGE_SEQ_METADATA_KEY: str(index + 1),
                    MODEL_REQUEST_SEQ_METADATA_KEY: str(index + 1),
                },
            )
        await history.state.run_store.flush_execution_projection(recorder.agent_run_id, execution_id="execution")
        materialized = 0
        for name in ("list_facts", "list_records", "get_records"):
            original = getattr(_MemoryTransaction, name)

            async def count_rows(self, *args, _original=original, **kwargs):
                nonlocal materialized
                rows = await _original(self, *args, **kwargs)
                materialized += len(rows)
                return rows

            monkeypatch.setattr(_MemoryTransaction, name, count_rows)
        for method, selectors in (
            (history.reader.history, {}),
            (history.reader.history, {"message_seq": message_count}),
            (history.reader.trace, {}),
        ):
            materialized = 0
            page = await method("execution", tenant_id="tenant", cursor=None, limit=1, **selectors)
            assert len(page.items) == 1
            assert materialized <= 128, "one small page must not fetch every step event"


@pytest.mark.asyncio
async def test_exact_memory_fact_lookup_does_not_enumerate_other_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStateStore()
    await store.initialize()
    owner, stream = b"o" * 32, b"s" * 32

    async def seed(transaction) -> None:
        await transaction.insert_record(StoredRecord(owner, None, None, "owner", "owner", None, 0, None, 0, None, {}))
        await transaction.insert_facts(tuple(
            StoredFact(stream, index, owner, "event", None, None, {"index": index})
            for index in (1, 2, 3, 5)
        ))

    await store.mutate(seed)
    examined = 0
    original = _MemoryOverlay.__iter__

    def iterate(self):
        nonlocal examined
        for key in original(self):
            examined += 1
            yield key

    monkeypatch.setattr(_MemoryOverlay, "__iter__", iterate)
    for query, expected in (
        (FactQuery(stream, after_sequence=1, limit=2), (2, 3)),
        (FactQuery(stream, after_sequence=4, limit=1), (5,)),
    ):
        result = await store.read(lambda tx: tx.list_facts(query))
        assert tuple(fact.sequence for fact in result) == expected
    assert examined == 0, "direct fact identities must not enumerate the entire fact map"
    sparse = await store.read(lambda tx: tx.list_facts(FactQuery(stream, after_sequence=3, limit=1)))
    assert tuple(fact.sequence for fact in sparse) == (5,)
    missing = await store.read(lambda tx: tx.list_facts(FactQuery(stream, after_sequence=5, limit=1)))
    assert missing == ()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ("no_match", "rare_match"))
async def test_selective_trace_examined_work_scales_linearly(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
) -> None:
    examined = 0
    measuring = False
    original = _MemoryOverlay.__iter__

    def iterate(self):
        nonlocal examined
        for key in original(self):
            if measuring:
                examined += 1
            yield key

    monkeypatch.setattr(_MemoryOverlay, "__iter__", iterate)
    costs = []
    for count in (32, 64, 128):
        async with _history() as history:
            recorder, = history.recorders
            for index in range(count):
                recorder.append_transcript_message(ModelResponse(parts=[TextPart(f"response {index}")]))
                await recorder.record_event("MODEL_REQUEST_SUCCEEDED", index + 1, metadata={
                    MESSAGE_SEQ_METADATA_KEY: str(index + 1),
                    MODEL_REQUEST_SEQ_METADATA_KEY: str(index + 1),
                })
            await history.state.run_store.flush_execution_projection(recorder.agent_run_id, execution_id="execution")
            examined = 0
            measuring = True
            filters = {"tool_call_id": "missing"} if selector == "no_match" else {"model_request_seq": count}
            page = await history.reader.trace("execution", tenant_id="tenant", cursor=None, limit=1, **filters)
            measuring = False
            assert len(page.items) == (0 if selector == "no_match" else 1)
            costs.append(examined)
    assert costs[1] <= costs[0] * 2.25
    assert costs[2] <= costs[1] * 2.25


@pytest.mark.asyncio
async def test_old_trace_cutoff_does_not_rescan_for_each_later_backdated_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _MemoryOverlay.__iter__
    examined = 0
    measuring = False

    def iterate(self):
        nonlocal examined
        for key in original(self):
            if measuring:
                examined += 1
            yield key

    monkeypatch.setattr(_MemoryOverlay, "__iter__", iterate)
    costs = []
    for count in (32, 64, 128):
        storage = RuntimeStorage.in_memory()
        try:
            archive, run_id = await _seed_trace(storage, 1)
            await _append_backdated_trace(storage, run_id, count)
            examined = 0
            measuring = True
            rows = await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=1,
                after_timestamp=None, after_sequence=0, limit=1,
            )
            measuring = False
            assert len(rows) == 1 and rows[0][0] == 1
            costs.append(examined)
        finally:
            await storage.close()
    assert costs[1] <= costs[0] * 2.25
    assert costs[2] <= costs[1] * 2.25


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("no_match", "old_cutoff"))
async def test_sql_trace_query_count_is_independent_of_examined_events(tmp_path: Path, scenario: str) -> None:
    counts = []
    for count in (32, 64, 128):
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'trace-{count}.db'}")
        await provision_database(engine)
        storage = RuntimeStorage.sql(engine)
        statements = []

        def capture(_connection, _cursor, statement, _parameters, _context, _executemany) -> None:
            statements.append(statement)

        try:
            archive, run_id = await _seed_trace(storage, count if scenario == "no_match" else 1)
            if scenario == "old_cutoff":
                await _append_backdated_trace(storage, run_id, count)
            sql_event.listen(engine.sync_engine, "before_cursor_execute", capture)
            rows = await archive.list_trace_events(
                agent_run_id=run_id,
                event_high_water=count if scenario == "no_match" else 1,
                after_timestamp=None, after_sequence=0, limit=1,
                **({"tool_call_id": "missing"} if scenario == "no_match" else {}),
            )
            sql_event.remove(engine.sync_engine, "before_cursor_execute", capture)
            assert len(rows) == (0 if scenario == "no_match" else 1)
            counts.append(len(statements))
        finally:
            await storage.close()
            await engine.dispose()
    assert max(counts) <= 12
    assert counts[1] <= counts[0] + 2
    assert counts[2] <= counts[0] + 2


@pytest.mark.asyncio
async def test_run_release_scans_owned_cleanup_data_once_per_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    original = _MemoryOverlay.__iter__
    examined = 0
    measuring = False

    def iterate(self):
        nonlocal examined
        for key in original(self):
            if measuring:
                examined += 1
            yield key

    monkeypatch.setattr(_MemoryOverlay, "__iter__", iterate)
    costs = []
    for count in (32, 64, 128):
        storage = RuntimeStorage.in_memory()
        try:
            archive, run_id = await _seed_trace(storage, count)
            examined = 0
            measuring = True
            await archive.release_agent_run(run_id, execution_id="execution")
            measuring = False
            costs.append(examined)
            assert await archive.get_agent_run(agent_run_id=run_id) is None
            assert await archive.list_events(agent_run_id=run_id) == []
        finally:
            await storage.close()
    assert costs[1] <= costs[0] * 2.25
    assert costs[2] <= costs[1] * 2.25
