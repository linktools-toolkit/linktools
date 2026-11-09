#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small history pages do not materialize an entire run's event facts."""

from linktools.ai.persistence import Persistence
import json
import weakref
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event as sql_event
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import ExecutionStatus, agent_conversation_id, agent_run_id, canonical_json_bytes
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.observe.journal import MESSAGE_SEQ_METADATA_KEY, MODEL_REQUEST_SEQ_METADATA_KEY
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import _sql
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._memory_transaction import _MemoryOverlay, _MemoryTransaction
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence._step_contracts import AgentRunRecord, StepEvent
from linktools.ai.persistence.state import FactQuery, StoredFact, StoredRecord

from .test_history_request_association import _history
from .test_history_projection_conformance import _record


async def _seed_trace(storage: Persistence, count: int, *, reverse_timestamps: bool = False):
    await storage.initialize(namespace="history", tenant_id="tenant")
    await storage.execution.executions.create_execution(_record(ExecutionStatus.STARTED, 1))
    run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="execution", agent_run_seq=1)
    run = AgentRunRecord(
        run_id, agent_id="default", metadata={"agent_run_seq": "1"},
        agent_conversation_id=agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="execution"),
    )
    await storage.run_store.register_agent_run(run, execution_id="execution")
    timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
    for index in range(count):
        await storage.run_store.append_event(StepEvent(
            run_id, "TOOL_CALL_STARTED", index + 1,
            timestamp=timestamp + timedelta(seconds=count - index) if reverse_timestamps else timestamp,
            tool_call_id=f"call-{index}", tool_name="tool", event_index=index,
            idempotency_key=f"event-{index}",
        ), execution_id="execution")
    await storage.run_store.flush_execution_projection(run_id, execution_id="execution")
    return storage.run_store.read_store(PersistenceDomain.TRACE), run_id


async def _append_backdated_trace(storage: Persistence, run_id: str, count: int) -> None:
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
        storage = Persistence(PersistenceStorage.in_memory())
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
        storage = Persistence(PersistenceStorage.sql(engine))
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
@pytest.mark.parametrize("page_size", (32, 64, 129))
@pytest.mark.parametrize("reverse_timestamps", (False, True))
async def test_trace_pages_batch_fact_reads_without_overfetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_size: int, reverse_timestamps: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'page.db'}")
    await provision_database(engine)
    storage = Persistence(PersistenceStorage.sql(engine))
    try:
        archive, run_id = await _seed_trace(storage, page_size * 2, reverse_timestamps=reverse_timestamps)
        batches: list[int] = []
        original = _SqlTransaction.list_facts
        original_fact = _sql._fact_from_row
        alive = maximum_alive = 0
        fact_selects = logical_json_bytes = 0

        class TrackedFact(StoredFact):
            pass

        def released() -> None:
            nonlocal alive
            alive -= 1

        def materialize(row):
            nonlocal alive, maximum_alive, logical_json_bytes
            value = original_fact(row)
            logical_json_bytes += len(canonical_json_bytes(value.data))
            tracked = TrackedFact(value.stream_digest, value.sequence, value.owner_key_digest,
                                  value.kind, value.subject_digest, value.state, value.data)
            alive += 1
            maximum_alive = max(maximum_alive, alive)
            weakref.finalize(tracked, released)
            return tracked

        async def capture(self, query):
            rows = await original(self, query)
            batches.append(len(rows))
            return rows

        def capture_sql(_connection, _cursor, statement, _parameters, _context, _many):
            nonlocal fact_selects
            if statement.lstrip().upper().startswith("SELECT") and "FROM AI_STATE_FACTS" in statement.upper():
                fact_selects += 1

        monkeypatch.setattr(_SqlTransaction, "list_facts", capture)
        monkeypatch.setattr(_sql, "_fact_from_row", materialize)
        sql_event.listen(engine.sync_engine, "before_cursor_execute", capture_sql)
        try:
            rows = await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=page_size * 2,
                after_timestamp=None, after_sequence=0, limit=page_size,
            )
        finally:
            sql_event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        expected = (range(page_size * 2, page_size, -1) if reverse_timestamps else range(1, page_size + 1))
        assert [sequence for sequence, _ in rows] == list(expected)
        assert sum(batches) == page_size, "normal pages must not read unselected event facts"
        print("TRACE_SELECTED_FACTS " + json.dumps({"page_size": page_size, "reverse": reverse_timestamps,
              "fact_selects": fact_selects, "fact_batches": len(batches), "fact_rows": sum(batches),
              "logical_json_bytes": logical_json_bytes, "peak_raw_facts": maximum_alive}))
        assert fact_selects == (page_size + 31) // 32, "selected consecutive facts must batch regardless of timestamp order"
        assert len(batches) == fact_selects
        assert maximum_alive <= 32, "raw facts from previous windows must not survive the next source read"
        assert alive == 0
    finally:
        await storage.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_ordered_sparse_trace_pages_keep_contiguous_fact_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "sparse-page.db"))
    try:
        archive, run_id = await _seed_trace(storage, 0)
        timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
        for sequence in range(1, 97):
            traced = sequence not in {32, 65}
            await storage.run_store.append_event(StepEvent(
                run_id, "TOOL_CALL_STARTED" if traced else "AGENT_RUN_STARTED", sequence,
                timestamp=timestamp, tool_call_id=f"call-{sequence}" if traced else None,
                tool_name="tool" if traced else None,
            ), execution_id="execution")
        await storage.run_store.flush_execution_projection(run_id, execution_id="execution")
        batches: list[tuple[int, ...]] = []
        original = _SqlTransaction.list_facts

        async def capture(transaction, query):
            facts = await original(transaction, query)
            batches.append(tuple(fact.sequence for fact in facts))
            return facts

        monkeypatch.setattr(_SqlTransaction, "list_facts", capture)
        rows = await archive.list_trace_events(
            agent_run_id=run_id, event_high_water=96,
            after_timestamp=None, after_sequence=0, limit=100,
        )
        expected = [sequence for sequence in range(1, 97) if sequence not in {32, 65}]
        assert [sequence for sequence, _ in rows] == expected
        assert [sequence for batch in batches for sequence in batch] == expected
        assert len(batches) == 3
    finally:
        await storage.close()


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
        storage = Persistence(PersistenceStorage.in_memory())
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


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_count,sparse", ((32, False), (65, False), (65, True)))
async def test_history_associations_batch_only_selected_source_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected_count: int, sparse: bool,
) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "association-page.db"))
    try:
        archive, run_id = await _seed_trace(storage, selected_count * 2)
        selected = tuple(range(1, selected_count * 2, 2)) if sparse else tuple(range(1, selected_count + 1))
        batches: list[tuple[int, ...]] = []
        original = _SqlTransaction.list_facts
        fact_selects = 0

        async def capture(transaction, query):
            facts = await original(transaction, query)
            batches.append(tuple(fact.sequence for fact in facts))
            return facts

        def capture_sql(_connection, _cursor, statement, _parameters, _context, _many):
            nonlocal fact_selects
            if statement.lstrip().upper().startswith("SELECT") and "FROM AI_STATE_FACTS" in statement.upper():
                fact_selects += 1

        monkeypatch.setattr(_SqlTransaction, "list_facts", capture)
        engine = archive.state_store.storage_group.context.engine.sync_engine
        sql_event.listen(engine, "before_cursor_execute", capture_sql)
        try:
            values = await archive.read_history_associations(
                agent_run_id=run_id, message_seqs=(),
                tool_call_ids=tuple(f"call-{sequence - 1}" for sequence in reversed(selected)),
                event_high_water=selected[-1],
            )
        finally:
            sql_event.remove(engine, "before_cursor_execute", capture_sql)
        assert [value.tool_call_id for value in values] == [f"call-{sequence - 1}" for sequence in selected]
        assert tuple(sequence for batch in batches for sequence in batch) == selected
        assert all(len(batch) <= 32 for batch in batches)
        assert fact_selects <= (selected_count if sparse else (selected_count + 31) // 32)
        print("HISTORY_SELECTED_FACTS " + json.dumps({"selected_count": selected_count, "sparse": sparse,
              "fact_selects": fact_selects, "fact_rows": sum(map(len, batches))}))
    finally:
        await storage.close()
