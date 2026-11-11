#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small history pages do not materialize an entire run's event facts."""

from linktools.ai.persistence import Persistence
import json
import weakref
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event as sql_event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import ExecutionStatus, agent_conversation_id, agent_run_id, canonical_json_bytes
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.observe.journal import MESSAGE_SEQ_METADATA_KEY, MODEL_REQUEST_SEQ_METADATA_KEY
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence import _sql
from linktools.ai.persistence._trace_event_store import SqlTraceEventStore
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._memory_transaction import _MemoryOverlay, _MemoryTransaction
from linktools.ai.persistence._step_contracts import AgentRunRecord, StepEvent
from linktools.ai.persistence.state import FactQuery, StoredAlias, StoredFact, StoredRecord

from .test_history_request_association import _history
from .test_history_projection_conformance import _record


@contextmanager
def _capture_fact_batches(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[list[int]]]:
    """Observe physical rows per SQL read independently of the query route."""
    batches: list[list[int]] = []
    original = _sql._fact_from_row
    original_trace = SqlTraceEventStore.row_to_fact

    def capture_sql(_connection, _cursor, statement, _parameters, _context, _many) -> None:
        if statement.lstrip().upper().startswith("SELECT") and any(
            name in statement.upper() for name in ("FROM AI_STATE_FACTS", "FROM AI_TRACE_EVENTS")
        ):
            batches.append([])

    def materialize(row):
        fact = original(row)
        assert batches, "materialized facts must belong to an observed SQL read"
        batches[-1].append(fact.sequence)
        return fact

    def materialize_trace(store, row):
        fact = original_trace(store, row)
        assert batches, "materialized Trace facts must belong to an observed SQL read"
        batches[-1].append(fact.sequence)
        return fact

    with monkeypatch.context() as patch:
        patch.setattr(_sql, "_fact_from_row", materialize)
        patch.setattr(SqlTraceEventStore, "row_to_fact", materialize_trace)
        sql_event.listen(engine, "before_cursor_execute", capture_sql)
        try:
            yield batches
        finally:
            sql_event.remove(engine, "before_cursor_execute", capture_sql)


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
@pytest.mark.parametrize("message_count", (
    3,
    pytest.param(32, marks=pytest.mark.manual, id="scale-32"),
    pytest.param(256, marks=pytest.mark.manual, id="scale-256"),
))
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
        event_rows = 0
        for name in ("list_facts", "get_facts", "list_records", "get_records"):
            original = getattr(_MemoryTransaction, name)

            async def count_rows(self, *args, _original=original, **kwargs):
                nonlocal materialized, event_rows
                rows = await _original(self, *args, **kwargs)
                materialized += len(rows)
                event_rows += sum(isinstance(row, StoredFact) and row.kind == "step_event" for row in rows)
                return rows

            monkeypatch.setattr(_MemoryTransaction, name, count_rows)
        for method, selectors in (
            (history.reader.history, {}),
            (history.reader.history, {"message_seq": message_count}),
            (history.reader.trace, {}),
        ):
            materialized = 0
            event_rows = 0
            page = await method("execution", tenant_id="tenant", cursor=None, limit=1, **selectors)
            assert len(page.items) == 1
            assert event_rows <= 2, "only the selected item and cursor lookahead may read event facts"
            assert materialized <= 128, "one small page must not fetch every step event"


@pytest.mark.asyncio
async def test_exact_memory_fact_lookup_does_not_enumerate_other_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryStateStore()
    await store.initialize()
    owner, stream = b"o" * 32, b"s" * 32

    async def seed(transaction) -> None:
        await transaction.insert_record(StoredRecord(owner, None, None, "owner", b'owner', None, 0, None, 0, None, {}))
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
    selected = await store.read(lambda tx: tx.get_facts(stream, (5, 2, 5, 4)))
    assert tuple(fact.sequence for fact in selected) == (2, 5)
    assert examined == 0, "direct fact identities must not enumerate the entire fact map"
    sparse = await store.read(lambda tx: tx.list_facts(FactQuery(stream, after_sequence=3, limit=1)))
    assert tuple(fact.sequence for fact in sparse) == (5,)
    missing = await store.read(lambda tx: tx.list_facts(FactQuery(stream, after_sequence=5, limit=1)))
    assert missing == ()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ("no_match", "rare_match"))
@pytest.mark.parametrize("counts", (
    pytest.param((3,), id="contract"),
    pytest.param((32, 64, 128), marks=pytest.mark.manual, id="growth"),
))
async def test_selective_trace_examined_work_scales_linearly(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    counts: tuple[int, ...],
) -> None:
    examined = 0
    measuring = False
    visits = Counter()
    original = _MemoryOverlay.__iter__

    def iterate(self):
        nonlocal examined
        for key in original(self):
            if measuring:
                examined += 1
                visits[id(self._base), key] += 1
            yield key

    monkeypatch.setattr(_MemoryOverlay, "__iter__", iterate)
    costs = []
    for count in counts:
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
            visits.clear()
            measuring = True
            filters = {"tool_call_id": "missing"} if selector == "no_match" else {"model_request_seq": count}
            page = await history.reader.trace("execution", tenant_id="tenant", cursor=None, limit=1, **filters)
            measuring = False
            assert len(page.items) == (0 if selector == "no_match" else 1)
            if selector == "rare_match":
                assert page.items[0].payload["model_request_seq"] == count
            assert max(visits.values(), default=0) <= 2, "selection must not rescan maps for each candidate"
            costs.append(examined)
    for smaller, larger in zip(costs, costs[1:]):
        assert larger <= smaller * 2.25


@pytest.mark.asyncio
@pytest.mark.parametrize("counts", (
    pytest.param((3,), id="contract"),
    pytest.param((32, 64, 128), marks=pytest.mark.manual, id="growth"),
))
async def test_old_trace_cutoff_does_not_rescan_for_each_later_backdated_event(
    monkeypatch: pytest.MonkeyPatch,
    counts: tuple[int, ...],
) -> None:
    original = _MemoryOverlay.__iter__
    examined = 0
    measuring = False
    visits = Counter()

    def iterate(self):
        nonlocal examined
        for key in original(self):
            if measuring:
                examined += 1
                visits[id(self._base), key] += 1
            yield key

    monkeypatch.setattr(_MemoryOverlay, "__iter__", iterate)
    costs = []
    for count in counts:
        storage = Persistence(PersistenceStorage.in_memory())
        try:
            archive, run_id = await _seed_trace(storage, 1)
            await _append_backdated_trace(storage, run_id, count)
            examined = 0
            visits.clear()
            measuring = True
            rows = await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=1,
                after_timestamp=None, after_sequence=0, limit=1,
            )
            measuring = False
            assert len(rows) == 1 and rows[0][0] == 1
            assert rows[0][1].tool_call_id == "call-0"
            assert max(visits.values(), default=0) <= 1, "later candidates must not rescan the captured prefix"
            costs.append(examined)
            current = await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=count + 1,
                after_timestamp=None, after_sequence=0, limit=count + 1,
            )
            assert [sequence for sequence, _ in current] == [*range(2, count + 2), 1]
        finally:
            await storage.close()
    for smaller, larger in zip(costs, costs[1:]):
        assert larger <= smaller * 2.25


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ("no_match", "old_cutoff"))
@pytest.mark.parametrize("sizes", (
    pytest.param((3,), id="contract"),
    pytest.param((32, 64, 128), marks=pytest.mark.manual, id="growth"),
))
async def test_sql_trace_query_count_is_independent_of_examined_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, sizes: tuple[int, ...],
) -> None:
    counts = []
    for count in sizes:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'trace-{count}.db'}")
        await provision_database(engine)
        storage = Persistence(PersistenceStorage.sql(engine))
        statements = []

        def capture(_connection, _cursor, statement, _parameters, _context, _executemany) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        try:
            archive, run_id = await _seed_trace(
                storage, count if scenario == "no_match" else 1, reverse_timestamps=True,
            )
            if scenario == "old_cutoff":
                await _append_backdated_trace(storage, run_id, count)
            high_water = count if scenario == "no_match" else 1
            sql_event.listen(engine.sync_engine, "before_cursor_execute", capture)
            try:
                with _capture_fact_batches(engine.sync_engine, monkeypatch) as batches:
                    rows = await archive.list_trace_events(
                        agent_run_id=run_id, event_high_water=high_water,
                        after_timestamp=None, after_sequence=0, limit=1,
                        **({"tool_call_id": "missing"} if scenario == "no_match" else {}),
                    )
            finally:
                sql_event.remove(engine.sync_engine, "before_cursor_execute", capture)
            assert len(rows) == (0 if scenario == "no_match" else 1)
            if scenario == "old_cutoff":
                assert [(sequence, value.tool_call_id) for sequence, value in rows] == [(1, "call-0")]
            expected_sequences = [] if scenario == "no_match" else [1]
            assert [sequence for batch in batches for sequence in batch] == expected_sequences, (
                "Trace filters must not materialize rejected or later event facts"
            )
            candidate_reads = sum(
                "FROM AI_STATE_RECORDS" in statement.upper() and "ORDER BY" in statement.upper()
                for statement in statements
            )
            # Rejected or later candidates must not cause per-event source reads.
            assert candidate_reads <= (1 if scenario == "old_cutoff" else 0)
            counts.append(len(statements))
        finally:
            await storage.close()
            await engine.dispose()
    assert max(counts) <= 12
    assert all(value <= counts[0] + 2 for value in counts[1:])


@pytest.mark.asyncio
@pytest.mark.parametrize("page_size", (
    1, 33,
    pytest.param(32, marks=pytest.mark.manual, id="scale-32"),
    pytest.param(64, marks=pytest.mark.manual, id="scale-64"),
    pytest.param(129, marks=pytest.mark.manual, id="scale-129"),
))
@pytest.mark.parametrize("reverse_timestamps", (False, True))
async def test_trace_pages_read_only_selected_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_size: int, reverse_timestamps: bool,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'page.db'}")
    await provision_database(engine)
    storage = Persistence(PersistenceStorage.sql(engine))
    try:
        event_count = page_size + 1
        archive, run_id = await _seed_trace(storage, event_count, reverse_timestamps=reverse_timestamps)
        original_fact = SqlTraceEventStore.row_to_fact
        alive = maximum_alive = 0
        logical_json_bytes = 0

        class TrackedFact(StoredFact):
            pass

        def released() -> None:
            nonlocal alive
            alive -= 1

        def materialize(store, row):
            nonlocal alive, maximum_alive, logical_json_bytes
            value = original_fact(store, row)
            logical_json_bytes += len(canonical_json_bytes(value.data))
            tracked = TrackedFact(value.stream_key, value.sequence, value.owner_key,
                                  value.kind, value.subject_id, value.state, value.data)
            alive += 1
            maximum_alive = max(maximum_alive, alive)
            weakref.finalize(tracked, released)
            return tracked

        monkeypatch.setattr(SqlTraceEventStore, "row_to_fact", materialize)
        with _capture_fact_batches(engine.sync_engine, monkeypatch) as batches:
            rows = await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=event_count,
                after_timestamp=None, after_sequence=0, limit=page_size,
            )
        fact_selects = len(batches)
        expected = (range(event_count, event_count - page_size, -1)
                    if reverse_timestamps else range(1, page_size + 1))
        assert [sequence for sequence, _ in rows] == list(expected)
        assert sum(map(len, batches)) == page_size, "normal pages must not read unselected event facts"
        assert sorted(sequence for batch in batches for sequence in batch) == sorted(expected)
        print("TRACE_SELECTED_FACTS " + json.dumps({"page_size": page_size, "reverse": reverse_timestamps,
              "fact_selects": fact_selects, "fact_batches": len(batches), "fact_rows": sum(map(len, batches)),
              "logical_json_bytes": logical_json_bytes, "peak_raw_facts": maximum_alive}))
        assert fact_selects <= (page_size + 31) // 32, "Trace page queries stay bounded"
        assert maximum_alive <= page_size, "raw Trace rows must remain bounded by the requested page"
        assert alive == 0
    finally:
        await storage.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_sparse_trace_pages_read_only_selected_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "sparse-page.db"))
    try:
        archive, run_id = await _seed_trace(storage, 0)
        timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
        # Gaps in the event sequence must not materialize non-Trace events.
        for sequence in range(1, 37):
            traced = sequence not in {2, 35}
            await storage.run_store.append_event(StepEvent(
                run_id, "TOOL_CALL_STARTED" if traced else "AGENT_RUN_STARTED", sequence,
                timestamp=timestamp, tool_call_id=f"call-{sequence}" if traced else None,
                tool_name="tool" if traced else None,
            ), execution_id="execution")
        await storage.run_store.flush_execution_projection(run_id, execution_id="execution")
        engine = archive.state_store.storage_group.context.engine.sync_engine
        with _capture_fact_batches(engine, monkeypatch) as batches:
            rows = await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=36,
                after_timestamp=None, after_sequence=0, limit=36,
            )
        expected = [sequence for sequence in range(1, 37) if sequence not in {2, 35}]
        assert [sequence for sequence, _ in rows] == expected
        assert [sequence for batch in batches for sequence in batch] == expected
        assert len(batches) <= 3
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("counts", (
    pytest.param((3,), id="contract"),
    pytest.param((32, 64, 128), marks=pytest.mark.manual, id="growth"),
))
async def test_run_release_scans_owned_cleanup_data_once_per_batch(
    monkeypatch: pytest.MonkeyPatch, counts: tuple[int, ...],
) -> None:
    original = _MemoryOverlay.__iter__
    examined = 0
    measuring = False
    visits = Counter()

    def iterate(self):
        nonlocal examined
        for key in original(self):
            if measuring:
                examined += 1
                visits[id(self._base), key] += 1
            yield key

    monkeypatch.setattr(_MemoryOverlay, "__iter__", iterate)
    costs = []
    for count in counts:
        storage = Persistence(PersistenceStorage.in_memory())
        try:
            archive, run_id = await _seed_trace(storage, count)
            owner, stream, alias = b"o" * 32, b"s" * 32, b"a" * 32
            unrelated = StoredRecord(owner, None, None, "owner", b"owner", None, 0, None, 0, None, {})
            fact = StoredFact(stream, 1, owner, "event", None, None, {"retained": True})

            async def seed_unrelated(transaction) -> None:
                await transaction.insert_record(unrelated)
                await transaction.insert_alias(StoredAlias(alias, owner))
                await transaction.insert_fact(fact)

            await archive.state_store.mutate(seed_unrelated)
            examined = 0
            visits.clear()
            measuring = True
            await archive.release_agent_run(run_id, execution_id="execution")
            measuring = False
            # Mapping item snapshots may enumerate once for their length and once for their values.
            assert max(visits.values(), default=0) <= 2, "cleanup must scan by owner batch, not by deleted record"
            costs.append(examined)
            assert await archive.get_agent_run(agent_run_id=run_id) is None
            assert await archive.list_events(agent_run_id=run_id) == []
            assert await archive.state_store.read(lambda tx: tx.get_record(owner)) == unrelated
            assert await archive.state_store.read(lambda tx: tx.resolve_alias(alias)) == owner
            assert await archive.state_store.read(lambda tx: tx.get_facts(stream, (1,))) == (fact,)
        finally:
            await storage.close()
    for smaller, larger in zip(costs, costs[1:]):
        assert larger <= smaller * 2.25


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_count,sparse", (
    (1, False), (33, False), (33, True),
    pytest.param(32, False, marks=pytest.mark.manual, id="scale-32-contiguous"),
    pytest.param(65, False, marks=pytest.mark.manual, id="scale-65-contiguous"),
    pytest.param(65, True, marks=pytest.mark.manual, id="scale-65-sparse"),
))
async def test_history_associations_batch_only_selected_source_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected_count: int, sparse: bool,
) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "association-page.db"))
    try:
        archive, run_id = await _seed_trace(storage, selected_count * 2 if sparse else selected_count + 1)
        selected = tuple(range(1, selected_count * 2, 2)) if sparse else tuple(range(1, selected_count + 1))
        engine = archive.state_store.storage_group.context.engine.sync_engine
        with _capture_fact_batches(engine, monkeypatch) as batches:
            values = await archive.read_history_associations(
                agent_run_id=run_id, message_seqs=(),
                tool_call_ids=tuple(f"call-{sequence - 1}" for sequence in reversed(selected)),
                event_high_water=selected[-1],
            )
        fact_selects = len(batches)
        assert [value.tool_call_id for value in values] == [f"call-{sequence - 1}" for sequence in selected]
        assert tuple(sorted(sequence for batch in batches for sequence in batch)) == selected
        assert all(len(batch) <= 32 for batch in batches)
        assert fact_selects <= (selected_count + 31) // 32
        print("HISTORY_SELECTED_FACTS " + json.dumps({"selected_count": selected_count, "sparse": sparse,
              "fact_selects": fact_selects, "fact_rows": sum(map(len, batches))}))
    finally:
        await storage.close()
