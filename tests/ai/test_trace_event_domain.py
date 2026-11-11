#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace storage preserves captured history and immutable event identity."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._step_contracts import StepEvent, TOOL_ERROR_CODE_METADATA_KEY
from linktools.ai.persistence.state import FactQuery, RecordQuery

from .test_incremental_history_read_cost import _seed_trace


REQUEST = "linktools.ai.model_request_seq"
MESSAGE = "linktools.ai.message_seq"
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


@asynccontextmanager
async def _archive(tmp_path: Path, backend: str):
    storage = Persistence(
        PersistenceStorage.in_memory() if backend == "memory"
        else PersistenceStorage.sqlite(tmp_path / "trace-domain.db"),
    )
    try:
        archive, run_id = await _seed_trace(storage, 0)
        run = await storage.run_store.get_agent_run(agent_run_id=run_id)
        assert run is not None
        await archive.register_agent_run(run, execution_id="execution")
        yield archive, run_id
    finally:
        await storage.close()


def _event(run_id: str, kind: str, *, request: str = "1", message: str | None = None,
           tool: str | None = None, step: int = 1, second: int = 0,
           unknown: bool = False) -> StepEvent:
    metadata = {REQUEST: request}
    if message is not None:
        metadata[MESSAGE] = message
    if unknown:
        metadata[TOOL_ERROR_CODE_METADATA_KEY] = ErrorCode.TOOL_EFFECT_UNKNOWN.value
    return StepEvent(run_id, kind, step, timestamp=EPOCH + timedelta(seconds=second),
                     metadata=metadata, tool_call_id=tool)


async def _append(archive, *events: StepEvent) -> None:
    for event in events:
        await archive.append_event(event, execution_id="execution")


async def _trace(archive, run_id: str, cutoff: int, **selectors):
    return await archive.list_trace_events(
        agent_run_id=run_id, event_high_water=cutoff, after_timestamp=None,
        after_sequence=0, limit=100, **selectors,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("origin", ("response", "tool"))
async def test_history_two_hops_respect_cutoff_and_raw_request_identity(
    tmp_path: Path, backend: str, origin: str,
) -> None:
    async with _archive(tmp_path, backend) as (archive, run_id):
        events = [
            _event(run_id, "MODEL_REQUEST_STARTED", request="01", second=8),
            _event(run_id, "MODEL_REQUEST_STARTED", request="1", second=7),
            _event(run_id, "MODEL_REQUEST_FAILED", request="01", second=6),
            _event(run_id, "MODEL_REQUEST_SUCCEEDED", request="01", message="2", second=5)
            if origin == "response" else
            _event(run_id, "TOOL_CALL_STARTED", request="01", tool="call", second=5),
            _event(run_id, "MODEL_REQUEST_CANCELLED", request="01", second=4),
            _event(run_id, "MODEL_REQUEST_SUCCEEDED", request="1", message="3", second=3),
        ]
        await _append(archive, *events)
        arguments = dict(agent_run_id=run_id, message_seqs=(2, 2) if origin == "response" else (),
                         tool_call_ids=("call", "call") if origin == "tool" else ())
        assert await archive.read_history_associations(**arguments, event_high_water=3) == []
        assert await archive.read_history_associations(**arguments, event_high_water=4) == [
            events[0], events[2], events[3],
        ]
        assert await archive.read_history_associations(**arguments, event_high_water=6) == [
            events[0], events[2], events[3], events[4],
        ]
        numeric = await _trace(archive, run_id, 6, model_request_seq=1)
        assert [sequence for sequence, _ in numeric] == [6, 5, 4, 3, 2, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_trace_filters_precede_limit_and_cursor_keeps_captured_cut(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (archive, run_id):
        await _append(archive,
                      _event(run_id, "TOOL_CALL_STARTED", tool="other", second=0),
                      _event(run_id, "TOOL_CALL_STARTED", tool="call", request="2", second=1),
                      _event(run_id, "TOOL_CALL_SUCCEEDED", tool="call", step=2, second=2),
                      _event(run_id, "TOOL_CALL_FAILED", tool="call", second=3),
                      _event(run_id, "MODEL_REQUEST_STARTED", tool="call", second=3))
        arguments = dict(agent_run_id=run_id, event_high_water=5, limit=1,
                         model_request_seq=1, step_index=1, tool_call_id="call")
        first = await archive.list_trace_events(**arguments, after_timestamp=None, after_sequence=0)
        assert [sequence for sequence, _ in first] == [4]
        await _append(archive, _event(run_id, "MODEL_REQUEST_CANCELLED", tool="call", second=-1))
        tail = await archive.list_trace_events(
            **arguments, after_timestamp=first[-1][1].timestamp, after_sequence=4,
        )
        assert [sequence for sequence, _ in tail] == [5]
        assert [sequence for sequence, _ in await _trace(archive, run_id, 5)] == [1, 2, 3, 4, 5]
        assert [sequence for sequence, _ in await _trace(archive, run_id, 6)] == [6, 1, 2, 3, 4, 5]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_unknown_tool_failure_is_trace_only_until_determined(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (archive, run_id):
        unknown = _event(run_id, "TOOL_CALL_FAILED", tool="call", unknown=True)
        known = _event(run_id, "TOOL_CALL_FAILED", tool="call", second=1)
        await _append(archive, unknown, known)
        arguments = dict(agent_run_id=run_id, message_seqs=(), tool_call_ids=("call",))
        assert await archive.read_history_associations(**arguments, event_high_water=1) == []
        assert await archive.read_history_associations(**arguments, event_high_water=2) == [known]
        assert [event for _, event in await _trace(archive, run_id, 2)] == [unknown, known]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("kind,tool", (("MODEL_REQUEST_STARTED", None), ("TOOL_CALL_STARTED", "call")))
async def test_same_identity_kind_conflict_rolls_back_without_sequence_gap(
    tmp_path: Path, backend: str, kind: str, tool: str | None,
) -> None:
    async with _archive(tmp_path, backend) as (archive, run_id):
        first = _event(run_id, kind, tool=tool)
        await _append(archive, first)
        with pytest.raises(AIError) as caught:
            await _append(archive, replace(first, timestamp=EPOCH + timedelta(seconds=1)))
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await archive.list_events(agent_run_id=run_id) == [first]
        next_event = _event(run_id, kind.replace("STARTED", "SUCCEEDED"), tool=tool, second=2)
        await _append(archive, next_event)
        assert await _trace(archive, run_id, 2) == [(1, first), (2, next_event)]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_trace_event_has_no_state_fact_or_association_shadow(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (archive, run_id):
        event = _event(run_id, "MODEL_REQUEST_SUCCEEDED", message="2")
        await _append(archive, event)
        records = await archive.state_store.read(lambda tx: tx.list_records(RecordQuery(
            parent_key=archive._agent_run_key(run_id), kind="history_association",
        )))
        facts = await archive.state_store.read(lambda tx: tx.list_facts(FactQuery(
            archive._stream(run_id, "event"),
        )))
        if backend == "sqlite":
            assert not records
            assert not facts
        assert await archive.list_events(agent_run_id=run_id) == [event]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_concurrent_role_conflict_keeps_one_event_and_allows_retry(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (archive, run_id):
        results = await asyncio.gather(
            _append(archive, _event(run_id, "TOOL_CALL_STARTED", tool="call")),
            _append(archive, _event(run_id, "TOOL_CALL_STARTED", tool="call", second=1)),
            return_exceptions=True,
        )
        assert sum(result is None for result in results) == 1
        failures = [result for result in results if isinstance(result, Exception)]
        assert len(failures) == 1
        assert isinstance(failures[0], AIError)
        assert failures[0].code in (ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_INTEGRITY_ERROR)
        published = await archive.list_events(agent_run_id=run_id)
        assert len(published) == 1
        completed = _event(run_id, "TOOL_CALL_SUCCEEDED", tool="call", second=2)
        await _append(archive, completed)
        assert await _trace(archive, run_id, 2) == [(1, published[0]), (2, completed)]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_promoted_and_residual_metadata_round_trip_together(
    tmp_path: Path, backend: str,
) -> None:
    async with _archive(tmp_path, backend) as (archive, run_id):
        event = _event(run_id, "MODEL_REQUEST_SUCCEEDED", request="01", message="2")
        event.metadata.update({"a": "first", "z": "last", "é": "unicode",
                               TOOL_ERROR_CODE_METADATA_KEY: "diagnostic"})
        await _append(archive, event)
        assert await archive.list_events(agent_run_id=run_id) == [event]
        assert await _trace(archive, run_id, 1, model_request_seq=1) == [(1, event)]
        assert await archive.read_history_associations(
            agent_run_id=run_id, message_seqs=(2,), tool_call_ids=(), event_high_water=1,
        ) == [event]


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("reordered", "duplicate"))
async def test_promoted_metadata_does_not_repair_corrupt_residual_mapping(
    tmp_path: Path, corruption: str,
) -> None:
    from linktools.ai.persistence.codec import _encode_step_envelope
    from linktools.ai.persistence.state import StoredFact

    async with _archive(tmp_path, "sqlite") as (archive, run_id):
        event = _event(run_id, "MODEL_REQUEST_SUCCEEDED", message="2")
        event.metadata.update({"a": "first", "z": "last"})
        fact = StoredFact(archive._stream(run_id, "event"), 1, archive._agent_run_key(run_id),
                          "step_event", None, event.event_type, _encode_step_envelope(event))
        row = archive._trace_events.row_values(fact, event)
        pairs = row["payload_json"]["value"]["payload"]["fields"]["metadata"]["$mapping"]
        if corruption == "reordered":
            pairs.reverse()
        else:
            pairs.append(pairs[-1].copy())
        with pytest.raises(AIError) as caught:
            archive._trace_events.row_to_fact(row)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
