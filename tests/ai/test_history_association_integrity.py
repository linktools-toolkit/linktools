#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History indexes preserve captured cuts and reject incomplete coverage."""

from linktools.ai.persistence import Persistence
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence._step_contracts import StepEvent
from linktools.ai.persistence.state import FactQuery, RecordQuery

from .test_incremental_history_read_cost import _capture_fact_batches, _seed_trace


async def _discard_locators(archive, run_id: str, keep_sequence: int | None) -> None:
    async def mutate(transaction) -> None:
        records = await transaction.list_records(RecordQuery(
            parent_digest=archive._agent_run_key(run_id), kind="history_association",
        ))
        await transaction.delete_records(tuple(
            record.key_digest for record in records
            if record.sort_key == b"coverage:event" or record.data["sequence"] != keep_sequence
        ))
    await archive.state_store.mutate(mutate)


@pytest.mark.asyncio
async def test_trace_indexes_keep_cursor_cut(
    tmp_path: Path,
) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "trace.db"))
    try:
        archive, run_id = await _seed_trace(storage, 4)
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        await archive.append_event(StepEvent(
            run_id, "TOOL_CALL_STARTED", 5, timestamp=timestamp,
            tool_call_id="new", tool_name="tool",
        ), execution_id="execution")
        marker = await archive.state_store.read(lambda tx: tx.get_record(
            archive._history_association_key(run_id, "coverage", "event"),
        ))
        assert marker is not None and marker.data == {"sequence": 5}
        first = await archive.list_trace_events(
            agent_run_id=run_id, event_high_water=4,
            after_timestamp=None, after_sequence=0, limit=2,
        )
        assert [sequence for sequence, _ in first] == [1, 2]
        tail = await archive.list_trace_events(
            agent_run_id=run_id, event_high_water=4,
            after_timestamp=first[-1][1].timestamp, after_sequence=2, limit=10,
        )
        assert [sequence for sequence, _ in tail] == [3, 4]
        fresh = await archive.list_trace_events(
            agent_run_id=run_id, event_high_water=5,
            after_timestamp=None, after_sequence=0, limit=10,
        )
        assert [sequence for sequence, _ in fresh] == [5, 1, 2, 3, 4]
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_associations_find_response_request_and_tools_at_captured_cut(
    tmp_path: Path,
) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "associations.db"))
    try:
        archive, run_id = await _seed_trace(storage, 0)
        run = await storage.run_store.get_agent_run(agent_run_id=run_id)
        assert run is not None
        await archive.register_agent_run(run, execution_id="execution")
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for index, (kind, metadata, tool_id) in enumerate((
            ("MODEL_REQUEST_STARTED", {"linktools.ai.model_request_seq": "1"}, None),
            ("MODEL_REQUEST_SUCCEEDED", {"linktools.ai.model_request_seq": "1", "linktools.ai.message_seq": "2"}, None),
            ("TOOL_CALL_STARTED", {}, "call"),
        ), 1):
            await archive.append_event(StepEvent(
                run_id, kind, 1, timestamp=timestamp + timedelta(seconds=index),
                metadata=metadata, tool_call_id=tool_id,
            ), execution_id="execution")
        await archive.append_event(StepEvent(
            run_id, "TOOL_CALL_SUCCEEDED", 1, timestamp=timestamp + timedelta(seconds=4),
            tool_call_id="call",
        ), execution_id="execution")
        arguments = dict(agent_run_id=run_id, message_seqs=(2, 2), tool_call_ids=("call", "call"))
        old = await archive.read_history_associations(**arguments, event_high_water=3)
        assert [event.event_type for event in old] == [
            "MODEL_REQUEST_STARTED", "MODEL_REQUEST_SUCCEEDED", "TOOL_CALL_STARTED",
        ]
        fresh = await archive.read_history_associations(**arguments, event_high_water=4)
        assert [event.event_type for event in fresh] == [
            "MODEL_REQUEST_STARTED", "MODEL_REQUEST_SUCCEEDED", "TOOL_CALL_STARTED", "TOOL_CALL_SUCCEEDED",
        ]
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ("trace", "association", "coverage"))
async def test_malformed_present_locator_is_rejected(tmp_path: Path, target: str) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "invalid.db"))
    try:
        archive, run_id = await _seed_trace(storage, 2)
        family, identity = {
            "trace": ("trace", "2"),
            "association": ("TOOL_CALL_STARTED", "call-1"),
            "coverage": ("coverage", "event"),
        }[target]
        key = archive._history_association_key(run_id, family, identity)
        async def corrupt(transaction) -> None:
            record = await transaction.get_record(key)
            assert record is not None
            await transaction.delete_record(key)
            await transaction.insert_record(replace(record, data={"sequence": True}))
        await archive.state_store.mutate(corrupt)
        with pytest.raises(AIError) as raised:
            if target == "association":
                await archive.read_history_associations(
                    agent_run_id=run_id, message_seqs=(), tool_call_ids=("call-1",), event_high_water=2,
                )
            else:
                await archive.list_trace_events(
                    agent_run_id=run_id, event_high_water=2,
                    after_timestamp=None, after_sequence=0, limit=10,
                )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", (False, True))
async def test_incomplete_event_index_coverage_rejects_reads_and_append(tmp_path: Path, partial: bool) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "incomplete.db"))
    try:
        archive, run_id = await _seed_trace(storage, 2)
        await _discard_locators(archive, run_id, 2 if partial else None)
        with pytest.raises(AIError) as trace:
            await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=2,
                after_timestamp=None, after_sequence=0, limit=10,
            )
        assert trace.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as associations:
            await archive.read_history_associations(
                agent_run_id=run_id, message_seqs=(), tool_call_ids=("call-1",), event_high_water=2,
            )
        assert associations.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as append:
            await archive.append_event(StepEvent(
                run_id, "TOOL_CALL_STARTED", 3, tool_call_id="new", tool_name="tool",
            ), execution_id="execution")
        assert append.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert len(await archive.list_events(agent_run_id=run_id)) == 2
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_empty_trace_does_not_require_index_coverage(tmp_path: Path) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "empty.db"))
    try:
        archive, run_id = await _seed_trace(storage, 0)
        assert await archive.list_trace_events(
            agent_run_id=run_id, event_high_water=0,
            after_timestamp=None, after_sequence=0, limit=10,
        ) == []
        assert await archive.read_history_associations(
            agent_run_id=run_id, message_seqs=(1,), tool_call_ids=(), event_high_water=0,
        ) == []
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_filtered_trace_rejects_missing_locator_under_complete_coverage(tmp_path: Path) -> None:
    storage = Persistence(PersistenceStorage.sqlite(tmp_path / "missing.db"))
    try:
        archive, run_id = await _seed_trace(storage, 2)
        await archive.state_store.mutate(lambda tx: tx.delete_record(
            archive._history_association_key(run_id, "trace", "2"),
        ))
        with pytest.raises(AIError) as raised:
            await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=2,
                after_timestamp=None, after_sequence=0, limit=10, tool_call_id="call-1",
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_trace_pages_preserve_sparse_timestamp_order_and_ties(
    tmp_path: Path, backend: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = Persistence(
        PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "order.db"),
    )
    try:
        archive, run_id = await _seed_trace(storage, 0)
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        expected = []
        for index in range(73):
            traced = index % 3 != 1
            event = StepEvent(
                run_id, "TOOL_CALL_STARTED" if traced else "AGENT_RUN_STARTED", index + 1,
                timestamp=timestamp + timedelta(seconds=((index * 37) % 73) // 2),
                tool_call_id=f"call-{index}" if traced else None,
                tool_name="tool" if traced else None,
            )
            await storage.run_store.append_event(event, execution_id="execution")
            if traced:
                expected.append((index + 1, event))
        await storage.run_store.flush_execution_projection(run_id, execution_id="execution")
        expected.sort(key=lambda item: (item[1].timestamp, item[0]))
        engine = None if backend == "memory" else archive.state_store.storage_group.context.engine.sync_engine
        for page_size in (1, 33, 65):
            result = []
            after_timestamp, after_sequence = None, 0
            while True:
                with (_capture_fact_batches(engine, monkeypatch) if engine is not None else nullcontext([])) as batches:
                    page = await archive.list_trace_events(
                        agent_run_id=run_id, event_high_water=73,
                        after_timestamp=after_timestamp, after_sequence=after_sequence, limit=page_size,
                    )
                if backend == "sqlite":
                    selected = [sequence for sequence, _ in page]
                    assert sorted(sequence for batch in batches for sequence in batch) == sorted(selected)
                    ordered_batches = 0
                    offset = 0
                    while offset < len(selected):
                        end = offset + 1
                        while (end < len(selected) and end - offset < 32
                               and selected[end] == selected[end - 1] + 1):
                            end += 1
                        ordered_batches += 1
                        offset = end
                    assert len(batches) <= ordered_batches
                if not page:
                    break
                result.extend(page)
                after_sequence, last_event = page[-1]
                after_timestamp = last_event.timestamp
            assert result == expected
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_trace_duplicate_sequence_locators_keep_individual_validation(
    tmp_path: Path, backend: str,
) -> None:
    storage = Persistence(
        PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "duplicates.db"),
    )
    try:
        archive, run_id = await _seed_trace(storage, 1)
        key = archive._history_association_key(run_id, "trace", "1")

        async def duplicate(transaction) -> None:
            record = await transaction.get_record(key)
            assert record is not None
            await transaction.insert_record(replace(
                record, key_digest=b"\xff" * 32,
                sort_key=record.sort_key + b"x",
            ))

        await archive.state_store.mutate(duplicate)
        arguments = dict(agent_run_id=run_id, event_high_water=1,
                         after_timestamp=None, after_sequence=0, limit=10)
        with pytest.raises(AIError) as raised:
            await archive.list_trace_events(**arguments)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("corruption", ("missing", "payload"))
@pytest.mark.parametrize("reverse_timestamps", (False, True))
async def test_trace_pages_reject_corrupt_facts_across_batches(
    tmp_path: Path, backend: str, corruption: str, reverse_timestamps: bool,
) -> None:
    storage = Persistence(
        PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "corrupt.db"),
    )
    try:
        archive, run_id = await _seed_trace(storage, 66, reverse_timestamps=reverse_timestamps)
        stream = archive._stream(run_id, "event")
        corrupt_sequence = 34 if reverse_timestamps else 33

        async def corrupt(transaction) -> None:
            facts = await transaction.list_facts(FactQuery(stream))
            owner = await transaction.get_record(archive._agent_run_key(run_id))
            assert owner is not None
            await transaction.guard_record(owner.key_digest, expected_storage_revision=owner.storage_revision)
            await transaction.delete_fact_streams(archive._agent_run_key(run_id))
            await transaction.insert_facts(tuple(
                replace(fact, data={"invalid": True}) if fact.sequence == corrupt_sequence else fact
                for fact in facts if corruption != "missing" or fact.sequence != corrupt_sequence
            ))

        await archive.state_store.mutate(corrupt)
        with pytest.raises(AIError) as raised:
            await archive.list_trace_events(
                agent_run_id=run_id, event_high_water=66,
                after_timestamp=None, after_sequence=0, limit=65,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("corruption", ("missing", "payload", "locator"))
async def test_association_source_batches_reject_corruption_only_inside_cutoff(
    tmp_path: Path, backend: str, corruption: str,
) -> None:
    storage = Persistence(
        PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "association-corrupt.db"),
    )
    try:
        archive, run_id = await _seed_trace(storage, 66)
        stream = archive._stream(run_id, "event")

        async def corrupt(transaction) -> None:
            if corruption == "locator":
                key = archive._history_association_key(run_id, "TOOL_CALL_STARTED", "call-32")
                record = await transaction.get_record(key)
                assert record is not None
                await transaction.delete_record(key)
                await transaction.insert_record(replace(record, data={"sequence": 33}, sort_key=b"bad"))
                return
            facts = await transaction.list_facts(FactQuery(stream))
            owner = await transaction.get_record(archive._agent_run_key(run_id))
            assert owner is not None
            await transaction.guard_record(owner.key_digest, expected_storage_revision=owner.storage_revision)
            await transaction.delete_fact_streams(archive._agent_run_key(run_id))
            await transaction.insert_facts(tuple(
                replace(fact, data={"invalid": True}) if fact.sequence == 33 else fact
                for fact in facts if corruption != "missing" or fact.sequence != 33
            ))

        await archive.state_store.mutate(corrupt)
        arguments = dict(agent_run_id=run_id, message_seqs=(),
                         tool_call_ids=tuple(f"call-{index}" for index in range(65)))
        prior = await archive.read_history_associations(**arguments, event_high_water=32)
        assert [value.tool_call_id for value in prior] == [f"call-{index}" for index in range(32)]
        with pytest.raises(AIError) as raised:
            await archive.read_history_associations(**arguments, event_high_water=65)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_associations_validate_each_locator_for_reused_source_fact(tmp_path: Path, backend: str) -> None:
    storage = Persistence(
        PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "association-alias.db"),
    )
    try:
        archive, run_id = await _seed_trace(storage, 1)
        key = archive._history_association_key(run_id, "TOOL_CALL_STARTED", "call-0")
        alias = archive._history_association_key(run_id, "TOOL_CALL_SUCCEEDED", "call-0")

        async def corrupt(transaction) -> None:
            record = await transaction.get_record(key)
            assert record is not None
            await transaction.insert_record(replace(record, key_digest=alias))

        await archive.state_store.mutate(corrupt)
        with pytest.raises(AIError) as raised:
            await archive.read_history_associations(
                agent_run_id=run_id, message_seqs=(), tool_call_ids=("call-0",), event_high_water=1,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()
