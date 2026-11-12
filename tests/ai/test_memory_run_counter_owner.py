#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory run counters share admission ownership and transaction lifetime."""

from dataclasses import replace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._run_counters import RUN_COUNTER_COLUMNS, run_counter_streams
from linktools.ai.persistence.state import StoredRecord, StateTransaction, record_key, stream_key


def _admission(run: str = "run") -> StoredRecord:
    key = record_key("run-counter", "tenant", "trace", "agent_run", (run,))
    return StoredRecord(key, None, None, "agent_run", key, None, 0, None, 0, None, {"run": run})


@pytest.mark.asyncio
async def test_memory_run_counters_follow_admission_cas_and_release() -> None:
    store = InMemoryStateStore()
    await store.initialize()
    try:
        record = _admission()
        event, checkpoint, interaction = run_counter_streams(record.key)
        opaque = b"opaque-counter"

        async def publish(transaction: StateTransaction) -> None:
            await transaction.insert_record(record)
            assert await transaction.reserve_sequences({event: 2, checkpoint: 1, opaque: 4}) == {
                event: 2, checkpoint: 1, opaque: 4,
            }
            current = await transaction.get_record(record.key)
            assert current is not None
            assert await transaction.replace_record(replace(current, storage_revision=1, data={"run": "updated"}),
                expected_storage_revision=0)
            assert await transaction.reserve_sequence(event, 3) == 5

        await store.mutate(publish)
        assert store._sequences == {opaque: 4}
        physical = store._step_run_admissions[record.key]
        assert physical.record.data == {"run": "updated"}
        assert physical.counters == (5, 1, None)
        assert await store.read(lambda tx: tx.get_sequences((event, checkpoint, interaction, opaque))) == {
            event: 5, checkpoint: 1, interaction: 0, opaque: 4,
        }
        all_values = await store.read(lambda tx: tx.scan_sequences())
        assert all_values == dict(sorted({event: 5, checkpoint: 1, opaque: 4}.items()))
        pages = []
        after = None
        while True:
            page = await store.read(lambda tx: tx.scan_sequences_page(after=after, limit=1))
            if not page:
                break
            pages.extend(page.items())
            after = next(iter(page))
        assert dict(pages) == all_values
        await store.mutate(lambda tx: tx.delete_sequence(event))
        assert store._step_run_admissions[record.key].counters == (None, 1, None)
        with pytest.raises(AIError) as raised:
            await store.mutate(lambda tx: tx.advance_sequence(event, 0))
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.mutate(lambda tx: tx.advance_sequence(b"new-opaque", 0)) == 1
        assert await store.mutate(lambda tx: tx.reserve_sequence(event, 2)) == 2

        async def recreate(transaction: StateTransaction) -> None:
            await transaction.delete_record(record.key)
            assert await transaction.get_sequence(event) == 0
            await transaction.insert_record(record)
            assert await transaction.get_sequence(checkpoint) == 0
            assert await transaction.reserve_sequence(interaction, 1) == 1

        await store.mutate(recreate)
        assert store._step_run_admissions[record.key].counters == (None, None, 1)
        await store.mutate(lambda tx: tx.delete_record(record.key))
        assert not store._step_run_admissions
        assert await store.read(lambda tx: tx.get_sequence(interaction)) == 0
        assert interaction not in await store.read(lambda tx: tx.scan_sequences())
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_memory_run_counter_mixed_reservation_rollback_and_logical_restore() -> None:
    store = InMemoryStateStore()
    restored = InMemoryStateStore()
    await store.initialize()
    await restored.initialize()
    try:
        record = _admission()
        streams = run_counter_streams(record.key)
        await store.mutate(lambda tx: tx.insert_record(record))
        await store.mutate(lambda tx: tx.reserve_sequences(dict(zip(streams, (2, 3, 4), strict=True))))
        missing = stream_key("run-counter", "tenant", "trace", "event", ("missing",))
        with pytest.raises(AIError) as raised:
            await store.mutate(lambda tx: tx.reserve_sequences({b"opaque": 9, streams[0]: 5, missing: 1}))
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        assert await store.read(lambda tx: tx.get_sequence(streams[0])) == 2
        assert b"opaque" not in store._sequences
        assert _admission("missing").key not in store._step_run_admissions

        sequences = await store.read(lambda tx: tx.scan_sequences())

        async def restore(transaction: StateTransaction) -> None:
            await transaction.insert_record(record)
            assert dict(await transaction.reserve_sequences(sequences)) == sequences

        await restored.mutate(restore)
        assert await restored.read(lambda tx: tx.scan_sequences()) == sequences
        assert not restored._sequences
        assert restored._step_run_admissions[record.key].counters == tuple(sequences[key] for key in streams)
        assert len(RUN_COUNTER_COLUMNS) == len(streams)
    finally:
        await store.close()
        await restored.close()
