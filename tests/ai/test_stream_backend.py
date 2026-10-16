#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordered stream retention, publication and independent-reader contracts."""

import asyncio

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryStreamBackend, StreamBatch


async def _read(backend: InMemoryStreamBackend, key: bytes = b"stream", after: int = 0) -> StreamBatch:
    return await backend.read(key, after, limit=100, wait_seconds=0)


@pytest.mark.asyncio
async def test_stream_records_are_ordered_and_reads_are_independent() -> None:
    backend = InMemoryStreamBackend(max_bytes=20)
    writer = await backend.open(b"stream", b"producer")
    await writer.append(1, b"one")
    await writer.append(2, b"two")
    expected = StreamBatch(((1, b"one"), (2, b"two")), gap=False, sealed=False)

    first, second = await asyncio.gather(_read(backend), _read(backend))

    assert first == second == expected
    assert await _read(backend, after=1) == StreamBatch(((2, b"two"),), False, False)
    assert await _read(backend, after=2) == StreamBatch((), False, False)


@pytest.mark.asyncio
async def test_existing_producer_open_and_identical_write_are_idempotent() -> None:
    backend = InMemoryStreamBackend(max_bytes=10)
    first = await backend.open(b"stream", b"producer")
    second = await backend.open(b"stream", b"producer")
    await first.append(1, b"value")
    await second.append(1, b"value")

    assert (await _read(backend)).records == ((1, b"value"),)
    with pytest.raises(AIError) as raised:
        await backend.open(b"stream", b"other")
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("sequence,payload", ((1, b"changed"), (3, b"hole")))
async def test_conflicting_and_out_of_order_writes_do_not_change_stream(
    sequence: int, payload: bytes,
) -> None:
    backend = InMemoryStreamBackend(max_bytes=20)
    writer = await backend.open(b"stream", b"producer")
    await writer.append(1, b"first")
    with pytest.raises(AIError) as raised:
        await writer.append(sequence, payload)
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    assert (await _read(backend)).records == ((1, b"first"),)


@pytest.mark.asyncio
async def test_seal_is_atomic_idempotent_and_reported_only_at_batch_tail() -> None:
    backend = InMemoryStreamBackend(max_bytes=20)
    writer = await backend.open(b"stream", b"producer")
    await writer.append(1, b"first")
    await writer.seal(2, b"final")
    await writer.seal(2, b"final")
    await writer.append(1, b"first")

    assert await backend.read(b"stream", 0, limit=1, wait_seconds=0) == StreamBatch(
        ((1, b"first"),), False, False,
    )
    assert await backend.read(b"stream", 1, limit=1, wait_seconds=0) == StreamBatch(
        ((2, b"final"),), False, True,
    )
    assert await _read(backend, after=2) == StreamBatch((), False, True)
    for sequence, payload in ((2, b"final"), (3, b"late")):
        with pytest.raises(AIError) as raised:
            await writer.append(sequence, payload)
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    with pytest.raises(AIError):
        await writer.seal(2, b"changed")


@pytest.mark.asyncio
async def test_seal_cannot_relabel_an_existing_nonterminal_record() -> None:
    backend = InMemoryStreamBackend(max_bytes=10)
    writer = await backend.open(b"stream", b"producer")
    await writer.append(1, b"first")
    with pytest.raises(AIError):
        await writer.seal(1, b"first")
    assert (await _read(backend)).sealed is False


@pytest.mark.asyncio
async def test_per_stream_trim_reports_gap_and_rejects_unverifiable_retry() -> None:
    backend = InMemoryStreamBackend(max_bytes=4)
    writer = await backend.open(b"stream", b"producer")
    for sequence in range(1, 4):
        await writer.append(sequence, b"ab")

    assert await _read(backend) == StreamBatch(((2, b"ab"), (3, b"ab")), True, False)
    assert await _read(backend, after=1) == StreamBatch(((2, b"ab"), (3, b"ab")), False, False)
    assert await _read(backend, after=4) == StreamBatch((), True, False)
    with pytest.raises(AIError) as raised:
        await writer.append(1, b"ab")
    assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    await writer.append(3, b"ab")


@pytest.mark.asyncio
async def test_total_capacity_evicts_other_stream_and_revokes_old_writer() -> None:
    backend = InMemoryStreamBackend(max_bytes=4)
    old = await backend.open(b"old", b"producer")
    await old.append(1, b"old")
    current = await backend.open(b"current", b"producer")
    await current.append(1, b"new")

    assert await _read(backend, b"old") == StreamBatch((), True, False)
    assert (await _read(backend, b"current")).records == ((1, b"new"),)
    # Even an erroneous key reuse must not restore an already-revoked handle.
    reopened = await backend.open(b"old", b"producer")
    for operation in (old.append, old.seal):
        with pytest.raises(AIError) as raised:
            await operation(2, b"late")
        assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    await reopened.append(1, b"ok")
    assert (await _read(backend, b"old")).records == ((1, b"ok"),)


@pytest.mark.asyncio
async def test_empty_streams_and_records_are_capacity_bounded() -> None:
    backend = InMemoryStreamBackend(max_bytes=2)
    await backend.open(b"a", b"producer")
    await backend.open(b"b", b"producer")
    writer = await backend.open(b"c", b"producer")
    assert (await _read(backend, b"a")).gap
    assert not (await _read(backend, b"b")).gap
    for sequence in range(1, 4):
        await writer.append(sequence, b"")
    assert (await _read(backend, b"b")).gap
    assert await _read(backend, b"c") == StreamBatch(((2, b""), (3, b"")), True, False)


@pytest.mark.asyncio
async def test_oversized_record_and_zero_budget_reject_without_mutation() -> None:
    disabled = InMemoryStreamBackend(max_bytes=0)
    with pytest.raises(AIError) as raised:
        await disabled.open(b"stream", b"producer")
    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
    assert (await _read(disabled)).gap

    backend = InMemoryStreamBackend(max_bytes=2)
    writer = await backend.open(b"stream", b"producer")
    await writer.append(1, b"a")
    for operation in (writer.append, writer.seal):
        with pytest.raises(AIError) as raised:
            await operation(2, b"large")
        assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE
        assert await _read(backend) == StreamBatch(((1, b"a"),), False, False)
    await writer.seal(2, b"b")
    assert (await _read(backend)).sealed


@pytest.mark.asyncio
@pytest.mark.parametrize("seal", (False, True))
async def test_all_waiting_readers_wake_for_publication(seal: bool) -> None:
    backend = InMemoryStreamBackend(max_bytes=10)
    writer = await backend.open(b"stream", b"producer")
    readers = [asyncio.create_task(backend.read(b"stream", 0, limit=1, wait_seconds=5)) for _ in range(2)]
    await asyncio.sleep(0)
    operation = writer.seal if seal else writer.append
    await operation(1, b"value")

    results = await asyncio.wait_for(asyncio.gather(*readers), timeout=1)
    assert results == [StreamBatch(((1, b"value"),), False, seal)] * 2


@pytest.mark.asyncio
async def test_eviction_wakes_waiting_reader_with_gap() -> None:
    backend = InMemoryStreamBackend(max_bytes=1)
    await backend.open(b"stream", b"producer")
    reader = asyncio.create_task(backend.read(b"stream", 0, limit=1, wait_seconds=5))
    await asyncio.sleep(0)
    await backend.open(b"other", b"producer")
    assert await asyncio.wait_for(reader, timeout=1) == StreamBatch((), True, False)


@pytest.mark.asyncio
async def test_waiting_read_detects_key_replacement_even_with_same_producer() -> None:
    backend = InMemoryStreamBackend(max_bytes=1)
    await backend.open(b"stream", b"producer")
    reader = asyncio.create_task(backend.read(b"stream", 0, limit=1, wait_seconds=5))
    await asyncio.sleep(0)
    await backend.open(b"other", b"producer")
    new = await backend.open(b"stream", b"producer")
    await new.append(1, b"x")
    assert await asyncio.wait_for(reader, timeout=1) == StreamBatch((), True, False)


@pytest.mark.asyncio
async def test_wait_timeout_and_cancel_do_not_cancel_producer_or_other_reader() -> None:
    backend = InMemoryStreamBackend(max_bytes=10)
    writer = await backend.open(b"stream", b"producer")
    assert await backend.read(b"stream", 0, limit=1, wait_seconds=0.001) == StreamBatch((), False, False)
    cancelled = asyncio.create_task(backend.read(b"stream", 0, limit=1, wait_seconds=5))
    reader = asyncio.create_task(backend.read(b"stream", 0, limit=1, wait_seconds=5))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    await writer.append(1, b"value")
    assert await asyncio.wait_for(reader, timeout=1) == StreamBatch(((1, b"value"),), False, False)


@pytest.mark.asyncio
async def test_cancelled_write_waiting_for_lock_has_no_effect() -> None:
    backend = InMemoryStreamBackend(max_bytes=10)
    writer = await backend.open(b"stream", b"producer")
    # Hold the serialization boundary to deterministically cancel before admission.
    async with backend._lock:
        pending = asyncio.create_task(writer.append(1, b"value"))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert await _read(backend) == StreamBatch((), False, False)
    await writer.append(1, b"value")


@pytest.mark.parametrize("max_bytes", (-1, True, 1.5))
def test_invalid_capacity_is_rejected(max_bytes: int) -> None:
    with pytest.raises(ValueError):
        InMemoryStreamBackend(max_bytes=max_bytes)


@pytest.mark.asyncio
@pytest.mark.parametrize("after,limit,wait", ((-1, 1, 0), (True, 1, 0), (0, 0, 0), (0, 1, -1), (0, 1, float("inf"))))
async def test_invalid_read_arguments_are_rejected(after: int, limit: int, wait: float) -> None:
    backend = InMemoryStreamBackend(max_bytes=1)
    with pytest.raises(ValueError):
        await backend.read(b"stream", after, limit=limit, wait_seconds=wait)
