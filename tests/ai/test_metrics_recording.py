#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit automatic-recording lifetimes and committed-only reads."""

import asyncio
from datetime import datetime, timezone

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.storage import WriteCache


def _observation(identity: str) -> Observation:
    return Observation(
        version=1, observation_id=identity, kind="test.recording",
        occurred_at=datetime.now(timezone.utc), source_namespace="example",
        tenant_id="tenant", status="SUCCEEDED", error_code=None,
        correlation={}, dimensions={}, measurements=(),
    )


class _ControlledStore(InMemoryMetricStore):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.task: asyncio.Task[object] | None = None

    async def put_observations(
        self, namespace: str, observations: tuple[Observation, ...],
    ) -> None:
        self.task = asyncio.current_task()
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        await super().put_observations(namespace, observations)


@pytest.mark.asyncio
async def test_direct_recorder_waits_in_caller_and_close_does_not_cancel_that_io() -> None:
    store = _ControlledStore()
    metrics = Metrics.from_store(store)
    recorder = metrics.recorder()
    writing = asyncio.create_task(recorder.try_record(_observation("direct")))
    await asyncio.wait_for(store.entered.wait(), 1)
    assert store.task is writing
    assert not writing.done()
    assert (await recorder.flush(timeout_seconds=0)).completed
    assert recorder.status().queue_capacity == recorder.status().queue_size == 0
    closing = asyncio.create_task(recorder.close())
    await asyncio.sleep(0)
    assert not closing.done()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert not store.cancelled
    assert not writing.done()
    store.release.set()
    assert await writing
    await recorder.close()
    assert await metrics.get_observation("direct") is not None
    assert recorder.status().persisted == 1
    assert not await recorder.try_record(_observation("closed"))


@pytest.mark.asyncio
async def test_direct_recording_preserves_caller_cancellation() -> None:
    store = _ControlledStore()
    recorder = Metrics.from_store(store).recorder()
    writing = asyncio.create_task(recorder.try_record(_observation("cancel")))
    await asyncio.wait_for(store.entered.wait(), 1)
    writing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writing
    assert store.cancelled
    status = recorder.status()
    assert status.accepted == status.lost == 1
    assert status.persisted == status.pending == 0
    await recorder.close()


@pytest.mark.asyncio
async def test_cached_recorder_accepts_without_source_io_and_reads_only_committed() -> None:
    store = _ControlledStore()
    metrics = Metrics.from_store(store, write_cache=WriteCache())
    recorder = metrics.recorder()
    assert await recorder.try_record(_observation("pending"))
    assert not store.entered.is_set()
    assert await metrics.get_observation("pending") is None
    flushing = asyncio.create_task(recorder.flush())
    await asyncio.wait_for(store.entered.wait(), 1)
    assert not flushing.done()
    assert await metrics.get_observation("pending") is None
    store.release.set()
    assert (await flushing).completed
    assert await metrics.get_observation("pending") is not None
    await recorder.close()


@pytest.mark.asyncio
async def test_shared_metrics_creates_independently_closed_recorders() -> None:
    metrics = Metrics.in_memory(write_cache=WriteCache())
    first, second = metrics.recorder(), metrics.recorder()
    assert await first.try_record(_observation("first"))
    assert await second.try_record(_observation("second"))
    await first.close()
    assert await metrics.get_observation("first") is not None
    assert await metrics.get_observation("second") is None
    assert second.status().accepting
    assert second.status().pending == 1
    assert await second.try_record(_observation("third"))
    await second.close()
    assert await metrics.get_observation("third") is not None
    assert first.status().accepted == 1
    assert second.status().persisted == 2


@pytest.mark.asyncio
async def test_configured_cache_does_not_weaken_explicit_record_observations() -> None:
    store = _ControlledStore()
    metrics = Metrics.from_store(store, write_cache=WriteCache())
    writing = asyncio.create_task(metrics.record_observations((_observation("strong"),)))
    await asyncio.wait_for(store.entered.wait(), 1)
    assert store.task is writing
    assert not writing.done()
    assert await metrics.get_observation("strong") is None
    store.release.set()
    await writing
    assert await metrics.get_observation("strong") is not None


@pytest.mark.asyncio
async def test_direct_recorder_fails_open_on_unconfirmed_storage() -> None:
    class FailingStore(InMemoryMetricStore):
        async def put_observations(
            self, namespace: str, observations: tuple[Observation, ...],
        ) -> None:
            await super().put_observations(namespace, observations)
            raise OSError("write response unavailable")

    metrics = Metrics.from_store(FailingStore())
    recorder = metrics.recorder()
    assert not await recorder.try_record(_observation("unknown"))
    assert recorder.status().lost == 1
    assert recorder.status().persisted == 0
    assert await metrics.get_observation("unknown") is not None
    await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", (False, True))
@pytest.mark.parametrize("timeout", (True, -1, float("nan"), float("inf")))
async def test_recorder_flush_rejects_invalid_deadlines(cached: bool, timeout: float) -> None:
    recorder = Metrics.in_memory(write_cache=WriteCache() if cached else None).recorder()
    try:
        with pytest.raises(AIError) as raised:
            await recorder.flush(timeout_seconds=timeout)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await recorder.close()
