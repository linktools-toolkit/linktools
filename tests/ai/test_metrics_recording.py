#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit automatic-recording lifetimes and committed-only reads."""

import asyncio
from datetime import datetime, timezone

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import (
    MetricAggregation, MetricDefinition, MetricQuery, Metrics, MetricSource, MetricType, MetricWindow, Observation,
)
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.storage import InMemoryCacheBackend, InMemoryStreamBackend


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
    metrics = Metrics.from_store(store, cache=InMemoryCacheBackend(max_bytes=1024 * 1024))
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
    metrics = Metrics.in_memory(cache=InMemoryCacheBackend(max_bytes=1024 * 1024))
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
    metrics = Metrics.from_store(store, cache=InMemoryCacheBackend(max_bytes=1024 * 1024))
    writing = asyncio.create_task(metrics.record_observations((_observation("strong"),)))
    await asyncio.wait_for(store.entered.wait(), 1)
    assert store.task is writing
    assert not writing.done()
    assert await metrics.get_observation("strong") is None
    store.release.set()
    await writing
    assert await metrics.get_observation("strong") is not None


@pytest.mark.asyncio
async def test_cached_recorder_respects_metric_write_batch_limit() -> None:
    class Store(InMemoryMetricStore):
        async def put_observations(
            self, namespace: str, observations: tuple[Observation, ...],
        ) -> None:
            assert len(observations) <= 128
            await super().put_observations(namespace, observations)

    cache = InMemoryCacheBackend(max_bytes=1024 * 1024)
    metrics = Metrics.from_store(Store(), cache=cache)
    observations = tuple(_observation(f"batch-{index}") for index in range(257))
    recorder = metrics.recorder()
    try:
        for observation in observations:
            assert await recorder.try_record(observation)
        flushed = await recorder.flush()
        assert flushed.completed
        assert flushed.status.persisted == flushed.status.accepted == len(observations)
        assert flushed.status.pending == flushed.status.lost == flushed.status.write_failures == 0
        for observation in observations:
            assert await metrics.get_observation(observation.observation_id) == observation
        assert recorder.status().queue_capacity is None
        with pytest.raises(AIError, match="REQUEST_FIELD_INVALID"):
            await metrics.record_observations(observations)
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_cached_recorder_preserves_confirmed_batch_when_later_batch_fails() -> None:
    class Store(InMemoryMetricStore):
        async def put_observations(
            self, namespace: str, observations: tuple[Observation, ...],
        ) -> None:
            if any(item.observation_id == "batch-256" for item in observations):
                raise OSError("tail batch failed")
            await super().put_observations(namespace, observations)

    metrics = Metrics.from_store(
        Store(), cache=InMemoryCacheBackend(max_bytes=1024 * 1024),
    )
    observations = tuple(_observation(f"batch-{index}") for index in range(257))
    recorder = metrics.recorder()
    try:
        for observation in observations:
            assert await recorder.try_record(observation)
        flushed = await recorder.flush()
        assert flushed.completed
        assert flushed.status.accepted == 257
        assert flushed.status.persisted == 256
        assert flushed.status.pending == 0
        assert flushed.status.lost == flushed.status.write_failures == 1
        for observation in observations[:256]:
            assert await metrics.get_observation(observation.observation_id) == observation
        assert await metrics.get_observation("batch-256") is None
    finally:
        await recorder.close()


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
    recorder = Metrics.in_memory(cache=InMemoryCacheBackend(max_bytes=1024 * 1024) if cached else None).recorder()
    try:
        with pytest.raises(AIError) as raised:
            await recorder.flush(timeout_seconds=timeout)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("none", "content", "stream"))
async def test_cache_without_buffer_methods_uses_direct_source(kind: str) -> None:
    class ContentCache:
        async def get(self, key: object) -> None:
            raise AssertionError("automatic recording does not use hot content")

        async def put(self, key: object, content: bytes) -> None:
            raise AssertionError("automatic recording does not use hot content")

        async def delete(self, key: object) -> None:
            raise AssertionError("automatic recording does not use hot content")

    cache = {"none": None, "content": ContentCache(), "stream": InMemoryStreamBackend(max_bytes=1024)}[kind]
    store = _ControlledStore()
    recorder = Metrics.from_store(store, cache=cache).recorder()
    writing = asyncio.create_task(recorder.try_record(_observation("direct-capability")))
    await asyncio.wait_for(store.entered.wait(), 1)
    assert store.task is writing
    assert not writing.done()
    assert recorder.status().queue_capacity == 0
    store.release.set()
    assert await writing
    await recorder.close()


@pytest.mark.parametrize("names", (("append",), ("read_batch", "ack"), ("append", "ack")))
def test_metrics_rejects_partial_buffer_capabilities(names: tuple[str, ...]) -> None:
    class PartialCache:
        pass

    async def method(*args: object, **kwargs: object) -> None:
        pass

    cache = PartialCache()
    for name in names:
        setattr(cache, name, method)
    with pytest.raises(TypeError, match="append, read_batch and ack"):
        Metrics.in_memory(cache=cache)


@pytest.mark.asyncio
async def test_shared_cache_budget_is_not_freed_by_another_recorder_close() -> None:
    cache = InMemoryCacheBackend(max_bytes=1024 * 1024, max_size=1)
    metrics = Metrics.in_memory(cache=cache)
    first, second = metrics.recorder(), metrics.recorder()
    assert await first.try_record(_observation("first-budget"))
    assert not await second.try_record(_observation("second-rejected"))
    await second.close()
    assert first.status().pending == 1
    assert (await first.flush()).completed
    third = metrics.recorder()
    assert await third.try_record(_observation("third-budget"))
    await third.close()
    await first.close()
    assert await metrics.get_observation("first-budget") is not None
    assert await metrics.get_observation("third-budget") is not None


@pytest.mark.asyncio
async def test_explicit_record_and_query_remain_authoritative_with_buffer_cache() -> None:
    from datetime import timedelta

    cache = InMemoryCacheBackend(max_bytes=1024 * 1024)
    store = _ControlledStore()
    metrics = Metrics.from_store(store, cache=cache)
    await metrics.define(MetricDefinition(
        name="test.value", revision=1, observation_kind="test.recording",
        source=MetricSource.measurement("value"), metric_type=MetricType.COUNTER,
        unit="1", default_aggregation=MetricAggregation.SUM,
    ))
    start = datetime.now(timezone.utc) - timedelta(seconds=1)
    writing = asyncio.create_task(metrics.record("test.value", 7, observation_id="strong-record"))
    await asyncio.wait_for(store.entered.wait(), 1)
    assert store.task is writing
    assert not writing.done()
    query = MetricQuery("test.value", MetricWindow.between(start, datetime.now(timezone.utc) + timedelta(seconds=1)))
    assert (await metrics.query(query)).points[0].sample_count == 0
    store.release.set()
    assert await writing == "strong-record"
    result = await metrics.query(query)
    assert result.points[0].value == 7


@pytest.mark.asyncio
async def test_buffer_only_cache_holds_observation_bytes_and_remains_borrowed() -> None:
    class BufferCache:
        def __init__(self) -> None:
            self.backend = InMemoryCacheBackend(max_bytes=1024 * 1024)
            self.contents: list[bytes] = []
            self.closed = False

        async def append(
            self, scope: str, id: str, key: object, content: bytes | None, *, wait: bool,
        ) -> bool:
            assert isinstance(content, bytes)
            self.contents.append(content)
            return await self.backend.append(scope, id, key, content, wait=wait)

        async def read_batch(
            self, scope: str, *, limit: int,
        ) -> tuple[tuple[str, object, bytes | None], ...]:
            return await self.backend.read_batch(scope, limit=limit)

        async def ack(self, scope: str, ids: tuple[str, ...]) -> None:
            await self.backend.ack(scope, ids)

        async def close(self) -> None:
            self.closed = True

    import json

    cache = BufferCache()
    metrics = Metrics.in_memory(namespace="buffer-only", cache=cache)
    observation = _observation("encoded")
    recorder = metrics.recorder()
    assert await recorder.try_record(observation)
    payload = json.loads(cache.contents[0])
    assert payload["namespace"] == "buffer-only"
    assert payload["observation"]["observation_id"] == "encoded"
    assert await metrics.get_observation("encoded") is None
    await recorder.close()
    assert await metrics.get_observation("encoded") == observation
    assert not cache.closed


@pytest.mark.asyncio
async def test_cache_admission_error_is_fail_open_but_close_exposes_unknown_cleanup() -> None:
    class FailingCache(InMemoryCacheBackend):
        async def append(
            self, scope: str, id: str, key: object, content: bytes | None, *, wait: bool,
        ) -> bool:
            raise OSError("cache unavailable")

    metrics = Metrics.in_memory(cache=FailingCache(max_bytes=1024))
    recorder = metrics.recorder()
    assert not await recorder.try_record(_observation("cache-failed"))
    assert recorder.status().last_failure_code == "CACHE_OPERATION_FAILED"
    assert recorder.status().rejected == 0
    assert recorder.status().write_failures == 1
    assert await metrics.get_observation("cache-failed") is None
    with pytest.raises(AIError, match="STORAGE_COMMIT_UNKNOWN"):
        await recorder.close()


@pytest.mark.asyncio
async def test_corrupt_buffer_content_cannot_become_a_committed_observation() -> None:
    class CorruptCache(InMemoryCacheBackend):
        async def read_batch(
            self, scope: str, *, limit: int,
        ) -> tuple[tuple[str, object, bytes | None], ...]:
            batch = await super().read_batch(scope, limit=limit)
            return tuple((id, key, b"{") for id, key, _ in batch)

    metrics = Metrics.in_memory(cache=CorruptCache(max_bytes=1024))
    recorder = metrics.recorder()
    assert await recorder.try_record(_observation("corrupt"))
    await recorder.close()
    assert await metrics.get_observation("corrupt") is None
    assert recorder.status().lost == recorder.status().write_failures == 1
