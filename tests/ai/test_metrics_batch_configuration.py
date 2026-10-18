#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset batching settings preserve delivery and Runtime ownership."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from linktools.ai.migrate import provision_metrics_sqlite
from linktools.ai.observe import MetricRecorder, Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.storage import BatchWriteScope, InMemoryCacheBackend
from linktools.ai.storage import _write_cache

from .test_explicit_runtime_features import _FeatureModels
from .test_metrics_recording import _observation


class _Store(InMemoryMetricStore):
    def __init__(self) -> None:
        super().__init__()
        self.written = asyncio.Event()
        self.batch_sizes: list[int] = []

    async def put_observations(self, namespace: str, observations: Sequence[Observation]) -> None:
        await super().put_observations(namespace, observations)
        self.batch_sizes.append(len(observations))
        self.written.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [None, 2, 256])
async def test_metrics_batch_threshold_flushes_without_explicit_flush(size: int | None) -> None:
    store = _Store()
    options = {} if size is None else {"batch_size": size}
    metrics = Metrics.from_store(store, cache=InMemoryCacheBackend(max_bytes=2**20), **options)
    recorder = metrics.recorder()
    threshold = 128 if size is None else size
    try:
        for index in range(threshold - 1):
            assert await recorder.try_record(_observation(str(index)))
        assert not store.written.is_set()
        assert await recorder.try_record(_observation("last"))
        await asyncio.wait_for(store.written.wait(), 1)
        assert store.batch_sizes == [threshold]
        assert await metrics.get_observation("last") is not None
    finally:
        await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("interval", [None, 7.0])
async def test_metrics_batch_interval_flushes_without_another_admission(monkeypatch: pytest.MonkeyPatch, interval: float | None) -> None:
    clock = [100.0]
    expected = 60.0 if interval is None else interval
    wait_for = asyncio.wait_for
    elapsed = []

    async def advancing_wait(awaitable, timeout):
        if timeout == expected:
            elapsed.append(timeout)
            clock[0] += timeout
            awaitable.close()
            raise asyncio.TimeoutError
        return await wait_for(awaitable, timeout)

    monkeypatch.setattr(_write_cache, "monotonic", lambda: clock[0])
    monkeypatch.setattr(_write_cache.asyncio, "wait_for", advancing_wait)
    store = _Store()
    options = {} if interval is None else {"flush_interval": interval}
    recorder = Metrics.from_store(store, cache=InMemoryCacheBackend(max_bytes=2**20), **options).recorder()
    try:
        assert await recorder.try_record(_observation("timed"))
        await wait_for(store.written.wait(), 1)
        assert elapsed == [expected]
        assert store.batch_sizes == [1]
    finally:
        await recorder.close()


@pytest.mark.parametrize("settings", [
    {"batch_size": 0}, {"batch_size": True}, {"batch_size": 1.5}, {"batch_size": 257},
    {"flush_interval": 0}, {"flush_interval": True}, {"flush_interval": float("nan")},
    {"flush_interval": float("inf")}, {"flush_interval": -1},
])
def test_metrics_batch_settings_reject_invalid_values_before_sqlite_creation(tmp_path: Path, settings: dict[str, object]) -> None:
    path = tmp_path / "not-created.sqlite"
    with pytest.raises(ValueError):
        Metrics.sqlite(path, **settings)
    assert not path.exists()
    with pytest.raises(ValueError):
        Metrics.in_memory(**settings)


@pytest.mark.asyncio
async def test_metrics_batch_options_leave_direct_and_explicit_writes_synchronous(tmp_path: Path) -> None:
    await provision_metrics_sqlite(tmp_path / "direct.sqlite")
    await provision_metrics_sqlite(tmp_path / "cached.sqlite")
    metrics = Metrics.sqlite(tmp_path / "direct.sqlite", batch_size=256, flush_interval=300)
    recorder = metrics.recorder()
    try:
        assert await recorder.try_record(_observation("direct"))
        assert await metrics.get_observation("direct") is not None
        assert recorder.status().pending == 0
    finally:
        await recorder.close()
    cached = Metrics.sqlite(tmp_path / "cached.sqlite", cache=InMemoryCacheBackend(max_bytes=2**20), batch_size=256, flush_interval=300)
    await cached.record_observations((_observation("explicit"),))
    assert await cached.get_observation("explicit") is not None


@pytest.mark.asyncio
async def test_shared_metrics_custom_batching_has_independent_runtime_flush_and_close(tmp_path: Path) -> None:
    await provision_metrics_sqlite(tmp_path / "shared.sqlite")
    recorders: list[MetricRecorder] = []

    class RecordingMetrics(Metrics):
        def recorder(self) -> MetricRecorder:
            recorder = super().recorder()
            recorders.append(recorder)
            return recorder

    metrics = RecordingMetrics.sqlite(tmp_path / "shared.sqlite", cache=InMemoryCacheBackend(max_bytes=2**20), batch_size=2, flush_interval=300)
    async with Runtime.open("first", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), metrics=metrics) as first:
        async with Runtime.open("second", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), metrics=metrics) as second:
            assert await recorders[0].try_record(_observation("first"))
            assert await recorders[1].try_record(_observation("second"))
            assert await metrics.get_observation("first") is None
            assert await metrics.get_observation("second") is None
            assert (await first.metrics.flush()).completed
            assert await metrics.get_observation("first") is not None
            assert await metrics.get_observation("second") is None
        assert await metrics.get_observation("second") is not None
        assert first.metrics.status().accepting
        assert await recorders[0].try_record(_observation("remaining"))
    assert await metrics.get_observation("remaining") is not None
    recorder = metrics.recorder()
    assert await recorder.try_record(_observation("reused"))
    await recorder.close()
    assert await metrics.get_observation("reused") is not None


@pytest.mark.parametrize("settings", [{"batch_size": 0}, {"batch_size": True}, {"flush_interval": float("inf")}, {"flush_interval": 0}])
def test_batch_scope_validates_settings_without_starting_work(settings: dict[str, object]) -> None:
    async def write(batch) -> None:
        raise AssertionError("not reached")
    with pytest.raises(ValueError):
        BatchWriteScope(InMemoryCacheBackend(max_bytes=1024), write, **settings)
