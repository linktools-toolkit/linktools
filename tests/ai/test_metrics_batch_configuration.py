#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset batching settings preserve delivery and Runtime ownership."""
from ._runtime_test_helpers import model_features

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from linktools.ai.migrate import provision_metrics_sqlite
from linktools.ai.observe import MetricRecorder, Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.storage import BatchWriteScopeFactory, BatchWriteScope, InMemoryCacheBackend
from linktools.ai.core import CommitPolicy
from linktools.ai.storage import _write_cache

from .test_explicit_runtime_features import _FeatureModels
from .test_metrics_recording import _ControlledStore, _observation


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
    options = {"commit": CommitPolicy.batch(batch_size=128 if size is None else size, flush_interval=60.0, buffer=InMemoryCacheBackend(max_bytes=2**20))}
    metrics = Metrics.from_store(store, **options)
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
    options = {"commit": CommitPolicy.batch(batch_size=128, flush_interval=60.0 if interval is None else interval, buffer=InMemoryCacheBackend(max_bytes=2**20))}
    recorder = Metrics.from_store(store, **options).recorder()
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
        Metrics.sqlite(path, commit=CommitPolicy.batch(**(dict(batch_size=128, flush_interval=60.0) | settings)))
    assert not path.exists()
    with pytest.raises(ValueError):
        Metrics.in_memory(commit=CommitPolicy.batch(**(dict(batch_size=128, flush_interval=60.0) | settings)))


@pytest.mark.asyncio
async def test_metrics_batch_options_leave_direct_and_explicit_writes_synchronous(tmp_path: Path) -> None:
    await provision_metrics_sqlite(tmp_path / "direct.sqlite")
    await provision_metrics_sqlite(tmp_path / "cached.sqlite")
    metrics = Metrics.sqlite(tmp_path / "direct.sqlite", commit=CommitPolicy.batch(batch_size=256, flush_interval=300))
    recorder = metrics.recorder()
    try:
        assert await recorder.try_record(_observation("direct"))
        assert await metrics.get_observation("direct") is not None
        assert recorder.status().pending == 0
    finally:
        await recorder.close()
    cached = Metrics.sqlite(tmp_path / "cached.sqlite", commit=CommitPolicy.batch(buffer=InMemoryCacheBackend(max_bytes=2**20), batch_size=256, flush_interval=300))
    await cached.record_observations((_observation("explicit"),))
    assert await cached.get_observation("explicit") is not None


@pytest.mark.asyncio
async def test_shared_metrics_custom_batching_has_independent_runtime_flush_and_close(tmp_path: Path) -> None:
    await provision_metrics_sqlite(tmp_path / "shared.sqlite")
    recorders: list[MetricRecorder] = []

    class RecordingMetrics(Metrics):
        def recorder(self, *, batch_scopes: BatchWriteScopeFactory | None = None) -> MetricRecorder:
            recorder = super().recorder(batch_scopes=batch_scopes)
            recorders.append(recorder)
            return recorder

    metrics = RecordingMetrics.sqlite(tmp_path / "shared.sqlite", commit=CommitPolicy.batch(buffer=InMemoryCacheBackend(max_bytes=2**20), batch_size=2, flush_interval=300))
    async with Runtime.open("first", storage=RuntimeStorage.in_memory(), features=model_features(_FeatureModels()), metrics=metrics) as first:
        async with Runtime.open("second", storage=RuntimeStorage.in_memory(), features=model_features(_FeatureModels()), metrics=metrics) as second:
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


@pytest.mark.parametrize("factory", ["constructor", "from_store", "in_memory", "sqlite", "sql"])
@pytest.mark.parametrize("commit", [None, CommitPolicy.terminal(), CommitPolicy.batch(batch_size=257, flush_interval=60.0)])
def test_metrics_rejects_unsupported_commit_policy_before_opening_source(
    tmp_path: Path, factory: str, commit: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.observe import _metrics, _sql

    def unexpected_source(*args, **kwargs):
        pytest.fail("invalid policy must be rejected before opening a source")

    monkeypatch.setattr(_metrics, "SQLiteMetricStore", unexpected_source)
    monkeypatch.setattr(_sql, "SqlMetricStore", unexpected_source)
    with pytest.raises(TypeError if commit is None else ValueError):
        if factory == "constructor":
            Metrics(_Store(), commit=commit)
        elif factory == "from_store":
            Metrics.from_store(_Store(), commit=commit)
        elif factory == "in_memory":
            Metrics.in_memory(commit=commit)
        elif factory == "sqlite":
            Metrics.sqlite(tmp_path / "not-created.sqlite", commit=commit)
        else:
            Metrics.sql(object(), commit=commit)
    assert not (tmp_path / "not-created.sqlite").exists()


@pytest.mark.asyncio
async def test_metrics_sync_policy_owns_no_source_task() -> None:
    store = _ControlledStore()
    metrics = Metrics.from_store(store, commit=CommitPolicy.sync())
    recorder = metrics.recorder()
    writing = asyncio.create_task(recorder.try_record(_observation("sync")))
    closing = None
    try:
        await asyncio.wait_for(store.entered.wait(), 1)
        assert store.task is writing
        assert not writing.done()
        assert recorder.status().queue_capacity == recorder.status().queue_size == 0
        assert (await recorder.flush(timeout_seconds=0)).completed
        closing = asyncio.create_task(recorder.close())
        await asyncio.sleep(0)
        assert not closing.done()
        assert not store.cancelled
        store.release.set()
        assert await writing
        await closing
        assert await metrics.get_observation("sync") is not None
        reused = metrics.recorder()
        try:
            assert await reused.try_record(_observation("reused-sync"))
            assert await metrics.get_observation("reused-sync") is not None
        finally:
            await reused.close()
    finally:
        store.release.set()
        await asyncio.gather(writing, *(() if closing is None else (closing,)), return_exceptions=True)
        await recorder.close()
