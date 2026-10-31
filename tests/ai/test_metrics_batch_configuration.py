#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Synchronous dataset recording preserves delivery and Runtime ownership."""
from ._runtime_test_helpers import model_features

import asyncio
from pathlib import Path

import pytest

from linktools.ai.migrate import provision_metrics_sqlite
from linktools.ai.observe import MetricRecorder, Metrics
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage import BatchWriteScope, InMemoryCacheBackend

from .test_explicit_runtime_features import _FeatureModels
from .test_metrics_recording import _ControlledStore, _observation


@pytest.mark.asyncio
async def test_metrics_direct_and_explicit_writes_are_synchronous(tmp_path: Path) -> None:
    await provision_metrics_sqlite(tmp_path / "direct.sqlite")
    await provision_metrics_sqlite(tmp_path / "cached.sqlite")
    metrics = Metrics.sqlite(tmp_path / "direct.sqlite")
    recorder = metrics.recorder()
    try:
        assert await recorder.try_record(_observation("direct"))
        assert await metrics.get_observation("direct") is not None
        assert recorder.status().pending == 0
    finally:
        await recorder.close()
    cached = Metrics.sqlite(tmp_path / "cached.sqlite")
    await cached.record_observations((_observation("explicit"),))
    assert await cached.get_observation("explicit") is not None


@pytest.mark.asyncio
async def test_shared_metrics_has_independent_runtime_flush_and_close(tmp_path: Path) -> None:
    await provision_metrics_sqlite(tmp_path / "shared.sqlite")
    recorders: list[MetricRecorder] = []

    class RecordingMetrics(Metrics):
        def recorder(self) -> MetricRecorder:
            recorder = super().recorder()
            recorders.append(recorder)
            return recorder

    metrics = RecordingMetrics.sqlite(tmp_path / "shared.sqlite")
    async with Persistence.open('first', storage=PersistenceStorage.in_memory()) as persistence_129, Runtime.attach(persistence=persistence_129, features=model_features(_FeatureModels()), metrics=metrics) as first:
        async with Persistence.open('second', storage=PersistenceStorage.in_memory()) as persistence_130, Runtime.attach(persistence=persistence_130, features=model_features(_FeatureModels()), metrics=metrics) as second:
            assert await recorders[0].try_record(_observation("first"))
            assert await recorders[1].try_record(_observation("second"))
            assert await metrics.get_observation("first") is not None
            assert await metrics.get_observation("second") is not None
            assert (await first.metrics.flush()).completed
            assert await metrics.get_observation("first") is not None
            assert await metrics.get_observation("second") is not None
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


@pytest.mark.asyncio
async def test_metrics_recorder_owns_no_source_task() -> None:
    store = _ControlledStore()
    metrics = Metrics.from_store(store)
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
