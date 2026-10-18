#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One Runtime owns recording/drain, while the Metrics dataset stays shared."""

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.observe import Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime._metrics_service import MetricsService
from linktools.ai.storage import InMemoryCacheBackend


def _observation(identity: str) -> Observation:
    return Observation(
        version=1, observation_id=identity, kind="business.sample",
        occurred_at=datetime.now(timezone.utc), source_namespace="business",
        tenant_id="tenant", status="SUCCEEDED", error_code=None,
        correlation={}, dimensions={}, measurements=(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_binding_owns_only_its_recording_and_reuses_dataset(cached: bool) -> None:
    class BorrowedCache(InMemoryCacheBackend):
        async def close(self) -> None:
            pytest.fail("recording must not close the borrowed cache")

    metrics = Metrics.in_memory(cache=BorrowedCache(max_bytes=2**20) if cached else None)
    first, second = MetricsService(metrics), MetricsService(metrics)
    try:
        assert await first.try_record(_observation("first"))
        assert await second.try_record(_observation("second"))
        if cached:
            assert await metrics.get_observation("first") is None
            assert await metrics.get_observation("second") is None
        await first.close()
        assert await metrics.get_observation("first") is not None
        assert second.status().accepting
        if cached:
            assert await metrics.get_observation("second") is None
            assert second.status().pending == 1
        await metrics.record_observations((_observation("explicit-business"),))
        assert await metrics.get_observation("explicit-business") is not None
        assert await second.try_record(_observation("after-first-close"))
        assert not await first.try_record(_observation("closed"))
    finally:
        await first.close()
        await second.close()
    assert await metrics.get_observation("second") is not None
    assert await metrics.get_observation("after-first-close") is not None


@pytest.mark.asyncio
async def test_shared_dataset_creates_independent_runtime_services() -> None:
    dataset = Metrics.in_memory(cache=InMemoryCacheBackend(max_bytes=2**20))
    first, second = MetricsService(dataset), MetricsService(dataset)
    try:
        assert first is not second
        await first.close()
        assert second.status().accepting
        assert await second.try_record(_observation("second-live"))
    finally:
        await first.close()
        await second.close()
    assert await dataset.get_observation("second-live") is not None


@pytest.mark.asyncio
async def test_binding_close_cancellation_does_not_cancel_direct_source_or_other_binding() -> None:
    class BlockingStore(InMemoryMetricStore):
        def __init__(self):
            super().__init__()
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.cancelled = False

        async def put_observations(self, namespace, observations):
            if observations[0].observation_id == "slow":
                self.entered.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            await super().put_observations(namespace, observations)

    store = BlockingStore()
    metrics = Metrics.from_store(store)
    first, second = MetricsService(metrics), MetricsService(metrics)
    writing = asyncio.create_task(first.try_record(_observation("slow")))
    await asyncio.wait_for(store.entered.wait(), 1)
    closing = asyncio.create_task(first.close())
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert not store.cancelled and not writing.done()
    assert await second.try_record(_observation("other-runtime"))
    store.release.set()
    assert await writing
    await first.close()
    await second.close()
    assert await metrics.get_observation("slow") is not None


@pytest.mark.asyncio
async def test_binding_failure_does_not_disable_another_runtime_or_explicit_recording() -> None:
    class FailingOnceStore(InMemoryMetricStore):
        failed = False

        async def put_observations(self, namespace, observations):
            if not self.failed:
                self.failed = True
                raise OSError("source unavailable")
            await super().put_observations(namespace, observations)

    metrics = Metrics.from_store(FailingOnceStore())
    first, second = MetricsService(metrics), MetricsService(metrics)
    try:
        assert not await first.try_record(_observation("failed"))
        assert first.status().unconfirmed == 1
        assert await second.try_record(_observation("other"))
        await metrics.record_observations((_observation("business"),))
        assert await metrics.get_observation("business") is not None
    finally:
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_recording_bindings_borrow_shared_sql_engine(tmp_path, monkeypatch) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.migrate import provision_metrics_database

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'metrics.db'}")
    await provision_metrics_database(engine)
    metrics = Metrics.sql(engine, cache=InMemoryCacheBackend(max_bytes=2**20))
    first, second = MetricsService(metrics), MetricsService(metrics)
    async def unexpected_dispose(*args, **kwargs):
        pytest.fail("recording must not dispose the borrowed SQL engine")
    try:
        with monkeypatch.context() as guard:
            guard.setattr(type(engine), "dispose", unexpected_dispose)
            assert await first.try_record(_observation("first-sql"))
            assert await second.try_record(_observation("second-sql"))
            await first.close()
            assert await metrics.get_observation("first-sql") is not None
            assert await metrics.get_observation("second-sql") is None
            assert second.status().pending == 1
            await second.close()
            assert await metrics.get_observation("second-sql") is not None
            await metrics.record_observations((_observation("after-close-sql"),))
            assert await metrics.get_observation("after-close-sql") is not None
    finally:
        await first.close()
        await second.close()
        await engine.dispose()


def test_metrics_service_requires_explicit_dataset() -> None:
    with pytest.raises(TypeError):
        MetricsService()
    with pytest.raises(TypeError, match="metrics must be Metrics"):
        MetricsService(None)
