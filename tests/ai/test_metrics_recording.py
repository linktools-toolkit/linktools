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
    assert status.accepted == status.unconfirmed == 1
    assert status.persisted == status.pending == 0
    await recorder.close()


@pytest.mark.asyncio
async def test_unconfirmed_recording_can_already_exist_in_source() -> None:
    class CommittedThenCancelledStore(InMemoryMetricStore):
        async def put_observations(
            self, namespace: str, observations: tuple[Observation, ...],
        ) -> None:
            await super().put_observations(namespace, observations)
            raise asyncio.CancelledError

    metrics = Metrics.from_store(CommittedThenCancelledStore())
    recorder = metrics.recorder()
    observation = _observation("committed-without-confirmation")
    with pytest.raises(asyncio.CancelledError):
        await recorder.try_record(observation)
    result = await recorder.flush()
    assert result.completed
    assert result.status.accepted == result.status.unconfirmed == 1
    assert result.status.persisted == result.status.pending == 0
    assert await metrics.get_observation(observation.observation_id) == observation
    await recorder.close()


@pytest.mark.asyncio
async def test_recorder_reads_only_committed_during_source_io() -> None:
    store = _ControlledStore()
    metrics = Metrics.from_store(store)
    recorder = metrics.recorder()
    writing = asyncio.create_task(recorder.try_record(_observation("pending")))
    await asyncio.wait_for(store.entered.wait(), 1)
    assert not writing.done()
    assert await metrics.get_observation("pending") is None
    store.release.set()
    assert await writing
    assert await metrics.get_observation("pending") is not None
    await recorder.close()


@pytest.mark.asyncio
async def test_shared_metrics_creates_independently_closed_recorders() -> None:
    metrics = Metrics.in_memory()
    first, second = metrics.recorder(), metrics.recorder()
    assert await first.try_record(_observation("first"))
    assert await second.try_record(_observation("second"))
    await first.close()
    assert await metrics.get_observation("first") is not None
    assert await metrics.get_observation("second") is not None
    assert second.status().accepting
    assert second.status().pending == 0
    assert await second.try_record(_observation("third"))
    await second.close()
    assert await metrics.get_observation("third") is not None
    assert first.status().accepted == 1
    assert second.status().persisted == 2


@pytest.mark.asyncio
async def test_explicit_record_observations_waits_for_source() -> None:
    store = _ControlledStore()
    metrics = Metrics.from_store(store)
    writing = asyncio.create_task(metrics.record_observations((_observation("strong"),)))
    await asyncio.wait_for(store.entered.wait(), 1)
    assert store.task is writing
    assert not writing.done()
    assert await metrics.get_observation("strong") is None
    store.release.set()
    await writing
    assert await metrics.get_observation("strong") is not None


@pytest.mark.asyncio
async def test_recorder_preserves_confirmed_observations_when_later_write_fails() -> None:
    class Store(InMemoryMetricStore):
        async def put_observations(
            self, namespace: str, observations: tuple[Observation, ...],
        ) -> None:
            if any(item.observation_id == "batch-256" for item in observations):
                raise OSError("tail batch failed")
            await super().put_observations(namespace, observations)

    metrics = Metrics.from_store(Store())
    observations = tuple(_observation(f"batch-{index}") for index in range(257))
    recorder = metrics.recorder()
    try:
        for observation in observations:
            assert await recorder.try_record(observation) is (observation.observation_id != "batch-256")
        flushed = await recorder.flush()
        assert flushed.completed
        assert flushed.status.accepted == 257
        assert flushed.status.persisted == 256
        assert flushed.status.pending == 0
        assert flushed.status.unconfirmed == flushed.status.write_failures == 1
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
    assert recorder.status().unconfirmed == 1
    assert recorder.status().persisted == 0
    assert await metrics.get_observation("unknown") is not None
    await recorder.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", (True, -1, float("nan"), float("inf")))
async def test_recorder_flush_rejects_invalid_deadlines(timeout: float) -> None:
    recorder = Metrics.in_memory().recorder()
    try:
        with pytest.raises(AIError) as raised:
            await recorder.flush(timeout_seconds=timeout)
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    finally:
        await recorder.close()


@pytest.mark.asyncio
async def test_explicit_record_and_query_remain_authoritative() -> None:
    from datetime import timedelta

    store = _ControlledStore()
    metrics = Metrics.from_store(store)
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


def test_standalone_metrics_does_not_import_runtime() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from linktools.ai.observe import Metrics; import sys; "
         "assert 'linktools.ai.runtime' not in sys.modules; Metrics.in_memory()"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_owned", [False, True])
async def test_metric_status_reads_preserve_failure_state_across_flush(runtime_owned: bool) -> None:
    from linktools.ai.observe.service import MetricsService

    metrics = Metrics.in_memory()
    owner = MetricsService(metrics) if runtime_owned else metrics.recorder()
    try:
        assert not await owner.try_record(None)
        initial = owner.status()
        assert initial.last_failure_code == ("ENRICHMENT_FAILED" if runtime_owned else "INVALID_OBSERVATION")
        fields_before_read = vars(owner).copy()
        assert owner.status() == owner.status() == initial
        assert vars(owner) == fields_before_read

        assert await owner.try_record(_observation("accepted"))
        assert not await owner.try_record(None)
        fields_before_read = vars(owner).copy()
        rejected = owner.status()
        assert rejected.last_failure_code == initial.last_failure_code
        assert rejected.rejected > initial.rejected
        assert owner.status() == rejected
        assert vars(owner) == fields_before_read

        flushed = await owner.flush()
        assert flushed.completed
        assert flushed.status.persisted == 1
        assert flushed.status.pending == 0
        assert flushed.status.rejected == rejected.rejected
        assert await metrics.get_observation("accepted") is not None
        fields_before_read = vars(owner).copy()
        assert owner.status() == owner.status() == flushed.status
        assert vars(owner) == fields_before_read
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_explicit_observation_batch_limit_rejects_before_partial_write() -> None:
    metrics = Metrics.in_memory()
    observations = tuple(_observation(f"limit-{index}") for index in range(257))
    with pytest.raises(AIError, match="REQUEST_FIELD_INVALID"):
        await metrics.record_observations(observations)
    assert await metrics.get_observation("limit-0") is None
    await metrics.record_observations(observations[:256])
    assert await metrics.get_observation("limit-255") == observations[255]
