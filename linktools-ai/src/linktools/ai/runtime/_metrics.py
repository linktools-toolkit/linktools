#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned best-effort Metrics buffering and observation helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from time import monotonic
from typing import TypeVar

from linktools.core import environ

from ..core import canonical_sha256
from ..errors import AIError
from ..observe import MetricMeasurement, MetricRecorder, Metrics, Observation

_logger = environ.get_logger("ai.runtime.metrics")
_QUEUE_CAPACITY = 1024
_BATCH_SIZE = 64
_WRITE_TIMEOUT_SECONDS = 2.0
_CLOSE_DEADLINE_SECONDS = 5.0
_WARNING_INTERVAL_SECONDS = 30.0
T = TypeVar("T")


class _RuntimeMetricBuffer(MetricRecorder):
    """Bound one Runtime's automatic observations without owning the Metrics store."""

    def __init__(self, metrics: Metrics) -> None:
        if not isinstance(metrics, Metrics):
            raise TypeError("metrics must be Metrics")
        self._metrics = metrics
        self._queue: asyncio.Queue[Observation | None] = asyncio.Queue(
            maxsize=_QUEUE_CAPACITY
        )
        self._accepting = True
        self._accepted = 0
        self._dropped = 0
        self._write_failures = 0
        self._last_warning_at = 0.0
        self._writer = asyncio.create_task(
            self._run(),
            name="linktools-runtime-metrics",
        )

    def try_record(self, observation: Observation) -> bool:
        if not self._accepting or not isinstance(observation, Observation):
            self._drop("invalid or closed metric observation")
            return False
        try:
            self._queue.put_nowait(observation)
        except asyncio.QueueFull:
            self._drop("runtime metric queue full")
            return False
        self._accepted += 1
        return True

    async def close(self) -> None:
        if not self._accepting and self._writer.done():
            self._consume_writer()
            return
        self._accepting = False
        started = monotonic()
        try:
            await asyncio.wait_for(
                self._queue.join(),
                timeout=_CLOSE_DEADLINE_SECONDS,
            )
        except (TimeoutError, asyncio.TimeoutError):
            self._drop_remaining("runtime metric close deadline exceeded")
        remaining = max(0.0, _CLOSE_DEADLINE_SECONDS - (monotonic() - started))
        if not self._writer.done():
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                self._drop_remaining("runtime metric close queue remained full")
                self._queue.put_nowait(None)
            try:
                await asyncio.wait_for(asyncio.shield(self._writer), timeout=remaining)
            except (TimeoutError, asyncio.TimeoutError):
                self._writer.cancel()
                await self._consume_cancelled_writer()
        self._consume_writer()
        if self._dropped or self._write_failures:
            _logger.warning(
                "runtime metrics closed with loss: accepted=%s dropped=%s write_failures=%s",
                self._accepted,
                self._dropped,
                self._write_failures,
            )
        else:
            _logger.debug(
                "runtime metrics closed: accepted=%s",
                self._accepted,
            )

    async def _run(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                return
            batch = [first]
            while len(batch) < _BATCH_SIZE:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is None:
                    self._queue.task_done()
                    await self._write(tuple(batch))
                    for _ in batch:
                        self._queue.task_done()
                    return
                batch.append(item)
            await self._write(tuple(batch))
            for _ in batch:
                self._queue.task_done()

    async def _write(self, batch: tuple[Observation, ...]) -> None:
        try:
            await asyncio.wait_for(
                self._metrics.record_observations(batch),
                timeout=_WRITE_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._write_failures += 1
            self._dropped += len(batch)
            self._warn("runtime metric batch write failed")

    def _drop(self, message: str) -> None:
        self._dropped += 1
        self._warn(message)

    def _drop_remaining(self, message: str) -> None:
        dropped = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if item is not None:
                dropped += 1
        if dropped:
            self._dropped += dropped
            self._warn(message)

    def _warn(self, message: str) -> None:
        now = monotonic()
        if now - self._last_warning_at < _WARNING_INTERVAL_SECONDS:
            return
        self._last_warning_at = now
        _logger.warning(
            "%s: accepted=%s dropped=%s write_failures=%s",
            message,
            self._accepted,
            self._dropped,
            self._write_failures,
        )

    async def _consume_cancelled_writer(self) -> None:
        try:
            await self._writer
        except asyncio.CancelledError:
            pass
        except BaseException:
            self._write_failures += 1
            self._warn("runtime metric writer failed during cancellation")

    def _consume_writer(self) -> None:
        if not self._writer.done():
            return
        try:
            self._writer.result()
        except asyncio.CancelledError:
            pass
        except BaseException:
            self._write_failures += 1
            self._warn("runtime metric writer failed")


def _stable_observation_id(*parts: str) -> str:
    return canonical_sha256({"parts": list(parts)})


def _measurement(name: str, value: int | float) -> MetricMeasurement:
    return MetricMeasurement(name, 1, value)


def _observation(
    *,
    observation_id: str,
    kind: str,
    source_namespace: str,
    tenant_id: str,
    status: str | None,
    error_code: str | None,
    correlation: Mapping[str, str | int] | None = None,
    dimensions: Mapping[str, str] | None = None,
    measurements: tuple[MetricMeasurement, ...] = (),
    occurred_at: datetime | None = None,
) -> Observation:
    return Observation(
        version=1,
        observation_id=observation_id,
        kind=kind,
        occurred_at=occurred_at or datetime.now(timezone.utc),
        source_namespace=source_namespace,
        tenant_id=tenant_id,
        status=status,
        error_code=error_code,
        correlation=dict(correlation or {}),
        dimensions=dict(dimensions or {}),
        measurements=measurements,
    )


def _try_record(
    recorder: MetricRecorder | None,
    factory: Callable[[], Observation],
) -> bool:
    if recorder is None:
        return False
    try:
        return recorder.try_record(factory())
    except (AIError, TypeError, ValueError):
        _logger.warning("runtime metric observation rejected", exc_info=False)
        return False


__all__: list[str] = []
