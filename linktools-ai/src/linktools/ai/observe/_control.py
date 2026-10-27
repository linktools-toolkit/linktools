#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close-free producer controls and aggregate recording results."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from time import monotonic

from ..errors import AIError, ErrorCode
from ._model import MetricFlushResult, MetricRecorderStatus


@dataclass(frozen=True, slots=True)
class RuntimeMetricsStatus:
    """Independent recorder snapshots, without a fictional global ordering."""

    recorders: tuple[MetricRecorderStatus, ...]

    @property
    def enabled(self) -> bool:
        return any(status.enabled for status in self.recorders)

    @property
    def accepting(self) -> bool:
        enabled = tuple(status for status in self.recorders if status.enabled)
        return bool(enabled) and all(status.accepting for status in enabled)

    @property
    def accepted(self) -> int:
        return sum(status.accepted for status in self.recorders)

    @property
    def persisted(self) -> int:
        return sum(status.persisted for status in self.recorders)

    @property
    def pending(self) -> int:
        return sum(status.pending for status in self.recorders)

    @property
    def rejected(self) -> int:
        return sum(status.rejected for status in self.recorders)

    @property
    def unconfirmed(self) -> int:
        return sum(status.unconfirmed for status in self.recorders)

    @property
    def write_failures(self) -> int:
        return sum(status.write_failures for status in self.recorders)

    @property
    def queue_size(self) -> int:
        return sum(status.queue_size for status in self.recorders)

    @property
    def queue_capacity(self) -> int | None:
        capacities = tuple(status.queue_capacity for status in self.recorders if status.enabled)
        if any(capacity is None for capacity in capacities):
            return None
        return sum(capacity for capacity in capacities if capacity is not None)

    @property
    def failure_codes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            status.last_failure_code for status in self.recorders
            if status.last_failure_code is not None
        ))


@dataclass(frozen=True, slots=True)
class RuntimeMetricsFlushResult:
    """Settlement of each recorder's admission prefix, not persistence success."""

    completed: bool
    status: RuntimeMetricsStatus


class MetricsControl:
    """Inspect and flush a producer without acquiring its recording lifetime."""

    def __init__(
        self,
        status: Callable[[], MetricRecorderStatus],
        flush: Callable[..., Awaitable[MetricFlushResult]],
    ) -> None:
        self._status = status
        self._flush = flush

    def status(self) -> MetricRecorderStatus:
        return self._status()

    async def flush(self, *, timeout_seconds: float = 5.0) -> MetricFlushResult:
        _validate_timeout(timeout_seconds)
        return await self._flush(timeout_seconds=timeout_seconds)


class RuntimeMetricsControl:
    """Inspect the composition owner's producer list without owning a registry."""

    def __init__(self, controls: Callable[[], Sequence[MetricsControl]]) -> None:
        self._controls = controls

    def status(self) -> RuntimeMetricsStatus:
        return RuntimeMetricsStatus(tuple(control.status() for control in self._controls()))

    async def flush(self, *, timeout_seconds: float = 5.0) -> RuntimeMetricsFlushResult:
        _validate_timeout(timeout_seconds)
        controls = tuple(self._controls())
        deadline = monotonic() + timeout_seconds
        results = []
        for control in controls:
            results.append(await control.flush(timeout_seconds=max(0.0, deadline - monotonic())))
        return RuntimeMetricsFlushResult(
            all(result.completed for result in results),
            RuntimeMetricsStatus(tuple(result.status for result in results)),
        )


def _validate_timeout(timeout_seconds: float) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


__all__ = [
    "MetricsControl", "RuntimeMetricsControl", "RuntimeMetricsStatus", "RuntimeMetricsFlushResult",
]
