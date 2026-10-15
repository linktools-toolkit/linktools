#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Best-effort automatic recording without changing explicit Metrics writes."""

from __future__ import annotations

import asyncio
import math
from time import monotonic
from typing import TYPE_CHECKING

from linktools.core import environ

from ..errors import AIError, ErrorCode
from ..storage import WriteCache
from ._model import MetricBufferStatus, MetricFlushResult, Observation

if TYPE_CHECKING:
    from ._metrics import Metrics

_logger = environ.get_logger("ai.observe.recorder")
_WARNING_INTERVAL_SECONDS = 30.0


class _MetricRecorder:
    def __init__(self, metrics: Metrics, cache: WriteCache | None) -> None:
        self._metrics = metrics
        self._scope = None if cache is None else cache.open(metrics.record_observations)
        self._accepting = True
        self._accepted = 0
        self._persisted = 0
        self._lost = 0
        self._rejected = 0
        self._write_failures = 0
        self._last_failure_code: str | None = None
        self._failure_baseline: tuple[int, int, int, str | None] | None = None
        self._last_warning_at = 0.0
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def try_record(self, observation: Observation) -> bool:
        if not isinstance(observation, Observation):
            self._rejected += 1
            self._fail("INVALID_OBSERVATION")
            return False
        if not self._accepting:
            self._rejected += 1
            self._fail("RECORDER_CLOSED")
            return False
        if self._scope is not None:
            return await self._scope.try_submit(observation)

        self._active += 1
        self._idle.clear()
        try:
            # The requesting task owns this I/O: no queue, task or timeout bridge.
            await self._metrics.record_observations((observation,))
        except asyncio.CancelledError:
            self._lost += 1
            self._last_failure_code = "WRITE_CANCELLED"
            raise
        except Exception:
            self._lost += 1
            self._write_failures += 1
            self._fail("WRITE_FAILED")
            return False
        else:
            self._persisted += 1
            return True
        finally:
            # Direct calls complete admission only after their source call settles.
            self._accepted += 1
            self._active -= 1
            if self._active == 0:
                self._idle.set()

    def status(self) -> MetricBufferStatus:
        if self._scope is not None:
            status = self._scope.status()
            if self._failure_baseline != (
                status.rejected, status.unconfirmed, status.write_failures, status.last_failure_code,
            ):
                self._last_failure_code = None
            return MetricBufferStatus(
                True, self._accepting and status.accepting,
                status.accepted, status.confirmed, status.pending,
                status.rejected + self._rejected, status.unconfirmed,
                status.write_failures, status.queue_size, status.queue_capacity,
                status.high_watermark,
                self._last_failure_code or status.last_failure_code,
            )
        return MetricBufferStatus(
            True, self._accepting, self._accepted, self._persisted, 0,
            self._rejected, self._lost, self._write_failures,
            last_failure_code=self._last_failure_code,
        )

    async def flush(self, *, timeout_seconds: float = 5.0) -> MetricFlushResult:
        if self._scope is not None:
            result = await self._scope.flush(timeout_seconds=timeout_seconds)
            return MetricFlushResult(result.completed, self.status())
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        # An unfinished direct call has not completed admission to this recorder.
        return MetricFlushResult(True, self.status())

    async def close(self) -> None:
        self._accepting = False
        if self._scope is not None:
            await self._scope.close()
            return
        # This wait never owns or cancels I/O running in other caller tasks.
        await self._idle.wait()

    def _fail(self, code: str) -> None:
        self._last_failure_code = code
        if self._scope is not None:
            status = self._scope.status()
            self._failure_baseline = (
                status.rejected, status.unconfirmed, status.write_failures, status.last_failure_code,
            )
        now = monotonic()
        if now - self._last_warning_at >= _WARNING_INTERVAL_SECONDS:
            self._last_warning_at = now
            _logger.warning("automatic metric delivery degraded: code=%s", code)
