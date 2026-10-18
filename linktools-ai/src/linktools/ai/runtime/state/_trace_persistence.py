#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace-owned observation buffering, batch pressure and timer lifetime."""

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Protocol

from linktools.core import environ

from ...core import (
    DEFAULT_TRACE_BATCH_SIZE as _OBSERVATION_FLUSH_BOUNDARIES,
    DEFAULT_TRACE_FLUSH_INTERVAL as _OBSERVATION_FLUSH_SECONDS,
)
from ...errors import AIError, ErrorCode
from ._step_archive import _AgentRunHistoryLock, _ProjectionOffset
from ._step_contracts import AgentRunStore

_logger = environ.get_logger("ai.runtime.state.trace")


class _FlushProjection(Protocol):
    async def __call__(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None = None, deferred: bool = False,
    ) -> None: ...


class TracePersistence:
    """Own trace buffering; delegate atomic publication to Runtime's narrow port."""

    def __init__(
        self, archive: AgentRunStore | None, *, history_lock: _AgentRunHistoryLock,
        ensure_ready: Callable[[], Awaitable[None]],
        require_mutable: Callable[[str], None],
        has_flight: Callable[[str], bool],
        flush: _FlushProjection,
        flush_boundary: _FlushProjection,
        background_tasks: set[asyncio.Task[object]],
    ) -> None:
        self.archive = archive
        self.offsets: dict[str, _ProjectionOffset] = {}
        self.dirty: set[str] = set()
        self._history_lock = history_lock
        self._ensure_ready = ensure_ready
        self._require_mutable = require_mutable
        self._has_flight = has_flight
        self._flush = flush
        self._flush_boundary = flush_boundary
        self._background_tasks = background_tasks
        self.scheduler_task: asyncio.Task[None] | None = None
        self.wake = asyncio.Event()

    def configure(self, agent_run_id: str, *, batch_size: int, flush_interval: float) -> None:
        """Bind this run's observation policy before its first boundary."""
        offset = self.offsets.setdefault(agent_run_id, _ProjectionOffset())
        if offset.observation_boundaries or offset.observation_task is not None:
            raise RuntimeError("observation batching cannot change during an active run")
        offset.observation_batch_size = batch_size
        offset.observation_flush_interval = flush_interval

    def reset(self) -> None:
        self.offsets.clear()
        self.dirty.clear()
        self.scheduler_task = None
        self.wake.clear()

    async def settle_scheduler(self) -> None:
        self.wake.set()
        if self.scheduler_task is not None:
            await asyncio.shield(self.scheduler_task)

    async def defer(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None,
    ) -> None:
        await self._ensure_ready()
        async with self._history_lock.hold(agent_run_id):
            self._require_mutable(agent_run_id)
            offset = self.offsets.setdefault(agent_run_id, _ProjectionOffset())
            task = offset.observation_task
            if task is not None and task.done():
                if not self._has_flight(agent_run_id):
                    offset.observation_task = None
                    offset.observation_boundaries = 0
                task.result()
            if agent_run_id not in self.dirty:
                return
            schedule = offset.observation_boundaries == 0
            offset.observation_boundaries += 1
            offset.observation_execution_id = execution_id
            offset.observation_producer_generation = producer_generation
            if schedule:
                interval = offset.observation_flush_interval or _OBSERVATION_FLUSH_SECONDS
                offset.observation_deadline = monotonic() + interval
            pressure = offset.observation_boundaries >= (
                offset.observation_batch_size or _OBSERVATION_FLUSH_BOUNDARIES
            )
        if pressure:
            await self._flush_boundary(
                agent_run_id, execution_id=execution_id,
                producer_generation=producer_generation,
            )
        elif schedule:
            self.schedule()


    def schedule(self) -> None:
        # Tasks inherit ContextVars; callers leave their run lock before this.
        self.wake.set()
        task = self.scheduler_task
        if task is not None:
            if not task.done():
                return
            task.result()
        self.scheduler_task = asyncio.create_task(
            self._run_observation_scheduler(), name="history-observation-scheduler",
        )
        self._track_observation_task(self.scheduler_task)


    def _track_observation_task(self, task: asyncio.Task[None]) -> None:
        self._background_tasks.add(task)

        def completed(value: asyncio.Task[None]) -> None:
            self._background_tasks.discard(value)
            self.wake.set()
            if not value.cancelled():
                value.exception()

        task.add_done_callback(completed)


    async def _run_observation_scheduler(self) -> None:
        while True:
            self.wake.clear()
            queued = tuple(
                agent_run_id for agent_run_id in self.dirty
                if (offset := self.offsets.get(agent_run_id)) is not None
                and offset.observation_boundaries
                and (offset.observation_task is None or not offset.observation_task.done())
            )
            if not queued:
                return
            waiting = tuple(
                run_id for run_id in queued if self.offsets[run_id].observation_task is None
            )
            if not waiting:
                await self.wake.wait()
                continue
            deadline = min(
                self.offsets[run_id].observation_deadline or monotonic() for run_id in waiting
            )
            try:
                await asyncio.wait_for(self.wake.wait(), max(0.0, deadline - monotonic()))
            except asyncio.TimeoutError:
                pass
            if self.wake.is_set():
                continue
            for agent_run_id in waiting:
                async with self._history_lock.hold(agent_run_id):
                    offset = self.offsets.get(agent_run_id)
                    if (offset is None or not offset.observation_boundaries
                            or offset.observation_task is not None
                            or agent_run_id not in self.dirty
                            or offset.observation_deadline is not None
                            and offset.observation_deadline > monotonic()):
                        continue
                    execution_id = offset.observation_execution_id
                    if execution_id is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    offset.observation_boundaries = 0
                    offset.observation_deadline = None
                    generation = offset.observation_producer_generation
                task = asyncio.create_task(
                    self._run_scheduled_projection(
                        agent_run_id, offset, execution_id=execution_id,
                        producer_generation=generation,
                    ),
                    name=f"history-observation-commit-{agent_run_id}",
                )
                offset.observation_task = task
                self._track_observation_task(task)


    async def _run_scheduled_projection(
        self, agent_run_id: str, offset: _ProjectionOffset, *,
        execution_id: str, producer_generation: int | None,
    ) -> None:
        await self._flush(
            agent_run_id, execution_id=execution_id,
            producer_generation=producer_generation,
        )
        async with self._history_lock.hold(agent_run_id):
            if offset.observation_task is not asyncio.current_task():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            offset.observation_task = None
            dirty = agent_run_id in self.dirty
            if dirty:
                offset.observation_boundaries = max(1, offset.observation_boundaries)
                if offset.observation_deadline is None:
                    interval = offset.observation_flush_interval or _OBSERVATION_FLUSH_SECONDS
                    offset.observation_deadline = monotonic() + interval
            else:
                offset.observation_boundaries = 0
                offset.observation_execution_id = None
                offset.observation_producer_generation = None
        if dirty:
            self.schedule()


    async def stop(self, agent_run_id: str) -> None:
        """Remove this run from the timer and settle only its captured commit."""
        async with self._history_lock.hold(agent_run_id):
            offset = self.offsets.get(agent_run_id)
            task = None if offset is None else offset.observation_task
            if offset is None or task is asyncio.current_task():
                return
            offset.observation_boundaries = 0
            offset.observation_deadline = None
            self.wake.set()
            if task is None:
                return
        try:
            await asyncio.shield(task)
        except Exception:
            async with self._history_lock.hold(agent_run_id):
                if self._has_flight(agent_run_id):
                    raise
                offset.observation_task = None
                offset.observation_boundaries = 0
            _logger.warning(
                "observation batch failed before commit; required barrier will retry: agent_run=%s",
                agent_run_id,
            )
