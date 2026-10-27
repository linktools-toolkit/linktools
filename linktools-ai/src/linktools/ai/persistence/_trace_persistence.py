#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trace projection scheduling over an explicitly borrowed write buffer."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from linktools.core import environ

from ..core import DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from ..storage import BatchWriteScope, BatchWriteScopeFactory, WriteBufferBackend, WriteBufferEntry
from ..errors import AIError, ErrorCode
from .durability import await_owned_task
from ._step_archive import _AgentRunHistoryLock, _ProjectionOffset
from ._step_contracts import AgentRunStore


_logger = environ.get_logger("ai.runtime.state.trace")


class _FlushProjection(Protocol):
    async def __call__(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None = None,
    ) -> None: ...


@dataclass
class _ObservationBuffer:
    scope: BatchWriteScope | None = None
    paused: bool = False
    error: BaseException | None = None


class TracePersistence:
    """Keep semantic facts in staging and optional flush requests in the buffer.

    Queue acknowledgements settle scheduling requests, never durable facts.
    Only Runtime's guarded projection commit may advance offsets or clear dirty
    staging. Each run owns its consumer; the buffer itself is borrowed.
    """

    def __init__(
        self, archive: AgentRunStore | None, *, history_lock: _AgentRunHistoryLock,
        ensure_ready: Callable[[], Awaitable[None]],
        require_mutable: Callable[[str], None],
        has_flight: Callable[[str], bool],
        flush: _FlushProjection,
        flush_boundary: _FlushProjection,
        background_tasks: set[asyncio.Task[object]],
    ) -> None:
        self._batch_scopes: BatchWriteScopeFactory = BatchWriteScope
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
        self._caches: dict[str, WriteBufferBackend | None] = {}
        self._buffers: dict[str, _ObservationBuffer] = {}
        self._terminal_runs: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    def bind_batch_scopes(self, factory: BatchWriteScopeFactory) -> None:
        """Borrow this Runtime's scheduling factory before any run is configured."""
        if self._caches or self._buffers:
            raise RuntimeError("batch scopes must be bound before run configuration")
        self._batch_scopes = factory

    def configure(
        self, agent_run_id: str, *, buffer: WriteBufferBackend | None = None,
        batch_size: int = DEFAULT_TRACE_BATCH_SIZE,
        flush_interval: float = DEFAULT_TRACE_FLUSH_INTERVAL,
        terminal: bool = False,
    ) -> None:
        """Bind this run's observation policy before admission."""
        if agent_run_id in self._buffers:
            raise RuntimeError("observation batching cannot change during an active run")
        if terminal and buffer is not None:
            raise ValueError("terminal observations do not use a write buffer")
        if buffer is not None and not isinstance(buffer, WriteBufferBackend):
            raise TypeError("buffer must implement append, read_batch and ack")
        self._caches[agent_run_id] = buffer
        if terminal:
            self._terminal_runs.add(agent_run_id)
        else:
            self._terminal_runs.discard(agent_run_id)
        offset = self.offsets.setdefault(agent_run_id, _ProjectionOffset())
        offset.observation_batch_size = batch_size
        offset.observation_flush_interval = flush_interval

    def reset(self) -> None:
        self.offsets.clear()
        self.dirty.clear()
        self._caches.clear()
        self._buffers.clear()
        self._terminal_runs.clear()
        self._locks.clear()

    async def settle_buffers(self) -> None:
        for agent_run_id in tuple(self._buffers):
            await self.stop(agent_run_id)

    def release(self, agent_run_id: str) -> None:
        """Forget a quiescent run after its semantic owner released staging."""
        if agent_run_id in self._buffers:
            raise RuntimeError("observation buffer is not quiescent")
        self._caches.pop(agent_run_id, None)
        self._terminal_runs.discard(agent_run_id)
        self._locks.pop(agent_run_id, None)

    async def defer(
        self, agent_run_id: str, *, execution_id: str,
        producer_generation: int | None,
    ) -> None:
        await self._ensure_ready()
        async with self._history_lock.hold(agent_run_id):
            self._require_mutable(agent_run_id)
            if agent_run_id not in self.dirty:
                return
            offset = self.offsets.setdefault(agent_run_id, _ProjectionOffset())
            offset.observation_execution_id = execution_id
            offset.observation_producer_generation = producer_generation
            if agent_run_id in self._terminal_runs:
                # Facts stay in staging; guarded safety and terminal boundaries
                # own capture, source commit and finalization. No queue receipt
                # may stand in for a durable projection.
                return
        accepted = False
        async with self._locks.setdefault(agent_run_id, asyncio.Lock()):
            cache = self._caches.get(agent_run_id)
            if cache is not None:
                buffer = self._buffers.get(agent_run_id)
                if buffer is None:
                    buffer = _ObservationBuffer()
                    async def write_batch(batch: tuple[WriteBufferEntry, ...]) -> None:
                        if buffer.paused:
                            return
                        offset.observation_task = asyncio.current_task()
                        try:
                            if buffer.error is not None:
                                raise buffer.error
                            if any(key != (agent_run_id, execution_id, producer_generation) or content is not None
                                   for _, key, content in batch):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            # One run/producer owns this scope. A marker requests
                            # capture of its latest staged facts, not a replay of
                            # a previously committed payload.
                            # A buffer callback timeout must not cancel durable
                            # publication. Wait for its actual outcome; caller
                            # cancellation remains owned by the outer boundary.
                            _, error, _ = await await_owned_task(
                                asyncio.create_task(self._flush(
                                    agent_run_id, execution_id=execution_id,
                                    producer_generation=producer_generation,
                                )), self._background_tasks,
                            )
                            if error is not None:
                                raise error
                        except BaseException as error:
                            buffer.error = error
                            raise
                        finally:
                            offset.observation_task = None
                    buffer.scope = self._batch_scopes(
                        cache, write_batch,
                        batch_size=offset.observation_batch_size or DEFAULT_TRACE_BATCH_SIZE,
                        flush_interval=offset.observation_flush_interval or DEFAULT_TRACE_FLUSH_INTERVAL,
                    )
                    self._buffers[agent_run_id] = buffer
                if buffer.error is None:
                    assert buffer.scope is not None
                    try:
                        accepted = await buffer.scope.try_submit(
                            (agent_run_id, execution_id, producer_generation), None,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # An uncertain append may already own a callback. The
                        # required path quiesces it before attempting source I/O.
                        self._caches[agent_run_id] = None
        if not accepted:
            await self._flush_boundary(
                agent_run_id, execution_id=execution_id,
                producer_generation=producer_generation,
            )

    async def stop(self, agent_run_id: str) -> None:
        """Quiesce scheduling without publishing the caller's dirty tail.

        Terminal seal and revoked-producer discard choose their own atomic
        action. Undispatched markers can be acknowledged once paused because
        their facts remain in staging; an entered callback must settle first.
        """
        _, error, cancelled = await await_owned_task(
            asyncio.create_task(self._stop(agent_run_id)), self._background_tasks,
        )
        if error is not None:
            raise error
        if cancelled:
            raise asyncio.CancelledError

    async def _stop(self, agent_run_id: str) -> None:
        async with self._locks.setdefault(agent_run_id, asyncio.Lock()):
            buffer = self._buffers.get(agent_run_id)
            if buffer is None:
                return
            buffer.paused = True
            assert buffer.scope is not None
            try:
                await buffer.scope.close()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Buffer transport failure cannot erase a source failure or make
                # a durable receipt. Disable optional buffering; caller's direct
                # guarded publication still decides whether staging is durable.
                self._caches[agent_run_id] = None
            self._buffers.pop(agent_run_id, None)
            if buffer.error is not None:
                if (not isinstance(buffer.error, Exception) or isinstance(buffer.error, AIError)
                        or self._has_flight(agent_run_id)):
                    raise buffer.error
                _logger.warning(
                    "observation publication failed before commit; required boundary will retry: agent_run=%s",
                    agent_run_id,
                )
