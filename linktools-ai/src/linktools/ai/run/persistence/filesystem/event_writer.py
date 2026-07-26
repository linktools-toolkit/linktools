#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemFencedRunEventWriter: Filesystem implementation of
FencedRunEventWriter.

Acquire a per-run in-process lock -> read the current execution_token from the
RunRecord -> verify the presented fence matches -> append the security event
-> fsync the event file -> release. A stale fence (or a run with no token, or
a missing run) raises RunFenceLostError BEFORE the event lands, so the
security-sensitive action that triggered the append does NOT proceed and the
Coordinator routes the run into fail/fencing-loss convergence.

Filesystem storage is single-process (one writer per run), so an in-process
``asyncio.Lock`` per run_id is the correct mutual-exclusion primitive -- the
on-disk RunRecord + event files are this process's exclusive state, and a
cross-process file lock would only guard against a second writer that cannot
exist. The lock serializes the read-check-append cycle so a concurrent
transition (e.g. the RunController marking the run FAILED after losing the
lease) cannot interleave between the fence check and the append."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..event_writer import RunFenceLostError

if TYPE_CHECKING:
    from ....events.context import EventStreamContext
    from ....events.store import EventStore
    from ....run.store import RunStore
    from ...commit import ExecutionFence


class FilesystemFencedRunEventWriter:
    """Filesystem implementation of FencedRunEventWriter."""

    def __init__(self, *, run_store: "RunStore", event_store: "EventStore") -> None:
        self._runs = run_store
        self._events = event_store
        self._locks: "dict[str, asyncio.Lock]" = {}
        self._locks_guard = asyncio.Lock()

    async def _run_lock(self, run_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(run_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[run_id] = lock
            return lock

    async def append_security(
        self,
        *,
        context: "EventStreamContext",
        fence: "ExecutionFence",
        event: Any,
    ) -> None:
        # Per-run lock: serialize the read-check-append cycle against a
        # concurrent transition on the same run.
        lock = await self._run_lock(context.run_id)
        async with lock:
            record = await self._runs.get(context.run_id)
            if record is None:
                raise RunFenceLostError(
                    f"fenced security event for run {context.run_id!r} cannot "
                    f"be appended: the run does not exist"
                )
            stored_token = record.execution_token or ""
            if not stored_token:
                raise RunFenceLostError(
                    f"fenced security event for run {context.run_id!r} cannot "
                    f"be appended: the run has no execution token to fence "
                    f"against"
                )
            if stored_token != fence.token:
                raise RunFenceLostError(
                    f"fence lost for run {context.run_id!r}: stored execution "
                    f"token differs from the presented fence"
                )
            # Fence verified; append the security event. The FilesystemEventStore
            # fsyncs each event file as part of its append, so the event is
            # durable when this call returns.
            from ....events.context import append_event

            await append_event(self._events, context, event)


__all__: "list[str]" = ["FilesystemFencedRunEventWriter"]
