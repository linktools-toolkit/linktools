#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunEventBus: in-process fan-out of a Run's live streaming events (text
deltas, tool phases, pause signals) to a concurrent subscriber. This is the
seam that lets ``run_stream()`` be reimplemented as a pure consumer of
published events instead of iterating ``AgentEngine.execute()``'s own
yields -- the change that lets ``execute()`` collapse to a single
Outcome-returning awaitable (spec section 12.4) without losing the CLI's
live incremental output."""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class RunEventBus:
    """One queue per ``run_id``, explicitly opened before a run starts and
    closed once it ends. Publishing to a run that was never opened (or has
    already closed) is a silent no-op -- there is nothing to buffer for, and
    a caller that never subscribes should not leak an unbounded queue."""

    def __init__(self) -> None:
        self._queues: "dict[str, asyncio.Queue]" = {}

    def open(self, run_id: str) -> None:
        self._queues[run_id] = asyncio.Queue()

    async def publish(self, run_id: str, event: dict) -> None:
        queue = self._queues.get(run_id)
        if queue is not None:
            await queue.put(event)

    async def subscribe(self, run_id: str) -> "AsyncIterator[dict]":
        """Yield published events for ``run_id`` until :meth:`close` is
        called. Requires :meth:`open` to have been called first."""
        queue = self._queues[run_id]
        try:
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            self._queues.pop(run_id, None)

    def close(self, run_id: str) -> None:
        """Signal the end of this run's event stream to its subscriber.
        A no-op if the run was never opened or already closed."""
        queue = self._queues.get(run_id)
        if queue is not None:
            queue.put_nowait(None)


__all__: "list[str]" = ["RunEventBus"]
