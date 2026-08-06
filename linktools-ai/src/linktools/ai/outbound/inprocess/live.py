#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Best-effort in-process Live Delta publisher."""

import asyncio

from ...domain.execution import LiveDeltaEnvelope


class InProcessLiveEventPublisher:
    def __init__(self) -> None:
        self._queues: "dict[str, set[asyncio.Queue[object]]]" = {}

    async def publish(self, event: LiveDeltaEnvelope) -> None:
        for queue in tuple(self._queues.get(event.execution_id, ())):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    async def subscribe(self, execution_id: str) -> "asyncio.Queue[object]":
        queue: "asyncio.Queue[object]" = asyncio.Queue(maxsize=256)
        self._queues.setdefault(execution_id, set()).add(queue)
        return queue


__all__ = ["InProcessLiveEventPublisher"]
