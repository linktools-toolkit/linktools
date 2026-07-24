#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/run/test_events_bus.py"""

import asyncio

from linktools.ai.run.events_bus import RunEventBus


def test_publish_then_subscribe_drains_in_order():
    async def _run():
        bus = RunEventBus()
        bus.open("run-1")
        await bus.publish("run-1", {"type": "text", "text": "a"})
        await bus.publish("run-1", {"type": "text", "text": "b"})
        bus.close("run-1")

        received = [event async for event in bus.subscribe("run-1")]
        return received

    assert asyncio.run(_run()) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_concurrent_publish_and_subscribe():
    async def _run():
        bus = RunEventBus()
        bus.open("run-2")
        received = []

        async def _consume():
            async for event in bus.subscribe("run-2"):
                received.append(event)

        async def _produce():
            for i in range(3):
                await bus.publish("run-2", {"type": "tool", "phase": str(i)})
            bus.close("run-2")

        await asyncio.gather(_consume(), _produce())
        return received

    result = asyncio.run(_run())
    assert [e["phase"] for e in result] == ["0", "1", "2"]


def test_publish_without_open_is_a_silent_noop():
    async def _run():
        bus = RunEventBus()
        await bus.publish("never-opened", {"type": "text", "text": "x"})

    asyncio.run(_run())


def test_close_without_open_is_a_silent_noop():
    bus = RunEventBus()
    bus.close("never-opened")


def test_subscribe_pops_queue_after_close():
    async def _run():
        bus = RunEventBus()
        bus.open("run-3")
        bus.close("run-3")
        async for _ in bus.subscribe("run-3"):
            pass
        return bus._queues

    assert asyncio.run(_run()) == {}
