#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/run/test_live_events.py"""

import asyncio

import pytest

from linktools.ai.errors import RunLiveStreamAlreadyOpenError
from linktools.ai.run.live_events import (
    NullRunLiveEventSink,
    NullSecurityEventSink,
    RunLiveEventHub,
)


def test_publish_then_drain_in_order():
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-1")
        await handle.publish({"type": "text", "text": "a"})
        await handle.publish({"type": "text", "text": "b"})
        await handle.close()
        return [event async for event in handle.events()]

    assert asyncio.run(_run()) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_concurrent_publish_and_consume():
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-2")
        received = []

        async def _consume():
            async for event in handle.events():
                received.append(event)

        async def _produce():
            for i in range(3):
                await handle.publish({"type": "tool", "phase": str(i)})
            await handle.close()

        await asyncio.gather(_consume(), _produce())
        return received

    result = asyncio.run(_run())
    assert [e["phase"] for e in result] == ["0", "1", "2"]


def test_duplicate_open_for_same_run_id_raises():
    async def _run():
        hub = RunLiveEventHub()
        await hub.open("run-3")
        with pytest.raises(RunLiveStreamAlreadyOpenError):
            await hub.open("run-3")

    asyncio.run(_run())


def test_reopen_after_close_succeeds_with_new_stream_id():
    async def _run():
        hub = RunLiveEventHub()
        first = await hub.open("run-4")
        await first.close()
        second = await hub.open("run-4")
        return first.stream_id, second.stream_id

    first_id, second_id = asyncio.run(_run())
    assert first_id != second_id


def test_stale_handle_close_does_not_evict_newer_handle():
    async def _run():
        hub = RunLiveEventHub()
        stale = await hub.open("run-5")
        await stale.close()
        fresh = await hub.open("run-5")
        # A redundant close on the already-closed stale handle must not
        # disturb the hub's registration of the newer handle.
        await stale.close()
        return hub.active_stream_count, hub._active["run-5"] is fresh

    count, is_fresh = asyncio.run(_run())
    assert count == 1
    assert is_fresh


def test_active_stream_count_tracks_open_and_close():
    async def _run():
        hub = RunLiveEventHub()
        assert hub.active_stream_count == 0
        handle = await hub.open("run-6")
        assert hub.active_stream_count == 1
        await handle.close()
        assert hub.active_stream_count == 0

    asyncio.run(_run())


def test_publish_backpressures_on_a_full_queue_instead_of_dropping():
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-7", capacity=1)
        await handle.publish({"type": "text", "text": "first"})

        second_published = False

        async def _publish_second():
            nonlocal second_published
            await handle.publish({"type": "text", "text": "second"})
            second_published = True

        task = asyncio.create_task(_publish_second())
        await asyncio.sleep(0.05)
        # The queue (capacity=1) is already full with "first" -- the second
        # publish must still be blocked awaiting room, never silently dropped.
        assert not second_published

        drained = []
        async for event in handle.events():
            drained.append(event)
            if len(drained) == 2:
                break
        await task
        return second_published, drained

    published, drained = asyncio.run(_run())
    assert published
    assert drained == [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]


def test_events_stop_on_close_with_no_consumer_ever_attached():
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-8")
        await handle.close()
        return [event async for event in handle.events()]

    assert asyncio.run(_run()) == []


def test_null_sinks_are_no_ops():
    async def _run():
        await NullRunLiveEventSink().publish({"type": "text", "text": "x"})
        await NullSecurityEventSink().emit(object())

    asyncio.run(_run())
