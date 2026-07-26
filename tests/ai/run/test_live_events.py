#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/run/test_live_events.py"""

import asyncio

import pytest

from linktools.ai.errors import RunLiveStreamAlreadyOpenError, RunLiveStreamClosedError
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


# --- P6 cancellation-safe close (no blocking sentinel) ----------------


def test_close_does_not_block_on_a_full_queue():
    """The old design pushed a sentinel into the queue on close, which blocked
    on a full buffer. The new design signals close via asyncio.Event and never
    touches the queue -- close returns even when the buffer is full."""
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-close-full", capacity=1)
        await handle.publish({"type": "text", "text": "first"})  # fills the queue
        # close must not block waiting to push a sentinel into the full queue.
        await asyncio.wait_for(handle.close(), timeout=1.0)
        assert handle.is_closed

    asyncio.run(_run())


def test_publish_racing_close_raises_closed():
    """A publish that loses the race against close raises
    RunLiveStreamClosedError rather than silently dropping or deadlocking."""
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-race", capacity=1)
        await handle.publish({"type": "text", "text": "first"})  # fill the queue

        publish_started = asyncio.Event()
        publish_done = {}

        async def _blocked_publish():
            publish_started.set()
            try:
                await handle.publish({"type": "text", "text": "second"})
                publish_done["outcome"] = "delivered"
            except RunLiveStreamClosedError:
                publish_done["outcome"] = "closed"

        task = asyncio.create_task(_blocked_publish())
        await publish_started.wait()
        # The publish is blocked on the full queue. Close mid-publish.
        await handle.close()
        await task
        return publish_done.get("outcome")

    assert asyncio.run(_run()) == "closed"


def test_publish_on_already_closed_raises():
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-closed")
        await handle.close()
        with pytest.raises(RunLiveStreamClosedError):
            await handle.publish({"type": "text", "text": "late"})

    asyncio.run(_run())


def test_events_drains_pre_close_enqueued_events_then_returns():
    """Events enqueued BEFORE close are still delivered; the iterator returns
    once the queue drains + close is signaled."""
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-drain")
        await handle.publish({"type": "text", "text": "a"})
        await handle.publish({"type": "text", "text": "b"})
        await handle.publish({"type": "text", "text": "c"})
        await handle.close()
        return [event async for event in handle.events()]

    assert asyncio.run(_run()) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c"},
    ]


def test_close_is_idempotent():
    """Closing an already-closed handle is a no-op (never raises, never
    double-signals)."""
    async def _run():
        hub = RunLiveEventHub()
        handle = await hub.open("run-idempotent")
        await handle.close()
        await handle.close()  # no exception
        await handle.close()  # still no exception
        assert handle.is_closed
        assert hub.active_stream_count == 0

    asyncio.run(_run())


def test_concurrent_open_close_are_serialized():
    """Hub lock: a concurrent open + close pair cannot race a stale handle's
    identity check against a fresh open's slot claim."""
    async def _run():
        hub = RunLiveEventHub()
        # Open + immediately close in concurrent tasks; the lock guarantees
        # at most one handle per run_id at a time, so the second open either
        # sees the slot empty (and succeeds) or occupied (and raises), never
        # a partial-state write.
        async def _open_close(run_id):
            try:
                h = await hub.open(run_id)
                await h.close()
            except RunLiveStreamAlreadyOpenError:
                pass

        await asyncio.gather(*[_open_close("run-A") for _ in range(8)])
        # Every opener either closed cleanly or bailed on AlreadyOpen; the
        # active map ends up empty.
        assert hub.active_stream_count == 0

    asyncio.run(_run())
