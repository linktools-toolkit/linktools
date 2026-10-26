#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local preview resume preserves coalesced ranges without duplicate text."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from linktools.ai.core import ExecutionDeltaType, ExecutionLineageKind, Principal
from linktools.ai.execution import Execution
from linktools.ai.execution.events import DefaultEventService, ExecutionDelta, LiveExecutionEventBroker
from linktools.ai.execution.stream import shared_stream_positions
from linktools.ai.observe.cursors import decode_execution_watch_cursor, get_preview_positions
from linktools.ai.execution import ExecutionTreeEvent


@asynccontextmanager
async def _watch(broker, cursor=None):
    principal = Principal("reader", "tenant")
    service = DefaultEventService(None, None, None, None, broker)
    positions = {} if cursor is None else decode_execution_watch_cursor(
        "preview-test", "tenant", "execution", cursor, include_content=True,
    )

    async def tree():
        stream = service._stream_with_live(
            "execution", principal=principal, after_event_seq=0, authorized=True,
        )
        try:
            async for event in stream:
                yield ExecutionTreeEvent(
                    "execution", "agent", ExecutionLineageKind.RUN,
                    None, "execution", None, 0, event,
                )
        finally:
            await stream.aclose()

    execution = Execution(SimpleNamespace(namespace="preview-test"), "execution", principal, None)
    stream = execution._watch_with_cursor(tree(), positions, True, cursor)
    with shared_stream_positions(get_preview_positions(positions)):
        try:
            yield stream
        finally:
            await stream.aclose()


@pytest.mark.asyncio
async def test_local_resume_slices_coalesced_text_at_received_position():
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "hello "))
    async with _watch(broker) as first:
        received = await asyncio.wait_for(anext(first), 1)
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "世界"))
    async with _watch(broker, received.cursor) as resumed:
        suffix = await asyncio.wait_for(anext(resumed), 1)
        assert suffix.event.event_type == "ASSISTANT_TEXT_DELTA"
        assert suffix.event.payload["text"] == "世界"
        assert suffix.cursor != received.cursor


@pytest.mark.asyncio
async def test_local_epoch_change_resets_before_rebuilding_preview():
    first = LiveExecutionEventBroker()
    first.register_local_producer("execution", 0)
    first.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "old"))
    async with _watch(first) as stream:
        received = await asyncio.wait_for(anext(stream), 1)
    replacement = LiveExecutionEventBroker()
    replacement.register_local_producer("execution", 0)
    replacement.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "new"))
    async with _watch(replacement, received.cursor) as resumed:
        assert (await asyncio.wait_for(anext(resumed), 1)).event.event_type == "STREAM_RESET"
        assert (await asyncio.wait_for(anext(resumed), 1)).event.payload["text"] == "new"


@pytest.mark.asyncio
async def test_local_resume_preserves_text_thinking_order_and_live_suffix():
    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "A"))
    async with _watch(broker) as stream:
        received = await asyncio.wait_for(anext(stream), 1)
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_THINKING_DELTA, "思考"))
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "B"))
    async with _watch(broker, received.cursor) as resumed:
        thinking = await asyncio.wait_for(anext(resumed), 1)
        text = await asyncio.wait_for(anext(resumed), 1)
        assert thinking.event.event_type == "ASSISTANT_THINKING_DELTA"
        assert thinking.event.payload["text"] == "思考"
        assert text.event.payload["text"] == "B"
    async with _watch(broker, text.cursor) as exact:
        pending = asyncio.create_task(anext(exact))
        await asyncio.sleep(0)
        assert not pending.done()
        broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "C"))
        suffix = await asyncio.wait_for(pending, 1)
        assert suffix.event.payload["text"] == "C"


@pytest.mark.asyncio
async def test_local_trimmed_preview_resets_and_rebuilds_retained_range():
    broker = LiveExecutionEventBroker(max_bytes=4)
    broker.register_local_producer("execution", 0)
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "a"))
    async with _watch(broker) as stream:
        received = await asyncio.wait_for(anext(stream), 1)
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_THINKING_DELTA, "1234"))
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "tail"))
    async with _watch(broker, received.cursor) as resumed:
        assert (await asyncio.wait_for(anext(resumed), 1)).event.event_type == "STREAM_RESET"
        retained = await asyncio.wait_for(anext(resumed), 1)
        assert retained.event.payload["text"] == "tail"
        assert retained.event.payload["stream_truncated"]


@pytest.mark.asyncio
async def test_local_resume_position_ahead_of_retained_head_resets():
    from linktools.ai.observe.cursors import encode_execution_watch_cursor

    broker = LiveExecutionEventBroker()
    broker.register_local_producer("execution", 0)
    broker.publish(ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "current"))
    cursor = encode_execution_watch_cursor(
        "preview-test", "tenant", "execution", include_content=True, event_seqs={},
        preview_positions={"execution": (broker.preview_epoch("execution"), 100)},
    )
    async with _watch(broker, cursor) as resumed:
        assert (await asyncio.wait_for(anext(resumed), 1)).event.event_type == "STREAM_RESET"
        assert (await asyncio.wait_for(anext(resumed), 1)).event.payload["text"] == "current"
