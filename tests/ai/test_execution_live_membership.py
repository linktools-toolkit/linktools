#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model metadata uses the live tree's authorized discovery without a duplicate scan."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionStreamEvent, ExecutionTreeEvent, TaskModelProjection
from linktools.ai.execution import tree as tree_module

from .test_execution_root_observation import _History, _handle


class _ScopedHistory(_History):
    async def read_model_interaction_metadata(self, execution_id, **kwargs):
        rows = await super().read_model_interaction_metadata(execution_id, **kwargs)
        return tuple(replace(row, execution_id=execution_id) for row in rows)


@pytest.mark.asyncio
async def test_model_watch_discovery_finds_a_child_before_any_child_event(monkeypatch):
    handle, reader, events, broker, active = _handle()
    history = _ScopedHistory()
    present = False
    authority_reads = []
    captures = 0
    closed = asyncio.Event()

    async def children(execution_id, *, principal):
        authority_reads.append(execution_id)
        return (reader.child,) if execution_id == "execution" and present else ()

    async def parked(execution_id, **kwargs):
        await closed.wait()
        if False:
            yield

    async def capture(*args, **kwargs):
        nonlocal captures
        captures += 1
        return await original_capture(*args, **kwargs)

    original_capture = handle._binding.capture_tree
    reader.list_children = children
    events.stream = parked
    handle = replace(handle, _binding=replace(
        handle._binding, history=history, capture_tree=capture,
    ))
    monkeypatch.setattr(tree_module, "_DISCOVERY_BACKOFF_INITIAL", 0.01)
    watch = handle.watch(include_model_interactions=True)
    try:
        first = await asyncio.wait_for(anext(watch), 1)
        assert isinstance(first.event, TaskModelProjection) and first.event.item.execution_id == "execution"
        present = True
        child = await asyncio.wait_for(anext(watch), 1)
        assert isinstance(child.event, TaskModelProjection) and child.event.item.execution_id == "child"
        assert child.event.item.depth == 1
        assert "child" in authority_reads
        assert authority_reads.count("execution") == 2
        assert captures == 0
    finally:
        await watch.aclose()
    assert not active and not broker._subscriptions
    assert all(subscription.closed for subscription in history.subscriptions)


@pytest.mark.asyncio
@pytest.mark.parametrize("regression", ["event_watermark", "deleted_child"])
async def test_composed_idle_sample_enforces_latest_delivered_cursor(monkeypatch, regression):
    handle, reader, events, broker, active = _handle()
    tick = asyncio.Event()
    deleted = False

    async def parked(execution_id, **kwargs):
        if execution_id == "child":
            yield ExecutionStreamEvent("child", 1, "MODEL_REQUEST_STARTED", {})
        await asyncio.Event().wait()

    async def inspect(execution_id, **kwargs):
        if deleted and execution_id == "child":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return await original_inspect(execution_id, **kwargs)

    async def children(execution_id, **kwargs):
        if deleted:
            return ()
        return await original_list_children(execution_id, **kwargs)

    async def sleep(delay):
        assert delay == 1.0
        await tick.wait()

    original_inspect = reader.inspect
    original_list_children = reader.list_children
    reader.inspect = inspect
    reader.list_children = children
    events.stream = parked
    history = _ScopedHistory()
    handle = replace(handle, _binding=replace(handle._binding, history=history))
    monkeypatch.setattr(tree_module, "asyncio", SimpleNamespace(**(vars(asyncio) | {"sleep": sleep})))
    watch = handle.watch(include_model_interactions=True)
    try:
        while True:
            item = await asyncio.wait_for(anext(watch), 1)
            if isinstance(item.event, ExecutionTreeEvent):
                assert item.event.execution_id == "child" and item.event.event.durable_seq == 1
                break
        if regression == "event_watermark":
            reader.child = replace(reader.child, event_seq=0)
            error = ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            deleted = True
            error = ErrorCode.AUTHORIZATION_DENIED
        tick.set()
        with pytest.raises(AIError) as caught:
            await asyncio.wait_for(anext(watch), 1)
        assert caught.value.code is error
    finally:
        await watch.aclose()
    assert not active and not broker._subscriptions


@pytest.mark.asyncio
async def test_capture_checks_cursors_at_read_start_not_after_a_concurrent_event(monkeypatch):
    handle, reader, events, broker, active = _handle()
    history = _ScopedHistory()
    entered = asyncio.Event()
    capture_release = asyncio.Event()
    event_release = asyncio.Event()
    tick = asyncio.Event()
    reader.root = replace(reader.root, event_seq=0)
    root_inspects = 0

    async def parked(execution_id, **kwargs):
        if execution_id == "execution":
            await event_release.wait()
            yield ExecutionStreamEvent(execution_id, 1, "MODEL_REQUEST_STARTED", {})
        await asyncio.Event().wait()

    async def inspect(execution_id, **kwargs):
        nonlocal root_inspects
        view = await original_inspect(execution_id, **kwargs)
        if execution_id == "execution":
            root_inspects += 1
            if root_inspects == 2:
                entered.set()
                await capture_release.wait()
        return view

    async def sleep(delay):
        await tick.wait()
        tick.clear()

    original_inspect = reader.inspect
    reader.inspect = inspect
    events.stream = parked
    handle = replace(handle, _binding=replace(handle._binding, history=history))
    monkeypatch.setattr(tree_module, "asyncio", SimpleNamespace(**(vars(asyncio) | {"sleep": sleep})))
    watch = handle.watch(include_descendants=True, include_model_interactions=True)
    try:
        assert isinstance((await asyncio.wait_for(anext(watch), 1)).event, TaskModelProjection)
        assert isinstance((await asyncio.wait_for(anext(watch), 1)).event, TaskModelProjection)
        pending = asyncio.create_task(anext(watch))
        tick.set()
        await asyncio.wait_for(entered.wait(), 1)
        event_release.set()
        event = (await asyncio.wait_for(pending, 1)).event
        assert isinstance(event, ExecutionTreeEvent) and event.event.durable_seq == 1
        reader.root = replace(reader.root, event_seq=1)
        history.rows.append(replace(history.rows[0], model_request_seq=2))
        capture_release.set()
        projected = (await asyncio.wait_for(anext(watch), 1)).event
        assert isinstance(projected, TaskModelProjection) and projected.item.model_request_seq == 2
    finally:
        await watch.aclose()
    assert not active and not broker._subscriptions
