#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root observations reuse authorized event/model delivery without child discovery."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionDeltaType, Page, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionEvent, ExecutionObservationEvent, ExecutionStreamEvent, ExecutionTreeEvent, ExecutionView, ModelInteractionItem, TaskModelProjection
from linktools.ai.execution.handle import Execution, ExecutionBinding
from linktools.ai.execution._tree_ready import ExecutionTreeReady
from linktools.ai.execution.tree import ExecutionTreeBroker, ExecutionTreeStreamer, _ExecutionTreeSubscription
from linktools.ai.observe.cursors import ExecutionWatchPositions, encode_execution_observation_cursor, encode_execution_watch_cursor

from .test_execution_model_observation import _History
from .test_subagent_execution_tree import _ExecutionReader
from .test_unified_wait_contract import _NAMESPACE, _PRINCIPAL


class _Reader(_ExecutionReader):
    def __init__(self) -> None:
        super().__init__()
        self.root = replace(self.root, execution_id="execution", root_execution_id="execution", event_seq=4)
        self.child = replace(self.child, parent_execution_id="execution", root_execution_id="execution", event_seq=4)
        self.children_reads = 0
        self.revoked = False

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        assert principal == _PRINCIPAL
        if self.revoked:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return self.root if execution_id == "execution" else self.child

    async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
        self.children_reads += 1
        return (self.child,) if execution_id == "execution" else ()


class _Broker(ExecutionTreeBroker):
    def __init__(self) -> None:
        super().__init__()
        self.subscriptions = 0

    def subscribe(self, parent_execution_id: str) -> _ExecutionTreeSubscription:
        self.subscriptions += 1
        return super().subscribe(parent_execution_id)


class _Events:
    def __init__(self, reader: _Reader) -> None:
        self.reader = reader
        self.opened: list[str] = []
        self.closed: list[str] = []

    def records(self, execution_id: str) -> tuple[ExecutionEvent, ...]:
        return tuple(ExecutionEvent(execution_id, seq, kind, payload) for seq, kind, payload in (
            (1, "MODEL_REQUEST_STARTED", {"model": "fixture"}),
            (2, "TOOL_CALL_STARTED", {"tool_name": "read", "arguments": {"path": "test"}}),
            (3, "TOOL_CALL_FINISHED", {"tool_name": "read"}),
            (4, "EXECUTION_SUCCEEDED", {}),
        ))

    async def list(self, execution_id: str, *, principal: Principal, after_event_seq: int = 0, limit: int = 100) -> Page[ExecutionEvent]:
        await self.reader.inspect(execution_id, principal=principal)
        return Page(tuple(row for row in self.records(execution_id) if row.event_seq > after_event_seq)[:limit])

    async def stream(self, execution_id: str, *, principal: Principal, after_event_seq: int = 0) -> AsyncIterator[ExecutionStreamEvent]:
        self.opened.append(execution_id)
        try:
            if not after_event_seq:
                await self.reader.inspect(execution_id, principal=principal)
                yield ExecutionStreamEvent(execution_id, None, ExecutionDeltaType.ASSISTANT_OUTPUT_PREVIEW.value,
                    {"text": "visible answer", "preview_field": "answer", "private_reasoning": "must not escape"})
            for row in self.records(execution_id):
                if row.event_seq > after_event_seq:
                    await self.reader.inspect(execution_id, principal=principal)
                    yield ExecutionStreamEvent(execution_id, row.event_seq, row.event_type, row.payload)
        finally:
            self.closed.append(execution_id)


def _handle():
    reader = _Reader()
    events = _Events(reader)
    broker = _Broker()
    streamer = ExecutionTreeStreamer(reader, events, broker)
    active = set()
    binding = ExecutionBinding(_NAMESPACE, reader, lambda: None, active.add, active.discard,
        streamer.capture, streamer.replay, history=_History(), share_tree_capture=True)
    return Execution(binding, "execution", _PRINCIPAL, streamer.stream), reader, events, broker, active


@pytest.mark.asyncio
@pytest.mark.parametrize("models", [False, True])
async def test_root_watch_keeps_events_preview_models_and_resume_without_child_work(models: bool) -> None:
    handle, reader, events, broker, active = _handle()
    observed = [item async for item in handle.watch(include_descendants=False,
        include_content=True, include_output_preview=True, include_model_interactions=models)]
    trees = [item.event if isinstance(item, ExecutionObservationEvent) else item for item in observed]
    tree_events = [item for item in trees if isinstance(item, ExecutionTreeEvent)]
    assert all(item.execution_id == "execution" and item.depth == 0 for item in tree_events)
    assert [item.event.durable_seq for item in tree_events if item.event.durable_seq is not None] == [1, 2, 3, 4]
    preview = next(item.event.payload for item in tree_events if item.event.event_type == "ASSISTANT_OUTPUT_PREVIEW")
    assert preview == {"text": "visible answer", "preview_field": "answer"}
    if models:
        projected = [item for item in trees if isinstance(item, TaskModelProjection)]
        assert projected and all(item.item.request == {} and not item.item.content_included for item in projected)
    resumed = [item async for item in handle.watch(cursor=observed[-1].cursor, include_descendants=False,
        include_content=True, include_output_preview=True, include_model_interactions=models)]
    resumed_trees = [item.event if isinstance(item, ExecutionObservationEvent) else item for item in resumed]
    assert not any(isinstance(item, ExecutionTreeEvent) for item in resumed_trees)
    assert reader.children_reads == broker.subscriptions == 0
    assert set(events.opened) == set(events.closed) == {"execution"}
    assert not active


class _ScopedHistory(_History):
    async def read_model_interaction_metadata(self, execution_id: str, *, principal: Principal,
            agent_run_seq: int, after_model_request_seq: int, through_model_request_seq: int, limit: int) -> tuple[ModelInteractionItem, ...]:
        rows = await super().read_model_interaction_metadata(
            execution_id, principal=principal, agent_run_seq=agent_run_seq,
            after_model_request_seq=after_model_request_seq,
            through_model_request_seq=through_model_request_seq, limit=limit,
        )
        return tuple(replace(row, execution_id=execution_id) for row in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("descendants", [False, True])
@pytest.mark.parametrize("share_tree_capture", [False, True])
async def test_live_models_share_prepared_tree_but_final_drain_still_captures(
    descendants: bool, share_tree_capture: bool,
) -> None:
    handle, _, _, broker, active = _handle()
    captured = []
    capture = handle._binding.capture_tree

    async def record_capture(*args, **kwargs):
        result = await capture(*args, **kwargs)
        captured.append(result)
        return result

    handle = replace(handle, _binding=replace(handle._binding,
        capture_tree=record_capture, history=_ScopedHistory(), share_tree_capture=share_tree_capture))
    watch = handle.watch(include_descendants=descendants, include_model_interactions=True)
    expected = {("execution", 0, 4)}
    if descendants:
        expected.add(("child", 1, 4))
    try:
        first = await anext(watch)
        assert isinstance(first.event, TaskModelProjection)
        assert len(captured) == (0 if share_tree_capture else 1)
        rest = [value async for value in watch]
        assert {value.event.execution_id for value in rest if isinstance(value.event, ExecutionTreeEvent)} == {
            execution_id for execution_id, _, _ in expected
        }
        assert len(captured) == (1 if share_tree_capture else 2)
        assert {(view.execution_id, depth, cutoff) for view, depth, cutoff in captured[-1]} == expected
    finally:
        await watch.aclose()
    assert not active
    assert not broker._subscriptions


@pytest.mark.asyncio
async def test_live_model_startup_rejects_cursor_ahead_of_tree_before_subscribing() -> None:
    handle, _, _, broker, active = _handle()
    cursor = encode_execution_observation_cursor(_NAMESPACE, "tenant", "execution",
        include_content=False, event_seqs={"execution": 5})
    with pytest.raises(AIError) as caught:
        _ = [value async for value in handle.watch(cursor=cursor,
            include_descendants=False, include_model_interactions=True)]
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert not handle._binding.history.subscriptions
    assert not active
    assert not broker._subscriptions


@pytest.mark.asyncio
async def test_explicit_tree_watch_keeps_descendant_observation() -> None:
    handle, reader, events, broker, active = _handle()
    observed = [item async for item in handle.watch()]
    assert {item.execution_id for item in observed} == {"execution", "child"}
    assert reader.children_reads > 0 and broker.subscriptions > 0
    assert not active


@pytest.mark.asyncio
async def test_root_watch_does_not_bypass_event_reauthorization() -> None:
    handle, reader, events, broker, active = _handle()
    watch = handle.watch(include_descendants=False)
    await watch.__anext__()
    reader.revoked = True
    with pytest.raises(AIError) as caught:
        await watch.__anext__()
    assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    await watch.aclose()
    assert not active


@pytest.mark.asyncio
@pytest.mark.parametrize("preview_only", [False, True])
async def test_root_watch_rejects_descendant_cursor_scope(preview_only: bool) -> None:
    handle, reader, events, broker, active = _handle()
    positions = ExecutionWatchPositions({}, {"child": ("producer", 1)}) if preview_only else {"child": 1}
    cursor = encode_execution_watch_cursor(_NAMESPACE, "tenant", "execution", include_content=False,
        event_seqs=positions, preview_positions={"child": ("producer", 1)} if preview_only else None)
    with pytest.raises(AIError) as caught:
        _ = [item async for item in handle.watch(cursor=cursor, include_descendants=False)]
    assert caught.value.code is ErrorCode.CURSOR_INVALID
    assert reader.children_reads == broker.subscriptions == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("models", [False, True])
async def test_root_checkpoint_can_expand_to_tree_without_repeating_root_events(models: bool) -> None:
    handle, reader, events, broker, active = _handle()
    history = _ScopedHistory()
    handle = replace(handle, _binding=replace(handle._binding, history=history))
    root = [item async for item in handle.watch(include_descendants=False,
        include_output_preview=True, include_model_interactions=models)]
    assert reader.children_reads == broker.subscriptions == 0
    assert not active

    expanded = [item async for item in handle.watch(cursor=root[-1].cursor,
        include_output_preview=True, include_model_interactions=models)]
    values = [item.event if isinstance(item, ExecutionObservationEvent) else item for item in expanded]
    tree = [item for item in values if isinstance(item, ExecutionTreeEvent)]
    assert tree and {item.execution_id for item in tree} == {"child"}
    assert [item.event.durable_seq for item in tree if item.event.durable_seq is not None] == [1, 2, 3, 4]
    if models:
        # Model snapshots intentionally compensate on reconnect; they are upserts,
        # unlike the durable root event suffix that must not be replayed.
        projections = [item for item in values if isinstance(item, TaskModelProjection)]
        assert {item.item.execution_id for item in projections} == {"execution", "child"}
        assert all(subscription.closed for subscription in history.subscriptions)
    assert reader.children_reads > 0 and broker.subscriptions > 0
    assert sorted(events.opened) == sorted(events.closed)
    assert not broker._subscriptions
    assert not active


@pytest.mark.asyncio
async def test_live_tree_views_share_periodic_child_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.execution import tree

    class LiveEvents(_Events):
        async def stream(self, execution_id: str, *, principal: Principal,
                         after_event_seq: int = 0) -> AsyncIterator[ExecutionStreamEvent]:
            # Keep the execution open until the observer leaves.
            await asyncio.Event().wait()
            async for event in super().stream(execution_id, principal=principal, after_event_seq=after_event_seq):
                yield event

    monkeypatch.setattr(tree, "_DISCOVERY_BACKOFF_INITIAL", 0.05)
    reader = _Reader()
    streamer = ExecutionTreeStreamer(reader, LiveEvents(reader), ExecutionTreeBroker())
    readiness = [ExecutionTreeReady(dict) for _ in range(10)]
    streams = [streamer.stream("execution", principal=_PRINCIPAL, ready=ready) for ready in readiness]
    pending = [asyncio.create_task(anext(stream)) for stream in streams]
    try:
        await asyncio.wait_for(asyncio.gather(*(ready.wait() for ready in readiness)), 2)
        baseline = reader.children_reads
        await asyncio.wait_for(asyncio.gather(*(ready.sampled.wait() for ready in readiness)), 2)
        # The shared sample reads root and child, not one tree per observer.
        assert reader.children_reads - baseline == 2
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*(stream.aclose() for stream in streams))


@pytest.mark.asyncio
@pytest.mark.parametrize("lost", ["gap", "malformed", "unresolved"])
async def test_shared_topology_falls_back_when_coverage_is_unusable(
    monkeypatch: pytest.MonkeyPatch, lost: str,
) -> None:
    from linktools.ai.execution import tree
    from linktools.ai.execution.stream import deliver_shared_observation

    class RemoteTreeReader(_Reader):
        def __init__(self) -> None:
            super().__init__()
            self.root_reads = 0
            self.root_read = asyncio.Event()

        async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
            if execution_id == "not-found":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return await super().inspect(execution_id, principal=principal)

        async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
            self.children_reads += 1
            if execution_id == self.root.execution_id:
                self.root_reads += 1
                self.root_read.set()
            return ()

    attached = asyncio.Event()
    add_child = asyncio.Event()
    lose_coverage = asyncio.Event()

    class SharedEvents(_Events):
        async def stream(self, execution_id: str, *, principal: Principal,
                         after_event_seq: int = 0) -> AsyncIterator[ExecutionStreamEvent]:
            from time import time

            position = ("remote:" + execution_id, 1)
            deliver_shared_observation(execution_id, position, {
                "kind": "coverage", "tree": True, "published_at": time(),
                "authority_deadline": asyncio.get_running_loop().time() + 5,
            })
            if execution_id == "execution":
                attached.set()
                await add_child.wait()
                deliver_shared_observation(execution_id, (position[0], 2), {"kind": "child", "execution_id": "child"})
                await lose_coverage.wait()
                if lost == "gap":
                    record = {"kind": "gap"}
                else:
                    record = {"kind": "child", "execution_id": "" if lost == "malformed" else "not-found"}
                deliver_shared_observation(execution_id, (position[0], 3), record)
            else:
                yield ExecutionStreamEvent(execution_id, 1, "MODEL_REQUEST_STARTED", {})
            await asyncio.Event().wait()

    monkeypatch.setattr(tree, "_DISCOVERY_BACKOFF_INITIAL", 0.02)
    reader = RemoteTreeReader()
    streamer = ExecutionTreeStreamer(reader, SharedEvents(reader), ExecutionTreeBroker())
    ready = ExecutionTreeReady(lambda: {})
    stream = streamer.stream("execution", principal=_PRINCIPAL, ready=ready)
    pending = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(attached.wait(), 2)
        # Startup plus the first verified coverage boundary reconcile the index.
        async with asyncio.timeout(2):
            while reader.root_reads < 2:
                reader.root_read.clear()
                await reader.root_read.wait()
        baseline = reader.root_reads
        await asyncio.sleep(0.08)
        assert reader.root_reads == baseline
        add_child.set()
        child = await asyncio.wait_for(pending, 2)
        assert child.execution_id == "child"
        assert reader.root_reads == baseline

        reader.root_read.clear()
        pending = asyncio.create_task(anext(stream))
        lose_coverage.set()
        await asyncio.wait_for(reader.root_read.wait(), 2)
        assert reader.root_reads > baseline
        assert not pending.done()
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
