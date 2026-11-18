#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root observations reuse authorized event/model delivery without child discovery."""

from collections.abc import AsyncIterator
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionDeltaType, Page, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionEvent, ExecutionObservationEvent, ExecutionStreamEvent, ExecutionTreeEvent, ExecutionView, ModelInteractionItem, TaskModelProjection
from linktools.ai.execution.handle import Execution, ExecutionBinding
from linktools.ai.execution.tree import ExecutionTreeBroker, ExecutionTreeStreamer, _ExecutionTreeSubscription
from linktools.ai.observe.cursors import ExecutionWatchPositions, encode_execution_watch_cursor

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
        streamer.capture, streamer.replay, history=_History())
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

    class ScopedHistory(_History):
        async def read_model_interaction_metadata(self, execution_id: str, *, principal: Principal,
                agent_run_seq: int, after_model_request_seq: int, through_model_request_seq: int, limit: int) -> tuple[ModelInteractionItem, ...]:
            rows = await super().read_model_interaction_metadata(execution_id, principal=principal,
                agent_run_seq=agent_run_seq, after_model_request_seq=after_model_request_seq,
                through_model_request_seq=through_model_request_seq, limit=limit)
            return tuple(replace(row, execution_id=execution_id) for row in rows)

    history = ScopedHistory()
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
