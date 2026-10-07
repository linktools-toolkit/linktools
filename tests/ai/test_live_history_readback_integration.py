#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live event locators read the same transcript before and after archival."""

import asyncio
import json
from collections.abc import Mapping

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionEventType, ExecutionStatus, JsonValue
from linktools.ai.runtime import ExecutionHistoryItem, ExecutionTreeEvent, Runtime, RuntimeStorage

from ._runtime_test_helpers import _UsageFunctionModel


class _Models:
    route_id = "default"
    provider = "test"
    model_identity = "test:live-history"
    vision = False
    model_digest = "a" * 64
    contract: dict[str, JsonValue] = {"provider": "test", "model": "live-history"}

    def __init__(self, model: FunctionModel) -> None:
        self.model = model

    def capture(self) -> "_Models":
        return self

    def resolve(self, route_id: str) -> "_Models":
        assert route_id == "default"
        return self

    def restore(self, payload: Mapping[str, JsonValue], *, route_id: str | None = None) -> "_Models":
        assert route_id in {None, "default"}
        assert dict(payload) == self.contract
        return self

    def materialize(self) -> FunctionModel:
        return self.model


def _contents(items: tuple[ExecutionHistoryItem, ...]) -> list[tuple[object, ...]]:
    return [(item.item_kind, item.tool_call_id, item.content, item.content_included) for item in items]


@pytest.mark.asyncio
async def test_parallel_tool_events_read_live_parts_and_preserve_cursor_after_archive() -> None:
    slow_entered = asyncio.Event()
    release_slow = asyncio.Event()
    start_read = asyncio.Event()
    finished_read = asyncio.Event()
    assistant_reads: list[str] = []
    live_items: tuple[ExecutionHistoryItem, ...] = ()
    frozen_first = None
    frozen_partial = None
    frozen_request = None

    async def quick(_ctx: AgentContext[None], value: str) -> None:
        assert value == "quick-argument"
        await asyncio.wait_for(slow_entered.wait(), 5)
        await asyncio.wait_for(start_read.wait(), 5)
        return None

    async def slow(_ctx: AgentContext[None], value: str) -> str:
        assert value == "slow-argument"
        slow_entered.set()
        await asyncio.wait_for(release_slow.wait(), 10)
        return "slow-result"

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("all finished")])
        return ModelResponse(parts=[
            TextPart("before tools"),
            ToolCallPart("quick", {"value": "quick-argument"}, tool_call_id="quick-call"),
            ToolCallPart("slow", {"value": "slow-argument"}, tool_call_id="slow-call"),
        ])

    group = CapabilityGroup("live-history")
    group.tool(quick, effect_policy="replay_safe")
    group.tool(slow, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("quick", "slow"))
    async with Runtime.open(
        "live-history", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("run both tools")
        principal = runtime.default_principal

        async def observe(tree_event: ExecutionTreeEvent) -> None:
            nonlocal live_items, frozen_first, frozen_partial, frozen_request
            event = tree_event.event
            payload = event.payload
            assert not any(secret in json.dumps(payload) for secret in (
                "quick-argument", "slow-argument", "before tools", "all finished",
            ))
            if event.event_type == ExecutionEventType.ASSISTANT_PART_COMPLETED:
                page = await runtime.executions.history(
                    tree_event.execution_id, principal=principal, include_content=True,
                    agent_run_sequence=payload["agent_run_sequence"],
                    message_sequence=payload["message_sequence"], part_index=payload["part_index"],
                )
                assert len(page.items) == 1
                assert page.items[0].item_kind == "assistant"
                assistant_reads.append(page.items[0].content)
                if frozen_partial is None:
                    assert page.items[0].request_sequence is None
                    frozen_partial = await execution.history(include_content=True, limit=1)
            if payload.get("call_id") != "quick-call":
                return
            selector = {"agent_run_sequence": payload["agent_run_sequence"], "tool_call_id": payload["call_id"]}
            if event.event_type == ExecutionEventType.TOOL_CALL_STARTED:
                page = await runtime.executions.history(
                    tree_event.execution_id, principal=principal, include_content=True, **selector,
                )
                assert _contents(page.items) == [("tool_call", "quick-call", {"value": "quick-argument"}, True)]
                assert "arguments" not in payload and "content" not in payload
                start_read.set()
            elif event.event_type == ExecutionEventType.TOOL_CALL_FINISHED:
                assert slow_entered.is_set() and not release_slow.is_set(), payload
                exact = await runtime.executions.history(
                    tree_event.execution_id, principal=principal, include_content=True, **selector,
                )
                assert _contents(exact.items) == [
                    ("tool_call", "quick-call", {"value": "quick-argument"}, True),
                    ("tool_result", "quick-call", None, True),
                ]
                metadata = await runtime.executions.history(tree_event.execution_id, principal=principal, **selector)
                assert len(metadata.items) == 2
                assert all(item.content is None and item.content_included is False for item in metadata.items)
                assert all(item.request_sequence == 1 and item.step_index is not None for item in metadata.items)
                frozen_request = await execution.history(request_sequence=1, limit=1)
                assert frozen_request.next_cursor is not None
                normal = await runtime.executions.history(tree_event.execution_id, principal=principal, include_content=True)
                live_items = normal.items
                assert _contents(tuple(item for item in normal.items if item.tool_call_id == "quick-call")) == _contents(exact.items)
                assert not any(item.item_kind == "tool_result" and item.tool_call_id == "slow-call" for item in normal.items)
                transcript = await runtime.executions.transcript(tree_event.execution_id, principal=principal, include_content=True)
                assert [item.text for item in transcript.items] == ["run both tools", *assistant_reads]
                frozen_first = await runtime.executions.history(
                    tree_event.execution_id, principal=principal, include_content=True, limit=1,
                )
                assert frozen_first.next_cursor is not None
                finished_read.set()

        observed = asyncio.create_task(execution.wait(on_event=observe, timeout_seconds=10))
        finished = asyncio.create_task(finished_read.wait())
        try:
            done, _ = await asyncio.wait((finished, observed), timeout=8, return_when=asyncio.FIRST_COMPLETED)
            if observed in done:
                await observed
            assert finished in done, "TOOL_CALL_FINISHED must be readable while its sibling remains blocked"
        finally:
            release_slow.set()
            finished.cancel()
            await asyncio.gather(finished, return_exceptions=True)
            if not observed.done():
                await asyncio.wait_for(observed, 5)
        result = (await observed).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert assistant_reads == ["before tools", "all finished"]
        assert frozen_first is not None
        frozen_items = list(frozen_first.items)
        cursor = frozen_first.next_cursor
        while cursor is not None:
            page = await runtime.executions.history(
                execution.execution_id, principal=principal, include_content=True, limit=1, cursor=cursor,
            )
            frozen_items.extend(page.items)
            assert len(frozen_items) <= len(live_items)
            cursor = page.next_cursor
        assert _contents(tuple(frozen_items)) == _contents(live_items)
        fresh = await runtime.executions.history(execution.execution_id, principal=principal, include_content=True)
        assert [(item.tool_call_id, item.content) for item in fresh.items if item.item_kind == "tool_result"] == [
            ("quick-call", None), ("slow-call", "slow-result"),
        ]
        assert [item.content for item in fresh.items if item.item_kind == "assistant"] == assistant_reads
        interactions = (await execution.model_interactions()).items
        by_request = {item.request_sequence: item for item in interactions}
        assert set(by_request) == {1, 2}
        for item in fresh.items:
            if item.item_kind in {"user", "system"}:
                assert item.request_sequence is None and item.step_index is None
            else:
                expected = 2 if item.content == "all finished" else 1
                assert item.request_sequence == expected
                assert item.step_index == by_request[expected].step_index
        for method in (execution.history, execution.trace):
            assert (await method(request_sequence=99)).items == ()
            selected = (await method(request_sequence=1, tool_call_id="quick-call")).items
            assert len(selected) == 2
            assert (await method(step_index=by_request[1].step_index)).items
        assert frozen_partial is not None
        partial_tail = await execution.history(include_content=True, cursor=frozen_partial.next_cursor)
        partial_response = [item for item in partial_tail.items if item.item_kind == "assistant"]
        assert len(partial_response) == 1
        assert partial_response[0].content == "before tools"
        assert partial_response[0].request_sequence is None
        assert partial_response[0].step_index is None
        assert frozen_request is not None
        frozen_filtered = list(frozen_request.items)
        cursor = frozen_request.next_cursor
        while cursor is not None:
            page = await execution.history(request_sequence=1, cursor=cursor, limit=1)
            frozen_filtered.extend(page.items)
            cursor = page.next_cursor
        assert not any(item.item_kind == "tool_result" and item.tool_call_id == "slow-call" for item in frozen_filtered)
        assert len((await execution.history(request_sequence=1)).items) == len(frozen_filtered) + 1
        trace = (await execution.trace(request_sequence=2)).items
        response = next(item for item in trace if item.payload["kind"] == "MODEL_RESPONSE")
        final = next(item for item in fresh.items if item.content == "all finished")
        assert response.payload["message_sequence"] == final.sequence



@pytest.mark.asyncio
async def test_child_event_locator_does_not_mix_same_call_id_in_parent_history() -> None:
    child_read = asyncio.Event()
    child_ids: set[str] = set()

    async def echo(_ctx: AgentContext[None], value: str) -> str:
        return value

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        parent = any(tool.name == "delegate_task" for tool in info.function_tools)
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            if not parent:
                await asyncio.wait_for(child_read.wait(), 5)
            return ModelResponse(parts=[TextPart("parent done" if parent else "child done")])
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task" if parent else "echo",
            {"subagent_id": "child", "task": "echo child-value"} if parent else {"value": "child-value"},
            tool_call_id="same-call",
        )])

    group = CapabilityGroup("child-history")
    group.tool(echo, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=(), allow_subagents=("child",))
    group.agent("child", model="default", allow_tools=("echo",), allow_subagents=())
    async with Runtime.open(
        "child-history", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.in_memory(), capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("delegate")
        principal = runtime.default_principal

        async def observe(tree_event: ExecutionTreeEvent) -> None:
            if tree_event.execution_id == execution.execution_id:
                return
            child_ids.add(tree_event.execution_id)
            event = tree_event.event
            if event.event_type != ExecutionEventType.TOOL_CALL_FINISHED:
                return
            selector = {
                "agent_run_sequence": event.payload["agent_run_sequence"],
                "tool_call_id": event.payload["call_id"],
            }
            child = await runtime.executions.history(
                tree_event.execution_id, principal=principal, include_content=True, **selector,
            )
            assert _contents(child.items) == [
                ("tool_call", "same-call", {"value": "child-value"}, True),
                ("tool_result", "same-call", "child-value", True),
            ]
            parent = await runtime.executions.history(
                execution.execution_id, principal=principal, include_content=True, **selector,
            )
            assert len(parent.items) == 1
            assert parent.items[0].tool_name == "delegate_task"
            assert parent.items[0].item_kind == "tool_call"
            child_read.set()

        result = (await execution.wait(on_event=observe, timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert child_read.is_set() and len(child_ids) == 1
        child_id = next(iter(child_ids))
        history = await runtime.executions.history(child_id, principal=principal, include_content=True)
        assert [item.content for item in history.items if item.item_kind == "assistant"] == ["child done"]
        transcript = await runtime.executions.transcript(child_id, principal=principal, include_content=True)
        assert [item.text for item in transcript.items] == ["echo child-value", "child done"]
