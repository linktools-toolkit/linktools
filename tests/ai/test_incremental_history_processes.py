#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent readers observe committed model and tool boundaries."""

import asyncio
import multiprocessing
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import AgentContext, CapabilityGroup
from linktools.ai.core import ExecutionEventType, ExecutionStatus, Principal
from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage

from ._runtime_test_helpers import _UsageFunctionModel, _wait_for_committed
from .test_live_history_readback_integration import _Models


def _writer_process(root: str, pipe: Connection) -> None:
    async def run() -> None:
        slow_entered = asyncio.Event()

        async def quick(_context: AgentContext[None]) -> str:
            await slow_entered.wait()
            return "quick-result"

        async def slow(_context: AgentContext[None]) -> str:
            slow_entered.set()
            assert await asyncio.to_thread(pipe.recv) == "release-tool"
            return "slow-result"

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart("finished")])
            pipe.send(("provider-entered",))
            assert await asyncio.to_thread(pipe.recv) == "release-provider"
            return ModelResponse(parts=[
                TextPart("calling tools"),
                ToolCallPart("quick", {}, tool_call_id="quick"),
                ToolCallPart("slow", {}, tool_call_id="slow"),
            ])

        group = CapabilityGroup("process-history")
        group.tool(quick, effect_policy="replay_safe")
        group.tool(slow, effect_policy="replay_safe")
        group.agent("default", model="default", allow_tools=("quick", "slow"))
        async with Runtime.open(
            "process-history", storage=RuntimeStorage.filesystem(Path(root)),
            models=_Models(_UsageFunctionModel(model)), capabilities=(group,),
        ) as runtime:
            execution = await runtime.agents.get("default").start("process prompt")
            principal = runtime.default_principal
            pipe.send(("execution", execution.execution_id, principal))

            async def observe(tree_event) -> None:
                event = tree_event.event
                if event.event_type == ExecutionEventType.TOOL_CALL_FINISHED and event.payload.get("call_id") == "quick":
                    pipe.send(("quick-finished",))

            result = (await execution.wait(on_event=observe, timeout_seconds=50)).result
            pipe.send(("finished", result.status.value))
            assert await asyncio.to_thread(pipe.recv) == "close"

    try:
        asyncio.run(run())
    except BaseException:
        pipe.send(("error", traceback.format_exc()))
        raise
    finally:
        pipe.close()


async def _receive(pipe: Connection) -> tuple:
    if not await asyncio.to_thread(pipe.poll, 35):
        raise AssertionError("writer did not reach its next durable boundary")
    message = pipe.recv()
    assert message[0] != "error", message[1:]
    return message


def _start_writer(root: Path):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_writer_process, args=(str(root), child))
    process.start()
    child.close()
    return process, parent


async def _stop_writer(process, pipe: Connection) -> None:
    if process.is_alive():
        process.terminate()
    await asyncio.to_thread(process.join, 5)
    if process.is_alive():
        process.kill()
        await asyncio.to_thread(process.join, 5)
    pipe.close()


@pytest.mark.asyncio
async def test_independent_live_readers_keep_frozen_cursor_through_terminal_commit(tmp_path: Path) -> None:
    process, pipe = _start_writer(tmp_path)
    try:
        admission = await _receive(pipe)
        if admission[0] == "provider-entered":
            admission = await _receive(pipe)
            provider_entered = True
        else:
            provider_entered = False
        assert admission[0] == "execution"
        execution_id, principal = admission[1:]
        assert isinstance(principal, Principal)
        if not provider_entered:
            assert (await _receive(pipe))[0] == "provider-entered"
        async with RuntimeHistory.open("process-history", storage=RuntimeStorage.filesystem(tmp_path)) as first:
            async with RuntimeHistory.open("process-history", storage=RuntimeStorage.filesystem(tmp_path)) as second:
                for reader in (first, second):
                    interactions = await _wait_for_committed(
                        lambda: reader.model_interactions(execution_id, principal=principal, include_content=True),
                        lambda page: bool(page.items),
                    )
                    assert len(interactions.items) == 1
                    interaction = interactions.items[0]
                    assert interaction.status == "RUNNING"
                    assert interaction.request == {}
                    assert interaction.response is None
                pipe.send("release-provider")
                assert (await _receive(pipe))[0] == "quick-finished"
                for reader in (first, second):
                    live = await _wait_for_committed(
                        lambda: reader.history(execution_id, principal=principal, include_content=True),
                        lambda page: any(item.item_kind == "tool_result" and item.tool_call_id == "quick" for item in page.items),
                    )
                    results = {item.tool_call_id: item.content for item in live.items if item.item_kind == "tool_result"}
                    assert results == {"quick": "quick-result"}
                    interactions = await reader.model_interactions(execution_id, principal=principal)
                    assert interactions.items[0].status == "SUCCEEDED"
                frozen = await first.history(execution_id, principal=principal, include_content=True, limit=1)
                assert frozen.next_cursor is not None
                pipe.send("release-tool")
                assert await _receive(pipe) == ("finished", ExecutionStatus.SUCCEEDED.value)
                frozen_items = list(frozen.items)
                cursor = frozen.next_cursor
                while cursor is not None:
                    page = await second.history(
                        execution_id, principal=principal, include_content=True, limit=1, cursor=cursor,
                    )
                    frozen_items.extend(page.items)
                    cursor = page.next_cursor
                assert {item.tool_call_id for item in frozen_items if item.item_kind == "tool_result"} == {"quick"}
                fresh = await first.history(execution_id, principal=principal, include_content=True)
                assert {item.tool_call_id for item in fresh.items if item.item_kind == "tool_result"} == {"quick", "slow"}
        pipe.send("close")
        await asyncio.to_thread(process.join, 10)
        assert process.exitcode == 0
        async with RuntimeHistory.open("process-history", storage=RuntimeStorage.filesystem(tmp_path)) as reopened:
            archived = await reopened.history(execution_id, principal=principal, include_content=True)
            assert archived.items == fresh.items
    finally:
        await _stop_writer(process, pipe)


@pytest.mark.asyncio
async def test_reader_retains_request_metadata_after_writer_process_dies(tmp_path: Path) -> None:
    process, pipe = _start_writer(tmp_path)
    try:
        messages = [await _receive(pipe), await _receive(pipe)]
        admission = next(message for message in messages if message[0] == "execution")
        assert any(message[0] == "provider-entered" for message in messages)
        execution_id, principal = admission[1:]
        async with RuntimeHistory.open("process-history", storage=RuntimeStorage.filesystem(tmp_path)) as observer:
            committed = await _wait_for_committed(
                lambda: observer.model_interactions(execution_id, principal=principal, include_content=True),
                lambda page: bool(page.items),
            )
            assert committed.items[0].status == "RUNNING"
        process.kill()
        await asyncio.to_thread(process.join, 5)
        async with RuntimeHistory.open("process-history", storage=RuntimeStorage.filesystem(tmp_path)) as reader:
            interactions = await reader.model_interactions(execution_id, principal=principal, include_content=True)
            assert len(interactions.items) == 1
            assert interactions.items[0].status == "RUNNING"
            assert interactions.items[0].request == {}
            assert interactions.items[0].response is None
            history = await reader.history(execution_id, principal=principal, include_content=True)
            assert [item.content for item in history.items if item.item_kind == "user"] == ["process prompt"]
            assert not any(item.item_kind == "assistant" for item in history.items)
    finally:
        await _stop_writer(process, pipe)
