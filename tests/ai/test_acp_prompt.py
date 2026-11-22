#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACP uses authorized session admission, prompt-scoped cancellation and committed history."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.ai.acp import ACPAgent
from linktools.ai.core import ExecutionDeltaType, ExecutionEventType, ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.session import ToolProcessItem


@pytest.mark.asyncio
async def test_acp_cancel_during_admission_targets_only_the_prompt_its_connection_owns() -> None:
    acp = pytest.importorskip("acp")
    started, admit, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    principal = Principal("caller", "tenant")
    connection = SimpleNamespace(session_update=AsyncMock())

    async def cancel_execution(**kwargs):
        assert not kwargs.get("force", False)
        finished.set()

    async def wait(**kwargs):
        await finished.wait()
        return SimpleNamespace(result=SimpleNamespace(status=ExecutionStatus.CANCELLED), observation_error=None)

    execution = SimpleNamespace(execution_id="owned", wait=wait, cancel=AsyncMock(side_effect=cancel_execution))

    async def start(*args, **kwargs):
        started.set()
        await admit.wait()
        return execution

    runtime = SimpleNamespace(sessions=SimpleNamespace(start=start))
    agent, other = ACPAgent(runtime, principal=principal), ACPAgent(runtime, principal=principal)
    agent.on_connect(connection)
    other.on_connect(SimpleNamespace(session_update=AsyncMock()))
    for instance in (agent, other):
        await instance.initialize(acp.PROTOCOL_VERSION)

    prompt = asyncio.create_task(agent.prompt("session", [SimpleNamespace(text="hi")]))
    try:
        await asyncio.wait_for(started.wait(), 1)
        await other.cancel("session")
        execution.cancel.assert_not_awaited()
        cancel = asyncio.create_task(agent.cancel("session"))
        await asyncio.sleep(0)
        assert not cancel.done()
        admit.set()
        await asyncio.wait_for(cancel, 1)
        assert (await asyncio.wait_for(prompt, 1)).stop_reason == "cancelled"
        await agent.cancel("session")
        execution.cancel.assert_awaited_once()
    finally:
        admit.set()
        finished.set()
        await asyncio.gather(prompt, return_exceptions=True)


@pytest.mark.asyncio
async def test_acp_admission_busy_is_a_safe_protocol_error() -> None:
    acp = pytest.importorskip("acp")
    error = AIError(ErrorCode.SESSION_BUSY, "private provider error", safe_details={"owner": "other execution"})
    start = AsyncMock(side_effect=error)
    agent = ACPAgent(SimpleNamespace(sessions=SimpleNamespace(start=start)), principal=Principal("caller", "tenant"))
    await agent.initialize(acp.PROTOCOL_VERSION)
    with pytest.raises(acp.RequestError) as failed:
        await agent.prompt("session", [SimpleNamespace(text="hi")])
    assert str(failed.value) == "Session busy"
    assert failed.value.data["code"] == "SESSION_BUSY"
    assert failed.value.data["safe_details"] == {}
    await agent.cancel("session")
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_acp_load_replays_only_committed_public_history_in_turn_order(tmp_path) -> None:
    acp = pytest.importorskip("acp")
    from linktools.ai.workspace import Workspace

    workspace = Workspace(tmp_path, {})
    principal = Principal("caller", "tenant")

    def turn(text, *, committed=True, process=()):
        return SimpleNamespace(
            conversation_committed=committed, process=process, execution_id=f"execution-{text}",
            user_input={"version": 1, "prompt": {"kind": "text", "text": text}},
            items=(
                SimpleNamespace(item_kind="thinking", content="private thinking"),
                SimpleNamespace(item_kind="tool_call", tool_call_id="c-1", content="private input"),
                SimpleNamespace(item_kind="tool_result", content="private result"),
                SimpleNamespace(item_kind="assistant", content=f"reply {text}"),
            ),
        )

    async def session_detail(session_id, *, principal, cursor):
        assert session_id == "session" and principal == expected_principal
        if cursor is None:
            page = SimpleNamespace(items=(turn("new"), turn("uncommitted", committed=False)), next_cursor="older")
        else:
            assert cursor == "older"
            page = SimpleNamespace(items=(turn("old", process=(ToolProcessItem("tool:c-1", "search", "SUCCEEDED"),)),),
                                   next_cursor=None)
        return SimpleNamespace(session=SimpleNamespace(cwd=None), timeline=page)

    expected_principal = principal
    agent = ACPAgent(SimpleNamespace(history=SimpleNamespace(session_detail=session_detail)),
                     principal=principal, workspace=workspace)
    connection = SimpleNamespace(session_update=AsyncMock())
    agent.on_connect(connection)
    await agent.initialize(acp.PROTOCOL_VERSION)
    await agent.load_session(str(workspace.root), "session")

    updates = [request.args[1] for request in connection.session_update.await_args_list]
    assert [value.session_update for value in updates] == [
        "user_message_chunk", "tool_call", "agent_message_chunk", "user_message_chunk", "agent_message_chunk",
    ]
    assert [value.content.text for value in updates if hasattr(value.content, "text")] == [
        "old", "reply old", "new", "reply new",
    ]
    assert (updates[1].tool_call_id, updates[1].title, updates[1].status) == ("execution-old:c-1", "search", "completed")


@pytest.mark.asyncio
async def test_acp_prompt_streams_public_tool_status_and_answer_without_thinking_or_payloads() -> None:
    acp = pytest.importorskip("acp")

    async def wait(*, on_event, **kwargs):
        for event_type, payload in (
            (ExecutionDeltaType.ASSISTANT_THINKING_DELTA, {"text": "private"}),
            (ExecutionEventType.TOOL_CALL_STARTED, {"call_id": "c-1", "tool_name": "search", "args": "private"}),
            (ExecutionEventType.TOOL_CALL_FINISHED, {
                "call_id": "c-1", "tool_name": "search", "status": "SUCCEEDED", "result": "private",
            }),
            (ExecutionDeltaType.ASSISTANT_TEXT_DELTA, {"text": "public"}),
        ):
            await on_event(SimpleNamespace(depth=0, execution_id="execution",
                                           event=SimpleNamespace(event_type=event_type.value, payload=payload)))
        return SimpleNamespace(result=SimpleNamespace(status=ExecutionStatus.SUCCEEDED), observation_error=None)

    start = AsyncMock(return_value=SimpleNamespace(execution_id="execution", wait=wait))
    agent = ACPAgent(SimpleNamespace(sessions=SimpleNamespace(start=start)), principal=Principal("caller", "tenant"))
    connection = SimpleNamespace(session_update=AsyncMock())
    agent.on_connect(connection)
    await agent.initialize(acp.PROTOCOL_VERSION)
    await agent.prompt("session", [SimpleNamespace(text="hi")])
    updates = [request.args[1] for request in connection.session_update.await_args_list]
    assert [value.session_update for value in updates] == ["tool_call", "tool_call_update", "agent_message_chunk"]
    assert (updates[0].tool_call_id, updates[0].status, updates[1].status) == ("execution:c-1", "in_progress", "completed")
    assert updates[2].content.text == "public"
    assert all("private" not in value.model_dump_json() for value in updates)
