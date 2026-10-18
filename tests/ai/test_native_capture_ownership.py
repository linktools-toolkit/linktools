#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native persistence declarations keep run-local timing and context ownership."""

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from linktools.ai.runtime import CheckpointCapability, ConversationCapability, TraceCapability
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._run_hooks import AgentRunCapture
from linktools.ai.runtime._journal import DURATION_NS_METADATA_KEY
from linktools.ai.runtime.state._steps import StagingAgentRunStore


@pytest.mark.asyncio
async def test_reused_trace_declaration_keeps_interleaved_tool_durations_per_run(monkeypatch) -> None:
    from linktools.ai.runtime import _persistence_capabilities

    store = StagingAgentRunStore()
    await store.initialize()
    declaration = TraceCapability()
    contexts = []
    for identity in ("first", "second"):
        recorder = AgentRunRecorder(store, execution_id=identity, agent_run_id=identity)
        capture = AgentRunCapture(recorder=recorder, agent_id="agent", agent_run_id=identity)
        ctx = SimpleNamespace(messages=[], conversation_id=identity, run_step=1)
        bound = declaration.bind(capture)
        await bound.before_run(ctx)
        contexts.append((recorder, ctx, bound))
    clock = iter((10, 20, 30, 50))
    monkeypatch.setattr(_persistence_capabilities, "monotonic_ns", lambda: next(clock))
    call = ToolCallPart("tool", {}, tool_call_id="same-provider-id")
    tool = ToolDefinition(name="tool")
    for recorder, ctx, bound in contexts:
        await bound.before_tool_execute(ctx, call=call, tool_def=tool, args={})
    for recorder, ctx, bound in contexts:
        assert await bound.after_tool_execute(ctx, call=call, tool_def=tool, args={}, result="done") == "done"
        await recorder.record_tool_result_boundary(ToolReturnPart("tool", "done", tool_call_id=call.tool_call_id), 1)
    durations = []
    for identity in ("first", "second"):
        events = await store.list_events(agent_run_id=identity)
        finished = [event for event in events if event.event_type == "TOOL_CALL_SUCCEEDED"]
        assert len(finished) == 1
        durations.append(int(finished[0].metadata[DURATION_NS_METADATA_KEY]))
    assert durations == [20, 30]


@pytest.mark.asyncio
async def test_reused_conversation_and_checkpoint_declarations_keep_context_per_run() -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    conversation = ConversationCapability()
    checkpoint = CheckpointCapability()
    for identity in ("first", "second"):
        user = ModelRequest(parts=[UserPromptPart(identity)])
        response = ModelResponse(parts=[TextPart("answer " + identity)])
        recorder = AgentRunRecorder(store, execution_id=identity, agent_run_id=identity)
        capture = AgentRunCapture(recorder=recorder, agent_id="agent", agent_run_id=identity)
        ctx = SimpleNamespace(messages=[user], conversation_id=identity, run_step=1)
        await capture.before_run(ctx)
        await capture.before_model_request(ctx, SimpleNamespace())
        await checkpoint.bind(capture).before_model_request(ctx, SimpleNamespace())
        recorder.append_transcript_message(response)
        ctx.messages.append(response)
        result = SimpleNamespace(all_messages=lambda: ctx.messages, output="done")
        await conversation.bind(capture).after_run(ctx, result=result)
        saved = await store.latest_checkpoint(agent_run_id=identity)
        assert saved.context_messages == [user, response]
        assert saved.messages == [user, response]
    first = await store.latest_checkpoint(agent_run_id="first")
    second = await store.latest_checkpoint(agent_run_id="second")
    assert first.context_messages != second.context_messages
