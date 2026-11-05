#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native persistence declarations keep run-local timing and context ownership."""

from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import ToolDefinition

from linktools.ai.capability import ExecutionCapability, TraceCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._run_hooks import AgentRunCapture
from linktools.ai.runtime._conversation_service import ConversationService
from linktools.ai.runtime._persistence_services import TraceService
from linktools.ai.observe.journal import DURATION_NS_METADATA_KEY
from linktools.ai.persistence.runs import StagingAgentRunStore


@pytest.mark.asyncio
async def test_reused_trace_declaration_keeps_interleaved_tool_durations_per_run(monkeypatch) -> None:
    from linktools.ai.runtime import _persistence_services

    store = StagingAgentRunStore()
    await store.initialize()
    declaration = TraceCapability()
    contexts = []
    for identity in ("first", "second"):
        recorder = AgentRunRecorder(store, execution_id=identity, agent_run_id=identity)
        capture = AgentRunCapture(recorder=recorder, agent_id="agent", agent_run_id=identity)
        ctx = SimpleNamespace(messages=[], conversation_id=identity, run_step=1)
        bound = declaration.bind(TraceService(capture))
        await bound.before_run(ctx)
        contexts.append((recorder, ctx, bound))
    clock = iter((10, 20, 30, 50))
    monkeypatch.setattr(_persistence_services, "monotonic_ns", lambda: next(clock))
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
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    store = StagingAgentRunStore()
    await store.initialize()
    conversation = ExecutionCapability(retain_history=True)
    checkpoint = native_checkpoint()
    for identity in ("first", "second"):
        recorder = AgentRunRecorder(store, execution_id=identity, agent_run_id=identity)
        capture = AgentRunCapture(recorder=recorder, agent_id="agent", agent_run_id=identity)
        agent = Agent(TestModel(custom_output_text="answer " + identity), capabilities=(
            capture, conversation.bind(ConversationService(capture), retain_history=True), checkpoint,
        ))
        result = await agent.run(identity, run_id=identity, conversation_id=identity)
        saved = await checkpoint.store.latest_snapshot(run_id=identity)
        assert saved is not None and saved.messages == result.all_messages()
        assert [part.content for message in saved.messages for part in message.parts
                if isinstance(part, UserPromptPart)] == [identity]
    first = await checkpoint.store.latest_snapshot(run_id="first")
    second = await checkpoint.store.latest_snapshot(run_id="second")
    assert first.messages != second.messages
