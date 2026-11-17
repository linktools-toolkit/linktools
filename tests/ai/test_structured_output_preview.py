#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in structured output previews never replace validated, durable results."""

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.messages import FinalResultEvent, PartDeltaEvent, PartStartEvent, ToolCallPart, ToolCallPartDelta
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from linktools.ai.agent import bind_output
from linktools.ai.capability import ExecutionCapability, CapabilityContribution
from linktools.ai.core import ExecutionDeltaType
from linktools.ai.execution.events import ExecutionDelta, project_event_payload
from linktools.ai.runtime._agent_executor import _event_stream_capability
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._output_preview import OutputPreview
from linktools.ai.persistence._step_archive import StagingAgentRunStore


class Answer(BaseModel):
    message: str = Field(max_length=300)
    internal: str


def preview():
    return OutputPreview.from_binding(bind_output(Answer), ExecutionCapability(output_preview_field="message"))


def test_requires_explicit_field_and_output_tool_identity():
    assert OutputPreview.from_binding(bind_output(Answer), ExecutionCapability()) is None
    assert OutputPreview.from_binding(bind_output(Answer), ExecutionCapability(output_preview_field="missing")) is None
    assert OutputPreview.from_binding(bind_output(), ExecutionCapability(output_preview_field="message")) is None
    current = preview()
    assert current.observe(PartStartEvent(index=0, part=ToolCallPart("business", '{"message":"secret"}', "call"))) is None
    assert current.observe(FinalResultEvent(tool_name=None, tool_call_id=None)) is None
    assert current.observe(FinalResultEvent(tool_name="output", tool_call_id="other")) is None


def test_partial_string_unicode_replacement_and_bounded_snapshot():
    current = preview()
    current.observe(PartStartEvent(index=0, part=ToolCallPart("output", '{"internal":"secret","message":"你', "call")))
    assert current.observe(FinalResultEvent(tool_name="output", tool_call_id="call"))[0] == "你"
    assert current.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='好\\n\\"ok\\"')))[0] == '你好\n"ok"'
    assert current.observe(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta="x" * 400)))[0].endswith("x")
    assert len(current.text) == 300
    # Reusing a part index is a replacement, not a continuation of the authenticated tool.
    assert current.observe(PartStartEvent(index=0, part=ToolCallPart("other", '{"message":"secret"}', "new")))[0] == ""
    assert current.new_stream().selected is None


@pytest.mark.parametrize("maximum, expected", [(None, 300), (500, 300), (40, 40)])
def test_preview_honors_sdk_cap_and_smaller_schema_limit(maximum, expected):
    class Result(BaseModel):
        message: str = Field(max_length=maximum)

    current = OutputPreview.from_binding(bind_output(Result), ExecutionCapability(output_preview_field="message"))
    current.observe(PartStartEvent(index=0, part=ToolCallPart("output", {"message": "x" * 500}, "call")))
    assert current.observe(FinalResultEvent(tool_name="output", tool_call_id="call"))[0] == "x" * expected


def test_observation_requires_independent_opt_in_and_keeps_other_content_redacted():
    payload = {"text": "safe", "preview_field": "message", "preview_seq": 1,
               "agent_run_seq": 1, "model_request_seq": 1, "stream_attempt_seq": 1,
               "raw_args": "secret", "thinking": "secret"}
    kind = ExecutionDeltaType.ASSISTANT_OUTPUT_PREVIEW
    assert project_event_payload(kind, payload, include_content=False) == {}
    assert project_event_payload(kind, payload, include_content=True) == {}
    shown = project_event_payload(kind, payload, include_content=False, include_output_preview=True)
    assert shown["text"] == "safe" and "raw_args" not in shown and "thinking" not in shown
    assert project_event_payload(ExecutionDeltaType.ASSISTANT_THINKING_DELTA, payload,
                                 include_content=False, include_output_preview=True) == {}
    delta = ExecutionDelta("e", kind, "whole snapshot", position={"preview_field": "message", "preview_seq": 1})
    assert delta.preview_span == 1
    with pytest.raises(ValueError):
        ExecutionDelta("e", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "bad", position={"preview_field": "message"})


@pytest.mark.asyncio
async def test_real_sdk_output_stream_is_visible_before_completion_and_final_validation_is_unchanged():
    emissions = []

    async def sink(value):
        emissions.append(value)

    async def model(messages, info):
        name = info.output_tools[0].name
        yield {0: DeltaToolCall(name=name, json_args='{"message":"first', tool_call_id="output-1")}
        yield {0: DeltaToolCall(json_args=' second')}
        # This is checked before the JSON is complete or the model returns.
        assert any(value.event_type is ExecutionDeltaType.ASSISTANT_OUTPUT_PREVIEW and value.content == "first second"
                   for value in emissions)
        yield {0: DeltaToolCall(json_args='","internal":"not for preview"}')}

    recorder = AgentRunRecorder(StagingAgentRunStore(), execution_id="execution", agent_run_id="run")
    capability = _event_stream_capability(sink, recorder, 1, output_preview=preview())
    result = await Agent(FunctionModel(stream_function=model), output_type=Answer, capabilities=[capability]).run("answer")
    assert result.output == Answer(message="first second", internal="not for preview")
    values = [item.content for item in emissions if item.event_type is ExecutionDeltaType.ASSISTANT_OUTPUT_PREVIEW]
    assert values == ["first", "first second"]
    assert all("internal" not in value and "not for preview" not in value for value in values)


def test_preview_configuration_is_not_durable_capability_identity():
    plain = ExecutionCapability()
    opted = ExecutionCapability(output_preview_field="message")
    assert CapabilityContribution.from_opaque("capability", plain.id, plain).contract == CapabilityContribution.from_opaque("capability", opted.id, opted).contract


def test_multiple_tool_proposals_clear_previous_preview():
    current = preview()
    current.observe(PartStartEvent(index=0, part=ToolCallPart("output", '{"message":"first"}', "call")))
    assert current.observe(FinalResultEvent(tool_name="output", tool_call_id="call"))[0] == "first"
    assert current.observe(PartStartEvent(index=1, part=ToolCallPart("business", '{"message":"secret"}', "other")))[0] == ""
    assert current.observe(PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta=' '))) is None
    assert current.observe(FinalResultEvent(tool_name=None, tool_call_id=None)) is None


def test_output_preview_never_writes_shared_stream_and_keeps_local_order():
    from linktools.ai.execution.events import LiveExecutionEventBroker

    class Shared:
        calls = []
        def delta(self, *args):
            self.calls.append(args)

    shared = Shared()
    broker = LiveExecutionEventBroker(shared=shared)
    broker.publish(ExecutionDelta("e", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "one"))
    broker.publish(ExecutionDelta("e", ExecutionDeltaType.ASSISTANT_OUTPUT_PREVIEW, "secret snapshot",
                                  position={"preview_field": "message", "preview_seq": 1}))
    broker.publish(ExecutionDelta("e", ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "two"))
    assert [args[2] for args in shared.calls] == ["one", "two"]
    assert [item.start_offset for item in broker._buffers["e"]] == [0, 3, 4]
    assert [item.content for item in broker._buffers["e"]] == ["one", "secret snapshot", "two"]


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_runtime_watch_preview_opt_in_does_not_change_other_output(structured):
    import asyncio
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from ._runtime_test_helpers import _Models, model_features

    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        if structured:
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"message":"hello', tool_call_id="call")}
        else:
            yield "hello"
        entered.set()
        await release.wait()
        if structured:
            yield {0: DeltaToolCall(json_args=' world","internal":"hidden"}')}
        else:
            yield " world"

    features = RuntimeFeatures("preview-test")
    features.agent("default", model="default", allow_tools=(), **({"output_schema": Answer.model_json_schema()} if structured else {}))
    from linktools.ai.capability import TraceCapability
    features.capability(TraceCapability(), scope="runtime")
    features.capability(ExecutionCapability(output_preview_field="message"), scope="runtime")
    async with Persistence.open("preview-test", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(FunctionModel(stream_function=model)), features)) as runtime:
            execution = await runtime.agents.get().start("answer")
            await asyncio.wait_for(entered.wait(), 5)
            seen = []

            async def consume():
                async for item in execution.watch(include_content=False, include_output_preview=True, include_model_interactions=True):
                    event = item.event.event if hasattr(item.event, "event") else None
                    if event is not None and event.event_type == ExecutionDeltaType.ASSISTANT_OUTPUT_PREVIEW:
                        seen.append(event.payload)
                        release.set()

            observer = asyncio.create_task(consume())
            try:
                if not structured:
                    release.set()
                result = await asyncio.wait_for(execution.wait(), 10)
                await asyncio.wait_for(observer, 10)
            finally:
                release.set()
                if not observer.done():
                    observer.cancel()
                await asyncio.gather(observer, return_exceptions=True)
            assert result.result.output == ({"message": "hello world", "internal": "hidden"} if structured else {"text": "hello world"})
            assert bool(seen) is structured
            if structured:
                assert seen[0]["text"] == "hello" and seen[0]["preview_field"] == "message"
                assert all("hidden" not in str(item) for item in seen)


@pytest.mark.asyncio
async def test_stream_attempt_changes_discard_authenticated_parts():
    emissions = []
    recorder = AgentRunRecorder(StagingAgentRunStore(), execution_id="execution", agent_run_id="run")

    async def sink(value):
        emissions.append(value)

    async def events():
        recorder._stream_position = {"model_request_seq": 1, "stream_attempt_seq": 1}
        yield PartStartEvent(index=0, part=ToolCallPart("output", '{"message":"old', "call"))
        yield FinalResultEvent(tool_name="output", tool_call_id="call")
        recorder._stream_position = {"model_request_seq": 1, "stream_attempt_seq": 2}
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta=' stale'))
        yield PartStartEvent(index=0, part=ToolCallPart("output", '{"message":"new', "call-2"))
        yield FinalResultEvent(tool_name="output", tool_call_id="call-2")

    capability = _event_stream_capability(sink, recorder, 1, output_preview=preview())
    await capability.handler(None, events())
    values = [item for item in emissions if item.event_type is ExecutionDeltaType.ASSISTANT_OUTPUT_PREVIEW]
    assert [(item.content, item.position["stream_attempt_seq"], item.position["preview_seq"]) for item in values] == [
        ("old", 1, 1), ("new", 2, 1),
    ]
