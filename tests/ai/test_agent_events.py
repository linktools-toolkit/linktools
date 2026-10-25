#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent stream event mapping and native capability checks."""

import pytest
from pydantic_ai.capabilities import ProcessEventStream, Thinking
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
    ToolCallPart,
)

from linktools.ai.core import ExecutionDeltaType, ExecutionEventType
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime._agent_executor import (
    DurableBoundary,
    LiveDelta,
    _event_stream_capability,
    _map_event,
)
from linktools.ai.runtime._event import ExecutionDelta
from linktools.ai.runtime._tool_return_codec import tool_return_content_digest


def test_thinking_parts_are_forwarded_as_thinking_events() -> None:
    assert _map_event(
        PartStartEvent(index=0, part=ThinkingPart(content="initial"))
    ) == LiveDelta(ExecutionDeltaType.ASSISTANT_THINKING_DELTA, "initial")
    assert _map_event(
        PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="continued"))
    ) == LiveDelta(ExecutionDeltaType.ASSISTANT_THINKING_DELTA, "continued")


def test_text_parts_are_forwarded_as_text_events() -> None:
    assert _map_event(
        PartStartEvent(index=1, part=TextPart(content="hello"))
    ) == LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "hello")
    assert _map_event(
        PartDeltaEvent(index=1, delta=TextPartDelta(content_delta="world"))
    ) == LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "world")


def test_tool_result_uses_model_visible_content_digest() -> None:
    content = {"ok": True}
    event = FunctionToolResultEvent(
        part=ToolReturnPart("tool", content, tool_call_id="call-1")
    )

    emission = _map_event(event)

    assert emission == DurableBoundary(
        ExecutionEventType.TOOL_CALL_FINISHED,
        {
            "call_id": "call-1",
            "tool_name": "tool",
            "result_digest": tool_return_content_digest(content),
            "status": "SUCCEEDED",
        },
    )


@pytest.mark.parametrize("part", [
    RetryPromptPart("retry", tool_name="tool", tool_call_id="call-1"),
    ToolReturnPart("tool", "retry", tool_call_id="call-1", outcome="retried"),
])
def test_tool_retry_boundary_carries_error_code(part: object) -> None:
    event = FunctionToolResultEvent(
        part=part,
    )

    assert _map_event(event) == DurableBoundary(
        ExecutionEventType.TOOL_CALL_FINISHED,
        {
            "call_id": "call-1",
            "tool_name": "tool",
            "result_digest": None,
            "status": "FAILED",
            "error_code": ErrorCode.TOOL_RETRY_REQUIRED.value,
        },
    )


@pytest.mark.asyncio
async def test_event_stream_forwarding_uses_native_capability() -> None:
    emissions: list[object] = []

    async def sink(emission: object) -> None:
        emissions.append(emission)

    from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
    from linktools.ai.runtime.state._step_archive import StagingAgentRunStore

    recorder = AgentRunRecorder(StagingAgentRunStore(), execution_id="execution", agent_run_id="run")
    capability = _event_stream_capability(sink, recorder, 1)
    assert isinstance(capability, ProcessEventStream)
    assert capability.id == "linktools.ai.event-stream"

    async def events():  # type: ignore[no-untyped-def]
        yield PartStartEvent(index=0, part=TextPart(content="hello"))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" world"))

    await capability.handler(None, events())  # type: ignore[arg-type]
    assert emissions == [
        LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "hello"),
        LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, " world"),
    ]


@pytest.mark.parametrize("effort", (None, False, True, "minimal", "low", "medium", "high", "xhigh", "max"))
def test_thinking_values_round_trip_to_native_capability(effort) -> None:
    from linktools.ai.spec import AgentSpec, AgentSpecCodec
    spec = AgentSpec("thinking", thinking=effort)
    codec = AgentSpecCodec()
    restored = codec.decode(codec.encode(spec))
    assert restored.thinking == effort
    if effort is not None:
        assert Thinking(effort=restored.thinking).get_model_settings() == {"thinking": effort}


@pytest.mark.parametrize(
    ("event", "event_type"),
    (
        (
            LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "text"),
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA,
        ),
        (
            DurableBoundary(ExecutionEventType.TOOL_CALL_FINISHED, {"status": "FAILED"}),
            ExecutionEventType.TOOL_CALL_FINISHED,
        ),
        (
            ExecutionDelta("execution", ExecutionDeltaType.ASSISTANT_THINKING_DELTA, "thought"),
            ExecutionDeltaType.ASSISTANT_THINKING_DELTA,
        ),
    ),
)
def test_event_emissions_expose_owner_specific_event_type(
    event: LiveDelta | DurableBoundary | ExecutionDelta,
    event_type: ExecutionDeltaType | ExecutionEventType,
) -> None:
    assert event.event_type is event_type


@pytest.mark.asyncio
@pytest.mark.parametrize("visible_part", (TextPart("answer"), ThinkingPart("thought")))
async def test_stream_parts_stay_local_until_the_request_or_run_finishes(
    visible_part: TextPart | ThinkingPart,
) -> None:
    from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
    from linktools.ai.runtime.state._step_archive import StagingAgentRunStore

    staged = StagingAgentRunStore()
    published: list[tuple[object, ...]] = []
    first_call = ToolCallPart("work", {}, tool_call_id="first")
    second_call = ToolCallPart("work", {}, tool_call_id="second")

    async def commit() -> None:
        transcript = staged.staged_transcript("run")
        if transcript is not None:
            assert transcript.pending is None
            published.append(tuple(transcript.messages[-1].parts))

    async def sink(emission: object) -> None:
        assert isinstance(emission, DurableBoundary)
        assert emission.event_type is ExecutionEventType.ASSISTANT_PART_COMPLETED
        assert emission.payload["part_index"] == 1
        assert not published
        assert staged.staged_transcript("run") is None

    recorder = AgentRunRecorder(
        staged, execution_id="execution", agent_run_id="run", history_boundary=commit,
    )
    capability = _event_stream_capability(sink, recorder, 1)

    async def events():  # type: ignore[no-untyped-def]
        yield PartEndEvent(index=0, part=first_call)
        assert not published
        yield PartEndEvent(index=1, part=visible_part)
        yield PartEndEvent(index=2, part=second_call)

    await capability.handler(None, events())  # type: ignore[arg-type]
    assert not published
    await recorder.commit_history_boundary()
    assert not published
    recorder.finish_transcript(interrupted=True)
    await recorder.commit_history_boundary()
    assert published == [(first_call, visible_part, second_call)]
    assert recorder.transcript_messages()[-1].state == "interrupted"
