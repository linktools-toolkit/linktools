#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent stream event mapping and native capability checks."""

import pytest
from pydantic_ai.capabilities import ProcessEventStream, Thinking
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.test import TestModel

from linktools.ai.core import ExecutionDeltaType, ExecutionEventType
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import (
    DurableBoundary,
    LiveDelta,
    _event_stream_capability,
    _map_event,
    _thinking_capability,
    _validate_thinking_model,
)
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


@pytest.mark.asyncio
async def test_event_stream_forwarding_uses_native_capability() -> None:
    emissions: list[object] = []

    async def sink(emission: object) -> None:
        emissions.append(emission)

    capability = _event_stream_capability(sink)  # type: ignore[arg-type]
    assert isinstance(capability, ProcessEventStream)
    assert capability.id == "linktools.ai.event-stream"

    async def events():  # type: ignore[no-untyped-def]
        yield PartStartEvent(index=0, part=TextPart(content="hello"))
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="thinking"))

    await capability.handler(None, events())  # type: ignore[arg-type]
    assert emissions == [
        LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, "hello"),
        LiveDelta(ExecutionDeltaType.ASSISTANT_THINKING_DELTA, "thinking"),
    ]


def test_thinking_uses_native_capability_with_request_model_validation() -> None:
    capability = _thinking_capability("high")
    assert isinstance(capability, Thinking)
    assert capability.id == "linktools.ai.thinking"
    assert capability.effort == "high"
    assert capability.get_model_settings() == {"thinking": "high"}

    supported = TestModel(
        profile={"supports_thinking": True, "thinking_always_enabled": False}
    )
    _validate_thinking_model(supported, "high")

    unsupported = TestModel(
        profile={"supports_thinking": False, "thinking_always_enabled": False}
    )
    with pytest.raises(AIError) as unsupported_error:
        _validate_thinking_model(unsupported, True)
    assert unsupported_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert unsupported_error.value.safe_details == {
        "field": "thinking",
        "reason": "model_not_supported",
    }

    always_on = TestModel(
        profile={"supports_thinking": True, "thinking_always_enabled": True}
    )
    with pytest.raises(AIError) as always_on_error:
        _validate_thinking_model(always_on, False)
    assert always_on_error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert always_on_error.value.safe_details == {
        "field": "thinking",
        "reason": "model_always_enabled",
    }
