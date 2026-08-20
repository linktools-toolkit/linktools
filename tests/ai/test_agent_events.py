#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent stream event mapping checks."""

from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from linktools.ai.agent._executor import LiveDelta, _map_event
from linktools.ai.core import ExecutionDeltaType


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
