#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent stream event mapping checks."""

from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, ThinkingPart, ThinkingPartDelta

from linktools.ai.agent._executor import _map_event
from linktools.ai.core import ExecutionEventType


def test_thinking_parts_are_forwarded_as_thinking_events() -> None:
    assert _map_event(PartStartEvent(index=0, part=ThinkingPart(content="initial"))) == (
        ExecutionEventType.ASSISTANT_THINKING_DELTA,
        {"text": "initial"},
    )
    assert _map_event(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="continued"))) == (
        ExecutionEventType.ASSISTANT_THINKING_DELTA,
        {"text": "continued"},
    )
