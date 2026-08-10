#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent stream event mapping checks."""

from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, ThinkingPart, ThinkingPartDelta

from linktools.ai.agent._runner import _map_event


def test_thinking_parts_are_forwarded_as_thinking_events() -> None:
    assert _map_event(PartStartEvent(index=0, part=ThinkingPart(content="initial"))) == {
        "type": "thinking",
        "text": "initial",
    }
    assert _map_event(PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta="continued"))) == {
        "type": "thinking",
        "text": "continued",
    }
