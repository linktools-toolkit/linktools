#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from linktools.ai.agent.checkpoint_io import deserialize_messages, serialize_messages

def _messages():
    return [
        ModelRequest(parts=[UserPromptPart(content="hi")]),
        ModelResponse(parts=[TextPart(content="hello")]),
    ]

def test_serialize_then_deserialize_round_trips_messages():
    restored = deserialize_messages(serialize_messages(_messages()))
    assert len(restored) == 2
    assert isinstance(restored[0], ModelRequest)
    assert isinstance(restored[1], ModelResponse)
    assert restored[0].parts[0].content == "hi"
    assert restored[1].parts[0].content == "hello"

def test_serialize_returns_bytes_not_str():
    assert isinstance(serialize_messages(_messages()), bytes)

def test_deserialize_handles_empty_list_round_trip():
    assert deserialize_messages(serialize_messages([])) == []
