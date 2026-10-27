#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recorded model settings survive durable history without changing meaning."""

import json

import pytest
from pydantic_ai import messages
from pydantic_ai.messages import ModelRequest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import decode_model_messages, encode_model_messages


@pytest.mark.parametrize("settings", [{}, {"thinking": "medium"}, {"thinking": None}, {"future": {"values": [1, None, False]}}])
@pytest.mark.parametrize("top_level", [False, True])
def test_model_settings_delta_round_trips_exact_window(settings, top_level):
    part_type = getattr(messages, "ModelSettingsDeltaPart", None)
    if part_type is None:
        pytest.skip("SDK does not emit recorded settings before 2.62")
    request = ModelRequest(parts=[part_type(settings=settings, top_level=top_level)])
    wire = encode_model_messages((request,))
    restored = decode_model_messages(wire)[0]
    assert isinstance(restored.parts[0], part_type)
    assert restored.parts[0].settings == settings
    assert restored.parts[0].top_level is top_level
    assert encode_model_messages((restored,)) == wire


@pytest.mark.parametrize("top_level", [None, 0, "false", []])
def test_model_settings_delta_rejects_corrupt_window_flag(top_level):
    wire = json.loads(encode_model_messages((ModelRequest(parts=[]),)))
    wire[0]["parts"] = [{
        "part_kind": "model-settings-delta", "settings": {"thinking": "medium"},
        "top_level": top_level,
    }]
    with pytest.raises(AIError):
        decode_model_messages(canonical_json_bytes(wire))


def test_old_sdk_rejects_recorded_settings_instead_of_dropping_them(monkeypatch):
    import linktools.ai.model.messages as codec

    wire = json.loads(encode_model_messages((ModelRequest(parts=[]),)))
    wire[0]["parts"] = [{
        "part_kind": "model-settings-delta", "settings": {"thinking": None},
        "top_level": True,
    }]
    monkeypatch.setattr(codec, "ModelSettingsDeltaPart", None)
    with pytest.raises(AIError) as caught:
        decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code == ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.parametrize("settings", [None, [], "medium", {"thinking": 1}, {"thinking": False}, {"thinking": "unknown"}])
def test_model_settings_delta_rejects_invalid_known_settings(settings):
    wire = json.loads(encode_model_messages((ModelRequest(parts=[]),)))
    wire[0]["parts"] = [{"part_kind": "model-settings-delta", "settings": settings, "top_level": False}]
    with pytest.raises(AIError):
        decode_model_messages(canonical_json_bytes(wire))


@pytest.mark.parametrize("request_side", [False, True])
def test_compaction_metadata_round_trips_without_changing_speaker(request_side):
    from pydantic_ai.messages import CompactionPart, ModelResponse

    if not hasattr(CompactionPart, "metadata"):
        pytest.skip("SDK does not carry summary metadata before 2.62")
    part = CompactionPart(content="summary", metadata={"summary": True, "created_at": "2026-10-27"})
    message_type = ModelRequest if request_side else ModelResponse
    message = message_type(parts=[part])
    wire = encode_model_messages((message,))
    restored = decode_model_messages(wire)[0]
    assert type(restored) is message_type
    assert restored.parts == [part]
    assert encode_model_messages((restored,)) == wire
