#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool-return persistence preserves JSON and explicit media provenance."""

from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart

from linktools.ai.runtime._message import decode_model_messages, encode_model_messages


def _round_trip(content: object) -> object:
    messages = (
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "tool",
                    content,
                    tool_call_id="call-1",
                )
            ]
        ),
    )
    decoded = decode_model_messages(encode_model_messages(messages))
    part = decoded[0].parts[0]
    assert isinstance(part, ToolReturnPart)
    return part.content


def test_media_shaped_tool_json_round_trips_as_plain_mapping() -> None:
    value = {
        "kind": "binary",
        "media_type": "application/octet-stream",
        "data": "YWJj",
    }

    restored = _round_trip(value)

    assert restored == value
    assert isinstance(restored, dict)


def test_nested_tool_json_and_binary_content_keep_distinct_types() -> None:
    plain = {
        "kind": "binary",
        "media_type": "application/octet-stream",
        "data": "YWJj",
        "business_count": 2,
    }
    binary = BinaryContent(
        data=b"real-binary",
        media_type="application/octet-stream",
    )

    restored = _round_trip({"plain": plain, "binary": binary})

    assert isinstance(restored, dict)
    assert restored["plain"] == plain
    assert isinstance(restored["binary"], BinaryContent)
    assert restored["binary"].data == binary.data
    assert restored["binary"].media_type == binary.media_type
