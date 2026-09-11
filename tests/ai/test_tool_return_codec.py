#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable deferred tool-result contract tests."""

from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.messages import BinaryContent, ImageUrl
from pydantic_ai.tools import DeferredToolResults

from linktools.ai.runtime._tool_return_codec import (
    decode_tool_return_content,
    encode_tool_return_content,
    rehydrate_deferred_tool_results,
    tool_return_content_digest,
)


def test_tool_return_content_round_trips_nested_multimodal_values() -> None:
    value = {
        "items": [
            BinaryContent(data=b"image-bytes", media_type="image/png"),
            ImageUrl(url="https://example.com/image.png"),
        ],
        "plain": {"kind": "binary", "label": "not-multimodal"},
    }

    encoded = encode_tool_return_content(value)
    restored = decode_tool_return_content(encoded)

    assert isinstance(restored, dict)
    items = restored["items"]
    assert isinstance(items, list)
    assert isinstance(items[0], BinaryContent)
    assert items[0].data == b"image-bytes"
    assert items[0].media_type == "image/png"
    assert isinstance(items[1], ImageUrl)
    assert items[1].url == "https://example.com/image.png"
    assert restored["plain"] == {"kind": "binary", "label": "not-multimodal"}


def test_deferred_rehydrate_preserves_control_results() -> None:
    portable = encode_tool_return_content(
        {"image": BinaryContent(data=b"binary", media_type="image/png")}
    )
    retry = ModelRetry("retry later")
    failed = ToolFailed("failed")
    source = DeferredToolResults(
        calls={"success": portable, "retry": retry, "failed": failed},
        metadata={"success": {"source": "test"}},
    )

    restored = rehydrate_deferred_tool_results(source)

    success = restored.calls["success"]
    assert isinstance(success, dict)
    assert isinstance(success["image"], BinaryContent)
    assert restored.calls["retry"] is retry
    assert restored.calls["failed"] is failed
    assert restored.metadata == {"success": {"source": "test"}}


def test_tool_return_content_digest_is_canonical() -> None:
    left = {
        "b": [BinaryContent(data=b"same", media_type="image/png")],
        "a": 1,
    }
    right = {
        "a": 1,
        "b": [BinaryContent(data=b"same", media_type="image/png")],
    }

    assert tool_return_content_digest(left) == tool_return_content_digest(right)
