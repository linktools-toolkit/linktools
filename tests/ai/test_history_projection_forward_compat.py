#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forward-compatibility coverage for public history projections."""

from pydantic_ai.messages import BinaryContent, ImageUrl, UploadedFile, ModelRequest, ToolAvailabilityDeltaPart, UserPromptPart

from linktools.ai.runtime.state._views import project_session_history_message


def test_session_history_preserves_multimodal_user_content() -> None:
    message = ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    "inspect",
                    BinaryContent(b"payload", media_type="application/octet-stream"),
                ]
            )
        ]
    )

    projected = project_session_history_message(message)

    assert len(projected) == 1
    assert projected[0].item_kind == "user"
    assert isinstance(projected[0].content, list)
    assert projected[0].content[0] == "inspect"
    assert projected[0].content[1]["kind"] == "binary"
    assert projected[0].content[1]["mime_type"] == "application/octet-stream"
    assert "media_type" not in projected[0].content[1]


def test_session_history_preserves_new_pydantic_parts_as_open_items() -> None:
    message = ModelRequest(
        parts=[ToolAvailabilityDeltaPart(tools_added=["search"], tool_call_id="call")]
    )

    projected = project_session_history_message(message)

    assert len(projected) == 1
    assert projected[0].item_kind == "tool-availability-delta"
    assert projected[0].tool_call_id == "call"
    assert projected[0].content["part_kind"] == "tool-availability-delta"
    assert projected[0].content["tools_added"] == ["search"]


def test_history_projects_sdk_mime_fields_without_mutating_messages() -> None:
    url = ImageUrl("https://example.com/image", media_type="image/png")
    uploaded = UploadedFile("file-id", "openai", media_type="application/pdf")
    projected = project_session_history_message(ModelRequest(parts=[UserPromptPart([url, uploaded])]))
    assert [item["mime_type"] for item in projected[0].content] == ["image/png", "application/pdf"]
    assert all("media_type" not in item for item in projected[0].content)
    assert url.media_type == "image/png"
    assert uploaded.media_type == "application/pdf"
