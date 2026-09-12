#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient BinaryContent projection contracts."""

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from linktools.ai.runtime._harness import HarnessStepStoreAdapter
from linktools.ai.runtime._message import (
    binary_content_usage,
    project_transient_binary_content,
)


def _request(name: str, body: bytes) -> ModelRequest:
    return ModelRequest(
        parts=[
            UserPromptPart(
                content=[
                    f"Workspace file: {name}",
                    BinaryContent(body, media_type="image/png", identifier=name),
                ]
            )
        ]
    )


def test_pending_binary_is_kept_until_complete_model_response() -> None:
    request = _request("a.png", b"image")
    interrupted = ModelResponse(parts=[TextPart("partial")], state="interrupted")

    pending = project_transient_binary_content((request,))
    interrupted_projection = project_transient_binary_content((request, interrupted))

    assert pending == (request,)
    assert interrupted_projection == (request, interrupted)
    assert binary_content_usage(pending) == (1, 5)


def test_consumed_binary_is_removed_without_mutating_raw_transcript() -> None:
    request = _request("a.png", b"image")
    response = ModelResponse(parts=[TextPart("done")])
    raw: tuple[ModelMessage, ...] = (request, response)

    projected = project_transient_binary_content(raw)

    assert projected != raw
    assert binary_content_usage(projected) == (0, 0)
    assert binary_content_usage(raw) == (1, 5)
    projected_request = projected[0]
    assert isinstance(projected_request, ModelRequest)
    content = projected_request.parts[0].content  # type: ignore[attr-defined]
    assert "Workspace file: a.png" in content
    assert any(
        isinstance(item, str) and "binary content already consumed" in item
        for item in content
    )


def test_only_binary_after_latest_complete_response_remains_pending() -> None:
    first = _request("old.png", b"old")
    response = ModelResponse(parts=[TextPart("done")])
    second = _request("new.png", b"new")

    projected = project_transient_binary_content((first, response, second))

    assert binary_content_usage(projected) == (1, 3)
    assert binary_content_usage((first, response, second)) == (2, 6)


def test_snapshot_context_projects_consumed_binary_even_without_compaction() -> None:
    adapter = HarnessStepStoreAdapter(object(), execution_id="execution")  # type: ignore[arg-type]
    request = _request("a.png", b"image")
    response = ModelResponse(parts=[TextPart("done")])
    raw = [request, response]

    context = adapter.snapshot_context_messages(raw)

    assert context is not None
    assert binary_content_usage(context) == (0, 0)
    assert binary_content_usage(raw) == (1, 5)
