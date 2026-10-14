#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consumed attachment markers retain verified, portable request metadata."""

import hashlib
import json

import httpx2
import pytest
from openai import DefaultAsyncHttpxClient
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from linktools.ai.model import ModelInputCapability, ModelRegistry, _openai
from linktools.ai.runtime._attachment import bind_tool_return_attachments
from linktools.ai.runtime._message import (
    decode_model_messages,
    encode_model_messages,
    project_transient_binary_content,
)

_PREFIX = "[binary content already consumed] "
_BODY = b"\xff\x00\x80image"


def _bound_request(
    paths: tuple[str, ...], call_id: str = "read-call", tool: str = "read_file",
) -> ModelRequest:
    files = [
        {
            "path": path, "input_index": 2 * index + 1,
            "media_type": "image/png", "size": len(_BODY),
            "sha256": hashlib.sha256(_BODY).hexdigest(),
        }
        for index, path in enumerate(paths)
    ]
    result = bind_tool_return_attachments(tool, call_id, ToolReturn(
        return_value={"files": files},
        content=[BinaryContent(_BODY, media_type="image/png") for _ in paths],
    ))
    assert isinstance(result, ToolReturn)
    return ModelRequest(parts=[
        ToolReturnPart(tool, result.return_value, call_id),
        UserPromptPart(content=result.content or []),
    ])


def _metadata(request: ModelRequest) -> list[dict[str, object]]:
    return [
        json.loads(item.removeprefix(_PREFIX))
        for part in request.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, str) and item.startswith(_PREFIX)
    ]


def _consume(request: ModelRequest) -> ModelRequest:
    projected = project_transient_binary_content([
        request, ModelResponse(parts=[TextPart("done")]),
    ])
    assert isinstance(projected[0], ModelRequest)
    return projected[0]


@pytest.mark.parametrize("tool", ["read_file", "attach_files"])
def test_identical_bytes_keep_bound_occurrence_paths_without_positional_matching(tool: str) -> None:
    request = _bound_request(("first.png", "second.png"), tool=tool)
    content = request.parts[1].content
    content.reverse()
    identifiers = [item.identifier for item in content]

    metadata = _metadata(_consume(request))

    assert metadata == [
        {"attachment": identifier, "path": path, "size": len(_BODY), "mime": "image/png"}
        for identifier, path in zip(identifiers, ("second.png", "first.png"))
    ]
    assert identifiers[0] != identifiers[1]
    repeated = _metadata(_consume(_bound_request(("first.png",), "new-call", tool)))
    assert repeated[0]["attachment"] not in identifiers


def test_multiple_tool_returns_in_one_request_keep_their_own_attachment_paths() -> None:
    first = _bound_request(("first.png",), "first-call")
    second = _bound_request(("second.png",), "second-call", "attach_files")
    request = ModelRequest(parts=[
        first.parts[0], second.parts[0], second.parts[1], first.parts[1],
    ])

    metadata = _metadata(_consume(request))

    assert [item["path"] for item in metadata] == ["second.png", "first.png"]
    assert [item["attachment"] for item in metadata] == [
        second.parts[1].content[0].identifier, first.parts[1].content[0].identifier,
    ]


def test_conflicting_paths_for_one_occurrence_remain_unknown() -> None:
    first = _bound_request(("first.png",))
    conflicting = _bound_request(("other.png",))
    request = ModelRequest(parts=[first.parts[0], conflicting.parts[0], first.parts[1]])

    assert _metadata(_consume(request)) == [{
        "attachment": first.parts[1].content[0].identifier,
        "size": len(_BODY), "mime": "image/png",
    }]


def test_projection_is_idempotent_and_preserves_raw_history_and_old_markers() -> None:
    request = _bound_request(("images/evidence.png",))
    request.parts.append(UserPromptPart(content=["[binary content already consumed]"]))
    raw = [request, ModelResponse(parts=[TextPart("done")])]
    encoded = encode_model_messages(raw)
    recovered = decode_model_messages(encoded)

    projected = project_transient_binary_content(recovered)

    assert _metadata(projected[0])[0]["path"] == "images/evidence.png"
    assert encode_model_messages(raw) == encode_model_messages(recovered) == encoded
    assert project_transient_binary_content(projected) == projected
    assert decode_model_messages(encode_model_messages(projected)) == projected
    assert projected[0].parts[-1].content == ["[binary content already consumed]"]


@pytest.mark.parametrize("corruption", ["missing", "malformed", "digest", "size", "mime", "unbound"])
def test_missing_or_inconsistent_tool_metadata_does_not_invent_a_path(corruption: str) -> None:
    request = _bound_request(("evidence.png",))
    binary = request.parts[1].content[0]
    if corruption == "missing":
        request.parts.pop(0)
    elif corruption == "malformed":
        request.parts[0].content["files"] = [None]
    elif corruption == "unbound":
        binary = BinaryContent(_BODY, media_type="image/png", identifier="evidence.png")
        request.parts[1].content[0] = binary
    else:
        field, value = {
            "digest": ("sha256", "0" * 64),
            "size": ("size", len(_BODY) + 1),
            "mime": ("media_type", "application/pdf"),
        }[corruption]
        request.parts[0].content["files"][0][field] = value

    assert _metadata(_consume(request)) == [{
        "attachment": binary.identifier, "size": len(_BODY), "mime": "image/png",
    }]


def test_paths_are_not_borrowed_from_other_requests_or_user_labels() -> None:
    request = _bound_request(("private.png",))
    tool_return, prompt = request.parts
    prompt.content.insert(0, 'Workspace file path: "fake.png"')
    history = [
        ModelRequest(parts=[tool_return]), ModelRequest(parts=[prompt]),
        ModelResponse(parts=[TextPart("done")]),
    ]

    projected = project_transient_binary_content(history)

    assert "path" not in _metadata(projected[1])[0]
    assert projected[1].parts[0].content[0] == 'Workspace file path: "fake.png"'


@pytest.mark.parametrize("path", ["/host/private.png", "../private.png", "C:/host/private.png", "https://user:secret@host/file?token=secret"])
def test_tool_paths_must_be_canonical_workspace_paths(path: str) -> None:
    marker = _metadata(_consume(_bound_request((path,))))[0]
    assert "path" not in marker
    assert path not in json.dumps(marker)


@pytest.mark.parametrize("identifier", [None, "evidence.png", "文件", "/host/private.png", "C:\\host\\private.png", "https://user:secret@host/file?token=secret"])
def test_original_binary_keeps_content_identifier_without_claiming_a_path(identifier: str | None) -> None:
    binary = BinaryContent(_BODY, media_type="image/png", identifier=identifier)
    request = ModelRequest(parts=[UserPromptPart(content=[binary])])
    expected = binary.identifier if identifier in {None, "evidence.png", "文件"} else None

    assert _metadata(_consume(request)) == [{
        "attachment": expected, "size": len(_BODY), "mime": "image/png",
    }]


@pytest.mark.parametrize("state", [None, "incomplete", "suspended", "interrupted"])
def test_unfinished_responses_keep_attachment_bytes_and_metadata(state: str | None) -> None:
    history = [_bound_request(("evidence.png",))]
    if state is not None:
        history.append(ModelResponse(parts=[TextPart("pending")], state=state))
    assert project_transient_binary_content(history) == tuple(history)
    assert _metadata(history[0]) == []


def test_url_and_uploaded_file_lifetimes_are_unchanged() -> None:
    url = ImageUrl("https://host/image.png?token=secret")
    uploaded = UploadedFile("file-provider-id", "openai", media_type="image/png")
    request = ModelRequest(parts=[UserPromptPart(content=[
        url, BinaryContent(_BODY, media_type="image/png"), uploaded,
    ])])

    projected = _consume(request)

    assert projected.parts[0].content[0] is url
    assert projected.parts[0].content[2] is uploaded
    assert len(_metadata(projected)) == 1


@pytest.mark.asyncio
async def test_openai_requests_preserve_tool_return_pairing_and_replace_only_consumed_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "chatcmpl-consumed", "object": "chat.completion", "created": 1,
            "model": "gpt-file-test", "choices": [{
                "index": 0, "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }],
        })

    def client(**kwargs: object) -> httpx2.AsyncClient:
        return DefaultAsyncHttpxClient(transport=httpx2.MockTransport(respond), **kwargs)

    monkeypatch.setattr(_openai, "DefaultAsyncHttpxClient", client)
    binding = ModelRegistry.openai(
        model="gpt-file-test", input_capabilities=(ModelInputCapability("image/png"),),
        api_key="test-key", base_url="https://file.invalid/v1", max_retries=0,
    ).capture().resolve("default")
    request = _bound_request(("evidence.png",))
    returned = request.parts[0].content
    returned["texts"] = [{"path": "note.txt", "input_index": 0, "content": "kept text"}]
    binary = request.parts[1].content[0]
    history = [
        ModelRequest(parts=[UserPromptPart("Inspect these files")]),
        ModelResponse(parts=[ToolCallPart(
            "read_file", {"path": ["note.txt", "evidence.png"]}, "read-call",
        )]),
        request,
    ]
    async with binding.materialize() as model:
        response = await model.request(history, None, ModelRequestParameters())
        history.extend([response, ModelRequest(parts=[UserPromptPart("Continue")])])
        await model.request(
            list(project_transient_binary_content(history)), None, ModelRequestParameters(),
        )

    first, second = [outgoing["messages"] for outgoing in requests]
    assert [message["role"] for message in first] == ["user", "assistant", "tool", "user"]
    assert [message["role"] for message in second] == [
        "user", "assistant", "tool", "user", "assistant", "user",
    ]
    assert first[1]["tool_calls"] == second[1]["tool_calls"]
    assert first[1]["tool_calls"][0]["id"] == "read-call"
    assert first[2] == second[2]
    assert second[2]["tool_call_id"] == "read-call"
    assert json.loads(second[2]["content"]) == returned
    assert any(part["type"] == "image_url" for part in first[3]["content"])
    assert len(second[3]["content"]) == 1
    marker = second[3]["content"][0]
    assert marker["type"] == "text"
    assert json.loads(marker["text"].removeprefix(_PREFIX)) == {
        "attachment": binary.identifier, "path": "evidence.png",
        "size": len(_BODY), "mime": "image/png",
    }
    assert binary.data_uri not in json.dumps(second)
