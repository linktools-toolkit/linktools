#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consumed attachment markers retain verified, portable request metadata."""
from linktools.ai.model import create_model

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
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from linktools.ai.model import _openai
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._attachment import (
    binary_attachment_text,
    bind_tool_return_attachments,
    is_binary_attachment_text,
)
from linktools.ai.runtime._message import (
    decode_model_messages,
    encode_model_messages,
    project_transient_binary_content,
)

_BODY = b"\xff\x00\x80image"


def _bound_request(
    paths: tuple[str, ...], call_id: str = "read-call", tool: str = "read_file",
) -> ModelRequest:
    files = [
        {
            "path": path,
            **({"input_index": 2 * index + 1} if tool == "read_file" else {}),
            ("mime_type" if tool == "read_file" else "media_type"): "image/png", "size": len(_BODY),
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
        json.loads(item.content)
        for part in request.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, TextContent)
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

    projected = _consume(request)

    assert projected.parts == [request.parts[0]]
    files = projected.parts[0].content["files"]
    assert [file["path"] for file in files] == ["first.png", "second.png"]
    if tool == "read_file":
        assert [file["attachment"] for file in files] == list(reversed(identifiers))
    else:
        assert all(set(file) == {"path", "media_type", "size", "sha256"} for file in files)
    assert identifiers[0] != identifiers[1]
    repeated = _bound_request(("first.png",), "new-call", tool)
    assert repeated.parts[1].content[0].identifier not in identifiers


def test_multiple_tool_returns_in_one_request_keep_their_own_attachment_paths() -> None:
    first = _bound_request(("first.png",), "first-call")
    second = _bound_request(("second.png",), "second-call", "attach_files")
    request = ModelRequest(parts=[
        first.parts[0], second.parts[0], second.parts[1], first.parts[1],
    ])

    projected = _consume(request)

    assert projected.parts == [first.parts[0], second.parts[0]]
    assert [part.content["files"][0]["path"] for part in projected.parts] == [
        "first.png", "second.png",
    ]


def test_conflicting_paths_for_one_occurrence_remain_unknown() -> None:
    first = _bound_request(("first.png",))
    conflicting = _bound_request(("other.png",))
    request = ModelRequest(parts=[first.parts[0], conflicting.parts[0], first.parts[1]])

    assert _metadata(_consume(request)) == [{
        "attachment": first.parts[1].content[0].identifier,
        "size": len(_BODY), "mime_type": "image/png",
    }]


@pytest.mark.parametrize("with_path", [True, False])
@pytest.mark.parametrize("media_key", ["mime_type", "media_type"])
def test_bound_legacy_tool_metadata_is_retained_without_a_second_description(
    with_path: bool, media_key: str,
) -> None:
    request = _bound_request(("evidence.png",))
    request.parts[0].content["files"][0].pop("attachment")
    if media_key == "media_type":
        file = request.parts[0].content["files"][0]
        file["media_type"] = file.pop("mime_type")
    if not with_path:
        request.parts[0].content["files"][0].pop("path")
    before = encode_model_messages([request])

    projected = _consume(request)

    assert projected.parts == [request.parts[0]]
    assert encode_model_messages([request]) == before


def test_generated_metadata_is_retained_once_when_its_binary_is_consumed() -> None:
    binary = BinaryContent(_BODY, media_type="image/png", identifier="image-id")
    metadata = binary_attachment_text(binary, path="images/evidence.png")
    request = ModelRequest(parts=[UserPromptPart(content=["Inspect", metadata, binary])])

    projected = _consume(request)

    assert projected.parts[0].content == ["Inspect", metadata]
    assert _metadata(projected) == [{
        "attachment": "image-id", "path": "images/evidence.png",
        "size": len(_BODY), "mime_type": "image/png",
    }]
    assert is_binary_attachment_text(metadata, binary, path="images/evidence.png")
    assert not is_binary_attachment_text(metadata, binary, path="other.png")
    assert not is_binary_attachment_text(metadata, BinaryContent(b"different", media_type="image/png", identifier="image-id"))
    assert not is_binary_attachment_text(metadata, BinaryContent(_BODY, media_type="image/jpeg", identifier="image-id"))


def test_generated_metadata_requires_the_canonical_mime_field() -> None:
    binary = BinaryContent(_BODY, media_type="image/png", identifier="image-id")
    current = binary_attachment_text(binary, path="image.png")
    value = json.loads(current.content)
    value["media_type"] = value.pop("mime_type")
    legacy = TextContent(json.dumps(value), metadata=current.metadata)
    request = ModelRequest(parts=[UserPromptPart(content=[legacy, binary])])
    before = encode_model_messages([request])

    assert not is_binary_attachment_text(legacy, binary, path="image.png")
    assert _consume(request).parts[0].content == [legacy, binary_attachment_text(binary)]
    assert encode_model_messages([request]) == before
    assert encode_model_messages(decode_model_messages(before)) == before
    value["mime_type"] = "image/png"
    assert not is_binary_attachment_text(TextContent(json.dumps(value), metadata=current.metadata), binary)


def test_arbitrary_json_text_is_not_treated_as_generated_metadata() -> None:
    binary = BinaryContent(_BODY, media_type="image/png")
    text = binary_attachment_text(binary).content
    request = ModelRequest(parts=[UserPromptPart(content=[text, binary])])

    projected = _consume(request)

    assert projected.parts[0].content[0] == text
    assert len(_metadata(projected)) == 1
    assert not is_binary_attachment_text(text, binary)
    assert not is_binary_attachment_text(TextContent(text), binary)


def test_checkpoint_keeps_tool_return_and_pending_index_after_empty_user_part_removal() -> None:
    recorder = AgentRunRecorder(object(), execution_id="execution", agent_run_id="run")
    request = _bound_request(("evidence.png",))
    pending = ModelRequest(parts=[UserPromptPart(content=[BinaryContent(_BODY, media_type="image/png")])])

    context, pending_index = recorder.checkpoint_context(
        [request, ModelResponse(parts=[TextPart("done")])], pending=pending,
    )

    assert context[0].parts == [request.parts[0]]
    assert pending_index == 2
    assert context[pending_index] == pending


def test_projection_is_idempotent_and_preserves_raw_history_and_old_markers() -> None:
    request = _bound_request(("images/evidence.png",))
    request.parts.append(UserPromptPart(content=["[binary content already consumed]"]))
    raw = [request, ModelResponse(parts=[TextPart("done")])]
    encoded = encode_model_messages(raw)
    recovered = decode_model_messages(encoded)

    projected = project_transient_binary_content(recovered)

    assert projected[0].parts[0].content["files"][0]["path"] == "images/evidence.png"
    assert _metadata(projected[0]) == []
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
            "mime": ("mime_type", "application/pdf"),
        }[corruption]
        request.parts[0].content["files"][0][field] = value

    assert _metadata(_consume(request)) == [{
        "attachment": binary.identifier, "size": len(_BODY), "mime_type": "image/png",
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


@pytest.mark.parametrize("path", ["/host/private.png", "../private.png", "C:\\host\\private.png", "https://user:secret@host/file?token=secret"])
def test_tool_paths_must_be_canonical_workspace_paths(path: str) -> None:
    marker = _metadata(_consume(_bound_request((path,))))[0]
    assert "path" not in marker
    assert path not in json.dumps(marker)


def test_authorized_workspace_path_may_contain_a_colon() -> None:
    binary = BinaryContent(_BODY, media_type="image/png")
    metadata = binary_attachment_text(binary, path="report:2026.png")

    assert json.loads(metadata.content)["path"] == "report:2026.png"
    assert is_binary_attachment_text(metadata, binary, path="report:2026.png")
    request = _bound_request(("report:2026.png",))
    assert _consume(request).parts == [request.parts[0]]


@pytest.mark.parametrize("identifier", [None, "evidence.png", "文件", "/host/private.png", "C:\\host\\private.png", "https://user:secret@host/file?token=secret"])
def test_original_binary_keeps_content_identifier_without_claiming_a_path(identifier: str | None) -> None:
    binary = BinaryContent(_BODY, media_type="image/png", identifier=identifier)
    request = ModelRequest(parts=[UserPromptPart(content=[binary])])
    expected = binary.identifier if identifier in {None, "evidence.png", "文件"} else None

    assert _metadata(_consume(request)) == [{
        "attachment": expected, "size": len(_BODY), "mime_type": "image/png",
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
@pytest.mark.parametrize("source", ["tool", "direct"])
async def test_openai_requests_keep_unified_metadata_and_drop_only_consumed_bytes(
    monkeypatch: pytest.MonkeyPatch, source: str,
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
    binding = create_model(protocol='openai',
        model="gpt-file-test", input_types=("mime:image/png",),
        api_key="test-key", base_url="https://file.invalid/v1", max_retries=0,
    )
    request = _bound_request(("evidence.png",))
    returned = request.parts[0].content
    returned["texts"] = [{"path": "note.txt", "input_index": 0, "content": "kept text"}]
    binary = request.parts[1].content[0]
    metadata = binary_attachment_text(binary, path="evidence.png")
    if source == "tool":
        history = [
            ModelRequest(parts=[UserPromptPart("Inspect these files")]),
            ModelResponse(parts=[ToolCallPart(
                "read_file", {"path": ["note.txt", "evidence.png"]}, "read-call",
            )]),
            request,
        ]
    else:
        history = [ModelRequest(parts=[UserPromptPart([
            metadata, binary, "Inspect these files",
        ])])]
    async with binding.materialize() as model:
        response = await model.request(history, None, ModelRequestParameters())
        history.extend([response, ModelRequest(parts=[UserPromptPart("Continue")])])
        await model.request(
            list(project_transient_binary_content(history)), None, ModelRequestParameters(),
        )

    first, second = [outgoing["messages"] for outgoing in requests]
    if source == "tool":
        assert [message["role"] for message in first] == ["user", "assistant", "tool", "user"]
        assert [message["role"] for message in second] == [
            "user", "assistant", "tool", "assistant", "user",
        ]
        assert first[1]["tool_calls"] == second[1]["tool_calls"]
        assert first[1]["tool_calls"][0]["id"] == "read-call"
        assert first[2] == second[2]
        assert second[2]["tool_call_id"] == "read-call"
        assert json.loads(second[2]["content"]) == returned
        assert any(part["type"] == "image_url" for part in first[3]["content"])
        assert returned["files"][0]["attachment"] == binary.identifier
    else:
        assert [message["role"] for message in first] == ["user"]
        assert [message["role"] for message in second] == ["user", "assistant", "user"]
        assert any(part["type"] == "image_url" for part in first[0]["content"])
        assert second[0]["content"] == [
            {"type": "text", "text": metadata.content},
            {"type": "text", "text": "Inspect these files"},
        ]
        assert json.loads(metadata.content) == {
            "attachment": binary.identifier, "path": "evidence.png",
            "size": len(_BODY), "mime_type": "image/png",
        }
    assert binary.data_uri not in json.dumps(second)
    assert "linktools.attachment" not in json.dumps(requests)
    assert "binary content already consumed" not in json.dumps(requests)
    assert "Workspace file path:" not in json.dumps(requests)
