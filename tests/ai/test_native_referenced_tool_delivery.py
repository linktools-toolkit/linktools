#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native tool attachments retain ownership across durable and provider boundaries."""

import asyncio
import base64
import hashlib
import json
import struct
import zlib
from types import SimpleNamespace

import httpx2
import pytest
from pydantic_ai.messages import (
    BinaryContent, ModelMessage, ModelRequest, ModelResponse, TextContent, TextPart,
    ToolCallPart, ToolReturn, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.tools import ToolDefinition

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ReferencedInputsCapability
from linktools.ai.core import service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequestRecorder, ReferencedContent
from linktools.ai.runtime._harness import SummaryProviderModel
from linktools.ai.runtime._message import (
    binary_content_usage, decode_model_messages, encode_model_messages, project_transient_binary_content,
)
from linktools.ai.runtime._tool_return_codec import decode_tool_return_content, encode_tool_return_content
from linktools.ai.runtime._user_content_codec import encode_user_content_item
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

from .test_model_client_response_capture import _Stream
from .test_model_protocols import PROTOCOLS, model_binding, response, stream_events


_SOURCE = CapabilityRef("capability", "native-files", 1)


def _png(color: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\x00" + color)) + chunk(b"IEND", b""))


_IMAGE = _png(b"\x00\x00\xff")
_USER_IMAGE = _png(b"\xff\x00\x00")
_PDF = b"%PDF-1.7\nsynthetic-native-tool-document"
_TEXT = b"synthetic native tool text"


def _reference(body: bytes = _IMAGE, media_type: str = "image/png", representation: str = "binary") -> ReferencedContent:
    return ReferencedContent(
        AssetVersionRef(AssetKey("document", "synthetic-native-" + hashlib.sha256(body).hexdigest()[:12]), "configured", StorageEntryRevision(1),
                        hashlib.sha256(body).hexdigest(), len(body)),
        media_type=media_type, representation=representation, source=_SOURCE,
        source_parameters={"selector": "private-reconstruction-parameter"},
    )


def _context() -> AgentContext[None]:
    return AgentContext(None, service_principal("tenant", "admitted-owner"), "native-files", None, "run", {})


def _messages(reference: ReferencedContent, *, user_image: bool = False) -> list:
    parts = [ToolReturnPart("read_file", [{"name": "synthetic"}, reference], tool_call_id="call-file")]
    if user_image:
        parts.append(UserPromptPart(["An independent user image", BinaryContent(_USER_IMAGE, media_type="image/png")]))
    return [ModelRequest(parts=[UserPromptPart("Inspect the file")]),
            ModelResponse(parts=[ToolCallPart("read_file", {}, tool_call_id="call-file")]),
            ModelRequest(parts=parts)]


@pytest.mark.parametrize("shape", ["direct", "list", "tool_return"])
def test_native_tool_reference_codec_round_trip_without_payload_bytes(shape: str) -> None:
    reference = _reference()
    value = reference if shape == "direct" else [{"name": "synthetic"}, reference]
    if shape == "tool_return":
        value = ToolReturn(return_value=value, content=["Explicit user follow-up"], metadata={"tag": "native"})
    payload = encode_tool_return_content(value)
    restored = decode_tool_return_content(payload)
    if shape == "tool_return":
        assert restored.return_value == value.return_value
        assert restored.content == value.content and restored.metadata == value.metadata
    else:
        assert restored == value
    raw = json.dumps(payload).encode()
    assert _IMAGE not in raw and base64.b64encode(_IMAGE) not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api,model_name", PROTOCOLS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("representation", ["image", "pdf", "text"])
async def test_native_reference_reaches_sdk_owned_tool_encoding_and_omits_raw_capture(
    monkeypatch: pytest.MonkeyPatch, kind: str, api: str, model_name: str, streaming: bool, representation: str,
) -> None:
    body, mime, mode = ((_IMAGE, "image/png", "binary") if representation == "image" else
                        (_PDF, "application/pdf", "binary") if representation == "pdf" else
                        (_TEXT, "text/plain", "text"))
    reference = _reference(body, mime, mode)
    original = _messages(reference, user_image=representation == "image")
    encoded = encode_model_messages(original)
    restored = list(decode_model_messages(encoded))
    loaded, sent = [], []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        assert content == reference
        loaded.append(content)
        return body

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        chunks = stream_events(api) if streaming else [json.dumps(response(api)).encode()]
        return httpx2.Response(200, stream=_Stream(chunks), headers={
            "content-type": "text/event-stream" if streaming else "application/json",
        })

    binding, clients = model_binding(
        monkeypatch, kind, model_name, handler, input_types=["image", "mime:application/pdf"],
        sdk_profile={"openai_chat_supports_document_input": True} if api == "openai" else {},
    )
    model = SummaryProviderModel(binding.materialize(), context=_context(), binding=binding,
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    params = ModelRequestParameters(function_tools=[ToolDefinition(name="read_file", parameters_json_schema={"type": "object"})])
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            if streaming:
                async with model.request_stream(restored, None, params) as stream:
                    async for _event in stream:
                        pass
                    answer = stream.get()
            else:
                answer = await model.request(restored, None, params)
    assert any(isinstance(part, TextPart) and part.content == "answer" for part in answer.parts)
    assert loaded == [reference] and len(sent) == 1
    assert encode_model_messages(original) == encoded == encode_model_messages(restored)
    assert all(client.is_closed for client in clients)
    wire = sent[0]
    serialized = json.dumps(wire)
    assert "private-reconstruction-parameter" not in serialized and "referenced-content" not in serialized
    expected = base64.b64encode(body).decode() if mode == "binary" else body.decode()
    assert expected in serialized
    if api == "anthropic":
        tool = next(part for message in wire["messages"] if isinstance(message["content"], list)
                    for part in message["content"] if part["type"] == "tool_result")
        assert tool["tool_use_id"] == "call-file" and expected in json.dumps(tool["content"])
        if representation == "image":
            assert base64.b64encode(_USER_IMAGE).decode() not in json.dumps(tool)
    elif api == "openai-responses":
        tool = next(item for item in wire["input"] if item.get("type") == "function_call_output")
        assert tool["call_id"] == "call-file"
        assert expected in json.dumps(tool["output"])
        if representation == "image":
            assert base64.b64encode(_USER_IMAGE).decode() not in json.dumps(tool)
    else:
        tool = next(item for item in wire["messages"] if item["role"] == "tool")
        assert tool["tool_call_id"] == "call-file"
        # Chat Completions has no native image tool role: preserve the SDK's user-image fallback.
        if mode == "binary":
            assert expected not in json.dumps(tool)
            assert any(expected in json.dumps(item) for item in wire["messages"] if item["role"] == "user")
        else:
            assert expected in json.dumps(tool)
    attempt, = recorder.requests
    assert attempt.body is attempt.response.body is None
    assert attempt.diagnostics["body_omission_reason"] == "referenced_content_lifecycle"
    assert _IMAGE not in encoded and base64.b64encode(body) not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["denied", "changed", "cancelled"])
async def test_native_reference_failure_never_dispatches_or_changes_history(failure: str) -> None:
    reference = _reference()
    original = _messages(reference)
    encoded = encode_model_messages(original)
    called = []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        if failure == "cancelled":
            raise asyncio.CancelledError
        if failure == "denied":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return b"different"

    async def respond(messages, info):
        called.append(True)
        return ModelResponse(parts=[TextPart("unexpected")])

    binding = SimpleNamespace(provider="test", model_identity="test:native", contract={"input_mime_types": ["image/png"]})
    model = SummaryProviderModel(FunctionModel(respond), context=_context(), binding=binding,
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    with pytest.raises(asyncio.CancelledError if failure == "cancelled" else AIError) as raised:
        await model.request(original, None, ModelRequestParameters())
    if failure != "cancelled":
        assert raised.value.code is (ErrorCode.AUTHORIZATION_DENIED if failure == "denied" else ErrorCode.ASSET_CONTENT_MISMATCH)
    assert not called and encode_model_messages(original) == encoded


def test_only_complete_responses_retire_native_references_and_preserve_tool_identity() -> None:
    reference = _reference()
    pending = _messages(reference)
    assert binary_content_usage(pending) == (1, len(_IMAGE))
    encoded = encode_model_messages(pending)
    assert encode_model_messages(project_transient_binary_content(pending)) == encoded
    completed = [*pending, ModelResponse(parts=[TextPart("done")])]
    projected = project_transient_binary_content(completed)
    tool = projected[-2].parts[0]
    assert isinstance(tool, ToolReturnPart) and tool.tool_call_id == "call-file"
    assert isinstance(tool.content[1], TextContent)
    assert encode_user_content_item(tool.content[1])["metadata"]["referenced_content"]["ref"] == reference.ref.to_payload()
    assert encode_model_messages(pending) == encoded


@pytest.mark.parametrize("value", ["mapping", "nested_list"])
def test_nested_json_references_are_rejected_instead_of_guessed_as_attachments(value: str) -> None:
    reference = _reference()
    content = {"nested": reference} if value == "mapping" else [[reference]]
    with pytest.raises(AIError):
        encode_tool_return_content(ToolReturn(return_value=content))


@pytest.mark.parametrize("state", ["incomplete", "interrupted", "suspended"])
def test_unfinished_native_tool_consumption_retains_reference(state: str) -> None:
    pending = _messages(_reference())
    incomplete = ModelResponse(parts=[TextPart("partial")], state=state)
    original = [*pending, incomplete]
    encoded = encode_model_messages(original)
    assert encode_model_messages(project_transient_binary_content(original)) == encoded
    assert binary_content_usage(original) == (1, len(_IMAGE))


@pytest.mark.parametrize("limit", ["parts", "bytes"])
@pytest.mark.parametrize("representation", ["binary", "text"])
def test_native_parallel_attachments_share_the_pending_budget(limit: str, representation: str) -> None:
    from linktools.ai.runtime._compaction import _validate_pending_binary_content
    from linktools.ai.core import PromptLimits

    body = _IMAGE if representation == "binary" else _TEXT
    reference = _reference(body, "image/png" if representation == "binary" else "text/plain", representation)
    pending = ModelRequest(parts=[
        ToolReturnPart("read_a", [reference], tool_call_id="a"),
        ToolReturnPart("read_b", [reference], tool_call_id="b"),
    ])
    limits = PromptLimits(max_binary_input_parts=1 if limit == "parts" else 2,
                          max_binary_input_bytes=2 * len(body) if limit == "parts" else len(body))
    with pytest.raises(AIError) as error:
        _validate_pending_binary_content([pending], limits)
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert error.value.safe_details["reason"] == "file_request_budget_exceeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api,model_name", PROTOCOLS)
async def test_native_reference_http_retry_keeps_ownership_and_safe_capture(
    monkeypatch: pytest.MonkeyPatch, kind: str, api: str, model_name: str,
) -> None:
    reference = _reference()
    original = _messages(reference)
    encoded = encode_model_messages(original)
    sent, loaded = [], []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        loaded.append(content)
        return _IMAGE

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        if len(sent) == 1:
            return httpx2.Response(429, json={"error": {"type": "rate_limit_error", "message": "retry"}})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_Stream(stream_events(api)))

    binding, clients = model_binding(monkeypatch, kind, model_name, handler, input_types=["image"], max_retries=1)
    model = SummaryProviderModel(binding.materialize(), context=_context(), binding=binding,
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            async with model.request_stream(original, None, ModelRequestParameters()) as stream:
                async for _event in stream:
                    pass
                assert stream.get().finish_reason == "stop"
    assert len(sent) == 2 and sent[0] == sent[1] and loaded == [reference]
    assert all(attempt.body is None and attempt.response.body is None for attempt in recorder.requests)
    assert [attempt.response.status_code for attempt in recorder.requests] == [429, 200]
    assert encode_model_messages(original) == encoded and all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_cancelled_native_reference_stream_preserves_recoverable_input() -> None:
    reference = _reference()
    original = _messages(reference)
    encoded = encode_model_messages(original)
    loaded = []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        loaded.append(content)
        return _IMAGE

    async def stream(messages, info):
        assert isinstance(messages[-1].parts[0].content[1], BinaryContent)
        yield "partial"
        raise asyncio.CancelledError

    model = SummaryProviderModel(FunctionModel(stream_function=stream), context=_context(),
                                 binding=SimpleNamespace(provider="test", model_identity="test:native", contract={"input_mime_types": ["image/png"]}),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
        with pytest.raises(asyncio.CancelledError):
            async with model.request_stream(original, None, ModelRequestParameters()) as response_stream:
                async for _event in response_stream:
                    pass
    assert loaded == [reference]
    assert recorder.body_omission_reason == "referenced_content_lifecycle"
    assert encode_model_messages(original) == encoded


@pytest.mark.asyncio
async def test_multiple_native_tool_images_keep_order_without_absorbing_user_image(monkeypatch: pytest.MonkeyPatch) -> None:
    second_body = _png(b"\x00\xff\x00")
    first, second = _reference(), _reference(second_body)
    original = _messages(first, user_image=True)
    original[-1].parts[0].content.append(second)
    encoded = encode_model_messages(original)
    sent = []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        return _IMAGE if content == first else second_body

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=response("anthropic"))

    binding, clients = model_binding(monkeypatch, "anthropic", "synthetic", handler, input_types=["image"])
    model = SummaryProviderModel(binding.materialize(), context=_context(), binding=binding,
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    async with model:
        await model.request(original, None, ModelRequestParameters())
    content = sent[0]["messages"][-1]["content"]
    tool = next(part for part in content if part["type"] == "tool_result")
    images = [part["source"]["data"] for part in tool["content"] if part["type"] == "image"]
    assert images == [base64.b64encode(body).decode() for body in (_IMAGE, second_body)]
    assert tool["tool_use_id"] == "call-file"
    user_image = next(part for part in content if part["type"] == "image")
    assert user_image["source"]["data"] == base64.b64encode(_USER_IMAGE).decode()
    assert binary_content_usage(original) == (3, len(_IMAGE) + len(second_body) + len(_USER_IMAGE))
    assert encode_model_messages(original) == encoded and all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_native_tool_attachment_profile_rejection_precedes_loading() -> None:
    calls = []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        calls.append("load")
        return _IMAGE

    async def respond(messages, info):
        calls.append("provider")
        return ModelResponse(parts=[TextPart("unexpected")])

    model = SummaryProviderModel(FunctionModel(respond), context=_context(),
                                 binding=SimpleNamespace(provider="test", model_identity="test:native", contract={"input_mime_types": []}),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    with pytest.raises(AIError) as error:
        await model.request(_messages(_reference()), None, ModelRequestParameters())
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert not calls


def test_reference_identity_copy_and_sdk_instrumentation_do_not_expose_descriptor() -> None:
    from copy import copy, deepcopy
    from pydantic_ai.messages import tool_return_ta
    from pydantic_ai.models.instrumented import InstrumentationSettings

    first = _reference()
    other = _reference(_png(b"\x00\xff\x00"))
    assert first != other
    assert copy(first) == deepcopy(first) == first
    native = ToolReturnPart("read_file", [first], tool_call_id="call-file")
    serialized = tool_return_ta.dump_json(first).decode()
    serialized += native.model_response_str()
    serialized += json.dumps(native.otel_message_parts(InstrumentationSettings(include_content=True)))
    assert "private-reconstruction-parameter" not in serialized
    assert first.ref.digest not in serialized and "source_parameters" not in serialized
    assert "native-files" not in serialized
    restored = decode_tool_return_content(encode_tool_return_content(first))
    assert restored == first and restored.source_parameters == first.source_parameters


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api,model_name", PROTOCOLS)
async def test_retired_native_tool_descriptor_is_not_sent_on_later_provider_turn(
    monkeypatch: pytest.MonkeyPatch, kind: str, api: str, model_name: str,
) -> None:
    reference = _reference()
    original = [*_messages(reference), ModelResponse(parts=[TextPart("consumed")]),
                ModelRequest(parts=[UserPromptPart("Continue without reopening the file")])]
    retired = list(project_transient_binary_content(original))
    encoded = encode_model_messages(retired)
    assert b"private-reconstruction-parameter" in encoded
    restored = list(decode_model_messages(encoded))
    sent = []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        pytest.fail("retired references must not be loaded again")

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        return httpx2.Response(200, json=response(api))

    binding, clients = model_binding(monkeypatch, kind, model_name, handler, input_types=["image"])
    model = SummaryProviderModel(binding.materialize(), context=_context(), binding=binding,
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    async with model:
        await model.request(restored, None, ModelRequestParameters())
    assert len(sent) == 1
    assert all(marker not in sent[0] for marker in (
        b"private-reconstruction-parameter", b"source_parameters", reference.ref.digest.encode(),
        base64.b64encode(_IMAGE),
    ))
    assert encode_model_messages(restored) == encoded and all(client.is_closed for client in clients)


def test_native_binary_result_preserves_existing_executable_tool_references() -> None:
    value = ToolReturn(return_value=[BinaryContent(_IMAGE, media_type="image/png")], tools=["existing_tool"])
    restored = decode_tool_return_content(encode_tool_return_content(value))
    assert restored.return_value == value.return_value and restored.tools == ["existing_tool"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api,model_name", PROTOCOLS)
async def test_native_tool_provider_error_cannot_echo_attachment_bytes_into_diagnostics(
    monkeypatch: pytest.MonkeyPatch, kind: str, api: str, model_name: str,
) -> None:
    import traceback

    reference = _reference()
    original = _messages(reference)
    encoded = encode_model_messages(original)
    encoded_image = base64.b64encode(_IMAGE).decode()
    sent = []

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        return _IMAGE

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        return httpx2.Response(400, json={"error": {"type": "invalid_request_error", "message": encoded_image}})

    binding, clients = model_binding(monkeypatch, kind, model_name, handler, input_types=["image"])
    model = SummaryProviderModel(binding.materialize(), context=_context(), binding=binding,
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            with pytest.raises(AIError) as raised:
                await model.request(original, None, ModelRequestParameters())
    assert len(sent) == 1 and encoded_image.encode() in sent[0]
    assert raised.value.code is ErrorCode.MODEL_REQUEST_REJECTED
    rendered = (str(raised.value) + json.dumps(raised.value.safe_details)
                + json.dumps(raised.value.diagnostics.to_payload()) + "".join(traceback.format_exception(raised.value)))
    assert encoded_image not in rendered and "private-reconstruction-parameter" not in rendered
    attempt, = recorder.requests
    assert attempt.body is attempt.response.body is None and attempt.response.status_code == 400
    assert attempt.diagnostics["body_omission_reason"] == "referenced_content_lifecycle"
    assert encode_model_messages(original) == encoded and all(client.is_closed for client in clients)


def test_retired_native_tool_instrumentation_is_safe_after_canonical_restore() -> None:
    from pydantic_ai.messages import tool_return_ta
    from pydantic_ai.models.instrumented import InstrumentationSettings

    reference = _reference()
    retired = project_transient_binary_content([*_messages(reference), ModelResponse(parts=[TextPart("done")])])
    encoded = encode_model_messages(retired)
    assert b"private-reconstruction-parameter" in encoded
    restored = decode_model_messages(encoded)
    tool = restored[-2].parts[0]
    assert encode_user_content_item(tool.content[1])["metadata"]["referenced_content"]["ref"] == reference.ref.to_payload()
    rendered = (tool_return_ta.dump_json(tool.content).decode() + tool.model_response_str()
                + json.dumps(tool.otel_message_parts(InstrumentationSettings(include_content=True))))
    assert all(marker not in rendered for marker in (
        "private-reconstruction-parameter", "source_parameters", "native-files", reference.ref.digest,
    ))
    assert encode_model_messages(restored) == encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["count_tokens", "compact_messages"])
async def test_native_references_are_hydrated_for_counting_and_compaction_without_history_changes(operation: str) -> None:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.usage import RequestUsage

    original = _messages(_reference())
    encoded = encode_model_messages(original)
    calls = []

    class InspectingModel(TestModel):
        async def count_tokens(
            self, messages: list[ModelMessage], model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> RequestUsage:
            assert messages[-1].parts[0].tool_call_id == "call-file"
            assert messages[-1].parts[0].content[1].data == _IMAGE
            calls.append("count_tokens")
            return RequestUsage(input_tokens=7)

        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None,
        ) -> ModelResponse:
            assert request_context.messages[-1].parts[0].content[1].data == _IMAGE
            assert instructions == "Preserve file findings"
            calls.append("compact_messages")
            return ModelResponse(parts=[TextPart("compact")])

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        calls.append("load")
        return _IMAGE

    model = SummaryProviderModel(InspectingModel(), context=_context(),
                                 binding=SimpleNamespace(provider="test", model_identity="test:native", contract={"input_mime_types": ["image/png"]}),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
        if operation == "count_tokens":
            assert (await model.count_tokens(original, None, ModelRequestParameters())).input_tokens == 7
        else:
            request = ModelRequestContext(model=model, messages=original, model_settings=None,
                                          model_request_parameters=ModelRequestParameters())
            assert (await model.compact_messages(request, instructions="Preserve file findings")).parts[0].content == "compact"
    assert calls == ["load", operation]
    assert recorder.body_omission_reason == "referenced_content_lifecycle"
    assert encode_model_messages(original) == encoded


@pytest.mark.asyncio
async def test_native_text_reference_works_with_default_function_model_usage_estimation() -> None:
    original = _messages(_reference(_TEXT, "text/plain", "text"))
    encoded = encode_model_messages(original)

    async def load(ctx: AgentContext[None], content: ReferencedContent) -> bytes:
        return _TEXT

    async def respond(messages, info):
        assert messages[-1].parts[0].content[1] == _TEXT.decode()
        return ModelResponse(parts=[TextPart("done")])

    model = SummaryProviderModel(FunctionModel(respond), context=_context(),
                                 binding=SimpleNamespace(provider="test", model_identity="test:native", contract={"input_mime_types": []}),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    response = await model.request(original, None, ModelRequestParameters())
    assert response.parts[0].content == "done" and response.usage.input_tokens > 0
    assert encode_model_messages(original) == encoded
