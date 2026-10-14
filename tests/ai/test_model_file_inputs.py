#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured file support is shared across admission, replay and SDK requests."""

import json
from collections.abc import Mapping
from dataclasses import replace

import httpx2
import pytest
from openai import DefaultAsyncHttpxClient
from pydantic_ai.messages import (
    AudioUrl, BinaryContent, DocumentUrl, ImageUrl, ModelRequest,
    ToolReturnPart, UploadedFile, UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import (
    ModelBinding, ModelInputCapability, ModelRegistry,
    decide_model_input, validate_model_input, validate_model_inputs,
)
from linktools.ai.model import _openai
from linktools.ai.model._openai import _RetryingModel


def _binding(*capabilities: ModelInputCapability, **kwargs: object) -> ModelBinding:
    return ModelRegistry.openai(model="gpt-file-test", input_capabilities=capabilities, **kwargs).capture().resolve("default")


@pytest.mark.parametrize("status", ("supported", "unsupported", "unknown"))
@pytest.mark.parametrize("source", ("binary", "document-url", "uploaded"))
def test_declared_file_support_is_identical_for_bytes_urls_and_handles(status: str, source: str) -> None:
    resolver = ModelRegistry.openai(
        model="gpt-file-test", input_capabilities=(ModelInputCapability("application/pdf", status),),
    ).capture()
    original = resolver.resolve("default")
    binding = resolver.restore(json.loads(json.dumps(dict(original.contract))), route_id="default")
    items = {
        "binary": BinaryContent(b"%PDF", media_type="application/pdf"),
        "document-url": DocumentUrl("https://file.invalid/report.pdf"),
        "uploaded": UploadedFile("file-opaque", "openai", media_type="application/pdf"),
    }
    decision = decide_model_input(binding, "application/pdf", input_kind=source, size=4 if source == "binary" else None)
    assert decision.status == status
    assert decision.transport == "file-block"
    assert decision.allowed is (status == "supported")
    assert decision.verified is (status != "unknown")
    if status == "supported":
        assert validate_model_inputs(binding, items[source]) == (decision,)
    else:
        with pytest.raises(AIError) as raised:
            validate_model_inputs(binding, items[source])
        assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert raised.value.retryable is False


@pytest.mark.parametrize("vision", (False, True))
def test_legacy_contract_and_non_image_admission_survive_restore(vision: bool) -> None:
    registry = ModelRegistry.openai(model="gpt-file-test", vision=vision)
    resolver = registry.capture()
    original = resolver.resolve("default")
    stored = json.loads(json.dumps(dict(original.contract)))
    assert stored == {"provider": "openai", "model_identity": "openai:gpt-file-test", "vision": vision, "settings": {}}
    restored = resolver.restore(stored, route_id="default")
    decisions = validate_model_inputs(restored, [
        BinaryContent(b"content", media_type="text/plain"),
        BinaryContent(b"%PDF", media_type="application/pdf"),
        UploadedFile("file-opaque", "openai"),
    ])
    assert [item.transport for item in decisions] == ["text-inline", "file-block", "file-block"]
    assert all(item.allowed and item.status == "unknown" and not item.verified for item in decisions)
    image = decide_model_input(restored, "image/x-unverified")
    assert image.allowed is vision
    assert image.verified is (not vision)
    if vision:
        assert image.status == "unknown"


def test_explicit_empty_declaration_rejects_unknown_without_changing_ordinary_text() -> None:
    binding = _binding()
    assert validate_model_inputs(binding, "plain prompt") == ()
    for mime in ("application/pdf", "text/plain", "image/png"):
        decision = decide_model_input(binding, mime)
        assert not decision.allowed
        assert decision.status == "unknown"
        assert decision.reason == "model_input_unknown"


def test_declarations_are_captured_canonical_semantics_and_restore_rejects_drift() -> None:
    declarations = [ModelInputCapability("text/plain"), ModelInputCapability("application/pdf", max_bytes=20)]
    registry = ModelRegistry.openai(model="gpt-file-test", input_capabilities=declarations)
    registry.register_alias("files", "default")
    captured = registry.capture()
    binding = captured.resolve("files")
    declarations.clear()
    equivalent = ModelRegistry.openai(model="gpt-file-test", base_url="https://different.invalid/v1", input_capabilities=(
        ModelInputCapability("application/pdf", max_bytes=20, transport="file-block"),
        ModelInputCapability(" TEXT/PLAIN ", transport="text-inline"),
    )).capture().resolve("default")
    assert binding.contract == equivalent.contract
    stored = json.loads(json.dumps(dict(binding.contract)))
    assert captured.restore(stored, route_id="files") is binding
    for change in ("status", "transport", "max_bytes", "unknown", "endpoint"):
        drifted = json.loads(json.dumps(stored))
        inputs = drifted["inputs"]
        if change in {"unknown", "endpoint"}:
            inputs[change] = "passthrough" if change == "unknown" else "responses"
        else:
            inputs["capabilities"][0][change] = {"status": "unsupported", "transport": "text-inline", "max_bytes": 21}[change]
        with pytest.raises(AIError) as raised:
            captured.restore(drifted, route_id="files")
        assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE


@pytest.mark.parametrize("mime,transport", (("text/plain", "text-inline"), ("application/pdf", "file-block"), ("image/png", "image"), ("audio/mpeg", "audio")))
def test_passthrough_remains_unknown_and_cannot_override_declared_rejection(mime: str, transport: str) -> None:
    binding = _binding(unknown_inputs="passthrough")
    decision = validate_model_input(binding, mime, size=10)
    assert decision.status == "unknown"
    assert decision.allowed and not decision.verified
    rejected = _binding(ModelInputCapability(mime, "unsupported", transport=transport), unknown_inputs="passthrough")
    with pytest.raises(AIError):
        validate_model_input(rejected, mime, size=10)


@pytest.mark.parametrize("status", ("supported", "unknown"))
def test_byte_limits_also_bound_passthrough_and_unknown_size_references(status: str) -> None:
    binding = _binding(ModelInputCapability("application/pdf", status, 4), unknown_inputs="passthrough")
    validate_model_input(binding, "application/pdf", size=4)
    for source, size, reason in (("binary", 5, "model_input_size_exceeded"), ("document-url", None, "model_input_size_unknown"), ("uploaded", None, "model_input_size_unknown")):
        with pytest.raises(AIError) as raised:
            validate_model_input(binding, "application/pdf", input_kind=source, size=size)
        assert raised.value.safe_details["reason"] == reason


@pytest.mark.parametrize("mime,kind,transport", (
    ("audio/ogg", "binary", "audio"),
    ("audio/ogg", "audio-url", "audio"),
    ("video/mp4", "binary", "file-block"),
    ("video/mp4", "video-url", "file-block"),
    ("image/png", "uploaded", "image"),
    ("application/octet-stream", "binary", "file-block"),
))
def test_declared_support_cannot_override_sdk_encoding_limits(mime: str, kind: str, transport: str) -> None:
    binding = _binding(ModelInputCapability(mime, transport=transport), unknown_inputs="passthrough")
    decision = decide_model_input(binding, mime, input_kind=kind)
    assert not decision.allowed
    assert decision.reason == "input_encoding_not_supported"
    assert decision.status == "unsupported"


@pytest.mark.parametrize("mime", (
    "text/plain", "text/csv", "application/json", "application/problem+json",
    "application/xml", "application/soap+xml", "application/yaml", "application/x-yaml", "application/toml",
))
def test_text_like_files_use_sdk_text_encoding_including_structured_mime_types(mime: str) -> None:
    binding = _binding(ModelInputCapability(mime))
    assert validate_model_inputs(binding, BinaryContent(b"hello", media_type=mime))[0].transport == "text-inline"
    assert validate_model_inputs(binding, DocumentUrl("https://file.invalid/object", media_type=mime))[0].transport == "text-inline"
    with pytest.raises(AIError):
        validate_model_inputs(binding, UploadedFile("file-opaque", "openai", media_type=mime))


def test_explicit_image_support_is_authoritative_without_legacy_vision() -> None:
    binding = _binding(ModelInputCapability("image/png"))
    assert "vision" not in binding.contract
    decision = validate_model_inputs(binding, ImageUrl("https://file.invalid/image.png"))[0]
    assert decision.allowed and decision.verified
    with pytest.raises(AIError):
        validate_model_inputs(binding, ImageUrl("https://file.invalid/image.jpeg"))


@pytest.mark.parametrize("vision", (False, True))
def test_new_declarations_cannot_mix_with_legacy_vision_policy(vision: bool) -> None:
    with pytest.raises(ValueError):
        _binding(ModelInputCapability("image/png"), vision=vision)


@pytest.mark.parametrize("content,reason", (
    (BinaryContent(b"\xff", media_type="text/plain"), "text_input_not_utf8"),
    (UploadedFile("file-opaque", "anthropic", media_type="application/pdf"), "uploaded_file_provider_mismatch"),
    (BinaryContent(b"image", media_type="IMAGE/PNG"), "input_encoding_not_supported"),
))
def test_concrete_inputs_reject_unencodable_bytes_and_wrong_provider(content: object, reason: str) -> None:
    binding = _binding(unknown_inputs="passthrough")
    with pytest.raises(AIError) as raised:
        validate_model_inputs(binding, content)
    assert raised.value.safe_details["reason"] == reason


def test_mixed_request_and_tool_history_uses_same_declarations_without_partial_result() -> None:
    binding = _binding(ModelInputCapability("text/plain"))
    messages = [ModelRequest(parts=[
        UserPromptPart([BinaryContent(b"ok", media_type="text/plain")]),
        ToolReturnPart("read_file", {"nested": [BinaryContent(b"%PDF", media_type="application/pdf")]}),
    ])]
    with pytest.raises(AIError) as raised:
        validate_model_inputs(binding, messages)
    assert raised.value.safe_details["media_type"] == "application/pdf"


@pytest.mark.parametrize("value", (
    None,
    {"endpoint": "responses", "unknown": "reject", "capabilities": []},
    {"endpoint": "openai-chat", "unknown": "maybe", "capabilities": []},
    {"endpoint": "openai-chat", "unknown": "reject", "capabilities": [{"media_type": "application/pdf", "transport": "file-block", "status": "surprise"}]},
))
def test_malformed_captured_input_semantics_fail_closed(value: object) -> None:
    class Binding:
        provider = "openai"
        model_identity = "openai:test"
        vision = True
        contract: Mapping[str, JsonValue] = {"inputs": value}
    with pytest.raises(AIError) as raised:
        validate_model_input(Binding(), "application/pdf")
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_actual_request_revalidates_declarations_before_provider_or_retry(stream: bool) -> None:
    class Provider(TestModel):
        calls = 0

        async def request(self, *args: object, **kwargs: object) -> object:
            self.calls += 1
            return await super().request(*args, **kwargs)

    provider = Provider()
    binding = _binding()
    model = _RetryingModel(provider, 3, 0, input_binding=binding)
    messages = [ModelRequest(parts=[UserPromptPart([BinaryContent(b"%PDF", media_type="application/pdf")])])]
    with pytest.raises(AIError):
        if stream:
            async with model.request_stream(messages, None, ModelRequestParameters()):
                pytest.fail("rejected file reached streaming provider")
        else:
            await model.request(messages, None, ModelRequestParameters())
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_openai_request_uses_reported_text_file_image_and_audio_transports(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "chatcmpl-files", "object": "chat.completion", "created": 1, "model": "gpt-file-test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        })

    def client(**kwargs: object) -> httpx2.AsyncClient:
        return DefaultAsyncHttpxClient(transport=httpx2.MockTransport(respond), **kwargs)

    monkeypatch.setattr(_openai, "DefaultAsyncHttpxClient", client)
    binding = _binding(
        ModelInputCapability("text/csv"),
        ModelInputCapability("application/pdf"),
        ModelInputCapability("image/png"),
        ModelInputCapability("audio/wav"),
        ModelInputCapability("text/csv", transport="file-block"),
        api_key="test-key", base_url="https://file.invalid/v1", max_retries=0,
    )
    contents = [BinaryContent(data, media_type=mime) for mime, data in (
        ("text/csv", b"a,b\n1,2\n"), ("application/pdf", b"%PDF-1.0"), ("image/png", b"image"), ("audio/wav", b"audio"),
    )]
    contents.append(UploadedFile("file-csv", "openai", media_type="text/csv"))
    decisions = validate_model_inputs(binding, contents)
    async with binding.materialize() as model:
        await model.request([ModelRequest(parts=[UserPromptPart(contents)])], None, ModelRequestParameters())
    assert [item.transport for item in decisions] == ["text-inline", "file-block", "image", "audio", "file-block"]
    parts = requests[0]["messages"][0]["content"]
    assert [item["type"] for item in parts] == ["text", "file", "image_url", "input_audio", "file"]
    assert "a,b\n1,2" in parts[0]["text"]
    assert parts[1]["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert parts[4]["file"]["file_id"] == "file-csv"


def test_explicit_alternate_transport_preserves_verified_uploaded_text_support() -> None:
    default = _binding(ModelInputCapability("text/csv"))
    uploaded = UploadedFile("file-opaque", "openai", media_type="text/csv")
    assert decide_model_input(default, "text/csv", input_kind="uploaded").status == "unknown"
    binding = _binding(ModelInputCapability("text/csv"), ModelInputCapability("text/csv", transport="file-block"))
    decisions = validate_model_inputs(binding, [BinaryContent(b"a,b", media_type="text/csv"), uploaded])
    assert [item.transport for item in decisions] == ["text-inline", "file-block"]
    assert all(item.allowed and item.verified for item in decisions)
    denied = _binding(ModelInputCapability("text/csv", "unsupported"), unknown_inputs="passthrough")
    with pytest.raises(AIError) as raised:
        validate_model_inputs(denied, uploaded)
    assert raised.value.safe_details["reason"] == "model_input_not_supported"


def test_custom_legacy_binding_requires_no_new_protocol_and_keeps_non_image_inputs() -> None:
    class Binding:
        provider = "test"
        model_identity = "test:local"
        vision = False
        contract: Mapping[str, JsonValue] = {"provider": "test", "vision": False}
    binding = Binding()
    assert validate_model_inputs(binding, None) == ()
    decision = validate_model_inputs(binding, BinaryContent(b"any", media_type="application/x-custom"))[0]
    assert decision.allowed and decision.status == "unknown" and decision.transport is None
    with pytest.raises(AIError):
        validate_model_inputs(binding, BinaryContent(b"image", media_type="image/png"))


def test_duplicate_rules_and_invalid_limits_are_rejected_before_capture() -> None:
    declaration = ModelInputCapability("application/pdf")
    with pytest.raises(ValueError):
        _binding(declaration, ModelInputCapability("application/pdf", transport="file-block"))
    with pytest.raises(ValueError):
        replace(declaration, max_bytes=True)
    with pytest.raises(ValueError):
        _binding(unknown_inputs="ignore")


def test_model_input_readers_accept_added_metadata_without_reinterpreting_known_fields() -> None:
    payload = {
        "media_type": "application/pdf", "status": "supported", "transport": "file-block",
        "max_bytes": 4, "description": "Display-only evidence", "metadata": {"label": "document"},
    }
    capability = ModelInputCapability.from_payload(payload)
    assert capability == ModelInputCapability("application/pdf", max_bytes=4, transport="file-block")

    class Binding:
        provider = "openai"
        model_identity = "openai:test"
        contract: Mapping[str, JsonValue] = {"inputs": {
            "endpoint": "openai-chat", "unknown": "reject", "capabilities": [payload],
            "description": "Additional provider metadata", "metadata": {"source": "deployment"},
        }}
    decision = validate_model_input(Binding(), "application/pdf", size=4)
    assert decision.allowed and decision.verified
    with pytest.raises(AIError) as raised:
        validate_model_input(Binding(), "application/pdf", size=5)
    assert raised.value.safe_details["reason"] == "model_input_size_exceeded"
    with pytest.raises(ValueError):
        ModelInputCapability.from_payload(dict(payload, transport="future-transport"))


def test_unsupported_mime_needs_no_encodable_transport_and_round_trips() -> None:
    capability = ModelInputCapability("video/mp4", status="unsupported")
    assert ModelInputCapability.from_payload(capability.to_payload()) == capability
    registry = ModelRegistry.openai(model="gpt-file-test", input_capabilities=(capability,), unknown_inputs="passthrough")
    resolver = registry.capture()
    binding = resolver.resolve("default")
    stored = json.loads(json.dumps(dict(binding.contract)))
    assert stored["inputs"]["capabilities"] == [{"media_type": "video/mp4", "status": "unsupported"}]
    restored = resolver.restore(stored, route_id="default")
    for kind in ("binary", "video-url", "uploaded"):
        decision = decide_model_input(restored, "video/mp4", input_kind=kind)
        assert decision.status == "unsupported" and not decision.allowed
    explicit = _binding(ModelInputCapability("video/mp4", status="unsupported", transport="file-block"), unknown_inputs="passthrough")
    assert explicit.contract == binding.contract


@pytest.mark.parametrize("content", (
    DocumentUrl("https://file.invalid/report.pdf", media_type="APPLICATION/PDF"),
    DocumentUrl("https://file.invalid/report.txt", media_type="TEXT/PLAIN"),
    AudioUrl("https://file.invalid/audio.wav", media_type="AUDIO/WAV"),
))
def test_url_validation_rejects_original_mime_that_sdk_cannot_encode(content: DocumentUrl | AudioUrl) -> None:
    binding = _binding(ModelInputCapability(content.media_type.lower()))
    with pytest.raises(AIError) as raised:
        validate_model_inputs(binding, content)
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details["reason"] == "input_encoding_not_supported"
    assert content.media_type == content.media_type.upper()


@pytest.mark.parametrize("mime", ("text/plain", "application/json", "application/problem+json"))
def test_text_url_validation_uses_inline_encoding_without_document_format_lookup(mime: str) -> None:
    binding = _binding(ModelInputCapability(mime))
    content = DocumentUrl("https://file.invalid/object", media_type=mime)
    decision = validate_model_inputs(binding, content)[0]
    assert decision.allowed and decision.verified
    assert decision.transport == "text-inline"
    assert content.media_type == mime
