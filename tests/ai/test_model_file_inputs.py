#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured file support is shared across admission, replay and SDK requests."""

import json

import httpx2
import pytest
from openai import DefaultAsyncHttpxClient
from pydantic_ai.messages import (
    AudioUrl, BinaryContent, DocumentUrl, ImageUrl, ModelRequest,
    ToolReturnPart, UploadedFile, UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import (
    ModelBinding, ModelRegistry,
    validate_model_input, validate_model_inputs,
)
from linktools.ai.model import _openai
from linktools.ai.model._openai import _RetryingModel


def _binding(*mime_types: str, **kwargs: object) -> ModelBinding:
    return ModelRegistry.openai(model="gpt-file-test", input_mime_types=mime_types, **kwargs).capture().resolve("default")


@pytest.mark.parametrize("mime", ("application/pdf", "text/plain", "image/png", "audio/wav"))
def test_exact_mime_allowlist_matches_and_default_rejects(mime: str) -> None:
    assert validate_model_input(_binding(mime), mime_type=mime) is None
    with pytest.raises(AIError) as raised:
        validate_model_input(_binding(), mime_type=mime)
    assert raised.value.safe_details == {"reason": "model_input_not_supported", "mime_type": mime, "input_kind": "binary"}


@pytest.mark.parametrize("pattern,mime", (("image/*", "image/png"), ("text/*", "text/csv"), ("application/*", "application/pdf")))
def test_family_allowlist_matches_only_its_mime_family(pattern: str, mime: str) -> None:
    binding = _binding(pattern)
    assert validate_model_input(binding, mime) is None
    with pytest.raises(AIError):
        validate_model_input(binding, "audio/wav")


@pytest.mark.parametrize("values", (None, "image/png", [None], ["*/*"], ["image/p*"], ["image"], ["image/png;foo=bar"]))
def test_invalid_mime_declarations_fail_at_capture(values: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ModelRegistry.openai(model="test", input_mime_types=values)


def test_captured_allowlist_is_normalized_immutable_and_restores_strictly() -> None:
    values = [" IMAGE/PNG ", "image/png", "text/*"]
    resolver = ModelRegistry.openai(model="test", input_mime_types=values).capture()
    values.append("application/pdf")
    original = resolver.resolve("default")
    assert original.contract["input_mime_types"] == ["image/png", "text/*"]
    stored = json.loads(json.dumps(original.contract))
    assert resolver.restore(stored, route_id="default").contract == original.contract
    stored["input_mime_types"].append("application/pdf")
    with pytest.raises(AIError) as raised:
        resolver.restore(stored, route_id="default")
    assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE


@pytest.mark.parametrize("content", (
    BinaryContent(b"%PDF", media_type="application/pdf"),
    DocumentUrl("https://file.invalid/report.pdf"),
    UploadedFile("file-id", "openai", media_type="application/pdf"),
))
def test_allowlist_is_shared_by_bytes_urls_and_uploaded_handles(content: object) -> None:
    assert validate_model_inputs(_binding("application/pdf"), content) is None
    with pytest.raises(AIError):
        validate_model_inputs(_binding(), content)


@pytest.mark.parametrize("content,reason", (
    (BinaryContent(b"\xff", media_type="text/plain"), "text_input_not_utf8"),
    (UploadedFile("file-id", "anthropic", media_type="application/pdf"), "uploaded_file_provider_mismatch"),
    (BinaryContent(b"image", media_type="IMAGE/PNG"), "input_encoding_not_supported"),
    (BinaryContent(b"video", media_type="video/mp4"), "input_encoding_not_supported"),
    (DocumentUrl("https://file.invalid/a", media_type="application/x-private"), "input_encoding_not_supported"),
))
def test_allowlist_does_not_bypass_sdk_encoding(content: object, reason: str) -> None:
    with pytest.raises(AIError) as raised:
        validate_model_inputs(_binding(content.media_type.lower()), content)
    assert raised.value.safe_details["reason"] == reason


def test_nested_history_content_is_checked_without_mutation() -> None:
    binary = BinaryContent(b"%PDF", media_type="application/pdf")
    messages = [ModelRequest(parts=[ToolReturnPart("read_file", {"nested": [binary]})])]
    with pytest.raises(AIError):
        validate_model_inputs(_binding("text/plain"), messages)
    assert binary.media_type == "application/pdf"
    assert validate_model_inputs(_binding("application/pdf"), messages) is None


@pytest.mark.parametrize("vision", (False, True))
def test_committed_vision_contract_keeps_original_image_policy(vision: bool) -> None:
    class Binding:
        provider = "test"
        model_identity = "test:local"
        contract = {"vision": vision}
    assert validate_model_input(Binding(), "application/x-private") is None
    if vision:
        assert validate_model_input(Binding(), "image/png") is None
    else:
        with pytest.raises(AIError) as raised:
            validate_model_input(Binding(), "image/png")
        assert raised.value.safe_details == {"reason": "image_input_not_supported"}


@pytest.mark.parametrize("contract", ({}, {"input_mime_types": None}, {"inputs": {}, "vision": True}))
def test_malformed_or_unsupported_model_input_contract_fails_closed(contract: dict) -> None:
    class Binding:
        provider = "test"
        model_identity = "test:local"
    binding = Binding()
    binding.contract = contract
    with pytest.raises(AIError) as raised:
        validate_model_input(binding, "image/png")
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


@pytest.mark.parametrize("name", ("AudioMediaType", "DocumentMediaType", "ImageMediaType", "VideoMediaType"))
def test_model_exports_original_sdk_mime_type_aliases(name: str) -> None:
    import pydantic_ai.messages as sdk_messages
    import linktools.ai.model as model
    assert getattr(model, name) is getattr(sdk_messages, name)


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
        "text/csv",
        "application/pdf",
        "image/png",
        "audio/wav",
        api_key="test-key", base_url="https://file.invalid/v1", max_retries=0,
    )
    contents = [BinaryContent(data, media_type=mime) for mime, data in (
        ("text/csv", b"a,b\n1,2\n"), ("application/pdf", b"%PDF-1.0"), ("image/png", b"image"), ("audio/wav", b"audio"),
    )]
    contents.append(UploadedFile("file-csv", "openai", media_type="text/csv"))
    assert validate_model_inputs(binding, contents) is None
    async with binding.materialize() as model:
        await model.request([ModelRequest(parts=[UserPromptPart(contents)])], None, ModelRequestParameters())
    parts = requests[0]["messages"][0]["content"]
    assert [item["type"] for item in parts] == ["text", "file", "image_url", "input_audio", "file"]
    assert "a,b\n1,2" in parts[0]["text"]
    assert parts[1]["file"]["file_data"].startswith("data:application/pdf;base64,")
    assert parts[4]["file"]["file_id"] == "file-csv"



def test_custom_mime_allowlist_can_use_provider_uploaded_file_encoding() -> None:
    binding = _binding("application/x-private")
    assert validate_model_inputs(binding, UploadedFile("file-id", "openai", media_type="application/x-private")) is None


@pytest.mark.parametrize("vision", (False, True))
def test_committed_openai_vision_payload_restores_only_matching_host_binding(vision: bool) -> None:
    payload = {
        "provider": "openai", "model_identity": "openai:gpt-file-test",
        "vision": vision, "settings": {"max_tokens": 2048},
    }

    class HostBinding:
        route_id = "default"
        provider = "openai"
        model_identity = "openai:gpt-file-test"
        contract = payload

        def materialize(self) -> TestModel:
            return TestModel()

    registry = ModelRegistry()
    registry.register(HostBinding())
    stored = json.loads(json.dumps(payload))
    restored = registry.capture().restore(stored, route_id="default")
    assert restored.contract == stored
    assert validate_model_input(restored, "application/pdf") is None
    if vision:
        assert validate_model_input(restored, "image/png") is None
    else:
        with pytest.raises(AIError) as raised:
            validate_model_input(restored, "image/png")
        assert raised.value.safe_details == {"reason": "image_input_not_supported"}

    current = ModelRegistry.openai(
        model="gpt-file-test", max_tokens=2048,
        input_mime_types=("image/*",) if vision else (),
    ).capture()
    with pytest.raises(AIError) as raised:
        current.restore(stored, route_id="default")
    assert raised.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
