#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured file support is shared across admission, replay and SDK requests."""
from linktools.ai.agent._model_resolver import _ModelResolver
from linktools.ai.model import create_model

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
from linktools.ai.model import ModelBinding, validate_model_input, validate_model_inputs
from linktools.ai.model import _openai
from linktools.ai.model._binding import _RetryingModel


def _binding(*mime_types: str, **kwargs: object) -> ModelBinding:
    return create_model(api='openai', model="gpt-file-test", input_types=tuple("mime:" + item for item in mime_types), **kwargs)


@pytest.mark.parametrize("mime", ("application/pdf", "text/plain", "image/png", "audio/wav"))
def test_exact_mime_allowlist_matches_and_default_rejects(mime: str) -> None:
    assert validate_model_input(_binding(mime), mime_type=mime) is None
    with pytest.raises(AIError) as raised:
        validate_model_input(_binding(), mime_type=mime)
    assert raised.value.safe_details == {
        "reason": "model_input_not_supported", "mime_type": mime, "input_kind": "binary",
        "provider": "openai", "model_identity": "openai:gpt-file-test", "api": "openai",
    }


@pytest.mark.parametrize("pattern,mime", (("image/*", "image/png"), ("text/*", "text/csv"), ("application/*", "application/pdf")))
def test_family_allowlist_matches_only_its_mime_family(pattern: str, mime: str) -> None:
    binding = _binding(pattern)
    assert validate_model_input(binding, mime) is None
    with pytest.raises(AIError):
        validate_model_input(binding, "audio/wav")


@pytest.mark.parametrize("values", (None, "image/png", [None], ["mime:*/*"], ["mime:image/p*"], ["text"], ["mime:image/png;foo=bar"]))
def test_invalid_mime_declarations_fail_at_capture(values: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        create_model(api='openai', model="test", input_types=values)


def test_captured_allowlist_is_normalized_immutable_and_restore_uses_current_route() -> None:
    values = ["mime: IMAGE/PNG ", "mime:image/png", "mime:text/*"]
    resolver = _ModelResolver({"default": create_model(api='openai', model="test", input_types=values)})
    values.append("mime:application/pdf")
    original = resolver.resolve("default")
    assert original.contract["input_mime_types"] == ["image/png", "text/*"]
    stored = json.loads(json.dumps(original.contract))
    assert resolver.resolve("default").contract == original.contract
    stored["input_mime_types"].append("application/pdf")
    restored = resolver.resolve("default")
    assert restored.contract == original.contract
    assert restored.contract != stored


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


@pytest.mark.parametrize("metadata", ({"vision": True}, {"vision": False}, {"inputs": {}}, {"label": "custom"}))
def test_current_mime_permissions_are_independent_of_extra_metadata(metadata: dict) -> None:
    class Binding:
        provider = "test"
        model_identity = "test:local"
        contract = {**metadata, "input_mime_types": ["application/pdf"]}

    assert validate_model_input(Binding(), "application/pdf") is None
    with pytest.raises(AIError) as raised:
        validate_model_input(Binding(), "image/png")
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details["reason"] == "model_input_not_supported"


@pytest.mark.parametrize("contract", ({}, {"input_mime_types": None}, {"vision": False}, {"vision": True}, {"inputs": {}}))
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
def test_registered_old_contract_cannot_grant_file_permissions(vision: bool) -> None:
    class HostBinding:
        route_id = "default"
        provider = "openai"
        model_identity = "openai:gpt-file-test"
        contract = {"provider": provider, "model_identity": model_identity, "vision": vision}

        def materialize(self) -> TestModel:
            return TestModel()

    registry = _ModelResolver({"default": HostBinding()})
    restored = registry.resolve("default")
    for mime_type in ("application/pdf", "image/png"):
        with pytest.raises(AIError) as raised:
            validate_model_input(restored, mime_type)
        assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
        assert raised.value.safe_details == {"reason": "model_input_contract_invalid"}


@pytest.mark.parametrize("category,alias_name", (
    ("image", "ImageMediaType"), ("audio", "AudioMediaType"),
    ("document", "DocumentMediaType"), ("video", "VideoMediaType"),
))
def test_input_categories_capture_the_official_sdk_mime_set(category: str, alias_name: str) -> None:
    from typing import get_args
    import pydantic_ai.messages as messages

    binding = create_model(api='openai', model="test", input_types=[category])
    assert binding.contract["input_mime_types"] == sorted(set(get_args(getattr(messages, alias_name))))


def test_categories_explicit_mime_and_duplicates_share_canonical_contract() -> None:
    from typing import get_args
    from pydantic_ai.messages import ImageMediaType, DocumentMediaType

    registry = create_model(api='openai', model="test", input_types=[
        "image", "document", "mime:audio/*", "mime:image/png", "image", "mime:application/x-private",
    ])
    resolver = _ModelResolver({"default": registry})
    expected = sorted(set((*get_args(ImageMediaType), *get_args(DocumentMediaType), "audio/*", "application/x-private")))
    explicit = _ModelResolver({"default": create_model(api='openai', model="test", input_types=["mime:" + mime for mime in expected])})
    binding = resolver.resolve("default")
    assert binding.contract["input_mime_types"] == expected
    assert binding.contract == explicit.resolve("default").contract
    assert resolver.resolve("default").contract == binding.contract
    assert validate_model_input(binding, "audio/wav") is None
    assert validate_model_input(binding, "application/x-private", input_kind="uploaded") is None


@pytest.mark.parametrize("value", ("ImageMediaType", "image/png", "text", "IMAGE", "mime:", "mime:mime:audio/*", 1, None))
def test_authoring_rejects_unknown_tokens_without_coercion(value: object) -> None:
    with pytest.raises(ValueError):
        create_model(api='openai', model="test", input_types=[value])


def test_authoring_rejects_literal_type_objects() -> None:
    from typing import Literal
    from pydantic_ai.messages import ImageMediaType

    for value in (ImageMediaType, Literal["application/x-private"]):
        with pytest.raises(ValueError):
            create_model(api='openai', model="test", input_types=[value])


@pytest.mark.parametrize("value", ("image", "ImageMediaType", "mime:audio/*", "mime:application/pdf"))
def test_captured_contract_does_not_expand_authoring_tokens(value: str) -> None:
    class Binding:
        provider = "test"
        model_identity = "test:local"
        contract = {"input_mime_types": [value]}

    with pytest.raises(AIError) as raised:
        validate_model_input(Binding(), "image/png")
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


def test_captured_mime_list_is_not_reexpanded_after_sdk_category_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Literal
    from linktools.ai.model import _input

    original = create_model(api='openai', model="test", input_types=["image"])
    stored = json.loads(json.dumps(original.contract))
    monkeypatch.setattr(_input, "ImageMediaType", Literal["image/png", "image/x-future"])
    assert original.contract == stored
    assert validate_model_input(original, "image/jpeg") is None
    current = _ModelResolver({"default": create_model(api='openai', model="test", input_types=["image"])})
    restored = current.resolve("default")
    assert restored.contract != stored
    assert validate_model_input(restored, "image/x-future") is None


def test_captured_contract_rejects_sdk_literal_objects() -> None:
    from pydantic_ai.messages import ImageMediaType

    class Binding:
        provider = "test"
        model_identity = "test:local"
        contract = {"input_mime_types": [ImageMediaType]}

    with pytest.raises(AIError) as raised:
        validate_model_input(Binding(), "image/png")
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
