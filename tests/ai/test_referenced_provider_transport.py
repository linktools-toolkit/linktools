#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized referenced bytes reach native SDKs without becoming trace bodies."""

import base64
import hashlib
import json
import traceback

import httpx2
import pytest
from pydantic_ai import exceptions as sdk_errors
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ReferencedInputsCapability
from linktools.ai.core import service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequestRecorder, ReferencedContent
from linktools.ai.runtime._harness import SummaryProviderModel
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

from .test_model_client_response_capture import _Stream
from .test_model_protocols import model_binding, response, stream_events


_PROTOCOLS = (
    ("openai", "openai", "/chat/completions"),
    ("openai_responses", "openai-responses", "/responses"),
    ("anthropic", "anthropic", "/messages"),
)
_MODEL = "synthetic-shared-model"
_BODY = b"%PDF-1.7\nsynthetic-referenced-document-not-durable"
_BASE64 = base64.b64encode(_BODY).decode()
_SOURCE = CapabilityRef("capability", "files", 1)
_REASON = "referenced_content_lifecycle"


def _reference(representation: str = "binary") -> ReferencedContent:
    return ReferencedContent(
        AssetVersionRef(
            AssetKey("document", "immutable-file"), "configured", StorageEntryRevision(1),
            hashlib.sha256(_BODY).hexdigest(), len(_BODY),
        ),
        media_type="application/pdf", source=_SOURCE, representation=representation,
        source_parameters={"selection": "reconstruction-only-parameter-never-in-provider-summary"},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api,suffix", _PROTOCOLS)
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("representation", ("binary", "text"))
async def test_referenced_pdf_reaches_explicit_native_protocol_without_retained_http_bodies(
    monkeypatch: pytest.MonkeyPatch, kind: str, api: str, suffix: str, streaming: bool, representation: str,
) -> None:
    sent: list[bytes] = []
    sources: list[_Stream] = []
    content = _reference(representation)
    context = AgentContext(None, service_principal("tenant", "admitted-user"),
                           "provider-test", None, "execution", {})
    loaded = []

    async def authorized_load(ctx: AgentContext[None], item: ReferencedContent) -> bytes:
        assert ctx is context and item is content
        assert item.source == _SOURCE and item.ref.digest == hashlib.sha256(_BODY).hexdigest()
        loaded.append((ctx, item))
        return _BODY

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path.endswith(suffix)
        assert json.loads(request.content)["model"] == _MODEL
        sent.append(request.content)
        source = _Stream(stream_events(api) if streaming else [json.dumps(response(api)).encode()])
        sources.append(source)
        return httpx2.Response(200, stream=source, headers={
            "content-type": "text/event-stream" if streaming else "application/json",
            "request-id" if api == "anthropic" else "x-request-id": "req-referenced-pdf",
        })

    binding, clients = model_binding(
        monkeypatch, kind, _MODEL, handler,
        input_types=[] if representation == "text" else ["mime:application/pdf"],
        sdk_profile={"openai_chat_supports_document_input": True} if api == "openai" and representation == "binary" else {},
    )
    model = SummaryProviderModel(
        binding.materialize(), context=context,
        referenced_inputs={_SOURCE: ReferencedInputsCapability(authorized_load)}, binding=binding,
    )
    original = [ModelRequest(parts=[UserPromptPart(["inspect file", content])])]
    encoded = encode_model_messages(original)
    captures = []
    async with model:
        for messages in (original, [ModelRequest(parts=[UserPromptPart("ordinary follow-up")])]):
            with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
                captures.append(recorder)
                if streaming:
                    async with model.request_stream(messages, None, ModelRequestParameters()) as streamed:
                        async for _event in streamed:
                            pass
                        answer = streamed.get()
                else:
                    answer = await model.request(messages, None, ModelRequestParameters())
                assert any(getattr(part, "content", None) == "answer" for part in answer.parts)

    assert loaded == [(context, content)]
    assert all(client.is_closed for client in clients)
    assert encode_model_messages(original) == encoded
    assert original[0].parts[0].content[1] is content
    assert _BODY not in encoded and _BASE64.encode() not in encoded
    assert len(sent) == len(captures) == 2
    assert b"reconstruction-only-parameter-never-in-provider-summary" not in b"".join(sent)
    outgoing = json.loads(sent[0])
    if representation == "text":
        messages = outgoing["input"] if api == "openai-responses" else outgoing["messages"]
        assert messages[0]["content"][1]["text"] == _BODY.decode()
    elif api == "openai":
        assert outgoing["messages"][0]["content"][1]["file"]["file_data"] == (
            f"data:application/pdf;base64,{_BASE64}"
        )
        assert "input" not in outgoing
    elif api == "openai-responses":
        assert outgoing["input"][0]["content"][1]["file_data"] == f"data:application/pdf;base64,{_BASE64}"
        assert "messages" not in outgoing
    else:
        assert outgoing["messages"][0]["content"][1]["source"] == {
            "type": "base64", "media_type": "application/pdf", "data": _BASE64,
        }
    assert (_BASE64.encode() in sent[0]) is (representation == "binary")
    assert _BASE64.encode() not in sent[1]
    omitted, ordinary = (capture.requests[0] for capture in captures)
    assert captures[0].record_http_bodies is True and captures[0].body_omission_reason == _REASON
    assert omitted.api == ordinary.api == api
    assert omitted.body is omitted.response.body is None and omitted.record_http_bodies is False
    assert omitted.capture_error is omitted.response.capture_error is None
    assert omitted.response.status_code == 200 and omitted.response.complete is False
    assert omitted.diagnostics["body_omission_reason"] == _REASON
    assert omitted.diagnostics["request_id"] == "req-referenced-pdf"
    assert omitted.diagnostics["received_bytes"] == sum(map(len, sources[0].received))
    assert omitted.diagnostics["received_chunks"] == len(sources[0].received)
    assert (omitted.diagnostics["headers_received_ns"] <= omitted.diagnostics["first_chunk_ns"]
            <= omitted.diagnostics["last_chunk_ns"] <= omitted.diagnostics["close_started_ns"]
            <= omitted.diagnostics["close_finished_ns"])
    assert captures[1].body_omission_reason is None and ordinary.record_http_bodies is True
    assert ordinary.body == sent[1] and ordinary.response.body == b"".join(sources[1].received)
    assert ordinary.response.complete is True
    assert "body_omission_reason" not in ordinary.diagnostics
    assert all(source.closes == 1 for source in sources)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api,suffix", _PROTOCOLS)
async def test_native_error_echoes_of_referenced_bytes_are_omitted_from_safe_diagnostics(
    monkeypatch: pytest.MonkeyPatch, kind: str, api: str, suffix: str,
) -> None:
    content = _reference()
    context = AgentContext(None, service_principal("tenant", "admitted-user"),
                           "provider-test", None, "execution", {})
    loaded = []
    sent = []
    error_body = json.dumps({"error": {
        "type": "invalid_request_error", "message": _BODY.decode() + " " + _BASE64,
    }}).encode()
    source = _Stream([error_body])

    async def authorized_load(ctx: AgentContext[None], item: ReferencedContent) -> bytes:
        assert ctx is context and item is content
        loaded.append(item)
        return _BODY

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path.endswith(suffix)
        sent.append(request.content)
        return httpx2.Response(400, headers={"content-type": "application/json"}, stream=source)

    binding, clients = model_binding(
        monkeypatch, kind, _MODEL, handler, input_types=["mime:application/pdf"],
        sdk_profile={"openai_chat_supports_document_input": True} if api == "openai" else {},
    )
    model = SummaryProviderModel(
        binding.materialize(), context=context,
        referenced_inputs={_SOURCE: ReferencedInputsCapability(authorized_load)}, binding=binding,
    )
    original = [ModelRequest(parts=[UserPromptPart(["inspect file", content])])]
    encoded = encode_model_messages(original)
    async with model:
        with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
            with pytest.raises(AIError) as failed:
                await model.request(original, None, ModelRequestParameters())

    assert loaded == [content] and len(sent) == 1 and _BASE64.encode() in sent[0]
    assert encode_model_messages(original) == encoded
    assert _BODY not in encoded and _BASE64.encode() not in encoded
    attempt, = recorder.requests
    assert attempt.api == api and attempt.body is attempt.response.body is None
    assert attempt.capture_error is attempt.response.capture_error is None
    assert attempt.response.status_code == 400 and attempt.record_http_bodies is False
    assert attempt.diagnostics["body_omission_reason"] == _REASON
    assert attempt.diagnostics["received_bytes"] == len(error_body)
    assert attempt.diagnostics["read_outcome"] == "eof"
    assert "close_finished_ns" in attempt.diagnostics
    error = failed.value
    assert error.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert error.safe_details == {
        "content_omission_reason": _REASON, "status_code": 400, "model_name": _MODEL,
    }
    assert error.diagnostics.content_omission_reason == _REASON
    assert error.diagnostics.exception_message == f"[content omitted: {_REASON}]"
    rejection_type = getattr(sdk_errors, "ModelRequestRejectedError", sdk_errors.ModelHTTPError)
    assert error.diagnostics.exception_type == rejection_type.__name__
    assert error.diagnostics.exception_chain[0]["exception_type"] == rejection_type.__name__
    assert f"root: {rejection_type.__name__}:" in error.diagnostics.traceback
    assert "in request\n" in error.diagnostics.traceback
    rendered = json.dumps(error.diagnostics.to_payload()) + str(error) + "".join(traceback.format_exception(error))
    assert _BODY.decode() not in rendered and _BASE64 not in rendered
    assert source.closes == 1 and all(client.is_closed for client in clients)
