#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider-only reference lowering works without Runtime construction."""

import asyncio
import hashlib
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelHTTPError
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequestRecorder, ModelInputValidator, ReferencedContent, ReferencedInputModel, create_model
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision


_BODY = b"private-reference-input-body"
_SOURCE = CapabilityRef("capability", "files", 1)


def _messages(representation: str = "text") -> list[ModelMessage]:
    reference = ReferencedContent(
        AssetVersionRef(AssetKey("document", "file"), "files", StorageEntryRevision(1),
                        hashlib.sha256(_BODY).hexdigest(), len(_BODY)),
        media_type="application/pdf", source=_SOURCE, representation=representation,
        identifier="attachment-1", source_parameters={"reconstruction": "not-provider-input"},
    )
    return [ModelRequest(parts=[UserPromptPart(["read", reference])])]


class _Provider(TestModel):
    def __init__(self, *, fail: str | None = None) -> None:
        super().__init__(custom_output_text="answer")
        self.inputs: list[list[ModelMessage]] = []
        self.fail = fail
        self.entries = 0
        self.exits = 0
        self.stream_closes = 0

    async def __aenter__(self) -> "_Provider":
        self.entries += 1
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exits += 1

    def _record(self, operation: str, messages: list[ModelMessage]) -> ModelResponse:
        self.inputs.append(messages)
        if self.fail == operation:
            raise ModelHTTPError(400, "fixture", body={"echo": _BODY.decode()})
        return ModelResponse(parts=[TextPart("answer")], model_name="fixture")

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return self._record("request", messages)

    async def count_tokens(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        self._record("count", messages)
        return RequestUsage(input_tokens=3)

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None,
    ) -> ModelResponse:
        return self._record("compact", request_context.messages)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        result = self._record("stream", messages)
        try:
            yield _Response(result, model_request_parameters, self.fail)
        finally:
            self.stream_closes += 1
            if self.fail == "exit":
                raise ModelHTTPError(400, "fixture", body={"echo": _BODY.decode()})


class _Response(CompletedStreamedResponse):
    def __init__(self, response: ModelResponse, parameters: ModelRequestParameters, fail: str | None) -> None:
        super().__init__(response, model_request_parameters=parameters)
        self.failed_attempts = response.failed_attempts
        self.fail = fail

    async def _get_event_iterator(self) -> AsyncIterator[object]:
        async for event in super()._get_event_iterator():
            yield event
        if self.fail == "pull":
            raise ModelHTTPError(400, "fixture", body={"echo": _BODY.decode()})

    async def close_stream(self) -> None:
        if self.fail == "close":
            raise ModelHTTPError(400, "fixture", body={"echo": _BODY.decode()})
        await super().close_stream()


def _binding(monkeypatch: pytest.MonkeyPatch, provider: _Provider, *, allowed: bool = True):
    binding = create_model(api="openai-responses", model="fixture", api_key="unused", max_retries=0,
                           input_types=["mime:application/pdf"] if allowed else [])
    monkeypatch.setattr(type(binding), "_build_model", lambda self, settings: provider)
    return binding


async def _invoke(model: Model, messages: list[ModelMessage], operation: str) -> None:
    parameters = ModelRequestParameters()
    if operation == "request":
        await model.request(messages, None, parameters)
    elif operation == "count":
        await model.count_tokens(messages, None, parameters)
    elif operation == "compact":
        await model.compact_messages(ModelRequestContext(
            model=model, messages=messages, model_settings=None, model_request_parameters=parameters,
        ))
    else:
        async with model.request_stream(messages, None, parameters) as response:
            async for _event in response:
                pass
            if operation == "close":
                await response.close_stream()
            response.get()


@pytest.mark.asyncio
@pytest.mark.parametrize("representation", ("text", "binary"))
@pytest.mark.parametrize("position", ("user", "tool_list", "tool_scalar"))
async def test_materialized_contract_and_shared_owner_cover_every_provider_operation(
    monkeypatch: pytest.MonkeyPatch, representation: str, position: str,
) -> None:
    provider = _Provider()
    binding = _binding(monkeypatch, provider)
    actual = binding.materialize()
    assert isinstance(actual, ModelInputValidator)
    original = _messages(representation)
    content = original[0].parts[0].content[1]
    if position != "user":
        original = [ModelRequest(parts=[ToolReturnPart(
            "read", ["read", content] if position == "tool_list" else content, "call-1",
        )])]
    loaded = []

    async def load(item: ReferencedContent) -> bytes:
        loaded.append(item)
        return _BODY

    # SDK-owned wrappers must not hide the selected model's public contract.
    model = ReferencedInputModel(WrapperModel(actual), load=load)
    async with actual:
        for operation in ("count", "compact", "request", "stream"):
            with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
                await _invoke(model, original, operation)
                assert recorder.body_omission_reason == "referenced_content_lifecycle"
    assert provider.entries == provider.exits == 1
    assert provider.stream_closes == 1
    assert loaded == [content] * 4
    original_content = original[0].parts[0].content
    assert (original_content if position == "tool_scalar" else original_content[1]) is content
    for messages in provider.inputs:
        delivered = messages[0].parts[0].content
        outgoing = delivered if position == "tool_scalar" else delivered[1]
        if representation == "binary":
            assert isinstance(outgoing, BinaryContent)
            assert outgoing.data == _BODY and outgoing.identifier == "attachment-1"
        else:
            assert outgoing == _BODY.decode()
        assert messages is not original and messages[0] is not original[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("request", "count", "compact", "stream", "pull", "exit", "close"))
async def test_provider_echoes_are_omitted_at_all_boundaries(
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    provider = _Provider(fail=operation)
    actual = _binding(monkeypatch, provider).materialize()

    async def load(item: ReferencedContent) -> bytes:
        return _BODY

    with pytest.raises(AIError) as caught:
        await _invoke(ReferencedInputModel(actual, load=load), _messages(), operation)
    error = caught.value
    assert error.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert error.safe_details["status_code"] == 400
    assert error.diagnostics.content_omission_reason == "referenced_content_lifecycle"
    assert _BODY.decode() not in str(error)
    assert _BODY.decode() not in repr(error.diagnostics.to_payload())
    assert _BODY.decode() not in "".join(traceback.format_exception(error))


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", (False, True))
async def test_unknown_actual_contract_fails_before_loader_or_provider(fallback: bool) -> None:
    provider = _Provider()
    actual = FallbackModel(provider, _Provider()) if fallback else provider

    async def load(item: ReferencedContent) -> bytes:
        pytest.fail("unknown contracts must not hydrate")

    with pytest.raises(AIError) as caught:
        await _invoke(ReferencedInputModel(actual, load=load), _messages(), "request")
    assert caught.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert caught.value.safe_details["reason"] == "model_input_contract_unknown"
    assert not provider.inputs


@pytest.mark.asyncio
async def test_actual_contract_rejects_binary_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _Provider()
    actual = _binding(monkeypatch, provider, allowed=False).materialize()

    async def load(item: ReferencedContent) -> bytes:
        pytest.fail("forbidden MIME must not hydrate")

    with pytest.raises(AIError) as caught:
        await _invoke(ReferencedInputModel(actual, load=load), _messages("binary"), "request")
    assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert not provider.inputs


@pytest.mark.asyncio
@pytest.mark.parametrize("error", (ValueError("consumer failure"), asyncio.CancelledError()))
async def test_consumer_error_and_cancellation_are_not_provider_failures(
    monkeypatch: pytest.MonkeyPatch, error: BaseException,
) -> None:
    provider = _Provider()
    actual = _binding(monkeypatch, provider).materialize()

    async def load(item: ReferencedContent) -> bytes:
        return _BODY

    model = ReferencedInputModel(actual, load=load)
    with pytest.raises(type(error)) as caught:
        async with model.request_stream(_messages(), None, ModelRequestParameters()):
            raise error
    assert caught.value is error
    assert provider.stream_closes == 1


@pytest.mark.asyncio
async def test_arbitrary_provider_group_cannot_bypass_reference_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    class Provider(_Provider):
        def _record(self, operation: str, messages: list[ModelMessage]) -> ModelResponse:
            raise FallbackExceptionGroup("echo " + _BODY.decode(), [
                ModelHTTPError(400, "fixture", body={"echo": _BODY.decode()}),
            ])

    actual = _binding(monkeypatch, Provider()).materialize()

    async def load(item: ReferencedContent) -> bytes:
        return _BODY

    with pytest.raises(AIError) as caught:
        await _invoke(ReferencedInputModel(actual, load=load), _messages(), "request")
    assert _BODY.decode() not in "".join(traceback.format_exception(caught.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("request", "count", "compact", "stream"))
async def test_plain_native_requests_preserve_original_provider_error(operation: str) -> None:
    error = ValueError("ordinary provider failure")

    class Provider(_Provider):
        def _record(self, operation: str, messages: list[ModelMessage]) -> ModelResponse:
            raise error

    async def load(item: ReferencedContent) -> bytes:
        pytest.fail("ordinary requests must not load references")

    with pytest.raises(ValueError) as caught:
        await _invoke(ReferencedInputModel(Provider(), load=load),
                      [ModelRequest(parts=[UserPromptPart("hello")])], operation)
    assert caught.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,api", (("openai", "openai"), ("openai_responses", "openai-responses"),
                                     ("anthropic", "anthropic")))
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("rejected", (False, True))
async def test_native_builtin_provider_uses_its_own_contract_and_omits_http_bodies(
    monkeypatch: pytest.MonkeyPatch, kind: str, api: str, streaming: bool, rejected: bool,
) -> None:
    import base64
    import json

    import httpx2

    from .test_model_client_response_capture import _Stream
    from .test_model_protocols import model_binding, response, stream_events

    sent = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        if rejected:
            return httpx2.Response(400, json={"error": {"message": _BODY.decode(), "type": "invalid_request_error"}})
        chunks = stream_events(api) if streaming else [json.dumps(response(api)).encode()]
        return httpx2.Response(200, stream=_Stream(chunks), headers={
            "content-type": "text/event-stream" if streaming else "application/json",
        })

    binding, clients = model_binding(
        monkeypatch, kind, "fixture", handler, input_types=["mime:application/pdf"],
        sdk_profile={"openai_chat_supports_document_input": True} if api == "openai" else {},
    )

    async def load(item: ReferencedContent) -> bytes:
        return _BODY

    actual = binding.materialize()
    model = ReferencedInputModel(actual, load=load)
    original = _messages("binary")
    with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
        async with actual:
            if rejected:
                with pytest.raises(AIError) as caught:
                    await _invoke(model, original, "stream" if streaming else "request")
                assert _BODY.decode() not in "".join(traceback.format_exception(caught.value))
                assert _BODY.decode() not in repr(caught.value.diagnostics.to_payload())
            else:
                await _invoke(model, original, "stream" if streaming else "request")
    assert len(sent) == 1 and base64.b64encode(_BODY) in sent[0]
    assert b"not-provider-input" not in sent[0]
    assert recorder.body_omission_reason == "referenced_content_lifecycle"
    assert recorder.requests[0].body is None
    assert recorder.requests[0].response.body is None
    assert all(client.is_closed for client in clients)
    assert isinstance(original[0].parts[0].content[1], ReferencedContent)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("request", "compact", "stream"))
async def test_custom_model_contract_does_not_allow_failed_attempt_body_retention(operation: str) -> None:
    from datetime import datetime, timedelta, timezone

    from pydantic_ai.messages import ModelRequestAttempt

    attempt = ModelRequestAttempt(
        model_name="first", provider_name="fixture", outcome="error", error=_BODY.decode(),
        timestamp=datetime.now(timezone.utc), duration=timedelta(seconds=1),
        usage=RequestUsage(input_tokens=2),
    )
    response = ModelResponse(parts=[TextPart("answer")], model_name="second", failed_attempts=[attempt])

    class Provider(_Provider):
        def validate_inputs(self, value: object) -> None:
            # This custom model explicitly owns the input contract.
            assert isinstance(value, list)

        def _record(self, operation: str, messages: list[ModelMessage]) -> ModelResponse:
            return response

    async def load(item: ReferencedContent) -> bytes:
        return _BODY

    model = ReferencedInputModel(Provider(), load=load)
    parameters = ModelRequestParameters()
    messages = _messages()
    if operation == "request":
        result = await model.request(messages, None, parameters)
    elif operation == "compact":
        result = await model.compact_messages(ModelRequestContext(
            model=model, messages=messages, model_settings=None, model_request_parameters=parameters,
        ))
    else:
        async with model.request_stream(messages, None, parameters) as streamed:
            async for _event in streamed:
                pass
            result = streamed.get()
            assert streamed.failed_attempts[0].error is None
    assert result.failed_attempts[0].error is None
    assert result.failed_attempts[0].usage.input_tokens == 2
    assert response.failed_attempts == [attempt] and attempt.error == _BODY.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("request", "count", "compact", "stream"))
async def test_builtin_provider_rejects_unhydrated_references_before_io(
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    provider = _Provider()
    actual = _binding(monkeypatch, provider).materialize()
    messages = _messages()
    # Admission is distinct from lowering; hooks may inspect references safely.
    actual.validate_inputs(messages)
    with pytest.raises(AIError) as caught:
        await _invoke(actual, messages, operation)
    assert caught.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
    assert caught.value.safe_details["reason"] == "referenced_input_not_materialized"
    assert not provider.inputs
