#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider identifiers survive public failure projection without body echoes."""

from pathlib import Path
import builtins

import httpx2
import pytest
from anthropic import APIStatusError as AnthropicStatusError
from openai import APIStatusError as OpenAIStatusError
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from linktools.ai.capability import AgentContext, ExecutionCapability, ReferencedInputsCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.model import ReferencedContent, create_model, model_binding_error, model_request_error
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from .test_referenced_input_failure import _BODY, _REF, _SOURCE


@pytest.mark.parametrize("provider", ("openai", "anthropic"))
@pytest.mark.parametrize("wrapped", (False, True))
def test_provider_sdk_identifiers_are_selected_without_free_text(provider: str, wrapped: bool) -> None:
    body = {"type": "invalid_request_error", "code": "invalid_parameter", "param": "messages[2].content", "message": "private-input-echo"}
    if provider == "anthropic":
        body = {"error": body}
    raw = (OpenAIStatusError if provider == "openai" else AnthropicStatusError)(
        "private-exception-message", response=httpx2.Response(400,
            request=httpx2.Request("POST", "https://private.invalid/?api_key=secret"),
            headers={"x-request-id": "req_fixture-1", "request-id": "req_fixture-1", "authorization": "Bearer secret"}),
        body=body,
    )
    error = raw
    if wrapped:
        error = ModelHTTPError(400, "fixture", body=body,
                               provider_error_code="invalid_parameter" if provider == "openai" else None,
                               provider_error_type="invalid_request_error")
        error.__cause__ = raw
    projected = model_request_error(error, content_omission_reason="referenced_content_lifecycle")
    assert isinstance(projected, AIError)
    assert projected.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert projected.retryable is False
    expected = {"type": "invalid_request_error", "request_id": "req_fixture-1"}
    if provider == "openai":
        expected.update(code="invalid_parameter", param="messages[2].content")
    assert projected.safe_details["provider_error"] == expected
    again = model_request_error(projected, content_omission_reason="referenced_content_lifecycle")
    assert again.safe_details["provider_error"] == expected
    for hidden in ("private-input-echo", "private-exception-message", "private.invalid", "Bearer secret"):
        assert hidden not in repr(projected.safe_details)
        assert hidden not in repr(projected.diagnostics.to_payload())


@pytest.mark.parametrize("value", (None, 42, True, ["invalid_parameter"], {"code": "invalid_parameter"},
    "", "x" * 129, "private input text", "https://private.invalid", "Bearer credential", "code\nsecret", "错误", "sk-proj-credential", "eyJhbGciOiJIUzI1NiJ9.payload.signature"))
def test_invalid_provider_identifiers_are_omitted_not_stringified(value: object) -> None:
    error = ModelHTTPError(400, "fixture", provider_error_code=value, provider_error_type=value,
                           headers={"x-request-id": value})
    projected = model_binding_error(error)
    assert projected is not None
    assert "provider_error" not in projected.safe_details


def test_unstructured_bodies_and_unrelated_exception_attributes_are_not_diagnostics() -> None:
    for body in ({"error": {"code": "invented", "type": "invented", "param": "messages"}},
                 '{"code":"invented"}', "invalid_parameter"):
        projected = model_binding_error(ModelHTTPError(400, "fixture", body=body))
        assert projected is not None
        assert "provider_error" not in projected.safe_details
    error = ValueError("provider_error_code=invalid_parameter")
    error.provider_error_code = "invented"
    projected = model_request_error(error, content_omission_reason="referenced_content_lifecycle")
    assert isinstance(projected, AIError)
    assert projected.code is ErrorCode.INTERNAL_ERROR
    assert "provider_error" not in projected.safe_details



@pytest.mark.parametrize("raw_request_id", (None, "sk-proj-not-a-request-id", "https://private.invalid"))
def test_wrapped_request_id_is_not_lost_to_missing_or_malformed_sdk_value(raw_request_id: object) -> None:
    raw = OpenAIStatusError("private text", response=httpx2.Response(400,
        request=httpx2.Request("POST", "https://offline.invalid")), body={"param": "input[0].content"})
    raw.request_id = raw_request_id
    error = ModelHTTPError(400, "fixture", headers={"x-request-id": "req_valid"})
    error.__cause__ = raw
    projected = model_binding_error(error)
    assert projected is not None
    assert projected.safe_details["provider_error"] == {"param": "input[0].content", "request_id": "req_valid"}


@pytest.mark.parametrize("failure", (AttributeError, TypeError, ValueError, RuntimeError))
def test_malformed_optional_metadata_does_not_replace_the_provider_failure(failure: type[Exception]) -> None:
    class MalformedMetadata(ModelHTTPError):
        def __getattribute__(self, name: str) -> object:
            if name == "provider_error_code":
                raise failure("broken metadata extension")
            return super().__getattribute__(name)

    projected = model_binding_error(MalformedMetadata(400, "fixture"))
    assert projected is not None
    assert projected.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert projected.retryable is False
    assert projected.safe_details == {"model_name": "fixture", "status_code": 400}


def test_provider_metadata_does_not_require_optional_sdk_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def without_optional_sdk(name: str, *args: object, **kwargs: object) -> object:
        if name in {"openai", "anthropic"}:
            raise ModuleNotFoundError(name=name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_optional_sdk)
    error = ModelHTTPError(400, "fixture", provider_error_code="invalid_parameter",
                           headers={"x-request-id": "req_valid"})
    error.__cause__ = ValueError("unrecognized transport cause")
    projected = model_binding_error(error)
    assert projected is not None
    assert projected.safe_details["provider_error"] == {"code": "invalid_parameter", "request_id": "req_valid"}



@pytest.mark.parametrize("echo", ("PRIVATE_INPUT_123", "secret_token"))
@pytest.mark.parametrize("param", ("secret_token", "input[0].PRIVATE_INPUT_123", "messages[2].secret_token"))
@pytest.mark.parametrize("reproject", (False, True))
def test_content_omission_drops_identifier_shaped_echoes(echo: str, param: str, reproject: bool) -> None:
    raw = OpenAIStatusError("private echo", response=httpx2.Response(400,
        request=httpx2.Request("POST", "https://offline.invalid"), headers={"x-request-id": "req_safe"}),
        body={"code": echo, "type": echo, "param": param})
    error = model_binding_error(raw) if reproject else raw
    if reproject:
        assert error.safe_details["provider_error"] == {"code": echo, "type": echo, "param": param, "request_id": "req_safe"}
    projected = model_request_error(error, content_omission_reason="referenced_content_lifecycle")
    assert isinstance(projected, AIError)
    assert projected.safe_details["provider_error"] == {"request_id": "req_safe"}
    assert echo not in repr(projected.safe_details)
    assert param not in repr(projected.safe_details)
    assert echo not in repr(projected.diagnostics.to_payload())
    assert param not in repr(projected.diagnostics.to_payload())


def test_safe_provider_details_survive_stream_failure_without_reclassification() -> None:
    error = ModelAPIError("fixture", "private text", provider_error_code="context_length_exceeded",
                           provider_error_type="invalid_request_error", in_stream=True)
    projected = model_binding_error(error)
    assert projected is not None
    assert projected.code is ErrorCode.MODEL_API_ERROR
    assert projected.retryable is False
    assert projected.safe_details["provider_error"] == {
        "code": "context_length_exceeded", "type": "invalid_request_error",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("rejection", ("invalid_parameter", "invalid_message", "context_length_exceeded"))
@pytest.mark.parametrize("referenced", (False, True))
async def test_provider_rejection_identifiers_survive_history_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rejection: str, referenced: bool,
) -> None:
    calls: list[httpx2.Request] = []
    clients: list[httpx2.AsyncClient] = []
    loads: list[ReferencedContent] = []
    param = "max_output_tokens" if rejection == "invalid_parameter" else "input[0].content"

    def respond(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(400, headers={"x-request-id": "req_offline-diagnostic"}, json={
            "error": {"type": "invalid_request_error", "code": rejection, "param": param,
                      "message": "private-provider-echo " + _BODY.decode()},
        })

    def client(**kwargs: object) -> httpx2.AsyncClient:
        result = httpx2.AsyncClient(transport=httpx2.MockTransport(respond), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr("linktools.ai.model._openai.DefaultAsyncHttpxClient", client)
    features = RuntimeFeatures("provider-diagnostics")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.model("default", create_model(api="openai-responses", model="gpt-4.1", input_types=("document",), api_key="synthetic-secret",
        base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0))
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(TraceCapability(record_http_bodies=False))

    async def load(context: AgentContext[None], content: ReferencedContent) -> bytes:
        loads.append(content)
        return _BODY

    if referenced:
        features.capability(ReferencedInputsCapability(load), id=_SOURCE.id, revision=_SOURCE.revision, scope="agent")
    prompt = (["inspect", ReferencedContent(_REF, media_type="application/pdf", source=_SOURCE)]
              if referenced else "inspect")
    storage = lambda: PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    expected = {"type": "invalid_request_error", "code": rejection, "param": param,
                "request_id": "req_offline-diagnostic"}
    async with Persistence.open("provider-diagnostics", storage=storage()) as persistence, Runtime.open(
        persistence=persistence, features=features,
    ) as runtime:
        execution = await runtime.agents.get().start(prompt)
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.MODEL_REQUEST_REJECTED.value
        assert result.safe_error_details["status_code"] == 400
        assert result.safe_error_details["provider_error"] == expected
        principal, execution_id = runtime.default_principal, execution.execution_id
        async with History.open(persistence=persistence) as history:
            info = await history.inspect_execution(execution_id, principal=principal)
            assert info.safe_error_details == result.safe_error_details
            interactions = await history.model_interactions(execution_id, principal=principal,
                include_content=True, include_client_requests=True)
            if referenced:
                assert _BODY.decode() not in repr(interactions)
                assert "private-provider-echo" not in repr(interactions)
                assert "synthetic-secret" not in repr(interactions)
        if referenced:
            assert result.error_diagnostics.content_omission_reason == "referenced_content_lifecycle"
            assert _BODY.decode() not in repr(result.error_diagnostics.to_payload())
            assert "private-provider-echo" not in repr(result.error_diagnostics.to_payload())
    async with Persistence.open("provider-diagnostics", storage=storage(), read_only=True) as persistence, History.open(
        persistence=persistence,
    ) as history:
        restored = await history.result(execution_id, principal=principal)
        info = await history.inspect_execution(execution_id, principal=principal)
        assert restored.safe_error_details == info.safe_error_details == result.safe_error_details
        assert restored.error_diagnostics == result.error_diagnostics
        assert restored.status is ExecutionStatus.FAILED
        interactions = await history.model_interactions(execution_id, principal=principal,
            include_content=True, include_client_requests=True)
        if referenced:
            assert _BODY.decode() not in repr(interactions)
            assert "private-provider-echo" not in repr(interactions)
            assert "synthetic-secret" not in repr(interactions)
    assert len(calls) == 1
    assert len(loads) == int(referenced)
    assert all(client.is_closed for client in clients)
