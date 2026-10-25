#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model error categories preserve retry and diagnostic boundaries."""

from collections.abc import Callable

import httpx2
import pytest
from pydantic_ai import exceptions as sdk_errors
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import ModelRequestParameters

from linktools.ai.errors import ErrorCode, ErrorDiagnostics
from linktools.ai.model import create_model, model_binding_error


@pytest.mark.parametrize(
    ("category", "in_stream", "expected", "retryable"),
    (
        ("ModelQuotaExceededError", False, ErrorCode.MODEL_REQUEST_REJECTED, False),
        ("ModelRequestTimeout", False, ErrorCode.MODEL_TIMEOUT, False),
        ("ModelTimeoutError", False, ErrorCode.MODEL_TIMEOUT, True),
        ("ModelTimeoutError", True, ErrorCode.MODEL_TIMEOUT, False),
        ("ModelConnectionError", False, ErrorCode.MODEL_UNAVAILABLE, True),
        ("ModelConnectionError", True, ErrorCode.MODEL_UNAVAILABLE, False),
    ),
)
def test_public_typed_model_failures_keep_category_and_retry_scope(
    category: str, in_stream: bool, expected: ErrorCode, retryable: bool,
) -> None:
    error_type = getattr(sdk_errors, category, None)
    if error_type is None:
        pytest.skip(f"SDK has no public {category}")
    if category == "ModelRequestTimeout":
        error = error_type("fixture", 30)
    elif category == "ModelQuotaExceededError":
        error = error_type("fixture", "synthetic account quota")
    else:
        error = error_type("fixture", "synthetic transport fault", phase="read", in_stream=in_stream)

    projected = model_binding_error(error)

    assert projected is not None
    assert projected.code is expected
    assert projected.retryable is retryable
    assert projected.safe_details["model_name"] == "fixture"
    if category == "ModelQuotaExceededError":
        assert projected.safe_details["reason"] == "quota_exceeded"
    elif category != "ModelRequestTimeout":
        assert projected.safe_details["phase"] == "read"


@pytest.mark.parametrize("status_code", (400, 429))
def test_typed_quota_category_takes_precedence_over_http_status(status_code: int) -> None:
    quota = getattr(sdk_errors, "ModelQuotaExceededError", None)
    if quota is None:
        pytest.skip("SDK has no public typed quota category")
    error = ModelHTTPError.for_category(quota, status_code=status_code, model_name="fixture")

    projected = model_binding_error(error)

    assert projected is not None
    assert projected.code is ErrorCode.MODEL_REQUEST_REJECTED
    assert projected.retryable is False
    assert projected.safe_details == {
        "model_name": "fixture", "status_code": status_code, "reason": "quota_exceeded",
    }


@pytest.mark.parametrize(
    ("body", "quota"),
    (
        ({"code": "insufficient_quota"}, True),
        ({"error": {"code": "insufficient_quota"}}, True),
        ({"error": {"code": "rate_limit_exceeded"}}, False),
        ({"error": {"message": "insufficient_quota"}}, False),
        ('{"error":{"code":"insufficient_quota"}}', False),
    ),
)
def test_http_quota_uses_only_structured_public_code(body: object, quota: bool) -> None:
    projected = model_binding_error(ModelHTTPError(429, "fixture", body=body))

    assert projected is not None
    assert projected.code is (ErrorCode.MODEL_REQUEST_REJECTED if quota else ErrorCode.MODEL_RATE_LIMITED)
    assert projected.retryable is not quota
    assert (projected.safe_details.get("reason") == "quota_exceeded") is quota


def test_generic_model_message_does_not_invent_quota_category() -> None:
    projected = model_binding_error(ModelAPIError("fixture", "insufficient_quota"))

    assert projected is not None
    assert projected.code is ErrorCode.MODEL_API_ERROR
    assert "reason" not in projected.safe_details


@pytest.mark.parametrize("quota", (False, True))
def test_projection_preserves_explicit_content_omission_diagnostics(quota: bool) -> None:
    body = {"code": "insufficient_quota" if quota else "rate_limit_exceeded", "message": "private body"}
    error = ModelHTTPError(429, "fixture", body=body)
    diagnostics = ErrorDiagnostics.from_exception(error, content_omission_reason="model_body_omitted")

    projected = model_binding_error(error, diagnostics=diagnostics)

    assert projected is not None
    assert projected.diagnostics is diagnostics
    assert projected.diagnostics.content_omission_reason == "model_body_omitted"
    assert "private body" not in str(projected.diagnostics.to_payload())


def _mock_clients(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx2.Request], httpx2.Response],
) -> list[httpx2.AsyncClient]:
    clients: list[httpx2.AsyncClient] = []

    def client(**kwargs: object) -> httpx2.AsyncClient:
        result = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr("linktools.ai.model._openai.DefaultAsyncHttpxClient", client)
    return clients


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
@pytest.mark.parametrize("quota", (False, True))
async def test_public_binding_distinguishes_quota_from_temporary_rate_limit(
    monkeypatch: pytest.MonkeyPatch, streaming: bool, quota: bool,
) -> None:
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(429, json={"error": {
            "code": "insufficient_quota" if quota else "rate_limit_exceeded",
            "message": "synthetic provider failure",
        }})

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(
        protocol="openai", model="fixture", api_key="synthetic",
        base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0,
    )

    async with binding.materialize() as model:
        with pytest.raises(ModelHTTPError) as caught:
            if streaming:
                async with model.request_stream([], None, ModelRequestParameters()) as stream:
                    async for _ in stream:
                        pass
            else:
                await model.request([], None, ModelRequestParameters())

    projected = model_binding_error(caught.value)
    assert projected is not None
    assert projected.code is (ErrorCode.MODEL_REQUEST_REJECTED if quota else ErrorCode.MODEL_RATE_LIMITED)
    assert projected.retryable is not quota
    assert len(sent) == (1 if quota else 3)
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_public_responses_server_error_before_first_event_uses_transport_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=(
            b'event: error\ndata: {"type":"error","code":"server_error",'
            b'"message":"synthetic provider fault","sequence_number":1}\n\n'
        ))

    clients = _mock_clients(monkeypatch, handle)
    binding = create_model(
        protocol="openai-responses", model="fixture", api_key="synthetic",
        base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0,
    )

    async with binding.materialize() as model:
        with pytest.raises(ModelAPIError) as caught:
            async with model.request_stream([], None, ModelRequestParameters()) as stream:
                async for _ in stream:
                    pass

    projected = model_binding_error(caught.value)
    assert projected is not None
    if hasattr(sdk_errors, "ModelServerError"):
        assert caught.value.in_stream is True
        assert projected.code is ErrorCode.MODEL_UNAVAILABLE
    else:
        assert projected.code is ErrorCode.MODEL_API_ERROR
    assert projected.retryable is False
    assert len(sent) == 3
    assert all(client.is_closed for client in clients)
