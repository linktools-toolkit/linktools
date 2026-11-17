#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider error projection for model bindings."""

from collections.abc import Mapping
import re

from pydantic_ai import exceptions as sdk_errors
from pydantic_ai.exceptions import (
    ContentFilterError, FallbackExceptionGroup, ModelAPIError, ModelHTTPError,
    RunCancelled, UnexpectedModelBehavior,
)

from ..core import JsonValue
from ..errors import AIError, ErrorCode, ErrorDiagnostics



# Omission mode retains only fixed protocol vocabulary: even identifier-shaped
# values in a provider error can echo omitted user input. These are diagnostic
# retention rules, not provider acceptance, error classification or retry rules.
_OMISSION_PROVIDER_TYPES = frozenset({
    "invalid_request_error", "authentication_error", "permission_error", "not_found_error",
    "rate_limit_error", "timeout_error", "overloaded_error", "api_error", "billing_error", "server_error",
})
_OMISSION_PROVIDER_CODES = frozenset({
    "invalid_parameter", "invalid_value", "invalid_message", "unsupported_parameter", "unknown_parameter",
    "context_length_exceeded", "invalid_prompt", "invalid_image", "rate_limit_exceeded", "insufficient_quota",
    "server_error", "model_not_found", "invalid_api_key",
})
_OMISSION_PROVIDER_FIELDS = frozenset({
    "model", "messages", "input", "content", "role", "type", "text", "image_url", "file_id", "file_data",
    "max_tokens", "max_output_tokens", "temperature", "top_p", "top_k", "stream", "stop", "stop_sequences",
    "tools", "tool_choice", "function", "name", "parameters", "response_format", "format", "json_schema",
    "schema", "strict", "instructions", "system", "reasoning", "effort", "thinking", "budget_tokens",
})


def model_binding_error(
    error: Exception, *, diagnostics: ErrorDiagnostics | None = None,
) -> "AIError | None":
    """Project provider failures; leave execution and output failures to callers."""
    details: dict[str, JsonValue] = {}
    retryable: bool | None = None
    if _is_quota_error(error):
        code = ErrorCode.MODEL_REQUEST_REJECTED
        details = {"model_name": error.model_name, "reason": "quota_exceeded"}
        if isinstance(error, ModelHTTPError):
            details["status_code"] = error.status_code
    elif _is_context_window_error(error):
        code = ErrorCode.MODEL_REQUEST_REJECTED
        details = {"model_name": error.model_name, "reason": "context_window_exceeded"}
        if isinstance(error, ModelHTTPError):
            details["status_code"] = error.status_code
    elif isinstance(error, sdk_errors.ModelRequestTimeout):
        code = ErrorCode.MODEL_TIMEOUT
        retryable = False
        details = {"model_name": error.model_name}
    elif isinstance(error, sdk_errors.ModelTimeoutError):
        code = ErrorCode.MODEL_TIMEOUT
        details = {"model_name": error.model_name}
        if error.phase is not None:
            details["phase"] = error.phase
    elif isinstance(error, sdk_errors.ModelConnectionError):
        code = ErrorCode.MODEL_UNAVAILABLE
        details = {"model_name": error.model_name}
        if error.phase is not None:
            details["phase"] = error.phase
    elif isinstance(error, sdk_errors.ModelRateLimitError):
        code = ErrorCode.MODEL_RATE_LIMITED
        details = {"model_name": error.model_name}
    elif (isinstance(error, sdk_errors.ModelUnavailableError)
          or isinstance(error, sdk_errors.ModelServerError)):
        code = ErrorCode.MODEL_UNAVAILABLE
        details = {"model_name": error.model_name}
    elif isinstance(error, ModelHTTPError):
        code = _http_error_code(error.status_code)
        details = {
            "model_name": error.model_name,
            "status_code": error.status_code,
        }
    elif isinstance(error, ModelAPIError):
        code = ErrorCode.MODEL_API_ERROR
        details = {"model_name": error.model_name}
    else:
        provider_errors = []
        try:
            from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError
            provider_errors.append((APITimeoutError, APIConnectionError, APIStatusError, APIError))
        except ModuleNotFoundError as missing:
            if missing.name != "openai":
                raise
        try:
            from anthropic import APIConnectionError, APIError, APIStatusError, APITimeoutError
            provider_errors.append((APITimeoutError, APIConnectionError, APIStatusError, APIError))
        except ModuleNotFoundError as missing:
            if missing.name != "anthropic":
                raise
        for timeout, connection, status, api in provider_errors:
            if isinstance(error, timeout):
                code = ErrorCode.MODEL_TIMEOUT
            elif isinstance(error, connection):
                code = ErrorCode.MODEL_UNAVAILABLE
            elif isinstance(error, status):
                code = _http_error_code(error.status_code)
                details = {"status_code": error.status_code}
            elif isinstance(error, api):
                code = ErrorCode.MODEL_API_ERROR
            else:
                continue
            break
        else:
            return None
    if isinstance(error, ModelAPIError):
        if isinstance(error, ModelHTTPError):
            details["status_code"] = error.status_code
        retry_after = error.retry_after
        if isinstance(retry_after, (int, float, str)) and not isinstance(retry_after, bool):
            details["retry_after"] = retry_after
        if error.in_stream:
            retryable = False
    try:
        provider_details = _bounded_provider_error(_provider_error_details(error),
            omit_content=diagnostics is not None and diagnostics.content_omission_reason is not None)
    except Exception:
        # Optional metadata must not replace the provider failure when an SDK
        # extension exposes malformed attributes.
        provider_details = {}
    if provider_details:
        details["provider_error"] = provider_details
    return AIError(
        code,
        retryable=retryable,
        safe_details=details,
        diagnostics=diagnostics if diagnostics is not None else ErrorDiagnostics.from_exception(error),
    )


def model_request_error(
    error: Exception, *, content_omission_reason: str | None,
    code: ErrorCode | None = None, safe_details: Mapping[str, JsonValue] | None = None,
) -> Exception:
    """Omit provider input echoes before failures reach an observer.

    Callers may add their own already-safe request facts and classification.
    Original provider messages, bodies, causes and arbitrary AIError details are
    never copied into the public envelope when input retention is disabled.
    """
    if isinstance(error, RunCancelled):
        return error
    if content_omission_reason is None and not safe_details and code is None:
        return error
    diagnostics = (
        error.diagnostics
        if isinstance(error, AIError) and error.diagnostics is not None
        and error.diagnostics.content_omission_reason == content_omission_reason
        else ErrorDiagnostics.from_exception(error, content_omission_reason=content_omission_reason)
    )
    provider = model_binding_error(error, diagnostics=diagnostics)
    classified = provider if provider is not None else error
    details: dict[str, JsonValue] = dict(safe_details or {})
    if content_omission_reason is not None:
        details["content_omission_reason"] = content_omission_reason
    if isinstance(classified, AIError):
        for key in ("model_name", "phase", "reason", "status_code", "retry_after"):
            if key in classified.safe_details:
                details[key] = classified.safe_details[key]
        provider_details = _bounded_provider_error(classified.safe_details.get("provider_error"),
            omit_content=content_omission_reason is not None)
        if provider_details:
            details["provider_error"] = provider_details
    if code is None:
        if isinstance(classified, AIError):
            code = classified.code
        elif isinstance(classified, ContentFilterError):
            code = ErrorCode.MODEL_CONTENT_FILTERED
        elif isinstance(classified, UnexpectedModelBehavior):
            code = ErrorCode.MODEL_RESPONSE_INVALID
        elif isinstance(classified, FallbackExceptionGroup):
            code = ErrorCode.MODEL_API_ERROR
        else:
            code = ErrorCode.INTERNAL_ERROR
    return AIError(
        code, retryable=classified.retryable if isinstance(classified, AIError) else False,
        safe_details=details, diagnostics=diagnostics,
    )


def _bounded_provider_error(value: object, *, omit_content: bool = False) -> dict[str, JsonValue]:
    """Copy identifier-shaped diagnostics, never free text or an arbitrary body."""
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, JsonValue] = {}
    patterns = {
        "type": (64, r"[A-Za-z0-9_]+"),
        "code": (64, r"[A-Za-z0-9_]+"),
        "param": (128, r"[A-Za-z_][A-Za-z0-9_]*(?:(?:\.[A-Za-z_][A-Za-z0-9_]*)|(?:\[[0-9]+\]))*"),
        "request_id": (128, r"(?:req_[A-Za-z0-9_-]+|[A-Fa-f0-9]{8}(?:-[A-Fa-f0-9]{4}){3}-[A-Fa-f0-9]{12})"),
    }
    for key, (limit, pattern) in patterns.items():
        token = value.get(key)
        if (isinstance(token, str) and 0 < len(token) <= limit
                and re.fullmatch(pattern, token, flags=re.ASCII) is not None):
            if omit_content:
                if key == "type" and token not in _OMISSION_PROVIDER_TYPES:
                    continue
                if key == "code" and token not in _OMISSION_PROVIDER_CODES:
                    continue
                if key == "param" and any(segment not in _OMISSION_PROVIDER_FIELDS
                        for segment in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", token, flags=re.ASCII)):
                    continue
            result[key] = token
    return result


def _provider_error_details(error: Exception) -> dict[str, JsonValue]:
    """Read only public provider identifiers from recognized SDK failures."""
    values: dict[str, object] = {}
    provider = error
    if isinstance(error, ModelAPIError):
        values.update(code=error.provider_error_code, type=error.provider_error_type)
        if isinstance(error, ModelHTTPError) and error.headers is not None:
            values["request_id"] = error.headers.get("x-request-id") or error.headers.get("request-id")
        # PydanticAI documents this immediate cause as the provider SDK error.
        provider = error.__cause__
        if provider is None:
            return _bounded_provider_error(values)
    try:
        from openai import APIError, APIStatusError
    except ModuleNotFoundError as missing:
        if missing.name != "openai":
            raise
    else:
        if isinstance(provider, APIError):
            values["code"] = values.get("code") or provider.code
            values["type"] = values.get("type") or provider.type
            values["param"] = provider.param
            if isinstance(provider, APIStatusError):
                request_id = _bounded_provider_error({"request_id": provider.request_id}).get("request_id")
                if request_id is not None:
                    values["request_id"] = request_id
    try:
        from anthropic import APIStatusError
    except ModuleNotFoundError as missing:
        if missing.name != "anthropic":
            raise
    else:
        if isinstance(provider, APIStatusError):
            values["type"] = values.get("type") or provider.type
            request_id = _bounded_provider_error({"request_id": provider.request_id}).get("request_id")
            if request_id is not None:
                values["request_id"] = request_id
    return _bounded_provider_error(values)


def _is_context_window_error(error: Exception) -> bool:
    """Use the SDK's explicit category, not an ambiguous provider message."""
    return isinstance(error, sdk_errors.ModelContextWindowExceededError)


def _is_quota_error(error: Exception) -> bool:
    """Recognize explicit quota exhaustion, including structured model HTTP errors."""
    if isinstance(error, sdk_errors.ModelQuotaExceededError):
        return True
    if not isinstance(error, ModelHTTPError) or not isinstance(error.body, Mapping):
        return False
    # OpenAI SDKs can expose either the error envelope or its unwrapped value.
    detail = error.body.get("error", error.body)
    return isinstance(detail, Mapping) and detail.get("code") == "insufficient_quota"


def _http_error_code(status_code: int) -> ErrorCode:
    if status_code == 408:
        return ErrorCode.MODEL_TIMEOUT
    if status_code == 429:
        return ErrorCode.MODEL_RATE_LIMITED
    if status_code >= 500:
        return ErrorCode.MODEL_UNAVAILABLE
    if 400 <= status_code < 500:
        return ErrorCode.MODEL_REQUEST_REJECTED
    return ErrorCode.MODEL_API_ERROR


__all__ = ["model_binding_error", "model_request_error"]
