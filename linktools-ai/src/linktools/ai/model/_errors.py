#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider error projection for model bindings."""

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from ..core import JsonValue
from ..errors import AIError, ErrorCode, ErrorDiagnostics


def model_binding_error(
    error: Exception, *, diagnostics: ErrorDiagnostics | None = None,
) -> "AIError | None":
    """Project provider failures; leave execution and output failures to callers."""
    details: dict[str, JsonValue] = {}
    if isinstance(error, ModelHTTPError):
        code = _http_error_code(error.status_code)
        details = {
            "model_name": error.model_name,
            "status_code": error.status_code,
        }
        retry_after = error.retry_after
        if isinstance(retry_after, (int, float, str)) and not isinstance(
            retry_after, bool
        ):
            details["retry_after"] = retry_after
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
    return AIError(
        code,
        safe_details=details,
        diagnostics=diagnostics if diagnostics is not None else ErrorDiagnostics.from_exception(error),
    )


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


__all__ = ["model_binding_error"]
