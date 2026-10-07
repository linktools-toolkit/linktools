#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider error projection for model bindings."""

from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIError as OpenAIAPIError,
    APIStatusError as OpenAIAPIStatusError,
    APITimeoutError as OpenAIAPITimeoutError,
)
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

from ..core import JsonValue
from ..errors import AIError, ErrorCode, ErrorDiagnostics


def model_binding_error(error: Exception) -> "AIError | None":
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
    elif isinstance(error, OpenAIAPITimeoutError):
        code = ErrorCode.MODEL_TIMEOUT
    elif isinstance(error, OpenAIAPIConnectionError):
        code = ErrorCode.MODEL_UNAVAILABLE
    elif isinstance(error, OpenAIAPIStatusError):
        code = _http_error_code(error.status_code)
        details = {"status_code": error.status_code}
    elif isinstance(error, OpenAIAPIError):
        code = ErrorCode.MODEL_API_ERROR
    else:
        return None
    return AIError(
        code,
        safe_details=details,
        diagnostics=ErrorDiagnostics.from_exception(error),
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
