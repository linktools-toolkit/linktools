#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution failure diagnostic contract tests."""

import hashlib
import json

import pytest
from httpx import Request
from openai import APIConnectionError
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.core import ExecutionStatus, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.runtime._agent_executor import _execution_error
from linktools.ai.runtime.service_api import ExecutionResult


def _digest(exception_type: str, exception_message: str) -> str:
    raw = json.dumps(
        {
            "exception_message": exception_message,
            "exception_type": exception_type,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_error_diagnostics_preserve_raw_exception_message() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("connection reset"))
    assert diagnostics.exception_type == "RuntimeError"
    assert diagnostics.exception_message == "connection reset"
    assert diagnostics.cause_digest == _digest("RuntimeError", "connection reset")


def test_error_diagnostics_truncate_display_but_digest_full_message() -> None:
    message = "x" * 3000
    error_type = type("X" * 300, (RuntimeError,), {})
    error = error_type(message)
    diagnostics = ErrorDiagnostics.from_exception(error)
    assert diagnostics.exception_type == "X" * 256
    assert diagnostics.exception_message == "x" * 2048
    assert diagnostics.cause_digest == _digest("X" * 300, message)


def test_error_diagnostics_survive_broken_exception_string() -> None:
    class BrokenError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    diagnostics = ErrorDiagnostics.from_exception(BrokenError())
    assert diagnostics.exception_type == "BrokenError"
    assert diagnostics.exception_message == ""
    assert diagnostics.cause_digest == _digest("BrokenError", "")


def test_unknown_agent_exception_keeps_safe_contract_and_adds_diagnostics() -> None:
    mapped = _execution_error(
        RuntimeError("provider disconnected"),
        usage_limits=UsageLimits(),
        run_usage=RunUsage(),
    )
    assert mapped.code is ErrorCode.INTERNAL_ERROR
    assert mapped.safe_details == {"phase": "agent_execution"}
    assert mapped.diagnostics == ErrorDiagnostics.from_exception(
        RuntimeError("provider disconnected")
    )
    safe = mapped.to_safe_error(operation_id="operation")
    assert safe.safe_details == {"phase": "agent_execution"}
    assert "provider disconnected" not in str(safe)


def test_provider_connection_error_preserves_original_diagnostics() -> None:
    error = APIConnectionError(
        request=Request("POST", "https://provider.invalid/v1/chat/completions")
    )
    mapped = _execution_error(
        error,
        usage_limits=UsageLimits(),
        run_usage=RunUsage(),
    )
    assert mapped.code is ErrorCode.MODEL_UNAVAILABLE
    assert mapped.retryable is True
    assert mapped.safe_details == {}
    assert mapped.diagnostics == ErrorDiagnostics.from_exception(error)
    assert mapped.diagnostics.exception_type == "APIConnectionError"


def test_failed_execution_result_exposes_diagnostics() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("boom"))
    result = ExecutionResult(
        "execution",
        ExecutionStatus.FAILED,
        None,
        None,
        UsageMetrics(),
        ErrorCode.INTERNAL_ERROR.value,
        {"phase": "agent_execution"},
        diagnostics,
    )
    assert result.error_diagnostics is diagnostics


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        (ExecutionStatus.SUCCEEDED, None),
        (ExecutionStatus.CANCELLED, ErrorCode.EXECUTION_CANCELLED.value),
    ],
)
def test_non_failed_execution_result_rejects_diagnostics(
    status: ExecutionStatus,
    error_code: str | None,
) -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("boom"))
    output = "ok" if status is ExecutionStatus.SUCCEEDED else None
    output_fingerprint = "0" * 64 if status is ExecutionStatus.SUCCEEDED else None
    with pytest.raises(ValueError):
        ExecutionResult(
            "execution",
            status,
            output,
            output_fingerprint,
            UsageMetrics(),
            error_code,
            {},
            diagnostics,
        )


def test_ai_error_safe_projection_ignores_diagnostics() -> None:
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("secret diagnostic"))
    error = AIError(
        ErrorCode.INTERNAL_ERROR,
        safe_details={"phase": "agent_execution"},
        diagnostics=diagnostics,
    )
    safe = error.to_safe_error(operation_id="operation")
    assert safe.safe_details == {"phase": "agent_execution"}
    assert "secret diagnostic" not in str(safe)
