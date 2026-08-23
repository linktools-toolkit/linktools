#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable error classification and propagation contracts."""

from datetime import datetime, timezone

import httpx
import pytest
from linktools.ai.agent._executor import _execution_error
from linktools.ai.core import ExecutionStatus, ToolOperationStatus, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionResult
from linktools.ai.runtime._evaluation import _stable_error
from linktools.ai.runtime._execution import _terminal_error
from linktools.ai.runtime._local import _secondary_execution_error
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy
from openai import APIError as OpenAIAPIError
from pydantic_ai.exceptions import ModelHTTPError, RunCancelled
from pydantic_ai.usage import RunUsage, UsageLimits


def _map(error: Exception) -> AIError:
    return _execution_error(
        error,
        usage_limits=UsageLimits(),
        run_usage=RunUsage(),
    )


@pytest.mark.parametrize(
    ("status_code", "expected", "retryable"),
    [
        (400, ErrorCode.MODEL_REQUEST_REJECTED, False),
        (408, ErrorCode.MODEL_TIMEOUT, True),
        (429, ErrorCode.MODEL_RATE_LIMITED, True),
        (500, ErrorCode.MODEL_UNAVAILABLE, True),
        (503, ErrorCode.MODEL_UNAVAILABLE, True),
    ],
)
def test_model_http_errors_have_stable_domain_codes(
    status_code: int,
    expected: ErrorCode,
    retryable: bool,
) -> None:
    mapped = _map(
        ModelHTTPError(
            status_code=status_code,
            model_name="model",
            body={"secret": "provider body must not escape"},
            headers={"Retry-After": "3"} if status_code == 429 else None,
        )
    )
    assert mapped.code is expected
    assert mapped.retryable is retryable
    assert mapped.safe_details["status_code"] == status_code
    assert mapped.safe_details["model_name"] == "model"
    assert "secret" not in mapped.safe_details
    if status_code == 429:
        assert mapped.safe_details["retry_after"] == 3.0


def test_first_party_run_cancelled_is_an_execution_cancellation() -> None:
    mapped = _map(RunCancelled("cancelled by application"))
    assert mapped.code is ErrorCode.EXECUTION_CANCELLED
    assert mapped.retryable is False


def test_unknown_agent_exception_is_internal_not_storage() -> None:
    mapped = _map(RuntimeError("boom"))
    assert mapped.code is ErrorCode.INTERNAL_ERROR
    assert mapped.safe_details == {"phase": "agent_execution"}


def test_raw_openai_api_error_stays_in_model_domain_without_payload_leak() -> None:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    mapped = _map(
        OpenAIAPIError(
            "provider secret",
            request=request,
            body={"secret": "provider body must not escape"},
        )
    )
    assert mapped.code is ErrorCode.MODEL_API_ERROR
    assert mapped.safe_details == {}


def test_execution_result_enforces_terminal_error_contract() -> None:
    usage = UsageMetrics()
    succeeded = ExecutionResult(
        "success",
        ExecutionStatus.SUCCEEDED,
        {"ok": True},
        "schema",
        1,
        "a" * 64,
        usage,
    )
    assert succeeded.error_code is None
    assert succeeded.safe_error_details == {}

    failed = ExecutionResult(
        "failed",
        ExecutionStatus.FAILED,
        None,
        None,
        None,
        None,
        usage,
        ErrorCode.MODEL_REQUEST_REJECTED.value,
        {"status_code": 400},
    )
    assert failed.error_code == ErrorCode.MODEL_REQUEST_REJECTED.value
    assert failed.safe_error_details == {"status_code": 400}

    cancelled = ExecutionResult(
        "cancelled",
        ExecutionStatus.CANCELLED,
        None,
        None,
        None,
        None,
        usage,
        ErrorCode.EXECUTION_CANCELLED.value,
    )
    assert cancelled.error_code == ErrorCode.EXECUTION_CANCELLED.value

    with pytest.raises(ValueError):
        ExecutionResult(
            "invalid",
            ExecutionStatus.FAILED,
            None,
            None,
            None,
            None,
            usage,
            ErrorCode.EXECUTION_CANCELLED.value,
        )
    with pytest.raises(ValueError):
        ExecutionResult(
            "unknown",
            ExecutionStatus.FAILED,
            None,
            None,
            None,
            None,
            usage,
            "NOT_A_REAL_ERROR",
        )
    with pytest.raises(ValueError):
        ExecutionResult(
            "failed-with-output",
            ExecutionStatus.FAILED,
            {"unexpected": True},
            None,
            None,
            None,
            usage,
            ErrorCode.INTERNAL_ERROR.value,
        )
    with pytest.raises(ValueError):
        ExecutionResult(
            "cancelled-with-schema",
            ExecutionStatus.CANCELLED,
            None,
            "unexpected-schema",
            None,
            None,
            usage,
            ErrorCode.EXECUTION_CANCELLED.value,
        )


def test_persisted_cancelled_execution_requires_explicit_cancel_code() -> None:
    class PersistedExecution:
        status = ExecutionStatus.CANCELLED
        error_code = None
        safe_error_details: tuple[tuple[str, object], ...] = ()

    with pytest.raises(AIError) as error:
        _terminal_error(PersistedExecution())  # type: ignore[arg-type]
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_ai_error_rejects_non_json_safe_details() -> None:
    with pytest.raises(TypeError):
        AIError(ErrorCode.INTERNAL_ERROR, safe_details={"bad": object()})
    with pytest.raises(TypeError):
        AIError(ErrorCode.INTERNAL_ERROR, safe_details={1: "bad"})  # type: ignore[dict-item]
    with pytest.raises(ValueError):
        AIError(ErrorCode.INTERNAL_ERROR, safe_details={"bad": float("nan")})
    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError):
        AIError(ErrorCode.INTERNAL_ERROR, safe_details={"bad": cycle})


def test_ai_error_copies_nested_safe_details() -> None:
    source = {"nested": [{"value": 1}]}
    error = AIError(ErrorCode.INTERNAL_ERROR, safe_details=source)
    source["nested"][0]["value"] = 2
    assert error.safe_details == {"nested": [{"value": 1}]}


def test_secondary_terminal_error_preserves_primary_contract() -> None:
    primary = AIError(
        ErrorCode.MODEL_REQUEST_REJECTED,
        safe_details={"status_code": 400},
    )
    secondary = AIError(
        ErrorCode.STORAGE_UNAVAILABLE,
        retryable=True,
        operation_id="terminal-write",
        safe_details={"phase": "commit"},
    )
    combined = _secondary_execution_error(secondary, primary)
    assert combined.code is ErrorCode.STORAGE_UNAVAILABLE
    assert combined.retryable is True
    assert combined.operation_id == "terminal-write"
    assert combined.safe_details == {
        "phase": "commit",
        "primary_error_code": ErrorCode.MODEL_REQUEST_REJECTED.value,
        "primary_safe_error_details": {"status_code": 400},
    }


def test_persisted_evaluation_error_rejects_missing_or_unknown_code() -> None:
    assert _stable_error(None).code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert _stable_error("UNKNOWN").code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert _stable_error(ErrorCode.MODEL_TIMEOUT.value).code is ErrorCode.MODEL_TIMEOUT


def _tool_bridge() -> RuntimeToolOperationBridge:
    return RuntimeToolOperationBridge(
        None,  # type: ignore[arg-type]
        InMemoryObjectStore(),
        namespace="error-contract",
        tenant_id="tenant",
        execution_id="execution",
        step_run_id="run",
        binding_fingerprint="a" * 64,
        owner="worker",
        payload_policy=PayloadPolicy(),
    )


def _failed_tool_record(
    *,
    error_code: str,
    error_payload: object,
) -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        tool_operation_id="operation",
        tenant_id="tenant",
        step_run_id="run",
        tool_call_id="call",
        idempotency_key_digest="b" * 64,
        tool_name="tool",
        arguments_digest="c" * 64,
        binding_fingerprint="a" * 64,
        replay_safe=True,
        status=ToolOperationStatus.FAILED,
        owner=None,
        fence=1,
        lease_expires_at=None,
        error_code=error_code,
        created_at=now,
        updated_at=now,
        error_payload=error_payload,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_tool_error_codec_preserves_ai_error_code_and_safe_details() -> None:
    bridge = _tool_bridge()
    code, payload = await bridge._error_payload(
        AIError(
            ErrorCode.MODEL_RATE_LIMITED,
            safe_details={"status_code": 429, "retry_after": 2.0},
        )
    )
    assert code == ErrorCode.MODEL_RATE_LIMITED.value
    decoded = await bridge._decode_error(
        _failed_tool_record(error_code=code, error_payload=payload)
    )
    assert isinstance(decoded, AIError)
    assert decoded.code is ErrorCode.MODEL_RATE_LIMITED
    assert decoded.safe_details == {"status_code": 429, "retry_after": 2.0}


@pytest.mark.asyncio
async def test_tool_error_codec_maps_generic_failure_without_message_leak() -> None:
    bridge = _tool_bridge()
    code, payload = await bridge._error_payload(RuntimeError("provider secret"))
    assert code == ErrorCode.TOOL_EXECUTION_FAILED.value
    decoded = await bridge._decode_error(
        _failed_tool_record(error_code=code, error_payload=payload)
    )
    assert isinstance(decoded, AIError)
    assert decoded.code is ErrorCode.TOOL_EXECUTION_FAILED
    assert decoded.safe_details["phase"] == "tool_execution"
    assert "provider secret" not in str(decoded.safe_details)

    second_code, second_payload = await bridge._error_payload(RuntimeError("other secret"))
    second = await bridge._decode_error(
        _failed_tool_record(error_code=second_code, error_payload=second_payload)
    )
    assert isinstance(second, AIError)
    assert second.safe_details["error_digest"] == decoded.safe_details["error_digest"]


@pytest.mark.asyncio
async def test_tool_error_codec_rejects_unknown_persisted_error_code() -> None:
    bridge = _tool_bridge()
    with pytest.raises(AIError) as captured:
        await bridge._decode_error(
            _failed_tool_record(error_code="UNKNOWN_ERROR", error_payload=None)
        )
    assert captured.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
