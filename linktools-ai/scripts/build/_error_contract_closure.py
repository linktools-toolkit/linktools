#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporary exact patch driver for the error-contract closure branch."""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    target.write_text(text.replace(old, new))


def replace_count(path: str, old: str, new: str, expected: int) -> None:
    target = Path(path)
    text = target.read_text()
    count = text.count(old)
    if count != expected:
        raise SystemExit(f"{path}: expected {expected} replacements, found {count}")
    target.write_text(text.replace(old, new))


def main() -> None:
    path = "linktools-ai/src/linktools/ai/agent/_executor.py"
    replace_once(
        path,
        '''from linktools.core import environ
from pydantic import ValidationError
''',
        '''from linktools.core import environ
from openai import APIError as OpenAIAPIError
from pydantic import ValidationError
''',
    )
    replace_once(
        path,
        '''    if isinstance(error, ModelAPIError):
        return AIError(
            ErrorCode.MODEL_API_ERROR,
            retryable=False,
            safe_details={"model_name": error.model_name},
        )
    if isinstance(error, UnexpectedModelBehavior):
''',
        '''    if isinstance(error, ModelAPIError):
        return AIError(
            ErrorCode.MODEL_API_ERROR,
            retryable=False,
            safe_details={"model_name": error.model_name},
        )
    if isinstance(error, OpenAIAPIError):
        return AIError(ErrorCode.MODEL_API_ERROR, retryable=False)
    if isinstance(error, UnexpectedModelBehavior):
''',
    )

    path = "linktools-ai/src/linktools/ai/runtime/_local.py"
    replace_once(
        path,
        '''        if isinstance(error, AIError):
            failure = _WorkerFailure(error.code, dict(error.safe_details))
        else:
            failure = _WorkerFailure(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                {"phase": "local_execution_worker"},
            )
''',
        '''        if isinstance(error, AIError):
            failure = _WorkerFailure(error.code, dict(error.safe_details))
        else:
            failure = _WorkerFailure(
                ErrorCode.INTERNAL_ERROR,
                {"phase": "local_execution_worker"},
            )
''',
    )
    replace_once(
        path,
        '''                if current is not None and current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    await self._commit_failure(current, error, run_id=run_id)
                persisted = await self._execution.executions.get(
                    execution_id,
                    tenant_id=original.tenant_id,
                )
''',
        '''                if current is not None and current.status not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }:
                    try:
                        await self._commit_failure(current, error, run_id=run_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception as commit_error:
                        try:
                            persisted = await self._execution.executions.get(
                                execution_id,
                                tenant_id=original.tenant_id,
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as readback_error:
                            raise _secondary_execution_error(readback_error, error) from error
                        if persisted is not None and persisted.status in {
                            ExecutionStatus.SUCCEEDED,
                            ExecutionStatus.FAILED,
                            ExecutionStatus.CANCELLED,
                        }:
                            operation_result = _execution_operation_result(persisted.status)
                            _logger.error(
                                "terminal finalization failed after durable execution terminal: execution=%s",
                                execution_id,
                                exc_info=True,
                            )
                            return
                        raise _secondary_execution_error(commit_error, error) from error
                persisted = await self._execution.executions.get(
                    execution_id,
                    tenant_id=original.tenant_id,
                )
''',
    )
    replace_once(
        path,
        '''    async def _commit_failure(self, execution: ExecutionRecord, error: Exception, *, run_id: str | None = None) -> None:
        code = ErrorCode.OUTPUT_VALIDATION_FAILED if isinstance(error, ValidationError) else error.code if isinstance(error, AIError) else ErrorCode.EXECUTION_FAILED
        details = error.safe_details if isinstance(error, AIError) else {}
        await self._commit_terminal(
            execution,
            ExecutionStatus.FAILED,
            None,
            code.value,
            StopReason.ERROR,
            run_id=run_id,
            safe_error_details=details,
        )
''',
        '''    async def _commit_failure(self, execution: ExecutionRecord, error: Exception, *, run_id: str | None = None) -> None:
        code = _execution_error_code(error)
        details = _execution_error_details(error)
        cancelled = code is ErrorCode.EXECUTION_CANCELLED
        await self._commit_terminal(
            execution,
            ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.FAILED,
            None,
            code.value,
            StopReason.CANCELLED if cancelled else StopReason.ERROR,
            run_id=run_id,
            safe_error_details=details,
        )
''',
    )
    replace_once(
        path,
        '''        recovery_run = None
        recovery_snapshot = None
        if run_id is not None and status is not ExecutionStatus.SUCCEEDED:
            recovery_run = await self._steps.get_run(run_id=run_id)
            recovery_snapshot = await self._steps.latest_snapshot(
                run_id=run_id,
                include_interrupted=True,
            )
''',
        '''        recovery_run = None
        recovery_snapshot = None
        if run_id is not None and status is not ExecutionStatus.SUCCEEDED:
            candidate_run = await self._steps.get_run(run_id=run_id)
            candidate_snapshot = await self._steps.latest_snapshot(
                run_id=run_id,
                include_interrupted=True,
            )
            if candidate_snapshot is not None:
                if candidate_run is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                recovery_run = candidate_run
                recovery_snapshot = candidate_snapshot
''',
    )
    replace_once(
        path,
        '''def _is_infrastructure_error(error: Exception) -> bool:
''',
        '''def _execution_error_code(error: Exception) -> ErrorCode:
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED
    if isinstance(error, AIError):
        return error.code
    return ErrorCode.INTERNAL_ERROR


def _execution_error_details(error: Exception) -> dict[str, JsonValue]:
    return dict(error.safe_details) if isinstance(error, AIError) else {}


def _secondary_execution_error(error: Exception, primary: Exception) -> AIError:
    primary_details: dict[str, JsonValue] = {
        "primary_error_code": _execution_error_code(primary).value,
        "primary_safe_error_details": _execution_error_details(primary),
    }
    if isinstance(error, AIError):
        details = dict(error.safe_details)
        details.update(primary_details)
        return AIError(
            error.code,
            category=error.category,
            retryable=error.retryable,
            operation_id=error.operation_id,
            safe_details=details,
        )
    return AIError(
        ErrorCode.INTERNAL_ERROR,
        safe_details={
            "phase": "execution_terminal_commit",
            **primary_details,
        },
    )


def _is_infrastructure_error(error: Exception) -> bool:
''',
    )

    path = "linktools-ai/src/linktools/ai/runtime/_tool.py"
    replace_once(
        path,
        '''        digest = hashlib.sha256(f"{type(error).__name__}:{error}".encode()).hexdigest()
        return ErrorCode.INTERNAL_ERROR.value, await self._json_payload(
            {
                "kind": "error",
                "code": ErrorCode.INTERNAL_ERROR.value,
                "safe_details": {
                    "error_digest": digest,
                    "phase": "tool_execution",
                },
            }
        )
''',
        '''        digest = hashlib.sha256(type(error).__qualname__.encode("utf-8")).hexdigest()
        return ErrorCode.TOOL_EXECUTION_FAILED.value, await self._json_payload(
            {
                "kind": "error",
                "code": ErrorCode.TOOL_EXECUTION_FAILED.value,
                "safe_details": {
                    "error_digest": digest,
                    "phase": "tool_execution",
                },
            }
        )
''',
    )

    path = "linktools-ai/src/linktools/ai/task/_local.py"
    replace_once(
        path,
        '''        except asyncio.CancelledError:
            raise
        except Exception as error:
            run.failure = error if isinstance(error, AIError) else AIError(ErrorCode.STORAGE_UNAVAILABLE)
            _logger.exception("local task graph scheduler failed: tenant=%s graph=%s", key[0], key[1])
''',
        '''        except asyncio.CancelledError:
            raise
        except Exception as error:
            run.failure = (
                error
                if isinstance(error, AIError)
                else AIError(
                    ErrorCode.INTERNAL_ERROR,
                    safe_details={"phase": "task_scheduler"},
                )
            )
            _logger.exception("local task graph scheduler failed: tenant=%s graph=%s", key[0], key[1])
''',
    )

    path = "linktools-ai/src/linktools/ai/asset/_repository.py"
    replace_once(
        path,
        '''        if len(first.candidates) == 0:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
''',
        '''        if len(first.candidates) == 0:
            raise AIError(ErrorCode.ASSET_NOT_FOUND)
''',
    )

    path = "linktools-ai/src/linktools/ai/runtime/_evaluation.py"
    replace_once(
        path,
        '''            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_error(existing.error_code, ErrorCode.STORAGE_UNAVAILABLE)
''',
        '''            if existing.status is IdempotencyStatus.FAILED:
                raise _stable_error(existing.error_code)
''',
    )
    replace_once(
        path,
        '''def _stable_error(error_code: str | None, fallback: ErrorCode) -> AIError:
    try:
        return AIError(fallback if error_code is None else ErrorCode(error_code))
    except ValueError:
        return AIError(fallback)
''',
        '''def _stable_error(error_code: str | None) -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(error_code)
    except ValueError:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AIError(code)
''',
    )

    path = "tests/ai/test_closure_fix.py"
    replace_count(
        path,
        "assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE",
        "assert raised.value.code is ErrorCode.INTERNAL_ERROR",
        2,
    )
    replace_count(
        path,
        "assert late_raised.value.code is ErrorCode.STORAGE_UNAVAILABLE",
        "assert late_raised.value.code is ErrorCode.INTERNAL_ERROR",
        1,
    )

    path = "tests/ai/test_asset_repository.py"
    replace_once(
        path,
        '''    with pytest.raises(AIError) as error:
        await repository.resolve(AssetRef("subagent", "team/foo/child"))
    assert error.value.code is ErrorCode.STORAGE_NOT_FOUND
''',
        '''    with pytest.raises(AIError) as error:
        await repository.resolve(AssetRef("subagent", "team/foo/child"))
    assert error.value.code is ErrorCode.ASSET_NOT_FOUND
''',
    )

    path = "tests/ai/test_error_contract.py"
    replace_once(
        path,
        '''from datetime import datetime, timezone

import pytest
from pydantic_ai.exceptions import ModelHTTPError, RunCancelled
''',
        '''from datetime import datetime, timezone

import httpx
import pytest
from openai import APIError as OpenAIAPIError
from pydantic_ai.exceptions import ModelHTTPError, RunCancelled
''',
    )
    replace_once(
        path,
        '''from linktools.ai.runtime import ExecutionResult
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
''',
        '''from linktools.ai.runtime import ExecutionResult
from linktools.ai.runtime._evaluation import _stable_error
from linktools.ai.runtime._local import _secondary_execution_error
from linktools.ai.runtime._tool import RuntimeToolOperationBridge, ToolOperationRecord
''',
    )
    replace_once(
        path,
        '''def test_unknown_agent_exception_is_internal_not_storage() -> None:
    mapped = _map(RuntimeError("boom"))
    assert mapped.code is ErrorCode.INTERNAL_ERROR
    assert mapped.safe_details == {"phase": "agent_execution"}


def test_execution_result_enforces_terminal_error_contract() -> None:
''',
        '''def test_unknown_agent_exception_is_internal_not_storage() -> None:
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
''',
    )
    replace_once(
        path,
        '''def test_ai_error_rejects_non_json_safe_details() -> None:
    with pytest.raises(TypeError):
        AIError(ErrorCode.INTERNAL_ERROR, safe_details={"bad": object()})
    with pytest.raises(ValueError):
        AIError(ErrorCode.INTERNAL_ERROR, safe_details={"bad": float("nan")})


def _tool_bridge() -> RuntimeToolOperationBridge:
''',
        '''def test_ai_error_rejects_non_json_safe_details() -> None:
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
''',
    )
    replace_once(
        path,
        '''@pytest.mark.asyncio
async def test_tool_error_codec_maps_unknown_runtime_error_to_internal() -> None:
    bridge = _tool_bridge()
    code, payload = await bridge._error_payload(RuntimeError("provider secret"))
    assert code == ErrorCode.INTERNAL_ERROR.value
    decoded = await bridge._decode_error(
        _failed_tool_record(error_code=code, error_payload=payload)
    )
    assert isinstance(decoded, AIError)
    assert decoded.code is ErrorCode.INTERNAL_ERROR
    assert decoded.safe_details["phase"] == "tool_execution"
    assert "provider secret" not in str(decoded.safe_details)
''',
        '''@pytest.mark.asyncio
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
''',
    )


if __name__ == "__main__":
    main()
