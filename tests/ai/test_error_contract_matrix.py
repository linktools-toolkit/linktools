#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Required error-contract matrix coverage."""

from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    ContentFilterError,
    ModelAPIError,
    UnexpectedModelBehavior,
)
from pydantic_ai.usage import RunUsage, UsageLimits

from linktools.ai.agent._executor import _execution_error
from linktools.ai.core import (
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    StructuredRedactor,
    TaskStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import DefaultExecutionService
from linktools.ai.task import DefaultTaskService


def _map(error: Exception) -> AIError:
    return _execution_error(
        error,
        usage_limits=UsageLimits(),
        run_usage=RunUsage(),
    )


@pytest.mark.parametrize(
    ("error", "expected", "retryable"),
    [
        (ModelAPIError("model", "provider failed"), ErrorCode.MODEL_API_ERROR, False),
        (ContentFilterError("filtered"), ErrorCode.MODEL_CONTENT_FILTERED, False),
        (
            UnexpectedModelBehavior("invalid response"),
            ErrorCode.MODEL_RESPONSE_INVALID,
            False,
        ),
        (
            ConcurrencyLimitExceeded("queue full"),
            ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED,
            True,
        ),
    ],
)
def test_pydantic_execution_errors_have_stable_domain_codes(
    error: Exception,
    expected: ErrorCode,
    retryable: bool,
) -> None:
    mapped = _map(error)
    assert mapped.code is expected
    assert mapped.retryable is retryable


def test_non_ai_redaction_error_is_internal_without_message_leak() -> None:
    safe = StructuredRedactor().safe_error(
        RuntimeError("secret failure text"),
        operation_id="operation",
    )
    assert safe.code == ErrorCode.INTERNAL_ERROR.value
    assert safe.retryable is False
    assert safe.operation_id == "operation"
    assert safe.safe_details == {}
    assert "secret failure text" not in str(safe)


def test_unknown_model_route_has_stable_connection_error() -> None:
    with pytest.raises(AIError) as error:
        ModelRegistry().snapshot().resolve("missing")
    assert error.value.code is ErrorCode.MODEL_CONNECTION_NOT_FOUND


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        (ErrorCode.INTERNAL_ERROR, False),
        (ErrorCode.MODEL_API_ERROR, False),
        (ErrorCode.MODEL_REQUEST_REJECTED, False),
        (ErrorCode.MODEL_RATE_LIMITED, True),
        (ErrorCode.MODEL_TIMEOUT, True),
        (ErrorCode.MODEL_UNAVAILABLE, True),
        (ErrorCode.MODEL_RESPONSE_INVALID, False),
        (ErrorCode.MODEL_CONTENT_FILTERED, False),
        (ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED, True),
        (ErrorCode.EXECUTION_NOT_READY, True),
        (ErrorCode.EXECUTION_WAIT_TIMEOUT, True),
        (ErrorCode.TASK_WAIT_TIMEOUT, True),
        (ErrorCode.TOOL_RETRY_REQUIRED, False),
        (ErrorCode.TOOL_EXECUTION_FAILED, False),
        (ErrorCode.ASSET_NOT_FOUND, False),
        (ErrorCode.STORAGE_PATH_INVALID, False),
    ],
)
def test_new_error_codes_have_fixed_retryability(
    code: ErrorCode,
    retryable: bool,
) -> None:
    assert AIError(code).retryable is retryable


@pytest.mark.asyncio
async def test_execution_result_before_terminal_is_not_ready() -> None:
    service = object.__new__(DefaultExecutionService)

    async def load_authorized(
        execution_id: str,
        principal: Principal,
        action: object,
    ) -> object:
        del execution_id, principal, action
        return SimpleNamespace(status=ExecutionStatus.STARTED)

    service._load_authorized = load_authorized  # type: ignore[method-assign]
    principal = Principal("principal", "tenant", "service")
    with pytest.raises(AIError) as error:
        await DefaultExecutionService.result.__wrapped__(
            service,
            "execution",
            principal=principal,
        )
    assert error.value.code is ErrorCode.EXECUTION_NOT_READY


@pytest.mark.asyncio
async def test_execution_wait_timeout_has_stable_code() -> None:
    service = object.__new__(DefaultExecutionService)

    async def inspect(
        execution_id: str,
        *,
        principal: Principal,
    ) -> object:
        del execution_id, principal
        return SimpleNamespace(status=ExecutionStatus.STARTED)

    service.inspect = inspect  # type: ignore[method-assign]
    service._backend = None
    service._local_waiter = None
    principal = Principal("principal", "tenant", "service")
    with pytest.raises(AIError) as error:
        await DefaultExecutionService.wait.__wrapped__(
            service,
            "execution",
            principal=principal,
            timeout_seconds=0.01,
        )
    assert error.value.code is ErrorCode.EXECUTION_WAIT_TIMEOUT


class _AllowAll:
    async def authorize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


class _RunningTasks:
    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef:
        return ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id)

    async def reconcile_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> object:
        del graph_id, tenant_id
        return SimpleNamespace(status=TaskStatus.RUNNING)


@pytest.mark.asyncio
async def test_task_wait_timeout_has_stable_code() -> None:
    persistence = SimpleNamespace(tasks=_RunningTasks())
    service = DefaultTaskService(persistence, _AllowAll())  # type: ignore[arg-type]
    principal = Principal("principal", "tenant", "service")
    with pytest.raises(AIError) as error:
        await service.wait_graph(
            "graph",
            principal=principal,
            timeout_seconds=0.01,
        )
    assert error.value.code is ErrorCode.TASK_WAIT_TIMEOUT
