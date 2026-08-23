#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Required error-contract matrix coverage."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.agent._executor import DurableBoundary, _execution_error, _map_event
from linktools.ai.core import (
    ExecutionStatus,
    JsonValue,
    Principal,
    ResourceKind,
    ResourceRef,
    StructuredRedactor,
    TaskStatus,
    UsageMetrics,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import DefaultExecutionService, ExecutionResult
from linktools.ai.runtime._planner import _execution_failure
from linktools.ai.runtime._subagent import _subagent_result
from linktools.ai.storage import StoragePath
from linktools.ai.task import (
    DefaultTaskService,
    TaskGraph,
    TaskLease,
    TaskNode,
    TaskNodeView,
)
from linktools.ai.temporal._task_operation import _RuntimeTaskOperation
from linktools.cli import CommandError
from linktools.commands.ai.run import _emit_result
from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    ContentFilterError,
    ModelAPIError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import FunctionToolResultEvent, RetryPromptPart
from pydantic_ai.usage import RunUsage, UsageLimits


def _map(error: Exception) -> AIError:
    return _execution_error(
        error,
        usage_limits=UsageLimits(),
        run_usage=RunUsage(),
    )


def _failed_result(
    execution_id: str = "execution",
    *,
    code: ErrorCode = ErrorCode.MODEL_RATE_LIMITED,
    details: "dict[str, JsonValue] | None" = None,
) -> ExecutionResult:
    return ExecutionResult(
        execution_id,
        ExecutionStatus.FAILED,
        None,
        None,
        None,
        None,
        UsageMetrics(),
        code.value,
        details or {"provider": "test"},
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


def test_tool_retry_event_uses_stable_retry_code() -> None:
    emission = _map_event(
        FunctionToolResultEvent(
            part=RetryPromptPart(
                content="retry",
                tool_name="tool",
                tool_call_id="call",
            )
        )
    )
    assert isinstance(emission, DurableBoundary)
    assert emission.payload["safe_error_code"] == ErrorCode.TOOL_RETRY_REQUIRED.value


def test_local_task_child_error_contract_is_preserved() -> None:
    failure = _execution_failure(
        _failed_result(details={"status_code": 429})
    )
    assert failure.code is ErrorCode.MODEL_RATE_LIMITED
    assert failure.safe_details == {"status_code": 429}


def test_subagent_result_preserves_child_error_contract() -> None:
    payload = _subagent_result(
        _failed_result(details={"status_code": 429})
    )
    assert payload["error_code"] == ErrorCode.MODEL_RATE_LIMITED.value
    assert payload["safe_error_details"] == {"status_code": 429}


def test_invalid_storage_path_has_stable_code() -> None:
    with pytest.raises(AIError) as error:
        StoragePath.parse("../outside")
    assert error.value.code is ErrorCode.STORAGE_PATH_INVALID


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


class _FailedAgent:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result

    async def run(
        self,
        prompt: str,
        *,
        session_id: str,
        memory_scope: str,
    ) -> ExecutionResult:
        del prompt, session_id, memory_scope
        return self._result

    def stream(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("JSON result path must not scan execution events")


class _FailedRuntime:
    def __init__(self, result: ExecutionResult) -> None:
        self._agent = _FailedAgent(result)

    def agent(self, *, planning: bool, thinking: bool) -> _FailedAgent:
        del planning, thinking
        return self._agent


@pytest.mark.asyncio
async def test_cli_json_failed_result_uses_result_contract_without_event_scan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _FailedRuntime(
        _failed_result(details={"status_code": 429})
    )
    with pytest.raises(CommandError):
        await _emit_result(
            runtime,  # type: ignore[arg-type]
            "prompt",
            "session",
            "memory",
            True,
            False,
            False,
        )
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["error_code"] == ErrorCode.MODEL_RATE_LIMITED.value
    assert payload["safe_error_details"] == {"status_code": 429}


class _TemporalRunner:
    def __init__(self, result: ExecutionResult) -> None:
        self._result = result

    async def terminal_result(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        del principal
        assert execution_id == self._result.execution_id
        return self._result


class _TemporalRepository:
    def __init__(self, view: TaskNodeView) -> None:
        self.view = view
        self.failed_code: str | None = None

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
    ) -> object:
        assert tenant_id == lease.tenant_id
        assert error_digest
        self.failed_code = error_code
        self.view = TaskNodeView(
            self.view.graph_id,
            self.view.node_id,
            self.view.dependencies,
            TaskStatus.FAILED,
            self.view.owner,
            self.view.fence,
            self.view.lease_expires_at,
            None,
            error_code,
            error_digest,
        )
        return self.view

    async def reconcile_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> object:
        del graph_id, tenant_id
        return SimpleNamespace(status=TaskStatus.FAILED)


class _TemporalOperation(_RuntimeTaskOperation):
    def __init__(
        self,
        runner: _TemporalRunner,
        repository: _TemporalRepository,
        stored: object,
    ) -> None:
        self._runner = runner
        self._repository = repository
        self._stored = stored

    async def _load_request(self, request: object) -> object:
        del request
        return self._stored

    async def _read_node(
        self,
        request: object,
        node: TaskNode,
        *,
        reconcile: bool,
    ) -> TaskNodeView:
        del request, node, reconcile
        return self._repository.view


@pytest.mark.asyncio
async def test_temporal_task_child_error_contract_is_preserved() -> None:
    principal = Principal("principal", "tenant", "service")
    node = TaskNode("node", (), input={})
    graph = TaskGraph("graph", (node,))
    lease = TaskLease(
        "graph",
        "node",
        "tenant",
        "worker",
        1,
        datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    repository = _TemporalRepository(
        TaskNodeView(
            "graph",
            "node",
            (),
            TaskStatus.RUNNING,
            "worker",
            1,
            lease.lease_expires_at,
            None,
            None,
            None,
        )
    )
    operation = _TemporalOperation(
        _TemporalRunner(
            _failed_result(
                "graph:node",
                details={"status_code": 429},
            )
        ),
        repository,
        SimpleNamespace(graph=graph, principal=principal),
    )
    updated = await operation.settle(
        SimpleNamespace(graph_id="graph", tenant_id="tenant"),  # type: ignore[arg-type]
        lease,
        SimpleNamespace(execution_id="graph:node", status="FAILED"),  # type: ignore[arg-type]
    )
    assert repository.failed_code == ErrorCode.MODEL_RATE_LIMITED.value
    assert updated.status is TaskStatus.FAILED
    assert updated.error_code == ErrorCode.MODEL_RATE_LIMITED.value
