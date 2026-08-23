"""Temporal client adapter implementing the runtime workflow port."""

from collections.abc import Mapping
from typing import Protocol

from linktools.core import environ

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..runtime import (
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionRequest,
    RuntimeObjectKeyFactory,
    WorkflowQueryResult,
    WorkflowUpdateResult,
)
from ..storage import ObjectStore
from ..task import TaskGraphHandle, TaskGraphRequest, TaskGraphView
from ._request import put_execution_request, put_task_request
from .workflow import ExecutionWorkflowInput, TaskWorkflowInput

_logger = environ.get_logger("ai.temporal.gateway")
QUERY_NAMES = frozenset({"inspect", "pending_approvals", "pending_external_calls"})
UPDATE_NAMES = frozenset({"approve", "supply_external_result", "cancel"})


class TemporalClient(Protocol):
    async def start_workflow(
        self,
        workflow: str,
        request: ExecutionWorkflowInput,
        *,
        workflow_id: str,
    ) -> ExecutionHandle: ...

    async def start_task_graph(
        self,
        request: TaskWorkflowInput,
        *,
        workflow_id: str,
    ) -> TaskGraphHandle:
        """Start a rearm-safe TaskWorkflow.

        A Temporal SDK adapter implementing this port must use
        ``WorkflowIDConflictPolicy.USE_EXISTING`` for an open workflow ID and
        ``WorkflowIDReusePolicy.ALLOW_DUPLICATE`` for a closed workflow ID.
        It must not use ``TERMINATE_EXISTING``.
        """
        ...

    async def update_workflow(
        self,
        workflow_id: str,
        operation: str,
        payload: "Mapping[str, JsonValue]",
    ) -> WorkflowUpdateResult: ...

    async def query_workflow(
        self, workflow_id: str, query: str
    ) -> WorkflowQueryResult: ...

    async def cancel_workflow(self, workflow_id: str) -> CancelExecutionResult: ...

    async def cancel_task_graph(
        self, workflow_id: str, idempotency_key: str
    ) -> TaskGraphView: ...


class WorkflowGateway:
    def __init__(
        self,
        client: TemporalClient,
        *,
        worker_build: str,
        request_store: ObjectStore,
        namespace: str,
    ) -> None:
        if not isinstance(worker_build, str) or not worker_build.strip():
            raise ValueError("Temporal worker build is required")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("Temporal namespace is required")
        self._client = client
        self._worker_build = worker_build
        self._request_store = request_store
        self._request_keys = RuntimeObjectKeyFactory(namespace)

    async def start_execution(
        self,
        workflow_id: str,
        request: ExecutionRequest,
        *,
        binding_digest: str,
        binding: Mapping[str, JsonValue],
    ) -> ExecutionHandle:
        if not workflow_id.strip():
            raise ValueError("workflow id is required")
        request_ref = await put_execution_request(
            self._request_store,
            self._request_keys,
            request,
            binding_digest=binding_digest,
            binding=binding,
        )
        workflow_request = ExecutionWorkflowInput(
            execution_id=workflow_id,
            tenant_id=request.principal.tenant_id,
            binding_digest=binding_digest,
            request_ref=request_ref,
            worker_build=self._worker_build,
        )
        _logger.debug(
            "starting durable execution workflow: workflow_id=%s binding=%s",
            workflow_id,
            binding_digest,
        )
        return await self._client.start_workflow(
            "execution",
            workflow_request,
            workflow_id=workflow_id,
        )

    async def update_execution(
        self,
        workflow_id: str,
        operation: str,
        payload: "Mapping[str, JsonValue]",
    ) -> WorkflowUpdateResult:
        if not workflow_id.strip() or operation not in UPDATE_NAMES:
            raise ValueError("unsupported execution update")
        if operation == "supply_external_result":
            required = {
                "operation_id",
                "call_id",
                "idempotency_key",
                "object_ref",
                "payload_digest",
                "principal_id",
            }
            if set(payload) != required or any(
                not isinstance(payload[key], str) or not payload[key].strip()
                for key in required
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif operation == "approve":
            required = {
                "operation_id",
                "approval_id",
                "idempotency_key",
                "decision",
                "principal_id",
                "decision_digest",
            }
            if set(payload) != required or any(
                not isinstance(payload[key], str) or not payload[key].strip()
                for key in required
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif operation == "cancel":
            required = {"operation_id"}
            if set(payload) != required or not isinstance(
                payload["operation_id"], str
            ) or not payload["operation_id"].strip():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._client.update_workflow(workflow_id, operation, payload)

    async def query_execution(
        self, workflow_id: str, query: str
    ) -> WorkflowQueryResult:
        if not workflow_id.strip() or query not in QUERY_NAMES:
            raise ValueError("unsupported execution query")
        return await self._client.query_workflow(workflow_id, query)

    async def cancel_execution(self, workflow_id: str) -> CancelExecutionResult:
        if not workflow_id.strip():
            raise ValueError("workflow id is required")
        return await self._client.cancel_workflow(workflow_id)

    async def start_task_graph(
        self, workflow_id: str, request: TaskGraphRequest
    ) -> TaskGraphHandle:
        if not workflow_id.strip():
            raise ValueError("workflow id is required")
        _logger.debug("starting durable task workflow: workflow_id=%s", workflow_id)
        request_ref = await put_task_request(
            self._request_store,
            self._request_keys,
            request,
        )
        workflow_request = TaskWorkflowInput.from_request(
            request,
            request_ref=request_ref,
            worker_build=self._worker_build,
        )
        return await self._client.start_task_graph(
            workflow_request, workflow_id=workflow_id
        )

    async def cancel_task_graph(
        self, workflow_id: str, idempotency_key: str
    ) -> TaskGraphView:
        if not workflow_id.strip() or not idempotency_key.strip():
            raise ValueError("workflow and idempotency keys are required")
        return await self._client.cancel_task_graph(workflow_id, idempotency_key)


__all__ = ["QUERY_NAMES", "UPDATE_NAMES", "TemporalClient", "WorkflowGateway"]
