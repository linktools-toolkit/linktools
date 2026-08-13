#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    WorkflowQueryResult,
    WorkflowUpdateResult,
)
from ..task import TaskGraphHandle, TaskGraphRequest, TaskGraphView

_logger = environ.get_logger("ai.temporal.gateway")
QUERY_NAMES = frozenset({"inspect", "pending_approvals", "pending_external_calls"})
UPDATE_NAMES = frozenset({"approve", "supply_external_result", "cancel"})


class TemporalClient(Protocol):
    async def start_workflow(
        self,
        workflow: str,
        request: ExecutionRequest,
        *,
        workflow_id: str,
    ) -> ExecutionHandle: ...

    async def start_task_graph(
        self,
        request: TaskGraphRequest,
        *,
        workflow_id: str,
    ) -> TaskGraphHandle: ...

    async def update_workflow(
        self,
        workflow_id: str,
        operation: str,
        payload: 'Mapping[str, JsonValue]',
    ) -> WorkflowUpdateResult: ...

    async def query_workflow(self, workflow_id: str, query: str) -> WorkflowQueryResult: ...

    async def cancel_workflow(self, workflow_id: str) -> CancelExecutionResult: ...

    async def cancel_task_graph(self, workflow_id: str, idempotency_key: str) -> TaskGraphView: ...


class WorkflowGateway:
    def __init__(self, client: TemporalClient) -> None:
        self._client = client

    async def start_execution(self, workflow_id: str, request: ExecutionRequest) -> ExecutionHandle:
        if not workflow_id.strip():
            raise ValueError("workflow id is required")
        _logger.debug("starting durable execution workflow: workflow_id=%s", workflow_id)
        return await self._client.start_workflow("execution", request, workflow_id=workflow_id)

    async def update_execution(
        self,
        workflow_id: str,
        operation: str,
        payload: 'Mapping[str, JsonValue]',
    ) -> WorkflowUpdateResult:
        if not workflow_id.strip() or operation not in UPDATE_NAMES:
            raise ValueError("unsupported execution update")
        if operation == "supply_external_result":
            required = {"call_id", "idempotency_key", "object_ref", "payload_digest", "principal_id"}
            if set(payload) != required or any(not isinstance(payload[key], str) or not payload[key].strip() for key in required):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if operation == "approve":
            required = {"approval_id", "idempotency_key", "decision", "principal_id", "decision_digest"}
            if set(payload) != required or any(not isinstance(payload[key], str) or not payload[key].strip() for key in required):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self._client.update_workflow(workflow_id, operation, payload)

    async def query_execution(self, workflow_id: str, query: str) -> WorkflowQueryResult:
        if not workflow_id.strip() or query not in QUERY_NAMES:
            raise ValueError("unsupported execution query")
        return await self._client.query_workflow(workflow_id, query)

    async def cancel_execution(self, workflow_id: str) -> CancelExecutionResult:
        if not workflow_id.strip():
            raise ValueError("workflow id is required")
        return await self._client.cancel_workflow(workflow_id)

    async def start_task_graph(self, workflow_id: str, request: TaskGraphRequest) -> TaskGraphHandle:
        if not workflow_id.strip():
            raise ValueError("workflow id is required")
        _logger.debug("starting durable task workflow: workflow_id=%s", workflow_id)
        return await self._client.start_task_graph(request, workflow_id=workflow_id)

    async def cancel_task_graph(self, workflow_id: str, idempotency_key: str) -> TaskGraphView:
        if not workflow_id.strip() or not idempotency_key.strip():
            raise ValueError("workflow and idempotency keys are required")
        return await self._client.cancel_task_graph(workflow_id, idempotency_key)


__all__ = ["QUERY_NAMES", "TemporalClient", "UPDATE_NAMES", "WorkflowGateway"]
