#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal client adapter implementing the runtime workflow port."""

from collections.abc import Mapping
from typing import Protocol

from linktools.core import environ

from ..core.json import JsonValue
from ..core import ExecutionProfile, require_profile_available
from ..core.errors import ErrorCode, LinktoolsAIError
from ..runtime.services import (
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionRequest,
    WorkflowQueryResult,
    WorkflowUpdateResult,
)
from ..task.model import TaskGraphHandle, TaskGraphRequest

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


class WorkflowGateway:
    def __init__(self, client: TemporalClient) -> None:
        self._client = client

    async def start_execution(self, workflow_id: str, request: ExecutionRequest) -> ExecutionHandle:
        if not workflow_id.strip():
            raise ValueError("workflow id is required")
        require_profile_available(request.requested_profile)
        if request.requested_profile is not ExecutionProfile.PRODUCTION_SERVICE or request.principal.kind == "LOCAL_TRUSTED":
            raise LinktoolsAIError(ErrorCode.PROFILE_NOT_ALLOWED)
        _logger.info("starting durable execution workflow: workflow_id=%s", workflow_id)
        return await self._client.start_workflow("execution", request, workflow_id=workflow_id)

    async def update_execution(
        self,
        workflow_id: str,
        operation: str,
        payload: 'Mapping[str, JsonValue]',
    ) -> WorkflowUpdateResult:
        if not workflow_id.strip() or operation not in UPDATE_NAMES:
            raise ValueError("unsupported execution update")
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
        require_profile_available(request.requested_profile)
        if request.requested_profile is not ExecutionProfile.PRODUCTION_SERVICE or request.principal.kind == "LOCAL_TRUSTED":
            raise LinktoolsAIError(ErrorCode.PROFILE_NOT_ALLOWED)
        _logger.info("starting durable task workflow: workflow_id=%s", workflow_id)
        return await self._client.start_task_graph(request, workflow_id=workflow_id)


__all__ = ["QUERY_NAMES", "TemporalClient", "UPDATE_NAMES", "WorkflowGateway"]
