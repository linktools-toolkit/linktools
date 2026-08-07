#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed task service used by the runtime composition root."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from linktools.core import environ

from ..agent.context import AgentBinding
from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256
from ..core.principal import AuthorizationAction, AuthorizationPolicy, ResourceRef
from ..core.value import OperationKind, OperationStatus, Principal, ResourceKind, TaskStatus
from ..task.model import CancelGraphRequest, TaskGraphRequest, TaskGraphResult, TaskGraphView
from .persistence import OperationLedgerRecord, RuntimePersistence
from .services import WorkflowGateway

_logger = environ.get_logger("ai.runtime.task")


class DefaultTaskService:
    """Validate and persist task graphs before scheduling any node."""

    def __init__(self, persistence: RuntimePersistence, authorization: AuthorizationPolicy, workflow_gateway: "WorkflowGateway | None" = None) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._workflow_gateway = workflow_gateway

    async def run_graph(self, binding: AgentBinding, request: TaskGraphRequest) -> TaskGraphResult:
        await self._authorization.authorize(request.principal, AuthorizationAction.TASK_RUN, ResourceRef(ResourceKind.TASK_GRAPH, request.graph.graph_id, request.principal.tenant_id))
        digest = canonical_sha256({"graph_id": request.graph.graph_id, "nodes": [node.task_id for node in request.graph.nodes], "binding": binding.digest, "limits": {"max_nodes": request.limits.max_nodes, "max_depth": request.limits.max_depth, "max_budget": request.limits.max_budget, "max_concurrency": request.limits.max_concurrency}})
        existing = await self._persistence.operations.get(request.idempotency_key, tenant_id=request.principal.tenant_id)
        if existing is not None:
            if existing.request_digest != digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            view = await self._persistence.tasks.get_plan(request.graph.graph_id, tenant_id=request.principal.tenant_id)
            return TaskGraphResult(request.graph.graph_id, view.status if view else TaskStatus.PENDING, ())
        now = datetime.now(timezone.utc)
        operation = OperationLedgerRecord(request.idempotency_key, request.principal.tenant_id, ResourceKind.TASK_GRAPH, request.graph.graph_id, None, OperationKind.TASK_NODE, OperationStatus.PENDING, digest, None, None, None, True, await self._persistence.operations.next_sequence(ResourceKind.TASK_GRAPH, request.graph.graph_id, tenant_id=request.principal.tenant_id), now, now)
        await self._persistence.operations.create(operation)
        try:
            view = await self._persistence.tasks.create_plan(request.graph, tenant_id=request.principal.tenant_id)
            if self._workflow_gateway is not None and request.requested_profile.value == "production-service":
                await self._workflow_gateway.start_task_graph(request.graph.graph_id, request)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=request.principal.tenant_id,
                expected_status=OperationStatus.PENDING,
                next_record=replace(
                    operation,
                    status=OperationStatus.FAILED,
                    error_code=ErrorCode.STORAGE_UNAVAILABLE.value,
                    updated_at=datetime.now(timezone.utc),
                ),
            )
            raise
        await self._persistence.operations.compare_and_swap(request.idempotency_key, tenant_id=request.principal.tenant_id, expected_status=OperationStatus.PENDING, next_record=OperationLedgerRecord(operation.operation_id, operation.tenant_id, operation.resource_kind, operation.resource_id, operation.execution_id, operation.kind, OperationStatus.SUCCEEDED, operation.request_digest, view.graph_id, canonical_sha256({"graph_id": view.graph_id, "status": view.status.value}), None, operation.compactable, operation.sequence, operation.created_at, datetime.now(timezone.utc)))
        _logger.info("task graph submitted: graph=%s tenant=%s profile=%s", view.graph_id, request.principal.tenant_id, request.requested_profile)
        return TaskGraphResult(view.graph_id, view.status, ())

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        header = await self._persistence.tasks.get_header(graph_id, tenant_id=principal.tenant_id)
        if header is None:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.TASK_READ, header)
        view = await self._persistence.tasks.get_plan(graph_id, tenant_id=principal.tenant_id)
        if view is None:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)
        return view

    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        header = await self._persistence.tasks.get_header(graph_id, tenant_id=request.principal.tenant_id)
        if header is None:
            raise LinktoolsAIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(request.principal, AuthorizationAction.TASK_CANCEL, header)
        operation_digest = canonical_sha256(
            {
                "action": "task.cancel",
                "tenant_id": request.principal.tenant_id,
                "principal_id": request.principal.principal_id,
                "graph_id": graph_id,
                "force": request.force,
            }
        )
        operation = await self._persistence.operations.get(
            request.cancel_request_id,
            tenant_id=request.principal.tenant_id,
        )
        if operation is not None:
            if operation.request_digest != operation_digest:
                raise LinktoolsAIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if operation.status in {OperationStatus.SUCCEEDED, OperationStatus.CANCELLED}:
                view = await self._persistence.tasks.get_plan(graph_id, tenant_id=request.principal.tenant_id)
                if view is None:
                    raise LinktoolsAIError(ErrorCode.STORAGE_NOT_FOUND)
                return view
        else:
            now = datetime.now(timezone.utc)
            operation = await self._persistence.operations.create(
                OperationLedgerRecord(
                    request.cancel_request_id,
                    request.principal.tenant_id,
                    ResourceKind.TASK_GRAPH,
                    graph_id,
                    None,
                    OperationKind.TASK_CANCEL,
                    OperationStatus.PENDING,
                    operation_digest,
                    None,
                    None,
                    None,
                    True,
                    await self._persistence.operations.next_sequence(
                        ResourceKind.TASK_GRAPH,
                        graph_id,
                        tenant_id=request.principal.tenant_id,
                    ),
                    now,
                    now,
                )
            )
        view = await self._persistence.tasks.cancel_plan(graph_id, tenant_id=request.principal.tenant_id)
        if self._workflow_gateway is not None:
            await self._workflow_gateway.cancel_task_graph(graph_id, request.cancel_request_id)
        await self._persistence.operations.compare_and_swap(
            operation.operation_id,
            tenant_id=request.principal.tenant_id,
            expected_status=operation.status,
            next_record=replace(
                operation,
                status=OperationStatus.SUCCEEDED,
                result_ref=graph_id,
                result_digest=canonical_sha256({"graph_id": graph_id, "status": view.status.value}),
                updated_at=datetime.now(timezone.utc),
            ),
        )
        _logger.info("task graph cancelled: graph=%s tenant=%s", graph_id, request.principal.tenant_id)
        return view


__all__ = ["DefaultTaskService"]
