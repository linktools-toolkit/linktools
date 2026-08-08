#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed task service used by the runtime composition root."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from linktools.core import environ

from ..core.errors import ErrorCode, AIError
from ..core.ids import canonical_sha256, idempotency_key_hash
from ..core.principal import AuthorizationAction, AuthorizationPolicy, ResourceRef
from ..core.value import OperationKind, OperationStatus, Principal, ResourceKind
from ..task import CancelGraphRequest, TaskGraphRequest, TaskGraphResult, TaskGraphView
from .persistence import OperationLedgerInput, OperationLedgerRecord, RuntimePersistence
from .services import WorkflowGateway

_logger = environ.get_logger("ai.runtime.planner")


class DefaultTaskService:
    """Validate and persist task graphs before scheduling any node."""

    def __init__(self, persistence: RuntimePersistence, authorization: AuthorizationPolicy, workflow_gateway: "WorkflowGateway | None" = None) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._workflow_gateway = workflow_gateway

    async def run_graph(self, binding_digest: str, request: TaskGraphRequest) -> TaskGraphResult:
        await self._authorization.authorize(request.principal, AuthorizationAction.TASK_RUN, ResourceRef(ResourceKind.TASK_GRAPH, request.graph.graph_id, request.principal.tenant_id))
        digest = canonical_sha256({"graph_id": request.graph.graph_id, "nodes": [node.task_id for node in request.graph.nodes], "binding": binding_digest, "limits": {"max_nodes": request.limits.max_nodes, "max_depth": request.limits.max_depth, "max_budget": request.limits.max_budget, "max_concurrency": request.limits.max_concurrency}})
        operation_id = idempotency_key_hash(request.idempotency_key)
        claimed, operation = await self._claim_operation(
            operation_id=operation_id,
            tenant_id=request.principal.tenant_id,
            graph_id=request.graph.graph_id,
            kind=OperationKind.TASK_NODE,
            request_digest=digest,
        )
        if not claimed:
            view = await self._persistence.tasks.get_plan(request.graph.graph_id, tenant_id=request.principal.tenant_id)
            if view is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _logger.info("task graph replayed: graph=%s operation=%s status=%s", view.graph_id, operation.operation_id, operation.status.value)
            return TaskGraphResult(view.graph_id, view.status, ())
        try:
            view = await self._persistence.tasks.create_plan(request.graph, tenant_id=request.principal.tenant_id)
            if self._workflow_gateway is not None:
                await self._workflow_gateway.start_task_graph(request.graph.graph_id, request)
        except asyncio.CancelledError:
            raise
        except AIError as error:
            current = await self._record_failure(operation, request.principal.tenant_id, error.code.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_result(request.graph.graph_id, request.principal.tenant_id)
            if current.status is OperationStatus.FAILED and current.error_code != error.code.value:
                raise _stable_operation_error(current.error_code)
            raise
        except Exception:
            current = await self._record_failure(operation, request.principal.tenant_id, ErrorCode.STORAGE_UNAVAILABLE.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_result(request.graph.graph_id, request.principal.tenant_id)
            if current.status is OperationStatus.FAILED and current.error_code != ErrorCode.STORAGE_UNAVAILABLE.value:
                raise _stable_operation_error(current.error_code)
            raise
        current = await self._record_success(operation, request.principal.tenant_id, view)
        if current.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info("task graph submitted: graph=%s tenant=%s gateway=%s", view.graph_id, request.principal.tenant_id, self._workflow_gateway is not None)
        return TaskGraphResult(view.graph_id, view.status, ())

    async def _claim_operation(
        self,
        *,
        operation_id: str,
        tenant_id: str,
        graph_id: str,
        kind: OperationKind,
        request_digest: str,
    ) -> "tuple[bool, OperationLedgerRecord]":
        operation = await self._persistence.operations.get(operation_id, tenant_id=tenant_id)
        if operation is None:
            now = datetime.now(timezone.utc)
            operation = await self._persistence.operations.append(
                OperationLedgerInput(
                    operation_id,
                    tenant_id,
                    ResourceKind.TASK_GRAPH,
                    graph_id,
                    None,
                    kind,
                    OperationStatus.PENDING,
                    request_digest,
                    None,
                    None,
                    None,
                    True,
                    now,
                    now,
                )
            )
        if operation.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        for _ in range(2):
            if operation.status is OperationStatus.PENDING:
                running = replace(operation, status=OperationStatus.RUNNING, updated_at=datetime.now(timezone.utc))
                try:
                    claimed = await self._persistence.operations.compare_and_swap(
                        operation.operation_id,
                        tenant_id=tenant_id,
                        expected_status=OperationStatus.PENDING,
                        next_record=running,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    reloaded = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
                    if reloaded is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    operation = reloaded
                    continue
                _logger.info("task operation claimed: operation=%s graph=%s status=%s", claimed.operation_id, graph_id, claimed.status.value)
                return True, claimed
            if operation.status is OperationStatus.SUCCEEDED:
                return False, operation
            if operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(operation.error_code)
            if operation.status in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN, OperationStatus.CANCELLED, OperationStatus.COMPACTED}:
                raise _operation_conflict(operation)
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raise _operation_conflict(operation)

    async def _record_success(self, operation: OperationLedgerRecord, tenant_id: str, view: TaskGraphView) -> OperationLedgerRecord:
        completed = replace(
            operation,
            status=OperationStatus.SUCCEEDED,
            result_ref=view.graph_id,
            result_digest=canonical_sha256({"graph_id": view.graph_id, "status": view.status.value}),
            error_code=None,
            updated_at=datetime.now(timezone.utc),
        )
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=OperationStatus.RUNNING,
                next_record=completed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is OperationStatus.SUCCEEDED:
                return current
            if current.status is OperationStatus.FAILED:
                raise _stable_operation_error(current.error_code)
            raise

    async def _record_failure(self, operation: OperationLedgerRecord, tenant_id: str, error_code: str) -> OperationLedgerRecord:
        failed = replace(operation, status=OperationStatus.FAILED, error_code=error_code, updated_at=datetime.now(timezone.utc))
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=OperationStatus.RUNNING,
                next_record=failed,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {OperationStatus.SUCCEEDED, OperationStatus.FAILED}:
                return current
            raise

    async def _replay_result(self, graph_id: str, tenant_id: str) -> TaskGraphResult:
        view = await self._persistence.tasks.get_plan(graph_id, tenant_id=tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskGraphResult(view.graph_id, view.status, ())

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        header = await self._persistence.tasks.get_header(graph_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.TASK_READ, header)
        view = await self._persistence.tasks.get_plan(graph_id, tenant_id=principal.tenant_id)
        if view is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return view

    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        header = await self._persistence.tasks.get_header(graph_id, tenant_id=request.principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
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
        operation_id = idempotency_key_hash(request.cancel_request_id)
        claimed, operation = await self._claim_operation(
            operation_id=operation_id,
            tenant_id=request.principal.tenant_id,
            graph_id=graph_id,
            kind=OperationKind.TASK_CANCEL,
            request_digest=operation_digest,
        )
        if not claimed:
            return await self._replay_view(graph_id, request.principal.tenant_id)
        try:
            view = await self._persistence.tasks.cancel_plan(graph_id, tenant_id=request.principal.tenant_id)
            if self._workflow_gateway is not None:
                await self._workflow_gateway.cancel_task_graph(graph_id, request.cancel_request_id)
        except asyncio.CancelledError:
            raise
        except AIError as error:
            current = await self._record_failure(operation, request.principal.tenant_id, error.code.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_view(graph_id, request.principal.tenant_id)
            if current.status is OperationStatus.FAILED and current.error_code != error.code.value:
                raise _stable_operation_error(current.error_code)
            raise
        except Exception:
            current = await self._record_failure(operation, request.principal.tenant_id, ErrorCode.STORAGE_UNAVAILABLE.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_view(graph_id, request.principal.tenant_id)
            if current.status is OperationStatus.FAILED and current.error_code != ErrorCode.STORAGE_UNAVAILABLE.value:
                raise _stable_operation_error(current.error_code)
            raise
        current = await self._record_success(operation, request.principal.tenant_id, view)
        if current.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info("task graph cancelled: graph=%s tenant=%s", graph_id, request.principal.tenant_id)
        return view

    async def _replay_view(self, graph_id: str, tenant_id: str) -> TaskGraphView:
        view = await self._persistence.tasks.get_plan(graph_id, tenant_id=tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return view


__all__ = ["DefaultTaskService"]


def _operation_conflict(operation: OperationLedgerRecord) -> AIError:
    return AIError(
        ErrorCode.STORAGE_CONFLICT,
        safe_details={"operation_id": operation.operation_id, "status": operation.status.value},
    )


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(error_code)
    except ValueError:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AIError(code)
