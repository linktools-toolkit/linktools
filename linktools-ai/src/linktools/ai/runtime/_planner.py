#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed task service used by the runtime composition root."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    OperationKind,
    OperationStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    canonical_sha256,
    idempotency_key_hash,
)
from ..errors import AIError, ErrorCode
from ..task import (
    CancelGraphRequest,
    TaskGraphHandle,
    TaskGraphLauncher,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphView,
    TaskNodeResult,
)
from ._persistence import (
    OperationLedgerInput,
    OperationLedgerRecord,
    RuntimePersistence,
)
from ._services import WorkflowGateway

_logger = environ.get_logger("ai.runtime.planner")


class DefaultTaskService:
    """Validate and persist task graphs before scheduling any node."""

    def __init__(self, persistence: RuntimePersistence, authorization: AuthorizationPolicy, launcher: "TaskGraphLauncher | None" = None) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher

    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._authorization.authorize(request.principal, AuthorizationAction.TASK_RUN, ResourceRef(ResourceKind.TASK_GRAPH, request.graph.graph_id, request.principal.tenant_id))
        digest = canonical_sha256(
            {
                "graph_id": request.graph.graph_id,
                "nodes": [
                    {
                        "task_id": node.task_id,
                        "dependencies": sorted(node.dependencies),
                        "binding_digest": node.binding_digest,
                        "budget_cost": node.budget_cost,
                    }
                    for node in sorted(request.graph.nodes, key=lambda item: item.task_id)
                ],
                "limits": {
                    "max_nodes": request.limits.max_nodes,
                    "max_depth": request.limits.max_depth,
                    "max_budget": request.limits.max_budget,
                    "max_concurrency": request.limits.max_concurrency,
                },
            }
        )
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
            return await self._result(view, request.principal.tenant_id)
        created = False
        try:
            view = await self._persistence.tasks.create_plan(request.graph, tenant_id=request.principal.tenant_id)
            created = True
            await self._launcher.start(request)
        except asyncio.CancelledError:
            raise
        except AIError as error:
            if created:
                await self._abort_unlaunched_plan(request)
            current = await self._record_failure(operation, request.principal.tenant_id, error.code.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_result(request.graph.graph_id, request.principal.tenant_id)
            if current.status is OperationStatus.FAILED and current.error_code != error.code.value:
                raise _stable_operation_error(current.error_code)
            raise
        except Exception:
            if created:
                await self._abort_unlaunched_plan(request)
            current = await self._record_failure(operation, request.principal.tenant_id, ErrorCode.STORAGE_UNAVAILABLE.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_result(request.graph.graph_id, request.principal.tenant_id)
            if current.status is OperationStatus.FAILED and current.error_code != ErrorCode.STORAGE_UNAVAILABLE.value:
                raise _stable_operation_error(current.error_code)
            raise
        current = await self._record_success(operation, request.principal.tenant_id, view)
        if current.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info("task graph submitted: graph=%s tenant=%s launcher=%s", view.graph_id, request.principal.tenant_id, type(self._launcher).__name__)
        return await self._result(view, request.principal.tenant_id)

    async def _abort_unlaunched_plan(self, request: TaskGraphRequest) -> None:
        try:
            await self._persistence.tasks.cancel_plan(request.graph.graph_id, tenant_id=request.principal.tenant_id)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_NOT_FOUND:
                _logger.exception("failed to close unlaunched task graph: graph=%s", request.graph.graph_id)

    async def run_graph_and_wait(self, request: TaskGraphRequest, *, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        handle = await self.run_graph(request)
        return await self.wait_graph(handle.graph_id, principal=request.principal, timeout_seconds=timeout_seconds)

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
        return await self._result(view, tenant_id)

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        header = await self._persistence.tasks.get_header(graph_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.TASK_READ, header)
        view = await self._persistence.tasks.reconcile_plan(graph_id, tenant_id=principal.tenant_id)
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
            if self._launcher is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            await self._launcher.cancel(graph_id, request)
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

    async def wait_graph(self, graph_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async def poll() -> TaskGraphResult:
            while True:
                header = await self._persistence.tasks.get_header(graph_id, tenant_id=principal.tenant_id)
                if header is None:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                await self._authorization.authorize(principal, AuthorizationAction.TASK_READ, header)
                view = await self._persistence.tasks.reconcile_plan(graph_id, tenant_id=principal.tenant_id)
                if view.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                    return await self._result(view, principal.tenant_id)
                await asyncio.sleep(0.05)

        try:
            return await asyncio.wait_for(poll(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE, "task graph wait timed out") from error

    async def _result(self, view: TaskGraphView, tenant_id: str) -> TaskGraphResult:
        nodes = await self._persistence.tasks.list_nodes(view.graph_id, tenant_id=tenant_id)
        results = tuple(TaskNodeResult(node.task_id, node.status, node.result_digest, node.execution_id, node.error_code, node.error_digest) for node in nodes)
        execution_ids = tuple(item.execution_id for item in results if item.status is TaskStatus.SUCCEEDED and item.execution_id is not None)
        return TaskGraphResult(view.graph_id, view.status, execution_ids, results)


class WorkflowTaskGraphLauncher:
    def __init__(self, gateway: WorkflowGateway) -> None:
        self._gateway = gateway

    async def start(self, request: TaskGraphRequest) -> TaskGraphHandle:
        return await self._gateway.start_task_graph(request.graph.graph_id, request)

    async def cancel(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        return await self._gateway.cancel_task_graph(graph_id, request.cancel_request_id)


__all__ = ["DefaultTaskService", "WorkflowTaskGraphLauncher"]


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
