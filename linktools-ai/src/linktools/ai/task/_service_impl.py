#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed Task API independent of Runtime composition."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    canonical_sha256,
    idempotency_key_hash,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ._graph import (
    CancelGraphRequest,
    TaskGraph,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphView,
    TaskNodeResult,
)
from ._service import TaskApi, TaskGraphLauncher

_logger = environ.get_logger("ai.task.service")


class _TaskRepository(Protocol):
    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create_plan(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView: ...
    async def get_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None: ...
    async def reconcile_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
    async def cancel_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> tuple[object, ...]: ...


class _OperationRepository(Protocol):
    async def append(self, record: OperationLedgerInput) -> OperationLedgerRecord: ...
    async def get(self, operation_id: str, *, tenant_id: str) -> OperationLedgerRecord | None: ...
    async def compare_and_swap(
        self,
        operation_id: str,
        *,
        tenant_id: str,
        expected_status: OperationStatus,
        next_record: OperationLedgerRecord,
    ) -> OperationLedgerRecord: ...


class TaskPersistence(Protocol):
    tasks: _TaskRepository
    operations: _OperationRepository


class DefaultTaskService(TaskApi):
    """Own durable Task submission, observation and cancellation semantics."""

    def __init__(
        self,
        persistence: TaskPersistence,
        authorization: AuthorizationPolicy,
        launcher: TaskGraphLauncher | None = None,
    ) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher

    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.TASK_RUN,
            ResourceRef(ResourceKind.TASK_GRAPH, request.graph.graph_id, request.principal.tenant_id),
        )
        digest = _graph_digest(request)
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
            if not _terminal(view.status):
                await self._launcher.start(request)
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
                await self._abort_plan(request)
            current = await self._record_failure(operation, request.principal.tenant_id, error.code.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_result(request.graph.graph_id, request.principal.tenant_id)
            raise
        except Exception:
            if created:
                await self._abort_plan(request)
            current = await self._record_failure(operation, request.principal.tenant_id, ErrorCode.STORAGE_UNAVAILABLE.value)
            if current.status is OperationStatus.SUCCEEDED:
                return await self._replay_result(request.graph.graph_id, request.principal.tenant_id)
            raise
        current = await self._record_success(operation, request.principal.tenant_id, view)
        if current.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info("task graph submitted: tenant=%s graph=%s", request.principal.tenant_id, view.graph_id)
        return await self._result(view, request.principal.tenant_id)

    async def run_graph_and_wait(self, request: TaskGraphRequest, *, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        handle = await self.run_graph(request)
        return await self.wait_graph(handle.graph_id, principal=request.principal, timeout_seconds=timeout_seconds)

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        header = await self._persistence.tasks.get_header(graph_id, tenant_id=principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(principal, AuthorizationAction.TASK_READ, header)
        return await self._persistence.tasks.reconcile_plan(graph_id, tenant_id=principal.tenant_id)

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
                if _terminal(view.status):
                    return await self._result(view, principal.tenant_id)
                await asyncio.sleep(0.05)

        try:
            return await asyncio.wait_for(poll(), timeout_seconds)
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE, "task graph wait timed out") from error

    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        header = await self._persistence.tasks.get_header(graph_id, tenant_id=request.principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(request.principal, AuthorizationAction.TASK_CANCEL, header)
        request_digest = canonical_sha256(
            {
                "action": "task.cancel",
                "principal": principal_identity_payload(request.principal),
                "graph_id": graph_id,
                "force": request.force,
            }
        )
        claimed, operation = await self._claim_cancel_operation(
            idempotency_key_hash(request.cancel_request_id),
            request.principal.tenant_id,
            graph_id,
            request_digest,
        )
        finalizer = asyncio.create_task(self._cancel_finalizer(graph_id, request, operation, claimed))
        try:
            return await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            await asyncio.shield(finalizer)
            raise

    async def _claim_operation(self, *, operation_id: str, tenant_id: str, graph_id: str, kind: OperationKind, request_digest: str) -> tuple[bool, OperationLedgerRecord]:
        operation = await self._persistence.operations.get(operation_id, tenant_id=tenant_id)
        if operation is None:
            now = datetime.now(timezone.utc)
            operation = await self._persistence.operations.append(OperationLedgerInput(operation_id, tenant_id, ResourceKind.TASK_GRAPH, graph_id, None, kind, OperationStatus.PENDING, request_digest, None, None, None, True, now, now))
        if operation.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if operation.status is OperationStatus.PENDING:
            running = replace(operation, status=OperationStatus.RUNNING, updated_at=datetime.now(timezone.utc))
            try:
                return True, await self._persistence.operations.compare_and_swap(operation_id, tenant_id=tenant_id, expected_status=OperationStatus.PENDING, next_record=running)
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                operation = await self._persistence.operations.get(operation_id, tenant_id=tenant_id)
                if operation is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if operation.status is OperationStatus.SUCCEEDED:
            return False, operation
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _claim_cancel_operation(self, operation_id: str, tenant_id: str, graph_id: str, request_digest: str) -> tuple[bool, OperationLedgerRecord]:
        operation = await self._persistence.operations.get(operation_id, tenant_id=tenant_id)
        if operation is None:
            now = datetime.now(timezone.utc)
            operation = await self._persistence.operations.append(OperationLedgerInput(operation_id, tenant_id, ResourceKind.TASK_GRAPH, graph_id, None, OperationKind.TASK_CANCEL, OperationStatus.PENDING, request_digest, None, None, None, True, now, now))
        if operation.request_digest != request_digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        if operation.status is OperationStatus.SUCCEEDED:
            return False, operation
        if operation.status is OperationStatus.FAILED:
            raise _stable_operation_error(operation.error_code)
        if operation.status not in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        view = await self._persistence.tasks.get_plan(graph_id, tenant_id=tenant_id)
        if view is None or not _terminal(view.status):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return False, operation
        running = replace(operation, status=OperationStatus.RUNNING, updated_at=datetime.now(timezone.utc))
        return True, await self._persistence.operations.compare_and_swap(operation_id, tenant_id=tenant_id, expected_status=OperationStatus.PENDING, next_record=running)

    async def _cancel_finalizer(self, graph_id: str, request: CancelGraphRequest, operation: OperationLedgerRecord, claimed: bool) -> TaskGraphView:
        before = await self._persistence.tasks.get_plan(graph_id, tenant_id=request.principal.tenant_id)
        if before is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        view = await self._persistence.tasks.cancel_plan(graph_id, tenant_id=request.principal.tenant_id) if claimed else before
        if claimed and self._launcher is not None and before.status not in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.BLOCKED}:
            try:
                await self._launcher.cancel(graph_id, request)
            except Exception as error:
                _logger.warning("task launcher cleanup failed after durable cancel: graph=%s error=%s", graph_id, type(error).__name__)
        current = await self._record_success(operation, request.principal.tenant_id, view, expected_status=operation.status)
        if current.status is not OperationStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info("task graph cancel settled: tenant=%s graph=%s", request.principal.tenant_id, graph_id)
        return view

    async def _record_success(self, operation: OperationLedgerRecord, tenant_id: str, view: TaskGraphView, *, expected_status: OperationStatus = OperationStatus.RUNNING) -> OperationLedgerRecord:
        completed = replace(operation, status=OperationStatus.SUCCEEDED, result_ref=view.graph_id, result_digest=canonical_sha256({"graph_id": view.graph_id, "status": view.status.value}), error_code=None, updated_at=datetime.now(timezone.utc))
        try:
            return await self._persistence.operations.compare_and_swap(operation.operation_id, tenant_id=tenant_id, expected_status=expected_status, next_record=completed)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is OperationStatus.SUCCEEDED:
                return current
            raise

    async def _record_failure(self, operation: OperationLedgerRecord, tenant_id: str, error_code: str) -> OperationLedgerRecord:
        failed = replace(operation, status=OperationStatus.FAILED, error_code=error_code, updated_at=datetime.now(timezone.utc))
        return await self._persistence.operations.compare_and_swap(operation.operation_id, tenant_id=tenant_id, expected_status=OperationStatus.RUNNING, next_record=failed)

    async def _abort_plan(self, request: TaskGraphRequest) -> None:
        try:
            await self._persistence.tasks.cancel_plan(request.graph.graph_id, tenant_id=request.principal.tenant_id)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_NOT_FOUND:
                _logger.warning("failed to close unlaunched task graph: graph=%s error=%s", request.graph.graph_id, error.code.value)

    async def _replay_result(self, graph_id: str, tenant_id: str) -> TaskGraphResult:
        view = await self._persistence.tasks.get_plan(graph_id, tenant_id=tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._result(view, tenant_id)

    async def _result(self, view: TaskGraphView, tenant_id: str) -> TaskGraphResult:
        nodes = await self._persistence.tasks.list_nodes(view.graph_id, tenant_id=tenant_id)
        results = tuple(TaskNodeResult(node.task_id, node.status, node.result_digest, node.execution_id, node.error_code, node.error_digest) for node in nodes)
        execution_ids = tuple(item.execution_id for item in results if item.status is TaskStatus.SUCCEEDED and item.execution_id is not None)
        return TaskGraphResult(view.graph_id, view.status, execution_ids, results)


def _graph_digest(request: TaskGraphRequest) -> str:
    return canonical_sha256(
        {
            "principal": principal_identity_payload(request.principal),
            "graph_id": request.graph.graph_id,
            "nodes": [{"task_id": node.task_id, "dependencies": sorted(node.dependencies), "binding_digest": node.binding_digest, "budget_cost": node.budget_cost} for node in sorted(request.graph.nodes, key=lambda item: item.task_id)],
            "limits": {"max_nodes": request.limits.max_nodes, "max_depth": request.limits.max_depth, "max_budget": request.limits.max_budget, "max_concurrency": request.limits.max_concurrency},
        }
    )


def _terminal(status: TaskStatus) -> bool:
    return status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["DefaultTaskService", "TaskPersistence"]
