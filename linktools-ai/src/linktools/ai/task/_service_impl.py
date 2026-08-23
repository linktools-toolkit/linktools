#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-backed Task API independent of Runtime composition."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
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
    idempotency_key_digest,
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


class _TaskReleaseCallback(Protocol):
    async def __call__(self, graph_id: str, *, tenant_id: str) -> None: ...


class _LocalTaskWaiter(Protocol):
    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool: ...

    async def wait_graph_activity(self, graph_id: str, *, tenant_id: str) -> None: ...


async def _no_release_terminal(graph_id: str, *, tenant_id: str) -> None:
    del graph_id, tenant_id


@dataclass
class _TaskHandoffState:
    active_consumers: int = 0
    release_requested: bool = False
    release_in_progress: bool = False


class _TaskRepository(Protocol):
    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None: ...
    async def create_graph(self, graph: TaskGraph, *, tenant_id: str) -> TaskGraphView: ...
    async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView | None: ...
    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
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
        *,
        release_terminal: _TaskReleaseCallback | None = None,
        local_waiter: "_LocalTaskWaiter | None" = None,
    ) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._launcher = launcher
        self._release_terminal = release_terminal or _no_release_terminal
        self._local_waiter = local_waiter
        self._handoff_states: dict[tuple[str, str], _TaskHandoffState] = {}
        self._handoff_condition = asyncio.Condition()

    async def run_graph(self, request: TaskGraphRequest) -> TaskGraphResult:
        return await self._run_graph(request, release_terminal=True)

    async def _run_graph(self, request: TaskGraphRequest, *, release_terminal: bool) -> TaskGraphResult:
        if self._launcher is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        graph_id = request.graph.graph_id
        tenant_id = request.principal.tenant_id
        async with self._graph_consumer(graph_id, tenant_id):
            await self._authorization.authorize(
                request.principal,
                AuthorizationAction.TASK_RUN,
                ResourceRef(ResourceKind.TASK_GRAPH, graph_id, tenant_id),
            )
            digest = _graph_digest(request)
            operation_id = idempotency_key_digest(request.idempotency_key)
            claimed, operation = await self._claim_operation(
                operation_id=operation_id,
                tenant_id=tenant_id,
                graph_id=graph_id,
                kind=OperationKind.TASK_NODE,
                request_digest=digest,
            )
            if not claimed:
                view = await self._persistence.tasks.get_graph(graph_id, tenant_id=tenant_id)
                if view is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if not _terminal(view.status):
                    await self._launcher.start(request)
                result = await self._result(view, tenant_id)
                if release_terminal and _terminal(result.status):
                    await self._request_graph_release(graph_id, tenant_id)
                return result
            created = False
            try:
                view = await self._persistence.tasks.create_graph(request.graph, tenant_id=tenant_id)
                created = True
                await self._launcher.start(request)
            except asyncio.CancelledError:
                raise
            except AIError as error:
                if created:
                    await self._abort_plan(request)
                current = await self._record_failure(operation, tenant_id, error.code.value)
                if current.status is OperationStatus.SUCCEEDED:
                    result = await self._replay_result(graph_id, tenant_id)
                    if release_terminal and _terminal(result.status):
                        await self._request_graph_release(graph_id, tenant_id)
                    return result
                raise
            except Exception as error:
                if created:
                    await self._abort_plan(request)
                current = await self._record_failure(
                    operation,
                    tenant_id,
                    ErrorCode.INTERNAL_ERROR.value,
                )
                if current.status is OperationStatus.SUCCEEDED:
                    result = await self._replay_result(graph_id, tenant_id)
                    if release_terminal and _terminal(result.status):
                        await self._request_graph_release(graph_id, tenant_id)
                    return result
                raise AIError(
                    ErrorCode.INTERNAL_ERROR,
                    safe_details={"phase": "task_graph_start"},
                ) from error
            current = await self._record_success(operation, tenant_id, view)
            if current.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            _logger.info("task graph submitted: tenant=%s graph=%s", tenant_id, graph_id)
            result = await self._result(view, tenant_id)
            if release_terminal and _terminal(result.status):
                await self._request_graph_release(graph_id, tenant_id)
            return result

    async def run_graph_and_wait(self, request: TaskGraphRequest, *, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        submitted = await self._run_graph(request, release_terminal=False)
        return await self.wait_graph(submitted.graph_id, principal=request.principal, timeout_seconds=timeout_seconds)

    async def inspect_graph(self, graph_id: str, *, principal: Principal) -> TaskGraphView:
        async with self._graph_consumer(graph_id, principal.tenant_id):
            header = await self._persistence.tasks.get_header(graph_id, tenant_id=principal.tenant_id)
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(principal, AuthorizationAction.TASK_READ, header)
            return await self._persistence.tasks.reconcile_graph(graph_id, tenant_id=principal.tenant_id)

    async def wait_graph(self, graph_id: str, *, principal: Principal, timeout_seconds: "float | None" = None) -> TaskGraphResult:
        if timeout_seconds is not None and timeout_seconds < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        async with self._graph_consumer(graph_id, principal.tenant_id):
            header = await self._persistence.tasks.get_header(graph_id, tenant_id=principal.tenant_id)
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await self._authorization.authorize(principal, AuthorizationAction.TASK_READ, header)

            async def poll() -> TaskGraphResult:
                while True:
                    view = await self._persistence.tasks.reconcile_graph(
                        graph_id,
                        tenant_id=principal.tenant_id,
                    )
                    if _terminal(view.status):
                        result = await self._result(view, principal.tenant_id)
                        await self._request_graph_release(graph_id, principal.tenant_id)
                        return result
                    waiter = self._local_waiter
                    if waiter is not None and waiter.owns_graph(
                        graph_id,
                        tenant_id=principal.tenant_id,
                    ):
                        await waiter.wait_graph_activity(
                            graph_id,
                            tenant_id=principal.tenant_id,
                        )
                    else:
                        await asyncio.sleep(1.0)

            try:
                return await asyncio.wait_for(poll(), timeout_seconds)
            except asyncio.TimeoutError as error:
                raise AIError(ErrorCode.TASK_WAIT_TIMEOUT) from error

    async def cancel_graph(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        async with self._graph_consumer(graph_id, request.principal.tenant_id):
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
            finalizer = asyncio.create_task(
                self._cancel_finalizer(
                    graph_id,
                    request,
                    idempotency_key_digest(request.idempotency_key),
                    request_digest,
                )
            )
            caller_cancellation: "asyncio.CancelledError | None" = None
            finalizer_error: "BaseException | None" = None
            result: "TaskGraphView | None" = None
            while not finalizer.done():
                try:
                    result = await asyncio.shield(finalizer)
                except asyncio.CancelledError as error:
                    caller_cancellation = caller_cancellation or error
                    continue
                except BaseException as error:
                    finalizer_error = error
                    break
            if finalizer.done() and result is None:
                try:
                    result = finalizer.result()
                except BaseException as error:
                    finalizer_error = error
            if caller_cancellation is not None:
                if finalizer_error is not None:
                    _logger.warning(
                        "task graph cancel finalizer failed after caller cancellation: graph=%s error=%s",
                        graph_id,
                        type(finalizer_error).__name__,
                    )
                raise caller_cancellation
            if finalizer_error is not None:
                raise finalizer_error
            if result is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._request_graph_release(graph_id, request.principal.tenant_id)
            return result

    @asynccontextmanager
    async def _graph_consumer(self, graph_id: str, tenant_id: str):
        key = (tenant_id, graph_id)
        async with self._handoff_condition:
            while True:
                state = self._handoff_states.get(key)
                if state is None:
                    state = _TaskHandoffState()
                    self._handoff_states[key] = state
                if not state.release_in_progress:
                    state.active_consumers += 1
                    break
                await self._handoff_condition.wait()
        cleanup_owner = False
        try:
            yield
        finally:
            async with self._handoff_condition:
                state.active_consumers -= 1
                if state.active_consumers < 0:
                    raise RuntimeError("task graph consumer count became negative")
                if state.active_consumers == 0:
                    if state.release_requested and not state.release_in_progress:
                        state.release_in_progress = True
                        cleanup_owner = True
                    elif not state.release_requested and self._handoff_states.get(key) is state:
                        self._handoff_states.pop(key, None)
                self._handoff_condition.notify_all()
            if cleanup_owner:
                cleanup_succeeded = False
                cleanup_error: BaseException | None = None
                try:
                    await self._release_terminal(graph_id, tenant_id=tenant_id)
                    cleanup_succeeded = True
                except BaseException as error:
                    cleanup_error = error
                    if isinstance(error, Exception):
                        _logger.error("task graph transient handoff cleanup failed: graph=%s", graph_id, exc_info=environ.debug)
                async with self._handoff_condition:
                    if self._handoff_states.get(key) is state:
                        if cleanup_succeeded and state.active_consumers == 0:
                            self._handoff_states.pop(key, None)
                        else:
                            state.release_in_progress = False
                            state.release_requested = True
                    self._handoff_condition.notify_all()
                if cleanup_error is not None and not isinstance(cleanup_error, Exception):
                    raise cleanup_error

    async def _request_graph_release(self, graph_id: str, tenant_id: str) -> None:
        async with self._handoff_condition:
            state = self._handoff_states.get((tenant_id, graph_id))
            if state is None:
                raise RuntimeError("task graph release requested without consumer")
            state.release_requested = True
            self._handoff_condition.notify_all()

    async def _claim_operation(self, *, operation_id: str, tenant_id: str, graph_id: str, kind: OperationKind, request_digest: str) -> "tuple[bool, OperationLedgerRecord]":
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
        while True:
            operation = await self._persistence.operations.get(operation_id, tenant_id=tenant_id)
            if operation is None:
                now = datetime.now(timezone.utc)
                try:
                    operation = await self._persistence.operations.append(
                        OperationLedgerInput(
                            operation_id,
                            tenant_id,
                            ResourceKind.TASK_GRAPH,
                            graph_id,
                            None,
                            OperationKind.TASK_CANCEL,
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
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    continue
            if operation.request_digest != request_digest:
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if operation.status is OperationStatus.PENDING:
                running = replace(operation, status=OperationStatus.RUNNING, updated_at=datetime.now(timezone.utc))
                try:
                    claimed = await self._persistence.operations.compare_and_swap(
                        operation_id,
                        tenant_id=tenant_id,
                        expected_status=OperationStatus.PENDING,
                        next_record=running,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    continue
                return True, claimed
            if operation.status is OperationStatus.SUCCEEDED:
                return False, operation
            if operation.status is OperationStatus.FAILED:
                raise _stable_operation_error(operation.error_code)
            if operation.status in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}:
                return False, operation
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _cancel_finalizer(
        self,
        graph_id: str,
        request: CancelGraphRequest,
        operation_id: str,
        request_digest: str,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        claimed, operation = await self._claim_cancel_operation(operation_id, tenant_id, graph_id, request_digest)
        try:
            view = await self._persistence.tasks.get_graph(graph_id, tenant_id=tenant_id)
        except BaseException as error:
            if claimed or operation.status in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}:
                return await self._settle_cancel_error(operation, graph_id, request, error)
            raise
        if view is None:
            if claimed or operation.status in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}:
                return await self._settle_cancel_error(operation, graph_id, request, AIError(ErrorCode.STORAGE_INTEGRITY_ERROR))
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if claimed:
            try:
                if not _terminal(view.status):
                    view = await self._persistence.tasks.cancel_graph(graph_id, tenant_id=tenant_id)
                view = await self._persistence.tasks.get_graph(graph_id, tenant_id=tenant_id)
                if view is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if not _terminal(view.status):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            except BaseException as error:
                return await self._settle_cancel_error(operation, graph_id, request, error)
        elif operation.status in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}:
            if not _terminal(view.status):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        elif not _terminal(view.status):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not _terminal(view.status):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (claimed or operation.status in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}) and view.status is TaskStatus.CANCELLED:
            await self._cleanup_cancelled_graph(graph_id, request)
        if claimed or operation.status in {OperationStatus.RUNNING, OperationStatus.EFFECT_UNKNOWN}:
            current = await self._record_success(operation, tenant_id, view, expected_status=operation.status)
            if current.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        _logger.info("task graph cancel settled: tenant=%s graph=%s status=%s", tenant_id, graph_id, view.status.value)
        return view

    async def _settle_cancel_error(
        self,
        operation: OperationLedgerRecord,
        graph_id: str,
        request: CancelGraphRequest,
        error: BaseException,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        try:
            view = await self._persistence.tasks.get_graph(graph_id, tenant_id=tenant_id)
        except Exception as reload_error:
            await self._record_effect_unknown(operation, tenant_id)
            if isinstance(error, AIError):
                raise error
            raise AIError(
                ErrorCode.INTERNAL_ERROR,
                safe_details={"phase": "task_cancel_readback"},
            ) from reload_error
        if view is None:
            await self._record_failure(operation, tenant_id, ErrorCode.STORAGE_INTEGRITY_ERROR.value)
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if _terminal(view.status):
            if view.status is TaskStatus.CANCELLED:
                await self._cleanup_cancelled_graph(graph_id, request)
            current = await self._record_success(operation, tenant_id, view, expected_status=operation.status)
            if current.status is not OperationStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return view
        if isinstance(error, AIError):
            await self._record_failure(operation, tenant_id, error.code.value)
            raise error
        await self._record_effect_unknown(operation, tenant_id)
        raise AIError(
            ErrorCode.INTERNAL_ERROR,
            safe_details={"phase": "task_cancel"},
        ) from error

    async def _cleanup_cancelled_graph(self, graph_id: str, request: CancelGraphRequest) -> None:
        if self._launcher is None:
            return
        try:
            await self._launcher.cancel(graph_id, request)
        except Exception as error:
            _logger.warning("task launcher cleanup failed after durable cancel: graph=%s error=%s", graph_id, type(error).__name__)

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
            if current.status is OperationStatus.FAILED:
                raise _stable_operation_error(current.error_code)
            raise

    async def _record_failure(self, operation: OperationLedgerRecord, tenant_id: str, error_code: str) -> OperationLedgerRecord:
        failed = replace(operation, status=OperationStatus.FAILED, error_code=error_code, updated_at=datetime.now(timezone.utc))
        try:
            return await self._persistence.operations.compare_and_swap(operation.operation_id, tenant_id=tenant_id, expected_status=operation.status, next_record=failed)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status is OperationStatus.FAILED:
                return current
            raise

    async def _record_effect_unknown(self, operation: OperationLedgerRecord, tenant_id: str) -> OperationLedgerRecord:
        if operation.status is OperationStatus.EFFECT_UNKNOWN:
            return operation
        unknown = replace(operation, status=OperationStatus.EFFECT_UNKNOWN, error_code=None, updated_at=datetime.now(timezone.utc))
        try:
            return await self._persistence.operations.compare_and_swap(
                operation.operation_id,
                tenant_id=tenant_id,
                expected_status=operation.status,
                next_record=unknown,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._persistence.operations.get(operation.operation_id, tenant_id=tenant_id)
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current.status in {OperationStatus.EFFECT_UNKNOWN, OperationStatus.SUCCEEDED}:
                return current
            raise

    async def _abort_plan(self, request: TaskGraphRequest) -> None:
        try:
            await self._persistence.tasks.cancel_graph(request.graph.graph_id, tenant_id=request.principal.tenant_id)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_NOT_FOUND:
                _logger.warning("failed to close unlaunched task graph: graph=%s error=%s", request.graph.graph_id, error.code.value)

    async def _replay_result(self, graph_id: str, tenant_id: str) -> TaskGraphResult:
        view = await self._persistence.tasks.get_graph(graph_id, tenant_id=tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._result(view, tenant_id)

    async def _result(self, view: TaskGraphView, tenant_id: str) -> TaskGraphResult:
        nodes = await self._persistence.tasks.list_nodes(view.graph_id, tenant_id=tenant_id)
        results = tuple(TaskNodeResult(node.node_id, node.status, node.result_digest, node.execution_id, node.error_code, node.error_digest) for node in nodes)
        execution_ids = tuple(item.execution_id for item in results if item.status is TaskStatus.SUCCEEDED and item.execution_id is not None)
        return TaskGraphResult(view.graph_id, view.status, execution_ids, results)


def _graph_digest(request: TaskGraphRequest) -> str:
    return canonical_sha256(
        {
            "principal": principal_identity_payload(request.principal),
            "graph_id": request.graph.graph_id,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "dependencies": sorted(node.dependencies),
                    "input": node.input,
                    "budget_cost": node.budget_cost,
                }
                for node in sorted(request.graph.nodes, key=lambda item: item.node_id)
            ],
            "limits": {
                "max_nodes": request.limits.max_nodes,
                "max_depth": request.limits.max_depth,
                "max_budget": request.limits.max_budget,
                "max_concurrency": request.limits.max_concurrency,
            },
        }
    )


def _terminal(status: TaskStatus) -> bool:
    return status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}


def _stable_operation_error(error_code: "str | None") -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["DefaultTaskService", "TaskPersistence"]
