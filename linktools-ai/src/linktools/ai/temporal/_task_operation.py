#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-backed Temporal TaskGraph activity operation."""

import re
from collections.abc import Mapping
from datetime import datetime, timezone

from linktools.core import environ

from ..core import Principal, TaskStatus, canonical_sha256
from ..errors import AIError, ErrorCode
from ..runtime import RuntimeObjectKeyFactory, RuntimeTaskNodeRunner
from ..runtime.state import TaskState
from ..storage import ObjectStore
from ..task import TaskDependencyResult, TaskLease, TaskNode, TaskNodeView
from ._request import put_execution_request, read_task_request
from .workflow import ExecutionWorkflowInput, ExecutionWorkflowResult, TaskWorkflowInput

_logger = environ.get_logger("ai.temporal.task_operation")
_LEASE_SECONDS = 60


class _RuntimeTaskOperation:
    def __init__(
        self,
        *,
        task_state: TaskState,
        runner: RuntimeTaskNodeRunner,
        request_store: ObjectStore,
        namespace: str,
    ) -> None:
        self._repository = task_state.tasks
        self._runner = runner
        self._request_store = request_store
        self._request_keys = RuntimeObjectKeyFactory(namespace)

    async def prepare(
        self,
        request: TaskWorkflowInput,
        node_id: str,
        dependency_results: Mapping[str, TaskDependencyResult],
    ) -> "tuple[TaskLease, ExecutionWorkflowInput] | None":
        stored = await self._load_request(request)
        node = _find_node(stored.graph.nodes, node_id)
        await self._repository.reconcile_graph(
            request.graph_id,
            tenant_id=request.tenant_id,
        )
        nodes = await self._repository.list_nodes(
            request.graph_id,
            tenant_id=request.tenant_id,
        )
        current = _find_node_view(nodes, node_id)
        if current.status is TaskStatus.FAILED:
            _validate_prepare_failure(current)
            _logger.info(
                "replaying failed Temporal task node: graph=%s node=%s",
                request.graph_id,
                node_id,
            )
            return None
        owner = _task_owner(request, node_id)
        lease = await self._acquire_lease(current, request, owner)
        try:
            binding_digest, execution_request = await self._runner.prepare(
                node,
                graph_id=request.graph_id,
                principal=stored.principal,
                dependency_results=dependency_results,
            )
        except AIError as error:
            if error.retryable:
                raise
            await self._fail_prepare(lease, request.tenant_id, error)
            return None
        request_ref = await put_execution_request(
            self._request_store,
            self._request_keys,
            execution_request,
        )
        operation_id = canonical_sha256(
            {
                "tenant_id": request.tenant_id,
                "graph_id": request.graph_id,
                "node_id": node_id,
                "fence": lease.fence,
            }
        )
        child = ExecutionWorkflowInput(
            execution_id=f"{request.graph_id}:{node_id}",
            tenant_id=request.tenant_id,
            binding_digest=binding_digest,
            bundle_digest=binding_digest,
            request_ref=request_ref,
            worker_build=request.worker_build,
            owner=lease.owner,
            fence=lease.fence,
            operation_id=operation_id,
        )
        _logger.info(
            "Temporal task node prepared: graph=%s node=%s fence=%s request_ref=%s",
            request.graph_id,
            node_id,
            lease.fence,
            request_ref,
        )
        return lease, child

    async def renew(self, lease: TaskLease) -> TaskLease:
        return await self._repository.renew(
            lease,
            tenant_id=lease.tenant_id,
            lease_seconds=_LEASE_SECONDS,
        )

    async def settle(
        self,
        request: TaskWorkflowInput,
        lease: TaskLease,
        result: ExecutionWorkflowResult,
    ) -> TaskDependencyResult | None:
        stored = await self._load_request(request)
        if lease.graph_id != request.graph_id or lease.tenant_id != request.tenant_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _find_node(stored.graph.nodes, lease.node_id)
        expected_execution_id = f"{request.graph_id}:{lease.node_id}"
        if result.execution_id != expected_execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        nodes = await self._repository.list_nodes(
            request.graph_id,
            tenant_id=request.tenant_id,
        )
        current = _find_node_view(nodes, lease.node_id)
        if current.status is not TaskStatus.RUNNING:
            return await self._settle_terminal(
                request,
                stored.principal,
                current,
                expected_execution_id,
                result,
            )
        if current.owner != lease.owner or current.fence != lease.fence:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        if result.status == "SUCCEEDED":
            node_result = await self._runner.result(
                result.execution_id,
                principal=stored.principal,
            )
            try:
                await self._repository.complete(
                    lease,
                    tenant_id=request.tenant_id,
                    execution_id=node_result.execution_id,
                    result_digest=node_result.result_digest,
                )
            except AIError as error:
                if error.code is ErrorCode.TASK_FENCE_STALE:
                    return await self._recover_stale_settle(
                        request,
                        stored.principal,
                        lease,
                        expected_execution_id,
                        result,
                        error,
                    )
                raise
            await self._repository.reconcile_graph(
                request.graph_id,
                tenant_id=request.tenant_id,
            )
            return _dependency_result(
                node_result.result_digest,
                node_result.execution_id,
            )
        if result.status in {"FAILED", "CANCELLED"}:
            try:
                await self._repository.fail(
                    lease,
                    tenant_id=request.tenant_id,
                    error_code=ErrorCode.EXECUTION_FAILED.value,
                    error_digest=canonical_sha256(
                        {"type": "ExecutionResult", "code": result.status}
                    ),
                )
            except AIError as error:
                if error.code is ErrorCode.TASK_FENCE_STALE:
                    return await self._recover_stale_settle(
                        request,
                        stored.principal,
                        lease,
                        expected_execution_id,
                        result,
                        error,
                    )
                raise
            await self._repository.reconcile_graph(
                request.graph_id,
                tenant_id=request.tenant_id,
            )
            return None
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _recover_stale_settle(
        self,
        request: TaskWorkflowInput,
        principal: Principal,
        lease: TaskLease,
        expected_execution_id: str,
        result: ExecutionWorkflowResult,
        stale_error: AIError,
    ) -> TaskDependencyResult | None:
        nodes = await self._repository.list_nodes(
            request.graph_id,
            tenant_id=request.tenant_id,
        )
        current = _find_node_view(nodes, lease.node_id)
        if current.status is TaskStatus.RUNNING:
            raise stale_error
        if current.status not in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._settle_terminal(
            request,
            principal,
            current,
            expected_execution_id,
            result,
        )

    async def _settle_terminal(
        self,
        request: TaskWorkflowInput,
        principal: Principal,
        current: TaskNodeView,
        expected_execution_id: str,
        result: ExecutionWorkflowResult,
    ) -> TaskDependencyResult | None:
        if current.status is TaskStatus.SUCCEEDED:
            if (
                current.owner is not None
                or current.lease_expires_at is not None
                or current.execution_id != expected_execution_id
                or not current.result_digest
                or current.error_code is not None
                or current.error_digest is not None
                or result.status != "SUCCEEDED"
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            node_result = await self._runner.result(
                current.execution_id,
                principal=principal,
            )
            if (
                node_result.execution_id != current.execution_id
                or node_result.result_digest != current.result_digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._repository.reconcile_graph(
                request.graph_id,
                tenant_id=request.tenant_id,
            )
            _logger.info(
                "replayed succeeded Temporal task node: graph=%s node=%s",
                request.graph_id,
                current.node_id,
            )
            return _dependency_result(current.result_digest, current.execution_id)
        if current.status is TaskStatus.FAILED:
            expected_error_digest = canonical_sha256(
                {"type": "ExecutionResult", "code": result.status}
            )
            if (
                current.owner is not None
                or current.lease_expires_at is not None
                or current.result_digest is not None
                or current.error_code != ErrorCode.EXECUTION_FAILED.value
                or result.status not in {"FAILED", "CANCELLED"}
                or current.error_digest != expected_error_digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._repository.reconcile_graph(
                request.graph_id,
                tenant_id=request.tenant_id,
            )
            _logger.info(
                "replayed failed Temporal task node: graph=%s node=%s",
                request.graph_id,
                current.node_id,
            )
            return None
        if current.status is TaskStatus.CANCELLED:
            await self._repository.reconcile_graph(
                request.graph_id,
                tenant_id=request.tenant_id,
            )
            _logger.info(
                "replayed cancelled Temporal task node: graph=%s node=%s",
                request.graph_id,
                current.node_id,
            )
            return None
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def _load_request(self, request: TaskWorkflowInput):
        stored = await read_task_request(
            self._request_store,
            self._request_keys,
            tenant_id=request.tenant_id,
            request_ref=request.request_ref,
        )
        try:
            expected = TaskWorkflowInput.from_request(
                stored,
                request_ref=request.request_ref,
                worker_build=request.worker_build,
            )
        except (AIError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if expected != request:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stored

    async def _acquire_lease(
        self,
        current: TaskNodeView,
        request: TaskWorkflowInput,
        owner: str,
    ) -> TaskLease:
        now = datetime.now(timezone.utc)
        if current.status is TaskStatus.READY:
            return await self._repository.claim(
                request.graph_id,
                current.node_id,
                tenant_id=request.tenant_id,
                owner=owner,
                lease_seconds=_LEASE_SECONDS,
            )
        if (
            current.status is TaskStatus.RUNNING
            and current.owner == owner
            and current.fence > 0
            and current.lease_expires_at is not None
            and current.lease_expires_at > now
        ):
            return TaskLease(
                request.graph_id,
                current.node_id,
                request.tenant_id,
                owner,
                current.fence,
                current.lease_expires_at,
            )
        if current.status is TaskStatus.RUNNING and (
            current.lease_expires_at is None or current.lease_expires_at <= now
        ):
            return await self._repository.claim(
                request.graph_id,
                current.node_id,
                tenant_id=request.tenant_id,
                owner=owner,
                lease_seconds=_LEASE_SECONDS,
            )
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def _fail_prepare(
        self,
        lease: TaskLease,
        tenant_id: str,
        error: AIError,
    ) -> None:
        error_digest = canonical_sha256(
            {"type": type(error).__name__, "code": error.code.value}
        )
        await self._repository.fail(
            lease,
            tenant_id=tenant_id,
            error_code=error.code.value,
            error_digest=error_digest,
        )
        await self._repository.reconcile_graph(
            lease.graph_id,
            tenant_id=tenant_id,
        )
        _logger.info(
            "Temporal task node failed during preparation: graph=%s node=%s code=%s",
            lease.graph_id,
            lease.node_id,
            error.code.value,
        )


def _find_node(nodes: tuple[TaskNode, ...], node_id: str) -> TaskNode:
    for node in nodes:
        if node.node_id == node_id:
            return node
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _find_node_view(nodes: tuple[TaskNodeView, ...], node_id: str) -> TaskNodeView:
    for node in nodes:
        if node.node_id == node_id:
            return node
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_prepare_failure(current: TaskNodeView) -> None:
    if (
        current.owner is not None
        or current.lease_expires_at is not None
        or current.result_digest is not None
        or current.execution_id is not None
        or not current.error_code
        or current.error_code == ErrorCode.EXECUTION_FAILED.value
        or not _is_digest(current.error_digest)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _is_digest(value: "str | None") -> bool:
    return value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _dependency_result(
    result_digest: str,
    execution_id: "str | None",
) -> TaskDependencyResult:
    try:
        return TaskDependencyResult(result_digest, execution_id)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _task_owner(request: TaskWorkflowInput, node_id: str) -> str:
    return "temporal-" + canonical_sha256(
        {
            "tenant_id": request.tenant_id,
            "graph_id": request.graph_id,
            "node_id": node_id,
        }
    )


__all__ = ["_RuntimeTaskOperation"]
