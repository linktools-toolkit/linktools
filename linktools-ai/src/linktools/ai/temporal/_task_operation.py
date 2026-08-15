#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-backed Temporal TaskGraph activity operation."""

from collections.abc import Mapping
from datetime import datetime, timezone

from linktools.core import environ

from ..core import TaskStatus, canonical_sha256
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
    ) -> tuple[TaskLease, ExecutionWorkflowInput]:
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
        owner = _task_owner(request, node_id)
        lease = await self._acquire_lease(current, request, owner)
        try:
            binding_digest, execution_request = await self._runner.prepare(
                node,
                graph_id=request.graph_id,
                principal=stored.principal,
                dependency_results=dependency_results,
            )
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
        except BaseException as error:
            await self._fail_prepare(lease, request.tenant_id, error)
            raise
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
                    return None
                raise
            await self._repository.reconcile_graph(
                request.graph_id,
                tenant_id=request.tenant_id,
            )
            return TaskDependencyResult(
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
                    return None
                raise
            await self._repository.reconcile_graph(
                request.graph_id,
                tenant_id=request.tenant_id,
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
        error: BaseException,
    ) -> None:
        code = error.code.value if isinstance(error, AIError) else ErrorCode.TASK_NODE_FAILED.value
        error_digest = canonical_sha256(
            {"type": type(error).__name__, "code": code}
        )
        try:
            await self._repository.fail(
                lease,
                tenant_id=tenant_id,
                error_code=code,
                error_digest=error_digest,
            )
        except AIError as failure:
            if failure.code is ErrorCode.TASK_FENCE_STALE:
                return
            _logger.warning(
                "Temporal task prepare failure could not be persisted: graph=%s node=%s error=%s",
                lease.graph_id,
                lease.node_id,
                failure.code.value,
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


def _task_owner(request: TaskWorkflowInput, node_id: str) -> str:
    return "temporal-" + canonical_sha256(
        {
            "tenant_id": request.tenant_id,
            "graph_id": request.graph_id,
            "node_id": node_id,
        }
    )


__all__ = ["_RuntimeTaskOperation"]
