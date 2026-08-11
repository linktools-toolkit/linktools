#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-local TaskGraph launcher backed by TaskRepository leases."""

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

from ..core import Principal, TaskStatus, canonical_sha256
from ..errors import AIError, ErrorCode
from ._graph import (
    CancelGraphRequest,
    TaskDependencyResult,
    TaskGraphHandle,
    TaskGraphRequest,
    TaskGraphView,
    TaskNode,
)

_logger = environ.get_logger("ai.task.local")
_LEASE_SECONDS = 60

@dataclass(frozen=True, slots=True)
class TaskNodeRunResult:
    result_digest: str
    execution_id: "str | None" = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task node result identity is invalid")


class TaskNodeRunner(Protocol):
    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
    ) -> TaskNodeRunResult: ...


class TaskNodeResultVerifier(Protocol):
    async def verify(self, result: TaskNodeRunResult, *, tenant_id: str) -> None: ...


class _ExecutionStatus(Protocol):
    value: str


class _ExecutionRecord(Protocol):
    status: _ExecutionStatus
    result_digest: "str | None"


class _ExecutionRepository(Protocol):
    async def get(self, execution_id: str, *, tenant_id: str) -> "_ExecutionRecord | None": ...


@dataclass(frozen=True, slots=True)
class RuntimeTaskNodeResultVerifier:
    executions: _ExecutionRepository

    async def verify(self, result: TaskNodeRunResult, *, tenant_id: str) -> None:
        if result.execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "runtime task result requires an execution id")
        record = await self.executions.get(result.execution_id, tenant_id=tenant_id)
        if (
            record is None
            or record.status.value != "SUCCEEDED"
            or record.result_digest is None
            or record.result_digest != result.result_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "task node execution result could not be verified")


class _TaskNodeState(Protocol):
    task_id: str
    status: TaskStatus
    execution_id: "str | None"
    result_digest: "str | None"
    lease_expires_at: "datetime | None"


class _TaskLease(Protocol):
    pass


class _TaskTerminal(Protocol):
    pass


class _TaskRepository(Protocol):
    async def reconcile_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> "tuple[_TaskNodeState, ...]": ...
    async def claim(self, graph_id: str, task_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> _TaskLease: ...
    async def renew(self, lease: _TaskLease, *, tenant_id: str) -> _TaskLease: ...
    async def complete(self, lease: _TaskLease, *, tenant_id: str, execution_id: "str | None", result_digest: str) -> _TaskTerminal: ...
    async def fail(self, lease: _TaskLease, *, tenant_id: str, error_code: str, error_digest: str) -> _TaskTerminal: ...
    async def cancel_plan(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...


class LocalTaskGraphLauncher:
    """Schedule DAG nodes while keeping durable truth in TaskRepository."""

    def __init__(self, repository: _TaskRepository, runner: TaskNodeRunner, verifier: TaskNodeResultVerifier, *, owner: str) -> None:
        if not owner.strip():
            raise ValueError("task launcher owner is required")
        self._repository = repository
        self._runner = runner
        self._verifier = verifier
        self._owner = owner
        self._graphs: dict[str, asyncio.Task[None]] = {}
        self._accepting = True

    async def start(self, request: TaskGraphRequest) -> TaskGraphHandle:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        existing = self._graphs.get(request.graph.graph_id)
        if existing is None or existing.done():
            self._graphs[request.graph.graph_id] = asyncio.create_task(self._run_graph(request))
        _logger.debug("local task graph launched: graph=%s nodes=%s", request.graph.graph_id, len(request.graph.nodes))
        return TaskGraphHandle(request.graph.graph_id, f"local:{request.graph.graph_id}")

    async def cancel(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        view = await self._repository.cancel_plan(graph_id, tenant_id=request.principal.tenant_id)
        task = self._graphs.get(graph_id)
        if task is not None and not task.done():
            task.cancel()
        _logger.debug("local task graph cancelled: graph=%s", graph_id)
        return view

    async def shutdown(self) -> None:
        self._accepting = False
        tasks = tuple(self._graphs.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._graphs.clear()

    async def _run_graph(self, request: TaskGraphRequest) -> None:
        graph_id = request.graph.graph_id
        try:
            while True:
                view = await self._repository.reconcile_plan(graph_id, tenant_id=request.principal.tenant_id)
                if view.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                    return
                nodes = await self._repository.list_nodes(graph_id, tenant_id=request.principal.tenant_id)
                now = datetime.now(timezone.utc)
                ready = tuple(
                    node
                    for node in request.graph.nodes
                    if (stored := _node_by_id(nodes, node.task_id)) is not None
                    and (
                        stored.status is TaskStatus.READY
                        or (stored.status is TaskStatus.RUNNING and stored.lease_expires_at is not None and stored.lease_expires_at <= now)
                    )
                )
                if ready:
                    semaphore = asyncio.Semaphore(request.limits.max_concurrency)
                    await asyncio.gather(*(self._run_node(request, node, semaphore) for node in ready), return_exceptions=True)
                    continue
                if any(node.status is TaskStatus.RUNNING for node in nodes):
                    await asyncio.sleep(0.05)
                    continue
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.exception("local task graph scheduler failed: graph=%s", graph_id)
            try:
                await self._repository.cancel_plan(graph_id, tenant_id=request.principal.tenant_id)
            except AIError:
                _logger.exception("local task graph could not be closed after scheduler failure: graph=%s", graph_id)
        finally:
            self._graphs.pop(graph_id, None)

    async def _run_node(self, request: TaskGraphRequest, node: TaskNode, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            try:
                lease = await self._repository.claim(request.graph.graph_id, node.task_id, tenant_id=request.principal.tenant_id, owner=self._owner, lease_seconds=_LEASE_SECONDS)
            except AIError as error:
                if error.code is not ErrorCode.TASK_NOT_READY:
                    _logger.debug("local task claim skipped: graph=%s task=%s code=%s", request.graph.graph_id, node.task_id, error.code.value)
                return
            heartbeat = asyncio.create_task(self._heartbeat(lease, request.principal.tenant_id))
            try:
                nodes = await self._repository.list_nodes(request.graph.graph_id, tenant_id=request.principal.tenant_id)
                dependency_results: dict[str, TaskDependencyResult] = {}
                for dependency in node.dependencies:
                    dependency_node = _node_by_id(nodes, dependency)
                    if (
                        dependency_node is None
                        or dependency_node.status is not TaskStatus.SUCCEEDED
                        or dependency_node.result_digest is None
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    dependency_results[dependency] = TaskDependencyResult(dependency_node.result_digest, dependency_node.execution_id)
                result = await self._runner.run(node, graph_id=request.graph.graph_id, principal=request.principal, dependency_results=dependency_results)
                await self._verifier.verify(result, tenant_id=request.principal.tenant_id)
                await self._repository.complete(lease, tenant_id=request.principal.tenant_id, execution_id=result.execution_id, result_digest=result.result_digest)
                _logger.debug("local task node completed: graph=%s task=%s execution=%s", request.graph.graph_id, node.task_id, result.execution_id)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                code = error.code.value if isinstance(error, AIError) else ErrorCode.TASK_NODE_FAILED.value
                digest = canonical_sha256({"graph_id": request.graph.graph_id, "task_id": node.task_id, "error_code": code})
                try:
                    await self._repository.fail(lease, tenant_id=request.principal.tenant_id, error_code=code, error_digest=digest)
                except AIError as terminal_error:
                    if terminal_error.code is not ErrorCode.TASK_FENCE_STALE:
                        raise
                _logger.exception("local task node failed: graph=%s task=%s code=%s", request.graph.graph_id, node.task_id, code)
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

    async def _heartbeat(self, lease: _TaskLease, tenant_id: str) -> None:
        while True:
            await asyncio.sleep(_LEASE_SECONDS / 2)
            lease = await self._repository.renew(lease, tenant_id=tenant_id)


def _node_by_id(nodes: "tuple[_TaskNodeState, ...]", task_id: str) -> "_TaskNodeState | None":
    return next((node for node in nodes if node.task_id == task_id), None)


__all__ = ["LocalTaskGraphLauncher", "RuntimeTaskNodeResultVerifier", "TaskNodeResultVerifier", "TaskNodeRunResult", "TaskNodeRunner"]
