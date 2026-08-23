"""Process-local TaskGraph launcher backed by TaskRepository leases."""

import asyncio
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol
from weakref import WeakKeyDictionary

from linktools.core import environ

from ..core import Principal, TaskStatus, canonical_sha256, validate_lease_owner
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


class _TaskNodeState(Protocol):
    node_id: str
    status: TaskStatus
    execution_id: "str | None"
    result_digest: "str | None"
    lease_expires_at: "datetime | None"
    fence: int


class _TaskLease(Protocol):
    owner: str
    fence: int
    lease_expires_at: datetime


class _TaskTerminal(Protocol):
    pass


class _TaskRepository(Protocol):
    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...
    async def get_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView | None": ...
    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> "tuple[_TaskNodeState, ...]": ...
    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> _TaskLease: ...
    async def renew(self, lease: _TaskLease, *, tenant_id: str, lease_seconds: int) -> _TaskLease: ...
    async def complete(
        self,
        lease: _TaskLease,
        *,
        tenant_id: str,
        execution_id: "str | None",
        result_digest: str,
    ) -> _TaskTerminal: ...
    async def fail(
        self,
        lease: _TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
    ) -> _TaskTerminal: ...


@dataclass
class _GraphRun:
    task: "asyncio.Task[None] | None" = None
    activity: asyncio.Event = field(default_factory=asyncio.Event)
    activity_generation: int = 0
    failure: "AIError | None" = None
    closed: bool = False
    waiters: int = 0


@dataclass
class _InflightNode:
    task: asyncio.Task[None]
    lease: _TaskLease | None = None


class LocalTaskGraphLauncher:
    """Schedule DAG nodes while keeping durable truth in TaskRepository."""

    def __init__(self, repository: _TaskRepository, runner: TaskNodeRunner, *, owner: str) -> None:
        try:
            validate_lease_owner(owner)
        except AIError as error:
            raise ValueError("task launcher lease owner is invalid") from error
        self._repository = repository
        self._runner = runner
        self._owner = owner
        self._graphs: dict[tuple[str, str], _GraphRun] = {}
        self._wait_observations: WeakKeyDictionary[asyncio.Task[Any], dict[tuple[str, str], int]] = WeakKeyDictionary()
        self._accepting = True

    async def start(self, request: TaskGraphRequest) -> TaskGraphHandle:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        key = request.principal.tenant_id, request.graph.graph_id
        existing = self._graphs.get(key)
        if existing is None or existing.closed or existing.task is None or existing.task.done():
            if existing is not None and not existing.closed:
                self._close_run(existing)
            run = _GraphRun()
            run.task = asyncio.create_task(
                self._run_graph(request, run),
                name=f"task-graph-{key[0]}-{key[1]}",
            )
            self._graphs[key] = run
            if existing is not None:
                _logger.info(
                    "task graph scheduler re-armed: tenant=%s graph=%s",
                    key[0],
                    key[1],
                )
        _logger.info(
            "task graph scheduler armed: tenant=%s graph=%s nodes=%s",
            key[0],
            key[1],
            len(request.graph.nodes),
        )
        return TaskGraphHandle(request.graph.graph_id, f"local:{request.principal.tenant_id}:{request.graph.graph_id}")

    async def cancel(self, graph_id: str, request: CancelGraphRequest) -> TaskGraphView:
        key = request.principal.tenant_id, graph_id
        run = self._graphs.get(key)
        if run is not None:
            self._close_run(run)
            if run.task is not None and not run.task.done():
                run.task.cancel()
                await asyncio.gather(run.task, return_exceptions=True)
        view = await self._repository.get_graph(graph_id, tenant_id=request.principal.tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if run is not None:
            self._remove_run(key, run)
        _logger.info("task graph scheduler disarmed: tenant=%s graph=%s", key[0], graph_id)
        return view

    async def shutdown(self) -> None:
        self._accepting = False
        runs = tuple(self._graphs.values())
        for run in runs:
            self._close_run(run)
            if run.task is not None and not run.task.done():
                run.task.cancel()
        tasks = tuple(run.task for run in runs if run.task is not None)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._graphs.clear()
        self._wait_observations.clear()
        _logger.info("local task graph launcher closed: graphs=%s", len(runs))

    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:
        key = tenant_id, graph_id
        run = self._graphs.get(key)
        if run is None:
            return False
        if run.closed:
            if run.failure is None:
                self._schedule_cleanup(run)
                return False
            self._remember_observation(key, run)
            return True
        owned = run.task is not None and not run.task.done()
        if owned:
            self._remember_observation(key, run)
        return owned

    async def wait_graph_activity(self, graph_id: str, *, tenant_id: str) -> None:
        key = tenant_id, graph_id
        run = self._graphs.get(key)
        if run is None:
            return
        run.waiters += 1
        event = run.activity
        task = asyncio.current_task()
        observations = self._wait_observations.get(task) if task is not None else None
        observed_generation = (
            observations.pop(key, run.activity_generation)
            if observations is not None
            else run.activity_generation
        )
        if observations is not None and not observations:
            self._wait_observations.pop(task, None)
        try:
            if run.failure is not None:
                raise AIError(run.failure.code, safe_details=dict(run.failure.safe_details))
            if not run.closed and run.activity_generation == observed_generation:
                await event.wait()
            if run.failure is not None:
                raise AIError(run.failure.code, safe_details=dict(run.failure.safe_details))
        finally:
            run.waiters -= 1
            self._cleanup_run(run)

    async def _run_graph(self, request: TaskGraphRequest, run: _GraphRun) -> None:
        key = request.principal.tenant_id, request.graph.graph_id
        inflight: dict[str, _InflightNode] = {}
        try:
            while not run.closed:
                _reap_inflight(inflight)
                view = await self._repository.reconcile_graph(
                    request.graph.graph_id,
                    tenant_id=request.principal.tenant_id,
                )
                if view.status in {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED}:
                    await _cancel_inflight(inflight)
                    return
                nodes = await self._repository.list_nodes(request.graph.graph_id, tenant_id=request.principal.tenant_id)
                now = datetime.now(timezone.utc)
                persisted = {
                    node.node_id
                    for node in nodes
                    if (
                        node.status is TaskStatus.RUNNING
                        and node.lease_expires_at is not None
                        and node.lease_expires_at > now
                    )
                }
                used = persisted | set(inflight)
                capacity = max(0, request.limits.max_concurrency - len(used))
                for node in request.graph.nodes:
                    if capacity <= 0:
                        break
                    if node.node_id in inflight or node.node_id in persisted:
                        continue
                    stored = _node_by_id(nodes, node.node_id)
                    if stored is None or not _runnable(stored, now):
                        continue
                    task = asyncio.create_task(
                        self._run_node(request, node, inflight, run),
                        name=f"task-node-{node.node_id}",
                    )
                    inflight[node.node_id] = _InflightNode(task)
                    capacity -= 1
                if inflight:
                    await asyncio.wait(
                        tuple(state.task for state in inflight.values()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                else:
                    await self._wait_for_activity(run, nodes, now)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            run.failure = (
                error
                if isinstance(error, AIError)
                else AIError(
                    ErrorCode.INTERNAL_ERROR,
                    safe_details={"phase": "task_scheduler"},
                )
            )
            _logger.exception("local task graph scheduler failed: tenant=%s graph=%s", key[0], key[1])
        finally:
            self._close_run(run)
            await _cancel_inflight(inflight)
            self._schedule_cleanup(run)

    async def _wait_for_activity(
        self,
        run: _GraphRun,
        nodes: "tuple[_TaskNodeState, ...]",
        now: datetime,
    ) -> None:
        timeout = 1.0
        for node in nodes:
            if node.status is TaskStatus.RUNNING and node.lease_expires_at is not None:
                timeout = min(timeout, max(0.0, (node.lease_expires_at - now).total_seconds()))
        event = run.activity
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return

    async def _run_node(
        self,
        request: TaskGraphRequest,
        node: TaskNode,
        inflight: dict[str, _InflightNode],
        run: _GraphRun,
    ) -> None:
        state = inflight[node.node_id]
        runner_task: asyncio.Task[TaskNodeRunResult] | None = None
        heartbeat_task: asyncio.Task[None] | None = None
        try:
            lease = await self._repository.claim(
                request.graph.graph_id,
                node.node_id,
                tenant_id=request.principal.tenant_id,
                owner=self._owner,
                lease_seconds=_LEASE_SECONDS,
            )
            state.lease = lease
            runner_task = asyncio.create_task(self._runner.run(
                node,
                graph_id=request.graph.graph_id,
                principal=request.principal,
                dependency_results=await self._dependency_results(request, node),
            ))
            heartbeat_task = asyncio.create_task(self._heartbeat(state, request.principal.tenant_id))
            done, _ = await asyncio.wait((runner_task, heartbeat_task), return_when=asyncio.FIRST_COMPLETED)
            heartbeat_error = _task_error(heartbeat_task) if heartbeat_task in done else None
            if heartbeat_error is not None:
                runner_task.cancel()
                await asyncio.gather(runner_task, return_exceptions=True)
                _logger.warning("task heartbeat lost ownership: graph=%s task=%s", request.graph.graph_id, node.node_id)
                return
            if runner_task in done:
                result = runner_task.result()
                lease = state.lease
                if lease is None:
                    raise AIError(ErrorCode.TASK_FENCE_STALE)
                try:
                    await self._repository.complete(
                        lease,
                        tenant_id=request.principal.tenant_id,
                        execution_id=result.execution_id,
                        result_digest=result.result_digest,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.TASK_FENCE_STALE:
                        raise
                return
            result = await runner_task
            lease = state.lease
            if lease is None:
                raise AIError(ErrorCode.TASK_FENCE_STALE)
            await self._repository.complete(
                lease,
                tenant_id=request.principal.tenant_id,
                execution_id=result.execution_id,
                result_digest=result.result_digest,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            lease = state.lease
            if lease is not None:
                code = error.code.value if isinstance(error, AIError) else ErrorCode.TASK_NODE_FAILED.value
                digest = canonical_sha256(
                    {
                        "graph_id": request.graph.graph_id,
                        "node_id": node.node_id,
                        "error_code": code,
                    }
                )
                try:
                    await self._repository.fail(
                        lease,
                        tenant_id=request.principal.tenant_id,
                        error_code=code,
                        error_digest=digest,
                    )
                except AIError as terminal_error:
                    if terminal_error.code is not ErrorCode.TASK_FENCE_STALE:
                        raise
                _logger.error(
                    "local task node failed: graph=%s task=%s code=%s",
                    request.graph.graph_id,
                    node.node_id,
                    code,
                )
        finally:
            for task in (runner_task, heartbeat_task):
                if task is not None and not task.done():
                    task.cancel()
            pending = tuple(task for task in (runner_task, heartbeat_task) if task is not None)
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if not run.closed:
                self._signal(run)

    async def _dependency_results(
        self,
        request: TaskGraphRequest,
        node: TaskNode,
    ) -> "Mapping[str, TaskDependencyResult]":
        nodes = await self._repository.list_nodes(request.graph.graph_id, tenant_id=request.principal.tenant_id)
        results: dict[str, TaskDependencyResult] = {}
        for dependency in node.dependencies:
            dependency_node = _node_by_id(nodes, dependency)
            if (
                dependency_node is None
                or dependency_node.status is not TaskStatus.SUCCEEDED
                or dependency_node.result_digest is None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            results[dependency] = TaskDependencyResult(dependency_node.result_digest, dependency_node.execution_id)
        return results

    async def _heartbeat(self, state: _InflightNode, tenant_id: str) -> None:
        while True:
            await asyncio.sleep(_LEASE_SECONDS / 2)
            if state.lease is None:
                continue
            state.lease = await self._repository.renew(state.lease, tenant_id=tenant_id, lease_seconds=_LEASE_SECONDS)

    def _close_run(self, run: _GraphRun) -> None:
        if run.closed:
            return
        run.closed = True
        run.activity.set()

    def _signal(self, run: _GraphRun) -> None:
        if run.closed:
            return
        event = run.activity
        run.activity = asyncio.Event()
        run.activity_generation += 1
        event.set()

    def _remember_observation(self, key: tuple[str, str], run: _GraphRun) -> None:
        task = asyncio.current_task()
        if task is not None:
            observations = self._wait_observations.setdefault(task, {})
            observations[key] = run.activity_generation

    def _schedule_cleanup(self, run: _GraphRun) -> None:
        asyncio.get_running_loop().call_soon(self._cleanup_run, run)

    def _cleanup_run(self, run: _GraphRun) -> None:
        if not run.closed or run.waiters or run.failure is not None:
            return
        for key, current in tuple(self._graphs.items()):
            if current is run:
                self._remove_run(key, run)

    def _remove_run(self, key: tuple[str, str], run: _GraphRun) -> None:
        if self._graphs.get(key) is run:
            self._graphs.pop(key, None)


def _runnable(node: _TaskNodeState, now: datetime) -> bool:
    return node.status is TaskStatus.READY or (
        node.status is TaskStatus.RUNNING
        and node.lease_expires_at is not None
        and node.lease_expires_at <= now
    )


def _task_error(task: asyncio.Task[object]) -> BaseException | None:
    if task.cancelled():
        return asyncio.CancelledError()
    return task.exception()


def _reap_inflight(inflight: dict[str, _InflightNode]) -> None:
    for node_id, state in tuple(inflight.items()):
        if state.task.done():
            inflight.pop(node_id, None)


async def _cancel_inflight(inflight: dict[str, _InflightNode]) -> None:
    tasks = tuple(state.task for state in inflight.values() if not state.task.done())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    inflight.clear()


def _node_by_id(nodes: "tuple[_TaskNodeState, ...]", node_id: str) -> "_TaskNodeState | None":
    return next((node for node in nodes if node.node_id == node_id), None)


__all__ = ["LocalTaskGraphLauncher", "TaskNodeRunResult", "TaskNodeRunner"]
