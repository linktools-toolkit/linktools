#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local TaskGraph scheduling over durable repository authority."""

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from linktools.core import environ

from ..core import JsonValue, Principal, TaskStatus, canonical_sha256, validate_lease_owner
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ._graph import (
    CancelGraphRequest,
    TaskDependencyResult,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
)

_logger = environ.get_logger("ai.task.local")
_HEARTBEAT_SECONDS = 30.0
_LEASE_SECONDS = 90
_SCHEDULER_RECHECK_SECONDS = 1.0
_RECOVERY_UNKNOWN_CODES = frozenset(
    {
        ErrorCode.STORAGE_COMMIT_UNKNOWN,
        ErrorCode.STORAGE_RECOVERY_REQUIRED,
        ErrorCode.EXECUTION_START_UNKNOWN,
        ErrorCode.TOOL_EFFECT_UNKNOWN,
    }
)
_TERMINAL = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class TaskNodeRunResult:
    result_digest: str
    execution_id: "str | None" = None
    result_payload: "StoredPayload | None" = None

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task node result digest is invalid")
        if self.execution_id is not None and (
            not isinstance(self.execution_id, str) or not self.execution_id.strip()
        ):
            raise ValueError("task node result execution id is invalid")
        if self.result_payload is not None:
            if not isinstance(self.result_payload, StoredPayload):
                raise TypeError("task node result payload is invalid")
            if self.result_payload.digest != self.result_digest:
                raise ValueError("task node result payload digest does not match result")


class TaskNodeRunError(AIError):
    """A task-node failure tied to one concrete execution."""

    def __init__(
        self,
        code: ErrorCode,
        execution_id: str,
        *,
        safe_details: "Mapping[str, JsonValue] | None" = None,
    ) -> None:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("task node failure execution id is required")
        super().__init__(code, safe_details=safe_details)
        self.execution_id = execution_id


@runtime_checkable
class TaskNodeRunControl(Protocol):
    async def bind_execution(self, execution_id: str) -> None: ...


class TaskNodeRunner(Protocol):
    async def run(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult: ...

    async def cancel(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
        dependency_results: "Mapping[str, TaskDependencyResult]",
    ) -> None: ...


@runtime_checkable
class _RunnerBackgroundOwner(Protocol):
    @property
    def pending_background_tasks(self) -> "tuple[asyncio.Task[object], ...]": ...

    @property
    def pending_cancelled_tasks(self) -> "tuple[asyncio.Task[object], ...]": ...

    @property
    def background_failure(self) -> "AIError | None": ...


class _TaskRepository(Protocol):
    async def reconcile_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView | None": ...

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> "tuple[TaskNodeView, ...]": ...

    async def get_results(
        self,
        graph_id: str,
        node_ids: "tuple[str, ...]",
        *,
        tenant_id: str,
    ) -> "Mapping[str, TaskResultRecord]": ...

    async def claim(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        owner: str,
        lease_seconds: int,
    ) -> TaskLease: ...

    async def renew(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        lease_seconds: int,
    ) -> TaskLease: ...

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView: ...

    async def complete(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: "str | None",
        result_digest: str,
        result_payload: "StoredPayload | None" = None,
    ) -> object: ...

    async def fail(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: "str | None" = None,
    ) -> object: ...

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...


@dataclass(slots=True)
class _GraphRun:
    request: TaskGraphLaunch
    owner: str
    task: "asyncio.Task[None] | None" = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    generation: int = 0
    failure: "AIError | None" = None
    closed: bool = False


@dataclass(slots=True)
class _LeaseState:
    lease: TaskLease
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class _InflightNode:
    task: "asyncio.Task[None]"
    lease_state: _LeaseState


class _TaskNodeRunControlImpl:
    def __init__(
        self,
        repository: _TaskRepository,
        lease_state: _LeaseState,
        *,
        tenant_id: str,
        on_activity: Callable[[], Awaitable[None]],
    ) -> None:
        self._repository = repository
        self._lease_state = lease_state
        self._tenant_id = tenant_id
        self._on_activity = on_activity

    async def bind_execution(self, execution_id: str) -> None:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution id is required")
        async with self._lease_state.lock:
            await self._repository.bind_execution(
                self._lease_state.lease,
                tenant_id=self._tenant_id,
                execution_id=execution_id,
            )
        await self._on_activity()


class LocalTaskGraphLauncher:
    """Run admitted TaskGraphs locally while durable state remains authoritative."""

    def __init__(
        self,
        repository: _TaskRepository,
        runner: TaskNodeRunner,
        *,
        owner: str,
    ) -> None:
        try:
            validate_lease_owner(owner)
        except AIError as error:
            raise ValueError("task launcher lease owner is invalid") from error
        self._repository = repository
        self._runner = runner
        self._owner = owner
        self._runs: dict[tuple[str, str], _GraphRun] = {}
        self._lock = asyncio.Lock()
        self._accepting = True

    async def start(self, request: TaskGraphLaunch) -> TaskGraphHandle:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        key = (request.principal.tenant_id, request.graph.graph_id)
        async with self._lock:
            existing = self._runs.get(key)
            if existing is not None and not existing.closed:
                if existing.failure is not None:
                    raise AIError(
                        existing.failure.code,
                        safe_details=dict(existing.failure.safe_details),
                    )
                return TaskGraphHandle(
                    request.graph.graph_id,
                    f"local:{key[0]}:{key[1]}",
                )
            run = _GraphRun(request, self._owner)
            self._runs[key] = run
            run.task = asyncio.create_task(
                self._run_graph(run),
                name=f"task-graph-{request.graph.graph_id}",
            )
            run.task.add_done_callback(
                lambda task, selected=run: self._consume_run(selected, task)
            )
        return TaskGraphHandle(
            request.graph.graph_id,
            f"local:{key[0]}:{key[1]}",
        )

    async def cancel(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView:
        tenant_id = request.principal.tenant_id
        key = (tenant_id, graph_id)
        view = await self._repository.cancel_graph(graph_id, tenant_id=tenant_id)
        async with self._lock:
            run = self._runs.get(key)
        if run is not None:
            states = await self._repository.list_nodes(graph_id, tenant_id=tenant_id)
            static = {node.node_id: node for node in run.request.graph.nodes}
            for state in states:
                if state.status is not TaskStatus.CANCELLED or state.fence < 1:
                    continue
                node = static.get(state.node_id)
                if node is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                dependency_results = await self._dependency_results(
                    graph_id,
                    node,
                    tenant_id=tenant_id,
                )
                await self._runner.cancel(
                    node,
                    graph_id=graph_id,
                    principal=run.request.principal,
                    dependency_results=dependency_results,
                )
            async with self._lock:
                current = self._runs.get(key)
                if current is run:
                    run.closed = True
                    task = run.task
                else:
                    task = None
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await self._notify(run)
        if view.status not in _TERMINAL:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return view

    async def shutdown(self) -> None:
        self._accepting = False
        async with self._lock:
            runs = tuple(self._runs.values())
            tasks = tuple(
                run.task
                for run in runs
                if run.task is not None and not run.task.done()
            )
            for run in runs:
                run.closed = True
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._drain_runner_background()
        async with self._lock:
            self._runs.clear()

    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:
        run = self._runs.get((tenant_id, graph_id))
        return run is not None and (not run.closed or run.failure is not None)

    def graph_activity_generation(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> int | None:
        run = self._runs.get((tenant_id, graph_id))
        if run is None or (run.closed and run.failure is None):
            return None
        return run.generation

    async def wait_graph_activity(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_generation: int,
    ) -> None:
        if after_generation < 0:
            raise ValueError("activity generation must be non-negative")
        async with self._lock:
            run = self._runs.get((tenant_id, graph_id))
        if run is None:
            return
        if run.failure is not None:
            raise AIError(run.failure.code, safe_details=dict(run.failure.safe_details))
        if run.closed:
            return
        async with run.condition:
            await run.condition.wait_for(
                lambda: run.generation != after_generation
                or run.closed
                or run.failure is not None
            )
        if run.failure is not None:
            raise AIError(run.failure.code, safe_details=dict(run.failure.safe_details))

    async def _run_graph(self, run: _GraphRun) -> None:
        request = run.request
        tenant_id = request.principal.tenant_id
        inflight: dict[str, _InflightNode] = {}
        try:
            while not run.closed:
                view = await self._repository.reconcile_graph(
                    request.graph.graph_id,
                    tenant_id=tenant_id,
                )
                states = await self._repository.list_nodes(
                    request.graph.graph_id,
                    tenant_id=tenant_id,
                )
                await self._notify(run)
                if view.status in _TERMINAL:
                    return
                _reap_inflight(inflight)
                now = datetime.now(timezone.utc)
                static = {node.node_id: node for node in request.graph.nodes}
                for state in states:
                    if len(inflight) >= request.limits.max_concurrency:
                        break
                    if state.node_id in inflight or not _runnable(state, now):
                        continue
                    node = static.get(state.node_id)
                    if node is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    try:
                        lease = await self._repository.claim(
                            request.graph.graph_id,
                            state.node_id,
                            tenant_id=tenant_id,
                            owner=run.owner,
                            lease_seconds=_LEASE_SECONDS,
                        )
                    except AIError as error:
                        if error.code in {
                            ErrorCode.TASK_NOT_READY,
                            ErrorCode.TASK_OWNER_CONFLICT,
                            ErrorCode.TASK_FENCE_STALE,
                        }:
                            continue
                        raise
                    lease_state = _LeaseState(lease)
                    task = asyncio.create_task(
                        self._run_node(run, node, lease_state),
                        name=f"task-node-{request.graph.graph_id}-{node.node_id}",
                    )
                    inflight[node.node_id] = _InflightNode(task, lease_state)
                    await self._notify(run)
                if inflight:
                    await asyncio.wait(
                        tuple(value.task for value in inflight.values()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    _reap_inflight(inflight)
                    continue
                await self._wait_scheduler(run, states)
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001
            run.failure = _scheduler_failure(error, request.graph.graph_id)
            await self._notify(run)
        finally:
            for value in inflight.values():
                if not value.task.done():
                    value.task.cancel()
            if inflight:
                await asyncio.gather(
                    *(value.task for value in inflight.values()),
                    return_exceptions=True,
                )
            run.closed = True
            await self._notify(run)

    async def _wait_scheduler(
        self,
        run: _GraphRun,
        states: "tuple[TaskNodeView, ...]",
    ) -> None:
        now = datetime.now(timezone.utc)
        live_expiries = tuple(
            state.lease_expires_at
            for state in states
            if state.status is TaskStatus.RUNNING
            and state.lease_expires_at is not None
            and state.lease_expires_at > now
        )
        timeout = _SCHEDULER_RECHECK_SECONDS
        if live_expiries:
            timeout = min(
                timeout,
                max(0.0, (min(live_expiries) - now).total_seconds()),
            )
        if timeout <= 0:
            return
        async with run.condition:
            generation = run.generation
            try:
                await asyncio.wait_for(
                    run.condition.wait_for(
                        lambda: run.generation != generation
                        or run.closed
                        or run.failure is not None
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return

    async def _run_node(
        self,
        run: _GraphRun,
        node: TaskNode,
        lease_state: _LeaseState,
    ) -> None:
        request = run.request
        graph_id = request.graph.graph_id
        tenant_id = request.principal.tenant_id
        dependency_results = await self._dependency_results(
            graph_id,
            node,
            tenant_id=tenant_id,
        )
        control = _TaskNodeRunControlImpl(
            self._repository,
            lease_state,
            tenant_id=tenant_id,
            on_activity=lambda: self._notify(run),
        )
        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(lease_state, tenant_id=tenant_id, stop=heartbeat_stop),
            name=f"task-heartbeat-{graph_id}-{node.node_id}",
        )
        completion: TaskNodeRunResult | None = None
        try:
            completion = await self._runner.run(
                node,
                graph_id=graph_id,
                principal=request.principal,
                dependency_results=dependency_results,
                control=control,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001
            await _stop_heartbeat(heartbeat_stop, heartbeat)
            if isinstance(error, AIError) and error.code in _RECOVERY_UNKNOWN_CODES:
                await self._defer_recovery(run, node, cause_code=error.code)
                return
            code = (
                error.code.value
                if isinstance(error, AIError)
                else ErrorCode.TASK_NODE_FAILED.value
            )
            execution_id = (
                error.execution_id if isinstance(error, TaskNodeRunError) else None
            )
            digest = canonical_sha256(
                {
                    "graph_id": graph_id,
                    "node_id": node.node_id,
                    "code": code,
                }
            )
            async with lease_state.lock:
                try:
                    await self._repository.fail(
                        lease_state.lease,
                        tenant_id=tenant_id,
                        error_code=code,
                        error_digest=digest,
                        execution_id=execution_id,
                    )
                except AIError as terminal_error:
                    if terminal_error.code is not ErrorCode.TASK_FENCE_STALE:
                        raise
            return
        await _stop_heartbeat(heartbeat_stop, heartbeat)
        async with lease_state.lock:
            try:
                await self._repository.complete(
                    lease_state.lease,
                    tenant_id=tenant_id,
                    execution_id=completion.execution_id,
                    result_digest=completion.result_digest,
                    result_payload=completion.result_payload,
                )
                return
            except AIError as error:
                if error.code not in _RECOVERY_UNKNOWN_CODES:
                    raise
                if await self._completion_committed(
                    graph_id,
                    node.node_id,
                    completion,
                    tenant_id=tenant_id,
                ):
                    return
                await self._defer_recovery(run, node, cause_code=error.code)

    async def _completion_committed(
        self,
        graph_id: str,
        node_id: str,
        completion: TaskNodeRunResult,
        *,
        tenant_id: str,
    ) -> bool:
        states = await self._repository.list_nodes(graph_id, tenant_id=tenant_id)
        state = next((value for value in states if value.node_id == node_id), None)
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state.status is TaskStatus.SUCCEEDED:
            if (
                state.result_digest != completion.result_digest
                or state.execution_id != completion.execution_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if completion.result_payload is not None:
                results = await self._repository.get_results(
                    graph_id,
                    (node_id,),
                    tenant_id=tenant_id,
                )
                record = results.get(node_id)
                if record is None or record.result_digest != completion.result_digest:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return True
        if state.status in _TERMINAL:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return False

    async def _defer_recovery(
        self,
        run: _GraphRun,
        node: TaskNode,
        *,
        cause_code: ErrorCode,
    ) -> None:
        graph_id = run.request.graph.graph_id
        failure = AIError(
            ErrorCode.STORAGE_RECOVERY_REQUIRED,
            safe_details={
                "phase": "task_node_recovery",
                "graph_id": graph_id,
                "node_id": node.node_id,
                "cause_code": cause_code.value,
            },
        )
        async with self._lock:
            current = self._runs.get((run.request.principal.tenant_id, graph_id))
            if current is run:
                run.failure = failure
                run.closed = True
        await self._notify(run)

    async def _heartbeat(
        self,
        lease_state: _LeaseState,
        *,
        tenant_id: str,
        stop: asyncio.Event,
    ) -> None:
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=_HEARTBEAT_SECONDS)
                return
            except asyncio.TimeoutError:
                pass
            async with lease_state.lock:
                lease_state.lease = await self._repository.renew(
                    lease_state.lease,
                    tenant_id=tenant_id,
                    lease_seconds=_LEASE_SECONDS,
                )

    async def _dependency_results(
        self,
        graph_id: str,
        node: TaskNode,
        *,
        tenant_id: str,
    ) -> "dict[str, TaskDependencyResult]":
        if not node.dependencies:
            return {}
        states = await self._repository.list_nodes(graph_id, tenant_id=tenant_id)
        by_id = {state.node_id: state for state in states}
        records = await self._repository.get_results(
            graph_id,
            tuple(node.dependencies),
            tenant_id=tenant_id,
        )
        result: dict[str, TaskDependencyResult] = {}
        for dependency_id in node.dependencies:
            state = by_id.get(dependency_id)
            if (
                state is None
                or state.status is not TaskStatus.SUCCEEDED
                or state.result_digest is None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            record = records.get(dependency_id)
            if record is not None and record.result_digest != state.result_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[dependency_id] = TaskDependencyResult(
                state.result_digest,
                state.execution_id,
                None if record is None else record.payload,
            )
        return result

    async def _notify(self, run: _GraphRun) -> None:
        async with run.condition:
            run.generation += 1
            run.condition.notify_all()

    async def _drain_runner_background(self) -> None:
        if not isinstance(self._runner, _RunnerBackgroundOwner):
            return
        pending = {
            *self._runner.pending_background_tasks,
            *self._runner.pending_cancelled_tasks,
        }
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        failure = self._runner.background_failure
        if failure is not None:
            raise AIError(failure.code, safe_details=dict(failure.safe_details))

    @staticmethod
    def _consume_run(run: _GraphRun, task: "asyncio.Task[None]") -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and run.failure is None:
            run.failure = _scheduler_failure(error, run.request.graph.graph_id)


def _scheduler_failure(error: BaseException, graph_id: str) -> AIError:
    if isinstance(error, AIError):
        return AIError(
            error.code,
            safe_details={**dict(error.safe_details), "graph_id": graph_id},
        )
    return AIError(
        ErrorCode.STORAGE_RECOVERY_REQUIRED,
        safe_details={"graph_id": graph_id, "phase": "task_scheduler"},
    )


async def _stop_heartbeat(stop: asyncio.Event, task: "asyncio.Task[None]") -> None:
    stop.set()
    if not task.done():
        task.cancel()
    result = await asyncio.gather(task, return_exceptions=True)
    error = result[0]
    if isinstance(error, BaseException) and not isinstance(error, asyncio.CancelledError):
        raise error


def _runnable(node: TaskNodeView, now: datetime) -> bool:
    return node.status is TaskStatus.READY or (
        node.status is TaskStatus.RUNNING
        and node.lease_expires_at is not None
        and node.lease_expires_at <= now
    )


def _reap_inflight(inflight: dict[str, _InflightNode]) -> None:
    for node_id, state in tuple(inflight.items()):
        if not state.task.done():
            continue
        inflight.pop(node_id, None)
        state.task.result()


__all__ = [
    "LocalTaskGraphLauncher",
    "TaskNodeRunControl",
    "TaskNodeRunError",
    "TaskNodeRunResult",
    "TaskNodeRunner",
]
