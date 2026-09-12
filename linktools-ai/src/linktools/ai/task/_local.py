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

from ..core import (
    JsonValue,
    Page,
    Principal,
    CorrelationData,
    TaskStatus,
    canonical_sha256,
    validate_lease_owner,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ._event import TaskEvent
from ._graph import (
    TaskDependencyResult,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphSnapshot,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
)
from ._metrics import _TaskMetricProjector

_logger = environ.get_logger("ai.task.local")
_HEARTBEAT_SECONDS = 30.0
_LEASE_SECONDS = 60
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


async def _noop_execution_hold(*args: object, **kwargs: object) -> None:
    del args, kwargs


@dataclass(frozen=True, slots=True)
class TaskNodeRunResult:
    result_digest: str
    execution_id: "str | None" = None
    result_payload: "StoredPayload | None" = None
    expanded_nodes: "tuple[TaskNode, ...]" = ()

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task node result digest is invalid")
        if self.execution_id is not None and (
            not isinstance(self.execution_id, str) or not self.execution_id.strip()
        ):
            raise ValueError("task node result execution id is invalid")
        if (
            self.result_payload is not None
            and self.result_payload.digest != self.result_digest
        ):
            raise ValueError("task node result payload digest does not match result")
        expanded_nodes = tuple(self.expanded_nodes)
        if any(not isinstance(node, TaskNode) for node in expanded_nodes):
            raise TypeError("expanded task nodes are invalid")
        object.__setattr__(self, "expanded_nodes", expanded_nodes)


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
    async def handoff_execution(self, execution_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskNodeInvocation:
    node: TaskNode
    graph_id: str
    principal: Principal
    correlation: CorrelationData
    dependency_results: "Mapping[str, TaskDependencyResult]"


class TaskNodeRunner(Protocol):
    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult: ...

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult: ...

    async def cancel(self, invocation: TaskNodeInvocation) -> None: ...


@runtime_checkable
class _RunnerBackgroundOwner(Protocol):
    @property
    def pending_background_tasks(self) -> "tuple[asyncio.Task[object], ...]": ...

    @property
    def pending_cancelled_tasks(self) -> "tuple[asyncio.Task[object], ...]": ...

    @property
    def background_failure(self) -> "AIError | None": ...


class _TaskRepository(Protocol):
    async def scheduler_snapshot(
        self, graph_id: str, *, tenant_id: str
    ) -> "TaskGraphSnapshot": ...

    async def snapshot_graph(
        self, graph_id: str, *, tenant_id: str
    ) -> "TaskGraphSnapshot | None": ...

    async def get_graph(self, graph_id: str, *, tenant_id: str) -> "TaskGraphView | None": ...

    async def list_nodes(self, graph_id: str, *, tenant_id: str) -> "tuple[TaskNodeView, ...]": ...

    async def get_results(
        self,
        graph_id: str,
        node_ids: "tuple[str, ...]",
        *,
        tenant_id: str,
    ) -> "Mapping[str, TaskResultRecord]": ...

    async def list_events(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[TaskEvent]: ...

    async def latest_event(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskEvent | None: ...

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

    async def handoff_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView: ...

    async def mark_recovery_required(
        self,
        lease: "TaskLease | None",
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: "str | None" = None,
        graph_id: "str | None" = None,
        node_id: "str | None" = None,
    ) -> TaskNodeView: ...

    async def complete(
        self,
        lease: "TaskLease | None",
        *,
        tenant_id: str,
        execution_id: "str | None",
        result_digest: str,
        result_payload: "StoredPayload | None" = None,
        graph_id: "str | None" = None,
        node_id: "str | None" = None,
        expanded_nodes: "tuple[TaskNode, ...]" = (),
    ) -> object: ...

    async def fail(
        self,
        lease: "TaskLease | None",
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        execution_id: "str | None" = None,
        graph_id: "str | None" = None,
        node_id: "str | None" = None,
    ) -> object: ...

    async def cancel_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView: ...


@dataclass(slots=True)
class _GraphRun:
    request: TaskGraphLaunch
    owner: str
    task: "asyncio.Task[None] | None" = None
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    generation: int = 0
    observation_backoff: float = 1.0
    failure: "AIError | None" = None
    closed: bool = False


@dataclass(slots=True)
class _LeaseState:
    lease: TaskLease
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(slots=True)
class _InflightNode:
    task: "asyncio.Task[None]"
    lease_state: "_LeaseState | None"


class _TaskNodeRunControlImpl:
    def __init__(
        self,
        repository: _TaskRepository,
        lease_state: _LeaseState,
        *,
        tenant_id: str,
        on_activity: Callable[[], Awaitable[None]],
        on_handoff: Callable[[], None],
    ) -> None:
        self._repository = repository
        self._lease_state = lease_state
        self._tenant_id = tenant_id
        self._on_activity = on_activity
        self._on_handoff = on_handoff
        self._execution_id: str | None = None

    @property
    def handed_off_execution_id(self) -> str | None:
        return self._execution_id

    async def handoff_execution(self, execution_id: str) -> None:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("execution id is required")
        async with self._lease_state.lock:
            try:
                await self._repository.handoff_execution(
                    self._lease_state.lease,
                    tenant_id=self._tenant_id,
                    execution_id=execution_id,
                )
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
                lease = self._lease_state.lease
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    safe_details={
                        "phase": "task_execution_handoff",
                        "graph_id": lease.graph_id,
                        "node_id": lease.node_id,
                    },
                ) from error
            lease = self._lease_state.lease
            self._execution_id = execution_id
            self._on_handoff()
        _logger.info(
            "task execution handed off: graph=%s node=%s execution=%s fence=%s",
            lease.graph_id,
            lease.node_id,
            execution_id,
            lease.fence,
        )
        await self._on_activity()


class LocalTaskGraphLauncher:
    """Run admitted TaskGraphs locally while durable state remains authoritative."""

    _metric_projector: _TaskMetricProjector | None = None

    def __init__(
        self,
        repository: _TaskRepository,
        runner: TaskNodeRunner,
        *,
        owner: str,
        acquire_execution_hold: "Callable[..., Awaitable[None]] | None" = None,
        release_execution_hold: "Callable[..., Awaitable[None]] | None" = None,
    ) -> None:
        try:
            validate_lease_owner(owner)
        except AIError as error:
            raise ValueError("task launcher lease owner is invalid") from error
        self._repository = repository
        self._runner = runner
        self._owner = owner
        self._acquire_execution_hold = (
            _noop_execution_hold
            if acquire_execution_hold is None
            else acquire_execution_hold
        )
        self._release_execution_hold = (
            _noop_execution_hold
            if release_execution_hold is None
            else release_execution_hold
        )
        self._metric_projector = None
        self._graphs: dict[tuple[str, str], _GraphRun] = {}
        self._lock = asyncio.Lock()
        self._accepting = True

    def bind_metric_projector(self, projector: _TaskMetricProjector) -> None:
        if not isinstance(projector, _TaskMetricProjector):
            raise TypeError("projector must be _TaskMetricProjector")
        if self._metric_projector is not None and self._metric_projector is not projector:
            raise RuntimeError("task metric projector is already bound")
        self._metric_projector = projector

    async def start(self, request: TaskGraphLaunch) -> TaskGraphHandle:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        key = (request.principal.tenant_id, request.graph_id)
        async with self._lock:
            existing = self._graphs.get(key)
            if existing is not None and not existing.closed:
                if existing.failure is not None:
                    raise _copy_ai_error(existing.failure)
                return TaskGraphHandle(
                    request.graph_id,
                    f"local:{key[0]}:{key[1]}",
                )
            run = _GraphRun(request, self._owner)
            self._graphs[key] = run
            run.task = asyncio.create_task(
                self._run_graph(run),
                name=f"task-graph-{request.graph_id}",
            )
            run.task.add_done_callback(
                lambda task, selected=run: self._consume_run(selected, task)
            )
        return TaskGraphHandle(
            request.graph_id,
            f"local:{key[0]}:{key[1]}",
        )

    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView:
        graph_id = launch.graph_id
        tenant_id = launch.principal.tenant_id
        key = (tenant_id, graph_id)
        async with self._lock:
            active_run = self._graphs.get(key)
        if active_run is not None and active_run.request != launch:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        view = await self._repository.get_graph(graph_id, tenant_id=tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        snapshot = await self._repository.snapshot_graph(
            graph_id,
            tenant_id=tenant_id,
        )
        if snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        states = snapshot.node_states
        static = {node.node_id: node for node in snapshot.nodes}
        cleanup_error: BaseException | None = None
        for state in states:
            if state.status is not TaskStatus.CANCELLED or state.fence < 1:
                continue
            node = static.get(state.node_id)
            if node is None:
                if cleanup_error is None:
                    cleanup_error = AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                continue
            try:
                await self._runner.cancel(
                    TaskNodeInvocation(
                        node,
                        graph_id,
                        launch.principal,
                        launch.correlation,
                        await self._dependency_results(
                            graph_id, node, tenant_id=tenant_id
                        ),
                    )
                )
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                if cleanup_error is None:
                    cleanup_error = error
        async with self._lock:
            run = self._graphs.pop(key, None)
            if run is not None:
                run.closed = True
                task = run.task
            else:
                task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if run is not None:
            await self._notify(run)
        if self._metric_projector is not None and view.status in _TERMINAL:
            self._metric_projector.trigger(graph_id, tenant_id=tenant_id)
        if cleanup_error is not None:
            if isinstance(cleanup_error, AIError):
                raise cleanup_error
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={"phase": "task_graph_cancel_cleanup", "graph_id": graph_id},
            ) from cleanup_error
        return view

    async def shutdown(self) -> None:
        cleanup = asyncio.create_task(
            self._shutdown_owned(),
            name="task-graph-launcher-shutdown",
        )
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            error = cleanup.exception()
            if error is not None:
                raise error
            raise

    async def _shutdown_owned(self) -> None:
        self._accepting = False
        async with self._lock:
            runs = tuple(self._graphs.values())
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
            self._graphs.clear()

    def owns_graph(self, graph_id: str, *, tenant_id: str) -> bool:
        run = self._graphs.get((tenant_id, graph_id))
        return run is not None and (not run.closed or run.failure is not None)

    def graph_activity_generation(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> int | None:
        run = self._graphs.get((tenant_id, graph_id))
        if run is None or (run.closed and run.failure is None):
            return None
        return run.generation

    async def wait_graph_activity(
        self,
        graph_id: str,
        *,
        tenant_id: str,
        after_generation: "int | None" = None,
    ) -> None:
        if after_generation is not None and after_generation < 0:
            raise ValueError("activity generation must be non-negative")
        key = (tenant_id, graph_id)
        async with self._lock:
            run = self._graphs.get(key)
        if run is None:
            return
        if run.failure is not None:
            raise _copy_ai_error(run.failure)
        if run.closed:
            async with self._lock:
                if self._graphs.get(key) is run:
                    self._graphs.pop(key, None)
            return
        observed_generation = run.generation if after_generation is None else after_generation
        async with run.condition:
            await run.condition.wait_for(
                lambda: run.generation != observed_generation
                or run.closed
                or run.failure is not None
            )
        if run.failure is not None:
            raise _copy_ai_error(run.failure)
        if run.closed:
            async with self._lock:
                if self._graphs.get(key) is run:
                    self._graphs.pop(key, None)

    async def _run_graph(self, run: _GraphRun) -> None:
        request = run.request
        tenant_id = request.principal.tenant_id
        inflight: dict[str, _InflightNode] = {}
        observed_fingerprint: str | None = None
        try:
            while not run.closed:
                try:
                    snapshot = await self._repository.scheduler_snapshot(
                        request.graph_id,
                        tenant_id=tenant_id,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    await asyncio.sleep(0)
                    continue
                view = TaskGraphView(
                    snapshot.graph_id,
                    snapshot.status,
                    snapshot.nodes,
                )
                states = snapshot.node_states
                now = datetime.now(timezone.utc)
                fingerprint = _scheduler_observation_fingerprint(view, states, now)
                if fingerprint != observed_fingerprint:
                    observed_fingerprint = fingerprint
                    run.observation_backoff = 1.0
                    await self._notify(run)
                if view.status is TaskStatus.RECOVERY_REQUIRED:
                    return
                if view.status in _TERMINAL:
                    if self._metric_projector is not None:
                        self._metric_projector.trigger(
                            request.graph_id,
                            tenant_id=tenant_id,
                        )
                    if view.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}:
                        await self._cancel_terminal_effects(run, states)
                    return
                _reap_inflight(inflight)
                persisted = {
                    state.node_id
                    for state in states
                    if state.status is TaskStatus.RUNNING
                    and state.lease_expires_at is not None
                    and state.lease_expires_at > now
                }
                waiting = {
                    state.node_id
                    for state in states
                    if state.status is TaskStatus.WAITING
                }
                static = {node.node_id: node for node in snapshot.nodes}
                for state in states:
                    if state.status is not TaskStatus.WAITING:
                        continue
                    if state.node_id in inflight:
                        continue
                    node = static.get(state.node_id)
                    if node is None or state.execution_id is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    task = asyncio.create_task(
                        self._wait_bound_node(run, node, state.execution_id),
                        name=f"task-wait-{request.graph_id}-{node.node_id}",
                    )
                    inflight[node.node_id] = _InflightNode(task, None)
                    await self._notify(run)
                used = persisted | waiting | set(inflight)
                capacity = max(
                    0,
                    request.limits.max_concurrency - len(used),
                )
                for state in states:
                    if capacity <= 0:
                        break
                    if state.node_id in inflight:
                        continue
                    node = static.get(state.node_id)
                    if node is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if state.node_id in used or not _runnable(state, now):
                        continue
                    try:
                        lease = await self._repository.claim(
                            request.graph_id,
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
                        name=f"task-node-{request.graph_id}-{node.node_id}",
                    )
                    inflight[node.node_id] = _InflightNode(task, lease_state)
                    used.add(node.node_id)
                    capacity -= 1
                    await self._notify(run)
                if inflight:
                    done, _pending = await asyncio.wait(
                        tuple(value.task for value in inflight.values()),
                        timeout=run.observation_backoff,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if done:
                        run.observation_backoff = 1.0
                    else:
                        run.observation_backoff = min(
                            30.0,
                            run.observation_backoff * 2,
                        )
                    continue
                await self._wait_scheduler(run, states)
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001
            run.failure = _scheduler_failure(error, request.graph_id)
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
            if run.failure is None:
                key = (tenant_id, request.graph_id)
                async with self._lock:
                    if self._graphs.get(key) is run:
                        self._graphs.pop(key, None)

    async def _cancel_terminal_effects(
        self,
        run: _GraphRun,
        states: "tuple[TaskNodeView, ...]",
    ) -> None:
        request = run.request
        snapshot = await self._repository.snapshot_graph(
            request.graph_id,
            tenant_id=request.principal.tenant_id,
        )
        if snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        static = {node.node_id: node for node in snapshot.nodes}
        tenant_id = request.principal.tenant_id
        for state in states:
            if state.status not in {TaskStatus.RUNNING, TaskStatus.WAITING}:
                continue
            if state.fence < 1:
                continue
            node = static.get(state.node_id)
            if node is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._runner.cancel(
                TaskNodeInvocation(
                    node,
                    request.graph_id,
                    request.principal,
                    request.correlation,
                    await self._dependency_results(
                        request.graph_id,
                        node,
                        tenant_id=tenant_id,
                    ),
                )
            )

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
        timeout = run.observation_backoff
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
                run.observation_backoff = min(30.0, timeout * 2)
                return
            else:
                run.observation_backoff = 1.0

    async def _wait_bound_node(
        self,
        run: _GraphRun,
        node: TaskNode,
        execution_id: str,
    ) -> None:
        request = run.request
        graph_id = request.graph_id
        tenant_id = request.principal.tenant_id
        hold_id = f"task:{graph_id}:{node.node_id}"
        await self._acquire_execution_hold(
            execution_id,
            tenant_id=tenant_id,
            hold_id=hold_id,
        )
        try:
            await self._settle_bound_node(run, node, execution_id)
        finally:
            await self._release_execution_hold(
                execution_id,
                tenant_id=tenant_id,
                hold_id=hold_id,
            )

    async def _settle_bound_node(
        self,
        run: _GraphRun,
        node: TaskNode,
        execution_id: str,
    ) -> None:
        request = run.request
        graph_id = request.graph_id
        tenant_id = request.principal.tenant_id
        invocation = TaskNodeInvocation(
            node,
            graph_id,
            request.principal,
            request.correlation,
            await self._dependency_results(graph_id, node, tenant_id=tenant_id),
        )
        try:
            completion = await self._runner.wait_bound(invocation, execution_id)
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001
            if isinstance(error, TaskNodeRunError):
                code = error.code.value
                digest = canonical_sha256(
                    {"graph_id": graph_id, "node_id": node.node_id, "code": code}
                )
                await self._repository.fail(
                    None,
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    execution_id=execution_id,
                    error_code=code,
                    error_digest=digest,
                )
                return
            if isinstance(error, AIError) and error.code is ErrorCode.TASK_DAG_INVALID:
                digest = canonical_sha256(
                    {"graph_id": graph_id, "node_id": node.node_id, "code": error.code.value}
                )
                await self._repository.fail(
                    None,
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    execution_id=execution_id,
                    error_code=error.code.value,
                    error_digest=digest,
                )
                return
            cause = (
                error
                if isinstance(error, AIError)
                else AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            )
            await self._defer_waiting_recovery(
                run,
                node,
                execution_id,
                cause=cause,
            )
            return
        try:
            await self._repository.complete(
                None,
                tenant_id=tenant_id,
                graph_id=graph_id,
                node_id=node.node_id,
                execution_id=completion.execution_id or execution_id,
                result_digest=completion.result_digest,
                result_payload=completion.result_payload,
                expanded_nodes=completion.expanded_nodes,
            )
        except AIError as error:
            if error.code not in _RECOVERY_UNKNOWN_CODES:
                raise
            await self._defer_waiting_recovery(
                run,
                node,
                execution_id,
                cause=error,
            )
            return
        await self._notify(run)

    async def _defer_waiting_recovery(
        self,
        run: _GraphRun,
        node: TaskNode,
        execution_id: str,
        *,
        cause: AIError,
    ) -> None:
        graph_id = run.request.graph_id
        tenant_id = run.request.principal.tenant_id
        code = (
            cause.code
            if cause.code in _RECOVERY_UNKNOWN_CODES
            else ErrorCode.STORAGE_RECOVERY_REQUIRED
        )
        digest = canonical_sha256(
            {
                "graph_id": graph_id,
                "node_id": node.node_id,
                "code": code.value,
            }
        )
        await self._repository.mark_recovery_required(
            None,
            tenant_id=tenant_id,
            graph_id=graph_id,
            node_id=node.node_id,
            execution_id=execution_id,
            error_code=code.value,
            error_digest=digest,
        )
        async with self._lock:
            current = self._graphs.get((tenant_id, graph_id))
            if current is run:
                run.closed = True
        _logger.warning(
            "waiting task graph requires recovery: graph=%s node=%s execution=%s code=%s",
            graph_id,
            node.node_id,
            execution_id,
            code.value,
        )
        await self._notify(run)

    async def _run_node(
        self,
        run: _GraphRun,
        node: TaskNode,
        lease_state: _LeaseState,
    ) -> None:
        request = run.request
        graph_id = request.graph_id
        tenant_id = request.principal.tenant_id
        dependency_results = await self._dependency_results(
            graph_id, node, tenant_id=tenant_id
        )
        heartbeat_stop = asyncio.Event()
        control = _TaskNodeRunControlImpl(
            self._repository,
            lease_state,
            tenant_id=tenant_id,
            on_activity=lambda: self._notify(run),
            on_handoff=heartbeat_stop.set,
        )
        runner_task = asyncio.create_task(
            self._runner.run(
                TaskNodeInvocation(
                    node,
                    graph_id,
                    request.principal,
                    request.correlation,
                    dependency_results,
                ),
                control=control,
            ),
            name=f"task-runner-{graph_id}-{node.node_id}",
        )
        heartbeat = asyncio.create_task(
            self._heartbeat(lease_state, tenant_id=tenant_id, stop=heartbeat_stop),
            name=f"task-heartbeat-{graph_id}-{node.node_id}",
        )
        try:
            done, _ = await asyncio.wait(
                (runner_task, heartbeat), return_when=asyncio.FIRST_COMPLETED
            )
            if (
                heartbeat in done
                and control.handed_off_execution_id is not None
                and not runner_task.done()
            ):
                await asyncio.wait(
                    (runner_task,), return_when=asyncio.FIRST_COMPLETED
                )
            if heartbeat in done and control.handed_off_execution_id is None:
                try:
                    heartbeat.result()
                except asyncio.CancelledError:
                    heartbeat_error: BaseException = AIError(
                        ErrorCode.STORAGE_RECOVERY_REQUIRED
                    )
                except BaseException as error:  # noqa: BLE001
                    heartbeat_error = error
                else:
                    heartbeat_error = AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
                if not runner_task.done():
                    runner_task.cancel()
                await asyncio.gather(runner_task, return_exceptions=True)
                if isinstance(heartbeat_error, AIError):
                    if heartbeat_error.code in _RECOVERY_UNKNOWN_CODES:
                        await self._defer_recovery(
                            run,
                            node,
                            lease_state,
                            cause=heartbeat_error,
                        )
                        return
                    if heartbeat_error.code in {
                        ErrorCode.TASK_FENCE_STALE,
                        ErrorCode.TASK_OWNER_CONFLICT,
                        ErrorCode.TASK_NOT_READY,
                    }:
                        return
                raise heartbeat_error
            try:
                completion = runner_task.result()
            except asyncio.CancelledError:
                raise
            except BaseException as error:  # noqa: BLE001
                await _stop_heartbeat(heartbeat_stop, heartbeat)
                if isinstance(error, AIError) and error.code in _RECOVERY_UNKNOWN_CODES:
                    await self._defer_recovery(
                        run,
                        node,
                        lease_state,
                        cause=error,
                        execution_id=(
                            error.execution_id
                            if isinstance(error, TaskNodeRunError)
                            else None
                        ),
                        waiting_execution_id=control.handed_off_execution_id,
                    )
                    return
                if isinstance(error, AIError) and error.code in {
                    ErrorCode.TASK_FENCE_STALE,
                    ErrorCode.TASK_OWNER_CONFLICT,
                    ErrorCode.TASK_NOT_READY,
                }:
                    return
                code = (
                    error.code.value
                    if isinstance(error, AIError)
                    else ErrorCode.TASK_NODE_FAILED.value
                )
                execution_id = (
                    error.execution_id
                    if isinstance(error, TaskNodeRunError)
                    else control.handed_off_execution_id
                )
                digest = canonical_sha256(
                    {"graph_id": graph_id, "node_id": node.node_id, "code": code}
                )
                async with lease_state.lock:
                    try:
                        await self._repository.fail(
                            None
                            if control.handed_off_execution_id is not None
                            else lease_state.lease,
                            tenant_id=tenant_id,
                            error_code=code,
                            error_digest=digest,
                            execution_id=execution_id,
                            graph_id=(
                                graph_id
                                if control.handed_off_execution_id is not None
                                else None
                            ),
                            node_id=(
                                node.node_id
                                if control.handed_off_execution_id is not None
                                else None
                            ),
                        )
                    except AIError as terminal_error:
                        if terminal_error.code is not ErrorCode.TASK_FENCE_STALE:
                            raise
                return
            await _stop_heartbeat(heartbeat_stop, heartbeat)
            recovery_error: AIError | None = None
            async with lease_state.lock:
                try:
                    await self._repository.complete(
                        None
                        if control.handed_off_execution_id is not None
                        else lease_state.lease,
                        tenant_id=tenant_id,
                        execution_id=(
                            completion.execution_id
                            or control.handed_off_execution_id
                        ),
                        result_digest=completion.result_digest,
                        result_payload=completion.result_payload,
                        graph_id=(
                            graph_id
                            if control.handed_off_execution_id is not None
                            else None
                        ),
                        node_id=(
                            node.node_id
                            if control.handed_off_execution_id is not None
                            else None
                        ),
                        expanded_nodes=completion.expanded_nodes,
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
                    recovery_error = error
            if recovery_error is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._defer_recovery(
                run,
                node,
                lease_state,
                cause=recovery_error,
                execution_id=completion.execution_id,
                waiting_execution_id=control.handed_off_execution_id,
            )
        except asyncio.CancelledError:
            if not runner_task.done():
                runner_task.cancel()
            await asyncio.gather(runner_task, return_exceptions=True)
            await _stop_heartbeat(heartbeat_stop, heartbeat)
            raise
        finally:
            execution_id = control.handed_off_execution_id
            if execution_id is not None:
                await self._release_execution_hold(
                    execution_id,
                    tenant_id=tenant_id,
                    hold_id=f"task:{graph_id}:{node.node_id}",
                )

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
        if state.status in _TERMINAL or state.status is TaskStatus.RECOVERY_REQUIRED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return False

    async def _defer_recovery(
        self,
        run: _GraphRun,
        node: TaskNode,
        lease_state: _LeaseState,
        *,
        cause: AIError,
        execution_id: "str | None" = None,
        waiting_execution_id: "str | None" = None,
    ) -> None:
        graph_id = run.request.graph_id
        tenant_id = run.request.principal.tenant_id
        recovery_code = (
            cause.code
            if cause.code in _RECOVERY_UNKNOWN_CODES
            else ErrorCode.STORAGE_RECOVERY_REQUIRED
        )
        digest = canonical_sha256(
            {
                "graph_id": graph_id,
                "node_id": node.node_id,
                "code": recovery_code.value,
            }
        )
        async with lease_state.lock:
            if waiting_execution_id is None:
                await self._repository.mark_recovery_required(
                    lease_state.lease,
                    tenant_id=tenant_id,
                    error_code=recovery_code.value,
                    error_digest=digest,
                    execution_id=execution_id,
                )
            else:
                await self._repository.mark_recovery_required(
                    None,
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    error_code=recovery_code.value,
                    error_digest=digest,
                    execution_id=waiting_execution_id,
                )
        async with self._lock:
            current = self._graphs.get((tenant_id, graph_id))
            if current is run:
                run.closed = True
        _logger.warning(
            "task graph requires recovery: graph=%s node=%s code=%s fence=%s",
            graph_id,
            node.node_id,
            recovery_code.value,
            lease_state.lease.fence,
        )
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
        while True:
            pending = {
                task
                for task in (
                    *self._runner.pending_background_tasks,
                    *self._runner.pending_cancelled_tasks,
                )
                if not task.done()
            }
            if not pending:
                break
            await asyncio.gather(
                *(asyncio.shield(task) for task in pending),
                return_exceptions=True,
            )
        failure = self._runner.background_failure
        if failure is not None:
            raise _copy_ai_error(failure)

    @staticmethod
    def _consume_run(run: _GraphRun, task: "asyncio.Task[None]") -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and run.failure is None:
            run.failure = _scheduler_failure(error, run.request.graph_id)


def _copy_ai_error(
    error: AIError,
    *,
    safe_details: "Mapping[str, JsonValue] | None" = None,
) -> AIError:
    return AIError(
        error.code,
        category=error.category,
        retryable=error.retryable,
        operation_id=error.operation_id,
        safe_details=(
            dict(error.safe_details) if safe_details is None else safe_details
        ),
        diagnostics=error.diagnostics,
    )


def _scheduler_failure(error: BaseException, graph_id: str) -> AIError:
    if isinstance(error, AIError):
        return _copy_ai_error(
            error,
            safe_details={**dict(error.safe_details), "graph_id": graph_id},
        )
    return AIError(
        ErrorCode.INTERNAL_ERROR,
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


def _scheduler_observation_fingerprint(
    view: TaskGraphView,
    states: "tuple[TaskNodeView, ...]",
    now: datetime,
) -> str:
    return canonical_sha256(
        {
            "status": view.status.value,
            "nodes": [
                {
                    "node_id": state.node_id,
                    "status": state.status.value,
                    "owner": state.owner,
                    "fence": state.fence,
                    "execution_id": state.execution_id,
                    "result_digest": state.result_digest,
                    "error_code": state.error_code,
                    "error_digest": state.error_digest,
                    "lease_phase": (
                        "none"
                        if state.lease_expires_at is None
                        else "expired"
                        if state.lease_expires_at <= now
                        else "live"
                    ),
                }
                for state in states
            ],
        }
    )


def _runnable(node: TaskNodeView, now: datetime) -> bool:
    return node.status is TaskStatus.READY or (
        node.status is TaskStatus.RUNNING
        and node.lease_expires_at is not None
        and node.lease_expires_at <= now
    )


def _reap_inflight(inflight: dict[str, _InflightNode]) -> bool:
    reaped = False
    for node_id, state in tuple(inflight.items()):
        if not state.task.done():
            continue
        inflight.pop(node_id, None)
        state.task.result()
        reaped = True
    return reaped


__all__ = [
    "LocalTaskGraphLauncher",
    "TaskNodeRunControl",
    "TaskNodeRunError",
    "TaskNodeRunResult",
    "TaskNodeRunner",
]
