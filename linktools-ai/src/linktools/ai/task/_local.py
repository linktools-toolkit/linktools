#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local TaskGraph scheduling over durable repository authority."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from linktools.core import environ

from ..core import (
    JsonValue,
    Page,
    Principal,
    TaskStatus,
    canonical_sha256,
    validate_lease_owner,
)
from ..errors import AIError, ErrorCode
from ._event import TaskEvent
from ._handler import TaskDependencyState, TaskEffectResolution
from ._graph import (
    TaskDependencyResult,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraphState,
    TaskGraphView,
    TaskLease,
    TaskNode,
    TaskNodeView,
    TaskResultRecord,
)
from ._metrics import _TaskMetricProjector
from ._runner import (
    TaskNodeInvocation,
    TaskNodeRunControl,
    TaskNodeRunError,
    TaskNodeRunner,
    TaskNodeRunResult,
)

_logger = environ.get_logger("ai.task.local")
_HEARTBEAT_SECONDS = 30.0
_LEASE_SECONDS = 60
_RECOVERY_UNKNOWN_CODES = frozenset(
    {
        ErrorCode.STORAGE_COMMIT_UNKNOWN,
        ErrorCode.STORAGE_RECOVERY_REQUIRED,
        ErrorCode.EXECUTION_START_UNKNOWN,
        ErrorCode.TOOL_EFFECT_UNKNOWN,
        ErrorCode.TASK_EFFECT_UNKNOWN,
    }
)
_EXECUTION_RESULT_CONFLICT_CODES = frozenset(
    {
        ErrorCode.STORAGE_CONFLICT,
        ErrorCode.EXECUTION_RESULT_CONFLICT,
        ErrorCode.TASK_RESULT_CONFLICT,
        ErrorCode.TASK_TERMINAL_CONFLICT,
        ErrorCode.TASK_FENCE_STALE,
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


@runtime_checkable
class _TaskDependencyPreparation(Protocol):
    async def prepare_node(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
    ) -> None: ...

    async def release_graph_dependencies(
        self,
        state: TaskGraphState,
        *,
        tenant_id: str,
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
    async def scheduler_state(
        self, graph_id: str, *, tenant_id: str
    ) -> "TaskGraphState": ...

    async def graph_state(
        self, graph_id: str, *, tenant_id: str
    ) -> "TaskGraphState | None": ...

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

    async def bind_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
    ) -> TaskNodeView: ...

    async def handoff_execution(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
        occupies_concurrency: bool = True,
    ) -> TaskNodeView: ...

    async def requeue_retry(
        self,
        lease: TaskLease,
        *,
        tenant_id: str,
        execution_id: str,
        next_attempt_at: datetime,
    ) -> TaskNodeView: ...

    async def requeue_waiting_retry(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        expected_fence: int,
        execution_id: str,
        next_attempt_at: datetime,
    ) -> TaskNodeView: ...

    async def requeue_recovery(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
        expected_fence: int,
        execution_id: "str | None" = None,
        next_attempt_at: "datetime | None" = None,
    ) -> TaskGraphView: ...

    async def mark_recovery_required(
        self,
        lease: "TaskLease | None",
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        error_origin: str = "node",
        safe_error_details: Mapping[str, JsonValue] | None = None,
        execution_id: "str | None" = None,
        graph_id: "str | None" = None,
        node_id: "str | None" = None,
        expected_fence: "int | None" = None,
    ) -> TaskNodeView: ...

    async def complete(
        self,
        lease: "TaskLease | None",
        *,
        tenant_id: str,
        execution_id: "str | None",
        result_digest: str,
        graph_id: "str | None" = None,
        node_id: "str | None" = None,
        expanded_nodes: "tuple[TaskNode, ...]" = (),
        expected_fence: "int | None" = None,
    ) -> object: ...

    async def fail(
        self,
        lease: "TaskLease | None",
        *,
        tenant_id: str,
        error_code: str,
        error_digest: str,
        error_origin: str = "node",
        safe_error_details: Mapping[str, JsonValue] | None = None,
        execution_id: "str | None" = None,
        graph_id: "str | None" = None,
        node_id: "str | None" = None,
        expected_fence: "int | None" = None,
    ) -> object: ...

    async def cancel_node(
        self,
        launch: TaskGraphLaunch,
        node_id: str,
        execution_id: str,
        *,
        invoke_effects: bool = True,
    ) -> TaskGraphView:
        tenant_id = launch.principal.tenant_id
        graph_id = launch.graph_id
        graph_state = await self._repository.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if graph_state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        state = next(
            (item for item in graph_state.node_states if item.node_id == node_id),
            None,
        )
        node = next(
            (item for item in graph_state.nodes if item.node_id == node_id),
            None,
        )
        if (
            state is None
            or node is None
            or state.execution_id != execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = (tenant_id, graph_id)
        async with self._lock:
            run = self._graphs.get(key)
        if run is not None and run.request != launch:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if run is not None and node_id in run.inflight:
            await self._quiesce_node(
                run,
                graph_state,
                node_id,
                invoke_cancel=invoke_effects,
            )
            await self._notify(run)
        elif invoke_effects and state.status not in _TERMINAL:
            dependency_results, dependency_states = await self._dependency_context(
                graph_id,
                node,
                tenant_id=tenant_id,
            )
            await self._runner.cancel(
                TaskNodeInvocation(
                    node,
                    graph_id,
                    launch.principal,
                    launch.correlation,
                    dependency_results,
                    state.execution_id,
                    dependency_states=dependency_states,
                )
            )

        refreshed = await self._repository.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if refreshed is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = next(
            (item for item in refreshed.node_states if item.node_id == node_id),
            None,
        )
        if state is None or state.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state.status not in _TERMINAL:
            settlement_run = run or _GraphRun(launch, self._owner)
            await self._settle_bound_node(
                settlement_run,
                node,
                execution_id,
                state.fence,
                wait_for_completion=False,
            )
        view = await self._repository.get_graph(
            graph_id,
            tenant_id=tenant_id,
        )
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return view

    async def _quiesce_node(
        self,
        run: _GraphRun,
        graph_state: TaskGraphState,
        node_id: str,
        *,
        invoke_cancel: bool,
    ) -> None:
        inflight = run.inflight.get(node_id)
        if inflight is None:
            return
        if invoke_cancel:
            inflight.invoke_cancel = True
        if inflight.quiesce_task is None:
            inflight.quiesce_task = asyncio.create_task(
                self._quiesce_node_owned(
                    run,
                    graph_state,
                    node_id,
                    inflight,
                    invoke_cancel=invoke_cancel,
                ),
                name=f"task-quiesce-{graph_state.graph_id}-{node_id}",
            )
        await asyncio.shield(inflight.quiesce_task)

    async def _quiesce_node_owned(
        self,
        run: _GraphRun,
        graph_state: TaskGraphState,
        node_id: str,
        inflight: _InflightNode,
        *,
        invoke_cancel: bool,
    ) -> None:
        cancellation_error: BaseException | None = None
        try:
            current_state = await self._repository.graph_state(
                graph_state.graph_id,
                tenant_id=run.request.principal.tenant_id,
            )
            if current_state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            node = next(
                (item for item in current_state.nodes if item.node_id == node_id),
                None,
            )
            state = next(
                (
                    item
                    for item in current_state.node_states
                    if item.node_id == node_id
                ),
                None,
            )
            if node is None or state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        except BaseException as error:  # noqa: BLE001
            cancellation_error = error
        finally:
            if not inflight.task.done():
                inflight.task.cancel()
            await asyncio.gather(inflight.task, return_exceptions=True)
        if (
            cancellation_error is None
            and (invoke_cancel or inflight.invoke_cancel)
            and current_state is not None
            and node is not None
            and state is not None
            and state.status
            in {
                TaskStatus.RUNNING,
                TaskStatus.WAITING,
                TaskStatus.READY,
            }
        ):
            try:
                dependency_results, dependency_states = await self._dependency_context(
                    current_state.graph_id,
                    node,
                    tenant_id=run.request.principal.tenant_id,
                )
                await self._runner.cancel(
                    TaskNodeInvocation(
                        node,
                        current_state.graph_id,
                        run.request.principal,
                        run.request.correlation,
                        dependency_results,
                        state.execution_id,
                        dependency_states=dependency_states,
                    )
                )
            except BaseException as error:  # noqa: BLE001
                cancellation_error = error
        if run.inflight.get(node_id) is inflight:
            run.inflight.pop(node_id, None)
        if cancellation_error is not None:
            if isinstance(cancellation_error, asyncio.CancelledError):
                raise cancellation_error
            if isinstance(cancellation_error, AIError):
                raise cancellation_error
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details={
                    "phase": "task_node_cancel_cleanup",
                    "graph_id": graph_state.graph_id,
                    "node_id": node_id,
                },
            ) from cancellation_error

    async def _quiesce_persisted_nodes(
        self,
        run: _GraphRun,
        graph_state: TaskGraphState,
    ) -> None:
        states = {
            node_state.node_id: node_state
            for node_state in graph_state.node_states
        }
        for node_id in tuple(run.inflight):
            state = states.get(node_id)
            if state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if state.status not in {
                TaskStatus.CANCELLED,
                TaskStatus.RECOVERY_REQUIRED,
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
            }:
                continue
            await self._quiesce_node(
                run,
                graph_state,
                node_id,
                invoke_cancel=False,
            )

    async def settle_cancel(
        self,
        launch: TaskGraphLaunch,
        *,
        invoke_effects: bool,
    ) -> TaskGraphView:
        graph_id = launch.graph_id
        tenant_id = launch.principal.tenant_id
        key = (tenant_id, graph_id)
        async with self._lock:
            run = self._graphs.get(key)
        if run is not None and run.request != launch:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        settlement_run = run or _GraphRun(launch, self._owner)
        graph_state = await self._repository.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if graph_state is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        definitions = {node.node_id: node for node in graph_state.nodes}
        unknown_effect = False

        for node_state in graph_state.node_states:
            if node_state.status in _TERMINAL:
                if run is not None and node_state.node_id in run.inflight:
                    await self._quiesce_node(
                        run,
                        graph_state,
                        node_state.node_id,
                        invoke_cancel=invoke_effects,
                    )
                continue
            if node_state.status is TaskStatus.RECOVERY_REQUIRED:
                if run is not None and node_state.node_id in run.inflight:
                    await self._quiesce_node(
                        run,
                        graph_state,
                        node_state.node_id,
                        invoke_cancel=False,
                    )
                if node_state.execution_id is not None:
                    node = definitions.get(node_state.node_id)
                    if node is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    await self._settle_bound_node(
                        settlement_run,
                        node,
                        node_state.execution_id,
                        node_state.fence,
                        wait_for_completion=False,
                    )
                    refreshed = await self._repository.graph_state(
                        graph_id,
                        tenant_id=tenant_id,
                    )
                    if refreshed is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    graph_state = refreshed
                continue
            if node_state.execution_id is None:
                if run is not None and node_state.node_id in run.inflight:
                    inflight = run.inflight.get(node_state.node_id)
                    await self._quiesce_node(
                        run,
                        graph_state,
                        node_state.node_id,
                        invoke_cancel=False,
                    )
                    if inflight is not None and inflight.lease_state is not None:
                        await self._mark_cancel_recovery(
                            settlement_run,
                            definitions[node_state.node_id],
                            node_state,
                            lease=inflight.lease_state.lease,
                            cause=AIError(ErrorCode.TASK_EFFECT_UNKNOWN),
                        )
                        unknown_effect = True
                elif node_state.status is TaskStatus.RUNNING:
                    raise AIError(
                        ErrorCode.STORAGE_RECOVERY_REQUIRED,
                        safe_details={
                            "phase": "task_cancel_unbound_node",
                            "graph_id": graph_id,
                            "node_id": node_state.node_id,
                        },
                    )
                continue

            node = definitions.get(node_state.node_id)
            if node is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            inflight = (
                None
                if run is None
                else run.inflight.get(node_state.node_id)
            )
            if inflight is not None:
                try:
                    await self._quiesce_node(
                        run,
                        graph_state,
                        node_state.node_id,
                        invoke_cancel=invoke_effects,
                    )
                except BaseException as error:  # noqa: BLE001
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    await self._mark_cancel_recovery(
                        settlement_run,
                        node,
                        node_state,
                        lease=(
                            inflight.lease_state.lease
                            if inflight.lease_state is not None
                            else None
                        ),
                        cause=error,
                    )
                    unknown_effect = True
                    continue
            elif invoke_effects:
                try:
                    dependency_results, dependency_states = await self._dependency_context(
                        graph_id,
                        node,
                        tenant_id=tenant_id,
                    )
                    await self._runner.cancel(
                        TaskNodeInvocation(
                            node,
                            graph_id,
                            launch.principal,
                            launch.correlation,
                            dependency_results,
                            node_state.execution_id,
                            dependency_states=dependency_states,
                        )
                    )
                except BaseException as error:  # noqa: BLE001
                    if isinstance(error, asyncio.CancelledError):
                        raise
                    await self._mark_cancel_recovery(
                        settlement_run,
                        node,
                        node_state,
                        lease=None,
                        cause=error,
                    )
                    unknown_effect = True
                    continue

            await self._settle_bound_node(
                settlement_run,
                node,
                node_state.execution_id,
                node_state.fence,
                wait_for_completion=invoke_effects,
            )
            graph_state = await self._repository.graph_state(
                graph_id,
                tenant_id=tenant_id,
            )
            if graph_state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if run is not None:
            async with self._lock:
                current = self._graphs.get(key)
                if current is run:
                    current.closed = True
                    task = current.task
                else:
                    task = None
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            await self._notify(run)

        view = await self._repository.get_graph(graph_id, tenant_id=tenant_id)
        if view is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._metric_projector is not None and view.status in _TERMINAL:
            self._metric_projector.trigger(graph_id, tenant_id=tenant_id)
        if unknown_effect:
            raise AIError(
                ErrorCode.TASK_EFFECT_UNKNOWN,
                safe_details={"phase": "task_graph_cancel"},
            )
        return view

    async def _mark_cancel_recovery(
        self,
        run: _GraphRun,
        node: TaskNode,
        state: TaskNodeView,
        *,
        lease: TaskLease | None,
        cause: BaseException,
    ) -> None:
        code = (
            cause.code
            if isinstance(cause, AIError)
            and cause.code in _RECOVERY_UNKNOWN_CODES
            else ErrorCode.TASK_EFFECT_UNKNOWN
        )
        digest = canonical_sha256(
            {
                "graph_id": run.request.graph_id,
                "node_id": node.node_id,
                "code": code.value,
            }
        )
        await self._repository.mark_recovery_required(
            lease,
            tenant_id=run.request.principal.tenant_id,
            error_code=code.value,
            error_digest=digest,
            error_origin="execution",
            safe_error_details=(
                cause.safe_details if isinstance(cause, AIError) else {}
            ),
            execution_id=state.execution_id,
            graph_id=None if lease is not None else run.request.graph_id,
            node_id=None if lease is not None else node.node_id,
            expected_fence=None if lease is not None else state.fence,
        )
        await self._notify(run)


    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView:
        return await self.settle_cancel(launch, invoke_effects=True)

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

    def graph_failure(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> AIError | None:
        run = self._graphs.get((tenant_id, graph_id))
        if run is None or run.failure is None:
            return None
        return _copy_ai_error(run.failure)

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

    async def _run_graph(self, run: _GraphRun) -> None:
        request = run.request
        tenant_id = request.principal.tenant_id
        inflight = run.inflight
        observed_fingerprint: str | None = None
        try:
            while not run.closed:
                try:
                    state = await self._repository.scheduler_state(
                        request.graph_id,
                        tenant_id=tenant_id,
                    )
                except AIError as error:
                    if error.code is not ErrorCode.STORAGE_CONFLICT:
                        raise
                    await asyncio.sleep(0)
                    continue
                view = TaskGraphView(
                    state.graph_id,
                    state.status,
                    state.nodes,
                )
                states = state.node_states
                now = datetime.now(timezone.utc)
                fingerprint = _scheduler_observation_fingerprint(view, states, now)
                if fingerprint != observed_fingerprint:
                    observed_fingerprint = fingerprint
                    run.observation_backoff = 1.0
                    await self._notify(run)
                await self._quiesce_persisted_nodes(run, state)
                if view.status is TaskStatus.RECOVERY_REQUIRED or view.status in _TERMINAL:
                    should_close, latest = await self._close_at_graph_boundary(run)
                    if should_close:
                        if latest.status in _TERMINAL:
                            if isinstance(self._runner, _TaskDependencyPreparation):
                                await self._runner.release_graph_dependencies(
                                    latest,
                                    tenant_id=tenant_id,
                                )
                            if self._metric_projector is not None:
                                self._metric_projector.trigger(
                                    request.graph_id,
                                    tenant_id=tenant_id,
                                )
                        return
                    continue
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
                static = {node.node_id: node for node in state.nodes}
                for state in states:
                    if state.status is not TaskStatus.WAITING:
                        continue
                    if state.node_id in inflight:
                        continue
                    node = static.get(state.node_id)
                    if node is None or state.execution_id is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if (
                        node.task is not None
                        and node.task.id == "linktools.ai.input"
                        and node.task.revision == 1
                    ):
                        continue
                    task = asyncio.create_task(
                        self._wait_bound_node(
                            run,
                            node,
                            state.execution_id,
                            state.fence,
                        ),
                        name=f"task-wait-{request.graph_id}-{node.node_id}",
                    )
                    inflight[node.node_id] = _InflightNode(task, None)
                    await self._notify(run)
                waiting = {
                    state.node_id
                    for state in states
                    if state.status is TaskStatus.WAITING
                    and state.occupies_concurrency
                }
                used = persisted | waiting | {
                    node_id
                    for node_id, value in inflight.items()
                    if value.lease_state is not None
                }
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
                    if isinstance(
                        self._runner,
                        _TaskDependencyPreparation,
                    ):
                        await self._runner.prepare_node(
                            node,
                            graph_id=request.graph_id,
                            principal=request.principal,
                        )
                    elif node.input_refs:
                        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
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
            cleanup = asyncio.create_task(
                self._drain_inflight(run),
                name=f"task-graph-drain-{request.graph_id}",
            )
            cancelled = False
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    cancelled = True
            cleanup.result()
            if cancelled:
                raise asyncio.CancelledError
            run.closed = True
            await self._notify(run)
            if run.failure is None:
                key = (tenant_id, request.graph_id)
                async with self._lock:
                    if self._graphs.get(key) is run:
                        self._graphs.pop(key, None)

    async def _close_at_graph_boundary(
        self,
        run: _GraphRun,
    ) -> tuple[bool, TaskGraphState]:
        key = (run.request.principal.tenant_id, run.request.graph_id)
        async with self._lock:
            if self._graphs.get(key) is not run:
                state = await self._repository.graph_state(
                    run.request.graph_id,
                    tenant_id=run.request.principal.tenant_id,
                )
                if state is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return True, state
            state = await self._repository.graph_state(
                run.request.graph_id,
                tenant_id=run.request.principal.tenant_id,
            )
            if state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if state.status not in _TERMINAL | {TaskStatus.RECOVERY_REQUIRED}:
                return False, state
            run.closed = True
            return True, state

    async def _drain_inflight(self, run: _GraphRun) -> None:
        values = tuple(run.inflight.items())
        for _node_id, value in values:
            if value.quiesce_task is None and not value.task.done():
                value.task.cancel()
        if values:
            await asyncio.gather(
                *(value.task for _node_id, value in values),
                return_exceptions=True,
            )
            quiescence = tuple(
                value.quiesce_task
                for _node_id, value in values
                if value.quiesce_task is not None
            )
            if quiescence:
                outcomes = await asyncio.gather(
                    *(asyncio.shield(task) for task in quiescence),
                    return_exceptions=True,
                )
                for outcome in outcomes:
                    if isinstance(outcome, BaseException) and not isinstance(
                        outcome, asyncio.CancelledError
                    ):
                        if run.failure is None:
                            run.failure = _scheduler_failure(
                                outcome,
                                run.request.graph_id,
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
        expected_fence: int,
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
            await self._settle_bound_node(
                run,
                node,
                execution_id,
                expected_fence,
            )
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
        expected_fence: int,
        *,
        wait_for_completion: bool = True,
    ) -> None:
        request = run.request
        graph_id = request.graph_id
        tenant_id = request.principal.tenant_id
        dependency_results, dependency_states = await self._dependency_context(
            graph_id,
            node,
            tenant_id=tenant_id,
        )
        invocation = TaskNodeInvocation(
            node,
            graph_id,
            request.principal,
            request.correlation,
            dependency_results,
            dependency_states=dependency_states,
        )
        try:
            if wait_for_completion:
                completion = await self._runner.wait_bound(invocation, execution_id)
            else:
                completion = await self._runner.inspect_bound(
                    invocation,
                    execution_id,
                )
                if completion is None:
                    return
        except asyncio.CancelledError:
            raise
        except BaseException as error:  # noqa: BLE001
            if isinstance(error, TaskNodeRunError):
                code = error.code.value
                digest = canonical_sha256(
                    {"graph_id": graph_id, "node_id": node.node_id, "code": code}
                )
                if error.code in _RECOVERY_UNKNOWN_CODES:
                    await self._defer_waiting_recovery(
                        run,
                        node,
                        execution_id,
                        expected_fence=expected_fence,
                        cause=error,
                    )
                    return
                if error.code is ErrorCode.EXECUTION_CANCELLED:
                    await self._repository.cancel_node(
                        graph_id,
                        node.node_id,
                        tenant_id=tenant_id,
                        execution_id=execution_id,
                        cancel_confirmed=True,
                        expected_fence=expected_fence,
                    )
                    await self._notify(run)
                    return
                if error.code in _EXECUTION_RESULT_CONFLICT_CODES:
                    await self._defer_waiting_recovery(
                        run,
                        node,
                        execution_id,
                        expected_fence=expected_fence,
                        cause=AIError(
                            ErrorCode.STORAGE_RECOVERY_REQUIRED,
                            safe_details={
                                "phase": "execution_result_settlement",
                                "cause": error.code.value,
                            },
                        ),
                    )
                    return
                await self._repository.fail(
                    None,
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    execution_id=execution_id,
                    error_code=code,
                    error_digest=digest,
                    error_origin="execution",
                    expected_fence=expected_fence,
                )
                await self._notify(run)
                return
            if isinstance(error, AIError) and error.code in _RECOVERY_UNKNOWN_CODES:
                await self._defer_waiting_recovery(
                    run,
                    node,
                    execution_id,
                    expected_fence=expected_fence,
                    cause=error,
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
                    safe_error_details=error.safe_details,
                    expected_fence=expected_fence,
                )
                await self._notify(run)
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
                expected_fence=expected_fence,
                cause=cause,
            )
            return
        if completion.execution_id not in {None, execution_id}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if completion.deferred:
            await self._notify(run)
            return
        try:
            if completion.retry_at is not None:
                await self._repository.requeue_waiting_retry(
                    graph_id,
                    node.node_id,
                    tenant_id=tenant_id,
                    expected_fence=expected_fence,
                    execution_id=execution_id,
                    next_attempt_at=completion.retry_at,
                )
            else:
                await self._repository.complete(
                    None,
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    execution_id=execution_id,
                    result_digest=completion.result_digest,
                    expanded_nodes=completion.expanded_nodes,
                    expected_fence=expected_fence,
                )
        except AIError as error:
            if error.code not in (
                _RECOVERY_UNKNOWN_CODES | _EXECUTION_RESULT_CONFLICT_CODES
            ):
                raise
            await self._defer_waiting_recovery(
                run,
                node,
                execution_id,
                expected_fence=expected_fence,
                cause=(
                    error
                    if error.code in _RECOVERY_UNKNOWN_CODES
                    else AIError(
                        ErrorCode.STORAGE_RECOVERY_REQUIRED,
                        safe_details={
                            "phase": "task_result_settlement",
                            "cause": error.code.value,
                        },
                    )
                ),
            )
            return
        await self._notify(run)

    async def _defer_waiting_recovery(
        self,
        run: _GraphRun,
        node: TaskNode,
        execution_id: str,
        *,
        expected_fence: int,
        cause: AIError,
    ) -> None:
        graph_id = run.request.graph_id
        tenant_id = run.request.principal.tenant_id
        current_state = await self._repository.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if current_state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current_node = next(
            (
                value
                for value in current_state.node_states
                if value.node_id == node.node_id
            ),
            None,
        )
        if current_node is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (
            current_node.execution_id != execution_id
            or current_node.fence != expected_fence
            or current_node.status in _TERMINAL
            or current_node.status is not TaskStatus.WAITING
        ):
            await self._notify(run)
            return
        if current_node.status is TaskStatus.RECOVERY_REQUIRED:
            await self._notify(run)
            return
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
            error_origin="execution",
            safe_error_details=cause.safe_details,
            expected_fence=expected_fence,
        )
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
        dependency_results, dependency_states = await self._dependency_context(
            graph_id,
            node,
            tenant_id=tenant_id,
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
                    execution_id=lease_state.lease.execution_id,
                    dependency_states=dependency_states,
                    task_lease=lease_state.lease,
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
                if not isinstance(error, asyncio.CancelledError):
                    _logger.error(
                        "task node runner failed: graph=%s node=%s type=%s",
                        graph_id,
                        node.node_id,
                        type(error).__name__,
                        exc_info=True,
                    )
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
                if (
                    control.execution_id is not None
                    and isinstance(error, AIError)
                    and error.code in _EXECUTION_RESULT_CONFLICT_CODES
                ):
                    await self._settle_bound_node(
                        run,
                        node,
                        control.execution_id,
                        lease_state.lease.fence,
                        wait_for_completion=False,
                    )
                    return
                if isinstance(error, AIError) and error.code in {
                    ErrorCode.TASK_FENCE_STALE,
                    ErrorCode.TASK_OWNER_CONFLICT,
                    ErrorCode.TASK_NOT_READY,
                }:
                    return
                if (
                    isinstance(error, TaskNodeRunError)
                    and error.code is ErrorCode.EXECUTION_CANCELLED
                ):
                    async with lease_state.lock:
                        try:
                            await self._repository.cancel_node(
                                graph_id,
                                node.node_id,
                                tenant_id=tenant_id,
                                execution_id=error.execution_id,
                                cancel_confirmed=True,
                                expected_fence=lease_state.lease.fence,
                            )
                        except AIError as terminal_error:
                            if terminal_error.code is not ErrorCode.TASK_FENCE_STALE:
                                raise
                    await self._notify(run)
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
                            error_origin=(
                                "execution"
                                if isinstance(error, TaskNodeRunError)
                                else "node"
                            ),
                            safe_error_details=(
                                {}
                                if isinstance(error, TaskNodeRunError)
                                or not isinstance(error, AIError)
                                else error.safe_details
                            ),
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
                            expected_fence=(
                                lease_state.lease.fence
                                if control.handed_off_execution_id is not None
                                else None
                            ),
                        )
                    except AIError as terminal_error:
                        if terminal_error.code is not ErrorCode.TASK_FENCE_STALE:
                            raise
                return
            if completion.deferred:
                execution_id = (
                    completion.execution_id
                    or control.handed_off_execution_id
                )
                if execution_id is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    control.handed_off_execution_id is not None
                    and control.handed_off_execution_id != execution_id
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if control.handed_off_execution_id is None:
                    await control.handoff_execution(execution_id)
            await _stop_heartbeat(heartbeat_stop, heartbeat)
            recovery_error: AIError | None = None
            async with lease_state.lock:
                try:
                    if completion.deferred:
                        await self._notify(run)
                        return
                    if completion.retry_at is not None:
                        if completion.execution_id is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        if control.handed_off_execution_id is None:
                            await self._repository.requeue_retry(
                                lease_state.lease,
                                tenant_id=tenant_id,
                                execution_id=completion.execution_id,
                                next_attempt_at=completion.retry_at,
                            )
                        else:
                            if (
                                control.handed_off_execution_id
                                != completion.execution_id
                            ):
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                            await self._repository.requeue_waiting_retry(
                                graph_id,
                                node.node_id,
                                tenant_id=tenant_id,
                                expected_fence=lease_state.lease.fence,
                                execution_id=completion.execution_id,
                                next_attempt_at=completion.retry_at,
                            )
                        await self._notify(run)
                        return
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
                        expected_fence=(
                            lease_state.lease.fence
                            if control.handed_off_execution_id is not None
                            else None
                        ),
                    )
                    return
                except AIError as error:
                    if error.code not in (
                        _RECOVERY_UNKNOWN_CODES | _EXECUTION_RESULT_CONFLICT_CODES
                    ):
                        raise
                    if await self._completion_committed(
                        graph_id,
                        node.node_id,
                        completion,
                        tenant_id=tenant_id,
                        expected_fence=lease_state.lease.fence,
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
            is_input_wait = (
                node.task is not None
                and node.task.id == "linktools.ai.input"
                and node.task.revision == 1
            )
            if execution_id is not None and not is_input_wait:
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
        expected_fence: int,
    ) -> bool:
        graph_state = await self._repository.graph_state(
            graph_id,
            tenant_id=tenant_id,
        )
        if graph_state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = next(
            (value for value in graph_state.node_states if value.node_id == node_id),
            None,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state.status is TaskStatus.SUCCEEDED:
            if (
                state.result_digest != completion.result_digest
                or state.execution_id != completion.execution_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return True
        if (
            state.fence != expected_fence
            or state.status in _TERMINAL
            or state.status is TaskStatus.RECOVERY_REQUIRED
        ):
            return True
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
            error_origin = (
                "execution"
                if isinstance(cause, TaskNodeRunError)
                or waiting_execution_id is not None
                else "node"
            )
            if waiting_execution_id is None:
                await self._repository.mark_recovery_required(
                    lease_state.lease,
                    tenant_id=tenant_id,
                    error_code=recovery_code.value,
                    error_digest=digest,
                    error_origin=error_origin,
                    safe_error_details=cause.safe_details,
                    execution_id=execution_id,
                    expected_fence=lease_state.lease.fence,
                )
            else:
                await self._repository.mark_recovery_required(
                    None,
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    node_id=node.node_id,
                    error_code=recovery_code.value,
                    error_digest=digest,
                    error_origin=error_origin,
                    safe_error_details=cause.safe_details,
                    execution_id=waiting_execution_id,
                    expected_fence=lease_state.lease.fence,
                )
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

    async def _dependency_context(
        self,
        graph_id: str,
        node: TaskNode,
        *,
        tenant_id: str,
    ) -> "tuple[dict[str, TaskDependencyResult], dict[str, TaskDependencyState]]":
        if not node.dependencies:
            return {}, {}
        raw_states = await self._repository.list_nodes(
            graph_id,
            tenant_id=tenant_id,
        )
        by_id = {state.node_id: state for state in raw_states}
        states = self._project_dependency_states(node, by_id)
        successful = tuple(
            dependency_id
            for dependency_id in node.dependencies
            if states[dependency_id].status is TaskStatus.SUCCEEDED
        )
        records = (
            {}
            if not successful
            else await self._repository.get_results(
                graph_id,
                successful,
                tenant_id=tenant_id,
            )
        )
        results: dict[str, TaskDependencyResult] = {}
        for dependency_id in successful:
            state = by_id[dependency_id]
            semantic = states[dependency_id]
            record = records.get(dependency_id)
            if (
                state.execution_id is None
                or record is None
                or record.result_digest != semantic.result_digest
                or (
                    record.execution_id is not None
                    and record.execution_id != state.execution_id
                )
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result_digest = semantic.result_digest
            if result_digest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            results[dependency_id] = TaskDependencyResult(
                result_digest,
                state.execution_id,
            )
        if (
            node.dependency_policy == "all_succeeded"
            and set(results) != set(node.dependencies)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return results, states

    async def _dependency_states(
        self,
        graph_id: str,
        node: TaskNode,
        *,
        tenant_id: str,
    ) -> "dict[str, TaskDependencyState]":
        if not node.dependencies:
            return {}
        raw_states = await self._repository.list_nodes(
            graph_id,
            tenant_id=tenant_id,
        )
        return self._project_dependency_states(
            node,
            {state.node_id: state for state in raw_states},
        )

    @staticmethod
    def _project_dependency_states(
        node: TaskNode,
        states: Mapping[str, TaskNodeView],
    ) -> "dict[str, TaskDependencyState]":
        values: dict[str, TaskDependencyState] = {}
        for dependency_id in node.dependencies:
            state = states.get(dependency_id)
            if state is None or state.status not in _TERMINAL:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                node.dependency_policy == "all_succeeded"
                and state.status is not TaskStatus.SUCCEEDED
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values[dependency_id] = TaskDependencyState(
                state.status,
                state.result_digest,
                state.error_code,
                state.error_digest,
            )
        return values

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
    return (
        node.status is TaskStatus.READY
        and (node.next_attempt_at is None or node.next_attempt_at <= now)
    ) or (
        node.status is TaskStatus.RUNNING
        and node.lease_expires_at is not None
        and node.lease_expires_at <= now
    )


def _reap_inflight(inflight: dict[str, _InflightNode]) -> bool:
    reaped = False
    for node_id, state in tuple(inflight.items()):
        if not state.task.done() or state.quiesce_task is not None:
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
