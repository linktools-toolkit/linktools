#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic task_graph scheduling and node-state convergence."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Awaitable, Callable, Protocol

from ...errors import (
    ChildExecutionPlatformError,
    CleanupDiagnostic,
    NodeLeaseLostError,
    ParentLeaseGuardError,
    ParentLeaseLostError,
    StorageConflictError,
    StorageError,
    SwarmLimitExceededError,
    TaskGraphCleanupError,
    TaskGraphInvariantError,
)
from ...execution.domain import RunError
from ...execution.commands import ParentLeaseGuard
from ...execution.identifiers import child_run_id
from ..models import (
    TaskExecution,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
    UsageAccumulator,
    merge_newer_cumulative_usage,
)
from ..store import TaskStore
from .validation import all_terminal, classify_readiness

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...governance.identity import PrincipalContext
    from .limits import SwarmLimits


DEFAULT_LEASE_DURATION = timedelta(minutes=10)
CONTROL_POLL_INTERVAL = 1.0


@dataclass(frozen=True, slots=True)
class NodeRunRequest:
    """The immutable inputs required to drive one child agent."""

    node: TaskNode
    execution: TaskExecution
    owner: str
    fence: int
    child_run_id: str
    dependencies: "tuple[TaskExecution, ...]"
    parent_guard: "ParentLeaseGuard"


@dataclass(frozen=True, slots=True)
class NodeUsageSnapshot:
    usage: TaskUsage
    snapshot_revision: int
    terminal: bool


class StopReason(StrEnum):
    NORMAL = "normal"
    USER_CANCELLED = "user_cancelled"
    CALLER_CANCELLED = "caller_cancelled"
    TIMEOUT = "timeout"
    TOKEN_LIMIT = "token_limit"
    COST_LIMIT = "cost_limit"
    PLATFORM_FAILURE = "platform_failure"
    PARENT_LEASE_LOST = "parent_lease_lost"


@dataclass(frozen=True, slots=True)
class NodeRunResult:
    """A child business outcome mapped to one task terminal transition."""

    status: TaskStatus
    result: "object | None" = None
    error: "RunError | None" = None
    reason: "str | None" = None
    usage: TaskUsage = field(default_factory=TaskUsage)
    snapshot_revision: int = 0


class NodeRunner(Protocol):
    """Drive children and request their cancellation; never write TaskStore."""

    async def run(self, request: NodeRunRequest) -> NodeRunResult: ...

    async def request_cancel(
        self, *, child_run_id: str, principal: "PrincipalContext | None", reason: str
    ) -> None: ...

    async def read_usage(self, *, child_run_id: str) -> NodeUsageSnapshot: ...


class ControlGate(Protocol):
    """Read parent control state and enforce run-level limits."""

    async def check(self) -> None: ...

    async def check_before_launch(self) -> None: ...

    def next_wake_delay(self, *, now_monotonic: float) -> float: ...

    @property
    def cancel_requested(self) -> bool: ...

    def record_usage(self, usage: TaskUsage) -> None: ...


@dataclass(slots=True)
class ActiveNode:
    node_id: str
    execution_id: str
    child_run_id: str
    owner: str
    fence: int
    task: "asyncio.Task[None]"
    next_renew_at: float


class TaskGraphEngine:
    """Own every TaskExecution transition for one immutable TaskPlan."""

    def __init__(
        self,
        store: TaskStore,
        runner: NodeRunner,
        gate: ControlGate,
        *,
        limits: "SwarmLimits",
        owner: str,
        parent_run_id: str,
        parent_owner: str,
        parent_fence: int,
        principal: "PrincipalContext | None" = None,
        on_skip: "Callable[[str, tuple[str, ...]], Awaitable[None]] | None" = None,
        on_node_terminal: "Callable[[str, NodeRunResult], Awaitable[None]] | None" = None,
        logger=None,
    ) -> None:
        self._store = store
        self._runner = runner
        self._gate = gate
        self._limits = limits
        self._owner = owner
        self._parent_run_id = parent_run_id
        self._parent_owner = parent_owner
        self._parent_fence = parent_fence
        self._principal = principal
        self._on_skip = on_skip
        self._on_node_terminal = on_node_terminal
        self._node_lease_duration = DEFAULT_LEASE_DURATION
        self._active_by_node: "dict[str, ActiveNode]" = {}
        self._active_by_task: "dict[asyncio.Task[None], ActiveNode]" = {}
        self._requests_by_node: "dict[str, NodeRunRequest]" = {}
        self._usage_by_node: "dict[str, TaskUsage]" = {}
        self._usage_revision_by_node: "dict[str, int]" = {}
        self._terminal_event_nodes: "set[str]" = set()
        self._raced_claimed_nodes: "set[str]" = set()
        self._stopping = False
        self._stop_reason = StopReason.NORMAL
        self._completion = asyncio.Event()
        self._logger = logger

    @property
    def active_count(self) -> int:
        return len(self._active_by_node)

    async def execute(self, plan: TaskPlan) -> TaskUsage:
        """Run the graph and always perform the final convergence pass."""
        primary_error: BaseException | None = None
        stop_reason = StopReason.NORMAL
        result: TaskUsage | None = None
        try:
            result = await self._run_loop(plan)
        except asyncio.CancelledError as exc:
            primary_error = exc
            stop_reason = StopReason.CALLER_CANCELLED
        except BaseException as exc:
            primary_error = exc
            stop_reason = self._classify_stop_reason(exc)
        self._stop_reason = stop_reason
        try:
            await self._shutdown_active_nodes(
                plan,
                stop_reason=stop_reason,
                primary_error=primary_error,
            )
        except BaseException as cleanup_error:
            diagnostic = CleanupDiagnostic(
                stage="engine_cleanup",
                node_id=None,
                error_type=type(cleanup_error).__name__,
                safe_message="task graph cleanup failed",
            )
            if self._logger is not None:
                self._logger.error(
                    "task graph cleanup failed parent_run_id=%s error_type=%s",
                    self._parent_run_id,
                    type(cleanup_error).__name__,
                )
            raise TaskGraphCleanupError(
                primary_error=primary_error,
                cleanup_error=cleanup_error,
                diagnostics=(diagnostic,),
            ) from cleanup_error
        if primary_error is not None:
            raise primary_error
        if result is None:
            raise TaskGraphInvariantError("task graph completed without usage")
        return result

    async def _run_loop(self, plan: TaskPlan) -> TaskUsage:
        while True:
            await self._gate.check()
            if self._gate.cancel_requested:
                raise SwarmLimitExceededError(
                    "parent cancellation requested",
                    kind="parent_cancelled",
                )
            await self._renew_due_nodes(time.monotonic())
            await self._reap_completed_tasks()

            executions = await self._execution_map(plan)
            await self._propagate_skips_to_fixpoint(plan, executions)
            executions = await self._execution_map(plan)

            if all_terminal(executions):
                if self._active_by_node:
                    raise TaskGraphInvariantError(
                        "terminal executions still have active local tasks"
                    )
                return await self._sum_usage(plan)

            if self._has_launchable_ready_node(plan, executions):
                await self._gate.check_before_launch()
            await self._launch_ready_nodes(plan, executions)
            await self._reap_completed_tasks()
            executions = await self._execution_map(plan)

            if all_terminal(executions):
                if self._active_by_node:
                    raise TaskGraphInvariantError(
                        "terminal executions still have active local tasks"
                    )
                return await self._sum_usage(plan)

            if self._active_by_node:
                await self._wait_for_task_or_control_signal()
                continue

            executions = await self._execution_map(plan)
            await self._propagate_skips_to_fixpoint(plan, executions)
            executions = await self._execution_map(plan)
            if all_terminal(executions):
                return await self._sum_usage(plan)
            if self._has_launchable_ready_node(plan, executions):
                continue
            if any(
                execution.status is TaskStatus.CLAIMED
                and execution.node_id in self._raced_claimed_nodes
                for execution in executions.values()
            ):
                if self._logger is not None:
                    self._logger.debug(
                        "task graph waiting for externally claimed nodes parent_run_id=%s",
                        self._parent_run_id,
                    )
                await self._wait_for_task_or_control_signal()
                continue
            raise TaskGraphInvariantError("task graph cannot make progress")

    async def _execution_map(self, plan: TaskPlan) -> "dict[str, TaskExecution]":
        return {
            execution.node_id: execution
            for execution in await self._store.list_executions(plan.id)
        }

    async def _renew_due_nodes(self, now_monotonic: float) -> None:
        due = tuple(
            active
            for active in self._active_by_node.values()
            if not active.task.done() and now_monotonic >= active.next_renew_at
        )
        for active in due:
            execution = await self._store.get_execution(active.execution_id)
            if execution is None:
                raise TaskGraphInvariantError("task execution disappeared")
            if execution.status is not TaskStatus.CLAIMED:
                if execution.terminal:
                    continue
                raise NodeLeaseLostError(
                    f"node {active.node_id!r} is no longer CLAIMED"
                )
            if execution.owner != active.owner or execution.fence != active.fence:
                raise NodeLeaseLostError(f"node {active.node_id!r} lease was replaced")
            try:
                renewed = await self._store.renew(
                    active.execution_id,
                    owner=active.owner,
                    fence=active.fence,
                    duration=self._node_lease_duration,
                )
            except StorageConflictError as exc:
                raise NodeLeaseLostError(
                    f"node {active.node_id!r} lease renewal failed"
                ) from exc
            active.fence = renewed.fence
            active.next_renew_at = (
                now_monotonic + self._node_lease_duration.total_seconds() / 3
            )
            if self._logger is not None:
                self._logger.debug(
                    "task graph renewed node=%s fence=%s",
                    active.node_id,
                    active.fence,
                )

    async def _launch_ready_nodes(
        self,
        plan: TaskPlan,
        executions: "dict[str, TaskExecution]",
    ) -> None:
        if self._stopping or self._gate.cancel_requested:
            return
        slots = self._limits.max_concurrency - len(self._active_by_node)
        if slots <= 0:
            return
        for node in plan.nodes:
            if slots <= 0:
                break
            execution = executions.get(node.id)
            if execution is None or execution.status is not TaskStatus.READY:
                continue
            readiness = classify_readiness(node, executions)
            if not readiness.ready or readiness.skip:
                continue
            bound = await self._spawn_node(node, executions)
            if bound is None:
                return
            executions[node.id] = bound
            slots -= 1

    def _has_launchable_ready_node(
        self,
        plan: TaskPlan,
        executions: "dict[str, TaskExecution]",
    ) -> bool:
        for node in plan.nodes:
            execution = executions.get(node.id)
            if execution is None or execution.status is not TaskStatus.READY:
                continue
            readiness = classify_readiness(node, executions)
            if readiness.ready and not readiness.skip:
                return True
        return False

    async def _spawn_node(
        self,
        node: TaskNode,
        executions: "dict[str, TaskExecution]",
    ) -> "TaskExecution | None":
        execution = executions[node.id]
        await self._gate.check_before_launch()
        if node.id in self._active_by_node or execution.active_run_id is not None:
            raise TaskGraphInvariantError("ready node already has an active child")
        try:
            claimed = await self._store.claim_ready(
                execution.id,
                owner=self._owner,
                duration=self._node_lease_duration,
            )
        except StorageConflictError as exc:
            current = await self._store.get_execution(execution.id)
            if current is None:
                raise TaskGraphInvariantError(
                    f"node {node.id!r} disappeared after claim race"
                ) from exc
            if current.status is TaskStatus.READY:
                if self._logger is not None:
                    self._logger.debug(
                        "task graph claim race node=%s remains READY; rereading",
                        node.id,
                    )
                return None
            if current.status in {
                TaskStatus.CLAIMED,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.SKIPPED,
                TaskStatus.CANCELLED,
            }:
                if self._logger is not None:
                    self._logger.debug(
                        "task graph claim race node=%s observed status=%s",
                        node.id,
                        current.status.value,
                    )
                if current.status is TaskStatus.CLAIMED:
                    self._raced_claimed_nodes.add(node.id)
                return current
            raise TaskGraphInvariantError(
                f"node {node.id!r} has invalid status after claim race: "
                f"{current.status.value}"
            ) from exc
        child_id = child_run_id(self._parent_run_id, node.id)
        task: "asyncio.Task[None] | None" = None
        active: ActiveNode | None = None
        registered = False
        try:
            bound = await self._store.bind_child_run(
                claimed.id,
                owner=claimed.owner or self._owner,
                fence=claimed.fence,
                child_run_id=child_id,
            )
            request = NodeRunRequest(
                node=node,
                execution=bound,
                owner=bound.owner or self._owner,
                fence=bound.fence,
                child_run_id=child_id,
                dependencies=tuple(
                    executions[dependency.node_id]
                    for dependency in node.dependencies
                    if dependency.node_id in executions
                ),
                parent_guard=ParentLeaseGuard(
                    run_id=self._parent_run_id,
                    owner=self._parent_owner,
                    fence=self._parent_fence,
                ),
            )
            self._requests_by_node[node.id] = request
            task = asyncio.create_task(self._drive_node(node.id))
            active = ActiveNode(
                node_id=node.id,
                execution_id=bound.id,
                child_run_id=child_id,
                owner=request.owner,
                fence=request.fence,
                task=task,
                next_renew_at=(
                    time.monotonic()
                    + self._node_lease_duration.total_seconds() / 3
                ),
            )
            self._active_by_node[node.id] = active
            self._active_by_task[task] = active
            registered = True
            task.add_done_callback(self._on_node_done)
            if self._logger is not None:
                self._logger.info(
                    "task graph launched node=%s child_run_id=%s",
                    node.id,
                    child_id,
                )
            return bound
        except BaseException as exc:
            if registered:
                raise
            if task is not None:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if active is not None:
                self._remove_active(active)
            self._requests_by_node.pop(node.id, None)
            try:
                cancelled = await self._store.cancel_claimed(
                    claimed.id,
                    owner=claimed.owner or self._owner,
                    fence=claimed.fence,
                    reason=(
                        "parent_lease_lost_before_child_start"
                        if isinstance(exc, ParentLeaseGuardError)
                        else "node_start_failed"
                    ),
                    snapshot_revision=claimed.usage_revision,
                    usage=claimed.usage,
                )
                await self._publish_node_terminal_once(
                    node.id,
                    NodeRunResult(
                        status=TaskStatus.CANCELLED,
                        reason=(
                            "parent_lease_lost_before_child_start"
                            if isinstance(exc, ParentLeaseGuardError)
                            else "node_start_failed"
                        ),
                        usage=cancelled.usage,
                        snapshot_revision=cancelled.usage_revision,
                    ),
                )
            except BaseException as cleanup_error:
                diagnostic = CleanupDiagnostic(
                    stage="claim_release",
                    node_id=node.id,
                    error_type=type(cleanup_error).__name__,
                    safe_message="node claim cleanup failed",
                )
                raise TaskGraphCleanupError(
                    primary_error=exc,
                    cleanup_error=cleanup_error,
                    diagnostics=(diagnostic,),
                ) from cleanup_error
            raise

    def _on_node_done(self, task: "asyncio.Task[None]") -> None:
        self._completion.set()

    async def _drive_node(self, node_id: str) -> None:
        active = self._active_by_node.get(node_id)
        request = self._requests_by_node.get(node_id)
        if active is None or request is None:
            raise TaskGraphInvariantError("active node registration disappeared")
        try:
            outcome = await self._runner.run(request)
        except ParentLeaseGuardError as exc:
            usage = self._usage_by_node.get(node_id, TaskUsage())
            try:
                cancelled = await self._store.cancel_claimed(
                    active.execution_id,
                    owner=active.owner,
                    fence=active.fence,
                    reason="parent_lease_lost_before_child_start",
                    snapshot_revision=0,
                    usage=usage,
                )
                await self._publish_node_terminal_once(
                    node_id,
                    NodeRunResult(
                        status=TaskStatus.CANCELLED,
                        reason="parent_lease_lost_before_child_start",
                        usage=cancelled.usage,
                        snapshot_revision=cancelled.usage_revision,
                    ),
                )
            except BaseException as cleanup_error:
                diagnostic = CleanupDiagnostic(
                    stage="guard_claim_release",
                    node_id=node_id,
                    error_type=type(cleanup_error).__name__,
                    safe_message="guard-failed claim cleanup failed",
                )
                raise TaskGraphCleanupError(
                    primary_error=exc,
                    cleanup_error=cleanup_error,
                    diagnostics=(diagnostic,),
                ) from cleanup_error
            raise
        except ChildExecutionPlatformError as exc:
            self._usage_by_node[node_id] = exc.usage
            self._usage_revision_by_node[node_id] = 0
            if self._stopping:
                return
            raise
        except BaseException:
            if self._stopping:
                return
            raise
        self._usage_by_node[node_id] = outcome.usage
        self._usage_revision_by_node[node_id] = outcome.snapshot_revision
        if self._stopping:
            return
        self._gate.record_usage(outcome.usage)
        if outcome.status is TaskStatus.COMPLETED:
            await self._store.complete(
                active.execution_id,
                owner=active.owner,
                fence=active.fence,
                result=outcome.result,
                snapshot_revision=outcome.snapshot_revision,
                usage=outcome.usage,
            )
        elif outcome.status is TaskStatus.FAILED:
            await self._store.fail(
                active.execution_id,
                owner=active.owner,
                fence=active.fence,
                error=outcome.error or RunError("node_failed", "node failed"),
                snapshot_revision=outcome.snapshot_revision,
                usage=outcome.usage,
            )
        elif outcome.status is TaskStatus.CANCELLED:
            await self._store.cancel_claimed(
                active.execution_id,
                owner=active.owner,
                fence=active.fence,
                reason=outcome.reason or "cancelled",
                snapshot_revision=outcome.snapshot_revision,
                usage=outcome.usage,
            )
        else:
            raise TaskGraphInvariantError("node runner returned unsupported status")
        await self._publish_node_terminal_once(node_id, outcome)

    async def _publish_node_terminal_once(
        self, node_id: str, outcome: NodeRunResult
    ) -> None:
        if node_id in self._terminal_event_nodes:
            return
        self._terminal_event_nodes.add(node_id)
        if self._on_node_terminal is None:
            return
        try:
            await self._on_node_terminal(node_id, outcome)
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "task graph terminal event degraded node=%s error_type=%s",
                    node_id,
                    type(exc).__name__,
                )

    async def _reap_completed_tasks(self) -> None:
        for active in tuple(self._active_by_node.values()):
            if not active.task.done():
                continue
            await active.task
            execution = await self._store.get_execution(active.execution_id)
            if execution is None:
                raise TaskGraphInvariantError("task execution disappeared")
            if not execution.terminal:
                raise TaskGraphInvariantError(
                    "node task completed without terminal task execution"
                )
            self._remove_active(active)

    def _remove_active(self, active: ActiveNode) -> None:
        if self._active_by_node.get(active.node_id) is active:
            del self._active_by_node[active.node_id]
        if self._active_by_task.get(active.task) is active:
            del self._active_by_task[active.task]
        self._requests_by_node.pop(active.node_id, None)
        self._usage_by_node.pop(active.node_id, None)
        self._usage_revision_by_node.pop(active.node_id, None)

    async def _propagate_skips_to_fixpoint(
        self,
        plan: TaskPlan,
        executions: "dict[str, TaskExecution]",
    ) -> None:
        changed = True
        while changed:
            changed = False
            for node in plan.nodes:
                execution = executions.get(node.id)
                if execution is None or execution.status is not TaskStatus.READY:
                    continue
                readiness = classify_readiness(node, executions)
                if not readiness.skip:
                    continue
                updated = await self._store.skip(
                    execution.id,
                    blocked_by=readiness.blocked_by,
                    reason="dependency_failed",
                )
                executions[node.id] = updated
                if self._on_skip is not None:
                    await self._on_skip(node.id, readiness.blocked_by)
                changed = True

    async def _wait_for_task_or_control_signal(self) -> None:
        self._completion.clear()
        waiter = asyncio.create_task(self._completion.wait())
        try:
            now = time.monotonic()
            delay = min(
                CONTROL_POLL_INTERVAL,
                max(0.0, self._gate.next_wake_delay(now_monotonic=now)),
                self._next_node_renew_delay(now),
            )
            await asyncio.wait(
                tuple(self._active_by_task) + (waiter,),
                timeout=delay,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    def _next_node_renew_delay(self, now_monotonic: float) -> float:
        if not self._active_by_node:
            return CONTROL_POLL_INTERVAL
        return max(
            0.0,
            min(
                active.next_renew_at - now_monotonic
                for active in self._active_by_node.values()
            ),
        )

    async def _shutdown_active_nodes(
        self,
        plan: TaskPlan,
        *,
        stop_reason: StopReason,
        primary_error: BaseException | None,
    ) -> None:
        self._stopping = True
        self._stop_reason = stop_reason
        active_nodes = tuple(self._active_by_node.values())
        if self._logger is not None:
            self._logger.info(
                "task graph stopping parent_run_id=%s reason=%s active=%s",
                self._parent_run_id,
                stop_reason,
                len(active_nodes),
            )
        cancel_errors: "list[BaseException]" = []
        cancel_failed_nodes: "set[str]" = set()
        reason = self._task_cancel_reason(stop_reason)
        parent_lease_lost = stop_reason is StopReason.PARENT_LEASE_LOST
        if not parent_lease_lost:
            for active in active_nodes:
                try:
                    await self._runner.request_cancel(
                        child_run_id=active.child_run_id,
                        principal=self._principal,
                        reason=reason,
                    )
                except BaseException as exc:
                    cancel_errors.append(exc)
                    cancel_failed_nodes.add(active.node_id)
        for active in active_nodes:
            if not active.task.done():
                active.task.cancel()
        await asyncio.gather(
            *(active.task for active in active_nodes),
            return_exceptions=True,
        )

        storage_errors: "list[BaseException]" = []
        usage_failed_nodes: "set[str]" = set()
        if not parent_lease_lost:
            for active in active_nodes:
                try:
                    snapshot = await self._runner.read_usage(
                        child_run_id=active.child_run_id
                    )
                    self._usage_by_node[active.node_id] = snapshot.usage
                    self._usage_revision_by_node[active.node_id] = (
                        snapshot.snapshot_revision
                    )
                    if not snapshot.terminal:
                        raise StorageError(
                            f"child {active.child_run_id!r} is not terminal"
                        )
                except BaseException as exc:
                    storage_errors.append(exc)
                    usage_failed_nodes.add(active.node_id)
        if not parent_lease_lost:
            try:
                executions = await self._store.list_executions(plan.id)
                for active in active_nodes:
                    execution = next(
                        (
                            item
                            for item in executions
                            if item.id == active.execution_id
                        ),
                        None,
                    )
                    if execution is None:
                        raise StorageError("task execution disappeared during cleanup")
                    if execution.status is TaskStatus.CLAIMED:
                        if (
                            execution.owner == active.owner
                            and execution.fence == active.fence
                            and active.node_id not in cancel_failed_nodes
                            and active.node_id not in usage_failed_nodes
                        ):
                            observed = self._usage_by_node.get(
                                active.node_id, execution.usage
                            )
                            merged = merge_newer_cumulative_usage(
                                previous=execution.usage,
                                newer=observed,
                            )
                            cancelled = await self._store.cancel_claimed(
                                execution.id,
                                owner=active.owner,
                                fence=active.fence,
                                reason=reason,
                                snapshot_revision=self._usage_revision_by_node.get(
                                    active.node_id, execution.usage_revision
                                ),
                                usage=merged,
                            )
                            await self._publish_node_terminal_once(
                                active.node_id,
                                NodeRunResult(
                                    status=TaskStatus.CANCELLED,
                                    reason=reason,
                                    usage=cancelled.usage,
                                    snapshot_revision=cancelled.usage_revision,
                                ),
                            )
                    elif not execution.terminal:
                        raise TaskGraphInvariantError(
                            "active node did not converge to a terminal execution"
                        )
                for execution in executions:
                    if execution.status is TaskStatus.READY:
                        await self._store.cancel_ready(
                            execution.id,
                            reason=reason,
                        )
            except BaseException as exc:
                storage_errors.append(exc)

        for active in active_nodes:
            self._remove_active(active)
        if not parent_lease_lost:
            try:
                final_executions = await self._store.list_executions(plan.id)
                if any(
                    execution.status is TaskStatus.CLAIMED
                    for execution in final_executions
                ):
                    raise StorageError("a worker still holds CLAIMED nodes")
            except BaseException as exc:
                storage_errors.append(exc)
        if self._active_by_node or self._active_by_task:
            raise TaskGraphInvariantError("active node indexes were not emptied")
        if storage_errors:
            raise storage_errors[0]
        if cancel_errors:
            raise cancel_errors[0]

    @staticmethod
    def _classify_stop_reason(exc: BaseException) -> StopReason:
        if isinstance(exc, TaskGraphCleanupError) and exc.primary_error is not None:
            return TaskGraphEngine._classify_stop_reason(exc.primary_error)
        kind = getattr(exc, "kind", None)
        if (
            kind == "parent_lease_lost"
            or isinstance(exc, (ParentLeaseLostError, ParentLeaseGuardError))
        ):
            return StopReason.PARENT_LEASE_LOST
        if kind == "parent_cancelled":
            return StopReason.USER_CANCELLED
        if kind in {"timeout"}:
            return StopReason.TIMEOUT
        if kind in {"max_total_tokens", "token_limit_reached"}:
            return StopReason.TOKEN_LIMIT
        if kind in {"max_total_cost", "cost_limit_reached", "cost_usage_unavailable"}:
            return StopReason.COST_LIMIT
        if isinstance(exc, NodeLeaseLostError) or isinstance(exc, ChildExecutionPlatformError):
            return StopReason.PLATFORM_FAILURE
        return StopReason.PLATFORM_FAILURE

    def _task_cancel_reason(self, stop_reason: StopReason) -> str:
        if stop_reason is StopReason.PARENT_LEASE_LOST:
            return "parent_lease_lost"
        if self._gate.cancel_requested:
            return "parent_cancelled"
        if stop_reason in {StopReason.USER_CANCELLED, StopReason.CALLER_CANCELLED}:
            return "parent_cancelled"
        if stop_reason is StopReason.TIMEOUT:
            return "parent_timeout"
        if stop_reason in {StopReason.TOKEN_LIMIT, StopReason.COST_LIMIT}:
            return "parent_limit"
        return "parent_failed"

    async def _sum_usage(self, plan: TaskPlan) -> TaskUsage:
        total = UsageAccumulator()
        for execution in await self._store.list_executions(plan.id):
            if execution.status in {TaskStatus.READY, TaskStatus.SKIPPED}:
                continue
            if execution.status is TaskStatus.CANCELLED and execution.attempt == 0:
                continue
            total.add(execution.usage)
        return total.freeze()


__all__ = [
    "ActiveNode",
    "ControlGate",
    "CONTROL_POLL_INTERVAL",
    "DEFAULT_LEASE_DURATION",
    "NodeRunRequest",
    "NodeRunResult",
    "NodeUsageSnapshot",
    "NodeRunner",
    "StopReason",
    "TaskGraphEngine",
]
