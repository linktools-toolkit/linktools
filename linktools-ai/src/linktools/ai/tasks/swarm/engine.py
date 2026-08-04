#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deterministic task_graph scheduling and node-state convergence."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Awaitable, Callable, Protocol

from ...errors import (
    ChildExecutionPlatformError,
    NodeLeaseLostError,
    ParentLeaseLostError,
    StorageConflictError,
    StorageError,
    SwarmLimitExceededError,
    TaskGraphInvariantError,
)
from ...execution.domain import RunError
from ...execution.identifiers import child_run_id
from ..models import (
    TaskExecution,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
    UsageAccumulator,
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


@dataclass(frozen=True, slots=True)
class NodeRunResult:
    """A child business outcome mapped to one task terminal transition."""

    status: TaskStatus
    result: "object | None" = None
    error: "RunError | None" = None
    reason: "str | None" = None
    usage: TaskUsage = field(default_factory=TaskUsage)


class NodeRunner(Protocol):
    """Drive children and request their cancellation; never write TaskStore."""

    async def run(self, request: NodeRunRequest) -> NodeRunResult: ...

    async def request_cancel(
        self, *, child_run_id: str, principal: "PrincipalContext | None", reason: str
    ) -> None: ...

    async def read_usage(self, *, child_run_id: str) -> TaskUsage: ...


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
        self._principal = principal
        self._on_skip = on_skip
        self._on_node_terminal = on_node_terminal
        self._node_lease_duration = DEFAULT_LEASE_DURATION
        self._active_by_node: "dict[str, ActiveNode]" = {}
        self._active_by_task: "dict[asyncio.Task[None], ActiveNode]" = {}
        self._requests_by_node: "dict[str, NodeRunRequest]" = {}
        self._usage_by_node: "dict[str, TaskUsage]" = {}
        self._stopping = False
        self._completion = asyncio.Event()
        self._logger = logger

    @property
    def active_count(self) -> int:
        return len(self._active_by_node)

    async def execute(self, plan: TaskPlan) -> TaskUsage:
        """Run the graph and always perform the final convergence pass."""
        primary_error: BaseException | None = None
        stop_reason = "normal"
        try:
            return await self._run_loop(plan)
        except asyncio.CancelledError as exc:
            primary_error = exc
            stop_reason = "caller_cancelled"
            raise
        except BaseException as exc:
            primary_error = exc
            stop_reason = self._classify_stop_reason(exc)
            raise
        finally:
            try:
                await self._shutdown_active_nodes(
                    plan,
                    stop_reason=stop_reason,
                    primary_error=primary_error,
                )
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(
                    "task graph cleanup failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
                if self._logger is not None:
                    self._logger.exception(
                        "task graph cleanup failed after primary error",
                        exc_info=cleanup_error,
                    )

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
    ) -> TaskExecution:
        execution = executions[node.id]
        if node.id in self._active_by_node or execution.active_run_id is not None:
            raise TaskGraphInvariantError("ready node already has an active child")
        claimed = await self._store.claim_ready(
            execution.id,
            owner=self._owner,
            duration=self._node_lease_duration,
        )
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
                await self._store.cancel_claimed(
                    claimed.id,
                    owner=claimed.owner or self._owner,
                    fence=claimed.fence,
                    reason="node_start_failed",
                    usage=claimed.usage,
                )
            except BaseException as cleanup_error:
                exc.add_note(
                    "failed to release node claim: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
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
        except ChildExecutionPlatformError as exc:
            self._usage_by_node[node_id] = exc.usage
            raise
        self._usage_by_node[node_id] = outcome.usage
        self._gate.record_usage(outcome.usage)
        if outcome.status is TaskStatus.COMPLETED:
            await self._store.complete(
                active.execution_id,
                owner=active.owner,
                fence=active.fence,
                result=outcome.result,
                usage=outcome.usage,
            )
        elif outcome.status is TaskStatus.FAILED:
            await self._store.fail(
                active.execution_id,
                owner=active.owner,
                fence=active.fence,
                error=outcome.error or RunError("node_failed", "node failed"),
                usage=outcome.usage,
            )
        elif outcome.status is TaskStatus.CANCELLED:
            await self._store.cancel_claimed(
                active.execution_id,
                owner=active.owner,
                fence=active.fence,
                reason=outcome.reason or "cancelled",
                usage=outcome.usage,
            )
        else:
            raise TaskGraphInvariantError("node runner returned unsupported status")
        if self._on_node_terminal is not None:
            await self._on_node_terminal(node_id, outcome)

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
        stop_reason: str,
        primary_error: BaseException | None,
    ) -> None:
        self._stopping = True
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

        parent_lease_lost = stop_reason == "parent_lease_lost"
        storage_errors: "list[BaseException]" = []
        usage_failed_nodes: "set[str]" = set()
        if not parent_lease_lost:
            for active in active_nodes:
                try:
                    self._usage_by_node[active.node_id] = await self._runner.read_usage(
                        child_run_id=active.child_run_id
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
                            await self._store.cancel_claimed(
                                execution.id,
                                owner=active.owner,
                                fence=active.fence,
                                reason=reason,
                                usage=self._usage_by_node.get(
                                    active.node_id,
                                    execution.usage,
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
                    and execution.owner == self._owner
                    for execution in final_executions
                ):
                    raise StorageError("current owner still holds CLAIMED nodes")
            except BaseException as exc:
                storage_errors.append(exc)
        if self._active_by_node or self._active_by_task:
            raise TaskGraphInvariantError("active node indexes were not emptied")
        if storage_errors:
            raise storage_errors[0]
        if cancel_errors:
            raise cancel_errors[0]

    @staticmethod
    def _classify_stop_reason(exc: BaseException) -> str:
        kind = getattr(exc, "kind", None)
        if kind == "parent_lease_lost" or isinstance(exc, ParentLeaseLostError):
            return "parent_lease_lost"
        if isinstance(exc, NodeLeaseLostError):
            return "node_lease_lost"
        if kind in {"timeout"}:
            return "timeout"
        if kind in {
            "max_total_tokens",
            "token_limit_reached",
            "max_total_cost",
            "cost_limit_reached",
            "cost_usage_unavailable",
        }:
            return "parent_limit"
        return "parent_failed"

    def _task_cancel_reason(self, stop_reason: str) -> str:
        if self._gate.cancel_requested or stop_reason in {
            "caller_cancelled",
            "parent_cancelled",
        }:
            return "parent_cancelled"
        if stop_reason == "timeout":
            return "parent_timeout"
        if stop_reason == "parent_limit":
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
    "NodeRunner",
    "TaskGraphEngine",
]
