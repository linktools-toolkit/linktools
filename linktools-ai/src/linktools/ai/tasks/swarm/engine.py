#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""TaskGraphEngine: deterministic DAG scheduler for the task_graph swarm.

The engine owns ONLY graph progress: it validates topology, finds ready
nodes, propagates skip/proceed_degraded, enforces max_concurrency, applies
timeout/token/cost limits, converges cancellation, and collects results. It
does not touch audit business objects, mutate ``task_plan``, or generate
nodes — those concerns live behind the NodeRunner and ControlGate protocols
the caller injects. ``RunError`` is the lone execution-domain type it uses
to label node-runner programming errors, accepted as a baselined import
cycle."""

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Awaitable, Callable, Protocol

from ...errors import StorageConflictError, SwarmLimitExceededError, TaskGraphInvariantError
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
from .validation import Readiness, all_terminal, classify_readiness

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...governance.identity import PrincipalContext
    from .limits import SwarmLimits


DEFAULT_LEASE_DURATION = timedelta(minutes=10)
CONTROL_POLL_INTERVAL = 1.0


@dataclass(frozen=True, slots=True)
class NodeRunRequest:
    """The pure inputs a NodeRunner needs to drive one node's child agent."""

    node: TaskNode
    execution: TaskExecution
    owner: str
    fence: int
    child_run_id: str
    dependencies: "tuple[TaskExecution, ...]"


@dataclass(frozen=True, slots=True)
class NodeRunResult:
    """The terminal outcome a NodeRunner reports for one node. Exactly one of
    result/error holds; usage always carries the real tokens/cost the child
    consumed (even on failure or cancellation)."""

    status: TaskStatus
    result: "object | None" = None
    error: "RunError | None" = None
    usage: TaskUsage = field(default_factory=TaskUsage)


class NodeRunner(Protocol):
    """Drives a single node's child agent to a terminal state and returns the
    outcome. Implemented by the execution layer; the engine never imports it."""

    async def run(self, request: NodeRunRequest) -> NodeRunResult: ...

    async def request_cancel(
        self, *, child_run_id: str, principal: "PrincipalContext", reason: str
    ) -> None: ...


class ControlGate(Protocol):
    """Checks the parent run's liveness before each scheduling pass. ``check``
    is async so it can read the parent run's persisted cancel/lease state.
    ``check`` raises to abort (timeout, token/cost limit, parent-lease lost,
    cancel requested) or returns silently to continue. ``cancel_requested`` is
    True when the parent run was asked to cancel. ``record_usage`` is called
    after each node terminal so the gate can accumulate token/cost spend and
    enforce caps mid-run."""

    async def check(self, *, now_monotonic: float) -> None: ...

    def next_wake_delay(self, *, now_monotonic: float) -> float: ...

    @property
    def cancel_requested(self) -> bool: ...

    def record_usage(self, usage: TaskUsage) -> None: ...


class TaskGraphEngine:
    """Schedules a TaskPlan against a TaskStore using an injected NodeRunner.

    Constructed once per swarm run; ``execute`` drives the loop to completion
    (all nodes terminal) or a run-level error, then returns the aggregate
    usage and a per-node projection for collect()."""

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
        self._tasks: "dict[asyncio.Task[None], str]" = {}
        self._stopping = False
        self._completion = asyncio.Event()
        self._logger = logger

    async def execute(self, plan: TaskPlan) -> TaskUsage:
        """Drive every node to a terminal status. Returns the summed real
        usage across all worker attempts (success, failure, cancellation)."""
        stop_reason: BaseException | None = None
        try:
            while True:
                now = time.monotonic()
                await self._check_gate(now)
                await self._renew_active_nodes(plan)
                if self._gate.cancel_requested:
                    await self._converge_cancellation(plan)
                executions = {
                    e.node_id: e for e in await self._store.list_executions(plan.id)
                }
                await self._propagate_skips(plan, executions)
                executions = {
                    e.node_id: e for e in await self._store.list_executions(plan.id)
                }
                if all_terminal(executions):
                    break
                check_before_launch = getattr(
                    self._gate, "check_before_launch", None
                )
                if check_before_launch is not None and self._has_launchable(
                    plan, executions
                ):
                    await check_before_launch()
                self._completion.clear()
                await self._launch_ready(plan, executions)
                await self._reap_completed()
                if not self._tasks and not self._has_launchable(plan, executions):
                    raise TaskGraphInvariantError("task graph cannot make progress")
                if self._tasks:
                    await self._wait_for_completion_or_signal()
            return await self._sum_usage(plan)
        except BaseException as exc:
            stop_reason = exc
            raise
        finally:
            if self._tasks:
                await self._shutdown_active_nodes(plan, stop_reason)

    async def _check_gate(self, now: float) -> None:
        parameters = inspect.signature(self._gate.check).parameters
        keyword = "now_monotonic" if "now_monotonic" in parameters else "now"
        await self._gate.check(**{keyword: now})

    async def _renew_active_nodes(self, plan: TaskPlan) -> None:
        if not self._tasks:
            return
        by_node = {
            execution.node_id: execution
            for execution in await self._store.list_executions(plan.id)
        }
        for task, node_id in tuple(self._tasks.items()):
            if task.done():
                continue
            execution = by_node.get(node_id)
            if execution is None or execution.status is not TaskStatus.CLAIMED:
                continue
            try:
                await self._store.renew(
                    execution.id,
                    owner=self._owner,
                    fence=execution.fence,
                    duration=DEFAULT_LEASE_DURATION,
                )
            except StorageConflictError as exc:
                raise SwarmLimitExceededError(
                    "node lease lost", kind="node_lease_lost"
                ) from exc

    async def _launch_ready(
        self,
        plan: TaskPlan,
        executions: "dict[str, TaskExecution]",
    ) -> None:
        if self._stopping or self._gate.cancel_requested:
            return
        running = len(self._tasks)
        slots = max(0, self._limits.max_concurrency - running)
        if slots == 0:
            return
        launched = 0
        for node in plan.nodes:
            if launched >= slots:
                break
            execution = executions.get(node.id)
            if execution is None or execution.status is not TaskStatus.READY:
                continue
            readiness = classify_readiness(node, executions)
            if readiness.skip:
                continue
            if not readiness.ready:
                continue
            await self._spawn(node, executions)
            launched += 1
            running += 1
            executions[node.id] = await self._store.get_execution(execution.id)

    def _has_launchable(
        self,
        plan: TaskPlan,
        executions: "dict[str, TaskExecution]",
    ) -> bool:
        for node in plan.nodes:
            execution = executions.get(node.id)
            if execution is None:
                continue
            if execution.status is not TaskStatus.READY:
                continue
            readiness = classify_readiness(node, executions)
            if readiness.ready:
                return True
        return False

    async def _spawn(
        self,
        node: TaskNode,
        executions: "dict[str, TaskExecution]",
    ) -> None:
        execution = executions[node.id]
        if execution.active_run_id is not None:
            raise TaskGraphInvariantError("ready node already has a child run")
        claimed = await self._store.claim_ready(
            execution.id, owner=self._owner, duration=DEFAULT_LEASE_DURATION
        )
        child_id = child_run_id(self._parent_run_id, node.id)
        try:
            bound = await self._store.bind_child_run(
                claimed.id,
                owner=self._owner,
                fence=claimed.fence,
                child_run_id=child_id,
            )
        except BaseException:
            await self._store.cancel_claimed(
                claimed.id,
                owner=self._owner,
                fence=claimed.fence,
                reason="bind_failed",
                usage=TaskUsage(),
            )
            raise
        deps = tuple(
            executions[dep.node_id]
            for dep in node.dependencies
            if dep.node_id in executions
        )
        request = NodeRunRequest(
            node=node,
            execution=bound,
            owner=self._owner,
            fence=bound.fence,
            child_run_id=child_id,
            dependencies=deps,
        )
        task = asyncio.create_task(self._drive_node(request))
        self._tasks[task] = node.id
        task.add_done_callback(self._on_node_done)
        if self._logger is not None:
            self._logger.info(
                "task graph launched node=%s child_run_id=%s",
                node.id,
                child_id,
            )

    def _on_node_done(self, task: "asyncio.Task[None]") -> None:
        self._completion.set()

    async def _reap_completed(self) -> None:
        for task in tuple(self._tasks):
            if not task.done():
                continue
            self._tasks.pop(task, None)
            await task

    async def _drive_node(self, request: NodeRunRequest) -> None:
        try:
            outcome = await self._runner.run(request)
        except asyncio.CancelledError:
            await self._cancel_claimed(request, request.execution.usage)
            raise
        self._gate.record_usage(outcome.usage)
        await self._apply_outcome(request, outcome, request.node.id)

    async def _apply_outcome(
        self, request: NodeRunRequest, outcome: NodeRunResult, node_id: str
    ) -> None:
        if outcome.status is TaskStatus.COMPLETED:
            await self._store.complete(
                request.execution.id,
                owner=request.owner,
                fence=request.fence,
                result=outcome.result,
                usage=outcome.usage,
            )
        elif outcome.status is TaskStatus.FAILED:
            await self._store.fail(
                request.execution.id,
                owner=request.owner,
                fence=request.fence,
                error=outcome.error or RunError("node_failed", "node failed"),
                usage=outcome.usage,
            )
        elif outcome.status is TaskStatus.CANCELLED:
            await self._cancel_claimed(request, outcome.usage)
        else:
            await self._cancel_claimed(request, outcome.usage)
        # The node's authoritative TaskExecution is now terminal; only now may
        # a step-terminal event be published (persist before event).
        if self._on_node_terminal is not None:
            await self._on_node_terminal(node_id, outcome)

    async def _cancel_claimed(
        self, request: NodeRunRequest, usage: TaskUsage
    ) -> None:
        await self._store.cancel_claimed(
            request.execution.id,
            owner=request.owner,
            fence=request.fence,
            reason="cancelled",
            usage=usage,
        )

    async def _propagate_skips(
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
                if readiness.skip:
                    updated = await self._store.skip(
                        execution.id,
                        blocked_by=readiness.blocked_by,
                        reason="dependency_failed",
                    )
                    executions[node.id] = updated
                    if self._on_skip is not None:
                        await self._on_skip(node.id, readiness.blocked_by)
                    changed = True

    async def _converge_cancellation(self, plan: TaskPlan) -> None:
        self._stopping = True
        cleanup_error: BaseException | None = None
        for task, node_id in tuple(self._tasks.items()):
            if not task.done():
                request = await self._store.get_execution(
                    next((e.id for e in await self._store.list_executions(plan.id) if e.node_id == node_id), "")
                )
                if request is not None and request.active_run_id:
                    try:
                        await self._request_child_cancel(request.active_run_id)
                    except BaseException as exc:
                        cleanup_error = cleanup_error or exc
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._tasks.clear()
        if cleanup_error is not None:
            raise cleanup_error
        executions = await self._store.list_executions(plan.id)
        for execution in executions:
            if execution.status is TaskStatus.READY:
                await self._store.cancel_ready(
                    execution.id, reason="parent_cancelled"
                )
            elif (
                execution.status is TaskStatus.CLAIMED
                and execution.owner == self._owner
            ):
                await self._store.cancel_claimed(
                    execution.id,
                    owner=self._owner,
                    fence=execution.fence,
                    reason="parent_cancelled",
                    usage=execution.usage,
                )

    async def _request_child_cancel(self, child_run_id: str) -> None:
        await self._runner.request_cancel(
            child_run_id=child_run_id,
            principal=self._principal,
            reason="parent_cancelled",
        )

    async def _shutdown_active_nodes(
        self, plan: TaskPlan, stop_reason: BaseException | None
    ) -> None:
        self._stopping = True
        if self._logger is not None:
            self._logger.info(
                "task graph stopping parent_run_id=%s reason=%s",
                self._parent_run_id,
                getattr(
                    stop_reason,
                    "kind",
                    type(stop_reason).__name__ if stop_reason else "complete",
                ),
            )
        reason = "parent_failed"
        if self._gate.cancel_requested:
            reason = "parent_cancelled"
        elif getattr(stop_reason, "kind", None) == "timeout":
            reason = "parent_timeout"
        elif getattr(stop_reason, "kind", None) in {
            "max_total_tokens",
            "token_limit_reached",
            "max_total_cost",
            "cost_limit_reached",
            "cost_usage_unavailable",
        }:
            reason = "parent_limit"
        cleanup_error: BaseException | None = None
        for task, node_id in tuple(self._tasks.items()):
            execution = next(
                (e for e in await self._store.list_executions(plan.id) if e.node_id == node_id),
                None,
            )
            if execution is not None and execution.active_run_id:
                try:
                    await self._request_child_cancel(execution.active_run_id)
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
            task.cancel()
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        self._tasks.clear()
        if cleanup_error is not None:
            raise cleanup_error
        if getattr(stop_reason, "kind", None) == "parent_lease_lost":
            return
        for execution in await self._store.list_executions(plan.id):
            if execution.status is TaskStatus.READY:
                await self._store.cancel_ready(execution.id, reason=reason)
            elif execution.status is TaskStatus.CLAIMED and execution.owner == self._owner:
                try:
                    await self._store.cancel_claimed(
                        execution.id,
                        owner=self._owner,
                        fence=execution.fence,
                        reason=reason,
                        usage=execution.usage,
                    )
                except StorageConflictError:
                    if getattr(stop_reason, "kind", None) != "node_lease_lost":
                        raise
    async def _wait_for_completion_or_signal(self) -> None:
        waiter = asyncio.create_task(self._completion.wait())
        try:
            now = time.monotonic()
            next_wake = getattr(self._gate, "next_wake_delay", None)
            delay = (
                CONTROL_POLL_INTERVAL
                if next_wake is None
                else next_wake(now_monotonic=now)
            )
            if delay is None:
                delay = CONTROL_POLL_INTERVAL
            await asyncio.wait(
                tuple(self._tasks) + (waiter,),
                timeout=max(0.0, min(float(delay), CONTROL_POLL_INTERVAL)),
                return_when=asyncio.FIRST_COMPLETED,
            )
            await self._reap_completed()
        finally:
            if not waiter.done():
                waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)

    async def _sum_usage(self, plan: TaskPlan) -> TaskUsage:
        total = UsageAccumulator()
        for execution in await self._store.list_executions(plan.id):
            if execution.status is TaskStatus.READY or execution.status is TaskStatus.SKIPPED:
                continue
            if execution.status is TaskStatus.CANCELLED and execution.attempt == 0:
                continue
            total.add(execution.usage)
        return total.freeze()


__all__ = [
    "ControlGate",
    "CONTROL_POLL_INTERVAL",
    "DEFAULT_LEASE_DURATION",
    "NodeRunRequest",
    "NodeRunResult",
    "NodeRunner",
    "TaskGraphEngine",
]
