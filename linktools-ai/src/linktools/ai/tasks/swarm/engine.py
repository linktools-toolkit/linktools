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
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Awaitable, Callable, Protocol

from ...errors import TaskGraphInvariantError
from ...execution.domain import RunError
from ..models import (
    TaskExecution,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
)
from ..store import TaskStore
from .validation import Readiness, all_terminal, classify_readiness

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .limits import SwarmLimits


DEFAULT_LEASE_DURATION = timedelta(minutes=10)


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


class ControlGate(Protocol):
    """Checks the parent run's liveness before each scheduling pass. ``check``
    is async so it can read the parent run's persisted cancel/lease state.
    ``check`` raises to abort (timeout, token/cost limit, parent-lease lost,
    cancel requested) or returns silently to continue. ``cancel_requested`` is
    True when the parent run was asked to cancel. ``record_usage`` is called
    after each node terminal so the gate can accumulate token/cost spend and
    enforce caps mid-run."""

    async def check(self, *, now: float) -> None: ...

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
        self._on_skip = on_skip
        self._on_node_terminal = on_node_terminal
        self._tasks: "dict[asyncio.Task[None], str]" = {}
        self._completion = asyncio.Event()
        self._logger = logger

    async def execute(self, plan: TaskPlan) -> TaskUsage:
        """Drive every node to a terminal status. Returns the summed real
        usage across all worker attempts (success, failure, cancellation)."""
        while True:
            await self._gate.check(now=time.monotonic())
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
            self._completion.clear()
            await self._launch_ready(plan, executions)
            if not self._tasks and not self._has_launchable(plan, executions):
                raise TaskGraphInvariantError("task graph cannot make progress")
            if self._tasks:
                await self._wait_for_completion_or_signal()
        return await self._sum_usage(plan)

    async def _launch_ready(
        self,
        plan: TaskPlan,
        executions: "dict[str, TaskExecution]",
    ) -> None:
        if self._gate.cancel_requested:
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
        claimed = await self._store.claim_ready(
            execution.id, owner=self._owner, duration=DEFAULT_LEASE_DURATION
        )
        child_id = _child_run_id(self._parent_run_id, node.id)
        bound = await self._store.bind_child_run(
            claimed.id,
            owner=self._owner,
            fence=claimed.fence,
            child_run_id=child_id,
        )
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

    def _on_node_done(self, task: "asyncio.Task[None]") -> None:
        self._tasks.pop(task, None)
        self._completion.set()

    async def _drive_node(self, request: NodeRunRequest) -> None:
        try:
            outcome = await self._runner.run(request)
        except asyncio.CancelledError:
            await self._cancel_claimed(request, TaskUsage())
            raise
        except Exception as exc:  # programming error -> FAIL the node
            outcome = NodeRunResult(
                status=TaskStatus.FAILED,
                error=RunError("node_runner_error", str(exc)),
                usage=TaskUsage(),
            )
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
        try:
            await self._store.cancel_claimed(
                request.execution.id,
                owner=request.owner,
                fence=request.fence,
                reason="cancelled",
                usage=usage,
            )
        except Exception:
            pass

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
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        self._tasks.clear()
        executions = await self._store.list_executions(plan.id)
        for execution in executions:
            if execution.status is TaskStatus.READY:
                await self._store.cancel_ready(
                    execution.id, reason="parent_cancelled"
                )

    async def _wait_for_completion_or_signal(self) -> None:
        await self._completion.wait()

    async def _sum_usage(self, plan: TaskPlan) -> TaskUsage:
        total = TaskUsage()
        for execution in await self._store.list_executions(plan.id):
            total = total.add(execution.usage)
        return total


def _child_run_id(parent_run_id: str, node_id: str) -> str:
    """Deterministic child run id; mirrors execution.domain.child_run_id so the
    engine does not import the execution layer."""
    import hashlib

    digest = hashlib.sha256(
        "task-graph-child-v1\0".encode()
        + parent_run_id.encode()
        + b"\0"
        + node_id.encode()
    ).hexdigest()
    return f"tg-child-{digest}"


__all__ = [
    "ControlGate",
    "DEFAULT_LEASE_DURATION",
    "NodeRunRequest",
    "NodeRunResult",
    "NodeRunner",
    "TaskGraphEngine",
]
