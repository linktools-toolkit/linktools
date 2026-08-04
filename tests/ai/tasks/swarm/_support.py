#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared fakes for TaskGraphEngine tests: a recording NodeRunner (tracks
in-flight high-water mark + run order), a no-op ControlGate, and plan/
execution builders."""

import asyncio
from typing import Mapping

from linktools.ai.execution.domain import RunError
from linktools.ai.tasks.models import (
    TaskDependency,
    TaskExecution,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
)
from linktools.ai.execution.identifiers import task_execution_id
from linktools.ai.tasks.swarm.engine import (
    ControlGate,
    NodeRunRequest,
    NodeRunResult,
    NodeRunner,
)


class RecordingRunner:
    """NodeRunner that resolves immediately from a result map, recording the
    concurrency high-water mark and the order nodes started."""

    def __init__(
        self,
        results: "Mapping[str, object] | None" = None,
        *,
        failures: "Mapping[str, RunError] | None" = None,
        usage: "Mapping[str, TaskUsage] | None" = None,
        fast: "set[str] | None" = None,
    ) -> None:
        self._results = results or {}
        self._failures = failures or {}
        self._usage = usage or {}
        self._fast = fast or set()
        self.in_flight = 0
        self.max_seen = 0
        self.order: "list[tuple[str, float]]" = []
        self.counts: "dict[str, int]" = {}

    async def run(self, request: NodeRunRequest) -> NodeRunResult:
        node_id = request.node.id
        self.in_flight += 1
        self.max_seen = max(self.max_seen, self.in_flight)
        self.order.append((node_id, asyncio.get_event_loop().time()))
        self.counts[node_id] = self.counts.get(node_id, 0) + 1
        try:
            if node_id not in self._fast:
                await asyncio.sleep(0)
            usage = self._usage.get(node_id, TaskUsage())
            if node_id in self._failures:
                return NodeRunResult(
                    status=TaskStatus.FAILED,
                    error=self._failures[node_id],
                    usage=usage,
                )
            result = self._results.get(node_id)
            return NodeRunResult(
                status=TaskStatus.COMPLETED,
                result=result,
                usage=usage,
            )
        finally:
            self.in_flight -= 1

    async def request_cancel(
        self, *, child_run_id: str, principal=None, reason: str
    ) -> None:
        return None

    async def read_usage(self, *, child_run_id: str) -> TaskUsage:
        return TaskUsage()


class NoopGate:
    """ControlGate that never aborts and never reports cancellation."""

    cancel_requested = False

    async def check(self) -> None:
        return None

    async def check_before_launch(self) -> None:
        return None

    def next_wake_delay(self, *, now_monotonic: float) -> float:
        return 1.0

    def record_usage(self, usage: TaskUsage) -> None:
        return None


class LimitGate:
    """ControlGate that enforces a total-token cap from recorded usage."""

    def __init__(self, *, max_total_tokens: int) -> None:
        self._max = max_total_tokens
        self._spent = 0
        self.cancel_requested = False

    def record_usage(self, usage: TaskUsage) -> None:
        self._spent += usage.input_tokens + usage.output_tokens

    async def check(self) -> None:
        from linktools.ai.errors import SwarmLimitExceededError

        if self._spent > self._max:
            raise SwarmLimitExceededError("token cap", kind="max_total_tokens")

    async def check_before_launch(self) -> None:
        if self._spent >= self._max:
            from linktools.ai.errors import SwarmLimitExceededError

            raise SwarmLimitExceededError("token cap", kind="token_limit_reached")

    def next_wake_delay(self, *, now_monotonic: float) -> float:
        return 1.0


def make_plan(
    node_ids: "tuple[str, ...]",
    edges: "Mapping[str, tuple[object, ...]] | None" = None,
) -> TaskPlan:
    """Build a plan where each node maps to its own agent (agent_id == node_id)
    unless edges rebind them. Edge values are node-id strings or TaskDependency."""
    nodes = []
    for nid in node_ids:
        deps: "list[TaskDependency]" = []
        if edges and nid in edges:
            for edge in edges[nid]:
                if isinstance(edge, TaskDependency):
                    deps.append(edge)
                else:
                    deps.append(TaskDependency(str(edge)))
        nodes.append(
            TaskNode(
                nid,
                TaskGraphNodePayload(agent_id=nid, prompt=f"prompt-{nid}"),
                dependencies=tuple(deps),
            )
        )
    return TaskPlan(f"plan-{abs(hash(node_ids)) % 100000}", tuple(nodes))


def ready_executions(plan: TaskPlan) -> "tuple[TaskExecution, ...]":
    return tuple(
        TaskExecution(
            id=task_execution_id(plan.id, node.id),
            plan_id=plan.id,
            node_id=node.id,
            status=TaskStatus.READY,
        )
        for node in plan.nodes
    )
