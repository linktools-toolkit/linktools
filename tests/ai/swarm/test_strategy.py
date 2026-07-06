#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for swarm.strategy: SwarmStrategy Protocol, build_strategy, the
two built-in strategies (CoordinatorDelegationStrategy, ParallelFanOutStrategy),
and the LLMCoordinator adapter. PROGRAMMATIC strategies -- workers are real
CompiledAgents driven by FunctionModel; the coordinator is a deterministic
injected async function. No real model calls."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.models import CompiledAgent
from linktools.ai.agent.runner import AgentRunner
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.errors import SwarmError, SwarmLimitExceededError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunInput, RunStatus, RunnableType
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore
from linktools.ai.swarm.aggregation import AggregationPolicy
from linktools.ai.swarm.limits import SwarmLimits
from linktools.ai.swarm.models import (
    AgentRef,
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskStatus,
    TaskInput,
    TokenUsage,
)
from linktools.ai.swarm.spec import (
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)
from linktools.ai.swarm.store import SwarmStore
from linktools.ai.swarm.strategy import SwarmExecutionContext


# --- in-memory SwarmStore (single-process, FIFO claim) ----------------------
# A real, fully-functional store (not a mock): persists state in dicts, executes
# the SwarmStore Protocol contract including the atomic FIFO claim_task that the
# FileSwarmStore backend (later phase) will mirror.

_NOW = datetime.now(timezone.utc)


class _MemorySwarmStore(SwarmStore):
    def __init__(self) -> None:
        self._runs: "dict[str, SwarmRun]" = {}
        self._tasks: "dict[str, SwarmTask]" = {}

    async def create_run(self, run: SwarmRun) -> SwarmRun:
        self._runs[run.id] = run
        return run

    async def get_run(self, swarm_run_id: str) -> "SwarmRun | None":
        return self._runs.get(swarm_run_id)

    async def update_run(
        self,
        swarm_run_id: str,
        *,
        expected_version: int,
        status: "SwarmStatus | None" = None,
        round: "int | None" = None,
        token_usage: "Any | None" = None,
        cost: "Any | None" = None,
        metadata: "dict | None" = None,
    ) -> SwarmRun:
        current = self._runs[swarm_run_id]
        updated = replace(
            current,
            status=status if status is not None else current.status,
            round=round if round is not None else current.round,
            token_usage=token_usage if token_usage is not None else current.token_usage,
            cost=cost if cost is not None else current.cost,
            metadata=metadata if metadata is not None else current.metadata,
            version=current.version + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._runs[swarm_run_id] = updated
        return updated

    async def create_task(self, task: SwarmTask) -> SwarmTask:
        self._tasks[task.id] = task
        return task

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmTask | None":
        # FIFO: oldest PENDING task matching (swarm_run_id, agent_id).
        candidates = [
            t for t in self._tasks.values()
            if t.swarm_run_id == swarm_run_id
            and t.assigned_agent_id == agent_id
            and t.status is SwarmTaskStatus.PENDING
        ]
        candidates.sort(key=lambda t: t.created_at)
        if not candidates:
            return None
        target = candidates[0]
        now = datetime.now(timezone.utc)
        claimed = replace(
            target,
            status=SwarmTaskStatus.CLAIMED,
            claimed_at=now,
            attempts=target.attempts + 1,
            version=target.version + 1,
            updated_at=now,
        )
        self._tasks[target.id] = claimed
        return claimed

    async def set_active_run(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmTask:
        from linktools.ai.errors import SwarmConflictError, SwarmTaskNotFoundError
        if task_id not in self._tasks:
            raise SwarmTaskNotFoundError(f"swarm task not found: {task_id}")
        current = self._tasks[task_id]
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        updated = replace(
            current,
            active_run_id=run_id,
            version=current.version + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = updated
        return updated

    async def complete_task(self, task_id: str, result) -> SwarmTask:
        current = self._tasks[task_id]
        done = replace(
            current,
            status=SwarmTaskStatus.SUCCEEDED,
            result=result,
            version=current.version + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = done
        return done

    async def fail_task(self, task_id: str, error) -> SwarmTask:
        current = self._tasks[task_id]
        failed = replace(
            current,
            status=SwarmTaskStatus.FAILED,
            error=error,
            version=current.version + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = failed
        return failed

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmTaskStatus | None" = None
    ) -> "tuple[SwarmTask, ...]":
        result = [
            t for t in self._tasks.values()
            if t.swarm_run_id == swarm_run_id and (status is None or t.status is status)
        ]
        result.sort(key=lambda t: t.created_at)
        return tuple(result)

    async def reclaim_expired_tasks(self, swarm_run_id: str) -> "tuple[SwarmTask, ...]":
        return ()


# --- helpers ----------------------------------------------------------------

def _compile_worker(agent_id: str, output_text: str) -> CompiledAgent:
    """Compile an AgentSpec over a FunctionModel that always returns output_text
    as a plain str output (output_schema=str)."""
    def _model_fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=output_text)])
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    compiler = AgentCompiler(model_router=ModelRouter(registry=registry))
    spec = AgentSpec(
        id=agent_id, name=agent_id,
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions=f"you are {agent_id}"),
        output_schema=str,
    )
    return asyncio.run(compiler.compile(spec))


def _limits(**overrides) -> SwarmLimits:
    base = dict(
        max_rounds=10, max_tasks=50, max_delegations=20, max_depth=5,
        max_concurrency=4, max_total_tokens=None, max_total_cost=None, timeout_seconds=None,
    )
    base.update(overrides)
    return SwarmLimits(**base)


def _swarm_run() -> SwarmRun:
    return SwarmRun(
        id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.RUNNING,
        version=1, token_usage=TokenUsage(), cost=Decimal("0"),
        created_at=_NOW, updated_at=_NOW,
    )


def _parent_context() -> RunContext:
    return RunContext(
        run_id="drive-run-1", root_run_id="drive-run-1", parent_run_id=None,
        session_id="shared-session", runnable_id="swarm-spec-1",
        runnable_type=RunnableType.SWARM, user_id=None, tenant_id=None, workspace=None,
    )


def _build_ctx(
    tmp_path: Path,
    *,
    agents: "Mapping[str, CompiledAgent]",
    spec: SwarmSpec,
    swarm_store: "SwarmStore | None" = None,
) -> "SwarmExecutionContext":
    run_store = FileRunStore(root=tmp_path / "runs")
    session_store = FileSessionStore(root=tmp_path / "sessions")
    event_store = FileEventStore(root=tmp_path / "events")
    checkpoint_store = FileCheckpointStore(root=tmp_path / "checkpoints")
    runner = AgentRunner(
        run_store=run_store, session_store=session_store,
        event_store=event_store, checkpoint_store=checkpoint_store,
    )
    # pre-seed the shared session so the driving RunContext is consistent.
    asyncio.run(session_store.create(SessionRecord(
        id="shared-session", parent_id=None, status=SessionStatus.ACTIVE,
        version=1, created_at=_NOW, updated_at=_NOW,
    )))
    compiler = AgentCompiler(model_router=ModelRouter(registry=ModelRegistry()))
    return SwarmExecutionContext(
        spec=spec, swarm_run=_swarm_run(), request=RunInput(prompt="do the work"),
        parent_context=_parent_context(), agent_runner=runner, compiler=compiler,
        agents=agents, swarm_store=swarm_store or _MemorySwarmStore(),
        run_store=run_store, session_store=session_store, event_store=event_store,
    )


def _make_spec(*, kind: str, limits: SwarmLimits, agents: "tuple[AgentRef, ...]",
               coordinator: AgentRef) -> SwarmSpec:
    return SwarmSpec(
        id="swarm-spec-1", name="test-swarm", agents=agents, coordinator=coordinator,
        strategy=SwarmStrategySpec(kind=kind), limits=limits,
        context_policy=SwarmContextPolicy(), aggregation=AggregationPolicy(),
    )


# --- 1. CoordinatorDelegationStrategy: 2 tasks round 1, empty round 2 --------

def test_coordinator_delegation_runs_two_workers_and_aggregates(tmp_path):
    compiled_a = _compile_worker("worker-a", "alpha-out")
    compiled_b = _compile_worker("worker-b", "beta-out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="coordinator_delegation", limits=_limits(max_rounds=10),
        agents=(AgentRef("coord"), AgentRef("worker-a"), AgentRef("worker-b")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a, "worker-b": compiled_b},
        spec=spec, swarm_store=swarm_store,
    )
    # deterministic coordinator: 2 TaskInputs round 1, empty thereafter.
    call_count = {"n": 0}

    async def coordinator_fn(swarm_run, completed, limits):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (TaskInput(prompt="task-a"), TaskInput(prompt="task-b"))
        return ()

    from linktools.ai.swarm.strategy import CoordinatorDelegationStrategy
    strategy = CoordinatorDelegationStrategy(coordinator_fn=coordinator_fn)

    async def _run():
        return await strategy.run(ctx)
    result = asyncio.run(_run())

    # output is the CONCAT of both workers' fixed strings (round-robin assignment).
    assert "alpha-out" in str(result.output)
    assert "beta-out" in str(result.output)
    assert result.metadata["task_count"] == 2

    # 2 child Runs exist in run_store, correctly parented.
    async def _verify():
        children = await ctx.run_store.list_children(ctx.swarm_run.run_id)
        tasks = await swarm_store.list_tasks(ctx.swarm_run.id)
        return children, tasks
    children, tasks = asyncio.run(_verify())
    assert len(children) == 2
    assert all(c.parent_run_id == ctx.swarm_run.run_id for c in children)
    assert all(c.root_run_id == ctx.parent_context.root_run_id for c in children)
    assert all(c.status is RunStatus.SUCCEEDED for c in children)
    # 2 SwarmTasks SUCCEEDED.
    assert len(tasks) == 2
    assert all(t.status is SwarmTaskStatus.SUCCEEDED for t in tasks)
    # coordinator was invoked exactly twice (round 1 produced work, round 2 empty -> stop).
    assert call_count["n"] == 2


# --- 2. CoordinatorDelegationStrategy: max_rounds exceeded -> raise ----------

def test_coordinator_delegation_raises_when_max_rounds_exceeded(tmp_path):
    compiled_a = _compile_worker("worker-a", "alpha-out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="coordinator_delegation", limits=_limits(max_rounds=1),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )

    # coordinator that ALWAYS returns a task -- tries to start round 2 after max_rounds=1.
    async def coordinator_fn(swarm_run, completed, limits):
        return (TaskInput(prompt="more work"),)

    from linktools.ai.swarm.strategy import CoordinatorDelegationStrategy
    strategy = CoordinatorDelegationStrategy(coordinator_fn=coordinator_fn)

    async def _run():
        await strategy.run(ctx)
    with pytest.raises(SwarmLimitExceededError) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.kind == "max_rounds"
    # SwarmLimitExceededError is a SwarmError.
    assert isinstance(exc_info.value, SwarmError)


# --- 3. ParallelFanOutStrategy: task_count=3, one worker -> 3 child Runs -----

def test_parallel_fan_out_runs_three_tasks_on_one_worker(tmp_path):
    compiled_a = _compile_worker("worker-a", "same-out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="parallel_fan_out", limits=_limits(max_concurrency=4),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )

    from linktools.ai.swarm.strategy import ParallelFanOutStrategy
    strategy = ParallelFanOutStrategy(task_count=3)

    async def _run():
        return await strategy.run(ctx)
    result = asyncio.run(_run())

    # 3 child Runs, all SUCCEEDED.
    async def _verify():
        children = await ctx.run_store.list_children(ctx.swarm_run.run_id)
        tasks = await swarm_store.list_tasks(ctx.swarm_run.id)
        return children, tasks
    children, tasks = asyncio.run(_verify())
    assert len(children) == 3
    assert all(c.status is RunStatus.SUCCEEDED for c in children)
    assert len(tasks) == 3
    assert all(t.status is SwarmTaskStatus.SUCCEEDED for t in tasks)
    # output is the CONCAT of 3 (same string repeated -> joined by newlines).
    assert str(result.output) == "same-out\nsame-out\nsame-out"
    assert result.metadata["task_count"] == 3


# --- 4. ParallelFanOutStrategy: max_concurrency bounds in-flight runs --------

class _ConcurrencyTrackingRunner:
    """Wraps AgentRunner; tracks the high-water mark of simultaneously-in-flight
    run() calls. Injects a tiny await sleep so overlap is observable even though
    FunctionModel + FileStore are otherwise near-synchronous between awaits."""

    def __init__(self, inner: AgentRunner) -> None:
        self._inner = inner
        self.current = 0
        self.max = 0

    async def run(self, agent, request, context):
        self.current += 1
        self.max = max(self.max, self.current)
        try:
            await asyncio.sleep(0.01)  # force a yield so the semaphore parks coroutines
            return await self._inner.run(agent, request, context)
        finally:
            self.current -= 1


def test_parallel_fan_out_bounds_concurrency_via_semaphore(tmp_path):
    compiled_a = _compile_worker("worker-a", "out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="parallel_fan_out", limits=_limits(max_concurrency=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )
    tracker = _ConcurrencyTrackingRunner(ctx.agent_runner)
    ctx = replace(ctx, agent_runner=tracker)

    from linktools.ai.swarm.strategy import ParallelFanOutStrategy
    strategy = ParallelFanOutStrategy(task_count=4)

    async def _run():
        return await strategy.run(ctx)
    result = asyncio.run(_run())

    # all 4 tasks SUCCEEDED.
    assert result.metadata["task_count"] == 4
    # at most max_concurrency=2 worker runs were in-flight at once.
    assert tracker.max <= 2
    # and concurrency actually happened (>= 2) -- semaphore was the bottleneck.
    assert tracker.max == 2


# --- 4b. ParallelFanOutStrategy: max_tasks exceeded -> raise ------------------

def test_parallel_fan_out_raises_when_max_tasks_exceeded(tmp_path):
    compiled_a = _compile_worker("worker-a", "out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="parallel_fan_out", limits=_limits(max_tasks=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )

    from linktools.ai.swarm.strategy import ParallelFanOutStrategy
    # task_count=3 would exceed max_tasks=2 -- must raise before dispatching.
    strategy = ParallelFanOutStrategy(task_count=3)

    async def _run():
        await strategy.run(ctx)
    with pytest.raises(SwarmLimitExceededError) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.kind == "max_tasks"
    # SwarmLimitExceededError is a SwarmError.
    assert isinstance(exc_info.value, SwarmError)
    # nothing was dispatched: no tasks created before the limit fired.
    async def _verify():
        return await swarm_store.list_tasks(ctx.swarm_run.id)
    tasks = asyncio.run(_verify())
    assert tasks == ()


# --- 5. build_strategy registry ------------------------------------------------

def test_build_strategy_returns_parallel_fan_out():
    from linktools.ai.swarm.strategy import (
        ParallelFanOutStrategy, build_strategy,
    )
    strategy = build_strategy(SwarmStrategySpec(kind="parallel_fan_out"))
    assert isinstance(strategy, ParallelFanOutStrategy)


def test_build_strategy_returns_coordinator_delegation():
    from linktools.ai.swarm.strategy import (
        CoordinatorDelegationStrategy, build_strategy,
    )
    strategy = build_strategy(SwarmStrategySpec(kind="coordinator_delegation"))
    assert isinstance(strategy, CoordinatorDelegationStrategy)


def test_build_strategy_unknown_kind_raises_swarm_error():
    from linktools.ai.swarm.strategy import build_strategy
    with pytest.raises(SwarmError):
        build_strategy(SwarmStrategySpec(kind="no_such_strategy"))


# --- 6. SwarmLimits.max_depth enforcement in _run_task (GAP-09) -------------

def test_run_task_raises_when_depth_exceeds_max_depth(tmp_path):
    """_run_task walks the parent_task_id chain via list_tasks and raises
    SwarmLimitExceededError(kind="max_depth") when the chain depth exceeds the
    limit -- before claiming or dispatching the worker. A depth-2 child
    (parent_task_id -> depth-1 parent) under max_depth=1 fires the guard."""
    from linktools.ai.swarm.strategy import _run_task

    compiled_a = _compile_worker("worker-a", "out")
    swarm_store = _MemorySwarmStore()
    asyncio.run(swarm_store.create_run(SwarmRun(
        id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.RUNNING,
        version=1, token_usage=TokenUsage(), cost=Decimal("0"),
        created_at=_NOW, updated_at=_NOW,
    )))
    spec = _make_spec(
        kind="coordinator_delegation", limits=_limits(max_depth=1),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )
    # parent task (depth 1) already SUCCEEDED -> not claimable; it exists only
    # so the child's parent_task_id chain resolves to a real parent.
    parent = SwarmTask(
        id="parent-1", swarm_run_id="swarm-1", parent_task_id=None,
        assigned_agent_id="worker-a", description="parent",
        status=SwarmTaskStatus.SUCCEEDED,
        dependencies=(), input=TaskInput(prompt="p"), result=None, error=None,
        attempts=1, version=1, claimed_at=None, lease_expires_at=None,
        created_at=_NOW, updated_at=_NOW,
    )
    # child task (depth 2) PENDING -> would be claimed, but the depth guard
    # fires first.
    child = SwarmTask(
        id="child-1", swarm_run_id="swarm-1", parent_task_id="parent-1",
        assigned_agent_id="worker-a", description="child",
        status=SwarmTaskStatus.PENDING,
        dependencies=(), input=TaskInput(prompt="c"), result=None, error=None,
        attempts=0, version=1, claimed_at=None, lease_expires_at=None,
        created_at=_NOW, updated_at=_NOW,
    )
    asyncio.run(swarm_store.create_task(parent))
    asyncio.run(swarm_store.create_task(child))

    with pytest.raises(SwarmLimitExceededError) as exc_info:
        asyncio.run(_run_task(ctx, child))
    assert exc_info.value.kind == "max_depth"

    # nothing was claimed -- the child is still PENDING (guard fired first).
    tasks = asyncio.run(swarm_store.list_tasks("swarm-1"))
    child_after = next(t for t in tasks if t.id == "child-1")
    assert child_after.status is SwarmTaskStatus.PENDING


def test_run_task_depth_chain_walks_multiple_ancestors(tmp_path):
    """A 3-deep chain (child -> parent -> grandparent, all parent_task_id linked)
    under max_depth=2 raises max_depth; confirms the walk iterates beyond the
    immediate parent."""
    from linktools.ai.swarm.strategy import _run_task

    compiled_a = _compile_worker("worker-a", "out")
    swarm_store = _MemorySwarmStore()
    asyncio.run(swarm_store.create_run(SwarmRun(
        id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.RUNNING,
        version=1, token_usage=TokenUsage(), cost=Decimal("0"),
        created_at=_NOW, updated_at=_NOW,
    )))
    spec = _make_spec(
        kind="coordinator_delegation", limits=_limits(max_depth=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )
    grandparent = SwarmTask(
        id="gp-1", swarm_run_id="swarm-1", parent_task_id=None,
        assigned_agent_id="worker-a", description="gp",
        status=SwarmTaskStatus.SUCCEEDED,
        dependencies=(), input=TaskInput(prompt="g"), result=None, error=None,
        attempts=1, version=1, claimed_at=None, lease_expires_at=None,
        created_at=_NOW, updated_at=_NOW,
    )
    parent = SwarmTask(
        id="p-1", swarm_run_id="swarm-1", parent_task_id="gp-1",
        assigned_agent_id="worker-a", description="p",
        status=SwarmTaskStatus.SUCCEEDED,
        dependencies=(), input=TaskInput(prompt="p"), result=None, error=None,
        attempts=1, version=1, claimed_at=None, lease_expires_at=None,
        created_at=_NOW, updated_at=_NOW,
    )
    child = SwarmTask(
        id="c-1", swarm_run_id="swarm-1", parent_task_id="p-1",
        assigned_agent_id="worker-a", description="c",
        status=SwarmTaskStatus.PENDING,
        dependencies=(), input=TaskInput(prompt="c"), result=None, error=None,
        attempts=0, version=1, claimed_at=None, lease_expires_at=None,
        created_at=_NOW, updated_at=_NOW,
    )
    asyncio.run(swarm_store.create_task(grandparent))
    asyncio.run(swarm_store.create_task(parent))
    asyncio.run(swarm_store.create_task(child))

    with pytest.raises(SwarmLimitExceededError) as exc_info:
        asyncio.run(_run_task(ctx, child))
    assert exc_info.value.kind == "max_depth"


# --- 7. Phase-5A: task.id != child RunRecord.id (active_run_id decoupling) ---
#
# review doc §19.1: "禁止 SwarmTask.id == child_run_id". Each execution mints a
# fresh run_id and stores it on task.active_run_id via SwarmStore.set_active_run.


def test_task_id_is_different_from_child_run_id(tmp_path):
    """After a successful run, every SUCCEEDED task's active_run_id is set and
    is DIFFERENT from task.id, and matches an existing child RunRecord id."""
    compiled_a = _compile_worker("worker-a", "alpha-out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="parallel_fan_out", limits=_limits(max_concurrency=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )

    from linktools.ai.swarm.strategy import ParallelFanOutStrategy
    strategy = ParallelFanOutStrategy(task_count=2)

    async def _run():
        return await strategy.run(ctx)
    asyncio.run(_run())

    async def _verify():
        children = await ctx.run_store.list_children(ctx.swarm_run.run_id)
        tasks = await swarm_store.list_tasks(ctx.swarm_run.id)
        return children, tasks
    children, tasks = asyncio.run(_verify())

    child_ids = {c.id for c in children}
    for t in tasks:
        # the invariant: task.id IS NOT its child RunRecord.id.
        assert t.active_run_id is not None
        assert t.active_run_id != t.id
        # active_run_id points at a real child RunRecord.
        assert t.active_run_id in child_ids


def test_two_executions_of_same_task_produce_different_active_run_ids(tmp_path):
    """A retry of the same task mints a NEW child run_id and overwrites
    task.active_run_id -- so two sequential executions of the same task leave
    DIFFERENT active_run_ids behind (the second one wins on the task, but the
    first child RunRecord still exists in RunStore)."""
    from linktools.ai.swarm.strategy import _run_task

    compiled_a = _compile_worker("worker-a", "alpha-out")
    swarm_store = _MemorySwarmStore()
    asyncio.run(swarm_store.create_run(SwarmRun(
        id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.RUNNING,
        version=1, token_usage=TokenUsage(), cost=Decimal("0"),
        created_at=_NOW, updated_at=_NOW,
    )))
    spec = _make_spec(
        kind="coordinator_delegation", limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path, agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec, swarm_store=swarm_store,
    )
    # one PENDING task to execute twice.
    task = SwarmTask(
        id="task-1", swarm_run_id="swarm-1", parent_task_id=None,
        assigned_agent_id="worker-a", description="x",
        status=SwarmTaskStatus.PENDING,
        dependencies=(), input=TaskInput(prompt="do"), result=None, error=None,
        attempts=0, version=1, claimed_at=None, lease_expires_at=None,
        created_at=_NOW, updated_at=_NOW,
    )
    asyncio.run(swarm_store.create_task(task))

    # First execution: succeeds -> task SUCCEEDED with active_run_id set.
    first = asyncio.run(_run_task(ctx, task))
    assert first is not None
    after_first = asyncio.run(swarm_store.list_tasks("swarm-1"))[0]
    first_active = after_first.active_run_id
    assert first_active is not None
    assert first_active != "task-1"

    # Simulate a retry: reset the task to PENDING (a reclaim-style reset) and
    # execute again. The NEW execution mints a fresh run_id.
    reset = replace(
        after_first,
        status=SwarmTaskStatus.PENDING,
        result=None,
        active_run_id=None,
        version=after_first.version + 1,
    )
    swarm_store._tasks["task-1"] = reset
    second = asyncio.run(_run_task(ctx, reset))
    assert second is not None
    after_second = asyncio.run(swarm_store.list_tasks("swarm-1"))[0]
    second_active = after_second.active_run_id

    # the decoupling: two executions -> two DIFFERENT active_run_ids, and neither
    # equals task.id.
    assert second_active is not None
    assert second_active != first_active
    assert second_active != "task-1"

    # both child RunRecords exist in RunStore (different ids, both parented to
    # the driving swarm run).
    async def _children():
        return await ctx.run_store.list_children(ctx.swarm_run.run_id)
    children = asyncio.run(_children())
    assert {first_active, second_active}.issubset({c.id for c in children})


def test_set_active_run_rejects_stale_expected_version(swarm_store_via_module=None):
    """SwarmStore.set_active_run honors expected_version: a concurrent update
    that bumps the version between claim_task and set_active_run surfaces a
    SwarmConflictError rather than silently overwriting."""
    from linktools.ai.errors import SwarmConflictError
    from linktools.ai.swarm.models import SwarmRun

    store = _MemorySwarmStore()
    asyncio.run(store.create_run(SwarmRun(
        id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.RUNNING,
        version=1, token_usage=TokenUsage(), cost=Decimal("0"),
        created_at=_NOW, updated_at=_NOW,
    )))
    asyncio.run(store.create_task(SwarmTask(
        id="task-1", swarm_run_id="swarm-1", parent_task_id=None,
        assigned_agent_id="worker-a", description="x",
        status=SwarmTaskStatus.PENDING,
        dependencies=(), input=TaskInput(prompt="x"), result=None, error=None,
        attempts=0, version=1, claimed_at=None, lease_expires_at=None,
        created_at=_NOW, updated_at=_NOW,
    )))
    claimed = asyncio.run(store.claim_task("swarm-1", "worker-a"))
    assert claimed is not None
    assert claimed.version == 2

    # Simulate a concurrent update bumping version (e.g., a reclaim or another
    # set_active_run) between claim_task and our set_active_run call.
    bumped = asyncio.run(store.set_active_run(
        "task-1", "child-run-concurrent", expected_version=claimed.version
    ))
    assert bumped.version == 3

    # Now the original caller tries with the STALE version -- conflict.
    with pytest.raises(SwarmConflictError):
        asyncio.run(store.set_active_run(
            "task-1", "child-run-stale", expected_version=claimed.version
        ))


def test_set_active_run_missing_task_raises_not_found():
    """set_active_run on a missing task id surfaces SwarmTaskNotFoundError
    (mirrors complete_task / fail_task behavior)."""
    from linktools.ai.errors import SwarmTaskNotFoundError

    store = _MemorySwarmStore()
    asyncio.run(store.create_run(SwarmRun(
        id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.RUNNING,
        version=1, token_usage=TokenUsage(), cost=Decimal("0"),
        created_at=_NOW, updated_at=_NOW,
    )))
    with pytest.raises(SwarmTaskNotFoundError):
        asyncio.run(store.set_active_run("nope", "child-1", expected_version=1))
