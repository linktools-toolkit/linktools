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
from linktools.ai.core.model_runtime import ModelRegistry
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
