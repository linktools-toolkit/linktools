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
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.errors import SwarmConflictError, SwarmError, SwarmLimitExceededError
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.resolver import ModelResolver
from linktools.ai.run.context import RunContext
from linktools.ai.run.dispatch import RunDispatcher
from linktools.ai.run.models import RunInput, RunStatus, RunnableType
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.definition import FilesystemRunDefinitionStore
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.swarm.aggregation import AggregationPolicy
from linktools.ai.swarm.limits import SwarmLimits
from linktools.ai.swarm.models import (
    AgentRef,
    SwarmRun,
    SwarmStatus,
    SwarmStep,
    SwarmStepStatus,
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
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


# --- in-memory SwarmStore (single-process, FIFO claim) ----------------------
# A real, fully-functional store (not a mock): persists state in dicts, executes
# the SwarmStore Protocol contract including the atomic FIFO claim_task that the
# FilesystemSwarmStore backend () will mirror.

_NOW = datetime.now(timezone.utc)


class _MemorySwarmStore(SwarmStore):
    def __init__(self) -> None:
        self._runs: "dict[str, SwarmRun]" = {}
        self._tasks: "dict[str, SwarmStep]" = {}
        self._attempts: "dict[str, list]" = {}  # task_id -> [SwarmStepAttempt, ...]

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

    async def create_task(self, task: SwarmStep) -> SwarmStep:
        self._tasks[task.id] = task
        return task

    async def claim_task(
        self, swarm_run_id: str, agent_id: str, *, lease_seconds: "float | None" = None
    ) -> "SwarmStep | None":
        # FIFO: oldest PENDING task matching (swarm_run_id, agent_id).
        candidates = [
            t
            for t in self._tasks.values()
            if t.swarm_run_id == swarm_run_id
            and t.assigned_agent_id == agent_id
            and t.status is SwarmStepStatus.PENDING
        ]
        candidates.sort(key=lambda t: t.created_at)
        if not candidates:
            return None
        target = candidates[0]
        now = datetime.now(timezone.utc)
        claimed = replace(
            target,
            status=SwarmStepStatus.CLAIMED,
            claimed_at=now,
            # Match File/SqlAlchemy backends: claim does NOT bump attempts.
            # Only fail_task bumps attempts (a retry happened). This keeps
            # SwarmStepAttempt.attempt numbering consistent across backends.
            attempts=target.attempts,
            version=target.version + 1,
            updated_at=now,
        )
        self._tasks[target.id] = claimed
        return claimed

    async def set_active_run(
        self, task_id: str, run_id: str, *, expected_version: int
    ) -> SwarmStep:
        from linktools.ai.errors import SwarmConflictError, SwarmStepNotFoundError

        if task_id not in self._tasks:
            raise SwarmStepNotFoundError(f"swarm task not found: {task_id}")
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

    async def complete_task(
        self,
        task_id: str,
        result,
        *,
        expected_version: int,
        active_run_id=None,
    ) -> SwarmStep:
        current = self._tasks[task_id]
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if active_run_id is not None and current.active_run_id != active_run_id:
            raise SwarmConflictError(
                f"task {task_id} active_run_id mismatch: expected {active_run_id!r}, "
                f"found {current.active_run_id!r}"
            )
        done = replace(
            current,
            status=SwarmStepStatus.SUCCEEDED,
            result=result,
            version=current.version + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = done
        return done

    async def fail_task(
        self,
        task_id: str,
        error,
        *,
        expected_version: int,
        active_run_id=None,
    ) -> SwarmStep:
        current = self._tasks[task_id]
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if active_run_id is not None and current.active_run_id != active_run_id:
            raise SwarmConflictError(
                f"task {task_id} active_run_id mismatch: expected {active_run_id!r}, "
                f"found {current.active_run_id!r}"
            )
        failed = replace(
            current,
            status=SwarmStepStatus.FAILED,
            error=error,
            version=current.version + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = failed
        return failed

    async def list_tasks(
        self, swarm_run_id: str, *, status: "SwarmStepStatus | None" = None
    ) -> "tuple[SwarmStep, ...]":
        result = [
            t
            for t in self._tasks.values()
            if t.swarm_run_id == swarm_run_id and (status is None or t.status is status)
        ]
        result.sort(key=lambda t: t.created_at)
        return tuple(result)

    async def reclaim_expired_tasks(self, swarm_run_id: str) -> "tuple[SwarmStep, ...]":
        return ()

    async def record_attempt(self, attempt) -> Any:
        # Upsert by attempt.id (mirrors File/SqlAlchemy backends). Track per
        # task_id so list_attempts returns them in insertion order.
        bucket = self._attempts.setdefault(attempt.task_id, [])
        idx = next(
            (i for i, a in enumerate(bucket) if a.id == attempt.id),
            None,
        )
        if idx is None:
            bucket.append(attempt)
        else:
            bucket[idx] = attempt
        return attempt

    async def list_attempts(self, task_id: str):
        return tuple(self._attempts.get(task_id, ()))

    async def renew_lease(
        self, task_id: str, *, expected_version: int, lease_seconds: float
    ) -> SwarmStep:
        from linktools.ai.errors import SwarmConflictError, SwarmStepNotFoundError

        if task_id not in self._tasks:
            raise SwarmStepNotFoundError(f"swarm task not found: {task_id}")
        current = self._tasks[task_id]
        if current.version != expected_version:
            raise SwarmConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        from datetime import timedelta

        new_lease = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
        updated = replace(
            current,
            lease_expires_at=new_lease,
            version=current.version + 1,
            updated_at=datetime.now(timezone.utc),
        )
        self._tasks[task_id] = updated
        return updated


# --- helpers ----------------------------------------------------------------


def _compile_worker(agent_id: str, output_text: str) -> CompiledAgent:
    """Compile an AgentSpec over a FunctionModel that always returns output_text
    as a plain str output (output_schema=str)."""

    def _model_fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=output_text)])

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    spec = AgentSpec(
        id=agent_id,
        name=agent_id,
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions=f"you are {agent_id}"),
        output_schema=str,
    )
    return asyncio.run(compiler.compile(spec))


def _limits(**overrides) -> SwarmLimits:
    base = dict(
        max_rounds=10,
        max_tasks=50,
        max_delegations=20,
        max_depth=5,
        max_concurrency=4,
        max_total_tokens=None,
        max_total_cost=None,
        timeout_seconds=None,
    )
    base.update(overrides)
    return SwarmLimits(**base)


def _swarm_run() -> SwarmRun:
    return SwarmRun(
        id="swarm-1",
        run_id="drive-run-1",
        round=0,
        status=SwarmStatus.RUNNING,
        version=1,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _parent_context() -> RunContext:
    return RunContext(
        run_id="drive-run-1",
        root_run_id="drive-run-1",
        parent_run_id=None,
        session_id="shared-session",
        runnable_id="swarm-spec-1",
        runnable_type=RunnableType.SWARM,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


def _build_ctx(
    tmp_path: Path,
    *,
    agents: "Mapping[str, CompiledAgent]",
    spec: SwarmSpec,
    swarm_store: "SwarmStore | None" = None,
) -> "SwarmExecutionContext":
    run_store = FilesystemRunStore(root=tmp_path / "runs")
    session_store = FilesystemSessionStore(root=tmp_path / "sessions")
    event_store = FilesystemEventStore(root=tmp_path / "events")
    checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
    from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    runner = AgentEngine(
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        commit_coordinator=FilesystemRunCommitCoordinator(
            approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
            checkpoint_store=checkpoint_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
        ),
    )
    # pre-seed the shared session so the driving RunContext is consistent.
    asyncio.run(
        session_store.create(
            SessionRecord(
                id="shared-session",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=ModelRegistry()),
    )
    return SwarmExecutionContext(
        spec=spec,
        swarm_run=_swarm_run(),
        request=RunInput(prompt="do the work"),
        parent_context=_parent_context(),
        dispatcher=runner,
        compiler=compiler,
        agents=agents,
        swarm_store=swarm_store or _MemorySwarmStore(),
        run_store=run_store,
        session_store=session_store,
        event_store=event_store,
        run_definitions=FilesystemRunDefinitionStore(root=tmp_path / "definitions"),
    )


def _make_spec(
    *,
    kind: str,
    limits: SwarmLimits,
    agents: "tuple[AgentRef, ...]",
    coordinator: AgentRef,
) -> SwarmSpec:
    return SwarmSpec(
        id="swarm-spec-1",
        name="test-swarm",
        agents=agents,
        coordinator=coordinator,
        strategy=SwarmStrategySpec(kind=kind),
        limits=limits,
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
    )


# --- 1. CoordinatorDelegationStrategy: 2 tasks round 1, empty round 2 --------


def test_coordinator_delegation_runs_two_workers_and_aggregates(tmp_path):
    compiled_a = _compile_worker("worker-a", "alpha-out")
    compiled_b = _compile_worker("worker-b", "beta-out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(max_rounds=10),
        agents=(AgentRef("coord"), AgentRef("worker-a"), AgentRef("worker-b")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a, "worker-b": compiled_b},
        spec=spec,
        swarm_store=swarm_store,
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
    # 2 SwarmSteps SUCCEEDED.
    assert len(tasks) == 2
    assert all(t.status is SwarmStepStatus.SUCCEEDED for t in tasks)
    # coordinator was invoked exactly twice (round 1 produced work, round 2 empty -> stop).
    assert call_count["n"] == 2


# --- 2. CoordinatorDelegationStrategy: max_rounds exceeded -> raise ----------


def test_coordinator_delegation_raises_when_max_rounds_exceeded(tmp_path):
    compiled_a = _compile_worker("worker-a", "alpha-out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(max_rounds=1),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
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
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=4),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
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
    assert all(t.status is SwarmStepStatus.SUCCEEDED for t in tasks)
    # output is the CONCAT of 3 (same string repeated -> joined by newlines).
    assert str(result.output) == "same-out\nsame-out\nsame-out"
    assert result.metadata["task_count"] == 3


# --- 4. ParallelFanOutStrategy: max_concurrency bounds in-flight runs --------


class _ConcurrencyTrackingDispatcher:
    """Wraps a RunDispatcher; tracks the high-water mark of simultaneously-in-flight
    dispatch() calls. Injects a tiny await sleep so overlap is observable even though
    FunctionModel + FileStore are otherwise near-synchronous between awaits."""

    def __init__(self, inner: RunDispatcher) -> None:
        self._inner = inner
        self.current = 0
        self.max = 0

    async def dispatch(self, request):
        self.current += 1
        self.max = max(self.max, self.current)
        try:
            await asyncio.sleep(0.01)  # force a yield so the semaphore parks coroutines
            return await self._inner.dispatch(request)
        finally:
            self.current -= 1


def test_parallel_fan_out_bounds_concurrency_via_semaphore(tmp_path):
    compiled_a = _compile_worker("worker-a", "out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
    )
    tracker = _ConcurrencyTrackingDispatcher(ctx.dispatcher)
    ctx = replace(ctx, dispatcher=tracker)

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
        kind="parallel_fan_out",
        limits=_limits(max_tasks=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
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
        ParallelFanOutStrategy,
        build_strategy,
    )

    strategy = build_strategy(SwarmStrategySpec(kind="parallel_fan_out"))
    assert isinstance(strategy, ParallelFanOutStrategy)


def test_build_strategy_returns_coordinator_delegation():
    from linktools.ai.swarm.strategy import (
        CoordinatorDelegationStrategy,
        build_strategy,
    )

    strategy = build_strategy(SwarmStrategySpec(kind="coordinator_delegation"))
    assert isinstance(strategy, CoordinatorDelegationStrategy)


def test_build_strategy_unknown_kind_raises_swarm_error():
    from linktools.ai.swarm.strategy import build_strategy

    with pytest.raises(SwarmError):
        build_strategy(SwarmStrategySpec(kind="no_such_strategy"))


# --- 6. SwarmLimits.max_depth enforcement in _run_task ----------------------


def test_run_task_raises_when_depth_exceeds_max_depth(tmp_path):
    """_run_task walks the parent_task_id chain via list_tasks and raises
    SwarmLimitExceededError(kind="max_depth") when the chain depth exceeds the
    limit -- before claiming or dispatching the worker. A depth-2 child
    (parent_task_id -> depth-1 parent) under max_depth=1 fires the guard."""
    from linktools.ai.swarm.strategy import _run_task

    compiled_a = _compile_worker("worker-a", "out")
    swarm_store = _MemorySwarmStore()
    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(max_depth=1),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
    )
    # parent task (depth 1) already SUCCEEDED -> not claimable; it exists only
    # so the child's parent_task_id chain resolves to a real parent.
    parent = SwarmStep(
        id="parent-1",
        swarm_run_id="swarm-1",
        parent_task_id=None,
        assigned_agent_id="worker-a",
        description="parent",
        status=SwarmStepStatus.SUCCEEDED,
        dependencies=(),
        input=TaskInput(prompt="p"),
        result=None,
        error=None,
        attempts=1,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    # child task (depth 2) PENDING -> would be claimed, but the depth guard
    # fires first.
    child = SwarmStep(
        id="child-1",
        swarm_run_id="swarm-1",
        parent_task_id="parent-1",
        assigned_agent_id="worker-a",
        description="child",
        status=SwarmStepStatus.PENDING,
        dependencies=(),
        input=TaskInput(prompt="c"),
        result=None,
        error=None,
        attempts=0,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    asyncio.run(swarm_store.create_task(parent))
    asyncio.run(swarm_store.create_task(child))

    with pytest.raises(SwarmLimitExceededError) as exc_info:
        asyncio.run(_run_task(ctx, child))
    assert exc_info.value.kind == "max_depth"

    # nothing was claimed -- the child is still PENDING (guard fired first).
    tasks = asyncio.run(swarm_store.list_tasks("swarm-1"))
    child_after = next(t for t in tasks if t.id == "child-1")
    assert child_after.status is SwarmStepStatus.PENDING


def test_run_task_depth_chain_walks_multiple_ancestors(tmp_path):
    """A 3-deep chain (child -> parent -> grandparent, all parent_task_id linked)
    under max_depth=2 raises max_depth; confirms the walk iterates beyond the
    immediate parent."""
    from linktools.ai.swarm.strategy import _run_task

    compiled_a = _compile_worker("worker-a", "out")
    swarm_store = _MemorySwarmStore()
    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(max_depth=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
    )
    grandparent = SwarmStep(
        id="gp-1",
        swarm_run_id="swarm-1",
        parent_task_id=None,
        assigned_agent_id="worker-a",
        description="gp",
        status=SwarmStepStatus.SUCCEEDED,
        dependencies=(),
        input=TaskInput(prompt="g"),
        result=None,
        error=None,
        attempts=1,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    parent = SwarmStep(
        id="p-1",
        swarm_run_id="swarm-1",
        parent_task_id="gp-1",
        assigned_agent_id="worker-a",
        description="p",
        status=SwarmStepStatus.SUCCEEDED,
        dependencies=(),
        input=TaskInput(prompt="p"),
        result=None,
        error=None,
        attempts=1,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    child = SwarmStep(
        id="c-1",
        swarm_run_id="swarm-1",
        parent_task_id="p-1",
        assigned_agent_id="worker-a",
        description="c",
        status=SwarmStepStatus.PENDING,
        dependencies=(),
        input=TaskInput(prompt="c"),
        result=None,
        error=None,
        attempts=0,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=_NOW,
        updated_at=_NOW,
    )
    asyncio.run(swarm_store.create_task(grandparent))
    asyncio.run(swarm_store.create_task(parent))
    asyncio.run(swarm_store.create_task(child))

    with pytest.raises(SwarmLimitExceededError) as exc_info:
        asyncio.run(_run_task(ctx, child))
    assert exc_info.value.kind == "max_depth"


# --- 7. task.id != child RunRecord.id (active_run_id decoupling) ---
#
# design note contract: "禁止 SwarmStep.id == child_run_id". Each execution mints a
# fresh run_id and stores it on task.active_run_id via SwarmStore.set_active_run.


def test_task_id_is_different_from_child_run_id(tmp_path):
    """After a successful run, every SUCCEEDED task's active_run_id is set and
    is DIFFERENT from task.id, and matches an existing child RunRecord id."""
    compiled_a = _compile_worker("worker-a", "alpha-out")
    swarm_store = _MemorySwarmStore()
    spec = _make_spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=2),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
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
    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
    )
    # one PENDING task to execute twice.
    task = SwarmStep(
        id="task-1",
        swarm_run_id="swarm-1",
        parent_task_id=None,
        assigned_agent_id="worker-a",
        description="x",
        status=SwarmStepStatus.PENDING,
        dependencies=(),
        input=TaskInput(prompt="do"),
        result=None,
        error=None,
        attempts=0,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=_NOW,
        updated_at=_NOW,
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
        status=SwarmStepStatus.PENDING,
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
    asyncio.run(
        store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    asyncio.run(
        store.create_task(
            SwarmStep(
                id="task-1",
                swarm_run_id="swarm-1",
                parent_task_id=None,
                assigned_agent_id="worker-a",
                description="x",
                status=SwarmStepStatus.PENDING,
                dependencies=(),
                input=TaskInput(prompt="x"),
                result=None,
                error=None,
                attempts=0,
                version=1,
                claimed_at=None,
                lease_expires_at=None,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    claimed = asyncio.run(store.claim_task("swarm-1", "worker-a"))
    assert claimed is not None
    assert claimed.version == 2

    # Simulate a concurrent update bumping version (e.g., a reclaim or another
    # set_active_run) between claim_task and our set_active_run call.
    bumped = asyncio.run(
        store.set_active_run(
            "task-1", "child-run-concurrent", expected_version=claimed.version
        )
    )
    assert bumped.version == 3

    # Now the original caller tries with the STALE version -- conflict.
    with pytest.raises(SwarmConflictError):
        asyncio.run(
            store.set_active_run(
                "task-1", "child-run-stale", expected_version=claimed.version
            )
        )


def test_set_active_run_missing_task_raises_not_found():
    """set_active_run on a missing task id surfaces SwarmStepNotFoundError
    (mirrors complete_task / fail_task behavior)."""
    from linktools.ai.errors import SwarmStepNotFoundError

    store = _MemorySwarmStore()
    asyncio.run(
        store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    with pytest.raises(SwarmStepNotFoundError):
        asyncio.run(store.set_active_run("nope", "child-1", expected_version=1))


# ---------------------------------------------------------------------------
# SwarmStepAttempt recording contract
# ---------------------------------------------------------------------------


def _seed_worker_task(
    swarm_store, *, task_id: str = "task-1", agent_id: str = "worker-a"
):
    """Seed a PENDING SwarmStep into the in-memory store for _run_task."""
    asyncio.run(
        swarm_store.create_task(
            SwarmStep(
                id=task_id,
                swarm_run_id="swarm-1",
                parent_task_id=None,
                assigned_agent_id=agent_id,
                description="x",
                status=SwarmStepStatus.PENDING,
                dependencies=(),
                input=TaskInput(prompt="do"),
                result=None,
                error=None,
                attempts=0,
                version=1,
                claimed_at=None,
                lease_expires_at=None,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )


def test_run_task_records_one_succeeded_attempt_with_run_id_matching_active_run_id(
    tmp_path,
):
    """A successful _run_task records exactly one SwarmStepAttempt whose run_id
    matches the task's active_run_id (child run) and whose attempt
    number is 1 (first execution: task.attempts started at 0 -> 0+1)."""
    from linktools.ai.swarm.models import AttemptStatus
    from linktools.ai.swarm.strategy import _run_task

    compiled_a = _compile_worker("worker-a", "alpha-out")
    swarm_store = _MemorySwarmStore()
    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_a, "worker-a": compiled_a},
        spec=spec,
        swarm_store=swarm_store,
    )
    _seed_worker_task(swarm_store)

    result = asyncio.run(_run_task(ctx, swarm_store._tasks["task-1"]))
    assert result is not None

    after = asyncio.run(swarm_store.list_tasks("swarm-1"))[0]
    attempts = asyncio.run(swarm_store.list_attempts("task-1"))
    assert len(attempts) == 1
    assert attempts[0].status is AttemptStatus.SUCCEEDED
    assert attempts[0].run_id == after.active_run_id
    assert attempts[0].task_id == "task-1"
    assert attempts[0].agent_id == "worker-a"
    assert attempts[0].attempt == 1  # 1-based: first attempt
    assert attempts[0].finished_at is not None


def test_run_task_records_failed_attempt_then_succeeded_on_retry_with_incrementing_numbers(
    tmp_path,
):
    """With max_task_retries=2 and a worker that fails once then succeeds, the
    audit trail records TWO attempts: #1=FAILED, #2=SUCCEEDED. Each attempt
    gets its OWN fresh child run_id and scratch session -- reusing one across
    attempts would make the retry's run_store.create() collide on the same
    primary key under SqlAlchemy storage (a real UNIQUE-constraint failure
    that silently turns every retry into a failure)."""
    from linktools.ai.swarm.models import AttemptStatus
    from linktools.ai.swarm.strategy import _run_task
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.agent.compiler import AgentCompiler
    from linktools.ai.agent.models import AgentSpec
    from linktools.ai.agent.spec import PromptSpec
    from pydantic_ai.messages import ModelResponse, TextPart

    # Stateful model: first call raises, second call succeeds. The exception
    # surfaces from dispatcher.dispatch into _run_task's retry loop.
    call_state = {"n": 0}

    def _flaky_model(messages, info):
        call_state["n"] += 1
        if call_state["n"] == 1:
            # Raise directly: AgentEngine surfaces exceptions to the strategy's
            # retry loop. This is the path _run_task catches.
            raise RuntimeError("transient boom")
        return ModelResponse(parts=[TextPart(content="recovered")])

    registry = ModelRegistry()
    registry.register("flaky-model", model=FunctionModel(_flaky_model))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    flaky_spec = AgentSpec(
        id="worker-flaky",
        name="worker-flaky",
        model=ModelPolicy(primary="flaky-model"),
        instructions=PromptSpec(instructions="you are flaky"),
        output_schema=str,
    )
    compiled_flaky = asyncio.run(compiler.compile(flaky_spec))

    swarm_store = _MemorySwarmStore()
    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-flaky")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled_flaky, "worker-flaky": compiled_flaky},
        spec=spec,
        swarm_store=swarm_store,
    )
    _seed_worker_task(swarm_store, agent_id="worker-flaky")

    # max_task_retries=1 -> two iterations: first fails, second succeeds.
    result = asyncio.run(
        _run_task(
            ctx,
            swarm_store._tasks["task-1"],
            max_task_retries=1,
        )
    )
    assert result is not None

    attempts = asyncio.run(swarm_store.list_attempts("task-1"))
    assert len(attempts) == 2
    # attempt numbers increment 1 -> 2
    assert [a.attempt for a in attempts] == [1, 2]
    # first FAILED, second SUCCEEDED
    assert attempts[0].status is AttemptStatus.FAILED
    assert attempts[1].status is AttemptStatus.SUCCEEDED
    # each attempt mints its OWN child run_id -- reusing one across attempts
    # is what caused the SqlAlchemy primary-key collision this test guards.
    assert attempts[0].run_id != attempts[1].run_id
    # the task's active_run_id tracks the CURRENT (latest) attempt's run_id
    after = asyncio.run(swarm_store.list_tasks("swarm-1"))[0]
    assert after.active_run_id == attempts[1].run_id
    # each attempt also gets its own scratch session -- a failed attempt's
    # partial conversation must never leak into the retry's prompt.
    session_1 = f"swarm:swarm-1:task-1:{attempts[0].run_id}"
    session_2 = f"swarm:swarm-1:task-1:{attempts[1].run_id}"
    assert session_1 != session_2
    assert asyncio.run(ctx.session_store.get(session_1)) is not None
    assert asyncio.run(ctx.session_store.get(session_2)) is not None
    # FAILED attempt carries the error
    assert attempts[0].error is not None
    assert attempts[0].error.error_type == "RuntimeError"
    # SUCCEEDED attempt has no error and a finished_at
    assert attempts[1].error is None
    assert attempts[1].finished_at is not None


def test_run_task_retry_survives_sqlalchemy_run_store_primary_key(tmp_path):
    """Regression: ``ai_runs.id`` is a PRIMARY KEY in SqlAlchemyRunStore.
    Retrying with the SAME child_run_id across attempts made the SECOND
    ``run_store.create()`` call collide on that key and raise IntegrityError,
    which turned a worker that would have succeeded on retry into a
    permanently FAILED task. FilesystemRunStore never caught this because it has no
    uniqueness check and silently overwrites on create() -- this test uses
    the real SqlAlchemy backend specifically so the collision cannot hide."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.agent.compiler import AgentCompiler
    from linktools.ai.agent.spec import AgentSpec, PromptSpec
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.run import SqlAlchemyRunStore
    from linktools.ai.swarm.strategy import _run_task

    call_state = {"n": 0}

    def _flaky_model(messages, info):
        call_state["n"] += 1
        if call_state["n"] == 1:
            raise RuntimeError("transient boom")
        return ModelResponse(parts=[TextPart(content="recovered")])

    async def _scenario():
        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        run_store = SqlAlchemyRunStore(session_factory=session_factory)
        session_store = FilesystemSessionStore(root=tmp_path / "sessions")
        event_store = FilesystemEventStore(root=tmp_path / "events")
        checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
        from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
        from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

        runner = AgentEngine(
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
            commit_coordinator=FilesystemRunCommitCoordinator(
                approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
                checkpoint_store=checkpoint_store,
                run_store=run_store,
                session_store=session_store,
                event_store=event_store,
            ),
        )
        await session_store.create(
            SessionRecord(
                id="shared-session",
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )

        registry = ModelRegistry()
        registry.register("flaky-model", model=FunctionModel(_flaky_model))
        compiler = AgentCompiler(
            tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
            model_resolver=ModelResolver(registry=registry),
        )
        flaky_spec = AgentSpec(
            id="worker-flaky",
            name="worker-flaky",
            model=ModelPolicy(primary="flaky-model"),
            instructions=PromptSpec(instructions="you are flaky"),
            output_schema=str,
        )
        compiled_flaky = await compiler.compile(flaky_spec)

        swarm_store = _MemorySwarmStore()
        await swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
        spec = _make_spec(
            kind="coordinator_delegation",
            limits=_limits(),
            agents=(AgentRef("coord"), AgentRef("worker-flaky")),
            coordinator=AgentRef("coord"),
        )
        ctx = SwarmExecutionContext(
            spec=spec,
            swarm_run=_swarm_run(),
            request=RunInput(prompt="do the work"),
            parent_context=_parent_context(),
            dispatcher=runner,
            compiler=compiler,
            agents={"coord": compiled_flaky, "worker-flaky": compiled_flaky},
            swarm_store=swarm_store,
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
            run_definitions=FilesystemRunDefinitionStore(root=tmp_path / "definitions"),
        )
        await swarm_store.create_task(
            SwarmStep(
                id="task-1",
                swarm_run_id="swarm-1",
                parent_task_id=None,
                assigned_agent_id="worker-flaky",
                description="x",
                status=SwarmStepStatus.PENDING,
                dependencies=(),
                input=TaskInput(prompt="do"),
                result=None,
                error=None,
                attempts=0,
                version=1,
                claimed_at=None,
                lease_expires_at=None,
                created_at=_NOW,
                updated_at=_NOW,
            )
        )

        return await _run_task(ctx, swarm_store._tasks["task-1"], max_task_retries=1)

    result = asyncio.run(_scenario())
    assert result is not None
    assert result.output == "recovered"


def test_run_task_complete_task_conflict_after_worker_success_is_not_a_retry(tmp_path):
    """A worker that ACTUALLY succeeds can still lose complete_task()'s
    fencing check if another caller reclaimed or cancelled the task while
    the worker was running -- genuinely losing ownership (a different
    active_run_id now owns the task), not just racing a stale version. That
    is a persistence conflict, not a worker failure: _run_task must not
    retry the worker and must not call fail_task() on a task whose work
    actually succeeded -- it discards this attempt's (superseded) result and
    returns None. The attempt's audit row, already written RUNNING before
    the worker started, must still be closed out (FAILED/"Superseded") --
    not left stuck RUNNING forever -- so a human reading the trail later can
    see this attempt actually finished, and why it doesn't own the result."""
    from linktools.ai.swarm.models import AttemptStatus
    from linktools.ai.swarm.strategy import _run_task
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.agent.compiler import AgentCompiler
    from linktools.ai.agent.models import AgentSpec
    from linktools.ai.agent.spec import PromptSpec

    call_count = {"n": 0}
    swarm_store = _MemorySwarmStore()

    def _model_fn(messages, info):
        call_count["n"] += 1
        # Simulate a concurrent reclaim landing WHILE this worker is
        # in-flight: something else bumps the version AND swaps
        # active_run_id to a DIFFERENT run (a new claimant's), so this is a
        # genuine ownership loss -- not merely a stale version this same
        # attempt could safely retry past.
        current = swarm_store._tasks["task-1"]
        swarm_store._tasks["task-1"] = replace(
            current,
            version=current.version + 1,
            active_run_id="some-other-run-id",
        )
        return ModelResponse(parts=[TextPart(content="done")])

    registry = ModelRegistry()
    registry.register("worker-model", model=FunctionModel(_model_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    worker_spec = AgentSpec(
        id="worker-a",
        name="worker-a",
        model=ModelPolicy(primary="worker-model"),
        instructions=PromptSpec(instructions="you work"),
        output_schema=str,
    )
    compiled = asyncio.run(compiler.compile(worker_spec))

    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled, "worker-a": compiled},
        spec=spec,
        swarm_store=swarm_store,
    )
    _seed_worker_task(swarm_store)

    result = asyncio.run(
        _run_task(
            ctx,
            swarm_store._tasks["task-1"],
            max_task_retries=2,
        )
    )

    # The conflict is discarded, not retried: the worker model ran exactly
    # once (a wrongful retry would call it again), and _run_task reports
    # None rather than fabricating a result for a task it no longer owns.
    assert call_count["n"] == 1
    assert result is None

    # fail_task was never called -- the task's real status (whatever the
    # reclaiming caller set it to) must be left alone.
    after = swarm_store._tasks["task-1"]
    assert after.status is not SwarmStepStatus.FAILED

    # The attempt's audit row is closed out, not left stuck RUNNING: it is
    # recorded FAILED (AttemptStatus has no other terminal state) but tagged
    # "Superseded" -- distinguishable from a genuine worker error -- rather
    # than silently vanishing from the trail.
    attempts = asyncio.run(swarm_store.list_attempts("task-1"))
    assert len(attempts) == 1
    assert attempts[0].status is AttemptStatus.FAILED
    assert attempts[0].error is not None
    assert attempts[0].error.error_type == "Superseded"
    assert attempts[0].finished_at is not None


def test_run_task_set_active_run_conflict_on_retry_does_not_crash_or_refail(tmp_path):
    """A legitimate worker failure on attempt #1 schedules a retry. If, before
    attempt #2 can even start, another caller reclaims the task (a lease
    expiry resets it PENDING, so it is no longer CLAIMED by this attempt at
    all -- a genuine ownership loss, not merely a stale version), attempt
    #2's set_active_run() loses that fencing race. That is a persistence
    conflict discovered BEFORE the worker ever ran again -- it must not be
    mistaken for a second worker failure (no second model call), must not
    crash _run_task by propagating the raw SwarmConflictError, and must not
    call fail_task() and overwrite whatever terminal status the reclaiming
    caller already set."""
    from linktools.ai.swarm.models import AttemptStatus
    from linktools.ai.swarm.strategy import _run_task
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.agent.compiler import AgentCompiler
    from linktools.ai.agent.models import AgentSpec
    from linktools.ai.agent.spec import PromptSpec

    call_count = {"n": 0}
    swarm_store = _MemorySwarmStore()

    def _model_fn(messages, info):
        call_count["n"] += 1
        # Attempt #1 fails for real (a genuine worker error). Simulate a
        # concurrent reclaim landing in the gap between attempt #1's failure
        # and attempt #2's set_active_run: the task's lease expired and
        # reclaim_expired_tasks() reset it to PENDING (version bumped, status
        # no longer CLAIMED) -- this attempt has genuinely lost ownership,
        # not merely raced a stale version it could safely retry past.
        current = swarm_store._tasks["task-1"]
        swarm_store._tasks["task-1"] = replace(
            current,
            version=current.version + 1,
            status=SwarmStepStatus.PENDING,
        )
        raise RuntimeError("transient boom")

    registry = ModelRegistry()
    registry.register("worker-model", model=FunctionModel(_model_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    worker_spec = AgentSpec(
        id="worker-a",
        name="worker-a",
        model=ModelPolicy(primary="worker-model"),
        instructions=PromptSpec(instructions="you work"),
        output_schema=str,
    )
    compiled = asyncio.run(compiler.compile(worker_spec))

    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled, "worker-a": compiled},
        spec=spec,
        swarm_store=swarm_store,
    )
    _seed_worker_task(swarm_store)

    # max_task_retries=2 -> would be up to 3 iterations, but attempt #2's
    # set_active_run should short-circuit after the FIRST failure.
    result = asyncio.run(
        _run_task(
            ctx,
            swarm_store._tasks["task-1"],
            max_task_retries=2,
        )
    )

    # No crash (SwarmConflictError did not propagate out of _run_task), and
    # the worker model ran exactly once -- attempt #2 never started it.
    assert result is None
    assert call_count["n"] == 1

    # Exactly one FAILED attempt (the real worker failure) -- no second one
    # was fabricated for the set_active_run conflict, and fail_task() was
    # never called to overwrite the task's status a second time.
    attempts = asyncio.run(swarm_store.list_attempts("task-1"))
    assert len(attempts) == 1
    assert attempts[0].status is AttemptStatus.FAILED


def test_run_task_complete_task_stale_version_retries_write_once_and_succeeds(tmp_path):
    """Not every complete_task() conflict means lost ownership. If the task's
    version is merely stale while this attempt still holds it (status still
    CLAIMED, active_run_id still this attempt's -- e.g. a lease renewal
    bumped the version while the worker was in flight), _run_task must retry
    the cheap persistence WRITE once with the fresh version rather than
    discarding a result the worker already successfully produced. Crucially,
    the WORKER itself must not be re-run -- only complete_task() is retried."""
    from linktools.ai.swarm.strategy import _run_task
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.agent.compiler import AgentCompiler
    from linktools.ai.agent.models import AgentSpec
    from linktools.ai.agent.spec import PromptSpec

    call_count = {"n": 0}
    swarm_store = _MemorySwarmStore()

    def _model_fn(messages, info):
        call_count["n"] += 1
        return ModelResponse(parts=[TextPart(content="done")])

    registry = ModelRegistry()
    registry.register("worker-model", model=FunctionModel(_model_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    worker_spec = AgentSpec(
        id="worker-a",
        name="worker-a",
        model=ModelPolicy(primary="worker-model"),
        instructions=PromptSpec(instructions="you work"),
        output_schema=str,
    )
    compiled = asyncio.run(compiler.compile(worker_spec))

    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled, "worker-a": compiled},
        spec=spec,
        swarm_store=swarm_store,
    )
    _seed_worker_task(swarm_store)

    original_complete_task = swarm_store.complete_task
    complete_task_calls = {"n": 0}

    async def _flaky_complete_task(
        task_id, result, *, expected_version, active_run_id=None
    ):
        complete_task_calls["n"] += 1
        if complete_task_calls["n"] == 1:
            # Simulate a lease renewal bumping the version WHILE the worker
            # was in flight -- ownership (status/active_run_id) is untouched.
            current = swarm_store._tasks[task_id]
            swarm_store._tasks[task_id] = replace(
                current,
                version=current.version + 1,
            )
        return await original_complete_task(
            task_id,
            result,
            expected_version=expected_version,
            active_run_id=active_run_id,
        )

    swarm_store.complete_task = _flaky_complete_task

    result = asyncio.run(
        _run_task(
            ctx,
            swarm_store._tasks["task-1"],
            max_task_retries=2,
        )
    )

    # The worker ran exactly once -- the stale-version conflict was resolved
    # by retrying the WRITE (complete_task, called twice: the losing attempt
    # + the fresh-version retry), never the worker.
    assert call_count["n"] == 1
    assert complete_task_calls["n"] == 2
    assert result is not None
    assert result.output == "done"

    after = swarm_store._tasks["task-1"]
    assert after.status is SwarmStepStatus.SUCCEEDED


def _worker_ctx_and_task(tmp_path, model_fn, *, swarm_store=None):
    """Shared setup for the propagation tests below: one CLAIMED task, one
    worker agent backed by ``model_fn``, wired into a fresh SwarmExecutionContext."""
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.agent.compiler import AgentCompiler
    from linktools.ai.agent.models import AgentSpec
    from linktools.ai.agent.spec import PromptSpec

    swarm_store = swarm_store if swarm_store is not None else _MemorySwarmStore()
    registry = ModelRegistry()
    registry.register("worker-model", model=FunctionModel(model_fn))
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry),
    )
    worker_spec = AgentSpec(
        id="worker-a",
        name="worker-a",
        model=ModelPolicy(primary="worker-model"),
        instructions=PromptSpec(instructions="you work"),
        output_schema=str,
    )
    compiled = asyncio.run(compiler.compile(worker_spec))

    asyncio.run(
        swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=_NOW,
                updated_at=_NOW,
            )
        )
    )
    spec = _make_spec(
        kind="coordinator_delegation",
        limits=_limits(),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    ctx = _build_ctx(
        tmp_path,
        agents={"coord": compiled, "worker-a": compiled},
        spec=spec,
        swarm_store=swarm_store,
    )
    _seed_worker_task(swarm_store)
    return ctx, swarm_store


def test_run_task_complete_task_not_found_propagates(tmp_path):
    """SwarmStepNotFoundError from complete_task() is not a fencing conflict
    _retry_fencing_conflict_once knows how to interpret (a task genuinely
    disappearing is not normal ownership transfer -- more likely data
    corruption or a wrong storage root) -- it must propagate out of
    _run_task, not be swallowed as a discarded attempt."""
    from linktools.ai.errors import SwarmStepNotFoundError
    from linktools.ai.swarm.strategy import _run_task

    def _model_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="done")])

    ctx, swarm_store = _worker_ctx_and_task(tmp_path, _model_fn)

    async def _not_found(task_id, result, *, expected_version, active_run_id=None):
        raise SwarmStepNotFoundError(f"swarm task not found: {task_id}")

    swarm_store.complete_task = _not_found

    with pytest.raises(SwarmStepNotFoundError):
        asyncio.run(
            _run_task(
                ctx,
                swarm_store._tasks["task-1"],
                max_task_retries=2,
            )
        )


def test_run_task_complete_task_storage_error_propagates(tmp_path):
    """A raw storage/connection error from complete_task() must not be
    mistaken for a fencing conflict and silently turned into 'this task was
    reclaimed' -- that would let a real outage masquerade as a benign
    ownership race and let the swarm's aggregate silently miss a task whose
    work actually succeeded. It must propagate out of _run_task."""
    from linktools.ai.swarm.strategy import _run_task

    def _model_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="done")])

    ctx, swarm_store = _worker_ctx_and_task(tmp_path, _model_fn)

    async def _db_down(task_id, result, *, expected_version, active_run_id=None):
        raise RuntimeError("db down")

    swarm_store.complete_task = _db_down

    with pytest.raises(RuntimeError, match="db down"):
        asyncio.run(
            _run_task(
                ctx,
                swarm_store._tasks["task-1"],
                max_task_retries=2,
            )
        )


def test_run_task_set_active_run_storage_error_propagates(tmp_path):
    """Same boundary as complete_task above, at the set_active_run() call
    site: a raw storage error is not a fencing conflict and must propagate,
    not be discarded as though the task had merely been reclaimed."""
    from linktools.ai.swarm.strategy import _run_task

    def _model_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="done")])

    ctx, swarm_store = _worker_ctx_and_task(tmp_path, _model_fn)

    async def _db_down(task_id, run_id, *, expected_version):
        raise RuntimeError("db down")

    swarm_store.set_active_run = _db_down

    with pytest.raises(RuntimeError, match="db down"):
        asyncio.run(
            _run_task(
                ctx,
                swarm_store._tasks["task-1"],
                max_task_retries=2,
            )
        )


def test_set_active_run_conflict_discards_without_retrying_with_fresh_claim(tmp_path):
    """Regression for the ownership hole in a retry-with-fresh-version design
    for set_active_run: status == CLAIMED on a re-read proves only that
    SOMEONE claimed the task, not that it is still THIS caller, because
    claim_task never touches active_run_id.

    Scenario: worker A claims the task (version=2). Before A calls
    set_active_run, A's lease expires, reclaim_expired_tasks() resets the
    task to PENDING, and worker B claims it fresh (version=4, CLAIMED). When
    A's set_active_run(expected_version=2) finally runs and conflicts, it
    must discard -- NOT retry with the fresh version=4, which would silently
    overwrite active_run_id with A's child_run_id and steal B's claim."""
    from linktools.ai.swarm.strategy import _run_task

    def _model_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="done")])

    ctx, swarm_store = _worker_ctx_and_task(tmp_path, _model_fn)

    # Worker A claims first -- this is the stale reference _run_task will be
    # driven with below, standing in for "A claimed this a while ago and is
    # only now reaching set_active_run."
    stale_claim = asyncio.run(swarm_store.claim_task("swarm-1", "worker-a"))
    assert stale_claim is not None
    assert stale_claim.version == 2

    # Simulate the lease expiring and reclaim_expired_tasks() resetting the
    # task to PENDING (bumping version, clearing claimed_at). Unlike the real
    # SqlAlchemy/File implementations, _MemorySwarmStore.claim_task() (below)
    # filters candidates by assigned_agent_id, so it is left as "worker-a"
    # here rather than cleared -- irrelevant to the ownership race under
    # test, which turns on status/version/active_run_id, not this field.
    reclaimed = replace(
        swarm_store._tasks["task-1"],
        status=SwarmStepStatus.PENDING,
        claimed_at=None,
        version=swarm_store._tasks["task-1"].version + 1,
    )
    swarm_store._tasks["task-1"] = reclaimed

    # Worker B claims the reclaimed task for real -- the store's actual
    # state is CLAIMED again, but now owned by B, not A.
    b_claim = asyncio.run(swarm_store.claim_task("swarm-1", "worker-a"))
    assert b_claim is not None
    assert b_claim.version == 4

    # Drive _run_task as worker A, still holding its STALE claim (version=2)
    # from before the reclaim. claim_task is monkeypatched to hand back that
    # stale record directly (bypassing the real store's claim queue, which
    # would correctly refuse a second claim on an already-CLAIMED task).
    async def _stale_claim_task(swarm_run_id, agent_id, **kwargs):
        return stale_claim

    swarm_store.claim_task = _stale_claim_task

    set_active_run_calls = {"n": 0}
    original_set_active_run = swarm_store.set_active_run

    async def _counting_set_active_run(task_id, run_id, *, expected_version):
        set_active_run_calls["n"] += 1
        return await original_set_active_run(
            task_id,
            run_id,
            expected_version=expected_version,
        )

    swarm_store.set_active_run = _counting_set_active_run

    result = asyncio.run(_run_task(ctx, stale_claim, max_task_retries=2))

    # Discarded, not retried: set_active_run was attempted exactly once (a
    # wrongful retry-with-fresh-version would call it again with
    # expected_version=4), and the worker model never ran.
    assert result is None
    assert set_active_run_calls["n"] == 1

    # Critically: B's claim on the real store is untouched -- A's conflicting
    # set_active_run must not have overwritten active_run_id.
    after = swarm_store._tasks["task-1"]
    assert after.version == 4
    assert after.active_run_id is None
    assert after.status is SwarmStepStatus.CLAIMED


def test_run_task_attempt_numbering_survives_a_superseded_attempt(tmp_path):
    """base_attempt is sourced from the actual attempt audit trail
    (list_attempts), not task.attempts -- because task.attempts is bumped
    only by fail_task(), and a superseded attempt (worker succeeded, but
    complete_task lost the fencing race) is deliberately never routed
    through fail_task(). Under the old `claimed.attempts + 1` formula, a
    fresh _run_task call after a supersession would see task.attempts still
    0 and wrongly restart numbering at attempt 1, colliding with the
    already-recorded attempt 1 in the trail."""
    from linktools.ai.swarm.models import AttemptStatus
    from linktools.ai.swarm.strategy import _run_task
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.resolver import ModelResolver
    from linktools.ai.agent.compiler import AgentCompiler
    from linktools.ai.agent.models import AgentSpec
    from linktools.ai.agent.spec import PromptSpec

    swarm_store = _MemorySwarmStore()

    # First _run_task call: worker succeeds, but another caller supersedes
    # the task (version + active_run_id swapped) before complete_task lands
    # -- the same scenario as test_run_task_complete_task_conflict_after_
    # worker_success_is_not_a_retry above.
    def _superseded_model_fn(messages, info):
        current = swarm_store._tasks["task-1"]
        swarm_store._tasks["task-1"] = replace(
            current,
            version=current.version + 1,
            active_run_id="some-other-run-id",
        )
        return ModelResponse(parts=[TextPart(content="done")])

    ctx, swarm_store = _worker_ctx_and_task(
        tmp_path,
        _superseded_model_fn,
        swarm_store=swarm_store,
    )

    first_result = asyncio.run(
        _run_task(
            ctx,
            swarm_store._tasks["task-1"],
            max_task_retries=0,
        )
    )
    assert first_result is None

    attempts_after_first = asyncio.run(swarm_store.list_attempts("task-1"))
    assert len(attempts_after_first) == 1
    assert attempts_after_first[0].attempt == 1
    assert attempts_after_first[0].status is AttemptStatus.FAILED
    assert attempts_after_first[0].error.error_type == "Superseded"
    # task.attempts is untouched -- fail_task was never called for a
    # superseded attempt -- so it still reads 0, the exact staleness this
    # fix works around.
    assert swarm_store._tasks["task-1"].attempts == 0

    # Make the task claimable again (standing in for the "other owner" --
    # whoever won the race above -- eventually finishing and the task
    # becoming available for a new attempt via a later PENDING transition;
    # the mechanism doesn't matter here, only that a fresh _run_task call
    # claims it and must number its attempt correctly).
    stale = swarm_store._tasks["task-1"]
    swarm_store._tasks["task-1"] = replace(
        stale,
        status=SwarmStepStatus.PENDING,
        claimed_at=None,
        active_run_id=None,
        version=stale.version + 1,
    )

    # Second _run_task call: worker succeeds cleanly this time (no
    # supersession).
    def _clean_model_fn(messages, info):
        return ModelResponse(parts=[TextPart(content="done-for-real")])

    registry2 = ModelRegistry()
    registry2.register("worker-model-2", model=FunctionModel(_clean_model_fn))
    compiler2 = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_resolver=ModelResolver(registry=registry2),
    )
    worker_spec_2 = AgentSpec(
        id="worker-a",
        name="worker-a",
        model=ModelPolicy(primary="worker-model-2"),
        instructions=PromptSpec(instructions="you work"),
        output_schema=str,
    )
    compiled_2 = asyncio.run(compiler2.compile(worker_spec_2))
    ctx.agents["worker-a"] = compiled_2

    second_result = asyncio.run(
        _run_task(
            ctx,
            swarm_store._tasks["task-1"],
            max_task_retries=0,
        )
    )
    assert second_result is not None
    assert second_result.output == "done-for-real"

    attempts_after_second = asyncio.run(swarm_store.list_attempts("task-1"))
    assert len(attempts_after_second) == 2
    assert [a.attempt for a in attempts_after_second] == [1, 2]
    assert attempts_after_second[1].status is AttemptStatus.SUCCEEDED
