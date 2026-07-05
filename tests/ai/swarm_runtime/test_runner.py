#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for swarm_runtime.runner.SwarmRunner: the orchestrator that compiles
member agents, creates the driving RunRecord + SwarmRun, builds the
SwarmExecutionContext, delegates the round loop to the resolved strategy, writes
ONLY the final aggregate to the shared Session, and transitions the driving Run
to SUCCEEDED. Plus resume() (explicit, caller-driven) and cancel() (store-level).

PROGRAMMATIC -- workers are real CompiledAgents driven by FunctionModel; no real
model calls. Mirrors the test_strategy.py harness conventions."""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent_runtime.compiler import AgentCompiler
from linktools.ai.agent_runtime.spec import AgentSpec, PromptSpec
from linktools.ai.core.model_runtime import ModelRegistry
from linktools.ai.errors import (
    SwarmLimitExceededError,
    SwarmRunNotFoundError,
)
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import (
    RunInput,
    RunRecord,
    RunStatus,
    RunnableType,
)
from linktools.ai.session.models import (
    MessageRole,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.storage.file.checkpoint import FileCheckpointStore
from linktools.ai.storage.file.event import FileEventStore
from linktools.ai.storage.file.run import FileRunStore
from linktools.ai.storage.file.session import FileSessionStore
from linktools.ai.storage.file.swarm import FileSwarmStore
from linktools.ai.swarm_runtime.aggregation import AggregationPolicy
from linktools.ai.swarm_runtime.limits import SwarmLimits
from linktools.ai.swarm_runtime.models import (
    AgentRef,
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskStatus,
    TaskInput,
    TokenUsage,
)
from linktools.ai.swarm_runtime.spec import (
    SwarmContextPolicy,
    SwarmSpec,
    SwarmStrategySpec,
)


_NOW = datetime.now(timezone.utc)


# --- helpers ----------------------------------------------------------------

def _make_model(output_text: str) -> FunctionModel:
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=output_text)])
    return FunctionModel(_fn)


def _build_compiler(*outputs: str) -> AgentCompiler:
    """Build an AgentCompiler with one registered model per output string. The
    model_type for the i-th output is ``f"model-{i}"`` so test specs can request
    a deterministic output by referencing that model_type."""
    registry = ModelRegistry()
    for i, out in enumerate(outputs):
        registry.register(f"model-{i}", model=_make_model(out))
    return AgentCompiler(model_router=ModelRouter(registry=registry))


def _agent_spec(agent_id: str, model_type: str) -> AgentSpec:
    return AgentSpec(
        id=agent_id, name=agent_id,
        model=ModelPolicy(primary=model_type),
        instructions=PromptSpec(instructions=f"you are {agent_id}"),
        output_schema=str,
    )


def _limits(**overrides) -> SwarmLimits:
    base = dict(
        max_rounds=10, max_tasks=50, max_delegations=20, max_depth=5,
        max_concurrency=4, max_total_tokens=None, max_total_cost=None, timeout_seconds=None,
    )
    base.update(overrides)
    return SwarmLimits(**base)


def _spec(
    *,
    kind: str,
    limits: "SwarmLimits | None" = None,
    agents: "tuple[AgentRef, ...]",
    coordinator: AgentRef,
    config: "dict[str, Any] | None" = None,
) -> SwarmSpec:
    return SwarmSpec(
        id="swarm-spec-1", name="test-swarm", agents=agents, coordinator=coordinator,
        strategy=SwarmStrategySpec(kind=kind, config=config or {}),
        limits=limits or _limits(),
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
    )


class _Stores:
    """Wires the five file-backed stores under tmp_path subdirs."""

    def __init__(self, tmp_path: Path) -> None:
        self.run_store = FileRunStore(root=tmp_path / "runs")
        self.session_store = FileSessionStore(root=tmp_path / "sessions")
        self.event_store = FileEventStore(root=tmp_path / "events")
        self.checkpoint_store = FileCheckpointStore(root=tmp_path / "checkpoints")
        self.swarm_store = FileSwarmStore(root=tmp_path / "swarm")

    def seed_shared_session(self, session_id: str) -> None:
        asyncio.run(self.session_store.create(SessionRecord(
            id=session_id, parent_id=None, status=SessionStatus.ACTIVE,
            version=1, created_at=_NOW, updated_at=_NOW,
        )))


def _driving_context(run_id: str, session_id: str) -> RunContext:
    return RunContext(
        run_id=run_id, root_run_id=run_id, parent_run_id=None,
        session_id=session_id, runnable_id="swarm-spec-1",
        runnable_type=RunnableType.SWARM, user_id=None, tenant_id=None, workspace=None,
    )


# --- 1. End-to-end run() with ParallelFanOutStrategy ------------------------

def test_run_parallel_fan_out_aggregates_and_marks_succeeded(tmp_path):
    from linktools.ai.swarm_runtime.runner import SwarmRunner

    # 3 distinct outputs -> 3 registered models; coord is compiled but does not
    # run as a worker (the worker pool excludes the coordinator, so the two
    # member agents worker-a/worker-b each receive one of the 2 fanned-out tasks).
    compiler = _build_compiler("coord-out", "alpha-out", "beta-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store, run_store=stores.run_store,
        session_store=stores.session_store, event_store=stores.event_store,
        checkpoint_store=stores.checkpoint_store, compiler=compiler,
    )

    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=4),
        agents=(AgentRef("coord"), AgentRef("worker-a"), AgentRef("worker-b")),
        coordinator=AgentRef("coord"),
        config={"task_count": 2},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-1"),
        "worker-b": _agent_spec("worker-b", "model-2"),
    }
    context = _driving_context("drive-run-1", "shared-session")

    async def _run():
        return await runner.run(spec, RunInput(prompt="do the work"), context, agents=agents)
    result = asyncio.run(_run())

    # aggregate contains BOTH workers' distinct strings (CONCAT).
    assert "alpha-out" in str(result.output)
    assert "beta-out" in str(result.output)
    assert result.metadata["task_count"] == 2

    async def _verify():
        driving = await stores.run_store.get(context.run_id)
        children = await stores.run_store.list_children(context.run_id)
        messages = await stores.session_store.list_messages(context.session_id)
        events = await stores.event_store.list(context.run_id, limit=100)
        return driving, children, messages, events
    driving, children, messages, events = asyncio.run(_verify())

    # driving Run is SUCCEEDED with runnable_type=SWARM.
    assert driving.status is RunStatus.SUCCEEDED
    assert driving.runnable_type is RunnableType.SWARM
    # 2 child Runs parented to the driving run.
    assert len(children) == 2
    assert all(c.parent_run_id == context.run_id for c in children)
    assert all(c.root_run_id == context.run_id for c in children)
    assert all(c.status is RunStatus.SUCCEEDED for c in children)
    # shared Session has EXACTLY ONE new assistant message (the aggregate).
    assert len(messages) == 1
    assert messages[0].role is MessageRole.ASSISTANT
    assert "alpha-out" in str(messages[0].content)
    assert "beta-out" in str(messages[0].content)
    # events include SwarmStarted + SwarmCompleted.
    payload_types = {type(e.payload).__name__ for e in events.items}
    assert "SwarmStarted" in payload_types
    assert "SwarmCompleted" in payload_types


# --- 2. cancel(swarm_run_id) ------------------------------------------------

def test_cancel_marks_swarm_and_in_flight_children_cancelled(tmp_path):
    from linktools.ai.swarm_runtime.runner import SwarmRunner

    stores = _Stores(tmp_path)
    # construct the in-flight state directly: a RUNNING SwarmRun with one
    # CLAIMED task whose id is also a RUNNING child RunRecord (the invariant
    # strategy._run_task establishes: task.id == child RunRecord.id).
    now = _NOW

    async def _seed():
        await stores.run_store.create(RunRecord(
            id="drive-run-1", root_run_id="drive-run-1", parent_run_id=None,
            session_id="shared", runnable_id="swarm-spec-1",
            runnable_type=RunnableType.SWARM, status=RunStatus.RUNNING,
            input=RunInput(prompt="x"), result=None, error=None, version=1,
            created_at=now, started_at=now, finished_at=None,
        ))
        await stores.swarm_store.create_run(SwarmRun(
            id="swarm-1", run_id="drive-run-1", round=1, status=SwarmStatus.RUNNING,
            version=1, token_usage=TokenUsage(), cost=Decimal("0"),
            created_at=now, updated_at=now,
        ))
        # child RunRecord (id == task.id) in RUNNING state.
        await stores.run_store.create(RunRecord(
            id="task-1", root_run_id="drive-run-1", parent_run_id="drive-run-1",
            session_id="swarm:swarm-1:task-1", runnable_id="worker-a",
            runnable_type=RunnableType.AGENT, status=RunStatus.RUNNING,
            input=RunInput(prompt="sub"), result=None, error=None, version=1,
            created_at=now, started_at=now, finished_at=None,
        ))
        await stores.swarm_store.create_task(SwarmTask(
            id="task-1", swarm_run_id="swarm-1", parent_task_id=None,
            assigned_agent_id="worker-a", description="x", status=SwarmTaskStatus.CLAIMED,
            dependencies=(), input=TaskInput(prompt="sub"), result=None, error=None,
            attempts=1, version=1, claimed_at=now, lease_expires_at=None,
            created_at=now, updated_at=now,
        ))
    asyncio.run(_seed())

    runner = SwarmRunner(
        swarm_store=stores.swarm_store, run_store=stores.run_store,
        session_store=stores.session_store, event_store=stores.event_store,
        checkpoint_store=stores.checkpoint_store,
        compiler=_build_compiler("coord-out"),
    )

    async def _cancel():
        await runner.cancel("swarm-1")
    asyncio.run(_cancel())

    async def _verify():
        swarm = await stores.swarm_store.get_run("swarm-1")
        child = await stores.run_store.get("task-1")
        return swarm, child
    swarm, child = asyncio.run(_verify())

    assert swarm.status is SwarmStatus.CANCELLED
    assert child.status is RunStatus.CANCELLED


def test_cancel_unknown_swarm_run_raises(tmp_path):
    from linktools.ai.swarm_runtime.runner import SwarmRunner

    stores = _Stores(tmp_path)
    runner = SwarmRunner(
        swarm_store=stores.swarm_store, run_store=stores.run_store,
        session_store=stores.session_store, event_store=stores.event_store,
        checkpoint_store=stores.checkpoint_store,
        compiler=_build_compiler("coord-out"),
    )

    async def _cancel():
        await runner.cancel("no-such-swarm")
    with pytest.raises(SwarmRunNotFoundError):
        asyncio.run(_cancel())


# --- 3. Strategy exceeding max_rounds surfaces failure ----------------------

def test_run_surfaces_strategy_limit_exceed_as_failed_run(tmp_path):
    from linktools.ai.swarm_runtime.runner import SwarmRunner

    compiler = _build_compiler("alpha-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store, run_store=stores.run_store,
        session_store=stores.session_store, event_store=stores.event_store,
        checkpoint_store=stores.checkpoint_store, compiler=compiler,
    )

    # coordinator that ALWAYS emits a task -> blows past max_rounds=1.
    async def coordinator_fn(swarm_run, completed, limits):
        return (TaskInput(prompt="more"),)

    spec = _spec(
        kind="coordinator_delegation",
        limits=_limits(max_rounds=1),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        config={"coordinator_fn": coordinator_fn},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-0"),
    }
    context = _driving_context("drive-run-1", "shared-session")

    async def _run():
        await runner.run(spec, RunInput(prompt="do the work"), context, agents=agents)
    with pytest.raises(SwarmLimitExceededError):
        asyncio.run(_run())

    # driving Run is FAILED; SwarmRun is FAILED (best-effort cleanup).
    async def _verify():
        driving = await stores.run_store.get(context.run_id)
        # the SwarmRun id is generated inside run(); find it via the store.
        runs_dir = stores.swarm_store._runs_dir
        swarm_run_files = sorted(runs_dir.glob("*.json"))
        swarm = None
        if swarm_run_files:
            swarm = await stores.swarm_store.get_run(
                json.loads(swarm_run_files[0].read_text())["id"]
            )
        return driving, swarm
    driving, swarm = asyncio.run(_verify())

    assert driving.status is RunStatus.FAILED
    assert driving.error is not None
    assert swarm is not None
    assert swarm.status is SwarmStatus.FAILED


# --- 4. resume(swarm_run_id) after partial failure --------------------------

def test_resume_after_partial_failure_completes(tmp_path):
    from linktools.ai.swarm_runtime.runner import SwarmRunner

    compiler = _build_compiler("resumed-out")
    stores = _Stores(tmp_path)
    now = _NOW

    # seed an interrupted swarm: driving Run RUNNING, SwarmRun RUNNING, one
    # already-FAILED task (a worker that crashed mid-flight). The default
    # coordinator returns one task on its first call (when no tasks have yet
    # SUCCEEDED), so resume re-enters the round loop, runs one new worker task
    # to success, then aggregates and completes.
    async def _seed():
        await stores.run_store.create(RunRecord(
            id="drive-run-1", root_run_id="drive-run-1", parent_run_id=None,
            session_id="shared", runnable_id="swarm-spec-1",
            runnable_type=RunnableType.SWARM, status=RunStatus.RUNNING,
            input=RunInput(prompt="do the work"), result=None, error=None, version=1,
            created_at=now, started_at=now, finished_at=None,
        ))
        await stores.swarm_store.create_run(SwarmRun(
            id="swarm-1", run_id="drive-run-1", round=0, status=SwarmStatus.RUNNING,
            version=1, token_usage=TokenUsage(), cost=Decimal("0"),
            created_at=now, updated_at=now,
        ))
        # one FAILED task from the interrupted attempt.
        await stores.swarm_store.create_task(SwarmTask(
            id="task-failed", swarm_run_id="swarm-1", parent_task_id=None,
            assigned_agent_id="worker-a", description="crashed", status=SwarmTaskStatus.FAILED,
            dependencies=(), input=TaskInput(prompt="crashed"), result=None,
            error=None, attempts=1, version=1, claimed_at=now, lease_expires_at=None,
            created_at=now, updated_at=now,
        ))
    asyncio.run(_seed())

    runner = SwarmRunner(
        swarm_store=stores.swarm_store, run_store=stores.run_store,
        session_store=stores.session_store, event_store=stores.event_store,
        checkpoint_store=stores.checkpoint_store, compiler=compiler,
    )

    spec = _spec(
        kind="coordinator_delegation",
        limits=_limits(max_rounds=10),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-0"),
    }

    async def _resume():
        return await runner.resume("swarm-1", spec, agents=agents)
    result = asyncio.run(_resume())

    # the new worker task ran to success -> aggregate carries its output.
    assert "resumed-out" in str(result.output)

    async def _verify():
        driving = await stores.run_store.get("drive-run-1")
        swarm = await stores.swarm_store.get_run("swarm-1")
        tasks = await stores.swarm_store.list_tasks("swarm-1")
        return driving, swarm, tasks
    driving, swarm, tasks = asyncio.run(_verify())

    assert driving.status is RunStatus.SUCCEEDED
    assert swarm.status is SwarmStatus.SUCCEEDED
    # the originally-FAILED task is still FAILED; at least one task SUCCEEDED.
    statuses = {t.status for t in tasks}
    assert SwarmTaskStatus.FAILED in statuses
    assert SwarmTaskStatus.SUCCEEDED in statuses


def test_resume_unknown_swarm_run_raises(tmp_path):
    from linktools.ai.swarm_runtime.runner import SwarmRunner

    stores = _Stores(tmp_path)
    runner = SwarmRunner(
        swarm_store=stores.swarm_store, run_store=stores.run_store,
        session_store=stores.session_store, event_store=stores.event_store,
        checkpoint_store=stores.checkpoint_store,
        compiler=_build_compiler("coord-out"),
    )
    spec = _spec(
        kind="parallel_fan_out",
        agents=(AgentRef("coord"),), coordinator=AgentRef("coord"),
    )

    async def _resume():
        await runner.resume("no-such-swarm", spec, agents={})
    with pytest.raises(SwarmRunNotFoundError):
        asyncio.run(_resume())
