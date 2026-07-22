#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for swarm.runner.SwarmRunner: the orchestrator that compiles
member agents, creates the driving RunRecord + SwarmRun, builds the
SwarmExecutionContext, delegates the round loop to the resolved strategy, writes
ONLY the final aggregate to the shared Session, and transitions the driving Run
to SUCCEEDED. Plus resume() (explicit, caller-driven) and cancel() (store-level).

PROGRAMMATIC -- workers are real CompiledAgents driven by FunctionModel; no real
model calls. Mirrors the test_strategy.py harness conventions."""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.engine import AgentEngine
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.run.controller import RunController
from linktools.ai.model.registry import ModelRegistry
from linktools.ai.errors import (
    SwarmLimitExceededError,
    SwarmRunNotFoundError,
)
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelGateway, ModelResolver
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import (
    RunErrorInfo,
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
    RunnableType,
)
from linktools.ai.session.models import (
    MessageRole,
    SessionRecord,
    SessionStatus,
)
from linktools.ai.storage.filesystem.approval import FilesystemApprovalStore
from linktools.ai.storage.filesystem.checkpoint import FilesystemCheckpointStore
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator
from linktools.ai.storage.filesystem.definition import FilesystemRunDefinitionStore
from linktools.ai.storage.filesystem.event import FilesystemEventStore
from linktools.ai.storage.filesystem.run import FilesystemRunStore
from linktools.ai.storage.filesystem.session import FilesystemSessionStore
from linktools.ai.storage.filesystem.swarm import FilesystemSwarmStore
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
from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.tool.executor import GovernedToolInvoker


_NOW = datetime.now(timezone.utc)


# --- helpers ----------------------------------------------------------------


def _make_model(output_text: str) -> FunctionModel:
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=output_text)])

    return FunctionModel(_fn)


def _make_model_with_usage(
    output_text: str, *, input_tokens: int, output_tokens: int
) -> FunctionModel:
    """Variant of _make_model that also reports token usage on each response --
    needed for max_total_tokens enforcement (the swarm accumulates
    RunResult.token_usage which AgentEngine populates from run_result.usage)."""
    from pydantic_ai.usage import RunUsage

    usage = RunUsage(input_tokens=input_tokens, output_tokens=output_tokens)

    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=output_text)], usage=usage)

    return FunctionModel(_fn)


def _build_compiler(*outputs: str) -> AgentCompiler:
    """Build an AgentCompiler with one registered model per output string. The
    model_type for the i-th output is ``f"model-{i}"`` so test specs can request
    a deterministic output by referencing that model_type."""
    registry = ModelRegistry()
    for i, out in enumerate(outputs):
        registry.register(f"model-{i}", model=_make_model(out))
    return AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_router=ModelGateway(ModelResolver(registry=registry)),
    )


def _agent_spec(agent_id: str, model_type: str) -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        name=agent_id,
        model=ModelPolicy(primary=model_type),
        instructions=PromptSpec(instructions=f"you are {agent_id}"),
        output_schema=str,
    )


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


def _spec(
    *,
    kind: str,
    limits: "SwarmLimits | None" = None,
    agents: "tuple[AgentRef, ...]",
    coordinator: AgentRef,
    config: "dict[str, Any] | None" = None,
) -> SwarmSpec:
    return SwarmSpec(
        id="swarm-spec-1",
        name="test-swarm",
        agents=agents,
        coordinator=coordinator,
        strategy=SwarmStrategySpec(kind=kind, config=config or {}),
        limits=limits or _limits(),
        context_policy=SwarmContextPolicy(),
        aggregation=AggregationPolicy(),
    )


class _Stores:
    """Wires the five file-backed stores under tmp_path subdirs, plus one
    Runtime-style AgentEngine + RunController -- SwarmRunner no longer builds
    its own AgentEngine (scenario: it must reuse the one Runtime.build()
    assembles), so tests build it here exactly like Runtime.build() does."""

    def __init__(self, tmp_path: Path) -> None:
        self.run_store = FilesystemRunStore(root=tmp_path / "runs")
        self.session_store = FilesystemSessionStore(root=tmp_path / "sessions")
        self.event_store = FilesystemEventStore(root=tmp_path / "events")
        self.checkpoint_store = FilesystemCheckpointStore(root=tmp_path / "checkpoints")
        self.swarm_store = FilesystemSwarmStore(root=tmp_path / "swarm")
        self.run_definitions = FilesystemRunDefinitionStore(root=tmp_path / "definitions")
        self.run_controller = RunController()
        self.agent_runner = AgentEngine(
            run_store=self.run_store,
            session_store=self.session_store,
            event_store=self.event_store,
            checkpoint_store=self.checkpoint_store,
            run_controller=self.run_controller,
            commit_coordinator=FilesystemRunCommitCoordinator(
                approval_store=FilesystemApprovalStore(root=tmp_path / "approvals"),
                checkpoint_store=self.checkpoint_store,
                run_store=self.run_store,
                session_store=self.session_store,
                event_store=self.event_store,
            ),
        )

    def seed_shared_session(self, session_id: str) -> None:
        asyncio.run(
            self.session_store.create(
                SessionRecord(
                    id=session_id,
                    parent_id=None,
                    status=SessionStatus.ACTIVE,
                    version=1,
                    created_at=_NOW,
                    updated_at=_NOW,
                )
            )
        )


def _driving_context(run_id: str, session_id: str) -> RunContext:
    return RunContext(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id=session_id,
        runnable_id="swarm-spec-1",
        runnable_type=RunnableType.SWARM,
        user_id=None,
        tenant_id=None,
        workspace=None,
    )


# --- 1. End-to-end run() with ParallelFanOutStrategy ------------------------


def test_run_parallel_fan_out_aggregates_and_marks_succeeded(tmp_path):
    from linktools.ai.swarm.runner import SwarmRunner

    # 3 distinct outputs -> 3 registered models; coord is compiled but does not
    # run as a worker (the worker pool excludes the coordinator, so the two
    # member agents worker-a/worker-b each receive one of the 2 fanned-out tasks).
    compiler = _build_compiler("coord-out", "alpha-out", "beta-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
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
        return await runner.run(
            spec, RunInput(prompt="do the work"), context, agents=agents
        )

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


def test_swarm_run_persists_driving_and_worker_run_definition_snapshots(tmp_path):
    """a swarm run persists a RunDefinitionSnapshot for the driving run
    AND for each worker child run, so Runtime.resume(child_run_id) can restore a
    worker that paused on approval. Both prepare_swarm_run (driving) and the
    worker's prepare_agent_run are unconditional -- this fails if either is
    re-gated behind an Optional RunDefinitionStore."""
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("coord-out", "alpha-out", "beta-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
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
    context = _driving_context("drive-run-snap", "shared-session")

    async def _run():
        return await runner.run(
            spec, RunInput(prompt="do the work"), context, agents=agents
        )

    result = asyncio.run(_run())
    assert "alpha-out" in str(result.output)

    async def _verify():
        children = await stores.run_store.list_children(context.run_id)
        driving_snapshot = await stores.run_definitions.get(context.run_id)
        worker_snapshots = {
            child.id: await stores.run_definitions.get(child.id) for child in children
        }
        return children, driving_snapshot, worker_snapshots

    children, driving_snapshot, worker_snapshots = asyncio.run(_verify())

    # Driving swarm run has a snapshot.
    assert driving_snapshot is not None, "driving swarm run has no snapshot"
    assert driving_snapshot.runnable_type == str(RunnableType.SWARM.value)
    # Each worker child run has a snapshot (the unconditional worker
    # prepare_agent_run in the strategy).
    assert len(children) == 2
    for child_id, snap in worker_snapshots.items():
        assert snap is not None, f"worker run {child_id} has no snapshot"
        assert snap.runnable_type == str(RunnableType.AGENT.value)


# --- 2. cancel(swarm_run_id) ------------------------------------------------


def test_cancel_marks_swarm_and_in_flight_children_cancelled(tmp_path):
    from linktools.ai.swarm.runner import SwarmRunner

    stores = _Stores(tmp_path)
    # construct the in-flight state directly: a RUNNING SwarmRun with one
    # CLAIMED task whose active_run_id points at a RUNNING child RunRecord
    # (the invariant: task.active_run_id == child RunRecord.id,
    # NOT task.id == child RunRecord.id).
    now = _NOW

    async def _seed():
        await stores.run_store.create(
            RunRecord(
                id="drive-run-1",
                root_run_id="drive-run-1",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=RunStatus.RUNNING,
                input=RunInput(prompt="x"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=1,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )
        # child RunRecord in RUNNING state. Its id is DIFFERENT from
        # the task's id -- cancel() must locate it via task.active_run_id.
        await stores.run_store.create(
            RunRecord(
                id="child-run-1",
                root_run_id="drive-run-1",
                parent_run_id="drive-run-1",
                session_id="swarm:swarm-1:task-1",
                runnable_id="worker-a",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.RUNNING,
                input=RunInput(prompt="sub"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        await stores.swarm_store.create_task(
            SwarmTask(
                id="task-1",
                swarm_run_id="swarm-1",
                parent_task_id=None,
                assigned_agent_id="worker-a",
                description="x",
                status=SwarmTaskStatus.CLAIMED,
                dependencies=(),
                input=TaskInput(prompt="sub"),
                result=None,
                error=None,
                attempts=1,
                version=1,
                claimed_at=now,
                lease_expires_at=None,
                created_at=now,
                updated_at=now,
                active_run_id="child-run-1",
            )
        )

    asyncio.run(_seed())

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=_build_compiler("coord-out"),
    )

    async def _cancel():
        await runner.cancel("swarm-1")

    asyncio.run(_cancel())

    async def _verify():
        swarm = await stores.swarm_store.get_run("swarm-1")
        child = await stores.run_store.get("child-run-1")
        # and the task.id RunRecord was never created -> cancel must NOT have
        # tried to transition it (proves cancel used active_run_id, not task.id).
        return swarm, child

    swarm, child = asyncio.run(_verify())

    assert swarm.status is SwarmStatus.CANCELLED
    assert child.status is RunStatus.CANCELLED


def test_cancel_propagates_through_run_controller_to_active_child(tmp_path):
    """scenario (actionable-fix-contract): when the driving run AND the
    active child run are both registered with the SAME RunController (as
    Runtime.build() wires them), cancel() must go through CANCELLING first
    (not straight to CANCELLED) and must call run_controller.cancel() for
    BOTH -- proving it's signaling the controller, not just flipping store
    status. Uses real (but idle) asyncio.Tasks as the "in-flight" registration
    so run_controller.get_token(...) returns non-None, matching what
    AgentEngine.execute()/SwarmRunner.run() do for real in-flight runs."""
    from linktools.ai.swarm.runner import SwarmRunner
    from linktools.ai.run.cancellation import CancellationToken

    stores = _Stores(tmp_path)
    now = _NOW

    async def _seed():
        await stores.run_store.create(
            RunRecord(
                id="drive-run-2",
                root_run_id="drive-run-2",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=RunStatus.RUNNING,
                input=RunInput(prompt="x"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm-2",
                run_id="drive-run-2",
                round=1,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )
        await stores.run_store.create(
            RunRecord(
                id="child-run-2",
                root_run_id="drive-run-2",
                parent_run_id="drive-run-2",
                session_id="swarm:swarm-2:task-2",
                runnable_id="worker-a",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.RUNNING,
                input=RunInput(prompt="sub"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        await stores.swarm_store.create_task(
            SwarmTask(
                id="task-2",
                swarm_run_id="swarm-2",
                parent_task_id=None,
                assigned_agent_id="worker-a",
                description="x",
                status=SwarmTaskStatus.CLAIMED,
                dependencies=(),
                input=TaskInput(prompt="sub"),
                result=None,
                error=None,
                attempts=1,
                version=1,
                claimed_at=now,
                lease_expires_at=None,
                created_at=now,
                updated_at=now,
                active_run_id="child-run-2",
            )
        )
        # Register both the driving run and the active child as "in-flight"
        # with the SAME RunController -- exactly what Runtime.build()-wired
        # SwarmRunner.run()/AgentEngine.execute() do for real runs.
        driving_task = asyncio.ensure_future(asyncio.sleep(100))
        child_task = asyncio.ensure_future(asyncio.sleep(100))
        await stores.run_controller.register(
            "drive-run-2", driving_task, CancellationToken()
        )
        await stores.run_controller.register(
            "child-run-2", child_task, CancellationToken()
        )
        return driving_task, child_task

    driving_task, child_task = asyncio.run(_seed())

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        run_controller=stores.run_controller,
        compiler=_build_compiler("coord-out"),
    )

    async def _cancel():
        await runner.cancel("swarm-2")
        # Give the cancelled tasks a tick to observe cancellation.
        for t in (driving_task, child_task):
            try:
                await t
            except asyncio.CancelledError:
                pass

    asyncio.run(_cancel())

    async def _verify():
        swarm = await stores.swarm_store.get_run("swarm-2")
        driving = await stores.run_store.get("drive-run-2")
        child = await stores.run_store.get("child-run-2")
        return swarm, driving, child

    swarm, driving, child = asyncio.run(_verify())

    # Real cancellation was signaled: the registered asyncio.Tasks were
    # actually cancelled (not just a store-status flip).
    assert driving_task.cancelled()
    assert child_task.cancelled()
    # Store transitioned through CANCELLING (SwarmRunner.run()'s own
    # CancelledError handler would finish CANCELLING -> CANCELLED for a real
    # run; here the driving/swarm records stay at CANCELLING because there is
    # no real run() coroutine draining -- proving cancel() itself does NOT
    # jump straight to CANCELLED when a controller registration exists).
    assert swarm.status is SwarmStatus.CANCELLING
    assert driving.status is RunStatus.CANCELLING
    # The active child, however, cancel() DOES drive through to CANCELLING
    # itself (no separate coroutine owns that transition for children).
    assert child.status is RunStatus.CANCELLING


def test_cancel_is_idempotent_on_terminal_swarm_run(tmp_path):
    from linktools.ai.swarm.runner import SwarmRunner

    stores = _Stores(tmp_path)
    now = _NOW

    async def _seed():
        await stores.run_store.create(
            RunRecord(
                id="drive-run-3",
                root_run_id="drive-run-3",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=RunStatus.SUCCEEDED,
                input=RunInput(prompt="x"),
                result=RunResult(output="done"),
                error=None,
                version=2,
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm-3",
                run_id="drive-run-3",
                round=1,
                status=SwarmStatus.SUCCEEDED,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )

    asyncio.run(_seed())

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        run_controller=stores.run_controller,
        compiler=_build_compiler("coord-out"),
    )

    async def _cancel():
        await runner.cancel("swarm-3")  # must be a no-op, not raise

    asyncio.run(_cancel())

    async def _verify():
        return await stores.swarm_store.get_run("swarm-3")

    swarm = asyncio.run(_verify())
    assert swarm.status is SwarmStatus.SUCCEEDED  # unchanged


def test_swarm_runner_reuses_injected_agent_runner(tmp_path):
    """scenario (actionable-fix-contract): SwarmRunner must not construct its
    own AgentEngine -- it stores exactly the instance it was given, so Swarm
    worker Runs inherit the SAME Tool/Policy/Middleware/UoW/Cancellation
    wiring as top-level Agent runs."""
    from linktools.ai.swarm.runner import SwarmRunner

    stores = _Stores(tmp_path)
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        run_controller=stores.run_controller,
        compiler=_build_compiler("coord-out"),
    )
    assert runner._dispatcher is stores.agent_runner
    assert runner._run_controller is stores.run_controller


def test_cancel_unknown_swarm_run_raises(tmp_path):
    from linktools.ai.swarm.runner import SwarmRunner

    stores = _Stores(tmp_path)
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=_build_compiler("coord-out"),
    )

    async def _cancel():
        await runner.cancel("no-such-swarm")

    with pytest.raises(SwarmRunNotFoundError):
        asyncio.run(_cancel())


# --- 3. Strategy exceeding max_rounds surfaces failure ----------------------


def test_run_surfaces_strategy_limit_exceed_as_failed_run(tmp_path):
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("alpha-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
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

    # driving Run is FAILED (SwarmRun failure is best-effort cleanup).
    driving = asyncio.run(stores.run_store.get(context.run_id))

    assert driving.status is RunStatus.FAILED
    assert driving.error is not None


# --- 4. resume(swarm_run_id) after partial failure --------------------------


def test_resume_after_partial_failure_completes(tmp_path):
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("resumed-out")
    stores = _Stores(tmp_path)
    now = _NOW

    # seed an explicitly recoverable swarm: the driving Run is RUNNING, the
    # SwarmRun is RECOVERABLE, and one already-FAILED task (a worker that
    # crashed mid-flight). The default
    # coordinator returns one task on its first call (when no tasks have yet
    # SUCCEEDED), so resume re-enters the round loop, runs one new worker task
    # to success, then aggregates and completes.
    async def _seed():
        await stores.run_store.create(
            RunRecord(
                id="drive-run-1",
                root_run_id="drive-run-1",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=RunStatus.RUNNING,
                input=RunInput(prompt="do the work"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=0,
                status=SwarmStatus.RECOVERABLE,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )
        # one FAILED task from the interrupted attempt.
        await stores.swarm_store.create_task(
            SwarmTask(
                id="task-failed",
                swarm_run_id="swarm-1",
                parent_task_id=None,
                assigned_agent_id="worker-a",
                description="crashed",
                status=SwarmTaskStatus.FAILED,
                dependencies=(),
                input=TaskInput(prompt="crashed"),
                result=None,
                error=None,
                attempts=1,
                version=1,
                claimed_at=now,
                lease_expires_at=None,
                created_at=now,
                updated_at=now,
            )
        )

    asyncio.run(_seed())

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
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

    # The test seeds state manually (no runner.run), so create the snapshot
    # explicitly so resume can restore the spec + member agents.
    import hashlib as _hashlib

    from linktools.ai.json import canonical_json as _cj
    from linktools.ai.run.definition import (
        RunDefinitionSnapshot as _Snap,
        serialize_agent_spec as _sa,
        serialize_swarm_spec as _ss,
    )

    _serialized = {
        "type": "swarm",
        "spec": _ss(spec),
        "members": {aid: _sa(a) for aid, a in agents.items()},
    }
    asyncio.run(
        stores.run_definitions.create(
            _Snap(
                run_id="drive-run-1",
                runnable_type="swarm",
                runnable_id="swarm-spec-1",
                serialized_spec=_serialized,
                spec_fingerprint=_hashlib.sha256(_cj(_serialized).encode()).hexdigest(),
                user_id=None,
                tenant_id=None,
                workspace=None,
                provider_revision=None,
                created_at=_NOW,
            )
        )
    )

    async def _resume():
        return await runner.resume("swarm-1")

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
    from linktools.ai.swarm.runner import SwarmRunner

    stores = _Stores(tmp_path)
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=_build_compiler("coord-out"),
    )

    async def _resume():
        await runner.resume("no-such-swarm")

    with pytest.raises(SwarmRunNotFoundError):
        asyncio.run(_resume())


# --- 5. SwarmLimits.timeout_seconds wraps strategy.run ----------------------


def test_run_timeout_transitions_driving_run_and_swarm_to_failed(tmp_path):
    """SwarmLimits.timeout_seconds wraps strategy.run(ctx) in asyncio.wait_for.
    A coordinator that sleeps past the timeout -> asyncio.TimeoutError ->
    driving Run FAILED + SwarmRun FAILED (best-effort cleanup in the except)."""
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("coord-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
    )

    async def slow_coordinator(swarm_run, completed, limits):
        # Force the strategy's round loop to block past the timeout.
        await asyncio.sleep(10)
        return ()

    spec = _spec(
        kind="coordinator_delegation",
        limits=_limits(timeout_seconds=0.05),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        config={"coordinator_fn": slow_coordinator},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-0"),
    }
    context = _driving_context("drive-to", "shared-session")

    async def _run():
        await runner.run(spec, RunInput(prompt="do the work"), context, agents=agents)

    with pytest.raises(Exception):
        asyncio.run(_run())

    async def _verify():
        driving = await stores.run_store.get(context.run_id)
        return driving

    driving = asyncio.run(_verify())
    assert driving.status is RunStatus.FAILED
    assert driving.error is not None
    # The timeout surfaces as a descriptive message, not a bare empty TimeoutError.
    assert "timeout" in driving.error.message.lower()


def test_run_cancelled_while_already_cancelling_still_reaches_cancelled(tmp_path):
    """P1-1 (current-review-actionable-fix-contract): reproduces the exact
    race Runtime.cancel(run_id) creates -- it transitions the driving Run to
    CANCELLING BEFORE calling run_controller.cancel(), which is what actually
    delivers the CancelledError into SwarmRunner.run(). Before the fix,
    SwarmRunner's CancelledError handler unconditionally tried
    RUNNING -> CANCELLING first, which fails with InvalidRunTransitionError
    when the record is ALREADY CANCELLING (CANCELLING is not a valid source
    for a CANCELLING target) -- silently swallowed by the best-effort
    except-Exception wrapper, leaving the run stuck in CANCELLING forever."""
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("coord-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        run_controller=stores.run_controller,
        compiler=compiler,
    )

    async def slow_coordinator(swarm_run, completed, limits):
        await asyncio.sleep(10)
        return ()

    spec = _spec(
        kind="coordinator_delegation",
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        config={"coordinator_fn": slow_coordinator},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-0"),
    }
    context = _driving_context("drive-cancel-race", "shared-session")

    async def _scenario():
        task = asyncio.ensure_future(
            runner.run(spec, RunInput(prompt="do the work"), context, agents=agents)
        )
        # Give run() a chance to create the driving RunRecord and register
        # with run_controller (mirrors AgentEngine.execute()'s own startup).
        for _ in range(50):
            if stores.run_controller.get_token(context.run_id) is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("run() never registered with run_controller")

        # Replicate exactly what Runtime.cancel(run_id) does: transition to
        # CANCELLING FIRST, then signal the controller.
        driving = await stores.run_store.get(context.run_id)
        await stores.run_store.transition(
            context.run_id,
            RunStatus.CANCELLING,
            expected_version=driving.version,
        )
        await stores.run_controller.cancel(context.run_id)

        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_scenario())

    async def _verify():
        driving = await stores.run_store.get(context.run_id)
        return driving

    driving = asyncio.run(_verify())
    assert driving.status is RunStatus.CANCELLED, (
        f"driving run stuck at {driving.status} -- must reach CANCELLED "
        f"even when it was already CANCELLING before the CancelledError handler ran"
    )


# --- 6. SwarmLimits.max_total_tokens accumulation ---------------------------


def test_run_max_total_tokens_exceeded_raises_and_marks_failed(tmp_path):
    """Two worker tasks each report input=100 + output=100 = 200 tokens; with
    max_total_tokens=300 the accumulated 400 > 300 fires after strategy.run
    returns -> SwarmLimitExceededError(kind="max_total_tokens") + FAILED."""
    from linktools.ai.swarm.runner import SwarmRunner

    # model-0 = coord (never runs as a worker); model-1 = worker producing usage.
    registry = ModelRegistry()
    registry.register("model-0", model=_make_model("coord-out"))
    registry.register(
        "model-1",
        model=_make_model_with_usage("worker-out", input_tokens=100, output_tokens=100),
    )
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_router=ModelGateway(ModelResolver(registry=registry)),
    )

    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
    )

    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_total_tokens=300),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        config={"task_count": 2},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-1"),
    }
    context = _driving_context("drive-tt", "shared-session")

    async def _run():
        await runner.run(spec, RunInput(prompt="do the work"), context, agents=agents)

    with pytest.raises(SwarmLimitExceededError) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.kind == "max_total_tokens"

    async def _verify():
        driving = await stores.run_store.get(context.run_id)
        return driving

    driving = asyncio.run(_verify())
    assert driving.status is RunStatus.FAILED


def test_run_under_max_total_tokens_succeeds(tmp_path):
    """Sanity: when accumulated tokens fit under max_total_tokens the run
    completes normally -- confirms the check doesn't fire false positives."""
    from linktools.ai.swarm.runner import SwarmRunner

    registry = ModelRegistry()
    registry.register("model-0", model=_make_model("coord-out"))
    registry.register(
        "model-1",
        model=_make_model_with_usage("worker-out", input_tokens=50, output_tokens=50),
    )
    compiler = AgentCompiler(
        tool_executor=GovernedToolInvoker(policy=PolicyEngine(rules=())),
        model_router=ModelGateway(ModelResolver(registry=registry)),
    )

    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
    )

    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_total_tokens=1000),
        agents=(AgentRef("coord"), AgentRef("worker-a")),
        coordinator=AgentRef("coord"),
        config={"task_count": 2},
    )
    agents = {
        "coord": _agent_spec("coord", "model-0"),
        "worker-a": _agent_spec("worker-a", "model-1"),
    }
    context = _driving_context("drive-ok", "shared-session")

    async def _run():
        return await runner.run(
            spec, RunInput(prompt="do the work"), context, agents=agents
        )

    result = asyncio.run(_run())

    # 2 tasks * (50 + 50) = 200 tokens accumulated, under the 1000 cap.
    assert result.token_usage.get("input_tokens") == 100
    assert result.token_usage.get("output_tokens") == 100
    driving = asyncio.run(stores.run_store.get(context.run_id))
    assert driving.status is RunStatus.SUCCEEDED


# --- 7. End-to-end active_run_id decoupling (real FilesystemSwarmStore) ---


def test_run_decouples_task_id_from_child_run_id_via_active_run_id(tmp_path):
    """End-to-end through the real FilesystemSwarmStore: after runner.run() completes,
    every SUCCEEDED task has active_run_id set, DIFFERENT from task.id, and
    matching a real child RunRecord id. This is the invariant the
    design note contract mandates (禁止 SwarmTask.id == child_run_id)."""
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("coord-out", "alpha-out", "beta-out")
    stores = _Stores(tmp_path)
    stores.seed_shared_session("shared-session")

    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
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
    context = _driving_context("drive-5a", "shared-session")

    async def _run():
        return await runner.run(
            spec, RunInput(prompt="do the work"), context, agents=agents
        )

    asyncio.run(_run())

    async def _verify():
        # discover the swarm_run_id by listing the run dir (only one exists).
        runs = []
        for p in (tmp_path / "swarm" / "runs").glob("*.json"):
            runs.append(p.stem)
        swarm_run_id = runs[0]
        tasks = await stores.swarm_store.list_tasks(swarm_run_id)
        children = await stores.run_store.list_children(context.run_id)
        return tasks, children

    tasks, children = asyncio.run(_verify())

    child_ids = {c.id for c in children}
    assert len(tasks) == 2
    for t in tasks:
        # the invariant: task.id != child run id; active_run_id is the handle.
        assert t.active_run_id is not None, f"task {t.id} has no active_run_id"
        assert t.active_run_id != t.id
        assert t.active_run_id in child_ids, (
            f"task {t.id} active_run_id {t.active_run_id} not in child runs {child_ids}"
        )


# --- recover() contract -------------------------------------------------------


def _seed_recover_state(
    stores: _Stores,
    *,
    task_id: str,
    child_run_id: "str | None",
    child_status: "RunStatus | None" = None,
    child_result: "RunResult | None" = None,
    child_error: "RunErrorInfo | None" = None,
    lease_expires_at: "datetime | None" = None,
) -> None:
    """Seed a SwarmRun + a CLAIMED SwarmTask (+ optionally a child RunRecord)
    for recover() to walk. ``lease_expires_at`` defaults to the past so the
    task is observable as expired -- pass a future datetime to leave it alone."""
    now = _NOW
    expired = lease_expires_at or (now - timedelta(seconds=60))

    async def _seed():
        # driving RunRecord (the swarm run's parent).
        await stores.run_store.create(
            RunRecord(
                id="drive-run-1",
                root_run_id="drive-run-1",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=RunStatus.RUNNING,
                input=RunInput(prompt="x"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=None,
            )
        )
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm-1",
                run_id="drive-run-1",
                round=1,
                status=SwarmStatus.RUNNING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )
        if child_run_id is not None:
            await stores.run_store.create(
                RunRecord(
                    id=child_run_id,
                    root_run_id="drive-run-1",
                    parent_run_id="drive-run-1",
                    session_id=f"swarm:swarm-1:{task_id}",
                    runnable_id="worker-a",
                    runnable_type=RunnableType.AGENT,
                    status=child_status or RunStatus.SUCCEEDED,
                    input=RunInput(prompt="sub"),
                    result=child_result,
                    error=child_error,
                    version=1,
                    created_at=now,
                    started_at=now,
                    finished_at=now,
                )
            )
        await stores.swarm_store.create_task(
            SwarmTask(
                id=task_id,
                swarm_run_id="swarm-1",
                parent_task_id=None,
                assigned_agent_id="worker-a",
                description="x",
                status=SwarmTaskStatus.CLAIMED,
                dependencies=(),
                input=TaskInput(prompt="sub"),
                result=None,
                error=None,
                attempts=1,
                version=1,
                claimed_at=now,
                lease_expires_at=expired,
                created_at=now,
                updated_at=now,
                active_run_id=child_run_id,
            )
        )

    asyncio.run(_seed())


def _make_runner(stores: _Stores):
    from linktools.ai.swarm.runner import SwarmRunner

    return SwarmRunner(
        swarm_store=stores.swarm_store,
        run_definitions=stores.run_definitions,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=_build_compiler("ok"),
    )


def test_recover_completes_task_when_child_run_succeeded(tmp_path):
    """contract: a CLAIMED task whose lease lapsed but whose child Run already
    SUCCEEDED is reconciled to SUCCEEDED (the strategy crashed between the
    child's SUCCEEDED transition and its complete_task call)."""
    stores = _Stores(tmp_path)
    child_result = RunResult(output="done", token_usage={}, metadata={})
    _seed_recover_state(
        stores,
        task_id="task-1",
        child_run_id="child-1",
        child_status=RunStatus.SUCCEEDED,
        child_result=child_result,
    )
    runner = _make_runner(stores)

    async def _recover():
        await runner.recover("swarm-1")

    asyncio.run(_recover())

    async def _verify():
        return await stores.swarm_store.list_tasks("swarm-1")

    tasks = asyncio.run(_verify())
    assert tasks[0].status is SwarmTaskStatus.SUCCEEDED
    assert tasks[0].result == child_result


def test_recover_fails_task_when_child_run_failed(tmp_path):
    """contract: a CLAIMED task whose child Run already FAILED is reconciled to
    FAILED (carrying the child Run's error forward)."""
    stores = _Stores(tmp_path)
    child_error = RunErrorInfo(error_type="ValueError", message="boom", detail={})
    _seed_recover_state(
        stores,
        task_id="task-1",
        child_run_id="child-1",
        child_status=RunStatus.FAILED,
        child_error=child_error,
    )
    runner = _make_runner(stores)

    async def _recover():
        await runner.recover("swarm-1")

    asyncio.run(_recover())

    async def _verify():
        return await stores.swarm_store.list_tasks("swarm-1")

    tasks = asyncio.run(_verify())
    assert tasks[0].status is SwarmTaskStatus.FAILED
    assert tasks[0].error == child_error


def test_recover_leaves_task_alone_when_child_run_still_running(tmp_path):
    """contract guard: even though the lease has lapsed, the child Run is still
    RUNNING -- the worker may yet finish. Leave the task CLAIMED (don't blindly
    re-run a side-effecting task)."""
    stores = _Stores(tmp_path)
    _seed_recover_state(
        stores,
        task_id="task-1",
        child_run_id="child-1",
        child_status=RunStatus.RUNNING,
    )
    runner = _make_runner(stores)

    async def _recover():
        await runner.recover("swarm-1")

    asyncio.run(_recover())

    async def _verify():
        return await stores.swarm_store.list_tasks("swarm-1")

    tasks = asyncio.run(_verify())
    assert tasks[0].status is SwarmTaskStatus.CLAIMED


def test_recover_skips_task_with_unexpired_lease(tmp_path):
    """A task whose lease is still live is presumed being worked -- recover()
    must not touch it (contract: don't blindly re-run)."""
    from datetime import timedelta

    stores = _Stores(tmp_path)
    future = _NOW + timedelta(seconds=300)
    _seed_recover_state(
        stores,
        task_id="task-1",
        child_run_id="child-1",
        child_status=RunStatus.SUCCEEDED,
        child_result=RunResult(output="done", token_usage={}, metadata={}),
        lease_expires_at=future,
    )
    runner = _make_runner(stores)

    async def _recover():
        await runner.recover("swarm-1")

    asyncio.run(_recover())

    async def _verify():
        return await stores.swarm_store.list_tasks("swarm-1")

    tasks = asyncio.run(_verify())
    # Lease not expired -> recover skipped it; task is still CLAIMED.
    assert tasks[0].status is SwarmTaskStatus.CLAIMED


def test_recover_unknown_swarm_run_raises(tmp_path):
    from linktools.ai.errors import SwarmRunNotFoundError

    stores = _Stores(tmp_path)
    runner = _make_runner(stores)

    async def _recover():
        await runner.recover("nope")

    with pytest.raises(SwarmRunNotFoundError):
        asyncio.run(_recover())


def test_recover_requeues_task_with_no_run_to_pending_on_sqlalchemy_backend(tmp_path):
    """On SqlAlchemySwarmStore a CLAIMED+expired task whose active_run_id is
    None gets re-queued to PENDING by the trailing reclaim_expired_tasks call
    (the strategy crashed between claim_task and set_active_run). FilesystemSwarmStore
    documents this as a no-op (single-process: nothing to reclaim at rest)."""
    from contextlib import asynccontextmanager
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.swarm import SqlAlchemySwarmStore
    from linktools.ai.swarm.runner import SwarmRunner

    now = _NOW
    expired = now - timedelta(seconds=60)

    @asynccontextmanager
    async def _swarm_store():
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/recover.db")
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            yield SqlAlchemySwarmStore(session_factory=session_factory)
        finally:
            await engine.dispose()

    async def _scenario():
        async with _swarm_store() as swarm_store:
            await swarm_store.create_run(
                SwarmRun(
                    id="swarm-1",
                    run_id="drive-run-1",
                    round=1,
                    status=SwarmStatus.RUNNING,
                    version=1,
                    token_usage=TokenUsage(),
                    cost=Decimal("0"),
                    created_at=now,
                    updated_at=now,
                )
            )
            await swarm_store.create_task(
                SwarmTask(
                    id="task-1",
                    swarm_run_id="swarm-1",
                    parent_task_id=None,
                    assigned_agent_id="worker-a",
                    description="x",
                    status=SwarmTaskStatus.CLAIMED,
                    dependencies=(),
                    input=TaskInput(prompt="sub"),
                    result=None,
                    error=None,
                    attempts=1,
                    version=1,
                    claimed_at=now,
                    lease_expires_at=expired,
                    created_at=now,
                    updated_at=now,
                    active_run_id=None,
                )
            )
            # File-backed RunStore is fine here: recover() only reads via
            # run_store.get(), and there's no child RunRecord to look up
            # (active_run_id is None -> the recover loop skips run_store.get).
            runner = SwarmRunner(
                swarm_store=swarm_store,
                run_store=stores.run_store,
                session_store=stores.session_store,
                event_store=stores.event_store,
                dispatcher=stores.agent_runner,
                compiler=_build_compiler("ok"),
                run_definitions=stores.run_definitions,
            )
            await runner.recover("swarm-1")
            return await swarm_store.list_tasks("swarm-1")

    stores = _Stores(tmp_path)
    tasks = asyncio.run(_scenario())
    assert tasks[0].status is SwarmTaskStatus.PENDING
    assert tasks[0].active_run_id is None
    assert tasks[0].assigned_agent_id is None


def test_resume_refused_for_terminal_swarm(tmp_path):
    """a terminal swarm (SUCCEEDED/FAILED/CANCELLED) must not resume
    -- re-running its strategy could repeat side-effecting tasks."""
    from linktools.ai.swarm.runner import SwarmRunner
    from linktools.ai.errors import InvalidRunTransitionError

    compiler = _build_compiler("term-out")
    stores = _Stores(tmp_path)
    now = _NOW

    async def _seed(status):
        await stores.run_store.create(
            RunRecord(
                id="drive-term",
                root_run_id="drive-term",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=RunStatus.SUCCEEDED,
                input=RunInput(prompt="done"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm-term",
                run_id="drive-term",
                round=0,
                status=status,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )

    for terminal in (SwarmStatus.SUCCEEDED, SwarmStatus.FAILED, SwarmStatus.CANCELLED):
        asyncio.run(_seed(terminal))
        runner = SwarmRunner(
            swarm_store=stores.swarm_store,
            run_store=stores.run_store,
            session_store=stores.session_store,
            event_store=stores.event_store,
            compiler=compiler,
            dispatcher=stores.agent_runner,
            run_controller=stores.run_controller,
            run_definitions=stores.run_definitions,
        )
        with pytest.raises(InvalidRunTransitionError):
            asyncio.run(runner.resume("swarm-term"))


@pytest.mark.parametrize(
    "driving_status",
    [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED],
    ids=["succeeded", "failed", "cancelled"],
)
def test_resume_refused_for_terminal_driving_run(tmp_path, driving_status):
    """v4 : a PAUSED (non-terminal) swarm whose DRIVING Run is already
    terminal must not resume -- re-entering the strategy could re-drive worker
    side effects. The driving Run is rejected before the snapshot is loaded or
    the strategy resumed, so nothing executes.

    RUNNING is intentionally NOT rejected: a RECOVERABLE swarm's driving Run is
    legitimately RUNNING mid-flight, and crash-recovery resume must keep working
    (see test_resume_after_partial_failure_completes). The guide's table
    lists RunStatus.RECOVERABLE, which does not exist in this codebase; RUNNING
    is the actual non-terminal recoverable driving state, so only the terminal
    states are rejected (per the guide's own note to map names to the project's
    actual statuses)."""
    from linktools.ai.errors import InvalidRunTransitionError
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("term-out")
    stores = _Stores(tmp_path)
    now = _NOW

    async def _seed():
        await stores.run_store.create(
            RunRecord(
                id="drive-term-d",
                root_run_id="drive-term-d",
                parent_run_id=None,
                session_id="shared",
                runnable_id="swarm-spec-1",
                runnable_type=RunnableType.SWARM,
                status=driving_status,
                input=RunInput(prompt="done"),
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=now,
                finished_at=now,
            )
        )
        # Non-terminal swarm -> passes the swarm-status check, so the driving
        # check is what rejects the resume.
        await stores.swarm_store.create_run(
            SwarmRun(
                id="swarm-term-d",
                run_id="drive-term-d",
                round=0,
                status=SwarmStatus.PAUSED,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
        )

    asyncio.run(_seed())
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        dispatcher=stores.agent_runner,
        compiler=compiler,
        run_controller=stores.run_controller,
        run_definitions=stores.run_definitions,
    )
    with pytest.raises(InvalidRunTransitionError):
        asyncio.run(runner.resume("swarm-term-d"))

    async def _verify():
        driving = await stores.run_store.get("drive-term-d")
        swarm = await stores.swarm_store.get_run("swarm-term-d")
        tasks = await stores.swarm_store.list_tasks("swarm-term-d")
        return driving, swarm, tasks

    driving, swarm, tasks = asyncio.run(_verify())
    # strategy.resume never ran: the driving Run stays terminal, the swarm stays
    # PAUSED, and no worker task was created.
    assert driving.status is driving_status
    assert swarm.status is SwarmStatus.PAUSED
    assert tasks == ()


def test_swarm_snapshot_failure_leaves_no_orphan_running(tmp_path):
    """when the run-definition snapshot cannot be serialized, the swarm
    run must fail BEFORE any Run/SwarmRun is created -- no orphan RUNNING. A set
    in the strategy config is rejected by canonical_json."""
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("snap-out")
    stores = _Stores(tmp_path)
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        compiler=compiler,
        dispatcher=stores.agent_runner,
        run_controller=stores.run_controller,
        run_definitions=stores.run_definitions,
    )
    spec = _spec(
        kind="parallel_fan_out",
        agents=(AgentRef("coord"),),
        coordinator=AgentRef("coord"),
    )
    agents = {"coord": _agent_spec("coord", "model-0")}
    context = _driving_context("drive-snap", "shared-session")

    # Force the snapshot store write to fail ( : Snapshot Store write
    # failure). The swarm run must fail BEFORE any Run/SwarmRun is created.
    async def _failing_create(_snapshot):
        raise RuntimeError("injected snapshot store failure")

    stores.run_definitions.create = _failing_create

    async def _run():
        await runner.run(spec, RunInput(prompt="do the work"), context, agents=agents)

    with pytest.raises(RuntimeError, match="injected snapshot store failure"):
        asyncio.run(_run())
    # No Run record was created (snapshot is prepared before any state).
    assert asyncio.run(stores.run_store.get("drive-snap")) is None
    # No SwarmRun either (the swarm runs subdir has no entry for this run).
    swarms_dir = (
        stores.swarm_store._root if hasattr(stores.swarm_store, "_root") else None
    )
    if swarms_dir is not None and swarms_dir.exists():
        assert not any("drive-snap" in f.name for f in swarms_dir.iterdir()), list(
            swarms_dir.iterdir()
        )


def test_swarm_worker_runs_have_resumable_snapshots(tmp_path):
    """a swarm worker Run (child of the driving swarm run) must persist a
    RunDefinitionSnapshot so Runtime.resume(child_run_id) can restore its spec
    if a worker tool pauses on approval."""
    from linktools.ai.swarm.runner import SwarmRunner

    compiler = _build_compiler("alpha-out", "beta-out")
    stores = _Stores(tmp_path)
    runner = SwarmRunner(
        swarm_store=stores.swarm_store,
        run_store=stores.run_store,
        session_store=stores.session_store,
        event_store=stores.event_store,
        compiler=compiler,
        dispatcher=stores.agent_runner,
        run_controller=stores.run_controller,
        run_definitions=stores.run_definitions,
    )
    spec = _spec(
        kind="parallel_fan_out",
        limits=_limits(max_concurrency=4),
        agents=(AgentRef("worker-a"), AgentRef("worker-b")),
        coordinator=AgentRef("worker-a"),
        config={"task_count": 2},
    )
    agents = {
        "worker-a": _agent_spec("worker-a", "model-0"),
        "worker-b": _agent_spec("worker-b", "model-1"),
    }
    context = _driving_context("drive-snap2", "shared-session")

    async def _run():
        return await runner.run(
            spec, RunInput(prompt="do the work"), context, agents=agents
        )

    asyncio.run(_run())

    async def _verify():
        children = await stores.run_store.list_children(context.run_id)
        snapshots = []
        for child in children:
            snapshots.append(await stores.run_definitions.get(child.id))
        return children, snapshots

    children, snapshots = asyncio.run(_verify())
    assert children, "expected at least one worker child run"
    # Every worker child run has a resumable snapshot.
    assert all(s is not None for s in snapshots), snapshots
    assert all(s.runnable_type == "agent" for s in snapshots)
