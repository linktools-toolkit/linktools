#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmStrategy: the PROGRAMMATIC orchestration layer (spec 22.4). A strategy
consumes a SwarmExecutionContext and produces one aggregated RunResult.

Decision #1 (PROGRAMMATIC): strategies never call a real model to DECIDE what to
do. They accept an injectable async ``coordinator_fn`` (CoordinatorDelegation)
or ``task_factory`` (ParallelFanOut). Tests inject deterministic functions and
use FunctionModel workers -- no real model calls.

Workers run against per-task SCRATCH sessions (spec invariant: only the final
aggregate is written to the shared/parent session). ``_run_task`` builds a CHILD
RunContext whose session_id is ``f"swarm:{swarm_run.id}:{task.id}"`` and creates
that SessionRecord before invoking AgentRunner.run.

claim_task is a WORK-QUEUE api: ``claim_task(swarm_run_id, agent_id)`` returns
the oldest PENDING task for that (run, agent) pair -- it does NOT take a task_id.
In the serial coordinator path the claimed task is the one just created; in the
parallel fan-out path several coroutines may compete for tasks of the same agent,
so ``_run_task`` processes the CLAIMED task (not the passed-in ``task``) and uses
its id for the scratch session, child run_id, and complete/fail calls. This keeps
every status transition consistent (PENDING -> CLAIMED -> SUCCEEDED|FAILED)."""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Protocol, Tuple, runtime_checkable

from ..agent.compiler import AgentCompiler
from ..agent.models import CompiledAgent
from ..agent.runner import AgentRunner
from ..errors import SwarmError, SwarmLimitExceededError
from ..run.context import RunContext
from ..run.models import RunErrorInfo, RunInput, RunResult, RunnableType
from ..session.models import SessionRecord, SessionStatus
from .aggregation import aggregate
from .limits import SwarmLimits
from .models import SwarmRun, SwarmTask, SwarmTaskStatus, TaskInput
from .spec import SwarmSpec, SwarmStrategySpec
from .store import SwarmStore


# --- coordinator_fn signature ------------------------------------------------

CoordinatorFn = Callable[
    [SwarmRun, "Tuple[SwarmTask, ...]", SwarmLimits],
    "Awaitable[Tuple[TaskInput, ...]]",
]


TaskFactory = Callable[[RunInput], "Tuple[TaskInput, ...]"]


# --- SwarmExecutionContext ---------------------------------------------------

@dataclass(frozen=True, slots=True)
class SwarmExecutionContext:
    """The bundle a strategy consumes: the spec, the driving swarm run, the
    request that initiated it, the parent RunContext (shared session lives
    here), the runner/compiler, the pre-compiled worker agents keyed by
    AgentRef.agent_id, and the four stores."""

    spec: SwarmSpec
    swarm_run: SwarmRun
    request: RunInput
    parent_context: RunContext
    agent_runner: AgentRunner
    compiler: AgentCompiler
    agents: "Mapping[str, CompiledAgent]"
    swarm_store: SwarmStore
    run_store: Any   # RunStore Protocol (typed loosely to avoid a cycle with run.store)
    session_store: Any  # SessionStore Protocol
    event_store: Any  # EventStore Protocol


# --- SwarmStrategy Protocol + registry --------------------------------------

@runtime_checkable
class SwarmStrategy(Protocol):
    async def run(self, ctx: SwarmExecutionContext) -> RunResult: ...


_STRATEGY_REGISTRY: "dict[str, type]" = {}


def register_strategy(kind: str) -> "Callable[[type], type]":
    """Class decorator: register ``kind`` -> the decorated strategy class."""
    def _decorator(cls: type) -> type:
        _STRATEGY_REGISTRY[kind] = cls
        return cls
    return _decorator


def build_strategy(spec: SwarmStrategySpec) -> "SwarmStrategy":
    """Construct a strategy from a SwarmStrategySpec. ``spec.config`` is spread
    as keyword args into the strategy class constructor."""
    try:
        cls = _STRATEGY_REGISTRY[spec.kind]
    except KeyError:
        raise SwarmError(f"unknown strategy kind: {spec.kind}") from None
    return cls(**dict(spec.config))


# --- shared worker-pool helper ----------------------------------------------

def _worker_pool(ctx: SwarmExecutionContext) -> "tuple[str, ...]":
    """Agent ids eligible for task assignment = spec.agents minus the
    coordinator. Falls back to all agents when the coordinator is the only
    member (coordinator-as-worker)."""
    workers = tuple(
        a.agent_id for a in ctx.spec.agents
        if a.agent_id != ctx.spec.coordinator.agent_id
    )
    if not workers:
        workers = tuple(a.agent_id for a in ctx.spec.agents)
    return workers


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _run_task(ctx: SwarmExecutionContext, task: SwarmTask, *, max_task_retries: int = 0) -> "RunResult | None":
    """Run a single SwarmTask against its assigned worker agent.

    Sequence:
      1. resolve the compiled worker.
      2. claim a pending task for (swarm_run, agent) via the store's work-queue
         api -- in the serial path this is ``task``; in the parallel path it may
         be a sibling task created by the same fan-out. We process whatever the
         store hands us so the PENDING->CLAIMED flip is consistent.
      3. build a per-task SCRATCH session and create it (workers must not touch
         the shared/parent session).
      4. build a CHILD RunContext parented to the swarm's driving run.
      5. retry loop: on exception retry up to ``max_task_retries`` extra times;
         on final failure mark the task FAILED and return None (catch-and-
         continue so a coordinator round completes even if a worker errors).

    Returns the worker's RunResult on success, or None if there was nothing to
    claim or every attempt failed.
    """
    compiled = ctx.agents[task.assigned_agent_id]
    claimed = await ctx.swarm_store.claim_task(ctx.swarm_run.id, task.assigned_agent_id)
    if claimed is None:
        return None

    scratch_session_id = f"swarm:{ctx.swarm_run.id}:{claimed.id}"
    now = _now()
    await ctx.session_store.create(SessionRecord(
        id=scratch_session_id, parent_id=None, status=SessionStatus.ACTIVE,
        version=1, created_at=now, updated_at=now,
    ))

    child_context = RunContext(
        run_id=claimed.id,
        root_run_id=ctx.parent_context.root_run_id,
        parent_run_id=ctx.swarm_run.run_id,
        session_id=scratch_session_id,
        runnable_id=claimed.assigned_agent_id,
        runnable_type=RunnableType.AGENT,
        user_id=ctx.parent_context.user_id,
        tenant_id=ctx.parent_context.tenant_id,
        workspace=None,
    )

    last_exc: "BaseException | None" = None
    for _attempt in range(max_task_retries + 1):
        try:
            result = await ctx.agent_runner.run(
                compiled, RunInput(prompt=claimed.input.prompt), child_context,
            )
            await ctx.swarm_store.complete_task(claimed.id, result)
            return result
        except Exception as exc:
            last_exc = exc

    await ctx.swarm_store.fail_task(claimed.id, RunErrorInfo(
        error_type=type(last_exc).__name__ if last_exc is not None else "Unknown",
        message=str(last_exc) if last_exc is not None else "",
    ))
    return None


def _make_task(ctx: SwarmExecutionContext, ti: TaskInput, agent_id: str) -> SwarmTask:
    now = _now()
    return SwarmTask(
        id=str(uuid.uuid4()),
        swarm_run_id=ctx.swarm_run.id,
        parent_task_id=None,
        assigned_agent_id=agent_id,
        description=ti.prompt[:80],
        status=SwarmTaskStatus.PENDING,
        dependencies=(),
        input=ti,
        result=None,
        error=None,
        attempts=0,
        version=1,
        claimed_at=None,
        lease_expires_at=None,
        created_at=now,
        updated_at=now,
    )


# --- CoordinatorDelegationStrategy ------------------------------------------

@register_strategy("coordinator_delegation")
class CoordinatorDelegationStrategy:
    """Round-based strategy: an injectable ``coordinator_fn`` decides, each
    round, what TaskInputs to create based on the swarm run and the tuple of
    SUCCEEDED tasks so far. Tasks are created and run serially within each
    round (round-robin across the worker pool).

    Limit enforcement:
      * ``max_rounds``    -- a coordinator that keeps producing tasks past this
                              many rounds raises SwarmLimitExceededError(kind=...).
      * ``max_delegations``-- one delegation == one round that produced work.
      * ``max_tasks``      -- cumulative created-task count across all rounds.
    """

    def __init__(
        self,
        *,
        coordinator_fn: "CoordinatorFn | None" = None,
        max_task_retries: int = 1,
    ) -> None:
        self._coordinator_fn = coordinator_fn
        self._max_task_retries = max_task_retries

    def _resolve_coordinator(self, ctx: SwarmExecutionContext) -> "CoordinatorFn":
        if self._coordinator_fn is not None:
            return self._coordinator_fn
        # default: one task carrying ctx.request.prompt on the first round,
        # nothing thereafter. The request prompt is captured via closure.
        request_prompt = ctx.request.prompt
        seen: "list[bool]" = [False]

        async def _default(swarm_run: SwarmRun, completed: "tuple[SwarmTask, ...]", limits: SwarmLimits) -> "tuple[TaskInput, ...]":
            if not seen[0] and not completed:
                seen[0] = True
                return (TaskInput(prompt=request_prompt),)
            return ()
        return _default

    async def run(self, ctx: SwarmExecutionContext) -> RunResult:
        # validate the coordinator agent is in the compiled agents mapping; the
        # default coordinator_fn handles the simple programmatic case.
        _ = ctx.agents[ctx.spec.coordinator.agent_id]
        coordinator_fn = self._resolve_coordinator(ctx)
        workers = _worker_pool(ctx)

        created_count = 0
        delegations = 0
        round_num = 0
        while True:
            completed = await ctx.swarm_store.list_tasks(
                ctx.swarm_run.id, status=SwarmTaskStatus.SUCCEEDED,
            )
            new_inputs = await coordinator_fn(ctx.swarm_run, completed, ctx.spec.limits)
            if not new_inputs:
                break
            round_num += 1
            if round_num > ctx.spec.limits.max_rounds:
                raise SwarmLimitExceededError(
                    f"coordinator exceeded max_rounds={ctx.spec.limits.max_rounds}",
                    kind="max_rounds",
                )
            delegations += 1
            if delegations > ctx.spec.limits.max_delegations:
                raise SwarmLimitExceededError(
                    f"coordinator exceeded max_delegations={ctx.spec.limits.max_delegations}",
                    kind="max_delegations",
                )
            if created_count + len(new_inputs) > ctx.spec.limits.max_tasks:
                raise SwarmLimitExceededError(
                    f"coordinator exceeded max_tasks={ctx.spec.limits.max_tasks}",
                    kind="max_tasks",
                )
            for ti in new_inputs:
                agent_id = workers[created_count % len(workers)]
                task = _make_task(ctx, ti, agent_id)
                await ctx.swarm_store.create_task(task)
                created_count += 1
                await _run_task(ctx, task, max_task_retries=self._max_task_retries)

        all_tasks = await ctx.swarm_store.list_tasks(ctx.swarm_run.id)
        return aggregate(ctx.spec.aggregation, all_tasks)


# --- ParallelFanOutStrategy -------------------------------------------------

@register_strategy("parallel_fan_out")
class ParallelFanOutStrategy:
    """Create N tasks up front (N = task_count, len(spec.agents), or via
    task_factory) and run them CONCURRENTLY bounded by an
    ``asyncio.Semaphore(spec.limits.max_concurrency)``. No rounds, no
    dependencies. Round-robin across the worker pool for assignment."""

    def __init__(
        self,
        *,
        task_count: "int | None" = None,
        task_factory: "TaskFactory | None" = None,
        max_task_retries: int = 0,
    ) -> None:
        self._task_count = task_count
        self._task_factory = task_factory
        self._max_task_retries = max_task_retries

    def _build_inputs(self, ctx: SwarmExecutionContext) -> "tuple[TaskInput, ...]":
        if self._task_factory is not None:
            return tuple(self._task_factory(ctx.request))
        n = self._task_count if self._task_count is not None else len(ctx.spec.agents)
        return tuple(TaskInput(prompt=ctx.request.prompt) for _ in range(n))

    async def run(self, ctx: SwarmExecutionContext) -> RunResult:
        inputs = self._build_inputs(ctx)
        workers = _worker_pool(ctx)

        # create all tasks up front, round-robin across the worker pool.
        tasks: "list[SwarmTask]" = []
        for i, ti in enumerate(inputs):
            agent_id = workers[i % len(workers)]
            task = _make_task(ctx, ti, agent_id)
            await ctx.swarm_store.create_task(task)
            tasks.append(task)

        # bounded concurrency: at most max_concurrency worker runs in flight.
        max_concurrency = max(1, ctx.spec.limits.max_concurrency)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _run_one(t: SwarmTask) -> None:
            async with semaphore:
                await _run_task(ctx, t, max_task_retries=self._max_task_retries)

        await asyncio.gather(*(_run_one(t) for t in tasks))

        all_tasks = await ctx.swarm_store.list_tasks(ctx.swarm_run.id)
        return aggregate(ctx.spec.aggregation, all_tasks)
