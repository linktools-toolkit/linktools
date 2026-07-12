#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmStrategy: the PROGRAMMATIC orchestration layer . A strategy
consumes a SwarmExecutionContext and produces one aggregated RunResult.

: strategies never call a real model to DECIDE what to
do. They accept an injectable async ``coordinator_fn`` (CoordinatorDelegation)
or ``task_factory`` (ParallelFanOut). Tests inject deterministic functions and
use FunctionModel workers -- no real model calls.

Workers run against per-task SCRATCH sessions (spec invariant: only the final
aggregate is written to the shared/parent session). ``_run_task`` builds a CHILD
RunContext whose session_id is ``f"swarm:{swarm_run.id}:{task.id}:{child_run_id}"``
and creates that SessionRecord before invoking AgentRunner.run. The
``child_run_id`` suffix keeps each retry attempt's scratch session distinct
(see the retry identity note below) so a failed attempt's partial conversation
can never leak into the next attempt's prompt.

The child RunRecord's id is NOT the task's id. ``_run_task``
mints a fresh ``str(uuid.uuid4())`` run_id per execution and stores it on the
task via ``SwarmStore.set_active_run`` -- so ``task.active_run_id`` is the
handle SwarmRunner.cancel uses to find the in-flight child Run. On retry the
same task gets a NEW run_id (active_run_id is overwritten), which is the
decoupling mandated for correctness.

claim_task is a WORK-QUEUE api: ``claim_task(swarm_run_id, agent_id)`` returns
the oldest PENDING task for that (run, agent) pair -- it does NOT take a task_id.
In the serial coordinator path the claimed task is the one just created; in the
parallel fan-out path several coroutines may compete for tasks of the same agent,
so ``_run_task`` processes the CLAIMED task (not the passed-in ``task``) and uses
its id for the scratch session, complete/fail calls, and the active-run lookup.
This keeps every status transition consistent (PENDING -> CLAIMED -> SUCCEEDED|FAILED).

Retry vs fencing conflict: only an exception from the worker call itself
(``ctx.agent_runner.run``) counts as an attempt failure that can trigger a
retry / eventual ``fail_task()``. ``set_active_run``/``complete_task``/
``record_attempt`` can also raise, but those are persistence fencing
conflicts -- another caller already reclaimed (lease expiry) or cancelled
this task -- not evidence the worker failed. ``_run_task`` logs those and
returns ``None`` without retrying or re-failing a task whose real outcome
belongs to someone else."""

import asyncio
import logging
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Protocol, Tuple, runtime_checkable

from ..agent.compiler import AgentCompiler
from ..agent.models import CompiledAgent
from ..agent.runner import AgentRunner
from ..errors import SwarmConflictError, SwarmError, SwarmLimitExceededError
from ..run.context import RunContext
from ..run.models import RunErrorInfo, RunInput, RunResult, RunnableType
from ..security.redact import redact_exception
from ..session.models import SessionRecord, SessionStatus
from .aggregation import aggregate
from .limits import SwarmLimits
from .models import (
    AttemptStatus,
    SwarmRun,
    SwarmTask,
    SwarmTaskAttempt,
    SwarmTaskStatus,
    TaskInput,
)
from .spec import SwarmSpec, SwarmStrategySpec
from .store import SwarmStore


_LOGGER = logging.getLogger(__name__)


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
    run_store: Any  # RunStore Protocol (typed loosely to avoid a cycle with run.store)
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
        a.agent_id
        for a in ctx.spec.agents
        if a.agent_id != ctx.spec.coordinator.agent_id
    )
    if not workers:
        workers = tuple(a.agent_id for a in ctx.spec.agents)
    return workers


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _compute_depth(ctx: SwarmExecutionContext, task: SwarmTask) -> int:
    """Walk ``task.parent_task_id`` up the ancestor chain to compute this task's
    depth. A top-level task (``parent_task_id is None``) is depth 1; each
    ancestor adds one. Guards against malformed cycles by capping the walk at a
    parent that is missing or already seen.

    Used by  max_depth enforcement. The current built-in
    strategies (``_make_task``) always set ``parent_task_id=None``, so every
    programmatically-created task is depth 1; this guard only fires for nested
    delegations (coordinator chains, future hierarchical strategies).
    """
    if task.parent_task_id is None:
        return 1
    # list_tasks is the only store api that returns tasks by id; one query feeds
    # the whole walk (the ancestor set is immutable for the duration of a single
    # _run_task call). A dedicated get_task is a separate concern.
    ancestors = await ctx.swarm_store.list_tasks(ctx.swarm_run.id)
    by_id: "dict[str, SwarmTask]" = {t.id: t for t in ancestors}
    depth = 1
    current_id = task.parent_task_id
    seen: "set[str]" = set()
    while current_id is not None and current_id not in seen:
        seen.add(current_id)
        parent = by_id.get(current_id)
        if parent is None:
            break
        depth += 1
        current_id = parent.parent_task_id
    return depth


async def _retry_fencing_conflict_once(
    ctx: SwarmExecutionContext,
    claimed: SwarmTask,
    *,
    still_owned: "Callable[[SwarmTask], bool]",
    op: "Callable[[int], Awaitable[SwarmTask]]",
) -> "SwarmTask | None":
    """Run a fenced SwarmTask write (``op(expected_version)`` -- a partial of
    complete_task/fail_task with the version filled in) and classify a
    ``SwarmConflictError`` instead of treating every conflict the same:

    * Re-read the task. If ``still_owned(fresh)`` says this call still holds
      it, the conflict was just a stale ``expected_version`` (e.g. a lease
      renewal bumped it while the worker was in flight) -- retry ``op()``
      ONCE with the fresh version. This is a retry of the cheap persistence
      write, never of the worker itself.
    * If ownership has genuinely moved on (reclaimed after lease expiry, or
      cancelled) -- or the retry with the fresh version ALSO conflicts --
      return None so the caller discards this attempt without mistaking the
      conflict for a worker failure.

    Any exception that is not a SwarmConflictError (e.g. SwarmTaskNotFoundError,
    a raw storage error) is not a fencing conflict this helper knows how to
    interpret and is left to propagate to the caller.

    NOT used for set_active_run: ``still_owned`` needs an owner token this
    same attempt uniquely holds (complete_task/fail_task use
    ``active_run_id == child_run_id``, which only THIS attempt could have
    set). set_active_run is what ASSIGNS that token in the first place, so
    at that point there is no owner token yet to distinguish "still this
    caller" from "a different worker also claimed it" -- both look
    identical (status CLAIMED) on a re-read. See _run_task's set_active_run
    call site, which discards on any conflict instead of retrying."""
    try:
        return await op(claimed.version)
    except asyncio.CancelledError:
        raise
    except SwarmConflictError as exc:
        current = await ctx.swarm_store.list_tasks(ctx.swarm_run.id)
        fresh = next((t for t in current if t.id == claimed.id), None)
        if fresh is None or not still_owned(fresh):
            _LOGGER.warning(
                "swarm task %s lost a fencing race (reclaimed or "
                "cancelled) -- discarding: %s",
                claimed.id,
                exc,
            )
            return None
        try:
            return await op(fresh.version)
        except asyncio.CancelledError:
            raise
        except SwarmConflictError as retry_exc:
            _LOGGER.warning(
                "swarm task %s lost a fencing race on retry with the "
                "fresh version -- discarding: %s",
                claimed.id,
                retry_exc,
            )
            return None


async def _run_task(
    ctx: SwarmExecutionContext, task: SwarmTask, *, max_task_retries: int = 0
) -> "RunResult | None":
    """Run a single SwarmTask against its assigned worker agent.

    Sequence:
      0. depth guard: walk ``parent_task_id`` chain; raise
         SwarmLimitExceededError(kind="max_depth") before claiming/dispatching.
      1. resolve the compiled worker.
      2. claim a pending task for (swarm_run, agent) via the store's work-queue
         api -- in the serial path this is ``task``; in the parallel path it may
         be a sibling task created by the same fan-out. We process whatever the
         store hands us so the PENDING->CLAIMED flip is consistent.
      3. mint a FRESH child run_id (task.id != run_id) and
         record it on the task via ``set_active_run`` so cancel() can find the
         child RunRecord later. On retry this same method is re-invoked, so a
         new run_id is minted and active_run_id is overwritten.
      4. build a per-task SCRATCH session and create it (workers must not touch
         the shared/parent session).
      5. build a CHILD RunContext parented to the swarm's driving run.
      6. retry loop: on exception retry up to ``max_task_retries`` extra times;
         on final failure mark the task FAILED and return None (catch-and-
         continue so a coordinator round completes even if a worker errors).

    Returns the worker's RunResult on success, or None if there was nothing to
    claim or every attempt failed.
    """
    # : max_depth guard. Fires before claim_task so a too-deep
    # task is rejected without consuming a worker slot.
    max_depth = ctx.spec.limits.max_depth
    if max_depth is not None:
        depth = await _compute_depth(ctx, task)
        if depth > max_depth:
            raise SwarmLimitExceededError(
                f"task {task.id} depth {depth} exceeds max_depth={max_depth}",
                kind="max_depth",
            )

    compiled = ctx.agents[task.assigned_agent_id]
    claimed = await ctx.swarm_store.claim_task(ctx.swarm_run.id, task.assigned_agent_id)
    if claimed is None:
        return None

    last_exc: "BaseException | None" = None
    child_run_id: "str | None" = None
    # Each retry iteration records one
    # SwarmTaskAttempt. ``base_attempt`` is the 1-based attempt number of the
    # FIRST iteration of this _run_task call, sourced from the actual audit
    # trail (``list_attempts``) rather than ``claimed.attempts`` -- the task
    # row's counter is bumped only by fail_task(), so it undercounts attempts
    # that were superseded by a reclaim/cancel (recorded in the attempt trail
    # via the "Superseded" close-out above, but never routed through
    # fail_task since the task's real terminal status belongs to whoever
    # else owns it). Counting the actual rows keeps numbering monotonic and
    # gap-free across reclaim/crash-recovery re-invocations of _run_task,
    # where claimed.attempts would otherwise start renumbering from a stale,
    # too-low count.
    base_attempt = len(await ctx.swarm_store.list_attempts(claimed.id)) + 1
    for _attempt in range(max_task_retries + 1):
        # Each attempt mints a FRESH child RunRecord id + scratch
        # session, not just the first one. Reusing one child_run_id across
        # attempts made AgentRunner.execute()'s run_store.create() collide on
        # the same primary key from attempt #2 onward -- a real
        # UNIQUE-constraint failure under SqlAlchemy storage that silently
        # turned every retry into a failure (masked under FileStore, which
        # overwrites on create() instead of rejecting the duplicate id).
        # Reusing one scratch session would also leak a failed attempt's
        # partial conversation into the retry's prompt, since AgentRunner
        # always prepends list_messages(session_id) for a fresh, non-resume
        # run. set_active_run records the fresh id on the task (bumping its
        # version) so SwarmRunner.cancel can locate the in-flight child Run.
        child_run_id = str(uuid.uuid4())
        # Unlike complete_task/fail_task below, a set_active_run conflict is
        # NOT retried with the fresh version -- it is discarded outright.
        # complete_task/fail_task can safely retry-once because
        # active_run_id == child_run_id is an owner token only THIS attempt
        # could have set; set_active_run is what ASSIGNS that token in the
        # first place, so at this point there is no owner token to check yet.
        # status == CLAIMED on a re-read proves only that *someone* claimed
        # the task, not that it is still this caller: claim_task never
        # touches active_run_id, so after this task's lease expired,
        # reclaim_expired_tasks() reset it PENDING, and a DIFFERENT worker
        # re-claimed it, a re-read still shows CLAIMED -- retrying with that
        # fresh version would silently steal the new owner's claim. No
        # worker has run yet for this attempt either, so there is no result
        # to lose by simply giving up.
        try:
            claimed = await ctx.swarm_store.set_active_run(
                claimed.id,
                child_run_id,
                expected_version=claimed.version,
            )
        except asyncio.CancelledError:
            raise
        except SwarmConflictError as exc:
            _LOGGER.warning(
                "swarm task %s lost set_active_run's fencing race before "
                "attempt %d started -- discarding: %s",
                claimed.id,
                base_attempt + _attempt,
                exc,
            )
            return None

        scratch_session_id = f"swarm:{ctx.swarm_run.id}:{claimed.id}:{child_run_id}"
        now = _now()
        await ctx.session_store.create(
            SessionRecord(
                id=scratch_session_id,
                parent_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )

        child_context = RunContext(
            run_id=child_run_id,
            root_run_id=ctx.parent_context.root_run_id,
            parent_run_id=ctx.swarm_run.run_id,
            session_id=scratch_session_id,
            runnable_id=claimed.assigned_agent_id,
            runnable_type=RunnableType.AGENT,
            user_id=ctx.parent_context.user_id,
            tenant_id=ctx.parent_context.tenant_id,
            workspace=None,
        )

        # Record the RUNNING attempt BEFORE invoking the worker so the audit
        # trail captures the start even if the worker never returns (crash
        # mid-run). record_attempt is an upsert on attempt.id, so the trailing
        # SUCCEEDED|FAILED write reuses the same id and finishes the row.
        current_attempt = SwarmTaskAttempt(
            id=str(uuid.uuid4()),
            task_id=claimed.id,
            run_id=child_run_id,
            agent_id=claimed.assigned_agent_id or "",
            attempt=base_attempt + _attempt,
            status=AttemptStatus.RUNNING,
            started_at=_now(),
            finished_at=None,
            error=None,
        )
        await ctx.swarm_store.record_attempt(current_attempt)

        # Only a failure of the worker itself is a retry-worthy attempt
        # failure. complete_task()/record_attempt() below can also raise --
        # but those are persistence/fencing conflicts (the task was already
        # reclaimed after its lease expired, or cancelled, by a different
        # caller), not evidence the worker failed, so they must not be
        # caught here and mistaken for a worker failure that warrants a
        # retry or an eventual fail_task() call.
        try:
            result = await ctx.agent_runner.run(
                compiled,
                RunInput(prompt=claimed.input.prompt),
                child_context,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_exc = exc
            await ctx.swarm_store.record_attempt(
                replace(
                    current_attempt,
                    status=AttemptStatus.FAILED,
                    finished_at=_now(),
                    error=RunErrorInfo(
                        error_type=type(exc).__name__,
                        message=redact_exception(exc),
                    ),
                )
            )
            continue

        # The worker succeeded. A genuine fencing conflict here (ownership
        # moved to a reclaim or cancel) means this attempt's result is moot,
        # not failed, so it must not trigger a retry or fail_task(). A
        # conflict where this call still owns the task (status still CLAIMED
        # AND active_run_id still this attempt's -- e.g. a stale version from
        # a lease renewal) is retried once with the fresh version instead of
        # being discarded, since the worker's result is already in hand.
        completed = await _retry_fencing_conflict_once(
            ctx,
            claimed,
            still_owned=lambda t: (
                t.status is SwarmTaskStatus.CLAIMED and t.active_run_id == child_run_id
            ),
            op=lambda v: ctx.swarm_store.complete_task(
                claimed.id,
                result,
                expected_version=v,
                active_run_id=child_run_id,
            ),
        )
        if completed is None:
            # The worker succeeded, but this attempt's completion write lost
            # the fencing race -- ownership moved to a reclaim or cancel, so
            # the task's real terminal status belongs to whoever owns it
            # now. This attempt is neither SUCCEEDED (its write never landed)
            # nor a genuine worker FAILED, but the RUNNING row already
            # written above must still be closed out -- leaving it RUNNING
            # forever would be a permanent, misleading gap in the audit
            # trail. AttemptStatus has no third terminal state, so record it
            # FAILED with an error_type that distinguishes "superseded" from
            # an actual worker error for anyone reading the trail.
            try:
                await ctx.swarm_store.record_attempt(
                    replace(
                        current_attempt,
                        status=AttemptStatus.FAILED,
                        finished_at=_now(),
                        error=RunErrorInfo(
                            error_type="Superseded",
                            message=(
                                "worker succeeded but complete_task lost a "
                                "fencing race (task reclaimed or cancelled by "
                                "another owner) before this attempt's "
                                "completion could be recorded"
                            ),
                        ),
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "swarm task %s: failed to close out the superseded "
                    "attempt's audit row: %s",
                    claimed.id,
                    exc,
                )
            return None
        claimed = completed

        try:
            await ctx.swarm_store.record_attempt(
                replace(
                    current_attempt,
                    status=AttemptStatus.SUCCEEDED,
                    finished_at=_now(),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # The task itself is correctly SUCCEEDED at this point (the
            # complete_task() call above did not raise) -- only the
            # best-effort audit-trail write failed, so still return the
            # real result rather than treating this as a failed attempt.
            _LOGGER.warning(
                "swarm task %s completed but recording its SUCCEEDED "
                "attempt audit row failed: %s",
                claimed.id,
                exc,
            )
        return result

    # Every attempt's worker call genuinely failed -- mark the task FAILED.
    # A fencing conflict here uses the same reclassify-and-retry-once policy
    # as complete_task above: a stale version while still owning the task is
    # retried once; ownership genuinely lost is discarded (the reclaiming or
    # cancelling caller owns the task's real terminal status now, so this
    # call must not overwrite it).
    await _retry_fencing_conflict_once(
        ctx,
        claimed,
        still_owned=lambda t: (
            t.status is SwarmTaskStatus.CLAIMED and t.active_run_id == child_run_id
        ),
        op=lambda v: ctx.swarm_store.fail_task(
            claimed.id,
            RunErrorInfo(
                error_type=type(last_exc).__name__
                if last_exc is not None
                else "Unknown",
                message=(redact_exception(last_exc) if last_exc is not None else ""),
            ),
            expected_version=v,
            active_run_id=child_run_id,
        ),
    )
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

        async def _default(
            swarm_run: SwarmRun, completed: "tuple[SwarmTask, ...]", limits: SwarmLimits
        ) -> "tuple[TaskInput, ...]":
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
                ctx.swarm_run.id,
                status=SwarmTaskStatus.SUCCEEDED,
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

        if len(inputs) > ctx.spec.limits.max_tasks:
            raise SwarmLimitExceededError(
                f"fan-out exceeded max_tasks={ctx.spec.limits.max_tasks}",
                kind="max_tasks",
            )

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
