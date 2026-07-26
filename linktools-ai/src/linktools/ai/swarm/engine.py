#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmEngine: the top-level orchestrator that ties the SwarmSpec -> strategy ->
child Runs flow together . It compiles the member agents, creates the
driving RunRecord (runnable_type=SWARM) + SwarmRun, builds the
SwarmExecutionContext, delegates the round loop to the resolved strategy, writes
ONLY the final aggregate to the shared/parent Session, and transitions the
driving Run to SUCCEEDED.

This module owns the DRIVING swarm lifecycle only. The round loop, per-round /
per-task events, task persistence, and aggregation are the strategy's job (see
swarm.strategy). SwarmEngine never calls a model itself -- it constructs
one AgentEngine and hands it to the SwarmExecutionContext so the strategy's
``_run_task`` can drive child Runs.

Critical invariant (established by strategy._run_task): a SwarmStep's
``active_run_id`` is its child RunRecord's ``id`` (NOT the task's own id).
cancel() exploits this -- ``list_tasks(..., status=CLAIMED)`` yields tasks
whose ``active_run_id`` is the in-flight child Run to cancel.

resume() is explicit and caller-driven : no auto-resume-on-construct.
cancel() uses RunController when available: it transitions the SwarmRun and
driving Run to CANCELLING, signals the driving coroutine, and propagates
cancellation to active child Runs via their active_run_id. When no controller
or no in-flight registration is available, it falls back to store-only
CANCELLED transitions for stale/cross-process records."""

import asyncio
import logging
import uuid
from decimal import Decimal
from typing import Any, Mapping

from ..agent.compiler import AgentCompiler
from ..agent.models import CompiledAgent
from ..agent.spec import AgentSpec
from ..clock import Clock, SystemClock
from ..errors import (
    InvalidRunTransitionError,
    InvalidSwarmTransitionError,
    MCPToolError,
    ModelInvocationDeniedError,
    ModelOutputValidationError,
    ModelPolicyExceededError,
    ModelResultDeniedError,
    ModelRoutingError,
    RunNotFoundError,
    SwarmConflictError,
    SwarmError,
    SwarmLimitExceededError,
    SwarmRunNotFoundError,
    SwarmResumeUnsupportedError,
    SwarmStepConflictError,
    SwarmStepNotFoundError,
    ToolError,
    ToolSchemaError,
)

from ..events.context import EventStreamContext
from ..events.payloads import SwarmCancelled, SwarmCompleted, SwarmFailed, SwarmStarted
from ..run.cancellation import CancellationToken
from ..run.context import RunContext
from ..run.dispatch import RunDispatcher
from ..run.models import (
    RunErrorInfo,
    RunInput,
    RunResult,
)
from ..session.models import MessageRole, NewSessionMessage
from .commit import (
    CancelSwarmPayload,
    CompleteSwarmPayload,
    FailSwarmPayload,
    StartSwarmPayload,
    SwarmCommitCoordinator,
    SwarmCommitId,
    SwarmExecutionFence,
)
from .models import (
    SwarmCheckpoint,
    SwarmCompleted as SwarmCompletedOutcome,
    SwarmExecutionOutcome,
    SwarmFailed as SwarmFailedOutcome,
    SwarmRun,
    SwarmStatus,
    SwarmStepStatus,
    SwarmUsage,
    TokenUsage,
)

from .spec import SwarmSpec
from .strategy import (
    ResumableSwarmStrategy,
    SwarmExecutionContext,
    build_strategy,
)


_LOGGER = logging.getLogger(__name__)

# Exception types that count as EXPECTED swarm/strategy/model/tool failures
# for execute()'s outcome classification. Only these are turned into a
# SwarmFailed outcome; every other exception (configuration / invariant /
# protocol violations such as RuntimeInitializationError, RunInvariantError,
# CapabilityResolutionError, MCPConnectionError, and all unknown programming
# errors like TypeError/AttributeError/KeyError) propagates unchanged -- the
# spec explicitly forbids an except-Exception catch-all that swallows them
# into a SwarmFailed outcome. This mirrors the agent path's allowlist.
#
# SwarmError is the base for the legitimate expected failures execute() raises
# itself (a round timeout, a missing referenced agent), so it stays in the
# allowlist -- but its conflict / invariant / not-found / unsupported subtypes
# are NOT per-run failures (a version conflict on the SUCCEEDED transition
# would otherwise mislabel a swarm that actually completed). They are carved
# out and re-raised, mirroring how the agent path excludes RunConflictError /
# RunInvariantError. ToolSchemaError subclasses ToolError but is a contract/
# config violation, so it is likewise re-raised rather than swallowed.
_EXPECTED_SWARM_FAILURES: "tuple[type[BaseException], ...]" = (
    SwarmError,
    ModelRoutingError,
    ModelPolicyExceededError,
    ModelOutputValidationError,
    ModelInvocationDeniedError,
    ModelResultDeniedError,
    ToolError,
    MCPToolError,
)

# SwarmError subtypes that are conflict / invariant / not-found / unsupported
# violations rather than expected per-run failures -- propagate even though
# their base (SwarmError) is in the allowlist above.
_SWARM_INVARIANT_FAILURES: "tuple[type[BaseException], ...]" = (
    SwarmConflictError,
    InvalidSwarmTransitionError,
    SwarmStepConflictError,
    SwarmRunNotFoundError,
    SwarmStepNotFoundError,
    SwarmResumeUnsupportedError,
)

class SwarmEngine:
    """Orchestrates one Swarm invocation end-to-end. Construct once, call
    ``run()`` per invocation. ``resume()`` re-enters the strategy after a
    partial run; ``cancel()`` propagates real cancellation through
    ``RunController`` when wired.

    SwarmEngine does NOT assemble an AgentEngine itself -- Runtime is the
    single assembly point. The caller (normally
    ``build_runtime()``) must hand in the SAME ``RunDispatcher`` (backed by the
    AgentEngine instance) used for top-level Agent runs, so Swarm worker Runs
    get identical Tool/Policy/Middleware/UoW/Cancellation semantics instead of
    a second, divergent execution path. Passing ``run_controller`` (the SAME
    instance the dispatcher's runner was built with) is what makes
    ``cancel()`` able to actually stop an in-flight child Run -- see
    ``cancel()``."""

    def __init__(
        self,
        *,
        compiler: AgentCompiler,
        dispatcher: RunDispatcher,
        swarm_commit_coordinator: "SwarmCommitCoordinator",
        clock: "Clock | None" = None,
    ) -> None:
        # Injected Clock so timestamp logic is deterministic under test
        # (FakeClock) and uses the wall clock in production (the default).
        self._clock = clock if clock is not None else SystemClock()
        self._compiler = compiler
        self._dispatcher = dispatcher
        # REQUIRED SwarmCommitCoordinator: every lifecycle commit (start,
        # complete, fail, cancel of the SwarmRun) routes through it so each
        # commit is idempotent by commit_id and recorded in the commit log.
        # The spec's swarm-commit-boundary rule forbids direct swarm_store
        # lifecycle writes -- a terminal transition that does NOT go through
        # the commit log would defeat replay detection.
        self._swarm_commit_coordinator = swarm_commit_coordinator
        self._state_store = swarm_commit_coordinator.state_store

    async def _commit_swarm_transition(
        self,
        *,
        swarm_run_id: str,
        expected_version: int,
        target: "SwarmStatus",
        commit_id_suffix: str,
        result: "RunResult | None" = None,
        error: "RunErrorInfo | None" = None,
        event_context: "EventStreamContext | None" = None,
    ) -> "int":
        """Apply a terminal SwarmRun transition through the
        SwarmCommitCoordinator so the commit is idempotent by commit_id and
        recorded in the commit log. Returns the new version.
        ``commit_id_suffix`` distinguishes the operations (complete/fail/cancel)
        so the same swarm_run_id can carry distinct commit_ids per terminal
        kind."""
        from .commit import (
            CancelSwarmCommand,
            CompleteSwarmCommand,
            FailSwarmCommand,
        )

        commit_id = f"{commit_id_suffix}:{swarm_run_id}"
        if target is SwarmStatus.SUCCEEDED:
            if result is None:
                raise SwarmError("successful swarm commit requires a result")
            lifecycle_event = SwarmCompleted(swarm_run_id=swarm_run_id)
        elif target is SwarmStatus.FAILED:
            if error is None:
                raise SwarmError("failed swarm commit requires an error")
            lifecycle_event = SwarmFailed(
                swarm_run_id=swarm_run_id,
                error=error.message,
            )
        else:
            lifecycle_event = SwarmCancelled(swarm_run_id=swarm_run_id)
        if target is SwarmStatus.SUCCEEDED:
            result = await self._swarm_commit_coordinator.complete(
                CompleteSwarmCommand(
                    commit_id=SwarmCommitId(commit_id),
                    swarm_run_id=swarm_run_id,
                    expected_version=expected_version,
                    payload=CompleteSwarmPayload(
                        result=result,
                        completed_event=lifecycle_event,
                        event_context=event_context,
                    ),
                    fence=SwarmExecutionFence(f"swarm:{swarm_run_id}"),
                )
            )
        elif target is SwarmStatus.FAILED:
            result = await self._swarm_commit_coordinator.fail(
                FailSwarmCommand(
                    commit_id=SwarmCommitId(commit_id),
                    swarm_run_id=swarm_run_id,
                    expected_version=expected_version,
                    payload=FailSwarmPayload(
                        error=error,
                        failed_event=lifecycle_event,
                        event_context=event_context,
                    ),
                    fence=SwarmExecutionFence(f"swarm:{swarm_run_id}"),
                )
            )
        elif target is SwarmStatus.CANCELLED:
            result = await self._swarm_commit_coordinator.cancel(
                CancelSwarmCommand(
                    commit_id=SwarmCommitId(commit_id),
                    swarm_run_id=swarm_run_id,
                    expected_version=expected_version,
                    payload=CancelSwarmPayload(
                        cancelled_event=lifecycle_event,
                        event_context=event_context,
                    ),
                    fence=SwarmExecutionFence(f"swarm:{swarm_run_id}"),
                )
            )
        else:  # pragma: no cover -- defensive
            raise SwarmError(
                f"unsupported terminal swarm status for commit: {target!r}"
            )
        return int(result.get("version", expected_version))

    # -- run() ----------------------------------------------------------------
    async def execute(
        self,
        spec: SwarmSpec,
        request: RunInput,
        context: RunContext,
        *,
        agents: "Mapping[str, AgentSpec]",
        cancellation: "CancellationToken",
    ) -> "SwarmExecutionOutcome":
        """The target swarm execution loop. Drives the strategy + manages
        SwarmRun/SwarmStep domain state and returns a
        SwarmExecutionOutcome -- it does NOT create/transition the driving
        RunRecord, write the parent-session aggregate, or own the driving Run's
        execution claim/heartbeat/fencing (RunCoordinator owns all of that,
        mirroring the agent path). On an explicit cancellation the SwarmRun is
        transitioned CANCELLED and CancelledError re-raises so the Coordinator
        can converge the driving Run to CANCELLED.

        ``run()`` (the legacy full-lifecycle entry, kept on the fix branch)
        remains the path tests/direct callers use until they migrate."""
        now = self._clock.now()
        swarm_run: "SwarmRun | None" = None
        swarm_version: "int | None" = None
        # Lifecycle events are carried by typed commit payloads; the
        # coordinator appends them at the same durability boundary as state.
        # SwarmEngine never appends the EventStore itself.

        try:
            await cancellation.raise_if_cancelled()
            compiled_agents = await self._compile_members(spec, agents)

            # 1. SwarmRun (the driving RunRecord + run-definition snapshot are the
            # Coordinator's concern -- prepared before the start command).
            swarm_run = SwarmRun(
                id=str(uuid.uuid4()),
                run_id=context.run_id,
                round=0,
                status=SwarmStatus.PENDING,
                version=1,
                token_usage=TokenUsage(),
                cost=Decimal("0"),
                created_at=now,
                updated_at=now,
            )
            swarm_event_context = EventStreamContext.from_run_context(context)
            # Route through the SwarmCommitCoordinator: one atomic, commit_id-
            # keyed idempotent start. The payload carries the full initial
            # SwarmRun, so the coordinator re-creates the same shape (id
            # included) rather than the engine minting a second id.
            from .commit import StartSwarmCommand

            await self._swarm_commit_coordinator.start(
                StartSwarmCommand(
                    commit_id=SwarmCommitId(f"start:{swarm_run.id}"),
                    swarm_run_id=swarm_run.id,
                    expected_version=1,
                    payload=StartSwarmPayload(
                        run=swarm_run,
                        started_event=SwarmStarted(
                            swarm_run_id=swarm_run.id, swarm_id=spec.id
                        ),
                        event_context=swarm_event_context,
                    ),
                    fence=SwarmExecutionFence(f"swarm:{swarm_run.id}"),
                )
            )
            # RUNNING is a mid-execution state change (not a terminal commit),
            # so it remains a coordinator-owned state update -- the lifecycle
            # entry write (PENDING create) is what the commit_log protects.
            created_swarm = await self._swarm_commit_coordinator.get_run(swarm_run.id)
            if created_swarm is None:
                raise SwarmConflictError(
                    f"swarm run {swarm_run.id} missing after coordinator.start"
                )
            swarm_run = await self._swarm_commit_coordinator.update_run(
                swarm_run.id,
                expected_version=created_swarm.version,
                status=SwarmStatus.RUNNING,
            )
            swarm_version = swarm_run.version

            # 3. Build the context the strategy consumes + delegate the round loop.
            ctx = SwarmExecutionContext(
                spec=spec,
                swarm_run=swarm_run,
                request=request,
                parent_context=context,
                dispatcher=self._dispatcher,
                agents=compiled_agents,
                swarm_store=self._state_store,
            )
            strategy = build_strategy(spec.strategy)
            timeout = spec.limits.timeout_seconds
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(strategy.run(ctx), timeout=timeout)
                else:
                    result = await strategy.run(ctx)
            except asyncio.TimeoutError:
                raise SwarmError(f"swarm timeout: exceeded timeout_seconds={timeout}")

            limits = spec.limits
            acc_input = int(result.token_usage.get("input_tokens", 0))
            acc_output = int(result.token_usage.get("output_tokens", 0))
            if (
                limits.max_total_tokens is not None
                and (acc_input + acc_output) > limits.max_total_tokens
            ):
                raise SwarmLimitExceededError(
                    f"swarm exceeded max_total_tokens={limits.max_total_tokens}: "
                    f"used {acc_input + acc_output}",
                    kind="max_total_tokens",
                )
            swarm_run = await self._swarm_commit_coordinator.update_run(
                swarm_run.id,
                expected_version=swarm_version,
                token_usage=TokenUsage(
                    input_tokens=acc_input, output_tokens=acc_output
                ),
            )
            swarm_version = swarm_run.version

            await cancellation.raise_if_cancelled()

            # 4. Transition SwarmRun SUCCEEDED + SwarmCompleted event. The
            # driving RunRecord + parent-session aggregate write are the
            # Coordinator's job (it reads them off SwarmCompleted).
            swarm_version = await self._commit_swarm_transition(
                swarm_run_id=swarm_run.id,
                expected_version=swarm_version,
                target=SwarmStatus.SUCCEEDED,
                commit_id_suffix="complete",
                result=result.result,
                event_context=swarm_event_context,
            )
            aggregate_messages: "tuple[NewSessionMessage, ...]" = ()
            if spec.context_policy.write_aggregate_to_session:
                aggregate_messages = (
                    NewSessionMessage(
                        role=MessageRole.ASSISTANT,
                        content=str(result.output),
                        run_id=context.run_id,
                    ),
                )
            return SwarmCompletedOutcome(
                result=result,
                aggregate_messages=aggregate_messages,
                usage=SwarmUsage(input_tokens=acc_input, output_tokens=acc_output),
            )
        except asyncio.CancelledError:
            # Explicit cancellation: transition SwarmRun CANCELLED (swarm
            # domain), then re-raise so the Coordinator cancels the driving Run.
            if swarm_run is not None and swarm_version is not None:
                try:
                    await self._finalize_cancelled_swarm_run(
                        swarm_run.id, event_context=EventStreamContext.from_run_context(context)
                    )
                except Exception as swarm_exc:  # noqa: BLE001
                    raise SwarmError(
                        f"swarm cancellation convergence failed for {swarm_run.id}"
                    ) from swarm_exc
            # Best-effort swarm-domain cancel event via the Coordinator-owned
            # sink (the Coordinator separately persists run-domain RunCancelled
            # via the acknowledge_cancel commit). Never masks the cancellation.
            raise
        except _EXPECTED_SWARM_FAILURES as exc:
            # Conflict / invariant / not-found swarm subtypes are NOT per-run
            # failures (a version conflict on the SUCCEEDED transition would
            # otherwise mislabel a swarm that actually completed) -- propagate
            # them, mirroring the agent path's RunConflictError/RunInvariantError
            # exclusion. A malformed tool schema (ToolSchemaError) is likewise a
            # contract/config violation, not a per-run tool failure -- even
            # though it subclasses ToolError, it must propagate instead of
            # becoming a SwarmFailed outcome.
            if isinstance(exc, _SWARM_INVARIANT_FAILURES) or isinstance(
                exc, ToolSchemaError
            ):
                raise
            from ..governance.security.redact import redact_exception

            error_info = RunErrorInfo(
                error_type=type(exc).__name__, message=redact_exception(exc)
            )
            if swarm_run is not None and swarm_version is not None:
                try:
                    swarm_version = await self._commit_swarm_transition(
                        swarm_run_id=swarm_run.id,
                        expected_version=swarm_version,
                        target=SwarmStatus.FAILED,
                        commit_id_suffix="fail",
                        error=error_info,
                        event_context=swarm_event_context,
                    )
                except Exception as swarm_exc:  # noqa: BLE001
                    raise SwarmError(
                        f"swarm failure convergence failed for {swarm_run.id}"
                    ) from swarm_exc
            return SwarmFailedOutcome(error=error_info)

    # -- resume() -------------------------------------------------------------
    async def _finalize_cancelled_swarm_run(
        self,
        swarm_run_id: str,
        *,
        event_context: "EventStreamContext | None" = None,
    ) -> None:
        """Same finalization semantics as :meth:`_finalize_cancelled_run`,
        for the SwarmRun record."""
        current = await self._swarm_commit_coordinator.get_run(swarm_run_id)
        if current is None:
            return
        if current.status in (
            SwarmStatus.SUCCEEDED,
            SwarmStatus.FAILED,
            SwarmStatus.CANCELLED,
        ):
            return
        if current.status == SwarmStatus.CANCELLING:
            await self._commit_swarm_transition(
                swarm_run_id=swarm_run_id,
                expected_version=current.version,
                target=SwarmStatus.CANCELLED,
                commit_id_suffix="cancel",
                event_context=event_context,
            )
            return
        cancelling = await self._swarm_commit_coordinator.update_run(
            swarm_run_id,
            expected_version=current.version,
            status=SwarmStatus.CANCELLING,
        )
        await self._commit_swarm_transition(
            swarm_run_id=swarm_run_id,
            expected_version=cancelling.version,
            target=SwarmStatus.CANCELLED,
            commit_id_suffix="cancel",
            event_context=event_context,
        )

    # -- helpers --------------------------------------------------------------

    async def _compile_members(
        self, spec: SwarmSpec, agents: "Mapping[str, AgentSpec]"
    ) -> "dict[str, CompiledAgent]":
        """Compile only the agents the spec references (coordinator + members).
        Raise SwarmError if any referenced agent_id is absent from ``agents``."""
        needed: "set[str]" = {spec.coordinator.agent_id}
        needed.update(a.agent_id for a in spec.agents)
        missing = needed - set(agents.keys())
        if missing:
            raise SwarmError(
                f"missing AgentSpec for referenced agent ids: {sorted(missing)}"
            )
        compiled: "dict[str, CompiledAgent]" = {}
        for agent_id in needed:
            compiled[agent_id] = await self._compiler.compile(agents[agent_id])
        return compiled
