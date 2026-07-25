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
from typing import TYPE_CHECKING, Mapping

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

from ..events.payloads import SwarmCompleted, SwarmStarted
from ..events.context import EventStreamContext, append_event
from ..run.live_events import NullSwarmEventSink, SwarmEventSink
from ..events.store import EventStore
from ..run.cancellation import CancellationToken
from ..run.context import RunContext
from ..run.controller import RunController
from ..run.dispatch import RunDispatcher
from ..run.lifecycle import mark_completed, mark_failed
from ..run.models import (
    RunErrorInfo,
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
    RunnableType,
)
from ..run.store import RunStore
from ..session.models import MessageRole, NewSessionMessage
from ..session.store import SessionStore
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

if TYPE_CHECKING:
    from ..run.definition import RunDefinitionStore
from .spec import SwarmSpec
from .store import SwarmStore
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
        swarm_store: SwarmStore,
        compiler: AgentCompiler,
        dispatcher: RunDispatcher,
        clock: "Clock | None" = None,
    ) -> None:
        self._swarm_store = swarm_store
        # Injected Clock so timestamp logic is deterministic under test
        # (FakeClock) and uses the wall clock in production (the default).
        self._clock = clock if clock is not None else SystemClock()
        self._compiler = compiler
        self._dispatcher = dispatcher

    # -- run() ----------------------------------------------------------------
    async def execute(
        self,
        spec: SwarmSpec,
        request: RunInput,
        context: RunContext,
        *,
        agents: "Mapping[str, AgentSpec]",
        cancellation: "CancellationToken",
        swarm_event_sink: "SwarmEventSink | None" = None,
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
        # The Coordinator owns the swarm event sink (durable, per-Run); direct
        # callers pass None for a null audit trail. SwarmEngine never appends the
        # EventStore itself.
        sink = swarm_event_sink or NullSwarmEventSink()

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
            created_swarm = await self._swarm_store.create_run(swarm_run)
            swarm_run = await self._swarm_store.update_run(
                swarm_run.id,
                expected_version=created_swarm.version,
                status=SwarmStatus.RUNNING,
            )
            swarm_version = swarm_run.version

            # 2. SwarmStarted event (swarm-domain; emitted via the
            # Coordinator-owned sink, never SwarmEngine's own EventStore).
            await sink.emit(
                SwarmStarted(swarm_run_id=swarm_run.id, swarm_id=spec.id)
            )

            # 3. Build the context the strategy consumes + delegate the round loop.
            ctx = SwarmExecutionContext(
                spec=spec,
                swarm_run=swarm_run,
                request=request,
                parent_context=context,
                dispatcher=self._dispatcher,
                agents=compiled_agents,
                swarm_store=self._swarm_store,
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
            swarm_run = await self._swarm_store.update_run(
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
            await self._swarm_store.update_run(
                swarm_run.id,
                expected_version=swarm_version,
                status=SwarmStatus.SUCCEEDED,
            )
            await sink.emit(SwarmCompleted(swarm_run_id=swarm_run.id))
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
                    await self._finalize_cancelled_swarm_run(swarm_run.id)
                except Exception as swarm_exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "failed to transition swarm run %s to CANCELLED: %s",
                        swarm_run.id,
                        swarm_exc,
                    )
            # Best-effort swarm-domain cancel event via the Coordinator-owned
            # sink (the Coordinator separately persists run-domain RunCancelled
            # via the acknowledge_cancel commit). Never masks the cancellation.
            if swarm_run is not None:
                try:
                    from ..events.payloads import SwarmCancelled

                    await sink.emit(SwarmCancelled(swarm_run_id=swarm_run.id))
                except Exception as swarm_exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "failed to emit SwarmCancelled event for swarm %s",
                        swarm_run.id,
                        swarm_exc,
                    )
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
                    await self._swarm_store.update_run(
                        swarm_run.id,
                        expected_version=swarm_version,
                        status=SwarmStatus.FAILED,
                    )
                except Exception as swarm_exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "failed to transition swarm run %s to FAILED: %s",
                        swarm_run.id,
                        swarm_exc,
                    )
            try:
                from ..events.payloads import SwarmFailed

                await sink.emit(
                    SwarmFailed(
                        swarm_run_id=swarm_run.id if swarm_run else "",
                        error=f"{type(exc).__name__}: {redact_exception(exc)}",
                    )
                )
            except Exception:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to emit SwarmFailed event for swarm %s",
                    swarm_run.id if swarm_run else "?",
                )
            return SwarmFailedOutcome(error=error_info)

    # -- resume() -------------------------------------------------------------
    async def _finalize_cancelled_swarm_run(self, swarm_run_id: str) -> None:
        """Same finalization semantics as :meth:`_finalize_cancelled_run`,
        for the SwarmRun record."""
        current = await self._swarm_store.get_run(swarm_run_id)
        if current is None:
            return
        if current.status in (
            SwarmStatus.SUCCEEDED,
            SwarmStatus.FAILED,
            SwarmStatus.CANCELLED,
        ):
            return
        if current.status == SwarmStatus.CANCELLING:
            await self._swarm_store.update_run(
                swarm_run_id,
                expected_version=current.version,
                status=SwarmStatus.CANCELLED,
            )
            return
        cancelling = await self._swarm_store.update_run(
            swarm_run_id,
            expected_version=current.version,
            status=SwarmStatus.CANCELLING,
        )
        await self._swarm_store.update_run(
            swarm_run_id,
            expected_version=cancelling.version,
            status=SwarmStatus.CANCELLED,
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
