#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmRunner: the top-level orchestrator that ties the SwarmSpec -> strategy ->
child Runs flow together . It compiles the member agents, creates the
driving RunRecord (runnable_type=SWARM) + SwarmRun, builds the
SwarmExecutionContext, delegates the round loop to the resolved strategy, writes
ONLY the final aggregate to the shared/parent Session, and transitions the
driving Run to SUCCEEDED.

This module owns the DRIVING swarm lifecycle only. The round loop, per-round /
per-task events, task persistence, and aggregation are the strategy's job (see
swarm.strategy). SwarmRunner never calls a model itself -- it constructs
one AgentEngine and hands it to the SwarmExecutionContext so the strategy's
``_run_task`` can drive child Runs.

Critical invariant (established by strategy._run_task): a SwarmTask's
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
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Mapping

from ..agent.compiler import AgentCompiler
from ..agent.models import CompiledAgent
from ..agent.spec import AgentSpec
from ..errors import (
    InvalidRunTransitionError,
    RunNotFoundError,
    SwarmError,
    SwarmLimitExceededError,
    SwarmRunNotFoundError,
    SwarmResumeUnsupportedError,
)

from ..events.payloads import SwarmCompleted, SwarmStarted
from ..events.context import EventStreamContext, append_event
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
from .models import SwarmCheckpoint, SwarmRun, SwarmStatus, SwarmTaskStatus, TokenUsage

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

# A driving Run in any of these states must not be resumed -- the swarm already
# reached a terminal outcome, so re-entering the strategy could re-drive worker
# side effects. Non-terminal states (RUNNING for an in-flight recoverable
# swarm) remain resumable. The driving Run is the SwarmRun's own RunRecord; a
# swarm pause is tracked in SwarmRun.status, so the driving Run stays RUNNING
# while paused and only leaves RUNNING when the swarm finishes.
_DRIVING_TERMINAL_STATUSES = frozenset(
    {
        RunStatus.SUCCEEDED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }
)


class SwarmRunner:
    """Orchestrates one Swarm invocation end-to-end. Construct once, call
    ``run()`` per invocation. ``resume()`` re-enters the strategy after a
    partial run; ``cancel()`` propagates real cancellation through
    ``RunController`` when wired.

    SwarmRunner does NOT assemble an AgentEngine itself -- Runtime is the
    single assembly point. The caller (normally
    ``Runtime.build()``) must hand in the SAME ``RunDispatcher`` (backed by the
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
        run_store: RunStore,
        session_store: SessionStore,
        event_store: EventStore,
        compiler: AgentCompiler,
        dispatcher: RunDispatcher,
        run_controller: "RunController | None" = None,
        run_definitions: "RunDefinitionStore",
    ) -> None:
        if run_definitions is None:
            # Defense in depth: Runtime.build rejects this up front, but a
            # hand-built SwarmRunner must also fail rather than silently
            # skipping worker snapshots.
            raise SwarmError("SwarmRunner requires a RunDefinitionStore")
        self._swarm_store = swarm_store
        self._run_store = run_store
        self._session_store = session_store
        self._event_store = event_store
        self._compiler = compiler
        self._dispatcher = dispatcher
        self._run_controller = run_controller
        self._run_definitions = run_definitions
        self._resume_locks: "dict[str, asyncio.Lock]" = {}
        # Single owner of the swarm RunDefinitionSnapshot (the prior double-create
        # with Runtime is gone -- Runtime delegates swarm snapshot creation here).
        from ..run.preparation import RunPreparationCoordinator

        self._prepare = RunPreparationCoordinator(run_definitions)

    # -- run() ----------------------------------------------------------------

    async def run(
        self,
        spec: SwarmSpec,
        request: RunInput,
        context: RunContext,
        *,
        agents: "Mapping[str, AgentSpec]",
    ) -> RunResult:
        compiled_agents = await self._compile_members(spec, agents)
        now = datetime.now(timezone.utc)
        # Initialized to None so a failure during snapshot preparation does not
        # leave the best-effort FAILED cleanup with unbound names.
        driving_running = None
        swarm_run = None
        swarm_version = None

        # 0. Persist the immutable run-definition snapshot BEFORE any state is
        # created. A non-serializable spec (e.g. strategy coordinator_fn) fails
        # fast here -- no orphan Run/SwarmRun left in RUNNING.
        await self._prepare.prepare_swarm_run(
            spec=spec, members=agents, context=context
        )

        # 1. driving RunRecord (the Swarm itself is a row in RunStore).
        record = RunRecord(
            id=context.run_id,
            root_run_id=context.root_run_id,
            parent_run_id=context.parent_run_id,
            session_id=context.session_id,
            runnable_id=spec.id,
            runnable_type=RunnableType.SWARM,
            status=RunStatus.PENDING,
            input=request,
            result=None,
            error=None,
            version=1,
            created_at=now,
            started_at=None,
            finished_at=None,
        )
        created = await self._run_store.create(record)
        driving_running = await self._run_store.transition(
            context.run_id, RunStatus.RUNNING, expected_version=created.version
        )

        # 2. SwarmRun.
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
        # version is now 2 after the PENDING -> RUNNING update.
        swarm_version = swarm_run.version

        # register the driving coroutine
        # + a fresh CancellationToken with run_controller, mirroring
        # AgentEngine.execute()'s own registration. Runtime.cancel(run_id) /
        # SwarmRunner.cancel() can then call run_controller.cancel(run_id)
        # to actually interrupt this coroutine (task.cancel()) instead of
        # only flipping store status. None (default) preserves the old
        # store-only behavior for callers that don't wire a controller.
        token: "CancellationToken | None" = None
        if self._run_controller is not None:
            token = CancellationToken()
            current_task = asyncio.current_task()
            if current_task is not None:
                await self._run_controller.register(context.run_id, current_task, token)

        try:
            if token is not None:
                await token.raise_if_cancelled()
            # 3. SwarmStarted event (store assigns the next sequence).
            await append_event(
                self._event_store,
                EventStreamContext.from_run_context(context),
                SwarmStarted(swarm_run_id=swarm_run.id, swarm_id=spec.id),
            )

            # 4. build the context the strategy consumes + delegate the round loop.
            ctx = SwarmExecutionContext(
                spec=spec,
                swarm_run=swarm_run,
                request=request,
                parent_context=context,
                dispatcher=self._dispatcher,
                compiler=self._compiler,
                agents=compiled_agents,
                swarm_store=self._swarm_store,
                run_store=self._run_store,
                session_store=self._session_store,
                event_store=self._event_store,
                run_definitions=self._run_definitions,
            )
            strategy = build_strategy(spec.strategy)
            # : SwarmLimits.timeout_seconds wraps the strategy
            # round loop in asyncio.wait_for. On timeout the TimeoutError is
            # translated to SwarmError("swarm timeout: ...") so the generic
            # FAILED handler below records a descriptive message; timeout_seconds
            # None means no timeout wrapper is needed.
            timeout = spec.limits.timeout_seconds
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(strategy.run(ctx), timeout=timeout)
                else:
                    result = await strategy.run(ctx)
            except asyncio.TimeoutError:
                raise SwarmError(f"swarm timeout: exceeded timeout_seconds={timeout}")

            # enforce SwarmLimits.max_total_tokens. aggregate() sums each
            # worker RunResult.token_usage (populated by AgentEngine from the
            # model's usage) into the aggregate result, so one comparison here
            # covers every task. max_total_cost is declared on SwarmLimits but
            # not enforced -- no cost-per-token rates are configured. The accumulated
            # usage is also persisted onto the SwarmRun (bumping its version,
            # which the trailing SUCCEEDED update_run picks up via swarm_version).
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

            # re-check right before committing success -- a cancel
            # that raced in while strategy.run() was wrapping up (and didn't
            # happen to land on an in-flight await) must not still write a
            # successful result.
            if token is not None:
                await token.raise_if_cancelled()

            # 5. write ONLY the final aggregate to the shared/parent Session.
            if spec.context_policy.write_aggregate_to_session:
                await self._write_aggregate(context, result)

            # 6. transition driving Run + SwarmRun to SUCCEEDED.
            await mark_completed(
                self._run_store,
                context.run_id,
                expected_version=driving_running.version,
                result=result,
            )
            await self._swarm_store.update_run(
                swarm_run.id,
                expected_version=swarm_version,
                status=SwarmStatus.SUCCEEDED,
            )

            # 7. SwarmCompleted event (store assigns the next sequence).
            await append_event(
                self._event_store,
                EventStreamContext.from_run_context(context),
                SwarmCompleted(swarm_run_id=swarm_run.id),
            )
            return result
        except asyncio.CancelledError:
            # real
            # cancel path -- CancelledError surfaces here either from
            # task.cancel() interrupting an in-flight await inside
            # strategy.run() (child agent call, store I/O, ...) or from one of
            # the token.raise_if_cancelled() checks above. Mirrors
            # AgentEngine.execute()'s own CancelledError handler exactly,
            # including the race AgentEngine defends against: Runtime.cancel(
            # run_id) may have ALREADY transitioned the driving Run to
            # CANCELLING (via
            # run_controller.cancel(), which is what triggered this
            # CancelledError in the first place) before this handler runs. A
            # naive "always transition RUNNING -> CANCELLING first" would then
            # hit InvalidRunTransitionError (CANCELLING is not a valid SOURCE
            # for a CANCELLING target) and never reach CANCELLED, leaving the
            # run stuck in CANCELLING forever. _finalize_cancelled_run/
            # _finalize_cancelled_swarm_run re-read current status and either
            # (a) skip if already terminal, (b) go straight to CANCELLED if
            # already CANCELLING, or (c) do the normal two-step transition.
            try:
                await self._finalize_cancelled_run(context.run_id)
            except Exception as transition_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition driving run %s to CANCELLED: %s",
                    context.run_id,
                    transition_exc,
                )
            try:
                await self._finalize_cancelled_swarm_run(swarm_run.id)
            except Exception as swarm_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition swarm run %s to CANCELLED: %s",
                    swarm_run.id,
                    swarm_exc,
                )
            raise
        except Exception as exc:
            # Best-effort cleanup: flip both records to FAILED, then re-raise.
            # The driving Run's expected version is the post-RUNNING version
            # captured in driving_running.version (no intermediate transition
            # bumps it inside the try block above); the SwarmRun's is tracked in
            # swarm_version.
            #
            # The FAILED transitions are kept
            # best-effort ONLY because we are already in the failing path:
            # letting either transition error escape would replace the ORIGINAL
            # exc (the actual cause) with a store/version error, losing the
            # cause for the caller. The warnings keep the transition failures
            # visible rather than silent.
            from ..governance.security.redact import redact_exception

            error_info = RunErrorInfo(
                error_type=type(exc).__name__, message=redact_exception(exc)
            )
            if driving_running is not None:
                try:
                    await mark_failed(
                        self._run_store,
                        context.run_id,
                        expected_version=driving_running.version,
                        error=error_info,
                    )
                except Exception as transition_exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "failed to transition driving run %s to FAILED: %s",
                        context.run_id,
                        transition_exc,
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
            # Best-effort terminal SwarmFailed event  A failure here is
            # an observability gap, not state corruption (the FAILED transition
            # above is the authoritative signal).
            try:
                from ..events.payloads import SwarmFailed

                await append_event(
                    self._event_store,
                    EventStreamContext.from_run_context(context),
                    SwarmFailed(
                        swarm_run_id=swarm_run.id,
                        error=f"{type(exc).__name__}: {redact_exception(exc)}",
                    ),
                )
            except Exception:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to append SwarmFailed event for swarm %s", swarm_run.id
                )
            raise
        finally:
            if self._run_controller is not None:
                await self._run_controller.unregister(context.run_id)

    # -- resume() -------------------------------------------------------------

    async def resume(
        self,
        swarm_run_id: str,
    ) -> RunResult:
        """Resume a paused swarm from its immutable persisted definition. Loads
        the SwarmSpec + member agents from the RunDefinitionSnapshot so the
        caller cannot inject a different spec or identity."""
        swarm_run = await self._swarm_store.get_run(swarm_run_id)
        if swarm_run is None:
            raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")
        driving = await self._run_store.get(swarm_run.run_id)
        if driving is None:
            raise RunNotFoundError(f"driving run not found: {swarm_run.run_id}")

        # Only explicit pause/recovery states have a checkpointed resume
        # protocol. RUNNING is never treated as implicitly recoverable.
        _TERMINAL = (
            SwarmStatus.SUCCEEDED,
            SwarmStatus.FAILED,
            SwarmStatus.CANCELLED,
        )
        if swarm_run.status in _TERMINAL or swarm_run.status not in (
            SwarmStatus.PAUSED,
            SwarmStatus.RECOVERABLE,
        ):
            raise InvalidRunTransitionError(
                f"cannot resume swarm run in terminal status {swarm_run.status}"
            )
        # The driving Run must also be non-terminal: a SUCCEEDED/FAILED/CANCELLED
        # driving Run means the swarm already reached its outcome, so resuming
        # would re-enter the strategy and could re-drive worker side effects.
        # Checked BEFORE the snapshot/compile/strategy.resume so a terminal
        # driving Run never reaches execution.
        if driving.status in _DRIVING_TERMINAL_STATUSES:
            raise InvalidRunTransitionError(
                f"cannot resume swarm when driving run is in terminal status "
                f"{driving.status}"
            )
        resume_lock = self._resume_locks.setdefault(swarm_run.id, asyncio.Lock())
        if resume_lock.locked():
            raise InvalidRunTransitionError(
                f"swarm resume already claimed: {swarm_run.id}"
            )
        await resume_lock.acquire()

        # Restore the immutable swarm definition (spec + member agents).
        try:
            snapshot = await self._run_definitions.get(swarm_run.run_id)
            if snapshot is None:
                raise SwarmRunNotFoundError(
                    f"no run-definition snapshot for swarm run: {swarm_run_id}"
                )
            from ..run.definition import deserialize_agent_spec, deserialize_swarm_spec

            spec_data = snapshot.serialized_spec
            spec = deserialize_swarm_spec(spec_data["spec"])
            agents = {
                aid: deserialize_agent_spec(a)
                for aid, a in spec_data.get("members", {}).items()
            }

            compiled_agents = await self._compile_members(spec, agents)
        except Exception:
            resume_lock.release()
            raise
        # Restore the ORIGINAL identity from the snapshot (user/tenant/workspace)
        # + lineage from the driving record -- never caller-supplied.
        parent_context = RunContext(
            run_id=driving.id,
            root_run_id=driving.root_run_id,
            parent_run_id=driving.parent_run_id,
            session_id=driving.session_id,
            runnable_id=driving.runnable_id,
            runnable_type=driving.runnable_type,
            user_id=snapshot.user_id,
            tenant_id=snapshot.tenant_id,
            workspace=snapshot.workspace,
        )
        # Capture the versions we read so the SUCCEEDED/FAILED transitions below
        # use the exact optimistic-concurrency token the store currently holds.
        driving_version = driving.version
        swarm_version = swarm_run.version
        swarm_run = await self._swarm_store.update_run(
            swarm_run.id,
            expected_version=swarm_run.version,
            status=SwarmStatus.RUNNING,
        )
        swarm_version = swarm_run.version

        try:
            ctx = SwarmExecutionContext(
                spec=spec,
                swarm_run=swarm_run,
                request=driving.input,
                parent_context=parent_context,
                dispatcher=self._dispatcher,
                compiler=self._compiler,
                agents=compiled_agents,
                swarm_store=self._swarm_store,
                run_store=self._run_store,
                session_store=self._session_store,
                event_store=self._event_store,
                run_definitions=self._run_definitions,
            )
            strategy = build_strategy(spec.strategy)
            tasks = await self._swarm_store.list_tasks(swarm_run.id)
            checkpoint = SwarmCheckpoint(
                completed_task_ids=tuple(
                    t.id for t in tasks if t.status is SwarmTaskStatus.SUCCEEDED
                ),
                failed_task_ids=tuple(
                    t.id for t in tasks if t.status is SwarmTaskStatus.FAILED
                ),
                pending_task_ids=tuple(
                    t.id for t in tasks if t.status is SwarmTaskStatus.PENDING
                ),
                active_task_ids=tuple(
                    t.id for t in tasks if t.status is SwarmTaskStatus.CLAIMED
                ),
                task_outputs={t.id: t.result for t in tasks if t.result is not None},
            )
            if not isinstance(strategy, ResumableSwarmStrategy):
                raise SwarmResumeUnsupportedError(
                    f"strategy {spec.strategy.kind!r} does not support resume"
                )
            result = await strategy.resume(ctx, checkpoint)

            if spec.context_policy.write_aggregate_to_session:
                await self._write_aggregate(parent_context, result)

            # The driving Run is non-terminal here (terminal driving Runs were
            # rejected before the snapshot was loaded), so it is always safe to
            # mark it SUCCEEDED with the version captured at read time.
            await mark_completed(
                self._run_store,
                parent_context.run_id,
                expected_version=driving_version,
                result=result,
            )
            await self._swarm_store.update_run(
                swarm_run.id,
                expected_version=swarm_version,
                status=SwarmStatus.SUCCEEDED,
            )

            # SwarmCompleted event -- the store assigns the next sequence
            # (events from the original run already occupy the low ones).
            await append_event(
                self._event_store,
                EventStreamContext.from_run_context(parent_context),
                SwarmCompleted(swarm_run_id=swarm_run.id),
            )
            return result
        except Exception as exc:
            from ..governance.security.redact import redact_exception

            error_info = RunErrorInfo(
                error_type=type(exc).__name__, message=redact_exception(exc)
            )
            # The driving Run is non-terminal here (terminal driving Runs were
            # rejected before the snapshot was loaded), so transition it to
            # FAILED. Kept best-effort ONLY because we are already in the
            # failing path -- letting the transition error escape would replace
            # the ORIGINAL exc with a store/version error. The warning keeps the
            # failure visible rather than silent.
            try:
                await self._run_store.transition(
                    parent_context.run_id,
                    RunStatus.FAILED,
                    expected_version=driving_version,
                    error=error_info,
                )
            except Exception as transition_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition driving run %s to FAILED: %s",
                    parent_context.run_id,
                    transition_exc,
                )
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
            raise
        finally:
            resume_lock.release()

    # -- cancel() -------------------------------------------------------------

    async def cancel(self, swarm_run_id: str) -> None:
        """Real cancel when a RunController is wired: transitions the SwarmRun
        and driving Run to
        CANCELLING and signals ``run_controller.cancel()`` for the driving
        run and every active child run, so an in-flight asyncio.Task (the
        swarm's own coroutine, or a child AgentEngine.execute()) actually
        stops -- not just a store-level status flip. Falls back to the old
        store-only CANCELLED transition when no controller is wired, or when
        a given run/child has no live registration (e.g. a stale record from
        a crashed worker, or cross-process where RunController cannot see
        the other process's tasks).

        Idempotent: a SwarmRun already in a terminal status is a no-op."""
        current = await self._swarm_store.get_run(swarm_run_id)
        if current is None:
            raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")
        if current.status in (
            SwarmStatus.SUCCEEDED,
            SwarmStatus.FAILED,
            SwarmStatus.CANCELLED,
        ):
            return  # already terminal -- no-op

        driving_run_id = current.run_id
        driving_in_flight = (
            self._run_controller is not None
            and self._run_controller.get_token(driving_run_id) is not None
        )
        if driving_in_flight:
            # CANCELLING first: the driving run()'s own
            # CancelledError handler finishes CANCELLING -> CANCELLED once it
            # actually stops. Both the SwarmRun and the driving RunRecord go
            # through this two-step transition.
            driving_record = await self._run_store.get(driving_run_id)
            if driving_record is not None and driving_record.status not in (
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.CANCELLING,
            ):
                await self._run_store.transition(
                    driving_run_id,
                    RunStatus.CANCELLING,
                    expected_version=driving_record.version,
                )
            await self._swarm_store.update_run(
                swarm_run_id,
                expected_version=current.version,
                status=SwarmStatus.CANCELLING,
            )
            await self._run_controller.cancel(driving_run_id)
        else:
            # No in-flight task (or no controller wired) -- nothing to
            # actually stop, so go straight to CANCELLED
            # behavior, preserved for the store-only / cross-process case).
            await self._swarm_store.update_run(
                swarm_run_id,
                expected_version=current.version,
                status=SwarmStatus.CANCELLED,
            )
            # Best-effort terminal SwarmCancelled event.
            await self._emit_swarm_terminal(
                driving_run_id, swarm_run_id, "SwarmCancelled"
            )

        claimed = await self._swarm_store.list_tasks(
            swarm_run_id, status=SwarmTaskStatus.CLAIMED
        )
        for task in claimed:
            # The child RunRecord's id is task.active_run_id (NOT
            # task.id). A CLAIMED task with active_run_id is None means the
            # strategy claimed it but crashed before set_active_run -- nothing
            # to cancel in RunStore, skip. Read the child's current version
            # rather than assuming it tracks task.version.
            #
            # best-effort per child: a child that
            # already completed or was cancelled concurrently is not a cancel
            # failure, and one stubborn child must not abort the rest of the
            # loop -- but the failure is logged (not silent) so a genuinely
            # unexpected error (e.g., OSError from a corrupted store) surfaces.
            if task.active_run_id is None:
                continue
            try:
                child = await self._run_store.get(task.active_run_id)
                if child is None:
                    continue
                if child.status in (
                    RunStatus.SUCCEEDED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                ):
                    continue
                child_in_flight = (
                    self._run_controller is not None
                    and self._run_controller.get_token(task.active_run_id) is not None
                )
                if child_in_flight:
                    if child.status != RunStatus.CANCELLING:
                        child = await self._run_store.transition(
                            task.active_run_id,
                            RunStatus.CANCELLING,
                            expected_version=child.version,
                        )
                    # The child's OWN AgentEngine.execute() -- registered with
                    # this SAME run_controller instance -- has its own
                    # CancelledError handler that finishes CANCELLING ->
                    # CANCELLED once it actually stops.
                    await self._run_controller.cancel(task.active_run_id)
                else:
                    await self._run_store.transition(
                        task.active_run_id,
                        RunStatus.CANCELLED,
                        expected_version=child.version,
                    )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to cancel child run %s for swarm run %s: %s",
                    task.active_run_id,
                    swarm_run_id,
                    exc,
                )

    async def _emit_swarm_terminal(
        self, driving_run_id: str, swarm_run_id: str, payload_name: str
    ) -> None:
        """Best-effort terminal swarm event (SwarmCancelled/SwarmFailed). Builds
        the EventStreamContext from the driving run's lineage; a failure here is an
        observability gap, not state corruption."""
        try:
            from ..events.payloads import SwarmCancelled, SwarmFailed

            driving = await self._run_store.get(driving_run_id)
            if driving is None:
                return
            ctx = EventStreamContext(
                stream_id=driving.id,
                run_id=driving.id,
                root_run_id=driving.root_run_id or driving.id,
                parent_run_id=driving.parent_run_id,
                session_id=driving.session_id,
                runnable_id=driving.runnable_id,
            )
            payload = (
                SwarmCancelled(swarm_run_id=swarm_run_id)
                if payload_name == "SwarmCancelled"
                else SwarmFailed(swarm_run_id=swarm_run_id, error="swarm error")
            )
            await append_event(self._event_store, ctx, payload)
        except Exception:  # noqa: BLE001
            _LOGGER.warning(
                "failed to append %s event for swarm %s", payload_name, swarm_run_id
            )

    # -- recover() -------------------------------------------------------------

    async def recover(self, swarm_run_id: str) -> None:
        """Best-effort restart-time scan. Walks every CLAIMED
        task whose lease has lapsed and reconciles it with the child RunRecord's
        current state:

          * Run SUCCEEDED -> ``complete_task`` (the strategy crashed between the
            child Run's SUCCEEDED transition and its ``complete_task`` call).
          * Run FAILED    -> ``fail_task`` (same gap, failed side).
          * Run RUNNING   -> leave alone. The worker may yet finish; this is the
            "禁止仅根据时间过期就盲目重复执行副作用任务" guard.
          * active_run_id is None / Run missing / Run CANCELLED or PENDING ->
            skip here, then ``reclaim_expired_tasks`` resets them to PENDING so
            the next ``resume()`` can pick them up.

        This is NOT distributed coordination. The caller MUST ensure no live
        worker is still processing this swarm run before invoking recover();
        otherwise a slow worker will have its task snatched on a stale lease.
        On FilesystemSwarmStore ``reclaim_expired_tasks`` is a no-op (single-process:
        nothing to reclaim at rest), so the requeue path effectively only fires
        on SqlAlchemySwarmStore -- which is the only backend that observes
        cross-process lease expiry anyway."""
        swarm_run = await self._swarm_store.get_run(swarm_run_id)
        if swarm_run is None:
            raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")

        claimed = await self._swarm_store.list_tasks(
            swarm_run_id,
            status=SwarmTaskStatus.CLAIMED,
        )
        now = datetime.now(timezone.utc)
        for task in claimed:
            # A task whose lease hasn't lapsed is presumed still being worked --
            # leave it alone (don't blindly re-run side-effecting tasks).
            if task.lease_expires_at is not None and task.lease_expires_at > now:
                continue
            # No active_run_id: the strategy crashed between claim_task and
            # set_active_run. Nothing to reconcile -- reclaim_expired_tasks
            # below will reset it to PENDING.
            if task.active_run_id is None:
                continue
            try:
                child = await self._run_store.get(task.active_run_id)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "recover: failed to read child run %s for task %s: %s",
                    task.active_run_id,
                    task.id,
                    exc,
                )
                continue
            # Missing Run record: lost. Reset to PENDING via reclaim below.
            if child is None:
                continue
            # Reconcile by terminal state. Best-effort per task so one bad
            # transition doesn't abort the whole recovery pass.
            try:
                if child.status == RunStatus.SUCCEEDED and child.result is not None:
                    await self._swarm_store.complete_task(
                        task.id,
                        child.result,
                        expected_version=task.version,
                        active_run_id=task.active_run_id,
                    )
                elif child.status == RunStatus.FAILED and child.error is not None:
                    await self._swarm_store.fail_task(
                        task.id,
                        child.error,
                        expected_version=task.version,
                        active_run_id=task.active_run_id,
                    )
                elif child.status == RunStatus.RUNNING:
                    # Worker may still be alive -- leave it. If the worker is
                    # actually dead, the next recover() pass after this Run
                    # reaches a terminal state will catch it.
                    pass
                # CANCELLED / PENDING Runs: leave for reclaim_expired_tasks.
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "recover: failed to reconcile task %s from child run %s (%s): %s",
                    task.id,
                    task.active_run_id,
                    child.status,
                    exc,
                )

        # Reset everything still CLAIMED with an expired lease (cases above we
        # didn't reconcile: no active_run_id, missing Run, non-terminal Run
        # state) to PENDING so resume() can re-drive them. On FilesystemSwarmStore
        # this is a documented no-op.
        await self._swarm_store.reclaim_expired_tasks(swarm_run_id)

    # -- cancellation finalization ---------------------------------------------

    async def _finalize_cancelled_run(self, run_id: str) -> None:
        """Drive the driving RunRecord to CANCELLED after a real
        ``CancelledError`` was observed. Mirrors ``AgentEngine.execute()``'s own
        CancelledError
        handler: re-reads current status rather than assuming RUNNING, since
        ``Runtime.cancel(run_id)`` may have ALREADY transitioned it to
        CANCELLING (that transition is precisely what preceded the
        ``run_controller.cancel()`` call that produced this CancelledError).

        * Already terminal (SUCCEEDED/FAILED/CANCELLED): no-op -- a
          concurrent terminal transition winning this race is not an error.
        * Already CANCELLING: go straight to CANCELLED -- attempting
          RUNNING/WAITING_APPROVAL/PAUSED -> CANCELLING again would hit
          InvalidRunTransitionError (CANCELLING is not a valid source for a
          CANCELLING target) and never reach CANCELLED.
        * Otherwise: the normal two-step CANCELLING -> CANCELLED transition."""
        current = await self._run_store.get(run_id)
        if current is None:
            return
        if current.status in (
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ):
            return
        if current.status == RunStatus.CANCELLING:
            await self._run_store.transition(
                run_id,
                RunStatus.CANCELLED,
                expected_version=current.version,
            )
            return
        cancelling = await self._run_store.transition(
            run_id,
            RunStatus.CANCELLING,
            expected_version=current.version,
        )
        await self._run_store.transition(
            run_id,
            RunStatus.CANCELLED,
            expected_version=cancelling.version,
        )

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

    async def _write_aggregate(self, context: RunContext, result: RunResult) -> None:
        """Append the single aggregate assistant message to the shared/parent
        Session. Sequence is assigned by the SessionStore itself, not
        computed here from `len(prior_messages) + 1`."""
        await self._session_store.append_messages(
            context.session_id,
            (
                NewSessionMessage(
                    role=MessageRole.ASSISTANT,
                    content=str(result.output),
                    run_id=context.run_id,
                ),
            ),
        )
