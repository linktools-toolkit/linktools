#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmRunner: the top-level orchestrator that ties the SwarmSpec -> strategy ->
child Runs flow together (spec 22.4). It compiles the member agents, creates the
driving RunRecord (runnable_type=SWARM) + SwarmRun, builds the
SwarmExecutionContext, delegates the round loop to the resolved strategy, writes
ONLY the final aggregate to the shared/parent Session, and transitions the
driving Run to SUCCEEDED.

This module owns the DRIVING swarm lifecycle only. The round loop, per-round /
per-task events, task persistence, and aggregation are the strategy's job (see
swarm.strategy). SwarmRunner never calls a model itself -- it constructs
one AgentRunner and hands it to the SwarmExecutionContext so the strategy's
``_run_task`` can drive child Runs.

Critical invariant (established by strategy._run_task): a SwarmTask's
``active_run_id`` is its child RunRecord's ``id`` (NOT the task's own id).
cancel() exploits this -- ``list_tasks(..., status=CLAIMED)`` yields tasks
whose ``active_run_id`` is the in-flight child Run to cancel.

resume() is explicit and caller-driven (Decision #3): no auto-resume-on-construct.
cancel() is store-level (Decision #4): no live asyncio task cancellation."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Mapping

from ..agent.compiler import AgentCompiler
from ..agent.models import CompiledAgent
from ..agent.runner import AgentRunner
from ..agent.spec import AgentSpec
from ..errors import (
    RunNotFoundError,
    SwarmError,
    SwarmLimitExceededError,
    SwarmRunNotFoundError,
)

if TYPE_CHECKING:
    from ..knowledge.retriever import Retriever
    from ..memory.store import MemoryStore
from ..events.payloads import SwarmCompleted, SwarmStarted
from ..events.store import EventStore
from ..run.checkpoint import CheckpointStore
from ..run.context import RunContext
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
from .models import SwarmRun, SwarmStatus, SwarmTaskStatus, TokenUsage
from .spec import SwarmSpec
from .store import SwarmStore
from .strategy import SwarmExecutionContext, build_strategy


_LOGGER = logging.getLogger(__name__)


class SwarmRunner:
    """Orchestrates one Swarm invocation end-to-end. Construct once, call
    ``run()`` per invocation. ``resume()`` re-enters the strategy after a
    partial run; ``cancel()`` flips the SwarmRun + in-flight child Runs to
    CANCELLED at the store level."""

    def __init__(
        self,
        *,
        swarm_store: SwarmStore,
        run_store: RunStore,
        session_store: SessionStore,
        event_store: EventStore,
        checkpoint_store: CheckpointStore,
        compiler: AgentCompiler,
        memory_store: "MemoryStore | None" = None,
        retriever: "Retriever | None" = None,
    ) -> None:
        self._swarm_store = swarm_store
        self._run_store = run_store
        self._session_store = session_store
        self._event_store = event_store
        self._compiler = compiler
        # One AgentRunner is reused for every child Run the strategy spawns.
        # SwarmRunner never calls it directly -- it is handed to the
        # SwarmExecutionContext so strategy._run_task can drive worker Runs.
        # memory_store + retriever are forwarded so the same Phase-5 prompt
        # injection that AgentRunner applies to top-level runs also applies to
        # each swarm worker Run (default None -> no change -> no injection).
        self._agent_runner = AgentRunner(
            run_store=run_store,
            session_store=session_store,
            event_store=event_store,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            retriever=retriever,
        )

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
            swarm_run.id, expected_version=created_swarm.version, status=SwarmStatus.RUNNING
        )
        # version is now 2 after the PENDING -> RUNNING update.
        swarm_version = swarm_run.version

        try:
            # 3. SwarmStarted event (store assigns the next sequence).
            await self._event_store.append(
                stream_id=context.run_id,
                run_id=context.run_id,
                root_run_id=context.root_run_id,
                parent_run_id=context.parent_run_id,
                session_id=context.session_id,
                runnable_id=context.runnable_id,
                payload=SwarmStarted(swarm_run_id=swarm_run.id, swarm_id=spec.id),
            )

            # 4. build the context the strategy consumes + delegate the round loop.
            ctx = SwarmExecutionContext(
                spec=spec,
                swarm_run=swarm_run,
                request=request,
                parent_context=context,
                agent_runner=self._agent_runner,
                compiler=self._compiler,
                agents=compiled_agents,
                swarm_store=self._swarm_store,
                run_store=self._run_store,
                session_store=self._session_store,
                event_store=self._event_store,
            )
            strategy = build_strategy(spec.strategy)
            # GAP-09 (spec 22.3): SwarmLimits.timeout_seconds wraps the strategy
            # round loop in asyncio.wait_for. On timeout the TimeoutError is
            # translated to SwarmError("swarm timeout: ...") so the generic
            # FAILED handler below records a descriptive message; timeout_seconds
            # left at None reproduces the pre-GAP-09 path (no wait_for wrapper).
            timeout = spec.limits.timeout_seconds
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(strategy.run(ctx), timeout=timeout)
                else:
                    result = await strategy.run(ctx)
            except asyncio.TimeoutError:
                raise SwarmError(f"swarm timeout: exceeded timeout_seconds={timeout}")

            # GAP-09: enforce SwarmLimits.max_total_tokens. aggregate() sums each
            # worker RunResult.token_usage (populated by AgentRunner from the
            # model's usage) into the aggregate result, so one comparison here
            # covers every task. max_total_cost is declared on SwarmLimits but
            # deferred -- no cost-per-token rates exist yet. The accumulated
            # usage is also persisted onto the SwarmRun (bumping its version,
            # which the trailing SUCCEEDED update_run picks up via swarm_version).
            limits = spec.limits
            acc_input = int(result.token_usage.get("input_tokens", 0))
            acc_output = int(result.token_usage.get("output_tokens", 0))
            if limits.max_total_tokens is not None and (acc_input + acc_output) > limits.max_total_tokens:
                raise SwarmLimitExceededError(
                    f"swarm exceeded max_total_tokens={limits.max_total_tokens}: "
                    f"used {acc_input + acc_output}",
                    kind="max_total_tokens",
                )
            swarm_run = await self._swarm_store.update_run(
                swarm_run.id, expected_version=swarm_version,
                token_usage=TokenUsage(input_tokens=acc_input, output_tokens=acc_output),
            )
            swarm_version = swarm_run.version

            # 5. write ONLY the final aggregate to the shared/parent Session.
            if spec.context_policy.write_aggregate_to_session:
                await self._write_aggregate(context, result)

            # 6. transition driving Run + SwarmRun to SUCCEEDED.
            await self._run_store.transition(
                context.run_id, RunStatus.SUCCEEDED,
                expected_version=driving_running.version, result=result,
            )
            await self._swarm_store.update_run(
                swarm_run.id, expected_version=swarm_version, status=SwarmStatus.SUCCEEDED
            )

            # 7. SwarmCompleted event (store assigns the next sequence).
            await self._event_store.append(
                stream_id=context.run_id,
                run_id=context.run_id,
                root_run_id=context.root_run_id,
                parent_run_id=context.parent_run_id,
                session_id=context.session_id,
                runnable_id=context.runnable_id,
                payload=SwarmCompleted(swarm_run_id=swarm_run.id),
            )
            return result
        except Exception as exc:
            # Best-effort cleanup: flip both records to FAILED, then re-raise.
            # The driving Run's expected version is the post-RUNNING version
            # captured in driving_running.version (no intermediate transition
            # bumps it inside the try block above); the SwarmRun's is tracked in
            # swarm_version.
            #
            # The FAILED transitions (§3.3 "Run status update") are kept
            # best-effort ONLY because we are already in the failing path:
            # letting either transition error escape would replace the ORIGINAL
            # exc (the actual cause) with a store/version error, losing the
            # cause for the caller. The warnings keep the transition failures
            # visible rather than silent.
            error_info = RunErrorInfo(
                error_type=type(exc).__name__, message=str(exc)
            )
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.FAILED,
                    expected_version=driving_running.version, error=error_info,
                )
            except Exception as transition_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition driving run %s to FAILED: %s",
                    context.run_id, transition_exc,
                )
            try:
                await self._swarm_store.update_run(
                    swarm_run.id, expected_version=swarm_version,
                    status=SwarmStatus.FAILED,
                )
            except Exception as swarm_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition swarm run %s to FAILED: %s",
                    swarm_run.id, swarm_exc,
                )
            raise

    # -- resume() -------------------------------------------------------------

    async def resume(
        self,
        swarm_run_id: str,
        spec: SwarmSpec,
        *,
        agents: "Mapping[str, AgentSpec]",
    ) -> RunResult:
        """Explicit, caller-driven re-entry (Decision #3). Re-reads the
        SwarmRun + driving RunRecord, reconstructs the SwarmExecutionContext, and
        re-enters ``strategy.run(ctx)`` -- the strategy's round loop observes
        already-SUCCEEDED tasks (via ``list_tasks(status=SUCCEEDED)``) and the
        coordinator decides whether more work is needed."""
        swarm_run = await self._swarm_store.get_run(swarm_run_id)
        if swarm_run is None:
            raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")
        driving = await self._run_store.get(swarm_run.run_id)
        if driving is None:
            raise RunNotFoundError(f"driving run not found: {swarm_run.run_id}")

        compiled_agents = await self._compile_members(spec, agents)
        parent_context = RunContext(
            run_id=driving.id,
            root_run_id=driving.root_run_id,
            parent_run_id=driving.parent_run_id,
            session_id=driving.session_id,
            runnable_id=driving.runnable_id,
            runnable_type=driving.runnable_type,
            user_id=None,
            tenant_id=None,
            workspace=None,
        )
        # Capture the versions we read so the SUCCEEDED/FAILED transitions below
        # use the exact optimistic-concurrency token the store currently holds.
        driving_version = driving.version
        swarm_version = swarm_run.version
        driving_was_terminal = driving.status in (
            RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
        )

        try:
            ctx = SwarmExecutionContext(
                spec=spec,
                swarm_run=swarm_run,
                request=driving.input,
                parent_context=parent_context,
                agent_runner=self._agent_runner,
                compiler=self._compiler,
                agents=compiled_agents,
                swarm_store=self._swarm_store,
                run_store=self._run_store,
                session_store=self._session_store,
                event_store=self._event_store,
            )
            strategy = build_strategy(spec.strategy)
            result = await strategy.run(ctx)

            if spec.context_policy.write_aggregate_to_session:
                await self._write_aggregate(parent_context, result)

            if not driving_was_terminal:
                await self._run_store.transition(
                    parent_context.run_id, RunStatus.SUCCEEDED,
                    expected_version=driving_version, result=result,
                )
            await self._swarm_store.update_run(
                swarm_run.id, expected_version=swarm_version, status=SwarmStatus.SUCCEEDED
            )

            # SwarmCompleted event -- the store assigns the next sequence
            # (events from the original run already occupy the low ones).
            await self._event_store.append(
                stream_id=parent_context.run_id,
                run_id=parent_context.run_id,
                root_run_id=parent_context.root_run_id,
                parent_run_id=parent_context.parent_run_id,
                session_id=parent_context.session_id,
                runnable_id=parent_context.runnable_id,
                payload=SwarmCompleted(swarm_run_id=swarm_run.id),
            )
            return result
        except Exception as exc:
            error_info = RunErrorInfo(
                error_type=type(exc).__name__, message=str(exc)
            )
            # §3.3 "Run status update": kept best-effort ONLY because we are
            # already in the failing path -- letting the transition error
            # escape would replace the ORIGINAL exc with a store/version error.
            # The warning keeps the failure visible rather than silent.
            if not driving_was_terminal:
                try:
                    await self._run_store.transition(
                        parent_context.run_id, RunStatus.FAILED,
                        expected_version=driving_version, error=error_info,
                    )
                except Exception as transition_exc:  # noqa: BLE001
                    _LOGGER.warning(
                        "failed to transition driving run %s to FAILED: %s",
                        parent_context.run_id, transition_exc,
                    )
            try:
                await self._swarm_store.update_run(
                    swarm_run.id, expected_version=swarm_version,
                    status=SwarmStatus.FAILED,
                )
            except Exception as swarm_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition swarm run %s to FAILED: %s",
                    swarm_run.id, swarm_exc,
                )
            raise

    # -- cancel() -------------------------------------------------------------

    async def cancel(self, swarm_run_id: str) -> None:
        """Store-level cancel (Decision #4): no live asyncio task cancellation.
        Flips the SwarmRun to CANCELLED, then enumerates CLAIMED tasks and
        transitions each task's child Run (located via ``task.active_run_id``,
        the Phase-5A handle set by strategy._run_task) to CANCELLED best-effort."""
        current = await self._swarm_store.get_run(swarm_run_id)
        if current is None:
            raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")
        await self._swarm_store.update_run(
            swarm_run_id, expected_version=current.version, status=SwarmStatus.CANCELLED
        )

        claimed = await self._swarm_store.list_tasks(
            swarm_run_id, status=SwarmTaskStatus.CLAIMED
        )
        for task in claimed:
            # Phase-5A: the child RunRecord's id is task.active_run_id (NOT
            # task.id). A CLAIMED task with active_run_id is None means the
            # strategy claimed it but crashed before set_active_run -- nothing
            # to cancel in RunStore, skip. Read the child's current version
            # rather than assuming it tracks task.version.
            #
            # best-effort per child (§3.3 "Cancel status confirm"): a child that
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
                await self._run_store.transition(
                    task.active_run_id, RunStatus.CANCELLED, expected_version=child.version
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to cancel child run %s for swarm run %s: %s",
                    task.active_run_id, swarm_run_id, exc,
                )

    # -- recover() -------------------------------------------------------------

    async def recover(self, swarm_run_id: str) -> None:
        """Best-effort restart-time scan (review doc §19.5). Walks every CLAIMED
        task whose lease has lapsed and reconciles it with the child RunRecord's
        current state:

          * Run SUCCEEDED -> ``complete_task`` (the strategy crashed between the
            child Run's SUCCEEDED transition and its ``complete_task`` call).
          * Run FAILED    -> ``fail_task`` (same gap, failed side).
          * Run RUNNING   -> leave alone. The worker may yet finish; this is the
            §19.4 "禁止仅根据时间过期就盲目重复执行副作用任务" guard.
          * active_run_id is None / Run missing / Run CANCELLED or PENDING ->
            skip here, then ``reclaim_expired_tasks`` resets them to PENDING so
            the next ``resume()`` can pick them up.

        This is NOT distributed coordination. The caller MUST ensure no live
        worker is still processing this swarm run before invoking recover();
        otherwise a slow worker will have its task snatched on a stale lease.
        On FileSwarmStore ``reclaim_expired_tasks`` is a no-op (single-process:
        nothing to reclaim at rest), so the requeue path effectively only fires
        on SqlAlchemySwarmStore -- which is the only backend that observes
        cross-process lease expiry anyway."""
        swarm_run = await self._swarm_store.get_run(swarm_run_id)
        if swarm_run is None:
            raise SwarmRunNotFoundError(f"swarm run not found: {swarm_run_id}")

        claimed = await self._swarm_store.list_tasks(
            swarm_run_id, status=SwarmTaskStatus.CLAIMED,
        )
        now = datetime.now(timezone.utc)
        for task in claimed:
            # A task whose lease hasn't lapsed is presumed still being worked --
            # leave it alone (§19.4: don't blindly re-run side-effecting tasks).
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
                    task.active_run_id, task.id, exc,
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
                        task.id, child.result, expected_version=task.version,
                    )
                elif child.status == RunStatus.FAILED and child.error is not None:
                    await self._swarm_store.fail_task(
                        task.id, child.error, expected_version=task.version,
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
                    task.id, task.active_run_id, child.status, exc,
                )

        # Reset everything still CLAIMED with an expired lease (cases above we
        # didn't reconcile: no active_run_id, missing Run, non-terminal Run
        # state) to PENDING so resume() can re-drive them. On FileSwarmStore
        # this is a documented no-op.
        await self._swarm_store.reclaim_expired_tasks(swarm_run_id)

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

    async def _write_aggregate(
        self, context: RunContext, result: RunResult
    ) -> None:
        """Append the single aggregate assistant message to the shared/parent
        Session. G6: sequence is assigned by the SessionStore itself, not
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
