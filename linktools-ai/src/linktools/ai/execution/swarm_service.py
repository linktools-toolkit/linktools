#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SwarmExecutionService: the task_graph orchestration layer.

Owns the parent RunRecord lifecycle (PENDING -> RUNNING -> terminal), the
atomic TaskPlan + executions creation, the mutating-tool precheck, the immutable
session snapshot, the ControlGate (timeout/token/cost/cancel), and the
NodeRunner that drives each node's child agent via AgentExecutionService. It
never touches audit business objects or mutates ``task_plan``; TaskGraphEngine
owns graph progress and TaskStore owns node state."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Mapping
from uuid import uuid4

from linktools.core import environ

from ..agent.spec import AgentSpec
from ..errors import (
    InvalidSpecError,
    PrincipalAccessDeniedError,
    SwarmLimitExceededError,
)
from ..execution.domain import (
    RunDefinition,
    RunError,
    RunKind,
    RunRecord,
    RunStatus,
    RunUsage,
    RunnableType,
    child_run_id,
)
from ..governance.authorization import AuthorizationPolicy, ExecutionAction
from ..governance.identity import PrincipalContext
from ..json import JsonValue, canonical_json_bytes
from ..observability.events.payloads import (
    SwarmCancelled,
    SwarmCompleted,
    SwarmFailed,
    SwarmLimitReached,
    SwarmStarted,
    SwarmStepCancelled,
    SwarmStepCompleted,
    SwarmStepFailed,
    SwarmStepSkipped,
    SwarmStepStarted,
)
from ..tasks.models import TaskExecution, TaskNode, TaskPlan, TaskStatus, TaskUsage
from ..tasks.store import TaskStore
from ..tasks.swarm.aggregation import collect
from ..tasks.swarm.codec import decode_swarm_spec, encode_swarm_spec
from ..tasks.swarm.engine import (
    ControlGate,
    NodeRunRequest,
    NodeRunResult,
    NodeRunner,
    TaskGraphEngine,
)
from ..tasks.swarm.validation import validate_plan_against_swarm
from ..tasks.swarm.models import SwarmCompleted as SwarmCompletedOutcome
from ..tasks.swarm.models import SwarmFailed as SwarmFailedOutcome
from ..tasks.swarm.models import SwarmRunView
from ..tasks.swarm.spec import SwarmSpec
from .commands import (
    AbortExecution,
    AcknowledgeCancellation,
    ClaimExecution,
    CompleteExecution,
    FailExecution,
    HeartbeatExecution,
    RequestCancellation,
    StartExecution,
)
from .live_events import RunLiveEventSink
from .service import ChildRunResult, ExecutionService
from .snapshots import AgentSnapshotData
from .store import ExecutionStore

logger = environ.get_logger("ai.execution.swarm_service")

_LEASE_DURATION = timedelta(minutes=10)


def _initial_executions(task_plan: TaskPlan) -> "tuple[TaskExecution, ...]":
    return tuple(
        TaskExecution(
            id=f"task-{task_plan.id}-{node.id}",
            plan_id=task_plan.id,
            node_id=node.id,
            status=TaskStatus.READY,
        )
        for node in task_plan.nodes
    )


def _swarm_definition(spec: SwarmSpec, task_plan: TaskPlan) -> RunDefinition:
    value = encode_swarm_spec(spec)
    return RunDefinition(
        spec.id,
        RunnableType.TASK,
        "swarm-spec.v1",
        value,
        sha256(canonical_json_bytes(value)).hexdigest(),
    )


class SwarmExecutionService:
    """Drives a task_graph swarm run: validate, persist, schedule, converge."""

    def __init__(
        self,
        store: ExecutionStore,
        tasks: TaskStore,
        agent_execution: ExecutionService,
        *,
        authorization: AuthorizationPolicy,
        live_events: RunLiveEventSink,
        agent_provider: "object | None" = None,
    ) -> None:
        self._store = store
        self._tasks = tasks
        self._agent_execution = agent_execution
        self._authorization = authorization
        self._live_events = live_events
        self._agent_provider = agent_provider

    async def run_swarm(
        self,
        spec: SwarmSpec,
        task_plan: TaskPlan,
        *,
        principal: PrincipalContext,
        session_id: "str | None" = None,
        execution_id: "str | None" = None,
    ) -> "SwarmCompletedOutcome | SwarmFailedOutcome":
        if spec.strategy.kind != "task_graph":
            raise InvalidSpecError(
                f"swarm {spec.id!r}: SwarmExecutionService only runs task_graph"
            )
        if not isinstance(principal, PrincipalContext):
            raise PrincipalAccessDeniedError("a valid PrincipalContext is required")
        if len(task_plan.nodes) > spec.limits.max_tasks:
            raise InvalidSpecError(
                f"task_plan has {len(task_plan.nodes)} nodes; limit is {spec.limits.max_tasks}"
            )
        self._authorization.assert_execution_access(
            principal=principal,
            tenant_id=principal.tenant_id,
            user_id=principal.user_id,
            action=ExecutionAction.SWARM,
        )
        # Full validation before any persistence: agent allow-set + mutating
        # tool precheck both run before start_run/create_plan.
        await self._validate_agents(spec, task_plan, principal)
        parent_run_id = execution_id or uuid4().hex
        resolved_session = session_id or uuid4().hex
        await self._store.create_session(
            session_id=resolved_session,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
        snapshot = await self._load_session_snapshot(resolved_session)
        # Persist the parent RunRecord BEFORE the TaskPlan: the encoded swarm
        # spec in its definition is what recover_swarm decodes, and a crash here
        # leaves nothing half-baked (start_run and create_plan are each atomic).
        record = await self._store.start_run(
            StartExecution(
                parent_run_id,
                resolved_session,
                RunKind.TASK,
                _swarm_definition(spec, task_plan),
                {"plan_id": task_plan.id},
                root_execution_id=parent_run_id,
                parent_execution_id=None,
            )
        )
        owner = f"swarm:{parent_run_id}"
        # Claim the parent run (PENDING -> RUNNING) so it carries a real lease
        # the terminal converge can fence against; an unclaimed PENDING run would
        # reject complete/fail on owner+fence and on the PENDING->terminal jump.
        claimed = await self._store.claim_run(
            ClaimExecution(parent_run_id, owner, datetime.now(timezone.utc), _LEASE_DURATION)
        )
        fence = claimed.lease.fence
        logger.debug(
            "swarm %s parent run %s claimed (owner=%s fence=%s nodes=%s)",
            spec.id, parent_run_id, owner, fence, len(task_plan.nodes),
        )
        executions = _initial_executions(task_plan)
        try:
            await self._tasks.create_plan(task_plan, executions)
        except Exception as plan_exc:
            logger.warning(
                "swarm %s create_plan failed for run %s: %s",
                spec.id, parent_run_id, plan_exc,
            )
            await self._abort_parent(parent_run_id, owner, fence, plan_exc)
            raise
        gate = _SwarmControlGate(
            limits=spec.limits,
            parent_run_id=parent_run_id,
            store=self._store,
            owner=owner,
            fence=fence,
            lease_duration=_LEASE_DURATION,
        )
        await self._publish(
            SwarmStarted(swarm_run_id=parent_run_id, swarm_id=spec.id)
        )
        try:
            usage = await self._drive(
                spec, task_plan, parent_run_id, owner, gate, principal,
                resolved_session, snapshot,
            )
        except SwarmLimitExceededError as exc:
            logger.info(
                "swarm %s run %s hit %s limit", spec.id, parent_run_id, exc.kind
            )
            await self._abort_inflight(task_plan)
            await self._publish(
                SwarmLimitReached(swarm_run_id=parent_run_id, kind=exc.kind)
            )
            if exc.kind != "parent_lease_lost":
                await self._fail_parent(
                    parent_run_id, owner, fence, exc.kind, str(exc)
                )
            await self._publish(SwarmFailed(swarm_run_id=parent_run_id, error=str(exc)))
            return SwarmFailedOutcome(error=RunError(exc.kind, str(exc)))
        except asyncio.CancelledError:
            logger.info("swarm %s run %s cancelled", spec.id, parent_run_id)
            await self._converge_cancel(task_plan, parent_run_id, owner, fence)
            await self._publish(SwarmCancelled(swarm_run_id=parent_run_id))
            raise
        except Exception as exc:
            logger.warning(
                "swarm %s run %s failed: %s", spec.id, parent_run_id, exc,
            )
            await self._fail_parent(
                parent_run_id, owner, fence, type(exc).__name__, str(exc)
            )
            await self._publish(SwarmFailed(swarm_run_id=parent_run_id, error=str(exc)))
            return SwarmFailedOutcome(error=RunError(type(exc).__name__, str(exc)))
        if gate.cancel_requested:
            logger.info(
                "swarm %s run %s cancelled mid-run", spec.id, parent_run_id
            )
            await self._converge_cancel(task_plan, parent_run_id, owner, fence)
            await self._publish(SwarmCancelled(swarm_run_id=parent_run_id))
            return SwarmFailedOutcome(
                error=RunError("swarm_cancelled", "swarm was cancelled mid-run")
            )
        await self._complete_parent(parent_run_id, owner, fence)
        nodes = await self._collect_nodes(task_plan)
        projection = collect(task_plan.id, nodes)
        await self._publish(SwarmCompleted(swarm_run_id=parent_run_id))
        return SwarmCompletedOutcome(collect=projection, usage=usage)

    async def recover_swarm(
        self,
        execution_id: str,
        *,
        principal: PrincipalContext,
    ) -> "SwarmCompletedOutcome | SwarmFailedOutcome":
        record = await self._store.get_run(execution_id)
        if record is None:
            raise KeyError(execution_id)
        self._authorize(principal, record, ExecutionAction.RESUME)
        spec = decode_swarm_spec(record.definition.spec)
        task_plan = await self._tasks.get_plan(_plan_id_from_run(record))
        if task_plan is None:
            return SwarmFailedOutcome(
                error=RunError("plan_integrity", "task graph record not found")
            )
        owner = f"swarm:{execution_id}"
        snapshot: "tuple[object, ...]" = ()
        gate = _SwarmControlGate(
            limits=spec.limits,
            parent_run_id=execution_id,
            store=self._store,
            owner=owner,
            fence=record.lease.fence,
            lease_duration=_LEASE_DURATION,
        )
        logger.debug("recovering swarm run %s", execution_id)
        await self._reconcile_inflight(task_plan, owner)
        usage = await self._drive(
            spec, task_plan, execution_id, owner, gate, principal,
            record.session_id, snapshot,
        )
        nodes = await self._collect_nodes(task_plan)
        projection = collect(task_plan.id, nodes)
        return SwarmCompletedOutcome(collect=projection, usage=usage)

    async def inspect_swarm(
        self,
        execution_id: str,
        *,
        principal: PrincipalContext,
    ) -> SwarmRunView:
        record = await self._store.get_run(execution_id)
        if record is None:
            raise KeyError(execution_id)
        self._authorize(principal, record, ExecutionAction.INSPECT)
        task_plan = await self._tasks.get_plan(_plan_id_from_run(record))
        if task_plan is None:
            return SwarmRunView(
                plan_id="",
                parent_run_id=execution_id,
                status=record.status.value,
                error=record.error,
                nodes=(),
                status_counts={},
            )
        executions = {
            e.node_id: e for e in await self._tasks.list_executions(task_plan.id)
        }
        node_views = tuple(self._node_view(node, executions) for node in task_plan.nodes)
        projection = collect(task_plan.id, node_views)
        return SwarmRunView(
            plan_id=task_plan.id,
            parent_run_id=execution_id,
            status=record.status.value,
            error=record.error,
            nodes=tuple(projection["nodes"].values()),  # type: ignore[arg-type]
            status_counts=projection["status_counts"],  # type: ignore[arg-type]
        )

    async def cancel_swarm(
        self, execution_id: str, *, principal: PrincipalContext
    ) -> None:
        """Externally request cancellation: persist CANCELLING on the parent run
        so the scheduler's gate observes it on its next async check()."""
        record = await self._store.get_run(execution_id)
        if record is None:
            raise KeyError(execution_id)
        self._authorize(principal, record, ExecutionAction.CANCEL)
        if record.status in {RunStatus.PENDING, RunStatus.RUNNING}:
            logger.info(
                "cancelling swarm run %s (current=%s)", execution_id, record.status.value
            )
            await self._store.request_cancel(
                RequestCancellation(
                    execution_id,
                    record.lease.owner or "swarm",
                    record.lease.fence,
                    datetime.now(timezone.utc),
                )
            )

    async def _drive(
        self,
        spec: SwarmSpec,
        task_plan: TaskPlan,
        parent_run_id: str,
        owner: str,
        gate: "_SwarmControlGate",
        principal: PrincipalContext,
        session_id: str,
        snapshot: "tuple[object, ...]",
    ) -> TaskUsage:
        runner = _ChildNodeRunner(
            agent_execution=self._agent_execution,
            agent_provider=self._agent_provider,
            principal=principal,
            session_id=session_id,
            parent_run_id=parent_run_id,
            root_execution_id=parent_run_id,
            message_history=snapshot,
            store=self._store,
            live_events=self._live_events,
        )
        engine = TaskGraphEngine(
            store=self._tasks,
            runner=runner,
            gate=gate,
            limits=spec.limits,
            owner=owner,
            parent_run_id=parent_run_id,
            on_skip=lambda nid, blk: self._publish_skip(parent_run_id, nid, blk),
            on_node_terminal=lambda nid, outcome: self._publish_node_terminal(
                parent_run_id, nid, outcome
            ),
        )
        return await engine.execute(task_plan)

    async def _validate_agents(
        self, spec: SwarmSpec, task_plan: TaskPlan, principal: PrincipalContext
    ) -> None:
        if self._agent_provider is None:
            raise InvalidSpecError(
                "no agent provider configured; cannot resolve or validate node agents"
            )
        validate_plan_against_swarm(
            task_plan,
            allowed_agent_ids={agent.agent_id for agent in spec.agents},
        )
        for node in task_plan.nodes:
            agent_spec = await self._agent_provider.get(node.payload.agent_id)  # type: ignore[attr-defined]
            await self._reject_mutating_tools(node, agent_spec)

    async def _reject_mutating_tools(
        self, node: TaskNode, agent_spec: AgentSpec
    ) -> None:
        assembly = await self._assemble_for_check(agent_spec)
        if assembly is None:
            return
        for tool in getattr(assembly, "tools", ()):
            if tool.descriptor.mutating:
                raise InvalidSpecError(
                    f"node {node.id!r}: task_graph forbids mutating tool "
                    f"{tool.descriptor.name!r} (mutating_tool_not_allowed)"
                )

    async def _assemble_for_check(self, agent_spec: AgentSpec):
        assembler = getattr(self._agent_execution, "_assembler", None)
        if assembler is None:
            return None
        from ..agent.assembly.provider import AgentFeatureContext

        ctx = AgentFeatureContext(
            agent_id=agent_spec.id,
            execution_id="precheck",
            root_execution_id="precheck",
            parent_execution_id=None,
            session_id="precheck",
            tenant_id=None,
            user_id=None,
            workspace=None,
            sandbox=getattr(self._agent_execution, "_sandbox", None),
        )
        return await assembler.assemble(agent_spec, ctx)

    async def _load_session_snapshot(self, session_id: str) -> "tuple[object, ...]":
        try:
            return await self._store.load_session_context(session_id)
        except Exception:
            return ()

    async def _collect_nodes(self, task_plan: TaskPlan) -> "tuple[dict, ...]":
        executions = {
            e.node_id: e for e in await self._tasks.list_executions(task_plan.id)
        }
        return tuple(self._node_view(node, executions) for node in task_plan.nodes)

    def _node_view(
        self, node: TaskNode, executions: "dict[str, TaskExecution]"
    ) -> dict:
        execution = executions.get(node.id)
        return {
            "node_id": node.id,
            "agent_id": node.payload.agent_id,
            "status": execution.status if execution else TaskStatus.SKIPPED,
            "output": execution.result if execution else None,
            "error": execution.error if execution else None,
            "blocked_by": execution.blocked_by if execution else (),
            "reason": execution.terminal_reason if execution else None,
            "attempts": execution.attempt if execution else 0,
            "child_run_id": execution.active_run_id if execution else None,
            "usage": {
                "input_tokens": execution.usage.input_tokens if execution else 0,
                "output_tokens": execution.usage.output_tokens if execution else 0,
                "total_cost": execution.usage.total_cost if execution else None,
            },
        }

    async def _abort_inflight(self, task_plan: TaskPlan) -> None:
        """On a run-level abort (limit/timeout), cancel CLAIMED nodes' children
        and mark READY nodes CANCELLED so no in-flight child is orphaned."""
        for execution in await self._tasks.list_executions(task_plan.id):
            if execution.status is TaskStatus.READY:
                await self._tasks.cancel_ready(execution.id, reason="swarm_aborted")
            elif execution.status is TaskStatus.CLAIMED and execution.active_run_id:
                child = await self._store.get_run(execution.active_run_id)
                if child is not None and child.status in {
                    RunStatus.RUNNING,
                    RunStatus.PENDING,
                }:
                    await self._store.request_cancel(
                        RequestCancellation(
                            child.id,
                            child.lease.owner or "swarm",
                            child.lease.fence,
                            datetime.now(timezone.utc),
                        )
                    )

    async def _converge_cancel(
        self,
        task_plan: TaskPlan,
        parent_run_id: str,
        owner: str,
        fence: int,
    ) -> None:
        await self._abort_inflight(task_plan)
        latest = await self._store.get_run(parent_run_id)
        if latest is not None and latest.status is RunStatus.RUNNING:
            await self._store.request_cancel(
                RequestCancellation(parent_run_id, owner, fence, datetime.now(timezone.utc))
            )
            latest = await self._store.get_run(parent_run_id)
        if latest is not None and latest.status is RunStatus.CANCELLING:
            await self._store.acknowledge_cancel(
                AcknowledgeCancellation(
                    parent_run_id,
                    owner,
                    fence,
                    AgentSnapshotData((), None, RunUsage(), 0),
                )
            )

    async def _complete_parent(
        self, parent_run_id: str, owner: str, fence: int
    ) -> None:
        await self._store.complete_run(
            CompleteExecution(
                parent_run_id, owner, fence,
                AgentSnapshotData((), None, RunUsage(), 0),
            )
        )

    async def _fail_parent(
        self,
        parent_run_id: str,
        owner: str,
        fence: int,
        error_type: str,
        message: str,
    ) -> None:
        await self._store.fail_run(
            FailExecution(
                parent_run_id, owner, fence,
                AgentSnapshotData((), None, RunUsage(), 0),
                RunError(error_type, message),
            )
        )

    async def _abort_parent(
        self,
        parent_run_id: str,
        owner: str,
        fence: int,
        cause: BaseException,
    ) -> None:
        """Abort a claimed parent run with no snapshot (the engine never ran),
        so a create_plan failure leaves the run FAILED rather than stranded."""
        await self._store.abort_run(
            AbortExecution(
                parent_run_id, owner, fence,
                RunError("plan_create_failed", str(cause)),
                0,
            )
        )

    async def _reconcile_inflight(self, task_plan: TaskPlan, owner: str) -> None:
        for execution in await self._tasks.list_executions(task_plan.id):
            if execution.status is TaskStatus.CLAIMED:
                await self._reconcile_claimed(task_plan, execution, owner)

    async def _reconcile_claimed(
        self,
        task_plan: TaskPlan,
        execution: TaskExecution,
        owner: str,
    ) -> None:
        if execution.active_run_id is None:
            await self._tasks.fail(
                execution.id, owner=owner, fence=execution.fence,
                error=RunError("orphaned_before_child_start", "claim without bind"),
                usage=TaskUsage(),
            )
            return
        child = await self._store.get_run(execution.active_run_id)
        if child is None:
            await self._tasks.fail(
                execution.id, owner=owner, fence=execution.fence,
                error=RunError("orphaned_before_child_start", "child run missing"),
                usage=TaskUsage(),
            )
            return
        if child.status is RunStatus.COMPLETED:
            output, recovered_usage = await self._recover_child_artifacts(child)
            await self._tasks.complete(
                execution.id, owner=owner, fence=execution.fence,
                result=output, usage=recovered_usage,
            )
        elif child.status is RunStatus.FAILED:
            _, recovered_usage = await self._recover_child_artifacts(child)
            await self._tasks.fail(
                execution.id, owner=owner, fence=execution.fence,
                error=child.error or RunError("child_failed", "child run failed"),
                usage=recovered_usage,
            )
        elif child.status is RunStatus.CANCELLED:
            _, recovered_usage = await self._recover_child_artifacts(child)
            await self._tasks.cancel_claimed(
                execution.id, owner=owner, fence=execution.fence,
                reason="child cancelled", usage=recovered_usage,
            )
        elif child.status is RunStatus.PAUSED:
            await self._tasks.fail(
                execution.id, owner=owner, fence=execution.fence,
                error=RunError("approval_not_supported", "child paused"),
                usage=TaskUsage(),
            )
        else:
            await self._store.request_cancel(
                RequestCancellation(
                    child.id, child.lease.owner or "swarm", child.lease.fence,
                    datetime.now(timezone.utc),
                )
            )
            await self._tasks.fail(
                execution.id, owner=owner, fence=execution.fence,
                error=RunError("interrupted_execution", "child still running on recover"),
                usage=TaskUsage(),
            )

    async def _recover_child_artifacts(
        self, child: RunRecord
    ) -> "tuple[object | None, TaskUsage]":
        """Best-effort recovery of a terminal child's structured output and real
        usage from its persisted snapshot. Output may be None if no snapshot
        exists (e.g. crash before first commit); usage zeros out but is still
        the most accurate recoverable figure."""
        snapshot = await self._store.get_snapshot(child.id)
        if snapshot is None:
            return None, TaskUsage()
        from decimal import Decimal

        cost: "Decimal | None" = None
        ru = snapshot.usage
        return snapshot.final_output, TaskUsage(
            input_tokens=ru.input_tokens,
            output_tokens=ru.output_tokens,
            total_cost=cost,
        )

    def _authorize(
        self, principal: PrincipalContext, record: RunRecord, action: ExecutionAction
    ) -> None:
        self._authorization.assert_execution_access(
            principal=principal,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            action=action,
        )

    async def _publish(self, event: object) -> None:
        try:
            await self._live_events.publish(event)
        except Exception as exc:
            if environ.debug:
                logger.debug("swarm event publish failed: %s: %s", type(exc).__name__, exc)

    async def _publish_skip(
        self, parent_run_id: str, node_id: str, blocked_by: "tuple[str, ...]"
    ) -> None:
        await self._publish(
            SwarmStepSkipped(
                swarm_run_id=parent_run_id, task_id=node_id, blocked_by=blocked_by
            )
        )

    async def _publish_node_terminal(
        self, parent_run_id: str, node_id: str, outcome: NodeRunResult
    ) -> None:
        if outcome.status is TaskStatus.COMPLETED:
            await self._publish(
                SwarmStepCompleted(swarm_run_id=parent_run_id, task_id=node_id)
            )
        elif outcome.status is TaskStatus.CANCELLED:
            await self._publish(
                SwarmStepCancelled(swarm_run_id=parent_run_id, task_id=node_id)
            )
        else:
            await self._publish(
                SwarmStepFailed(
                    swarm_run_id=parent_run_id,
                    task_id=node_id,
                    error_message=(
                        outcome.error.message if outcome.error else "failed"
                    ),
                )
            )


def _plan_id_from_run(record: RunRecord) -> str:
    raw = record.input if isinstance(record.input, dict) else {}
    value = raw.get("plan_id")
    return str(value) if isinstance(value, str) else ""


class _SwarmControlGate(ControlGate):
    """Monotonic timeout, token/cost caps, cancel-request propagation, and
    parent-lease liveness for one swarm run. ``check`` is async: it reads the
    parent RunRecord each pass so an external ``cancel_swarm`` (persisted
    CANCELLING) and a lost/reclaimed parent lease both propagate into the
    scheduler without an in-process channel. The parent lease is renewed
    periodically so long-running swarms in a multi-process deployment do not
    silently lose ownership."""

    _HEARTBEAT_INTERVAL = timedelta(minutes=3)

    def __init__(
        self,
        *,
        limits: "object",
        parent_run_id: str,
        store: ExecutionStore,
        owner: str,
        fence: int,
        lease_duration: timedelta,
    ) -> None:
        self._limits = limits
        self._parent_run_id = parent_run_id
        self._store = store
        self._owner = owner
        self._fence = fence
        self._lease_duration = lease_duration
        self._start: "float | None" = None
        self._last_heartbeat: "datetime | None" = None
        self._accumulated = TaskUsage()
        self._cancel = False

    def record_usage(self, usage: TaskUsage) -> None:
        self._accumulated = self._accumulated.add(usage)

    @property
    def cancel_requested(self) -> bool:
        return self._cancel

    async def _assert_and_renew_lease(self, record: RunRecord) -> None:
        """Verify the parent lease still belongs to this scheduler; renew it
        periodically so a long-running swarm does not silently expire. If the
        lease was reclaimed by another worker (different owner or fence), the
        scheduler must stop — continuing would write under a stale identity."""
        if record.lease.owner != self._owner or record.lease.fence != self._fence:
            raise SwarmLimitExceededError(
                "parent lease lost or reclaimed by another worker",
                kind="parent_lease_lost",
            )
        now_dt = datetime.now(timezone.utc)
        if self._last_heartbeat is None:
            self._last_heartbeat = now_dt
            return
        if now_dt - self._last_heartbeat < self._HEARTBEAT_INTERVAL:
            return
        try:
            renewed = await self._store.heartbeat_run(
                HeartbeatExecution(
                    self._parent_run_id,
                    self._owner,
                    self._fence,
                    now_dt,
                    self._lease_duration,
                )
            )
            self._last_heartbeat = now_dt
            if renewed.status is RunStatus.CANCELLING:
                self._cancel = True
        except Exception as exc:
            raise SwarmLimitExceededError(
                f"parent lease renewal failed: {exc}", kind="parent_lease_lost"
            ) from exc

    async def check(self, *, now: float) -> None:
        if self._start is None:
            self._start = now
        record = await self._store.get_run(self._parent_run_id)
        if record is None:
            raise SwarmLimitExceededError("parent run vanished", kind="parent_lost")
        if record.status in {
            RunStatus.CANCELLING,
            RunStatus.CANCELLED,
        }:
            self._cancel = True
        await self._assert_and_renew_lease(record)
        timeout = getattr(self._limits, "timeout_seconds", None)
        if timeout is not None and now - self._start > timeout:
            raise SwarmLimitExceededError("swarm timeout", kind="timeout")
        max_tokens = getattr(self._limits, "max_total_tokens", None)
        if max_tokens is not None:
            spent = self._accumulated.input_tokens + self._accumulated.output_tokens
            if spent > max_tokens:
                raise SwarmLimitExceededError(
                    "token limit exceeded", kind="max_total_tokens"
                )
        max_cost = getattr(self._limits, "max_total_cost", None)
        if max_cost is not None:
            spent_any = (
                self._accumulated.input_tokens > 0
                or self._accumulated.output_tokens > 0
            )
            # Before any node reports, an unknown cost is expected (nothing to
            # measure yet); cost_usage_unavailable fires only once a node HAS
            # reported usage but could not supply a cost.
            if spent_any and self._accumulated.total_cost is None:
                raise SwarmLimitExceededError(
                    "cost configured but usage unavailable", kind="cost_usage_unavailable"
                )
            if self._accumulated.total_cost is not None and self._accumulated.total_cost > max_cost:
                raise SwarmLimitExceededError(
                    "cost limit exceeded", kind="max_total_cost"
                )


@dataclass(frozen=True, slots=True)
class _ChildNodeRunner(NodeRunner):
    """NodeRunner that maps each node to an AgentExecutionService.run_child call,
    threading the dependency view into the child's AgentInput.metadata."""

    agent_execution: ExecutionService
    agent_provider: "object | None"
    principal: PrincipalContext
    session_id: str
    parent_run_id: str
    root_execution_id: str
    message_history: "tuple[object, ...]"
    store: ExecutionStore
    live_events: RunLiveEventSink

    async def run(self, request: NodeRunRequest) -> NodeRunResult:
        node = request.node
        agent_spec = await self._resolve(node.payload.agent_id)
        child_id = child_run_id(self.parent_run_id, node.id)
        deps_metadata = _dependency_metadata(self.parent_run_id, request)
        await self._publish(
            SwarmStepStarted(
                swarm_run_id=self.parent_run_id, task_id=node.id, child_run_id=child_id
            )
        )
        try:
            result = await self.agent_execution.run_child(
                agent_spec,
                node.payload.prompt,
                principal=self.principal,
                session_id=self.session_id,
                execution_id=child_id,
                root_execution_id=self.root_execution_id,
                parent_execution_id=self.parent_run_id,
                message_history=self.message_history,
                metadata={"task_graph": deps_metadata},
            )
        except Exception as exc:
            return NodeRunResult(
                status=TaskStatus.FAILED,
                error=RunError("node_runner_error", str(exc)),
                usage=TaskUsage(),
            )
        # Step-terminal events are published from the engine's on_node_terminal
        # hook AFTER the TaskExecution is persisted, not here (persist-before-event).
        if result.status is RunStatus.COMPLETED:
            return NodeRunResult(
                status=TaskStatus.COMPLETED, result=result.output, usage=result.usage
            )
        if result.status is RunStatus.CANCELLED:
            return NodeRunResult(
                status=TaskStatus.CANCELLED, error=result.error, usage=result.usage
            )
        return NodeRunResult(
            status=TaskStatus.FAILED,
            error=result.error or RunError("node_failed", "child failed"),
            usage=result.usage,
        )

    async def _resolve(self, agent_id: str) -> AgentSpec:
        if self.agent_provider is None:
            raise InvalidSpecError("no agent provider configured")
        return await self.agent_provider.get(agent_id)  # type: ignore[attr-defined]

    async def _publish(self, event: object) -> None:
        try:
            await self.live_events.publish(event)
        except Exception:
            pass


def _dependency_metadata(parent_run_id: str, request: NodeRunRequest) -> dict:
    deps: "dict[str, object]" = {}
    for dep_exec in request.dependencies:
        deps[dep_exec.node_id] = {
            "status": dep_exec.status.value,
            "output": dep_exec.result,
            "error": dep_exec.error.message if dep_exec.error else None,
            "blocked_by": list(dep_exec.blocked_by),
        }
    return {
        "swarm_run_id": parent_run_id,
        "plan_id": request.execution.plan_id,
        "node_id": request.node.id,
        "dependencies": deps,
    }


__all__ = ["SwarmExecutionService"]
