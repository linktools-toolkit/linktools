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
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Mapping
from uuid import uuid4

from linktools.core import environ

from ..agent.spec import AgentSpec
from ..errors import (
    ChildCancelNotConvergedError,
    ChildExecutionPlatformError,
    ChildRunMissingError,
    ChildSnapshotError,
    InvalidSpecError,
    PrincipalAccessDeniedError,
    RecoveryConflictError,
    RuntimeInitializationError,
    StorageConflictError,
    StorageError,
    SwarmLimitExceededError,
    TaskGraphInvariantError,
)
from ..execution.domain import (
    MessageCaptureState,
    RunDefinition,
    RunError,
    RunKind,
    RunRecord,
    RunStatus,
    RunUsage,
    RunnableType,
    sanitize_run_error,
)
from ..governance.authorization import AuthorizationPolicy, ExecutionAction
from ..governance.identity import PrincipalContext
from ..json import JsonValue, canonical_json_bytes, normalize_json
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
from ..tasks.codec import encode_plan
from ..tasks.models import (
    TaskExecution,
    TaskNode,
    TaskPlan,
    TaskStatus,
    TaskUsage,
    UsageAccumulator,
)
from ..tasks.store import TaskStore
from ..storage.coordination.lease import is_expired
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
from .identifiers import child_run_id, task_execution_id
from .live_events import RunLiveEventSink
from .service import ChildRunResult, ExecutionService, PreparedAgentExecution
from .snapshots import AgentSnapshotData
from .store import ExecutionStore

logger = environ.get_logger("ai.execution.swarm_service")

_LEASE_DURATION = timedelta(minutes=10)
RECOVERY_CHILD_CANCEL_TIMEOUT = 30.0
RECOVERY_CHILD_POLL_INTERVAL = 1.0


@dataclass(frozen=True, slots=True)
class ValidatedSwarmRun:
    record: RunRecord
    task_plan: TaskPlan
    spec: SwarmSpec
    snapshot: "tuple[object, ...]"
    executions: "tuple[TaskExecution, ...]"
    prepared_agents: "Mapping[str, PreparedAgentExecution]"


@dataclass(slots=True)
class _RecoveryLease:
    record: RunRecord
    next_heartbeat_at: float


def _initial_executions(task_plan: TaskPlan) -> "tuple[TaskExecution, ...]":
    return tuple(
        TaskExecution(
            id=task_execution_id(task_plan.id, node.id),
            plan_id=task_plan.id,
            node_id=node.id,
            status=TaskStatus.READY,
        )
        for node in task_plan.nodes
    )


def _swarm_definition(
    spec: SwarmSpec,
    task_plan: TaskPlan,
    *,
    snapshot_hash: str,
    agent_fingerprints: "Mapping[str, str]",
    deadline_at: "datetime | None",
) -> RunDefinition:
    plan_hash = sha256(canonical_json_bytes(encode_plan(task_plan))).hexdigest()
    value = {
        "swarm_spec": encode_swarm_spec(spec),
        "task_plan_id": task_plan.id,
        "task_plan_hash": plan_hash,
        "session_snapshot_hash": snapshot_hash,
        "agent_fingerprints": dict(agent_fingerprints),
        "deadline_at": deadline_at.isoformat() if deadline_at is not None else None,
    }
    return RunDefinition(
        spec.id,
        RunnableType.TASK,
        "swarm-task-graph.v1",
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
        parent_run_id = execution_id or uuid4().hex
        resolved_session = session_id or uuid4().hex
        prepared_agents, agent_fingerprints = await self._validate_agents(
            spec, task_plan, principal, session_id=resolved_session
        )
        await self._store.create_session(
            session_id=resolved_session,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
        snapshot = await self._load_session_snapshot(resolved_session)
        snapshot_hash = sha256(canonical_json_bytes(list(snapshot))).hexdigest()
        deadline_at = (
            datetime.now(timezone.utc) + timedelta(seconds=spec.limits.timeout_seconds)
            if spec.limits.timeout_seconds is not None
            else None
        )
        record = await self._store.start_run(
            StartExecution(
                parent_run_id,
                resolved_session,
                RunKind.TASK,
                _swarm_definition(
                    spec,
                    task_plan,
                    snapshot_hash=snapshot_hash,
                    agent_fingerprints=agent_fingerprints,
                    deadline_at=deadline_at,
                ),
                {
                    "task_plan_id": task_plan.id,
                    "session_snapshot": list(snapshot),
                    "session_snapshot_hash": snapshot_hash,
                },
                root_execution_id=parent_run_id,
                parent_execution_id=None,
            )
        )
        owner = f"swarm:{parent_run_id}"
        executions = _initial_executions(task_plan)
        try:
            await self._tasks.create_plan(task_plan, executions)
        except Exception as plan_exc:
            logger.warning(
                "swarm %s create_plan failed for run %s: %s",
                spec.id, parent_run_id, type(plan_exc).__name__,
            )
            await self._abort_parent(parent_run_id, owner, 0, plan_exc)
            raise
        # Parent ownership begins only after the task graph is durable.
        claimed = await self._store.claim_run(
            ClaimExecution(parent_run_id, owner, datetime.now(timezone.utc), _LEASE_DURATION)
        )
        fence = claimed.lease.fence
        logger.info(
            "swarm %s run %s claimed owner=%s fence=%s nodes=%s",
            spec.id, parent_run_id, owner, fence, len(task_plan.nodes),
        )
        gate = _SwarmControlGate(
            limits=spec.limits,
            parent_run_id=parent_run_id,
            store=self._store,
            owner=owner,
            fence=fence,
            lease_duration=_LEASE_DURATION,
            deadline_at=deadline_at,
        )
        await self._publish(
            SwarmStarted(swarm_run_id=parent_run_id, swarm_id=spec.id)
        )
        try:
            usage = await self._drive(
                spec, task_plan, parent_run_id, owner, gate, principal,
                resolved_session, snapshot, prepared_agents,
            )
        except SwarmLimitExceededError as exc:
            logger.info(
                "swarm %s run %s hit %s limit", spec.id, parent_run_id, exc.kind
            )
            if exc.kind == "parent_cancelled":
                await self._converge_cancel(task_plan, parent_run_id, owner, fence)
                await self._publish(SwarmCancelled(swarm_run_id=parent_run_id))
                return SwarmFailedOutcome(
                    error=RunError("swarm_cancelled", "swarm was cancelled")
                )
            await self._publish(
                SwarmLimitReached(swarm_run_id=parent_run_id, kind=exc.kind)
            )
            if exc.kind != "parent_lease_lost":
                await self._fail_parent(
                    parent_run_id,
                    owner,
                    fence,
                    exc.kind,
                    exc.kind,
                    task_plan=task_plan,
                )
            safe_error = sanitize_run_error(exc)
            await self._publish(SwarmFailed(swarm_run_id=parent_run_id, error=safe_error.message))
            return SwarmFailedOutcome(error=safe_error)
        except asyncio.CancelledError:
            logger.info("swarm %s run %s cancelled", spec.id, parent_run_id)
            await self._converge_cancel(task_plan, parent_run_id, owner, fence)
            await self._publish(SwarmCancelled(swarm_run_id=parent_run_id))
            raise
        except Exception as exc:
            logger.warning(
                "swarm %s run %s failed: %s", spec.id, parent_run_id, type(exc).__name__,
            )
            safe_error = sanitize_run_error(exc)
            await self._fail_parent(
                parent_run_id,
                owner,
                fence,
                safe_error.error_type,
                safe_error.message,
                task_plan=task_plan,
            )
            await self._publish(SwarmFailed(swarm_run_id=parent_run_id, error=safe_error.message))
            return SwarmFailedOutcome(error=safe_error)
        if gate.cancel_requested:
            logger.info(
                "swarm %s run %s cancelled mid-run", spec.id, parent_run_id
            )
            await self._converge_cancel(task_plan, parent_run_id, owner, fence)
            await self._publish(SwarmCancelled(swarm_run_id=parent_run_id))
            return SwarmFailedOutcome(
                error=RunError("swarm_cancelled", "swarm was cancelled mid-run")
            )
        nodes = await self._collect_nodes(task_plan)
        projection = collect(task_plan.id, nodes)
        await self._complete_parent(
            parent_run_id,
            owner,
            fence,
            task_plan=task_plan,
            usage=usage,
            final_output=projection,
        )
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
        task_plan = await self._tasks.get_plan(_plan_id_from_run(record))
        owner = f"swarm:{execution_id}"
        try:
            claimed = await self._claim_for_recovery(record, owner)
        except StorageConflictError as exc:
            raise RecoveryConflictError("recovery claim lost") from exc
        try:
            if task_plan is None:
                raise InvalidSpecError("plan_integrity")
            validated = await self.validate_persisted_swarm_run(
                record, principal=principal
            )
            task_plan = validated.task_plan
            spec = validated.spec
            prepared_agents = validated.prepared_agents
            snapshot = validated.snapshot
        except StorageError:
            raise
        except Exception as exc:
            error_type = _stable_integrity_error_type(exc)
            if task_plan is not None:
                await self._fail_parent(
                    execution_id,
                    owner,
                    claimed.lease.fence,
                    error_type,
                    error_type,
                    task_plan=task_plan,
                )
            else:
                await self._store.fail_run(
                    FailExecution(
                        execution_id,
                        owner,
                        claimed.lease.fence,
                        AgentSnapshotData((), None, RunUsage(), 0),
                        RunError(
                            error_type,
                            error_type,
                        ),
                    )
                )
            return SwarmFailedOutcome(error=RunError(error_type, error_type))
        gate = _SwarmControlGate(
            limits=spec.limits,
            parent_run_id=execution_id,
            store=self._store,
            owner=owner,
            fence=claimed.lease.fence,
            lease_duration=_LEASE_DURATION,
            deadline_at=_deadline_from_definition(record.definition.spec),
        )
        logger.debug("recovering swarm run %s", execution_id)
        try:
            recovery_lease = _RecoveryLease(
                record=claimed,
                next_heartbeat_at=(
                    time.monotonic() + _LEASE_DURATION.total_seconds() / 3
                ),
            )
            await self._reconcile_inflight(
                task_plan,
                recovery_lease,
                principal=principal,
            )
            gate.seed_usage(
                tuple(
                    execution.usage
                    for execution in await self._tasks.list_executions(task_plan.id)
                    if execution.status not in {TaskStatus.READY, TaskStatus.SKIPPED}
                    and not (
                        execution.status is TaskStatus.CANCELLED
                        and execution.attempt == 0
                    )
                )
            )
            usage = await self._drive(
                spec,
                task_plan,
                execution_id,
                owner,
                gate,
                principal,
                record.session_id,
                snapshot,
                prepared_agents,
            )
        except SwarmLimitExceededError as exc:
            if exc.kind == "parent_cancelled":
                await self._converge_cancel(
                    task_plan, execution_id, owner, claimed.lease.fence
                )
                return SwarmFailedOutcome(
                    error=RunError("swarm_cancelled", "swarm was cancelled")
                )
            if exc.kind != "parent_lease_lost":
                await self._fail_parent(
                    execution_id,
                    owner,
                    claimed.lease.fence,
                    exc.kind,
                    exc.kind,
                    task_plan=task_plan,
                )
            return SwarmFailedOutcome(error=sanitize_run_error(exc))
        except asyncio.CancelledError:
            await self._converge_cancel(
                task_plan, execution_id, owner, claimed.lease.fence
            )
            raise
        except ChildCancelNotConvergedError as exc:
            logger.warning(
                "swarm run %s child cancellation did not converge: %s",
                execution_id,
                exc,
            )
            return SwarmFailedOutcome(
                error=RunError("child_cancel_not_converged", "child cancellation did not converge")
            )
        except Exception as exc:
            safe_error = sanitize_run_error(exc)
            await self._fail_parent(
                execution_id,
                owner,
                claimed.lease.fence,
                safe_error.error_type,
                safe_error.message,
                task_plan=task_plan,
            )
            return SwarmFailedOutcome(error=safe_error)
        if gate.cancel_requested:
            await self._converge_cancel(
                task_plan, execution_id, owner, claimed.lease.fence
            )
            return SwarmFailedOutcome(
                error=RunError("swarm_cancelled", "swarm was cancelled")
            )
        nodes = await self._collect_nodes(task_plan)
        projection = collect(task_plan.id, nodes)
        await self._complete_parent(
            execution_id,
            owner,
            claimed.lease.fence,
            task_plan=task_plan,
            usage=usage,
            final_output=projection,
        )
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
        try:
            validated = await self.validate_persisted_swarm_run(
                record, principal=principal
            )
            executions = validated.executions
        except StorageError:
            raise
        except Exception as exc:
            return SwarmRunView(
                plan_id=task_plan.id,
                parent_run_id=execution_id,
                status=record.status.value,
                error=RunError("integrity_error", sanitize_run_error(exc).message),
                nodes=(),
                status_counts={},
            )
        executions = {
            e.node_id: e for e in await self._tasks.list_executions(task_plan.id)
        }
        node_views = tuple(self._node_view(node, executions) for node in task_plan.nodes)
        projection = collect(task_plan.id, node_views)
        final_output = None
        snapshot_usage = None
        if record.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            snapshot = await self._store.get_snapshot(execution_id)
            if snapshot is None or snapshot.final_output is None:
                return SwarmRunView(
                    plan_id=task_plan.id,
                    parent_run_id=execution_id,
                    status=record.status.value,
                    error=RunError("integrity_error", "terminal snapshot missing"),
                    nodes=(),
                    status_counts={},
                )
            if canonical_json_bytes(normalize_json(snapshot.final_output)) != canonical_json_bytes(normalize_json(projection)):
                return SwarmRunView(
                    plan_id=task_plan.id,
                    parent_run_id=execution_id,
                    status=record.status.value,
                    error=RunError("integrity_error", "terminal snapshot mismatch"),
                    nodes=(),
                    status_counts={},
                )
            final_output = snapshot.final_output
            snapshot_usage = TaskUsage(
                input_tokens=snapshot.usage.input_tokens,
                output_tokens=snapshot.usage.output_tokens,
                total_cost=snapshot.usage.total_cost,
                cache_write_tokens=snapshot.usage.cache_write_tokens,
                cache_read_tokens=snapshot.usage.cache_read_tokens,
            )
        output_nodes = projection["nodes"]
        status_counts = projection["status_counts"]
        if isinstance(final_output, dict):
            output_nodes = final_output.get("nodes", output_nodes)
            status_counts = final_output.get("status_counts", status_counts)
        return SwarmRunView(
            plan_id=task_plan.id,
            parent_run_id=execution_id,
            status=record.status.value,
            error=record.error,
            nodes=tuple(output_nodes.values()),  # type: ignore[union-attr]
            status_counts=status_counts,  # type: ignore[arg-type]
            final_output=final_output,
            usage=snapshot_usage,
        )

    async def validate_persisted_swarm_run(
        self,
        record: RunRecord,
        *,
        principal: PrincipalContext,
    ) -> ValidatedSwarmRun:
        task_plan = await self._tasks.get_plan(_plan_id_from_run(record))
        if task_plan is None:
            raise InvalidSpecError("plan_integrity")
        spec = decode_swarm_spec(_definition_swarm_spec(record.definition.spec))
        prepared_agents, fingerprints = await self._validate_agents(
            spec,
            task_plan,
            principal,
            session_id=record.session_id,
        )
        executions = await self._tasks.list_executions(task_plan.id)
        _assert_persisted_integrity(record, task_plan, fingerprints, executions)
        await _assert_child_context(self._store, record, executions)
        snapshot = _snapshot_from_record(record)
        return ValidatedSwarmRun(
            record=record,
            task_plan=task_plan,
            spec=spec,
            snapshot=snapshot,
            executions=executions,
            prepared_agents=prepared_agents,
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

    async def _claim_for_recovery(self, record: RunRecord, owner: str) -> RunRecord:
        now = datetime.now(timezone.utc)
        return await self._store.claim_run_for_recovery(
            record.id,
            owner=owner,
            now=now,
            duration=_LEASE_DURATION,
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
        prepared_agents: "Mapping[str, PreparedAgentExecution] | None" = None,
    ) -> TaskUsage:
        runner = _ChildNodeRunner(
            agent_execution=self._agent_execution,
            agent_provider=self._agent_provider,
            principal=principal,
            session_id=session_id,
            parent_run_id=parent_run_id,
            root_execution_id=parent_run_id,
            message_history=snapshot,
            execution_store=self._store,
            live_events=self._live_events,
            prepared_agents=prepared_agents or {},
        )
        engine = TaskGraphEngine(
            store=self._tasks,
            runner=runner,
            gate=gate,
            limits=spec.limits,
            owner=owner,
            parent_run_id=parent_run_id,
            principal=principal,
            on_skip=lambda nid, blk: self._publish_skip(parent_run_id, nid, blk),
            on_node_terminal=lambda nid, outcome: self._publish_node_terminal(
                parent_run_id, nid, outcome
            ),
            logger=logger,
        )
        return await engine.execute(task_plan)

    async def _validate_agents(
        self,
        spec: SwarmSpec,
        task_plan: TaskPlan,
        principal: PrincipalContext,
        *,
        session_id: str,
    ) -> "tuple[dict[str, PreparedAgentExecution], dict[str, str]]":
        if self._agent_provider is None:
            raise InvalidSpecError(
                "no agent provider configured; cannot resolve or validate node agents"
            )
        validate_plan_against_swarm(
            task_plan,
            allowed_agent_ids={agent.agent_id for agent in spec.agents},
        )
        prepared: "dict[str, PreparedAgentExecution]" = {}
        fingerprints: "dict[str, str]" = {}
        for node in task_plan.nodes:
            try:
                agent_spec = await self._agent_provider.get(node.payload.agent_id)  # type: ignore[attr-defined]
            except Exception as exc:
                raise RuntimeInitializationError("agent_preflight_failed") from exc
            prepared_execution = await self._agent_execution.prepare_agent_execution(
                agent_spec,
                principal=principal,
                session_id=session_id,
                execution_id=f"preflight:{spec.id}:{node.id}",
                root_execution_id=f"preflight:{spec.id}",
                parent_execution_id=None,
            )
            prepared[node.id] = prepared_execution
            fingerprints[agent_spec.id] = prepared_execution.fingerprint
        return prepared, fingerprints

    async def _load_session_snapshot(self, session_id: str) -> "tuple[object, ...]":
        snapshot = await self._store.load_session_context(session_id)
        if not isinstance(snapshot, tuple):
            raise InvalidSpecError("session_snapshot_invalid")
        try:
            return tuple(normalize_json(item) for item in snapshot)
        except (TypeError, ValueError) as exc:
            raise InvalidSpecError("session_snapshot_invalid") from exc

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
                "cache_write_tokens": (
                    execution.usage.cache_write_tokens if execution else 0
                ),
                "cache_read_tokens": (
                    execution.usage.cache_read_tokens if execution else 0
                ),
            },
        }

    async def _converge_cancel(
        self,
        task_plan: TaskPlan,
        parent_run_id: str,
        owner: str,
        fence: int,
    ) -> None:
        latest = await self._store.get_run(parent_run_id)
        if latest is not None and latest.status is RunStatus.RUNNING:
            await self._store.request_cancel(
                RequestCancellation(parent_run_id, owner, fence, datetime.now(timezone.utc))
            )
            latest = await self._store.get_run(parent_run_id)
        if latest is not None and latest.status is RunStatus.CANCELLING:
            usage = await self._sum_task_usage(task_plan)
            await self._assert_parent_terminal_gate(
                parent_run_id,
                owner,
                fence,
                task_plan,
            )
            await self._store.acknowledge_cancel(
                AcknowledgeCancellation(
                    parent_run_id,
                    owner,
                    fence,
                    self._snapshot(
                        task_plan,
                        usage,
                        final_output=collect(
                            task_plan.id, await self._collect_nodes(task_plan)
                        ),
                    ),
                )
            )

    async def _complete_parent(
        self,
        parent_run_id: str,
        owner: str,
        fence: int,
        *,
        task_plan: TaskPlan,
        usage: TaskUsage,
        final_output: object,
    ) -> None:
        await self._assert_parent_terminal_gate(
            parent_run_id,
            owner,
            fence,
            task_plan,
        )
        await self._store.complete_run(
            CompleteExecution(
                parent_run_id, owner, fence,
                self._snapshot(task_plan, usage, final_output=final_output),
            )
        )

    async def _fail_parent(
        self,
        parent_run_id: str,
        owner: str,
        fence: int,
        error_type: str,
        message: str,
        *,
        task_plan: TaskPlan,
    ) -> None:
        usage = await self._sum_task_usage(task_plan)
        projection = collect(task_plan.id, await self._collect_nodes(task_plan))
        await self._assert_parent_terminal_gate(
            parent_run_id,
            owner,
            fence,
            task_plan,
        )
        await self._store.fail_run(
            FailExecution(
                parent_run_id, owner, fence,
                self._snapshot(task_plan, usage, final_output=projection),
                RunError(error_type, message),
            )
        )

    async def _assert_parent_terminal_gate(
        self,
        parent_run_id: str,
        owner: str,
        fence: int,
        task_plan: TaskPlan,
    ) -> None:
        record = await self._store.get_run(parent_run_id)
        if record is None:
            raise StorageError("parent run disappeared before terminal commit")
        if record.lease.owner != owner or record.lease.fence != fence:
            raise StorageConflictError("parent lease lost before terminal commit")
        if record.lease.expires_at is None or record.lease.expires_at <= datetime.now(timezone.utc):
            raise StorageConflictError("parent lease expired before terminal commit")
        executions = await self._tasks.list_executions(task_plan.id)
        if any(
            execution.status is TaskStatus.CLAIMED
            and execution.owner == owner
            for execution in executions
        ):
            raise TaskGraphInvariantError(
                "parent terminal gate found current-owner CLAIMED nodes"
            )
        definition = record.definition.spec
        if (
            not isinstance(definition, dict)
            or sha256(canonical_json_bytes(definition)).hexdigest()
            != record.definition.spec_hash
        ):
            raise InvalidSpecError("definition_integrity")
        expected_hash = definition.get("task_plan_hash") if isinstance(definition, dict) else None
        actual_hash = sha256(canonical_json_bytes(encode_plan(task_plan))).hexdigest()
        if expected_hash != actual_hash:
            raise InvalidSpecError("plan_integrity")

    async def _sum_task_usage(self, task_plan: TaskPlan) -> TaskUsage:
        accumulator = UsageAccumulator()
        for execution in await self._tasks.list_executions(task_plan.id):
            if execution.status is TaskStatus.READY or execution.status is TaskStatus.SKIPPED:
                continue
            if execution.status is TaskStatus.CANCELLED and execution.attempt == 0:
                continue
            accumulator.add(execution.usage)
        return accumulator.freeze()

    def _snapshot(
        self,
        task_plan: TaskPlan,
        usage: TaskUsage,
        *,
        final_output: object | None = None,
    ) -> AgentSnapshotData:
        return AgentSnapshotData(
            delta_messages=(),
            final_output=(normalize_json(final_output) if final_output is not None else None),
            usage=RunUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                total_cost=usage.total_cost,
                cache_write_tokens=usage.cache_write_tokens,
                cache_read_tokens=usage.cache_read_tokens,
            ),
            trace_end_sequence=0,
        )

    async def _abort_parent(
        self,
        parent_run_id: str,
        owner: str,
        fence: int,
        cause: BaseException,
    ) -> None:
        """Persist a failed parent snapshot when graph creation cannot start."""
        await self._store.abort_run(
            AbortExecution(
                parent_run_id, owner, fence,
                AgentSnapshotData(
                    delta_messages=(),
                    final_output=None,
                    usage=RunUsage(),
                    trace_end_sequence=0,
                    capture_state=MessageCaptureState.UNAVAILABLE,
                ),
                RunError("plan_create_failed", "task plan creation failed"),
            )
        )

    async def _reconcile_inflight(
        self,
        task_plan: TaskPlan,
        recovery_lease: _RecoveryLease,
        *,
        principal: PrincipalContext,
    ) -> None:
        claimed = tuple(
            execution
            for execution in await self._tasks.list_executions(task_plan.id)
            if execution.status is TaskStatus.CLAIMED
        )
        for execution in claimed:
            await self._reconcile_claimed(
                execution,
                recovery_lease,
                principal=principal,
            )

    async def _reconcile_claimed(
        self,
        execution: TaskExecution,
        recovery_lease: _RecoveryLease,
        *,
        principal: PrincipalContext,
    ) -> None:
        current = execution
        while not current.terminal:
            await self._renew_recovery_parent(recovery_lease)
            latest = await self._tasks.get_execution(current.id)
            if latest is None:
                raise StorageError("task execution disappeared during recovery")
            current = latest
            if current.terminal:
                return
            if current.lease.expires_at is None or current.lease.expires_at > datetime.now(timezone.utc):
                await asyncio.sleep(RECOVERY_CHILD_POLL_INTERVAL)
                continue
            try:
                current = await self._tasks.take_over_expired_claim_for_reconcile(
                    current.id,
                    owner=recovery_lease.record.lease.owner or "",
                    now=datetime.now(timezone.utc),
                    duration=_LEASE_DURATION,
                )
            except StorageConflictError:
                continue
            logger.info(
                "recovery took over node=%s fence=%s child_run_id=%s",
                current.node_id,
                current.fence,
                current.active_run_id,
            )
            await self._reconcile_owned_claim(
                current,
                recovery_lease,
                principal=principal,
            )
            return

    async def _reconcile_owned_claim(
        self,
        execution: TaskExecution,
        recovery_lease: _RecoveryLease,
        *,
        principal: PrincipalContext,
    ) -> None:
        owner = recovery_lease.record.lease.owner or ""
        if execution.active_run_id is None:
            await self._tasks.fail(
                execution.id,
                owner=owner,
                fence=execution.fence,
                error=RunError(
                    "orphaned_before_child_start",
                    "claim without bind",
                ),
                usage=execution.usage,
            )
            return
        child = await self._store.get_run(execution.active_run_id)
        if child is None:
            await self._tasks.fail(
                execution.id,
                owner=owner,
                fence=execution.fence,
                error=RunError("child_run_missing", "child run missing"),
                usage=execution.usage,
            )
            return
        if child.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.PAUSED,
        }:
            await self._converge_recovered_child(execution, child)
            return

        logger.info(
            "recovery requesting child cancellation node=%s child_run_id=%s",
            execution.node_id,
            child.id,
        )
        await self._agent_execution.cancel(child.id, principal=principal)
        deadline = time.monotonic() + RECOVERY_CHILD_CANCEL_TIMEOUT
        next_node_renew = time.monotonic() + _LEASE_DURATION.total_seconds() / 3
        while True:
            await self._renew_recovery_parent(recovery_lease)
            now = time.monotonic()
            if now >= next_node_renew:
                execution = await self._tasks.renew(
                    execution.id,
                    owner=owner,
                    fence=execution.fence,
                    duration=_LEASE_DURATION,
                )
                next_node_renew = now + _LEASE_DURATION.total_seconds() / 3
            child = await self._store.get_run(child.id)
            if child is None:
                raise ChildRunMissingError(execution.active_run_id)
            if child.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.PAUSED,
            }:
                await self._converge_recovered_child(execution, child)
                return
            if time.monotonic() >= deadline:
                raise ChildCancelNotConvergedError(
                    f"child {child.id!r} did not converge after cancellation"
                )
            await asyncio.sleep(RECOVERY_CHILD_POLL_INTERVAL)

    async def _renew_recovery_parent(self, recovery_lease: _RecoveryLease) -> None:
        now = time.monotonic()
        current = await self._store.get_run(recovery_lease.record.id)
        if current is None:
            raise StorageError("parent run disappeared during recovery")
        if (
            current.lease.owner != recovery_lease.record.lease.owner
            or current.lease.fence != recovery_lease.record.lease.fence
            or is_expired(current.lease, datetime.now(timezone.utc))
        ):
            raise StorageConflictError("parent lease lost during recovery")
        recovery_lease.record = current
        if now < recovery_lease.next_heartbeat_at:
            return
        record = await self._store.heartbeat_run(
            HeartbeatExecution(
                recovery_lease.record.id,
                recovery_lease.record.lease.owner or "",
                recovery_lease.record.lease.fence,
                datetime.now(timezone.utc),
                _LEASE_DURATION,
            )
        )
        recovery_lease.record = record
        recovery_lease.next_heartbeat_at = (
            now + _LEASE_DURATION.total_seconds() / 3
        )

    async def _converge_recovered_child(
        self,
        execution: TaskExecution,
        child: RunRecord,
    ) -> None:
        owner = execution.owner or ""
        output, usage = await self._recover_child_artifacts(child)
        if child.status is RunStatus.COMPLETED:
            await self._tasks.complete(
                execution.id,
                owner=owner,
                fence=execution.fence,
                result=output,
                usage=usage,
            )
        elif child.status is RunStatus.FAILED:
            await self._tasks.fail(
                execution.id,
                owner=owner,
                fence=execution.fence,
                error=(
                    RunError(child.error.error_type, "child execution failed")
                    if child.error is not None
                    else RunError("child_failed", "child execution failed")
                ),
                usage=usage,
            )
        elif child.status is RunStatus.PAUSED:
            await self._tasks.fail(
                execution.id,
                owner=owner,
                fence=execution.fence,
                error=RunError(
                    "approval_not_supported",
                    "task_graph child paused for approval",
                ),
                usage=usage,
            )
        else:
            parent = await self._store.get_run(child.parent_execution_id or "")
            reason = (
                "parent_cancelled"
                if parent is not None and parent.status is RunStatus.CANCELLING
                else "interrupted_execution"
            )
            if reason == "parent_cancelled":
                await self._tasks.cancel_claimed(
                    execution.id,
                    owner=owner,
                    fence=execution.fence,
                    reason=reason,
                    usage=usage,
                )
            else:
                await self._tasks.fail(
                    execution.id,
                    owner=owner,
                    fence=execution.fence,
                    error=RunError(reason, "child cancelled during recovery"),
                    usage=usage,
                )

    async def _recover_child_artifacts(
        self, child: RunRecord
    ) -> "tuple[object | None, TaskUsage]":
        """Read a terminal child's output and usage from its persisted snapshot."""
        snapshot = await self._store.get_snapshot(child.id)
        if snapshot is None:
            raise ChildSnapshotError(child.id)
        try:
            ru = snapshot.usage
            usage = TaskUsage(
                input_tokens=ru.input_tokens,
                output_tokens=ru.output_tokens,
                total_cost=ru.total_cost,
                cache_write_tokens=ru.cache_write_tokens,
                cache_read_tokens=ru.cache_read_tokens,
            )
            return snapshot.final_output, usage
        except (AttributeError, TypeError, ValueError) as exc:
            raise ChildSnapshotError(child.id) from exc

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
    value = raw.get("task_plan_id", raw.get("plan_id"))
    return str(value) if isinstance(value, str) else ""


def _stable_integrity_error_type(exc: BaseException) -> str:
    if isinstance(exc, RuntimeInitializationError):
        return "agent_preflight_failed"
    value = str(exc)
    if value in {
        "agent_fingerprint_mismatch",
        "child_context_integrity",
        "child_run_integrity",
        "definition_integrity",
        "plan_integrity",
        "session_snapshot_invalid",
        "snapshot_integrity",
        "task_execution_integrity",
    }:
        return value
    return "definition_integrity"


def _definition_swarm_spec(value: JsonValue) -> JsonValue:
    if isinstance(value, dict) and "swarm_spec" in value:
        return value["swarm_spec"]
    return value


def _deadline_from_definition(value: JsonValue) -> "datetime | None":
    if not isinstance(value, dict):
        return None
    raw = value.get("deadline_at")
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise InvalidSpecError("definition_integrity")
    deadline = datetime.fromisoformat(raw)
    if deadline.tzinfo is None:
        raise InvalidSpecError("definition_integrity")
    return deadline


def _snapshot_from_record(record: RunRecord) -> "tuple[object, ...]":
    raw = record.input if isinstance(record.input, dict) else {}
    snapshot = raw.get("session_snapshot", ())
    if not isinstance(snapshot, (list, tuple)):
        raise InvalidSpecError("session_snapshot_invalid")
    actual = sha256(canonical_json_bytes(list(snapshot))).hexdigest()
    expected = raw.get("session_snapshot_hash")
    if expected is not None and actual != expected:
        raise InvalidSpecError("snapshot_integrity")
    return tuple(snapshot)


def _assert_persisted_integrity(
    record: RunRecord,
    task_plan: TaskPlan,
    fingerprints: "Mapping[str, str]",
    executions: "tuple[TaskExecution, ...]",
) -> None:
    definition = record.definition.spec
    if not isinstance(definition, dict):
        raise InvalidSpecError("definition_integrity")
    if sha256(canonical_json_bytes(definition)).hexdigest() != record.definition.spec_hash:
        raise InvalidSpecError("definition_integrity")
    if record.definition.schema != "swarm-task-graph.v1":
        raise InvalidSpecError("definition_integrity")
    if definition.get("task_plan_id") != task_plan.id:
        raise InvalidSpecError("plan_integrity")
    expected_plan_hash = definition.get("task_plan_hash")
    actual_plan_hash = sha256(canonical_json_bytes(encode_plan(task_plan))).hexdigest()
    if not isinstance(expected_plan_hash, str) or expected_plan_hash != actual_plan_hash:
        raise InvalidSpecError("plan_integrity")
    raw_input = record.input if isinstance(record.input, dict) else {}
    if raw_input.get("task_plan_id") != task_plan.id:
        raise InvalidSpecError("plan_integrity")
    expected_snapshot_hash = definition.get("session_snapshot_hash")
    if (
        not isinstance(expected_snapshot_hash, str)
        or expected_snapshot_hash != raw_input.get("session_snapshot_hash")
    ):
        raise InvalidSpecError("snapshot_integrity")
    expected_agents = definition.get("agent_fingerprints")
    if not isinstance(expected_agents, dict) or dict(expected_agents) != dict(fingerprints):
        raise InvalidSpecError("agent_fingerprint_mismatch")
    if record.definition.schema == "swarm-task-graph.v1":
        expected_nodes = {node.id for node in task_plan.nodes}
        if any(execution.plan_id != task_plan.id for execution in executions):
            raise InvalidSpecError("task_execution_integrity")
        if {execution.node_id for execution in executions} != expected_nodes or len(executions) != len(expected_nodes):
            raise InvalidSpecError("task_execution_integrity")
        for execution in executions:
            if execution.id != task_execution_id(task_plan.id, execution.node_id):
                raise InvalidSpecError("task_execution_integrity")
            if execution.active_run_id is not None and execution.active_run_id != child_run_id(record.id, execution.node_id):
                raise InvalidSpecError("child_run_integrity")


async def _assert_child_context(
    store: ExecutionStore,
    record: RunRecord,
    executions: "tuple[TaskExecution, ...]",
) -> None:
    for execution in executions:
        if execution.active_run_id is None:
            continue
        child = await store.get_run(execution.active_run_id)
        if child is None:
            continue
        if (
            child.parent_execution_id != record.id
            or child.root_execution_id != record.root_execution_id
            or child.session_id != record.session_id
            or child.tenant_id != record.tenant_id
            or child.user_id != record.user_id
        ):
            raise InvalidSpecError("child_context_integrity")




class _SwarmControlGate(ControlGate):
    """Monotonic timeout, token/cost caps, cancel-request propagation, and
    parent-lease liveness for one swarm run. ``check`` is async: it reads the
    parent RunRecord each pass so an external ``cancel_swarm`` (persisted
    CANCELLING) and a lost/reclaimed parent lease both propagate into the
    scheduler without an in-process channel. The parent lease is renewed
    periodically so long-running swarms in a multi-process deployment do not
    silently lose ownership."""

    def __init__(
        self,
        *,
        limits: "object",
        parent_run_id: str,
        store: ExecutionStore,
        owner: str,
        fence: int,
        lease_duration: timedelta,
        deadline_at: "datetime | None" = None,
    ) -> None:
        self._limits = limits
        self._parent_run_id = parent_run_id
        self._store = store
        self._owner = owner
        self._fence = fence
        self._lease_duration = lease_duration
        self._heartbeat_interval = max(lease_duration / 3, timedelta(seconds=1))
        self._deadline_at = deadline_at
        self._start: "float | None" = None
        self._last_heartbeat: "datetime | None" = None
        self._accumulated = UsageAccumulator()
        self._has_usage = False
        self._cancel = False

    def record_usage(self, usage: TaskUsage) -> None:
        self._has_usage = True
        self._accumulated.add(usage)

    def seed_usage(self, usages: "tuple[TaskUsage, ...]") -> None:
        for usage in usages:
            self.record_usage(usage)

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
        if is_expired(record.lease, now_dt):
            raise SwarmLimitExceededError(
                "parent lease expired", kind="parent_lease_lost"
            )
        if self._last_heartbeat is None:
            self._last_heartbeat = now_dt
            return
        if now_dt - self._last_heartbeat < self._heartbeat_interval:
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
        except StorageConflictError as exc:
            raise SwarmLimitExceededError(
                "parent lease renewal failed", kind="parent_lease_lost"
            ) from exc

    def next_wake_delay(self, *, now_monotonic: float) -> float:
        delay = 1.0
        timeout = getattr(self._limits, "timeout_seconds", None)
        if timeout is not None and self._start is not None:
            delay = min(delay, max(0.0, timeout - (now_monotonic - self._start)))
        return delay

    async def check(self) -> None:
        now = time.monotonic()
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
        if self._deadline_at is not None and datetime.now(timezone.utc) >= self._deadline_at:
            raise SwarmLimitExceededError("swarm timeout", kind="timeout")
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
            spent_any = self._has_usage
            # Before any node reports, an unknown cost is expected (nothing to
            # measure yet); cost_usage_unavailable fires only once a node HAS
            # reported usage but could not supply a cost.
            if spent_any and not self._accumulated.cost_known:
                raise SwarmLimitExceededError(
                    "cost configured but usage unavailable", kind="cost_usage_unavailable"
                )
            if self._accumulated.cost_known and self._accumulated.total_cost > max_cost:
                raise SwarmLimitExceededError(
                    "cost limit exceeded", kind="max_total_cost"
                )

    async def check_before_launch(self) -> None:
        max_tokens = getattr(self._limits, "max_total_tokens", None)
        if max_tokens is not None and self._accumulated.input_tokens + self._accumulated.output_tokens >= max_tokens:
            raise SwarmLimitExceededError("token limit reached", kind="token_limit_reached")
        max_cost = getattr(self._limits, "max_total_cost", None)
        if max_cost is not None and self._accumulated.cost_known and self._accumulated.total_cost >= max_cost:
            raise SwarmLimitExceededError("cost limit reached", kind="cost_limit_reached")


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
    execution_store: ExecutionStore
    live_events: RunLiveEventSink
    prepared_agents: "Mapping[str, PreparedAgentExecution]"

    async def run(self, request: NodeRunRequest) -> NodeRunResult:
        node = request.node
        prepared = self.prepared_agents.get(node.id)
        agent_spec = (
            prepared.agent_spec
            if prepared is not None
            else await self._resolve(node.payload.agent_id)
        )
        child_id = request.child_run_id
        deps_metadata = _dependency_metadata(self.parent_run_id, request)
        await self._publish(
            SwarmStepStarted(
                swarm_run_id=self.parent_run_id, task_id=node.id, child_run_id=child_id
            )
        )
        kwargs = dict(
            principal=self.principal,
            session_id=self.session_id,
            execution_id=child_id,
            root_execution_id=self.root_execution_id,
            parent_execution_id=self.parent_run_id,
            message_history=self.message_history,
            metadata={"task_graph": deps_metadata},
        )
        if prepared is not None:
            kwargs["prepared_execution"] = prepared
        try:
            result = await self.agent_execution.run_child(
                agent_spec,
                node.payload.prompt,
                **kwargs,
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            child = await self.execution_store.get_run(child_id)
            if child is None:
                raise ChildRunMissingError(child_id) from exc
            try:
                snapshot = await self.execution_store.get_snapshot(child_id)
            except StorageError:
                raise
            except (AttributeError, TypeError, ValueError) as snapshot_error:
                raise ChildSnapshotError(child_id) from snapshot_error
            if snapshot is None:
                raise ChildSnapshotError(child_id) from exc
            try:
                usage = TaskUsage(
                    input_tokens=snapshot.usage.input_tokens,
                    output_tokens=snapshot.usage.output_tokens,
                    total_cost=snapshot.usage.total_cost,
                    cache_write_tokens=snapshot.usage.cache_write_tokens,
                    cache_read_tokens=snapshot.usage.cache_read_tokens,
                )
            except (AttributeError, TypeError, ValueError) as snapshot_error:
                raise ChildSnapshotError(child_id) from snapshot_error
            raise ChildExecutionPlatformError(
                child_run_id=child_id,
                usage=usage,
                error_type=type(exc).__name__,
                safe_message="child execution failed",
                cause=exc,
            ) from exc
        # Step-terminal events are published from the engine's on_node_terminal
        # hook AFTER the TaskExecution is persisted, not here (persist-before-event).
        if result.status is RunStatus.COMPLETED:
            return NodeRunResult(
                status=TaskStatus.COMPLETED, result=result.output, usage=result.usage
            )
        if result.status is RunStatus.CANCELLED:
            parent = await self.execution_store.get_run(self.parent_run_id)
            if parent is None or parent.status not in {
                RunStatus.CANCELLING,
                RunStatus.CANCELLED,
            }:
                return NodeRunResult(
                    status=TaskStatus.FAILED,
                    error=RunError(
                        "unexpected_child_cancel",
                        "child cancelled without parent cancellation",
                    ),
                    usage=result.usage,
                )
            return NodeRunResult(
                status=TaskStatus.CANCELLED,
                reason="parent_cancelled",
                error=result.error,
                usage=result.usage,
            )
        if result.status is RunStatus.PAUSED:
            return NodeRunResult(
                status=TaskStatus.FAILED,
                error=RunError(
                    "approval_not_supported",
                    "task_graph child paused for approval",
                ),
                usage=result.usage,
            )
        if result.status is not RunStatus.FAILED:
            raise TaskGraphInvariantError(
                f"child {child_id!r} returned unsupported status"
            )
        safe_child_error = (
            RunError(result.error.error_type, "child execution failed")
            if result.error is not None
            else RunError("child_failed", "child execution failed")
        )
        return NodeRunResult(
            status=TaskStatus.FAILED,
            error=safe_child_error,
            usage=result.usage,
        )

    async def request_cancel(
        self,
        *,
        child_run_id: str,
        principal: "PrincipalContext | None",
        reason: str,
    ) -> None:
        if principal is None or principal is not self.principal:
            raise PrincipalAccessDeniedError("child cancellation principal mismatch")
        child = await self.execution_store.get_run(child_run_id)
        if child is None:
            raise ChildRunMissingError(child_run_id)
        if child.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return
        await self.agent_execution.cancel(child_run_id, principal=self.principal)

    async def read_usage(self, *, child_run_id: str) -> TaskUsage:
        child = await self.execution_store.get_run(child_run_id)
        if child is None:
            raise ChildRunMissingError(child_run_id)
        snapshot = await self.execution_store.get_snapshot(child.id)
        if snapshot is None:
            raise ChildSnapshotError(child.id)
        try:
            usage = snapshot.usage
            return TaskUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_cost=usage.total_cost,
                cache_write_tokens=usage.cache_write_tokens,
                cache_read_tokens=usage.cache_read_tokens,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ChildSnapshotError(child.id) from exc

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


__all__ = ["SwarmExecutionService", "ValidatedSwarmRun"]
