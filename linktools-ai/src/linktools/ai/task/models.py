#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task domain models.

The reliable-task domain's data shapes: jobs, tasks, attempts, transitions and
signals, plus the policies (retry, side-effect), principal/actor/budget context
and resource-snapshot references. Models are frozen ``slots=True`` dataclasses
so a record, once written, cannot be mutated in place -- the store is the only
thing that moves them between states.

State-machine legality lives here as transition tables plus validators; the
TaskStore enforces them inside its atomic domain operations (claim / commit /
recover), never as ad-hoc field writes.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..artifact.models import ArtifactRef, ResourceSnapshotRef
from ..security.principal import ActorRef, ScopeSet


# ----------------------------------------------------------------- enums --


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    PENDING = "pending"  # dependencies not yet satisfied
    READY = "ready"  # claimable
    CLAIMED = "claimed"  # taken by a worker
    WAITING = "waiting"  # awaiting an external signal
    RETRY_WAIT = "retry_wait"  # awaiting next retry time
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"  # task was reclaimed/cancelled under this attempt


class TaskFailureKind(str, Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    POLICY_DENIED = "policy_denied"
    HANDLER_NOT_FOUND = "handler_not_found"
    INVALID_INPUT = "invalid_input"
    BUDGET_EXCEEDED = "budget_exceeded"
    SIDE_EFFECT_UNKNOWN = "side_effect_unknown"
    SUPERSEDED = "superseded"
    INTERNAL = "internal"


class SideEffectMode(str, Enum):
    NONE = "none"
    IDEMPOTENT = "idempotent"
    GUARDED = "guarded"
    NON_IDEMPOTENT = "non_idempotent"


# --------------------------------------------------------------- policies --


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 1
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.1
    retryable_kinds: "tuple[TaskFailureKind, ...]" = (
        TaskFailureKind.TRANSIENT,
        TaskFailureKind.TIMEOUT,
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")
        if self.max_delay_seconds < self.initial_delay_seconds:
            raise ValueError("max_delay_seconds must be >= initial_delay_seconds")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SideEffectPolicy:
    mode: SideEffectMode = SideEffectMode.NONE
    idempotency_key: "str | None" = None


# ----------------------------------------------------------- context types --


@dataclass(frozen=True, slots=True)
class TaskPrincipal:
    tenant_id: str
    user_id: "str | None" = None
    workspace_key: "str | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("TaskPrincipal.tenant_id must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ActorChain:
    actors: "tuple[ActorRef, ...]"
    delegated_scopes: "ScopeSet" = field(default_factory=ScopeSet.empty)

    def __post_init__(self) -> None:
        # Runtime construction is strict; persistence adapters own legacy migration.
        if not isinstance(self.delegated_scopes, ScopeSet):
            raise TypeError("ActorChain.delegated_scopes requires explicit ScopeSet")


@dataclass(frozen=True, slots=True)
class TaskBudget:
    max_tasks: "int | None" = None
    max_depth: "int | None" = None
    max_attempts: "int | None" = None
    max_runtime_seconds: "float | None" = None
    max_total_tokens: "int | None" = None
    # Decimal-as-string so JSON round-trips without float drift.
    max_total_cost: "str | None" = None


# ResourceSnapshotRef is defined in ..artifact.models (the shared low layer) and
# re-exported here so existing ``from linktools.ai.task.models import
# ResourceSnapshotRef`` references keep working.


# ----------------------------------------------------------- record types --


@dataclass(frozen=True, slots=True)
class JobRecord:
    id: str
    status: JobStatus
    principal: TaskPrincipal
    actor_chain: ActorChain
    budget: TaskBudget
    root_task_id: "str | None"
    input_artifact_id: "str | None"
    output_artifact_id: "str | None"
    version: int
    created_at: datetime
    started_at: "datetime | None"
    finished_at: "datetime | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskWaitCondition:
    """A signal a WAITING task is blocked on (set by a WaitSignal command).
    Lives on the TaskRecord -- not in metadata -- so it is a first-class part of
    the state machine that serialization and reconciliation can trust."""

    name: str
    correlation_key: str


@dataclass(frozen=True, slots=True)
class TaskRecord:
    id: str
    job_id: str
    parent_task_id: "str | None"
    key: str
    handler: str
    status: TaskStatus
    input_artifact_id: "str | None"
    output_artifact_id: "str | None"
    dependencies: "tuple[str, ...]"
    retry_policy: RetryPolicy
    side_effect_policy: SideEffectPolicy
    attempt_count: int
    available_at: datetime
    lease_owner: "str | None"
    lease_expires_at: "datetime | None"
    fencing_token: int
    active_attempt_id: "str | None"
    timeout_seconds: "float | None"
    resource_snapshots: "tuple[ResourceSnapshotRef, ...]"
    version: int
    created_at: datetime
    updated_at: datetime
    depth: int = 0
    delegated_scopes: "ScopeSet" = field(default_factory=ScopeSet.empty)
    actor_chain: "ActorChain | None" = None

    def __post_init__(self) -> None:
        # Runtime construction is strict; persistence adapters own legacy migration.
        if not isinstance(self.delegated_scopes, ScopeSet):
            raise TypeError("TaskRecord.delegated_scopes requires explicit ScopeSet")
    # Signal-wait state (set when a handler returns WaitSignal). wait_deadline_at
    # is None for an unbounded wait; reconcile_due moves a WAITING task past its
    # deadline to a retry or a terminal cancel.
    wait_conditions: "tuple[TaskWaitCondition, ...]" = ()
    wait_deadline_at: "datetime | None" = None
    # Pinned runnable resolution (set on the first attempt by bind_runnable).
    # A retry re-resolves and bind_runnable rejects a drift -- so a mapping
    # change between attempts can never silently re-run a different agent.
    resolved_runnable_id: "str | None" = None
    resolved_runnable_revision: "str | None" = None
    resolved_runnable_fingerprint: "str | None" = None
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskAttemptRecord:
    id: str
    task_id: str
    job_id: str
    attempt: int
    worker_id: str
    fencing_token: int
    status: AttemptStatus
    run_id: "str | None"  # set only by a handler that drives the existing Runtime
    started_at: datetime
    finished_at: "datetime | None"
    failure_kind: "TaskFailureKind | None"
    error_type: "str | None"
    error_message: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskTransitionRecord:
    id: str
    job_id: str
    task_id: "str | None"
    attempt_id: "str | None"
    from_status: "str | None"
    to_status: str
    reason: str
    occurred_at: datetime
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskSignalRecord:
    id: str
    job_id: str
    name: str
    correlation_key: str
    payload_artifact_id: "str | None"
    created_at: datetime
    consumed_by_task_id: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


def resolve_effective_scopes(
    requested: "tuple[str, ...] | list[str] | ScopeSet | None",
    parent: "ScopeSet",
) -> "ScopeSet":
    """Resolve effective delegated scopes (never expand permission).

    ``requested is None`` means "inherit the parent's effective scopes"
    (unrestricted when the parent was unrestricted). An unrestricted requested
    ScopeSet is equivalent to inherit: a child can never hold more than the
    parent, so "everything requested" still collapses to the parent's set. A
    concrete requested set (tuple/list/restricted ScopeSet) intersects with the
    parent's scopes -- the child keeps only scopes the parent held, in the
    parent's order. Always returns a concrete :class:`ScopeSet`."""
    if requested is None:
        return parent
    if isinstance(requested, ScopeSet):
        if requested.unrestricted:
            return parent
        requested_values = requested.values
    else:
        requested_values = tuple(requested)
    if parent.unrestricted:
        return ScopeSet.of(*requested_values)
    requested_set = set(requested_values)
    return ScopeSet.of(*(s for s in parent.values if s in requested_set))


def narrow_child_principal(
    parent: "TaskRecord",
    cmd_delegated_scopes: "tuple[str, ...] | None",
    cmd_handler: str,
    job_actor_chain: ActorChain,
) -> "tuple[ScopeSet, ActorChain]":
    """Compute a child task's effective ``(delegated_scopes, actor_chain)``.

    Delegated scopes can only narrow: when a command requests scopes, the child
    receives their intersection with the parent's effective scopes -- never a
    union, never a scope the parent did not hold. The actor chain appends the
    current handler as a new Actor so each delegation step is attributable."""
    parent_scopes = parent.delegated_scopes  # always a ScopeSet (normalized)
    child_scopes = resolve_effective_scopes(cmd_delegated_scopes, parent_scopes)
    parent_chain = (
        parent.actor_chain if parent.actor_chain is not None else job_actor_chain
    )
    child_chain = ActorChain(
        actors=parent_chain.actors + (ActorRef(kind="TaskHandler", id=cmd_handler),)
    )
    return child_scopes, child_chain


# ------------------------------------------------------- state machines --


class IllegalTaskTransitionError(Exception):
    """Raised when a Job/Task/Attempt transition is not legal from its state."""


JOB_TRANSITIONS: "dict[JobStatus, frozenset[JobStatus]]" = {
    JobStatus.PENDING: frozenset({JobStatus.RUNNING, JobStatus.CANCELLING}),
    JobStatus.RUNNING: frozenset(
        {JobStatus.WAITING, JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLING}
    ),
    JobStatus.WAITING: frozenset({JobStatus.RUNNING, JobStatus.CANCELLING}),
    JobStatus.CANCELLING: frozenset({JobStatus.CANCELLED, JobStatus.FAILED}),
    JobStatus.SUCCEEDED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.CANCELLED: frozenset(),
}

TASK_TRANSITIONS: "dict[TaskStatus, frozenset[TaskStatus]]" = {
    TaskStatus.PENDING: frozenset({TaskStatus.READY, TaskStatus.CANCELLING}),
    TaskStatus.READY: frozenset({TaskStatus.CLAIMED, TaskStatus.CANCELLING}),
    # CLAIMED can resolve, retry, wait, fail, be reclaimed (lease expiry), or cancel.
    TaskStatus.CLAIMED: frozenset(
        {
            TaskStatus.SUCCEEDED,
            TaskStatus.RETRY_WAIT,
            TaskStatus.WAITING,
            TaskStatus.FAILED,
            TaskStatus.READY,
            TaskStatus.CANCELLING,
        }
    ),
    TaskStatus.RETRY_WAIT: frozenset({TaskStatus.READY, TaskStatus.CANCELLING}),
    TaskStatus.WAITING: frozenset({TaskStatus.READY, TaskStatus.CANCELLING}),
    TaskStatus.CANCELLING: frozenset({TaskStatus.CANCELLED}),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}

ATTEMPT_TRANSITIONS: "dict[AttemptStatus, frozenset[AttemptStatus]]" = {
    AttemptStatus.RUNNING: frozenset(
        {
            AttemptStatus.SUCCEEDED,
            AttemptStatus.FAILED,
            AttemptStatus.CANCELLED,
            AttemptStatus.SUPERSEDED,
        }
    ),
    AttemptStatus.SUCCEEDED: frozenset(),
    AttemptStatus.FAILED: frozenset(),
    AttemptStatus.CANCELLED: frozenset(),
    AttemptStatus.SUPERSEDED: frozenset(),
}


def assert_job_transition(current: JobStatus, target: JobStatus) -> None:
    if not isinstance(current, Enum):
        current = JobStatus(current)
    if not isinstance(target, Enum):
        target = JobStatus(target)
    if target not in JOB_TRANSITIONS.get(current, frozenset()):
        raise IllegalTaskTransitionError(
            f"illegal job transition {current.value} -> {target.value}"
        )


def assert_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    if not isinstance(current, Enum):
        current = TaskStatus(current)
    if not isinstance(target, Enum):
        target = TaskStatus(target)
    if target not in TASK_TRANSITIONS.get(current, frozenset()):
        raise IllegalTaskTransitionError(
            f"illegal task transition {current.value} -> {target.value}"
        )


def assert_attempt_transition(current: AttemptStatus, target: AttemptStatus) -> None:
    if not isinstance(current, Enum):
        current = AttemptStatus(current)
    if not isinstance(target, Enum):
        target = AttemptStatus(target)
    if target not in ATTEMPT_TRANSITIONS.get(current, frozenset()):
        raise IllegalTaskTransitionError(
            f"illegal attempt transition {current.value} -> {target.value}"
        )


JOB_TERMINAL = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED})

# Claimable jobs -- a whitelist so a future JobStatus is never silently claimable
# by default. A task in a terminal (SUCCEEDED/FAILED/CANCELLED) or CANCELLING job
# must never be claimed: the claim path checks membership here rather than
# enumerating excluded statuses.
CLAIMABLE_JOB_STATUSES = frozenset(
    {JobStatus.PENDING, JobStatus.RUNNING, JobStatus.WAITING}
)

TASK_TERMINAL = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED}
)
ATTEMPT_TERMINAL = frozenset(
    {
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
        AttemptStatus.CANCELLED,
        AttemptStatus.SUPERSEDED,
    }
)


# ------------------------------------------------------------- json serde --
# The generic to_jsonable/from_jsonable pair lives in the neutral ``json`` module
# (shared by every store); re-exported here so existing
# ``from linktools.ai.task.models import to_jsonable`` references keep working.
from ..json import from_jsonable, to_jsonable  # noqa: E402,F401  (re-export)


__all__: "list[str]" = [
    "JobStatus",
    "TaskStatus",
    "AttemptStatus",
    "TaskFailureKind",
    "SideEffectMode",
    "ArtifactRef",
    "RetryPolicy",
    "SideEffectPolicy",
    "TaskPrincipal",
    "ActorRef",
    "ActorChain",
    "ScopeSet",
    "TaskBudget",
    "ResourceSnapshotRef",
    "JobRecord",
    "TaskRecord",
    "TaskWaitCondition",
    "TaskAttemptRecord",
    "TaskTransitionRecord",
    "TaskSignalRecord",
    "IllegalTaskTransitionError",
    "JOB_TRANSITIONS",
    "TASK_TRANSITIONS",
    "ATTEMPT_TRANSITIONS",
    "JOB_TERMINAL",
    "TASK_TERMINAL",
    "ATTEMPT_TERMINAL",
    "CLAIMABLE_JOB_STATUSES",
    "assert_job_transition",
    "assert_task_transition",
    "assert_attempt_transition",
    "resolve_effective_scopes",
    "narrow_child_principal",
    "to_jsonable",
    "from_jsonable",
]
