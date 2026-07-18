#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task domain models: JSON round-trip, immutability, state-machine legality,
policy validation (plan section 28, phase-2 acceptance)."""

import dataclasses
import json
from datetime import datetime, timezone

import pytest

from linktools.ai.task.models import (
    ATTEMPT_TERMINAL,
    IllegalTaskTransitionError,
    JobRecord,
    JobStatus,
    RetryPolicy,
    SideEffectMode,
    SideEffectPolicy,
    TaskAttemptRecord,
    TaskBudget,
    TaskFailureKind,
    TaskPrincipal,
    TaskRecord,
    TaskSignalRecord,
    TaskStatus,
    TaskTransitionRecord,
    ActorChain,
    ActorRef,
    ScopeSet,
    ResourceSnapshotRef,
    assert_attempt_transition,
    assert_job_transition,
    assert_task_transition,
    from_jsonable,
    to_jsonable,
)

NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)
LATER = datetime(2026, 7, 16, 12, 5, tzinfo=timezone.utc)


def _principal() -> TaskPrincipal:
    return TaskPrincipal(tenant_id="t1", user_id="alice", workspace_key="ws")


def _actor_chain() -> ActorChain:
    return ActorChain(actors=(ActorRef("user", "alice"),), delegated_scopes=ScopeSet.of("read"))


def _budget() -> TaskBudget:
    return TaskBudget(max_tasks=10, max_depth=3, max_total_cost="1.25")


def _snapshot() -> ResourceSnapshotRef:
    return ResourceSnapshotRef(
        path="/skills/skill-a", version=1, etag="abc", artifact_id="aid", sha256="dead"
    )


def _job() -> JobRecord:
    return JobRecord(
        id="job-1",
        status=JobStatus.PENDING,
        principal=_principal(),
        actor_chain=_actor_chain(),
        budget=_budget(),
        root_task_id=None,
        input_artifact_id=None,
        output_artifact_id=None,
        version=1,
        created_at=NOW,
        started_at=None,
        finished_at=None,
        metadata={"k": "v"},
    )


def _task() -> TaskRecord:
    return TaskRecord(
        id="task-1",
        job_id="job-1",
        parent_task_id=None,
        key="do-thing",
        handler="runtime",
        status=TaskStatus.READY,
        input_artifact_id=None,
        output_artifact_id=None,
        dependencies=(),
        retry_policy=RetryPolicy(max_attempts=3),
        side_effect_policy=SideEffectPolicy(mode=SideEffectMode.IDEMPOTENT),
        attempt_count=0,
        available_at=NOW,
        lease_owner=None,
        lease_expires_at=None,
        fencing_token=0,
        active_attempt_id=None,
        timeout_seconds=30.0,
        resource_snapshots=(_snapshot(),),
        version=1,
        created_at=NOW,
        updated_at=NOW,
    )


def _attempt() -> TaskAttemptRecord:
    return TaskAttemptRecord(
        id="att-1",
        task_id="task-1",
        job_id="job-1",
        attempt=1,
        worker_id="w-1",
        fencing_token=1,
        status="running",
        started_at=NOW,
        run_id=None,
        finished_at=None,
        failure_kind=None,
        error_type=None,
        error_message=None,
    )


def _transition() -> TaskTransitionRecord:
    return TaskTransitionRecord(
        id="tr-1",
        job_id="job-1",
        task_id="task-1",
        attempt_id="att-1",
        from_status="ready",
        to_status="claimed",
        reason="claimed",
        occurred_at=NOW,
    )


def _signal() -> TaskSignalRecord:
    return TaskSignalRecord(
        id="sig-1",
        job_id="job-1",
        name="human-review",
        correlation_key="case-1",
        payload_artifact_id=None,
        created_at=NOW,
        consumed_by_task_id=None,
    )


CASES = [
    ("JobRecord", JobRecord, _job),
    ("TaskRecord", TaskRecord, _task),
    ("TaskAttemptRecord", TaskAttemptRecord, _attempt),
    ("TaskTransitionRecord", TaskTransitionRecord, _transition),
    ("TaskSignalRecord", TaskSignalRecord, _signal),
]


@pytest.mark.parametrize("name,cls,builder", CASES, ids=[c[0] for c in CASES])
def test_json_round_trip(name: str, cls: type, builder) -> None:
    original = builder()
    encoded = to_jsonable(original)
    json.dumps(encoded)  # JSON-native
    restored = from_jsonable(cls, encoded)
    assert restored == original


def test_decimal_cost_roundtrips_as_string() -> None:
    # max_total_cost stays a string through JSON (no float drift).
    job = _job()
    restored = from_jsonable(JobRecord, to_jsonable(job))
    assert restored.budget.max_total_cost == "1.25"


def test_records_are_frozen() -> None:
    for builder in (_job, _task, _attempt, _transition, _signal):
        obj = builder()
        first = next(iter(dataclasses.fields(obj)))
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, first.name, getattr(obj, first.name))


def test_retry_policy_validates_bounds() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(initial_delay_seconds=-1)
    with pytest.raises(ValueError):
        RetryPolicy(max_delay_seconds=0.5, initial_delay_seconds=1.0)
    with pytest.raises(ValueError):
        RetryPolicy(multiplier=0.5)
    with pytest.raises(ValueError):
        RetryPolicy(jitter_ratio=1.5)
    # valid policy constructs.
    assert RetryPolicy(max_attempts=3, retryable_kinds=(TaskFailureKind.TRANSIENT,))


def test_principal_requires_tenant() -> None:
    with pytest.raises(TypeError):
        TaskPrincipal()  # type: ignore[call-arg]
    assert TaskPrincipal(tenant_id="t1").tenant_id == "t1"


def test_legal_transitions_pass() -> None:
    assert_job_transition(JobStatus.PENDING, JobStatus.RUNNING)
    assert_task_transition(TaskStatus.READY, TaskStatus.CLAIMED)
    assert_task_transition(TaskStatus.CLAIMED, TaskStatus.RETRY_WAIT)
    assert_task_transition(TaskStatus.CLAIMED, TaskStatus.READY)  # lease expiry reclaim
    assert_attempt_transition("running", "succeeded")


@pytest.mark.parametrize("status", list(ATTEMPT_TERMINAL))
def test_terminal_attempt_is_sink(status) -> None:
    with pytest.raises(IllegalTaskTransitionError):
        assert_attempt_transition(status, "running")


def test_illegal_transition_rejected() -> None:
    with pytest.raises(IllegalTaskTransitionError):
        assert_task_transition(TaskStatus.SUCCEEDED, TaskStatus.CLAIMED)
    with pytest.raises(IllegalTaskTransitionError):
        assert_job_transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)
