#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for swarm.models, swarm.store.SwarmStore, and the
SwarmError family in errors.py. Pure data/Protocol checks -- no I/O."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from linktools.ai.errors import (
    InvalidSwarmTransitionError,
    LinktoolsAIError,
    SwarmConflictError,
    SwarmError,
    SwarmLimitExceededError,
    SwarmRunNotFoundError,
    SwarmTaskNotFoundError,
)
from linktools.ai.run.models import RunErrorInfo, RunResult
from linktools.ai.swarm.models import (
    ALLOWED_SWARM_TRANSITIONS,
    AgentRef,
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskStatus,
    TaskInput,
    TokenUsage,
)
from linktools.ai.swarm.store import SwarmStore


# --- SwarmStatus enum --------------------------------------------------------


def test_swarm_status_values():
    assert SwarmStatus.PENDING.value == "pending"
    assert SwarmStatus.RUNNING.value == "running"
    assert SwarmStatus.PAUSED.value == "paused"
    assert SwarmStatus.SUCCEEDED.value == "succeeded"
    assert SwarmStatus.FAILED.value == "failed"
    assert SwarmStatus.CANCELLED.value == "cancelled"


def test_swarm_status_is_str_enum():
    assert isinstance(SwarmStatus.PENDING, str)
    assert SwarmStatus.PENDING == "pending"


# --- SwarmTaskStatus enum ----------------------------------------------------


def test_swarm_task_status_values():
    assert SwarmTaskStatus.PENDING.value == "pending"
    assert SwarmTaskStatus.CLAIMED.value == "claimed"
    assert SwarmTaskStatus.SUCCEEDED.value == "succeeded"
    assert SwarmTaskStatus.FAILED.value == "failed"
    assert SwarmTaskStatus.CANCELLED.value == "cancelled"


# --- ALLOWED_SWARM_TRANSITIONS ----------------------------------------------


def test_allowed_swarm_transitions_pending():
    assert ALLOWED_SWARM_TRANSITIONS[SwarmStatus.PENDING] == frozenset(
        {SwarmStatus.RUNNING}
    )


def test_allowed_swarm_transitions_running():
    assert ALLOWED_SWARM_TRANSITIONS[SwarmStatus.RUNNING] == frozenset(
        {
            SwarmStatus.PAUSED,
            SwarmStatus.SUCCEEDED,
            SwarmStatus.FAILED,
            SwarmStatus.CANCELLING,
            SwarmStatus.CANCELLED,
        }
    )


def test_allowed_swarm_transitions_paused():
    assert ALLOWED_SWARM_TRANSITIONS[SwarmStatus.PAUSED] == frozenset(
        {
            SwarmStatus.RUNNING,
            SwarmStatus.CANCELLING,
            SwarmStatus.CANCELLED,
        }
    )


def test_allowed_swarm_transitions_cancelling():
    """CANCELLING distinguishes "cancel requested" from "actually cancelled"
    (mirrors RunStatus.CANCELLING) -- actionable-fix-contract."""
    assert ALLOWED_SWARM_TRANSITIONS[SwarmStatus.CANCELLING] == frozenset(
        {
            SwarmStatus.CANCELLED,
            SwarmStatus.FAILED,
        }
    )


@pytest.mark.parametrize(
    "status",
    [
        SwarmStatus.SUCCEEDED,
        SwarmStatus.FAILED,
        SwarmStatus.CANCELLED,
    ],
)
def test_allowed_swarm_transitions_terminals_empty(status):
    assert ALLOWED_SWARM_TRANSITIONS[status] == frozenset()


# --- AgentRef ---------------------------------------------------------------


def test_agent_ref_defaults():
    ref = AgentRef("a1")
    assert ref.agent_id == "a1"
    assert ref.role is None


def test_agent_ref_frozen():
    ref = AgentRef("a1", role="worker")
    with pytest.raises(FrozenInstanceError):
        ref.agent_id = "a2"


# --- TaskInput --------------------------------------------------------------


def test_task_input_defaults():
    ti = TaskInput(prompt="hi")
    assert ti.prompt == "hi"
    assert ti.metadata == {}


# --- TokenUsage -------------------------------------------------------------


def test_token_usage_defaults():
    tu = TokenUsage()
    assert tu.input_tokens == 0
    assert tu.output_tokens == 0
    assert tu.total_cost == Decimal("0")


def test_token_usage_add():
    a = TokenUsage(10, 5)
    b = TokenUsage(3, 2)
    assert a.add(b) == TokenUsage(13, 7, Decimal("0"))


def test_token_usage_add_is_immutable():
    a = TokenUsage(10, 5)
    b = TokenUsage(3, 2)
    result = a.add(b)
    assert a == TokenUsage(10, 5)
    assert result == TokenUsage(13, 7, Decimal("0"))


def test_token_usage_from_mapping():
    tu = TokenUsage.from_mapping({"input_tokens": 100, "output_tokens": 50})
    assert tu == TokenUsage(100, 50)


def test_token_usage_from_mapping_missing_keys():
    tu = TokenUsage.from_mapping({})
    assert tu == TokenUsage()


def test_token_usage_from_mapping_none_values():
    """None values are rejected (strict parsing, no implicit int() coercion)."""
    with pytest.raises(ValueError, match="input_tokens"):
        TokenUsage.from_mapping({"input_tokens": None, "output_tokens": None})


# --- SwarmRun ---------------------------------------------------------------


def _now():
    return datetime.now(timezone.utc)


def test_swarm_run_construct():
    now = _now()
    run = SwarmRun(
        id="sr-1",
        run_id="r-1",
        round=0,
        status=SwarmStatus.PENDING,
        version=0,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=now,
        updated_at=now,
    )
    assert run.id == "sr-1"
    assert run.run_id == "r-1"
    assert run.round == 0
    assert run.status is SwarmStatus.PENDING
    assert run.version == 0
    assert run.token_usage == TokenUsage()
    assert run.cost == Decimal("0")
    assert run.created_at == now
    assert run.updated_at == now
    assert run.metadata == {}


def test_swarm_run_frozen():
    now = _now()
    run = SwarmRun(
        id="sr-1",
        run_id="r-1",
        round=0,
        status=SwarmStatus.PENDING,
        version=0,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(FrozenInstanceError):
        run.status = SwarmStatus.RUNNING


# --- SwarmTask --------------------------------------------------------------


def test_swarm_task_construct_with_result_and_error():
    now = _now()
    result = RunResult(output="done")
    error = RunErrorInfo(error_type="ValueError", message="boom")
    task = SwarmTask(
        id="t-1",
        swarm_run_id="sr-1",
        parent_task_id="t-0",
        assigned_agent_id="a-1",
        description="do the thing",
        status=SwarmTaskStatus.SUCCEEDED,
        dependencies=("t-0",),
        input=TaskInput(prompt="hi", metadata={"k": 1}),
        result=result,
        error=error,
        attempts=2,
        version=3,
        claimed_at=now,
        lease_expires_at=now,
        created_at=now,
        updated_at=now,
    )
    assert task.parent_task_id == "t-0"
    assert task.assigned_agent_id == "a-1"
    assert task.status is SwarmTaskStatus.SUCCEEDED
    assert task.dependencies == ("t-0",)
    assert task.input.metadata == {"k": 1}
    assert task.result is result
    assert task.error is error
    assert task.attempts == 2
    assert task.version == 3
    assert task.claimed_at == now
    assert task.lease_expires_at == now


def test_swarm_task_construct():
    now = _now()
    task = SwarmTask(
        id="t-1",
        swarm_run_id="sr-1",
        parent_task_id=None,
        assigned_agent_id=None,
        description="do the thing",
        status=SwarmTaskStatus.PENDING,
        dependencies=(),
        input=TaskInput(prompt="hi"),
        result=None,
        error=None,
        attempts=0,
        version=0,
        claimed_at=None,
        lease_expires_at=None,
        created_at=now,
        updated_at=now,
    )
    assert task.id == "t-1"
    assert task.swarm_run_id == "sr-1"
    assert task.parent_task_id is None
    assert task.assigned_agent_id is None
    assert task.description == "do the thing"
    assert task.status is SwarmTaskStatus.PENDING
    assert task.dependencies == ()
    assert task.input == TaskInput(prompt="hi")
    assert task.result is None
    assert task.error is None
    assert task.attempts == 0
    assert task.version == 0
    assert task.claimed_at is None
    assert task.lease_expires_at is None
    assert task.created_at == now
    assert task.updated_at == now


def test_swarm_task_frozen():
    now = _now()
    task = SwarmTask(
        id="t-1",
        swarm_run_id="sr-1",
        parent_task_id=None,
        assigned_agent_id=None,
        description="x",
        status=SwarmTaskStatus.PENDING,
        dependencies=(),
        input=TaskInput(prompt="hi"),
        result=None,
        error=None,
        attempts=0,
        version=0,
        claimed_at=None,
        lease_expires_at=None,
        created_at=now,
        updated_at=now,
    )
    with pytest.raises(FrozenInstanceError):
        task.status = SwarmTaskStatus.CLAIMED


# --- SwarmError family ------------------------------------------------------


def test_swarm_error_is_linktools_ai_error():
    assert issubclass(SwarmError, LinktoolsAIError)


@pytest.mark.parametrize(
    "exc_cls",
    [
        SwarmRunNotFoundError,
        SwarmTaskNotFoundError,
        SwarmConflictError,
        InvalidSwarmTransitionError,
        SwarmLimitExceededError,
    ],
)
def test_swarm_error_subclasses(exc_cls):
    assert issubclass(exc_cls, SwarmError)


def test_swarm_limit_exceeded_carries_kind():
    err = SwarmLimitExceededError("too many", kind="max_rounds")
    assert err.kind == "max_rounds"
    assert str(err) == "too many"


def test_swarm_limit_exceeded_raises_as_swarm_error():
    with pytest.raises(SwarmError):
        raise SwarmLimitExceededError("msg", kind="max_rounds")


# --- SwarmStore Protocol ----------------------------------------------------


class _StubStore:
    async def create_run(self, run): ...

    async def get_run(self, swarm_run_id): ...

    async def update_run(
        self,
        swarm_run_id,
        *,
        expected_version,
        status=None,
        round=None,
        token_usage=None,
        cost=None,
        metadata=None,
    ): ...

    async def create_task(self, task): ...

    async def claim_task(self, swarm_run_id, agent_id, *, lease_seconds=None): ...

    async def set_active_run(self, task_id, run_id, *, expected_version): ...

    async def complete_task(self, task_id, result): ...

    async def fail_task(self, task_id, error): ...

    async def list_tasks(self, swarm_run_id, *, status=None): ...

    async def reclaim_expired_tasks(self, swarm_run_id): ...

    async def record_attempt(self, attempt): ...

    async def list_attempts(self, task_id): ...

    async def renew_lease(self, task_id, *, expected_version, lease_seconds): ...


def test_swarm_store_is_runtime_checkable():
    assert isinstance(_StubStore(), SwarmStore)


def test_swarm_store_rejects_non_implementor():
    class _Incomplete:
        async def create_run(self, run): ...

    assert not isinstance(_Incomplete(), SwarmStore)
