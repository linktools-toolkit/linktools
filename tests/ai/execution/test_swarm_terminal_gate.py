#!/usr/bin/env python3
"""Parent terminal writes require every task and child to be converged."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from types import SimpleNamespace

import pytest

from linktools.ai.errors import ParentTerminalGateError, RunDefinitionIntegrityError
from linktools.ai.execution.domain import RunDefinition, RunKind, RunStatus, RunnableType
from linktools.ai.execution.swarm_service import SwarmExecutionService
from linktools.ai.execution.identifiers import child_run_id
from linktools.ai.json import canonical_json_bytes
from linktools.ai.storage.coordination.lease import Lease
from linktools.ai.tasks.codec import encode_plan
from linktools.ai.tasks.models import (
    TaskExecution,
    TaskGraphNodePayload,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from linktools.ai.tasks.persistence.local import LocalTaskBackend


class _GateExecutionStore:
    def __init__(self, parent, children=()):
        self.parent = parent
        self.children = dict(children)

    async def assert_active_lease(self, run_id: str, *, owner: str, fence: int):
        return None

    async def get_run(self, run_id: str):
        return self.parent

    async def list_runs_by_ids(self, run_ids):
        return tuple(
            self.children[run_id]
            for run_id in dict.fromkeys(run_ids)
            if run_id in self.children
        )


def _plan() -> TaskPlan:
    return TaskPlan(
        "gate-plan",
        (TaskNode("a", TaskGraphNodePayload("agent-a", "prompt")),),
    )


def _parent(plan: TaskPlan):
    snapshot = []
    snapshot_hash = sha256(canonical_json_bytes(snapshot)).hexdigest()
    value = {
        "task_plan_id": plan.id,
        "task_plan_hash": sha256(canonical_json_bytes(encode_plan(plan))).hexdigest(),
        "session_snapshot_hash": snapshot_hash,
        "agent_fingerprints": {"agent-a": "fingerprint:agent-a"},
    }
    definition = RunDefinition(
        "swarm",
        RunnableType.TASK,
        "swarm-task-graph.v1",
        value,
        sha256(canonical_json_bytes(value)).hexdigest(),
    )
    return SimpleNamespace(
        definition=definition,
        input={
            "task_plan_id": plan.id,
            "session_snapshot": snapshot,
            "session_snapshot_hash": snapshot_hash,
        },
    )


def _service(store, tasks):
    return SwarmExecutionService(
        store=store,
        tasks=tasks,
        agent_execution=object(),
        authorization=object(),
        live_events=object(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (TaskStatus.READY, TaskStatus.CLAIMED))
async def test_parent_gate_rejects_nonterminal_task_execution(status):
    plan = _plan()
    lease = (
        Lease(
            owner="other-worker",
            fence=3,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        )
        if status is TaskStatus.CLAIMED
        else Lease()
    )
    execution = TaskExecution(
        "execution",
        plan.id,
        "a",
        status,
        lease=lease,
        attempt=1 if status is TaskStatus.CLAIMED else 0,
    )
    tasks = LocalTaskBackend()
    await tasks.create_plan(plan, (execution,))
    store = _GateExecutionStore(_parent(plan))

    with pytest.raises(ParentTerminalGateError):
        await _service(store, tasks)._assert_parent_terminal_gate(
            "parent", "owner", 1, plan
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("child_status", (RunStatus.RUNNING, RunStatus.CANCELLING))
async def test_parent_gate_checks_public_child_records_before_terminal_write(child_status):
    plan = _plan()
    execution = TaskExecution(
        "execution",
        plan.id,
        "a",
        TaskStatus.COMPLETED,
        active_run_id=child_run_id("parent", "a"),
    )
    tasks = LocalTaskBackend()
    await tasks.create_plan(plan, (execution,))
    store = _GateExecutionStore(
        _parent(plan),
        children=((child_run_id("parent", "a"), SimpleNamespace(status=child_status)),),
    )

    with pytest.raises(ParentTerminalGateError):
        await _service(store, tasks)._assert_parent_terminal_gate(
            "parent", "owner", 1, plan
        )


@pytest.mark.asyncio
async def test_parent_gate_rejects_missing_child_record():
    plan = _plan()
    execution = TaskExecution(
        "execution",
        plan.id,
        "a",
        TaskStatus.COMPLETED,
        active_run_id="missing-child",
    )
    tasks = LocalTaskBackend()
    await tasks.create_plan(plan, (execution,))

    with pytest.raises(RunDefinitionIntegrityError):
        await _service(
            _GateExecutionStore(_parent(plan)), tasks
        )._assert_parent_terminal_gate("parent", "owner", 1, plan)


@pytest.mark.asyncio
async def test_parent_gate_rejects_active_engine_task():
    plan = _plan()
    execution = TaskExecution(
        "execution",
        plan.id,
        "a",
        TaskStatus.COMPLETED,
        active_run_id="child",
    )
    tasks = LocalTaskBackend()
    await tasks.create_plan(plan, (execution,))
    store = _GateExecutionStore(
        _parent(plan),
        children=(("child", SimpleNamespace(status=RunStatus.COMPLETED)),),
    )

    with pytest.raises(ParentTerminalGateError):
        await _service(store, tasks)._assert_parent_terminal_gate(
            "parent", "owner", 1, plan, SimpleNamespace(active_count=1)
        )
