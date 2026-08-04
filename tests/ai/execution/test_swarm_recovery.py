#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery ownership and child-cancellation convergence tests."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.execution.domain import RunStatus, RunUsage
from linktools.ai.execution.identifiers import child_run_id
from linktools.ai.execution.snapshots import RunSnapshot
from linktools.ai.storage.coordination.lease import Lease
from linktools.ai.tasks.models import TaskStatus
from linktools.ai.tasks.persistence.local import LocalTaskBackend

from tests.ai.tasks.swarm._support import make_plan, ready_executions
from tests.ai.tasks.swarm.test_swarm_service import (
    FakeExecutionService,
    _persisted_swarm_run,
    _principal,
    _service,
    _spec,
)


@pytest.mark.asyncio
async def test_recovery_takes_over_expired_claim_with_new_fence():
    tasks = LocalTaskBackend()
    service, store = _service(fake_exec=FakeExecutionService(), tasks=tasks)
    plan = make_plan(("a",))
    spec = _spec(agents=("a",))
    parent_id = "parent-recovery"
    child_id = child_run_id(parent_id, "a")
    store._runs[parent_id] = _persisted_swarm_run(parent_id, spec, plan)
    child = replace(
        _persisted_swarm_run(child_id, spec, plan),
        parent_execution_id=parent_id,
        root_execution_id=parent_id,
        status=RunStatus.COMPLETED,
    )
    store._runs[child_id] = child
    store._snapshots[child_id] = RunSnapshot(
        schema="run-snapshot.v1",
        run_id=child_id,
        revision=1,
        resume_messages=(),
        final_output={"recovered": True},
        status=RunStatus.COMPLETED,
        usage=RunUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        trace_end_sequence=0,
        created_at=datetime.now(timezone.utc),
    )
    initial = replace(
        ready_executions(plan)[0],
        status=TaskStatus.CLAIMED,
        attempt=1,
        active_run_id=child_id,
        lease=Lease(
            owner=f"swarm:{parent_id}",
            fence=1,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        ),
    )
    await tasks.create_plan(plan, (initial,))

    outcome = await service.recover_swarm(parent_id, principal=_principal())

    current = (await tasks.list_executions(plan.id))[0]
    assert current.status is TaskStatus.COMPLETED
    assert current.fence == 2
    assert current.usage.input_tokens == 4
    assert outcome.collect["nodes"]["a"]["output"] == {"recovered": True}


@pytest.mark.asyncio
async def test_recovery_timeout_keeps_parent_and_claim_nonterminal(monkeypatch):
    import linktools.ai.execution.swarm_service as swarm_service_module

    monkeypatch.setattr(swarm_service_module, "RECOVERY_CHILD_CANCEL_TIMEOUT", 0.0)
    monkeypatch.setattr(swarm_service_module, "RECOVERY_CHILD_POLL_INTERVAL", 0.0)
    tasks = LocalTaskBackend()
    fake = FakeExecutionService()
    service, store = _service(fake_exec=fake, tasks=tasks)
    plan = make_plan(("a",))
    spec = _spec(agents=("a",))
    parent_id = "parent-timeout"
    child_id = child_run_id(parent_id, "a")
    store._runs[parent_id] = _persisted_swarm_run(parent_id, spec, plan)
    store._runs[child_id] = replace(
        _persisted_swarm_run(child_id, spec, plan),
        parent_execution_id=parent_id,
        root_execution_id=parent_id,
        status=RunStatus.RUNNING,
    )
    store._snapshots[child_id] = RunSnapshot(
        schema="run-snapshot.v1",
        run_id=child_id,
        revision=1,
        resume_messages=(),
        final_output=None,
        status=RunStatus.RUNNING,
        usage=RunUsage(input_tokens=8, output_tokens=1, total_tokens=9),
        trace_end_sequence=0,
        created_at=datetime.now(timezone.utc),
    )
    initial = replace(
        ready_executions(plan)[0],
        status=TaskStatus.CLAIMED,
        attempt=1,
        active_run_id=child_id,
        lease=Lease(
            owner=f"swarm:{parent_id}",
            fence=1,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        ),
    )
    await tasks.create_plan(plan, (initial,))

    outcome = await service.recover_swarm(parent_id, principal=_principal())

    current = (await tasks.list_executions(plan.id))[0]
    parent = await store.get_run(parent_id)
    assert outcome.error.error_type == "child_cancel_not_converged"
    assert parent.status is RunStatus.RUNNING
    assert current.status is TaskStatus.CLAIMED
    assert current.active_run_id == child_id
    assert current.usage.input_tokens == 0
