#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persisted swarm integrity gates."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.errors import InvalidSpecError
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
async def test_validate_rejects_definition_hash_tampering():
    tasks = LocalTaskBackend()
    service, store = _service(fake_exec=FakeExecutionService(), tasks=tasks)
    plan = make_plan(("a",))
    spec = _spec(agents=("a",))
    record = _persisted_swarm_run("parent", spec, plan)
    object.__setattr__(
        record.definition,
        "spec",
        {**record.definition.spec, "task_plan_id": "tampered"},
    )
    store._runs[record.id] = record
    await tasks.create_plan(plan, ready_executions(plan))

    with pytest.raises(InvalidSpecError, match="definition_integrity"):
        await service.validate_persisted_swarm_run(
            store._runs[record.id], principal=_principal()
        )


@pytest.mark.asyncio
async def test_validate_rejects_non_deterministic_child_id():
    tasks = LocalTaskBackend()
    service, store = _service(fake_exec=FakeExecutionService(), tasks=tasks)
    plan = make_plan(("a",))
    spec = _spec(agents=("a",))
    record = _persisted_swarm_run("parent", spec, plan)
    store._runs[record.id] = record
    execution = replace(
        ready_executions(plan)[0],
        status=TaskStatus.CLAIMED,
        attempt=1,
        active_run_id="wrong-child",
        lease=Lease(
            owner="swarm:parent",
            fence=1,
            expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
        ),
    )
    await tasks.create_plan(plan, (execution,))

    with pytest.raises(InvalidSpecError, match="child_run_integrity"):
        await service.validate_persisted_swarm_run(
            record, principal=_principal()
        )
