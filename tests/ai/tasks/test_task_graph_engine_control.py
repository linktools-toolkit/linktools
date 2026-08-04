#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Control-loop and node-lease tests for the task-graph engine."""

from datetime import timedelta

import pytest

from linktools.ai.tasks.persistence.local import LocalTaskBackend
from linktools.ai.tasks.swarm.engine import TaskGraphEngine

from tests.ai.tasks.swarm._support import (
    NoopGate,
    RecordingRunner,
    make_plan,
    ready_executions,
)
from tests.ai.tasks.swarm.test_engine import _limits


@pytest.mark.asyncio
async def test_immediate_completion_uses_fresh_authoritative_state():
    store = LocalTaskBackend()
    plan = make_plan(("a", "b"))
    await store.create_plan(plan, ready_executions(plan))
    runner = RecordingRunner(results={"a": "A", "b": "B"}, fast={"a", "b"})
    engine = TaskGraphEngine(
        store=store,
        runner=runner,
        gate=NoopGate(),
        limits=_limits(max_concurrency=2),
        owner="scheduler",
        parent_run_id="parent",
    )

    await engine.execute(plan)

    assert runner.counts == {"a": 1, "b": 1}
    assert engine.active_count == 0


@pytest.mark.asyncio
async def test_node_renewal_writes_only_when_due():
    class CountingStore(LocalTaskBackend):
        renew_count = 0

        async def renew(self, *args, **kwargs):
            self.renew_count += 1
            return await super().renew(*args, **kwargs)

    store = CountingStore()
    plan = make_plan(("a",))
    await store.create_plan(plan, ready_executions(plan))
    runner = RecordingRunner()
    engine = TaskGraphEngine(
        store=store,
        runner=runner,
        gate=NoopGate(),
        limits=_limits(max_concurrency=1),
        owner="scheduler",
        parent_run_id="parent",
    )
    engine._node_lease_duration = timedelta(seconds=30)
    executions = {item.node_id: item for item in await store.list_executions(plan.id)}
    await engine._spawn_node(plan.nodes[0], executions)
    active = next(iter(engine._active_by_node.values()))

    await engine._renew_due_nodes(active.next_renew_at - 0.001)
    assert store.renew_count == 0
    await engine._renew_due_nodes(active.next_renew_at)
    assert store.renew_count == 1

    await engine._shutdown_active_nodes(
        plan,
        stop_reason="timeout",
        primary_error=None,
    )
    assert engine.active_count == 0
