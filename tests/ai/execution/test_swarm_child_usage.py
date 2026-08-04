#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Child runner usage and platform-error boundary tests."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import ChildExecutionPlatformError
from linktools.ai.execution.domain import RunStatus, RunUsage
from linktools.ai.execution.commands import ParentLeaseGuard
from linktools.ai.execution.identifiers import child_run_id
from linktools.ai.execution.live_events import NoopRunLiveEventSink
from linktools.ai.execution.service import PreparedAgentExecution
from linktools.ai.execution.snapshots import RunSnapshot
from linktools.ai.execution.swarm_service import _ChildNodeRunner
from linktools.ai.tasks.models import TaskUsage
from linktools.ai.tasks.swarm.engine import NodeRunRequest

from tests.ai.tasks.swarm._support import make_plan, ready_executions
from tests.ai.tasks.swarm.test_swarm_service import (
    FakeExecutionService,
    _fake_agent_spec,
    _persisted_swarm_run,
    _principal,
    _spec,
)


@dataclass
class _ReadOnlyExecutionStore:
    runs: dict[str, object] = field(default_factory=dict)
    snapshots: dict[str, RunSnapshot] = field(default_factory=dict)

    async def get_run(self, run_id: str):
        return self.runs.get(run_id)

    async def get_snapshot(self, run_id: str):
        return self.snapshots.get(run_id)


@pytest.mark.asyncio
async def test_platform_error_carries_persisted_child_usage_and_cause():
    plan = make_plan(("a",))
    parent_id = "parent"
    child_id = child_run_id(parent_id, "a")
    spec = _spec(agents=("a",))
    child = _persisted_swarm_run(child_id, spec, plan)
    store = _ReadOnlyExecutionStore(runs={child_id: child})
    store.snapshots[child_id] = RunSnapshot(
        schema="run-snapshot.v1",
        run_id=child_id,
        revision=1,
        resume_messages=(),
        final_output=None,
        status=RunStatus.FAILED,
        usage=RunUsage(input_tokens=19, output_tokens=7, total_tokens=26),
        trace_end_sequence=0,
        created_at=datetime.now(timezone.utc),
    )

    class ExplodingExecutionService(FakeExecutionService):
        async def run_child(self, *args, **kwargs):
            raise RuntimeError("provider internals must stay in cause only")

    agent_spec = _fake_agent_spec("a")
    prepared = PreparedAgentExecution(
        agent_spec=agent_spec,
        assembled_agent=object(),
        tool_descriptors=(),
        fingerprint="fingerprint:a",
    )
    runner = _ChildNodeRunner(
        agent_execution=ExplodingExecutionService(),
        principal=_principal(),
        session_id="s",
        parent_run_id=parent_id,
        root_execution_id=parent_id,
        message_history=(),
        execution_store=store,
        live_events=NoopRunLiveEventSink(),
        prepared_agents={"a": prepared},
    )

    with pytest.raises(ChildExecutionPlatformError) as raised:
        await runner.run(
            NodeRunRequest(
                node=plan.nodes[0],
                execution=ready_executions(plan)[0],
                owner="scheduler",
                fence=0,
                child_run_id=child_id,
                dependencies=(),
                parent_guard=ParentLeaseGuard("parent", "scheduler", 0),
            )
        )

    assert raised.value.child_run_id == child_id
    assert raised.value.usage == TaskUsage(input_tokens=19, output_tokens=7)
    assert isinstance(raised.value.cause, RuntimeError)
    assert raised.value.safe_message == "child execution failed"


@pytest.mark.asyncio
async def test_child_runner_reads_usage_without_task_store_access():
    plan = make_plan(("a",))
    child_id = child_run_id("parent", "a")
    spec = _spec(agents=("a",))
    store = _ReadOnlyExecutionStore()
    store.runs[child_id] = _persisted_swarm_run(child_id, spec, plan)
    store.snapshots[child_id] = RunSnapshot(
        schema="run-snapshot.v1",
        run_id=child_id,
        revision=1,
        resume_messages=(),
        final_output=None,
        status=RunStatus.CANCELLED,
        usage=RunUsage(input_tokens=3, output_tokens=2, total_tokens=5),
        trace_end_sequence=0,
        created_at=datetime.now(timezone.utc),
    )
    runner = _ChildNodeRunner(
        agent_execution=FakeExecutionService(),
        principal=_principal(),
        session_id="s",
        parent_run_id="parent",
        root_execution_id="parent",
        message_history=(),
        execution_store=store,
        live_events=NoopRunLiveEventSink(),
        prepared_agents={},
    )

    snapshot = await runner.read_usage(child_run_id=child_id)
    assert snapshot.usage == TaskUsage(3, 2)
    assert snapshot.snapshot_revision == 1
