#!/usr/bin/env python3
"""Parent fence validation belongs to the execution-store child-start write."""

from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.errors import ParentLeaseGuardError, RunIdentityConflictError
from linktools.ai.execution.commands import (
    ClaimExecution,
    ParentLeaseGuard,
    StartExecution,
    StartClaimedChildExecution,
)
from linktools.ai.execution.domain import (
    RunDefinition,
    RunKind,
    RunStatus,
    RunnableType,
    compute_run_definition_hash,
)
from linktools.ai.execution.persistence.local import LocalExecutionBackend


def _definition() -> RunDefinition:
    schema = "swarm-task-graph.v1"
    spec = {}
    return RunDefinition(
        "child",
        RunnableType.TASK,
        schema,
        spec,
        compute_run_definition_hash(schema=schema, spec=spec),
    )


async def _assert_guarded_start(store) -> None:
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    definition = _definition()
    started = await store.start_run(
        StartExecution("parent", "s", RunKind.TASK, definition, {})
    )
    parent = await store.claim_run(
        ClaimExecution(
            started.record.id,
            "scheduler",
            datetime.now(timezone.utc),
            timedelta(minutes=5),
        )
    )
    guard = ParentLeaseGuard(parent.id, "scheduler", parent.lease.fence)
    child = await store.start_claimed_child(
        StartClaimedChildExecution(
            StartExecution(
                "child",
                "s",
                RunKind.TASK,
                definition,
                {},
                root_execution_id=parent.id,
                parent_execution_id=parent.id,
                parent_guard=guard,
            ),
            "swarm",
            timedelta(minutes=5),
        )
    )
    assert child.record.id == "child"
    assert child.record.status is RunStatus.RUNNING
    assert child.record.lease.owner == "swarm"
    assert child.record.lease.fence == 1
    events = (await store.list_run_events("child")).items
    assert tuple(event.type for event in events) == ("run.started", "run.claimed")
    assert all(
        (
            event.created_at
            if event.created_at.tzinfo is not None
            else event.created_at.replace(tzinfo=timezone.utc)
        )
        == child.record.created_at
        for event in events
    )
    assert child.record.updated_at == child.record.created_at
    with pytest.raises(ParentLeaseGuardError):
        await store.start_run(
            StartExecution(
                "pending-child",
                "s",
                RunKind.TASK,
                definition,
                {},
                root_execution_id=parent.id,
                parent_execution_id=parent.id,
                parent_guard=guard,
            )
        )
    assert await store.get_run("pending-child") is None
    with pytest.raises(ParentLeaseGuardError):
        await store.start_claimed_child(
            StartClaimedChildExecution(
                StartExecution(
                    "child", "s", RunKind.TASK, definition, {},
                    root_execution_id=parent.id,
                    parent_execution_id=parent.id,
                    parent_guard=ParentLeaseGuard(parent.id, "stale", 0),
                ),
                "swarm",
                timedelta(minutes=5),
            )
        )
    with pytest.raises(ParentLeaseGuardError):
        await store.start_claimed_child(
            StartClaimedChildExecution(
                StartExecution(
                    "stale-child", "s", RunKind.TASK, definition, {},
                    root_execution_id=parent.id,
                    parent_execution_id=parent.id,
                    parent_guard=ParentLeaseGuard(
                        parent.id, "scheduler", parent.lease.fence - 1
                    ),
                ),
                "swarm",
                timedelta(minutes=5),
            )
        )
    assert await store.get_run("stale-child") is None
    with pytest.raises(ParentLeaseGuardError):
        await store.start_claimed_child(
            StartClaimedChildExecution(
                StartExecution(
                    "missing-guard", "s", RunKind.TASK, definition, {},
                    root_execution_id=parent.id,
                    parent_execution_id=parent.id,
                ),
                "swarm",
                timedelta(minutes=5),
            )
        )
    assert await store.get_run("missing-guard") is None
    with pytest.raises(ParentLeaseGuardError):
        await store.start_claimed_child(
            StartClaimedChildExecution(
                StartExecution(
                    "mismatched-guard", "s", RunKind.TASK, definition, {},
                    root_execution_id=parent.id,
                    parent_execution_id=parent.id,
                    parent_guard=ParentLeaseGuard(
                        "other-parent", "scheduler", parent.lease.fence
                    ),
                ),
                "swarm",
                timedelta(minutes=5),
            )
        )
    assert await store.get_run("mismatched-guard") is None
    with pytest.raises(RunIdentityConflictError):
        await store.start_claimed_child(
            StartClaimedChildExecution(
                StartExecution(
                    "child", "s", RunKind.TASK,
                    RunDefinition(
                        "different", RunnableType.TASK, "swarm-task-graph.v1", {},
                        compute_run_definition_hash(
                            schema="swarm-task-graph.v1", spec={}
                        ),
                    ),
                    {},
                    root_execution_id=parent.id,
                    parent_execution_id=parent.id,
                    parent_guard=guard,
                    ),
                    "swarm",
                    timedelta(minutes=5),
            )
        )


@pytest.mark.asyncio
async def test_local_child_start_checks_parent_guard_atomically(tmp_path):
    store = LocalExecutionBackend(tmp_path / "execution")
    await store.initialize_storage()
    await _assert_guarded_start(store)


@pytest.mark.asyncio
async def test_sql_child_start_checks_parent_guard_atomically(tmp_path):
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from linktools.ai.execution.persistence.sqlalchemy import SqlAlchemyExecutionBackend

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'execution.db'}")
    store = SqlAlchemyExecutionBackend(
        async_sessionmaker(engine, expire_on_commit=False)
    )
    await store.initialize_storage(engine)
    try:
        await _assert_guarded_start(store)
    finally:
        await engine.dispose()
