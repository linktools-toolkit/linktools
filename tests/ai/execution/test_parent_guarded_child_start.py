#!/usr/bin/env python3
"""Parent fence validation belongs to the execution-store child-start write."""

from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.errors import ParentLeaseGuardError, StorageCorruptionError
from linktools.ai.execution.commands import (
    ClaimExecution,
    ParentLeaseGuard,
    StartExecution,
)
from linktools.ai.execution.domain import RunDefinition, RunKind, RunnableType
from linktools.ai.execution.persistence.local import LocalExecutionBackend


def _definition() -> RunDefinition:
    return RunDefinition("child", RunnableType.TASK, "swarm-task-graph.v1", {}, "hash")


async def _assert_guarded_start(store) -> None:
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    definition = _definition()
    parent = await store.start_run(
        StartExecution("parent", "s", RunKind.TASK, definition, {})
    )
    parent = await store.claim_run(
        ClaimExecution(
            parent.id,
            "scheduler",
            datetime.now(timezone.utc),
            timedelta(minutes=5),
        )
    )
    guard = ParentLeaseGuard(parent.id, "scheduler", parent.lease.fence)
    child = await store.start_run(
        StartExecution(
            "child",
            "s",
            RunKind.TASK,
            definition,
            {},
            root_execution_id=parent.id,
            parent_execution_id=parent.id,
            parent_guard=guard,
        )
    )
    assert child.id == "child"
    assert (
        await store.start_run(
            StartExecution(
                "child",
                "s",
                RunKind.TASK,
                definition,
                {},
                root_execution_id=parent.id,
                parent_execution_id=parent.id,
                parent_guard=ParentLeaseGuard(parent.id, "stale", 0),
            )
        )
    ).id == child.id
    with pytest.raises(ParentLeaseGuardError):
        await store.start_run(
            StartExecution(
                "stale-child",
                "s",
                RunKind.TASK,
                definition,
                {},
                root_execution_id=parent.id,
                parent_execution_id=parent.id,
                parent_guard=ParentLeaseGuard(
                    parent.id, "scheduler", parent.lease.fence - 1
                ),
            )
        )
    assert await store.get_run("stale-child") is None
    with pytest.raises(ParentLeaseGuardError):
        await store.start_run(
            StartExecution(
                "missing-guard",
                "s",
                RunKind.TASK,
                definition,
                {},
                root_execution_id=parent.id,
                parent_execution_id=parent.id,
            )
        )
    assert await store.get_run("missing-guard") is None
    with pytest.raises(ParentLeaseGuardError):
        await store.start_run(
            StartExecution(
                "mismatched-guard",
                "s",
                RunKind.TASK,
                definition,
                {},
                root_execution_id=parent.id,
                parent_execution_id=parent.id,
                parent_guard=ParentLeaseGuard(
                    "other-parent", "scheduler", parent.lease.fence
                ),
            )
        )
    assert await store.get_run("mismatched-guard") is None
    with pytest.raises(StorageCorruptionError):
        await store.start_run(
            StartExecution(
                "child",
                "s",
                RunKind.TASK,
                RunDefinition(
                    "different",
                    RunnableType.TASK,
                    "swarm-task-graph.v1",
                    {},
                    "different",
                ),
                {},
                root_execution_id=parent.id,
                parent_execution_id=parent.id,
                parent_guard=guard,
            )
        )


@pytest.mark.asyncio
async def test_local_child_start_checks_parent_guard_atomically(tmp_path):
    store = LocalExecutionBackend(tmp_path / "execution")
    await store.initialize_storage()
    await _assert_guarded_start(store)


@pytest.mark.asyncio
async def test_sql_child_start_checks_parent_guard_atomically(tmp_path):
    sqlalchemy = pytest.importorskip("sqlalchemy")
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
