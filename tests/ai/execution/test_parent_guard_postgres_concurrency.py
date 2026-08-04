#!/usr/bin/env python3
"""PostgreSQL concurrency contract for fenced task child starts."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from linktools.ai.errors import ParentLeaseGuardError
from linktools.ai.execution.commands import (
    ClaimExecution,
    ParentLeaseGuard,
    StartExecution,
)
from linktools.ai.execution.domain import RunDefinition, RunKind, RunnableType


def _definition(runnable_id: str) -> RunDefinition:
    return RunDefinition(
        runnable_id,
        RunnableType.TASK,
        "swarm-task-graph.v1",
        {},
        f"hash-{runnable_id}",
    )


@pytest.mark.asyncio
async def test_postgresql_parent_guard_and_child_insert_are_transactional():
    dsn = os.environ.get("LINKTOOLS_AI_TEST_POSTGRESQL_DSN")
    if not dsn:
        pytest.skip(
            "LINKTOOLS_AI_TEST_POSTGRESQL_DSN not set; real PostgreSQL is required"
        )
    pytest.importorskip("asyncpg")
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import select, update
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from linktools.ai.execution.persistence.sqlalchemy import (
        ExecutionRow,
        SessionRow,
        SqlAlchemyExecutionBackend,
    )

    token = uuid4().hex
    session_id = f"pg-session-{token}"
    parent_id = f"pg-parent-{token}"
    child_id = f"pg-child-{token}"
    stale_child_id = f"pg-stale-child-{token}"
    definition = _definition(f"task-{token}")
    engine_one = create_async_engine(dsn, poolclass=NullPool)
    engine_two = create_async_engine(dsn, poolclass=NullPool)
    store_one = SqlAlchemyExecutionBackend(
        async_sessionmaker(engine_one, expire_on_commit=False)
    )
    store_two = SqlAlchemyExecutionBackend(
        async_sessionmaker(engine_two, expire_on_commit=False)
    )

    try:
        await store_one.initialize_storage(engine_one)
        await store_one.create_session(
            session_id=session_id,
            user_id="u",
            tenant_id="t",
        )
        started = await store_one.start_run(
            StartExecution(parent_id, session_id, RunKind.TASK, definition, {})
        )
        now = datetime.now(timezone.utc)
        parent = await store_one.claim_run(
            ClaimExecution(
                started.record.id,
                "scheduler",
                now,
                timedelta(minutes=5),
            )
        )
        guard = ParentLeaseGuard(parent.id, "scheduler", parent.lease.fence)
        child_command = StartExecution(
            child_id,
            session_id,
            RunKind.TASK,
            definition,
            {},
            root_execution_id=parent.id,
            parent_execution_id=parent.id,
            parent_guard=guard,
        )

        results = await asyncio.gather(
            store_one.start_run(child_command),
            store_two.start_run(child_command),
        )
        assert sorted(result.created for result in results) == [False, True]
        assert len(await store_one.list_runs_by_ids((child_id,))) == 1

        async with store_two.session_factory() as session:
            async with session.begin():
                await session.scalar(
                    select(SessionRow)
                    .where(SessionRow.session_id == session_id)
                    .with_for_update()
                )
                await session.scalar(
                    select(ExecutionRow)
                    .where(ExecutionRow.execution_id == parent.id)
                    .with_for_update()
                )
                blocked_start = asyncio.create_task(
                    store_one.start_run(
                        StartExecution(
                            stale_child_id,
                            session_id,
                            RunKind.TASK,
                            definition,
                            {},
                            root_execution_id=parent.id,
                            parent_execution_id=parent.id,
                            parent_guard=guard,
                        )
                    )
                )
                await session.execute(
                    update(ExecutionRow)
                    .where(ExecutionRow.execution_id == parent.id)
                    .values(
                        owner="recovery",
                        fence=parent.lease.fence + 1,
                        lease_expires_at=now + timedelta(minutes=5),
                    )
                )
            with pytest.raises(ParentLeaseGuardError):
                await blocked_start
        assert await store_one.get_run(stale_child_id) is None
    finally:
        await engine_one.dispose()
        await engine_two.dispose()
