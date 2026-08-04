#!/usr/bin/env python3
"""PostgreSQL must serialize usage checkpoints with an exact revision CAS."""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from linktools.ai.execution.commands import (
    CheckpointExecutionUsage,
    ClaimExecution,
    StartExecution,
)
from linktools.ai.execution.domain import (
    RunDefinition,
    RunKind,
    RunUsage,
    RunnableType,
    compute_run_definition_hash,
)
from linktools.ai.execution.persistence.sqlalchemy import SqlAlchemyExecutionBackend


@pytest.mark.asyncio
async def test_postgresql_checkpoint_cas_has_one_winner():
    dsn = os.environ.get("LINKTOOLS_AI_TEST_POSTGRESQL_DSN")
    if not dsn:
        pytest.skip("LINKTOOLS_AI_TEST_POSTGRESQL_DSN not set; PostgreSQL is required")
    pytest.importorskip("asyncpg")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    token = uuid4().hex
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
        session_id = f"checkpoint-session-{token}"
        run_id = f"checkpoint-run-{token}"
        spec = {"id": "agent"}
        definition = RunDefinition(
            "agent",
            RunnableType.AGENT,
            "agent-spec.v1",
            spec,
            compute_run_definition_hash(schema="agent-spec.v1", spec=spec),
        )
        await store_one.create_session(
            session_id=session_id, user_id="u", tenant_id="t"
        )
        await store_one.start_run(
            StartExecution(run_id, session_id, RunKind.USER_TURN, definition, "prompt")
        )
        claimed = await store_one.claim_run(
            ClaimExecution(
                run_id,
                "worker",
                datetime.now(timezone.utc),
                timedelta(minutes=5),
            )
        )
        usage = RunUsage(
            input_tokens=2,
            output_tokens=1,
            total_tokens=3,
            total_cost=Decimal("0.2"),
        )
        command = CheckpointExecutionUsage(
            run_id,
            "worker",
            claimed.lease.fence,
            0,
            usage,
            1,
        )
        results = await asyncio.gather(
            store_one.checkpoint_run_usage(command),
            store_two.checkpoint_run_usage(command),
            return_exceptions=True,
        )
        assert sum(not isinstance(result, Exception) for result in results) == 2
        assert all(result.revision == 1 for result in results)
        record = await store_one.get_run(run_id)
        assert record is not None
        assert record.snapshot_revision == 1
        snapshot = await store_one.get_snapshot(run_id)
        assert snapshot is not None
        assert snapshot.usage == usage
    finally:
        await engine_one.dispose()
        await engine_two.dispose()
