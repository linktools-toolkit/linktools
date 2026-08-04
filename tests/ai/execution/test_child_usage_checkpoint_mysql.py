#!/usr/bin/env python3
"""MySQL reservation for the SQL execution usage-checkpoint contract."""

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from linktools.ai.execution.commands import CheckpointExecutionUsage, ClaimExecution, StartExecution
from linktools.ai.execution.domain import (
    RunDefinition,
    RunKind,
    RunUsage,
    RunnableType,
    compute_run_definition_hash,
)
from linktools.ai.execution.persistence.sqlalchemy import SqlAlchemyExecutionBackend


def _definition() -> RunDefinition:
    schema = "agent-spec.v1"
    spec = {"id": "agent"}
    return RunDefinition(
        "agent",
        RunnableType.AGENT,
        schema,
        spec,
        compute_run_definition_hash(schema=schema, spec=spec),
    )


@pytest.mark.asyncio
async def test_mysql_usage_checkpoint_contract() -> None:
    dsn = os.environ.get("LINKTOOLS_AI_TEST_MYSQL_DSN")
    if not dsn:
        pytest.skip("LINKTOOLS_AI_TEST_MYSQL_DSN not set; MySQL is downstream-owned")
    pytest.importorskip("asyncmy")
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(dsn)
    store = SqlAlchemyExecutionBackend(
        async_sessionmaker(engine, expire_on_commit=False)
    )
    run_id = "mysql-checkpoint-contract"
    try:
        await store.initialize_storage(engine)
        await store.create_session(session_id="mysql-session", user_id="u", tenant_id="t")
        await store.start_run(
            StartExecution(
                run_id,
                "mysql-session",
                RunKind.USER_TURN,
                _definition(),
                "prompt",
            )
        )
        claimed = await store.claim_run(
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
        snapshot = await store.checkpoint_run_usage(
            CheckpointExecutionUsage(
                run_id,
                "worker",
                claimed.lease.fence,
                0,
                usage,
                1,
            )
        )
        assert snapshot.revision == 1
        assert (await store.get_snapshot(run_id)).usage == usage
    finally:
        await engine.dispose()
