#!/usr/bin/env python3
"""Usage checkpoints advance the lifecycle revision and fence terminal writes."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from linktools.ai.errors import StorageConflictError
from linktools.ai.execution.commands import (
    CheckpointExecutionUsage,
    ClaimExecution,
    CompleteExecution,
    StartExecution,
)
from linktools.ai.execution.domain import (
    RunDefinition,
    RunKind,
    RunStatus,
    RunUsage,
    RunnableType,
    compute_run_definition_hash,
)
from linktools.ai.execution.persistence.local import LocalExecutionBackend
from linktools.ai.execution.persistence.sqlalchemy import SqlAlchemyExecutionBackend
from linktools.ai.execution.service import PersistedRunUsageSink
from linktools.ai.execution.snapshots import (
    AgentSnapshotData,
    ModelRequestUsageObservation,
    RequestUsage,
    RunUsageCapture,
)


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


def _usage(input_tokens: int, output_tokens: int, cost: str) -> RunUsage:
    return RunUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        total_cost=Decimal(cost),
    )


async def _claimed(store, *, initialize: bool = True) -> object:
    if initialize:
        await store.initialize_storage()
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    await store.start_run(
        StartExecution("run", "s", RunKind.USER_TURN, _definition(), "prompt")
    )
    return await store.claim_run(
        ClaimExecution(
            "run",
            "worker",
            datetime.now(timezone.utc),
            timedelta(minutes=5),
        )
    )


@pytest.mark.asyncio
async def test_local_checkpoint_usage_is_monotonic_and_terminal_is_exact(tmp_path):
    store = LocalExecutionBackend(tmp_path / "execution")
    claimed = await _claimed(store)
    first = _usage(2, 1, "0.2")
    updated = await store.checkpoint_run_usage(
        CheckpointExecutionUsage("run", "worker", claimed.lease.fence, 0, first, 3)
    )
    assert updated.revision == 1
    snapshot = await store.get_snapshot("run")
    assert snapshot is not None
    assert snapshot.status is RunStatus.RUNNING
    assert snapshot.final_output is None
    assert snapshot.usage == first

    second = _usage(4, 2, "0.5")
    updated = await store.checkpoint_run_usage(
        CheckpointExecutionUsage("run", "worker", claimed.lease.fence, 1, second, 4)
    )
    assert updated.revision == 2
    idempotent = await store.checkpoint_run_usage(
        CheckpointExecutionUsage("run", "worker", claimed.lease.fence, 0, first, 5)
    )
    assert idempotent.revision == 2
    with pytest.raises(StorageConflictError):
        await store.checkpoint_run_usage(
            CheckpointExecutionUsage(
                "run",
                "worker",
                claimed.lease.fence,
                2,
                _usage(1, 5, "0.6"),
                5,
            )
        )

    terminal = AgentSnapshotData((), {"answer": "ok"}, second, 6)
    completed = await store.complete_run(
        CompleteExecution("run", "worker", claimed.lease.fence, terminal, 2)
    )
    assert completed.snapshot_revision == 3
    with pytest.raises(StorageConflictError):
        await store.complete_run(
            CompleteExecution("run", "worker", claimed.lease.fence, terminal, 2)
        )


@pytest.mark.asyncio
async def test_duplicate_sink_observation_refreshes_persisted_revision(tmp_path):
    store = LocalExecutionBackend(tmp_path / "execution")
    claimed = await _claimed(store)
    sink = PersistedRunUsageSink(
        capture=RunUsageCapture(),
        store=store,
        run_id="run",
        owner="worker",
        fence=claimed.lease.fence,
        snapshot_revision=0,
        trace_sequence=lambda: 1,
    )
    observation = ModelRequestUsageObservation(
        request_key="request-1",
        usage=RequestUsage(input_tokens=2, output_tokens=1),
        provider_name=None,
        response_model_name=None,
    )
    await sink.observe_request(observation, pricing=None)
    await store.checkpoint_run_usage(
        CheckpointExecutionUsage(
            "run",
            "worker",
            claimed.lease.fence,
            1,
            RunUsage(input_tokens=3, output_tokens=2, total_tokens=5),
            2,
        )
    )
    await sink.observe_request(observation, pricing=None)
    assert sink.last_snapshot_revision == 2


@pytest.mark.asyncio
async def test_sqlite_checkpoint_matches_local_contract(tmp_path):
    pytest.importorskip("sqlalchemy")
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'execution.db'}")
    store = SqlAlchemyExecutionBackend(
        async_sessionmaker(engine, expire_on_commit=False)
    )
    try:
        await store.initialize_storage(engine)
        claimed = await _claimed(store, initialize=False)
        usage = _usage(1, 2, "0.4")
        updated = await store.checkpoint_run_usage(
            CheckpointExecutionUsage("run", "worker", claimed.lease.fence, 0, usage, 1)
        )
        assert updated.revision == 1
        terminal = AgentSnapshotData((), "done", usage, 2)
        completed = await store.complete_run(
            CompleteExecution("run", "worker", claimed.lease.fence, terminal, 1)
        )
        assert completed.status is RunStatus.COMPLETED
        assert completed.snapshot_revision == 2
    finally:
        await engine.dispose()
