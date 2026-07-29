import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.execution.snapshots import AgentSnapshotData, RunSnapshot
from linktools.ai.execution.trace_models import NewRunTraceStep
from linktools.ai.execution.persistence.sqlalchemy import SnapshotRow, SqlAlchemyExecutionBackend
from linktools.ai.execution.store import ExecutionStore
from linktools.ai.errors import StorageConflictError
from linktools.ai.execution.commands import (
    AbortExecution,
    AcknowledgeCancellation,
    ClaimExecution,
    CompleteExecution,
    DecideApproval,
    PauseExecution,
    RequestCancellation,
    ResumeExecution,
    StartExecution,
)
from linktools.ai.execution.domain import ApprovalDecision, RunApproval, RunDefinition, RunError, RunKind, RunRecord, RunStatus, RunUsage, RunnableType


@pytest.mark.asyncio
async def test_sqlalchemy_execution_pages_in_database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'execution.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyExecutionBackend(factory)
    await store.initialize_storage(engine)
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    definition = RunDefinition("a", RunnableType.AGENT, "agent-spec.v1", {"id": "a"}, "a")
    run = await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "p"))
    assert run.input == "p"
    claimed = await store.claim_run(ClaimExecution("r", "w", datetime.now(timezone.utc), __import__("datetime").timedelta(minutes=5)))
    snapshot = RunSnapshot("run-snapshot.v1", "r", 1, ({"role": "user", "content": "p"},), "done", RunStatus.COMPLETED, RunUsage(), 0, datetime.now(timezone.utc))
    async with factory() as session:
        async with session.begin():
            session.add(SnapshotRow(execution_id="r", revision=1, resume_messages=list(snapshot.resume_messages), final_output="done", status="completed", usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, trace_end_sequence=0, created_at=snapshot.created_at))
    assert (await store.list_session_turns("s", limit=1)).items[0].run_id == "r"
    assert (await store.get_snapshot("r")).final_output == "done"
    await engine.dispose()


async def _claimed_store(tmp_path, name: str = "lifecycle"):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'{name}.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyExecutionBackend(factory)
    await store.initialize_storage(engine)
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    definition = RunDefinition("a", RunnableType.AGENT, "agent-spec.v1", {"id": "a"}, "a")
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "p"))
    now = datetime.now(timezone.utc)
    claimed = await store.claim_run(ClaimExecution("r", "worker", now, timedelta(minutes=5)))
    return engine, store, claimed


@pytest.mark.asyncio
async def test_sqlalchemy_cancel_acknowledgement_keeps_fence_until_terminal_commit(tmp_path):
    engine, store, claimed = await _claimed_store(tmp_path, "cancel")
    now = datetime.now(timezone.utc)
    cancelling = await store.request_cancel(
        RequestCancellation("r", "worker", claimed.lease.fence, now)
    )
    assert cancelling.lease.owner == "worker"
    snapshot = AgentSnapshotData((), None, RunUsage(), 0)
    cancelled = await store.acknowledge_cancel(
        AcknowledgeCancellation("r", "worker", claimed.lease.fence, snapshot)
    )
    assert cancelled.status is RunStatus.CANCELLED
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlalchemy_abort_run_persists_error_without_a_snapshot(tmp_path):
    engine, store, claimed = await _claimed_store(tmp_path, "abort")
    aborted = await store.abort_run(AbortExecution("r", "worker", claimed.lease.fence, RunError("RuntimeError", "boom"), 3))
    assert aborted.status is RunStatus.FAILED
    assert aborted.error == RunError("RuntimeError", "boom")
    assert aborted.lease.owner is None
    assert aborted.trace_sequence == 3
    assert await store.get_snapshot("r") is None
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlalchemy_approval_decision_is_idempotent_and_immutable(tmp_path):
    engine, store, claimed = await _claimed_store(tmp_path, "approval")
    now = datetime.now(timezone.utc)
    snapshot = AgentSnapshotData((), None, RunUsage(), 0)
    approval = RunApproval("approval", "call", "tool", {})
    await store.pause_run(
        PauseExecution("r", "worker", claimed.lease.fence, snapshot, approval)
    )
    first = await store.decide_approval(
        DecideApproval("r", "approval", "allow", "reviewer")
    )
    replay = await store.decide_approval(
        DecideApproval("r", "approval", "allow", "reviewer")
    )
    assert replay.event_sequence == first.event_sequence
    with pytest.raises(StorageConflictError):
        await store.decide_approval(
            DecideApproval("r", "approval", "deny", "reviewer")
        )
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlalchemy_approval_deny_cancels_run(tmp_path):
    """Spec 2.13: a DENY decision is terminal -- execution -> CANCELLED, lease
    released, no tools run. The decision is then immutable."""
    engine, store, claimed = await _claimed_store(tmp_path, "deny")
    now = datetime.now(timezone.utc)
    snapshot = AgentSnapshotData((), None, RunUsage(), 0)
    await store.pause_run(
        PauseExecution("r", "worker", claimed.lease.fence, snapshot, RunApproval("approval", "call", "tool", {}))
    )
    denied = await store.decide_approval(
        DecideApproval("r", "approval", ApprovalDecision.DENY, "reviewer")
    )
    assert denied.status is RunStatus.CANCELLED
    assert denied.lease.owner is None
    # replaying the same DENY is idempotent (already terminal).
    replay = await store.decide_approval(
        DecideApproval("r", "approval", ApprovalDecision.DENY, "reviewer")
    )
    assert replay.status is RunStatus.CANCELLED
    assert replay.event_sequence == denied.event_sequence
    # flipping a denied (CANCELLED) run to ALLOW is rejected.
    with pytest.raises(StorageConflictError):
        await store.decide_approval(
            DecideApproval("r", "approval", ApprovalDecision.ALLOW, "reviewer")
        )
    await engine.dispose()


@pytest.mark.asyncio
async def test_store_allocates_monotonic_snapshot_revisions(tmp_path):
    # The STORE allocates the snapshot revision (expected + 1), not the engine.
    # Two pause cycles on one run must yield revisions 1 then 2, even though the
    # caller-built snapshot always carries revision=0.
    engine, store, claimed = await _claimed_store(tmp_path, "rev")
    now = datetime.now(timezone.utc)
    snapshot = AgentSnapshotData((), None, RunUsage(), 0)
    await store.pause_run(PauseExecution("r", "worker", claimed.lease.fence, snapshot, RunApproval("a", "c", "t", {})))
    assert (await store.get_snapshot("r")).revision == 1
    await store.decide_approval(DecideApproval("r", "a", ApprovalDecision.ALLOW, "reviewer"))
    await store.resume_run(ResumeExecution("r"))
    reclaimed = await store.claim_run(ClaimExecution("r", "worker", now, timedelta(minutes=5)))
    await store.pause_run(PauseExecution("r", reclaimed.lease.owner, reclaimed.lease.fence, snapshot, RunApproval("b", "c", "t", {})))
    assert (await store.get_snapshot("r")).revision == 2
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlalchemy_run_allows_only_one_concurrent_claim(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'run-claim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyExecutionBackend(factory)
    await store.initialize_storage(engine)
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    definition = RunDefinition("a", RunnableType.AGENT, "agent-spec.v1", {"id": "a"}, "a")
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "p"))
    now = datetime.now(timezone.utc)
    results = await asyncio.gather(
        store.claim_run(ClaimExecution("r", "worker-a", now, timedelta(minutes=5))),
        store.claim_run(ClaimExecution("r", "worker-b", now, timedelta(minutes=5))),
        return_exceptions=True,
    )
    assert sum(isinstance(result, RunRecord) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlalchemy_run_commits_terminal_snapshot_exactly_once(tmp_path):
    engine, store, claimed = await _claimed_store(tmp_path, "run-result")
    now = datetime.now(timezone.utc)
    first = RunSnapshot(
        "run-snapshot.v1",
        "r",
        1,
        (),
        "first",
        RunStatus.COMPLETED,
        RunUsage(),
        0,
        now,
    )
    second = RunSnapshot(
        "run-snapshot.v1",
        "r",
        1,
        (),
        "second",
        RunStatus.COMPLETED,
        RunUsage(),
        0,
        now,
    )
    results = await asyncio.gather(
        store.complete_run(CompleteExecution("r", "worker", claimed.lease.fence, first)),
        store.complete_run(CompleteExecution("r", "worker", claimed.lease.fence, second)),
        return_exceptions=True,
    )
    assert sum(isinstance(result, RunRecord) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sqlalchemy_trace_append_uses_sequence_cas(tmp_path):
    engine, store, _claimed = await _claimed_store(tmp_path, "trace")
    now = datetime.now(timezone.utc)
    results = await asyncio.gather(
        store.append_trace_steps(
            "r",
            expected_sequence=0,
            steps=(NewRunTraceStep("model_interaction", {"value": "first"}, now),),
        ),
        store.append_trace_steps(
            "r",
            expected_sequence=0,
            steps=(NewRunTraceStep("model_interaction", {"value": "second"}, now),),
        ),
        return_exceptions=True,
    )
    assert sum(result == 1 for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    assert len(await store.list_trace_steps("r")) == 1
    await engine.dispose()
