import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import pytest

from linktools.ai.errors import StorageConflictError
from linktools.ai.tasks.persistence.sqlalchemy import SqlAlchemyTaskBackend
from linktools.ai.agent.tool.persistence.sqlalchemy import OperationRow, SqlAlchemyToolStateBackend
from linktools.ai.agent.tool.models import ToolOperation, ToolOperationStatus
from linktools.ai.agent.tool.store import ToolStateStore

from tests.ai.tasks.contracts import (
    assert_task_store_contract,
    assert_usage_round_trips_through_complete,
)


@pytest.mark.asyncio
async def test_sql_task_store_meets_contract(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyTaskBackend(factory)
    await store.initialize_storage(engine)
    await assert_task_store_contract(store)
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_task_store_round_trips_usage(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks-usage.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyTaskBackend(factory)
    await store.initialize_storage(engine)
    await assert_usage_round_trips_through_complete(store)
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_task_store_concurrent_complete_has_one_winner(tmp_path):
    # complete vs cancel_claimed on the same CLAIMED node: exactly one terminal wins.
    from linktools.ai.tasks.models import TaskExecution, TaskPlan, TaskStatus, TaskUsage

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks-race.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyTaskBackend(factory)
    await store.initialize_storage(engine)
    await store.create_plan(
        TaskPlan("race", ()),
        (TaskExecution("e", "race", "n", TaskStatus.READY),),
    )
    claimed = await store.claim_ready("e", owner="w", duration=timedelta(seconds=30))
    await store.bind_child_run("e", owner="w", fence=claimed.fence, child_run_id="child-race")
    complete_coro = store.complete(
        "e", owner="w", fence=claimed.fence, result={"ok": True}, snapshot_revision=1, usage=TaskUsage(1, 1)
    )
    from linktools.ai.execution.domain import RunError

    cancel_coro = store.cancel_claimed(
        "e", owner="w", fence=claimed.fence, reason="abort", snapshot_revision=1, usage=TaskUsage(1, 1)
    )
    results = await asyncio.gather(complete_coro, cancel_coro, return_exceptions=True)
    final = await store.get_execution("e")
    assert final.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED}
    winners = [r for r in results if not isinstance(r, Exception)]
    conflicts = [r for r in results if isinstance(r, StorageConflictError)]
    assert len(winners) == 1 and len(conflicts) == 1, results
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_task_store_requires_reconcile_for_expired_claim(tmp_path):
    from linktools.ai.tasks.models import TaskExecution, TaskPlan, TaskStatus, TaskUsage

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tasks-reclaim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyTaskBackend(factory)
    await store.initialize_storage(engine)
    await store.create_plan(
        TaskPlan("reclaim", ()),
        (TaskExecution("e", "reclaim", "n", TaskStatus.READY),),
    )
    stale = await store.claim_ready("e", owner="w-a", duration=timedelta(seconds=-1))
    current = await store.take_over_expired_claim_for_reconcile(
        "e",
        owner="w-b",
        now=stale.lease.expires_at,
        duration=timedelta(seconds=30),
    )
    assert current.fence == stale.fence + 1
    with pytest.raises(StorageConflictError):
        await store.complete(
            "e", owner="w-a", fence=stale.fence, result=None, snapshot_revision=1, usage=TaskUsage()
        )
    await engine.dispose()


# --- Tool state store (unchanged) ---


@pytest.mark.asyncio
async def test_sql_tool_store_replays_completed_operation(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tools.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyToolStateBackend(factory)
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", "binding", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    claimed = await store.claim("o", owner="worker", duration=timedelta(minutes=1))
    assert claimed.lease.expires_at is not None
    await store.complete("o", owner="worker", fence=claimed.fence, result={"ok": True})
    assert (await store.prepare(operation)).result == {"ok": True}
    with pytest.raises(StorageConflictError):
        await store.complete("o", owner="worker", fence=claimed.fence, result={"ok": False})
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_concurrent_prepare_is_idempotent(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-prepare.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyToolStateBackend(factory)
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", "binding", ToolOperationStatus.PREPARED)
    results = await asyncio.gather(
        store.prepare(operation),
        store.prepare(operation),
    )
    assert results[0].id == results[1].id == "o"
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_allows_only_one_concurrent_claim(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-claim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyToolStateBackend(factory)
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", "binding", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    results = await asyncio.gather(
        store.claim("o", owner="worker-a"),
        store.claim("o", owner="worker-b"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ToolOperation) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_commits_terminal_result_exactly_once(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-result.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyToolStateBackend(factory)
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", "binding", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    claimed = await store.claim("o", owner="worker")
    results = await asyncio.gather(
        store.complete("o", owner="worker", fence=claimed.fence, result="first"),
        store.complete("o", owner="worker", fence=claimed.fence, result="second"),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ToolOperation) for result in results) == 1
    assert sum(isinstance(result, StorageConflictError) for result in results) == 1
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_reclaims_expired_lease_and_rejects_stale_fence(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-reclaim.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyToolStateBackend(factory)
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", "binding", ToolOperationStatus.PREPARED, replay_safe=True)
    await store.prepare(operation)
    stale = await store.claim("o", owner="worker-a", duration=timedelta(seconds=-1))
    current = await store.claim("o", owner="worker-b")
    assert current.fence == stale.fence + 1
    with pytest.raises(StorageConflictError):
        await store.complete("o", owner="worker-a", fence=stale.fence, result=None)
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_tool_store_stale_non_replay_safe_claim_marks_indeterminate(tmp_path):
    # claim()'s stale-lease branch surfaces an expired, non-replay-safe CLAIMED
    # lease as INDETERMINATE (the replay-safe reclaim case is covered by the test
    # above; this one exercises the other branch). Covers the post-lock-removal
    # path end-to-end so a regression that dropped the mark would surface here.
    #
    # Note on the CAS the branch uses: without a pessimistic lock, the
    # INDETERMINATE mark is a conditional UPDATE keyed on the exact row state
    # the branch read (status/owner/fence/lease_expires_at), so a concurrently
    # committed change to those columns makes the mark match zero rows and the
    # fresh state is returned instead. That property holds under the
    # READ-COMMITTED isolation of MySQL/PostgreSQL (production); SQLite
    # serializes a transaction's snapshot, so a same-transaction race window
    # cannot be exercised here and the CAS is verified structurally in code
    # review plus a standalone cross-session repro rather than by this test.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'tool-indeterminate.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyToolStateBackend(factory)
    await store.initialize_storage(engine)
    operation = ToolOperation("o", None, "r", "c", "key", "tool", "hash", "binding", ToolOperationStatus.PREPARED)
    await store.prepare(operation)
    # Establish a CLAIMED lease, then force its expiry timestamp into the past
    # directly (a real expiry would require sleeping past the lease window).
    await store.claim("o", owner="worker-a", duration=timedelta(minutes=1))
    expired_at = datetime.now(timezone.utc) - timedelta(minutes=5)
    async with factory() as expire_session:
        async with expire_session.begin():
            await expire_session.execute(
                update(OperationRow)
                .where(OperationRow.operation_id == "o")
                .values(lease_expires_at=expired_at)
            )
    # A fresh claim by a different owner reads status=CLAIMED with an expired
    # lease, and (non-replay-safe) marks it INDETERMINATE instead of reclaiming.
    stale = await store.claim("o", owner="worker-b")
    assert stale.status == ToolOperationStatus.INDETERMINATE
    # The stale claim is terminal-ish: a fresh claim observes INDETERMINATE and
    # short-circuits (never re-claims), surfacing the staleness to the caller.
    again = await store.claim("o", owner="worker-c")
    assert again.status == ToolOperationStatus.INDETERMINATE
    await engine.dispose()
