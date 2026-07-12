#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_swarm_store_contract.py — runs the same
SwarmStore contract against both FileSwarmStore and SqlAlchemySwarmStore (spec
contract backend parity). The parametrized ``store_factory`` fixture is copied
verbatim from ``test_run_store_contract.py`` (file + sqlalchemy branches,
including the ``_run_in_new_loop`` helper that bootstraps the SQL engine off the
test loop); ``Base.metadata.create_all`` already covers ``SwarmRunRow`` /
``SwarmTaskRow`` since they subclass the same ``Base``.

Uses the ``def test_x(store_factory):`` + ``asyncio.run(_run())`` style (sync
test wrapper driving its own event loop) — no pytest-asyncio mode config needed."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from linktools.ai.errors import (
    InvalidSwarmTransitionError,
    SwarmConflictError,
    SwarmRunNotFoundError,
    SwarmTaskNotFoundError,
)
from linktools.ai.run.models import RunErrorInfo, RunResult
from linktools.ai.storage.file.swarm import FileSwarmStore
from linktools.ai.swarm.models import (
    SwarmRun,
    SwarmStatus,
    SwarmTask,
    SwarmTaskStatus,
    TaskInput,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Record builders. Defaults use datetime.now(timezone.utc), TokenUsage() and
# Decimal("0") (zero-state) per the spec; round-trip tests override with
# non-zero Decimals to verify precision.
# ---------------------------------------------------------------------------


def make_run(
    swarm_run_id: str = "swarm-1",
    status: SwarmStatus = SwarmStatus.PENDING,
    version: int = 1,
    round: int = 0,
    cost: Decimal = Decimal("0"),
    token_usage: "TokenUsage | None" = None,
    metadata: "dict | None" = None,
) -> SwarmRun:
    now = datetime.now(timezone.utc)
    return SwarmRun(
        id=swarm_run_id,
        run_id="run-1",
        round=round,
        status=status,
        version=version,
        token_usage=token_usage if token_usage is not None else TokenUsage(),
        cost=cost,
        created_at=now,
        updated_at=now,
        metadata={"k": "v"} if metadata is None else metadata,
    )


def make_task(
    task_id: str = "task-1",
    swarm_run_id: str = "swarm-1",
    parent_task_id: "str | None" = None,
    status: SwarmTaskStatus = SwarmTaskStatus.PENDING,
    dependencies: "tuple[str, ...]" = (),
    assigned_agent_id: "str | None" = None,
    attempts: int = 0,
    version: int = 1,
) -> SwarmTask:
    now = datetime.now(timezone.utc)
    return SwarmTask(
        id=task_id,
        swarm_run_id=swarm_run_id,
        parent_task_id=parent_task_id,
        assigned_agent_id=assigned_agent_id,
        description="do thing",
        status=status,
        dependencies=dependencies,
        input=TaskInput(prompt="hi", metadata={"t": 1}),
        result=None,
        error=None,
        attempts=attempts,
        version=version,
        claimed_at=None,
        lease_expires_at=None,
        created_at=now,
        updated_at=now,
    )


# ---------------------------------------------------------------------------
# Parametrized store factory. The SQL branch (incl. ``_run_in_new_loop``) is
# copied verbatim from test_run_store_contract.py.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FileSwarmStore(root=tmp_path / f"swarm-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.swarm import SqlAlchemySwarmStore

    counter = {"n": 0}
    engines = []

    def _run_in_new_loop(coro):
        # This factory is called synchronously from inside an already-running
        # pytest-asyncio event loop (the async test function), so we cannot use
        # asyncio.get_event_loop().run_until_complete() here -- that raises
        # "This event loop is already running". Run the setup coroutine to
        # completion on a separate thread with its own fresh event loop instead.
        import asyncio
        import threading

        outcome = {}

        def _runner():
            try:
                outcome["value"] = asyncio.run(coro)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread below
                outcome["error"] = exc

        thread = threading.Thread(target=_runner)
        thread.start()
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome.get("value")

    def sqlalchemy_factory():
        counter["n"] += 1
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{tmp_path}/swarm-db-{counter['n']}.db"
        )
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                # SwarmRunRow + SwarmTaskRow subclass Base, so a single
                # create_all covers every table both backends need.
                await conn.run_sync(Base.metadata.create_all)
            # The connection pool otherwise holds a connection bound to this
            # thread's event loop; dispose it so later operations (running on
            # pytest-asyncio's loop) open fresh connections instead of reusing
            # one tied to a loop that is about to be closed.
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemySwarmStore(session_factory=session_factory)

    def _dispose_engines():
        # The store itself opens fresh connections on pytest-asyncio's loop
        # during the test. Those connections (and aiosqlite's background
        # worker threads) must be disposed before that loop closes at test
        # teardown, otherwise the worker thread tries to call back into an
        # already-closed loop and pytest reports an unraisable exception.
        for engine in engines:
            _run_in_new_loop(engine.dispose())

    request.addfinalizer(_dispose_engines)

    return sqlalchemy_factory


# ---------------------------------------------------------------------------
# 1. create_run -> get_run round-trip (all fields incl. metadata, Decimal,
#    enum values, datetime precision).
# ---------------------------------------------------------------------------


def test_create_run_then_get_run_roundtrip(store_factory):
    store = store_factory()

    async def _run():
        run = make_run(
            cost=Decimal("1.25"),
            token_usage=TokenUsage(
                input_tokens=10, output_tokens=20, total_cost=Decimal("1.25")
            ),
        )
        created = await store.create_run(run)
        fetched = await store.get_run("swarm-1")
        assert fetched is not None
        # Frozen dataclass equality: every field (id, run_id, round, status,
        # version, token_usage incl. Decimal total_cost, cost, created_at,
        # updated_at, metadata) round-trips identically on both backends.
        assert fetched == created
        # Targeted checks for the load-bearing fields ( Decimal precision,
        # enum value, metadata mapping, datetime tz-awareness).
        assert fetched.status == SwarmStatus.PENDING
        assert fetched.token_usage.input_tokens == 10
        assert fetched.token_usage.output_tokens == 20
        assert fetched.token_usage.total_cost == Decimal("1.25")
        assert fetched.cost == Decimal("1.25")
        assert dict(fetched.metadata) == {"k": "v"}
        assert fetched.created_at == created.created_at
        assert fetched.created_at.tzinfo is not None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2-4. update_run: version advance, conflict, invalid transition.
# ---------------------------------------------------------------------------


def test_update_run_advances_version_and_applies_fields(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        updated = await store.update_run(
            "swarm-1",
            expected_version=1,
            status=SwarmStatus.RUNNING,
            round=2,
            token_usage=TokenUsage(
                input_tokens=100, output_tokens=200, total_cost=Decimal("2")
            ),
            cost=Decimal("9.99"),
            metadata={"new": "k"},
        )
        assert updated.version == 2
        assert updated.status == SwarmStatus.RUNNING
        assert updated.round == 2
        assert updated.token_usage.input_tokens == 100
        assert updated.token_usage.output_tokens == 200
        assert updated.cost == Decimal("9.99")
        assert dict(updated.metadata) == {"new": "k"}

    asyncio.run(_run())


def test_update_run_wrong_expected_version_raises_conflict(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        with pytest.raises(SwarmConflictError):
            await store.update_run(
                "swarm-1", expected_version=99, status=SwarmStatus.RUNNING
            )

    asyncio.run(_run())


def test_update_run_invalid_transition_raises(store_factory):
    store = store_factory()

    async def _run():
        # PENDING -> SUCCEEDED is not in ALLOWED_SWARM_TRANSITIONS (only
        # PENDING -> RUNNING is).
        await store.create_run(make_run())
        with pytest.raises(InvalidSwarmTransitionError):
            await store.update_run(
                "swarm-1", expected_version=1, status=SwarmStatus.SUCCEEDED
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. create_task -> list_tasks(swarm_run_id) returns it; status filter narrows.
# ---------------------------------------------------------------------------


def test_create_task_then_list_with_status_filter(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        await store.create_task(
            make_task(task_id="t-pending", status=SwarmTaskStatus.PENDING)
        )
        await store.create_task(
            make_task(
                task_id="t-claimed",
                status=SwarmTaskStatus.CLAIMED,
                assigned_agent_id="agent-7",
            ),
        )
        all_tasks = await store.list_tasks("swarm-1")
        assert {t.id for t in all_tasks} == {"t-pending", "t-claimed"}
        claimed = await store.list_tasks("swarm-1", status=SwarmTaskStatus.CLAIMED)
        assert {t.id for t in claimed} == {"t-claimed"}
        pending = await store.list_tasks("swarm-1", status=SwarmTaskStatus.PENDING)
        assert {t.id for t in pending} == {"t-pending"}

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6-8. claim_task: PENDING -> CLAIMED, dependency gate, empty -> None.
# ---------------------------------------------------------------------------


def test_claim_task_marks_claimed_and_assigns_agent(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        await store.create_task(make_task(task_id="t-1"))
        claimed = await store.claim_task("swarm-1", "agent-9")
        assert claimed is not None
        assert claimed.id == "t-1"
        assert claimed.status == SwarmTaskStatus.CLAIMED
        assert claimed.assigned_agent_id == "agent-9"
        assert claimed.claimed_at is not None
        assert claimed.version == 2

    asyncio.run(_run())


def test_claim_task_respects_dependencies(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        # t-dep is PENDING (claimable, no deps); t-blocked depends on t-dep.
        await store.create_task(
            make_task(task_id="t-dep", status=SwarmTaskStatus.PENDING)
        )
        await store.create_task(make_task(task_id="t-blocked", dependencies=("t-dep",)))
        # First claim picks t-dep (created first, deps trivially satisfied),
        # NOT t-blocked whose only dependency is still PENDING.
        first = await store.claim_task("swarm-1", "agent-1")
        assert first is not None
        assert first.id == "t-dep"
        # Now nothing is claimable: t-dep is CLAIMED (no longer pending), and
        # t-blocked's dependency is CLAIMED, not SUCCEEDED.
        second = await store.claim_task("swarm-1", "agent-2")
        assert second is None

    asyncio.run(_run())


def test_claim_task_returns_none_when_nothing_claimable(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        # No tasks at all -> nothing to claim.
        assert await store.claim_task("swarm-1", "agent-1") is None
        # A terminal/SUCCEEDED task is not claimable either.
        await store.create_task(
            make_task(task_id="t-done", status=SwarmTaskStatus.SUCCEEDED)
        )
        assert await store.claim_task("swarm-1", "agent-1") is None

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 8b. set_active_run: stores the child RunRecord id on a CLAIMED task,
#     bumps version, and rejects a stale expected_version. Phase-5A.
# ---------------------------------------------------------------------------


def test_set_active_run_stores_run_id_and_advances_version(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        await store.create_task(make_task(task_id="t-1"))
        claimed = await store.claim_task("swarm-1", "agent-1")
        assert claimed is not None
        # claim advanced v1 -> v2; set_active_run must advance v2 -> v3.
        updated = await store.set_active_run(
            "t-1", "child-run-7", expected_version=claimed.version
        )
        assert updated.active_run_id == "child-run-7"
        assert updated.version == claimed.version + 1

    asyncio.run(_run())


def test_set_active_run_wrong_expected_version_raises_conflict(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        await store.create_task(make_task(task_id="t-1"))
        await store.claim_task("swarm-1", "agent-1")
        with pytest.raises(SwarmConflictError):
            await store.set_active_run("t-1", "child-1", expected_version=99)

    asyncio.run(_run())


def test_set_active_run_missing_task_raises_not_found(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        with pytest.raises(SwarmTaskNotFoundError):
            await store.set_active_run("nope", "child-1", expected_version=1)

    asyncio.run(_run())


def test_set_active_run_requires_claimed_status(store_factory):
    """set_active_run must reject a task that is no longer CLAIMED even when
    the caller's expected_version happens to be correct -- e.g. the task was
    completed by a racing writer between this caller's last read and this
    call. Mirrors complete_task/fail_task's own status fencing."""
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        await store.create_task(make_task(task_id="t-1"))
        claimed = await store.claim_task("swarm-1", "agent-1")
        assert claimed is not None
        completed = await store.complete_task(
            "t-1",
            RunResult(output="done"),
            expected_version=claimed.version,
        )
        # completed.version is the CORRECT current version -- but the task
        # is now SUCCEEDED, not CLAIMED, so set_active_run must still reject.
        with pytest.raises(SwarmConflictError):
            await store.set_active_run(
                "t-1",
                "child-1",
                expected_version=completed.version,
            )

    asyncio.run(_run())


def test_set_active_run_roundtrips_through_persistence(store_factory):
    """active_run_id round-trips through the store's serialization layer (JSON
    for FileSwarmStore, SQL column for SqlAlchemySwarmStore). Verified by
    reading back via list_tasks on the same store instance after write."""
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        await store.create_task(make_task(task_id="t-1"))
        claimed = await store.claim_task("swarm-1", "agent-1")
        assert claimed is not None
        await store.set_active_run(
            "t-1", "child-run-xyz", expected_version=claimed.version
        )
        # read back through the same serialization path the next process would.
        tasks = await store.list_tasks("swarm-1")
        assert len(tasks) == 1
        assert tasks[0].active_run_id == "child-run-xyz"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 9. complete_task -> SUCCEEDED + result; fail_task -> FAILED + error.
# ---------------------------------------------------------------------------


def test_complete_and_fail_task_store_result_and_error(store_factory):
    store = store_factory()

    async def _run():
        await store.create_run(make_run())
        await store.create_task(make_task(task_id="t-ok"))
        claimed_ok = await store.claim_task("swarm-1", "agent-1")
        result = RunResult(
            output={"done": True},
            token_usage={"input_tokens": 1},
            metadata={"m": "n"},
        )
        # Lifecycle create(v1) -> claim(v2) -> complete(v3): each step bumps
        # version, so a completed-via-claim task lands at version 3.
        completed = await store.complete_task(
            "t-ok", result, expected_version=claimed_ok.version
        )
        assert completed.status == SwarmTaskStatus.SUCCEEDED
        assert completed.result.output == {"done": True}
        assert dict(completed.result.metadata) == {"m": "n"}
        assert completed.version == 3

        await store.create_task(make_task(task_id="t-bad"))
        claimed_bad = await store.claim_task("swarm-1", "agent-2")
        err = RunErrorInfo(error_type="ValueError", message="boom", detail={"x": 1})
        failed = await store.fail_task(
            "t-bad", err, expected_version=claimed_bad.version
        )
        assert failed.status == SwarmTaskStatus.FAILED
        assert failed.error.error_type == "ValueError"
        assert failed.error.message == "boom"
        assert failed.attempts == 1
        assert failed.version == 3

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 10. get_run/update_run/complete_task on missing ids.
# ---------------------------------------------------------------------------


def test_missing_run_and_task_raise_not_found(store_factory):
    store = store_factory()

    async def _run():
        # get_run on missing id -> None (NOT an error).
        assert await store.get_run("nope") is None
        # update_run / complete_task / fail_task on missing ids -> typed errors.
        with pytest.raises(SwarmRunNotFoundError):
            await store.update_run(
                "nope", expected_version=1, status=SwarmStatus.RUNNING
            )
        with pytest.raises(SwarmTaskNotFoundError):
            await store.complete_task(
                "nope", RunResult(output=None), expected_version=1
            )
        with pytest.raises(SwarmTaskNotFoundError):
            await store.fail_task(
                "nope", RunErrorInfo(error_type="X", message="y"), expected_version=1
            )

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 11. File-only: path-traversal in swarm_run_id / task_id -> ValueError.
# (SQL ids are opaque primary-key strings, not path segments, so this guard is
# FileSwarmStore-specific — mirrors the file-only path-traversal test in
# test_run_store_contract.py.)
# ---------------------------------------------------------------------------


def test_path_traversal_in_swarm_ids_is_rejected(tmp_path):
    store = FileSwarmStore(root=tmp_path)

    async def _run():
        with pytest.raises(ValueError):
            await store.get_run("../evil")
        with pytest.raises(ValueError):
            await store.create_run(make_run(swarm_run_id="../evil"))
        with pytest.raises(ValueError):
            await store.create_task(make_task(task_id="../evil"))
        with pytest.raises(ValueError):
            await store.complete_task(
                "../evil", RunResult(output=None), expected_version=1
            )

    asyncio.run(_run())
