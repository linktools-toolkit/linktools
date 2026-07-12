#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_run_store_contract.py — runs the same RunStore
contract against both FileRunStore and SqlAlchemyRunStore."""

from datetime import datetime, timezone

import pytest

from linktools.ai.errors import (
    InvalidRunTransitionError,
    RunConflictError,
    RunNotFoundError,
)
from linktools.ai.run.models import (
    RunErrorInfo,
    RunInput,
    RunnableType,
    RunRecord,
    RunResult,
    RunStatus,
)
from linktools.ai.storage.file.run import FileRunStore


def _record(
    run_id="run-1", parent_run_id=None, status=RunStatus.PENDING, version=1
) -> RunRecord:
    now = datetime.now(timezone.utc)
    return RunRecord(
        id=run_id,
        root_run_id=run_id if parent_run_id is None else "run-1",
        parent_run_id=parent_run_id,
        session_id="session-1",
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        status=status,
        input=RunInput(prompt="hi"),
        result=None,
        error=None,
        version=version,
        created_at=now,
        started_at=None,
        finished_at=None,
    )


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FileRunStore(root=tmp_path / f"runs-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.run import SqlAlchemyRunStore

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
            f"sqlite+aiosqlite:///{tmp_path}/runs-db-{counter['n']}.db"
        )
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            # The connection pool otherwise holds a connection bound to this
            # thread's event loop; dispose it so later operations (running on
            # pytest-asyncio's loop) open fresh connections instead of reusing
            # one tied to a loop that is about to be closed.
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyRunStore(session_factory=session_factory)

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


@pytest.mark.asyncio
async def test_create_then_get_roundtrip(store_factory):
    store = store_factory()
    created = await store.create(_record())
    fetched = await store.get("run-1")
    assert fetched is not None
    assert fetched.id == "run-1"
    assert fetched.status == RunStatus.PENDING
    assert created == fetched


@pytest.mark.asyncio
async def test_get_missing_returns_none(store_factory):
    store = store_factory()
    assert await store.get("nope") is None


@pytest.mark.asyncio
async def test_transition_pending_to_running_succeeds(store_factory):
    store = store_factory()
    await store.create(_record())
    updated = await store.transition("run-1", RunStatus.RUNNING, expected_version=1)
    assert updated.status == RunStatus.RUNNING
    assert updated.version == 2


@pytest.mark.asyncio
async def test_transition_invalid_target_raises(store_factory):
    store = store_factory()
    await store.create(_record())
    with pytest.raises(InvalidRunTransitionError):
        await store.transition("run-1", RunStatus.SUCCEEDED, expected_version=1)


@pytest.mark.asyncio
async def test_transition_wrong_expected_version_raises_conflict(store_factory):
    store = store_factory()
    await store.create(_record())
    with pytest.raises(RunConflictError):
        await store.transition("run-1", RunStatus.RUNNING, expected_version=99)


@pytest.mark.asyncio
async def test_transition_missing_run_raises_not_found(store_factory):
    store = store_factory()
    with pytest.raises(RunNotFoundError):
        await store.transition("nope", RunStatus.RUNNING, expected_version=1)


@pytest.mark.asyncio
async def test_transition_to_succeeded_stores_result(store_factory):
    store = store_factory()
    await store.create(_record())
    await store.transition("run-1", RunStatus.RUNNING, expected_version=1)
    done = await store.transition(
        "run-1",
        RunStatus.SUCCEEDED,
        expected_version=2,
        result=RunResult(output={"ok": True}),
    )
    assert done.status == RunStatus.SUCCEEDED
    assert done.result.output == {"ok": True}


@pytest.mark.asyncio
async def test_transition_to_failed_stores_error(store_factory):
    store = store_factory()
    await store.create(_record())
    await store.transition("run-1", RunStatus.RUNNING, expected_version=1)
    failed = await store.transition(
        "run-1",
        RunStatus.FAILED,
        expected_version=2,
        error=RunErrorInfo(error_type="X", message="boom"),
    )
    assert failed.status == RunStatus.FAILED
    assert failed.error.message == "boom"


@pytest.mark.asyncio
async def test_list_children_returns_only_direct_children(store_factory):
    store = store_factory()
    await store.create(_record(run_id="parent", status=RunStatus.PENDING))
    await store.create(_record(run_id="child-1", parent_run_id="parent"))
    await store.create(_record(run_id="child-2", parent_run_id="parent"))
    await store.create(_record(run_id="unrelated", parent_run_id=None))
    children = await store.list_children("parent")
    assert {c.id for c in children} == {"child-1", "child-2"}


@pytest.mark.asyncio
async def test_concurrent_transitions_with_same_expected_version_only_one_succeeds(
    store_factory,
):
    """P0-4/P1-6: N coroutines race to transition the SAME run from PENDING
    to RUNNING using the SAME expected_version. Exactly one may succeed;
    every other one must observe a conflict (RunConflictError) rather than
    silently succeeding too (a lost update) or corrupting the version
    counter. Exercises the DB-level CAS (SqlAlchemy) and the in-process lock
    (File) that replaced the read-then-compare-then-write pattern."""
    import asyncio

    store = store_factory()
    await store.create(_record())

    successes = []
    conflicts = []

    async def _attempt():
        try:
            updated = await store.transition(
                "run-1", RunStatus.RUNNING, expected_version=1
            )
            successes.append(updated)
        except RunConflictError:
            conflicts.append(1)

    await asyncio.gather(*(_attempt() for _ in range(20)))

    assert len(successes) == 1, f"expected exactly one winner, got {len(successes)}"
    assert len(conflicts) == 19
    assert successes[0].version == 2

    final = await store.get("run-1")
    assert final.version == 2, "version must have been bumped exactly once"


@pytest.mark.asyncio
async def test_path_traversal_in_run_id_is_rejected(tmp_path):
    store = FileRunStore(root=tmp_path)
    with pytest.raises(ValueError):
        await store.get("../../etc/passwd")
