#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/contract/test_approval_store_contract.py — runs the same
ApprovalStore contract against both FileApprovalStore and
SqlAlchemyApprovalStore (contract backend parity). The parametrized
``store_factory`` fixture is copied verbatim from
``test_memory_store_contract.py`` (file + sqlalchemy branches, including the
``_run_in_new_loop`` helper that bootstraps the SQL engine off the test loop);
``Base.metadata.create_all`` already covers ``ApprovalRow`` since it subclasses
the same ``Base``.

ApprovalRequests are minted via ``build_approval_request`` (uuid4 id, fresh UTC
timestamps, version=1, PENDING). Uses the ``def test_x(store_factory):`` +
``asyncio.run(_run())`` style (sync test wrapper driving its own event loop) —
no pytest-asyncio mode config needed."""
import asyncio

import pytest

from linktools.ai.agent.approval import (
    ApprovalStatus,
    build_approval_request,
)
from linktools.ai.errors import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    InvalidApprovalTransitionError,
)
from linktools.ai.storage.file.approval import FileApprovalStore


# ---------------------------------------------------------------------------
# Parametrized store factory. The SQL branch (incl. ``_run_in_new_loop``) is
# copied verbatim from test_memory_store_contract.py / test_swarm_store_contract.py.
# ---------------------------------------------------------------------------


@pytest.fixture(params=["file", "sqlalchemy"])
def store_factory(request, tmp_path):
    if request.param == "file":
        counter = {"n": 0}

        def file_factory():
            counter["n"] += 1
            return FileApprovalStore(root=tmp_path / f"approval-{counter['n']}")

        return file_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from linktools.ai.storage.sqlalchemy.models import Base
    from linktools.ai.storage.sqlalchemy.approval import SqlAlchemyApprovalStore

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
            f"sqlite+aiosqlite:///{tmp_path}/approval-db-{counter['n']}.db"
        )
        engines.append(engine)

        async def _create():
            async with engine.begin() as conn:
                # ApprovalRow subclasses Base, so a single create_all covers
                # every table both backends need.
                await conn.run_sync(Base.metadata.create_all)
            # The connection pool otherwise holds a connection bound to this
            # thread's event loop; dispose it so later operations (running on
            # pytest-asyncio's loop) open fresh connections instead of reusing
            # one tied to a loop that is about to be closed.
            await engine.dispose()

        _run_in_new_loop(_create())
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        return SqlAlchemyApprovalStore(session_factory=session_factory)

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
# 1. create -> get round-trip (status PENDING, version 1, datetime tz-aware,
#    arguments mapping, default metadata).
# ---------------------------------------------------------------------------


def test_create_then_get_roundtrip(store_factory):
    store = store_factory()

    async def _run():
        req = build_approval_request(
            run_id="r1",
            tool_call_id="c1",
            tool_name="terminal",
            reason="need shell",
            arguments={"cmd": "ls", "args": ["-l"]},
        )
        created = await store.create(req)
        fetched = await store.get(req.id)
        assert fetched is not None
        # Frozen dataclass equality: every field round-trips identically on
        # both backends.
        assert fetched == created
        # Targeted checks for the load-bearing fields (status, version,
        # datetime tz-awareness, arguments mapping, default metadata).
        assert fetched.status is ApprovalStatus.PENDING
        assert fetched.version == 1
        assert fetched.resolved_at is None
        assert fetched.resolved_by is None
        assert fetched.run_id == "r1"
        assert fetched.tool_call_id == "c1"
        assert fetched.tool_name == "terminal"
        assert fetched.reason == "need shell"
        assert dict(fetched.arguments) == {"cmd": "ls", "args": ["-l"]}
        # build_approval_request leaves metadata as the empty default; it must
        # round-trip as an empty mapping (not None) on both backends.
        assert dict(fetched.metadata) == {}
        # created_at must remain tz-aware across both backends (SqlAlchemy
        # reattaches UTC on read; FileApprovalStore preserves isoformat tz).
        assert fetched.created_at.tzinfo is not None
        assert fetched.created_at == created.created_at

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 2. approve(id, expected_version=1, resolved_by=...) -> status APPROVED,
#    version 2, resolved_at set + tz-aware, resolved_by recorded.
# ---------------------------------------------------------------------------


def test_approve_transitions_to_approved(store_factory):
    store = store_factory()

    async def _run():
        req = build_approval_request(
            run_id="r1", tool_call_id="c1", tool_name="terminal", reason="x"
        )
        await store.create(req)
        approved = await store.approve(req.id, expected_version=1, resolved_by="alice")
        assert approved.status is ApprovalStatus.APPROVED
        assert approved.version == 2
        assert approved.resolved_by == "alice"
        assert approved.resolved_at is not None
        assert approved.resolved_at.tzinfo is not None
        # created_at is unchanged by resolution.
        assert approved.created_at == req.created_at
        # approve never touches metadata (so it can't shadow a prior rejection
        # reason on a different request).
        assert dict(approved.metadata) == {}
        # The change is persisted: a fresh get reflects the resolved state.
        refetched = await store.get(req.id)
        assert refetched is not None
        assert refetched.status is ApprovalStatus.APPROVED
        assert refetched.version == 2
        assert refetched.resolved_by == "alice"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3. reject(id, expected_version=1, resolved_by=..., reason=...) -> status
#    REJECTED, version 2, rejection reason recorded under
#    metadata["rejection_reason"].
# ---------------------------------------------------------------------------


def test_reject_transitions_to_rejected(store_factory):
    store = store_factory()

    async def _run():
        req = build_approval_request(
            run_id="r1", tool_call_id="c1", tool_name="terminal", reason="x"
        )
        await store.create(req)
        rejected = await store.reject(
            req.id, expected_version=1, resolved_by="bob", reason="no"
        )
        assert rejected.status is ApprovalStatus.REJECTED
        assert rejected.version == 2
        assert rejected.resolved_by == "bob"
        assert rejected.resolved_at is not None
        assert rejected.resolved_at.tzinfo is not None
        # reject always records the key (even when reason is None); a provided
        # reason lands under metadata["rejection_reason"].
        assert rejected.metadata.get("rejection_reason") == "no"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4. create duplicate id -> ApprovalConflictError. ``build_approval_request``
#    mints a fresh uuid each call, so to collide we reuse the same request
#    object (both backends enforce the uniqueness invariant — FileApprovalStore
#    via path.exists, SqlAlchemy via the primary-key constraint).
# ---------------------------------------------------------------------------


def test_create_duplicate_id_raises_conflict(store_factory):
    store = store_factory()

    async def _run():
        req = build_approval_request(
            run_id="r1", tool_call_id="c1", tool_name="terminal", reason="x"
        )
        await store.create(req)
        with pytest.raises(ApprovalConflictError):
            await store.create(req)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5. approve/reject wrong expected_version -> ApprovalConflictError; missing
#    id -> ApprovalNotFoundError.
# ---------------------------------------------------------------------------


def test_resolve_wrong_version_and_missing_id_raise(store_factory):
    store = store_factory()

    async def _run():
        req = build_approval_request(
            run_id="r1", tool_call_id="c1", tool_name="terminal", reason="x"
        )
        await store.create(req)
        # Wrong expected_version on both approve and reject -> conflict.
        with pytest.raises(ApprovalConflictError):
            await store.approve(req.id, expected_version=99, resolved_by="alice")
        with pytest.raises(ApprovalConflictError):
            await store.reject(req.id, expected_version=99, resolved_by="bob")
        # Missing id on both approve and reject -> not found.
        with pytest.raises(ApprovalNotFoundError):
            await store.approve("missing-id", expected_version=1, resolved_by="alice")
        with pytest.raises(ApprovalNotFoundError):
            await store.reject("missing-id", expected_version=1, resolved_by="bob")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 6. Double-resolve (approve then approve again) -> InvalidApprovalTransitionError.
#    APPROVED has an empty allowed-transitions set, so any further resolution
#    is rejected before the version check could pass.
# ---------------------------------------------------------------------------


def test_double_resolve_raises_invalid_transition(store_factory):
    store = store_factory()

    async def _run():
        req = build_approval_request(
            run_id="r1", tool_call_id="c1", tool_name="terminal", reason="x"
        )
        await store.create(req)
        await store.approve(req.id, expected_version=1, resolved_by="alice")
        # Second approve with the new expected_version (2) still fails: the
        # APPROVED -> APPROVED transition is not in ALLOWED_APPROVAL_TRANSITIONS.
        with pytest.raises(InvalidApprovalTransitionError):
            await store.approve(req.id, expected_version=2, resolved_by="alice")
        # And approve->reject on an already-approved request is likewise blocked
        # (transition guard fires before the version check).
        with pytest.raises(InvalidApprovalTransitionError):
            await store.reject(req.id, expected_version=2, resolved_by="bob")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 7. list_pending(run_id) filters by run_id + PENDING status: seed two runs,
#    approve one in run-1, list_pending("run-2") returns only run-2's pending.
# ---------------------------------------------------------------------------


def test_list_pending_filters_by_run_and_status(store_factory):
    store = store_factory()

    async def _run():
        a1 = build_approval_request(
            run_id="run-1", tool_call_id="c1", tool_name="terminal", reason="x"
        )
        a2 = build_approval_request(
            run_id="run-1", tool_call_id="c2", tool_name="terminal", reason="x"
        )
        b1 = build_approval_request(
            run_id="run-2", tool_call_id="c3", tool_name="terminal", reason="x"
        )
        await store.create(a1)
        await store.create(a2)
        await store.create(b1)
        # Before resolution: run-1 has both pending, run-2 has its one pending.
        assert {r.id for r in await store.list_pending("run-1")} == {a1.id, a2.id}
        assert {r.id for r in await store.list_pending("run-2")} == {b1.id}
        # Approve one in run-1 -> run-1's pending shrinks to a2; run-2 unchanged.
        await store.approve(a1.id, expected_version=1, resolved_by="alice")
        assert {r.id for r in await store.list_pending("run-1")} == {a2.id}
        assert {r.id for r in await store.list_pending("run-2")} == {b1.id}
        # Unknown run_id -> empty tuple (not None, not list).
        assert await store.list_pending("run-zzz") == ()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 8. File-only: path-traversal in approval_id -> ValueError. (SQL ids are
#    opaque primary-key strings, not path segments, so this guard is
#    FileApprovalStore-specific — mirrors the file-only path-traversal test in
#    test_memory_store_contract.py.)
# ---------------------------------------------------------------------------


def test_path_traversal_in_approval_id_is_rejected(tmp_path):
    store = FileApprovalStore(root=tmp_path)

    async def _run():
        with pytest.raises(ValueError):
            await store.get("../evil")
        with pytest.raises(ValueError):
            await store.approve("../evil", expected_version=1, resolved_by="x")
        with pytest.raises(ValueError):
            await store.reject("../evil", expected_version=1, resolved_by="x")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 9. list_for_run(run_id) is status-agnostic: seeds a run with one PENDING +
#    one APPROVED + one REJECTED request, asserts all three are returned
#    (ordered by created_at). This is the resume gate's read path -- the
#    executor's ``_already_approved`` helper consults it to recognize a call
#    that was approved externally. ``list_pending`` (case 7) filters these
#    down to just the PENDING one; ``list_for_run`` returns every status.
# ---------------------------------------------------------------------------


def test_list_for_run_is_status_agnostic(store_factory):
    store = store_factory()

    async def _run():
        # Three requests for run-1 in three statuses. created_at is minted by
        # build_approval_request in real UTC time; to make the order assertion
        # deterministic across the file backend (which sorts by created_at),
        # we sequence the build calls so their timestamps are strictly
        # increasing -- and we pin each request's id to its tool_call_id so
        # the assertions can refer to them by stable handles.
        a_pending = build_approval_request(
            run_id="run-1", tool_call_id="c-pending", tool_name="t", reason="x"
        )
        a_approved = build_approval_request(
            run_id="run-1", tool_call_id="c-approved", tool_name="t", reason="x"
        )
        a_rejected = build_approval_request(
            run_id="run-1", tool_call_id="c-rejected", tool_name="t", reason="x"
        )
        # A request in a DIFFERENT run -- must NOT appear in list_for_run("run-1").
        other_run = build_approval_request(
            run_id="run-2", tool_call_id="c-other", tool_name="t", reason="x"
        )
        await store.create(a_pending)
        await store.create(a_approved)
        await store.create(a_rejected)
        await store.create(other_run)
        # Flip the second into APPROVED and the third into REJECTED (version
        # moves to 2, resolved_at/resolved_by get set).
        await store.approve(
            a_approved.id, expected_version=1, resolved_by="alice"
        )
        await store.reject(
            a_rejected.id, expected_version=1, resolved_by="bob", reason="no"
        )

        all_for_run = await store.list_for_run("run-1")
        # Status-agnostic: all three run-1 requests are returned regardless of
        # status (PENDING + APPROVED + REJECTED). The other run's request is
        # excluded by the run_id filter.
        assert {r.tool_call_id for r in all_for_run} == {
            "c-pending",
            "c-approved",
            "c-rejected",
        }
        # All three statuses are represented exactly once.
        statuses = sorted(r.status.value for r in all_for_run)
        assert statuses == ["approved", "pending", "rejected"]
        # Ordered by created_at ascending (matches list_pending's ordering).
        created_ats = [r.created_at for r in all_for_run]
        assert created_ats == sorted(created_ats)
        # Other run excluded: list_for_run("run-1") does not leak run-2.
        assert all(r.run_id == "run-1" for r in all_for_run)

        # list_pending("run-1") on the same seed returns ONLY the PENDING one
        # (contrast against list_for_run's status-agnostic three).
        pending_only = await store.list_pending("run-1")
        assert {r.tool_call_id for r in pending_only} == {"c-pending"}

        # Unknown run_id -> empty tuple (not None, not list).
        assert await store.list_for_run("run-zzz") == ()

    asyncio.run(_run())
