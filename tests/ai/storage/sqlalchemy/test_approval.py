#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/sqlalchemy/test_approval.py — SqlAlchemyApprovalStore
contract. Uses the `def test_x(): asyncio.run(_run())` style (sync test wrapper
driving its own event loop) so no pytest-asyncio mode config is needed. Mirrors
how test_memory.py bootstraps the in-file aiosqlite engine; mirrors the File
test's 5-method contract (round-trip, conflict-on-dup, approve/reject semantics,
version conflict, not-found, double-resolve InvalidApprovalTransition,
list_pending filter)."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.agent.approval import (
    ApprovalStatus,
    build_approval_request,
)
from linktools.ai.errors import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    InvalidApprovalTransitionError,
)
from linktools.ai.storage.sqlalchemy.approval import SqlAlchemyApprovalStore
from linktools.ai.storage.sqlalchemy.models import Base


@asynccontextmanager
async def _store_ctx(tmp_path):
    """Build a SqlAlchemyApprovalStore against an in-file SQLite DB. The engine
    is disposed on exit so aiosqlite's background worker threads shut down before
    the per-test event loop closes (otherwise they call into a dead loop)."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/approval.db")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        yield SqlAlchemyApprovalStore(session_factory=session_factory)
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# 1. create -> get round-trips all fields
# ---------------------------------------------------------------------------


def test_create_then_get_roundtrips_all_fields(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            # build_approval_request doesn't accept metadata, so attach it via
            # replace.
            req = replace(
                build_approval_request(
                    run_id="run-1",
                    tool_call_id="call-1",
                    tool_name="shell",
                    reason="needs human review",
                    arguments={"cmd": "rm -rf /"},
                ),
                metadata={"source": "policy"},
            )
            await store.create(req)
            fetched = await store.get(req.id)
            assert fetched is not None
            assert fetched.id == req.id
            assert fetched.run_id == "run-1"
            assert fetched.tool_call_id == "call-1"
            assert fetched.tool_name == "shell"
            assert fetched.reason == "needs human review"
            assert fetched.arguments == {"cmd": "rm -rf /"}
            assert fetched.status is ApprovalStatus.PENDING
            assert fetched.version == 1
            assert fetched.created_at == req.created_at
            # datetime reattached with UTC tzinfo on read (aiosqlite strips it).
            assert fetched.created_at.tzinfo is not None
            assert fetched.resolved_at is None
            assert fetched.resolved_by is None
            assert fetched.metadata == {"source": "policy"}

    asyncio.run(_run_case())


def test_get_missing_returns_none(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            assert await store.get("nope") is None

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 2. create conflict on duplicate id (IntegrityError -> ApprovalConflictError)
# ---------------------------------------------------------------------------


def test_create_duplicate_id_raises_conflict(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            req = build_approval_request(run_id="run-1", tool_call_id="c", tool_name="t")
            await store.create(req)
            with pytest.raises(ApprovalConflictError):
                await store.create(req)

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 3. approve -> status APPROVED, version=2, resolved_at set, resolved_by
# ---------------------------------------------------------------------------


def test_approve_transitions_to_approved(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            req = build_approval_request(run_id="run-1", tool_call_id="c", tool_name="t")
            await store.create(req)
            resolved = await store.approve(req.id, expected_version=1, resolved_by="alice")
            assert resolved.status is ApprovalStatus.APPROVED
            assert resolved.version == 2
            assert resolved.resolved_at is not None
            assert resolved.resolved_by == "alice"
            assert resolved.created_at == req.created_at
            # approve does NOT touch metadata (no rejection_reason key).
            assert "rejection_reason" not in resolved.metadata
            # Persisted.
            refetched = await store.get(req.id)
            assert refetched is not None
            assert refetched.status is ApprovalStatus.APPROVED
            assert refetched.version == 2
            assert refetched.resolved_by == "alice"

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 4. reject -> status REJECTED, version=2, reason stored in metadata
# ---------------------------------------------------------------------------


def test_reject_transitions_to_rejected_and_stores_reason(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            req = replace(
                build_approval_request(run_id="run-1", tool_call_id="c", tool_name="t"),
                metadata={"source": "policy"},
            )
            await store.create(req)
            resolved = await store.reject(
                req.id, expected_version=1, resolved_by="bob", reason="too risky"
            )
            assert resolved.status is ApprovalStatus.REJECTED
            assert resolved.version == 2
            assert resolved.resolved_at is not None
            assert resolved.resolved_by == "bob"
            # Reason is stored under metadata["rejection_reason"].
            assert resolved.metadata.get("rejection_reason") == "too risky"
            # Pre-existing metadata is preserved alongside the new key.
            assert resolved.metadata.get("source") == "policy"
            # Persisted.
            refetched = await store.get(req.id)
            assert refetched is not None
            assert refetched.metadata.get("rejection_reason") == "too risky"

    asyncio.run(_run_case())


def test_reject_without_reason_stores_none_key(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            req = build_approval_request(run_id="run-1", tool_call_id="c", tool_name="t")
            await store.create(req)
            resolved = await store.reject(req.id, expected_version=1, resolved_by="bob")
            assert resolved.status is ApprovalStatus.REJECTED
            # When reason is None we still record the key as None.
            assert "rejection_reason" in resolved.metadata
            assert resolved.metadata["rejection_reason"] is None

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 5. approve/reject wrong expected_version / missing id
# ---------------------------------------------------------------------------


def test_resolve_wrong_expected_version_raises_conflict(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            req = build_approval_request(run_id="run-1", tool_call_id="c", tool_name="t")
            await store.create(req)
            with pytest.raises(ApprovalConflictError):
                await store.approve(req.id, expected_version=99, resolved_by="alice")
            with pytest.raises(ApprovalConflictError):
                await store.reject(req.id, expected_version=99, resolved_by="bob")

    asyncio.run(_run_case())


def test_resolve_missing_id_raises_not_found(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            with pytest.raises(ApprovalNotFoundError):
                await store.approve("ghost", expected_version=1, resolved_by="alice")
            with pytest.raises(ApprovalNotFoundError):
                await store.reject("ghost", expected_version=1, resolved_by="bob")

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 6. double-resolve -> InvalidApprovalTransitionError
# ---------------------------------------------------------------------------


def test_double_resolve_raises_invalid_transition(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            req = build_approval_request(run_id="run-1", tool_call_id="c", tool_name="t")
            await store.create(req)
            await store.approve(req.id, expected_version=1, resolved_by="alice")
            # Second approve with the new expected_version still can't transition
            # APPROVED -> APPROVED (terminal state).
            with pytest.raises(InvalidApprovalTransitionError):
                await store.approve(req.id, expected_version=2, resolved_by="alice")
            # APPROVED -> REJECTED is also forbidden.
            with pytest.raises(InvalidApprovalTransitionError):
                await store.reject(req.id, expected_version=2, resolved_by="bob")

    asyncio.run(_run_case())


def test_reject_then_reject_raises_invalid_transition(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            req = build_approval_request(run_id="run-1", tool_call_id="c", tool_name="t")
            await store.create(req)
            await store.reject(req.id, expected_version=1, resolved_by="bob", reason="no")
            with pytest.raises(InvalidApprovalTransitionError):
                await store.reject(req.id, expected_version=2, resolved_by="bob")

    asyncio.run(_run_case())


# ---------------------------------------------------------------------------
# 7. list_pending(run_id) returns only PENDING requests for that run
# ---------------------------------------------------------------------------


def test_list_pending_filters_by_run_and_status(tmp_path):
    async def _run_case():
        async with _store_ctx(tmp_path) as store:
            a1 = build_approval_request(run_id="run-a", tool_call_id="c1", tool_name="t")
            a2 = build_approval_request(run_id="run-a", tool_call_id="c2", tool_name="t")
            b1 = build_approval_request(run_id="run-b", tool_call_id="c3", tool_name="t")
            for r in (a1, a2, b1):
                await store.create(r)

            pending_a = await store.list_pending("run-a")
            assert {r.id for r in pending_a} == {a1.id, a2.id}

            # Resolve a1; it should drop out of run-a's pending list.
            await store.approve(a1.id, expected_version=1, resolved_by="alice")
            pending_a_after = await store.list_pending("run-a")
            assert {r.id for r in pending_a_after} == {a2.id}

            # run-b is unaffected.
            pending_b = await store.list_pending("run-b")
            assert {r.id for r in pending_b} == {b1.id}

            # Unknown run -> empty tuple.
            assert await store.list_pending("run-zzz") == ()

    asyncio.run(_run_case())
