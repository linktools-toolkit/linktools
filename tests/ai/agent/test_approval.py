#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for agent.approval: ApprovalStatus, ApprovalRequest,
ALLOWED_APPROVAL_TRANSITIONS, build_approval_request factory, and the
ApprovalStore Protocol. Pure data/Protocol checks -- no I/O."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from linktools.ai.agent.approval import (
    ALLOWED_APPROVAL_TRANSITIONS,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalStore,
    build_approval_request,
)


# --- ApprovalStatus enum ----------------------------------------------------


def test_approval_status_values():
    assert ApprovalStatus.PENDING.value == "pending"
    assert ApprovalStatus.APPROVED.value == "approved"
    assert ApprovalStatus.REJECTED.value == "rejected"


def test_approval_status_is_str_enum():
    assert isinstance(ApprovalStatus.PENDING, str)
    assert ApprovalStatus.PENDING == "pending"
    assert ApprovalStatus.APPROVED == "approved"
    assert ApprovalStatus.REJECTED == "rejected"


# --- ALLOWED_APPROVAL_TRANSITIONS -------------------------------------------


def test_allowed_transitions_pending():
    assert ALLOWED_APPROVAL_TRANSITIONS[ApprovalStatus.PENDING] == frozenset(
        {
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
        }
    )


@pytest.mark.parametrize("status", [ApprovalStatus.APPROVED, ApprovalStatus.REJECTED])
def test_allowed_transitions_terminals_empty(status):
    assert ALLOWED_APPROVAL_TRANSITIONS[status] == frozenset()


def test_allowed_transitions_values_are_frozensets():
    for status, allowed in ALLOWED_APPROVAL_TRANSITIONS.items():
        assert isinstance(allowed, frozenset), status


# --- ApprovalRequest --------------------------------------------------------


def _full_request():
    now = datetime.now(timezone.utc)
    return ApprovalRequest(
        id="ap-1",
        run_id="r-1",
        tool_call_id="c-1",
        tool_name="terminal",
        reason="rm -rf",
        redacted_arguments={"target": "/"},
        arguments_hash="hash-target",
        status=ApprovalStatus.PENDING,
        version=1,
        created_at=now,
        resolved_at=None,
        resolved_by=None,
        metadata={"k": "v"},
    )


def test_approval_request_construct_all_fields():
    req = _full_request()
    assert req.id == "ap-1"
    assert req.run_id == "r-1"
    assert req.tool_call_id == "c-1"
    assert req.tool_name == "terminal"
    assert req.reason == "rm -rf"
    assert req.redacted_arguments == {"target": "/"}
    assert req.status is ApprovalStatus.PENDING
    assert req.version == 1
    assert req.resolved_at is None
    assert req.resolved_by is None
    assert req.metadata == {"k": "v"}


def test_approval_request_created_at_is_tz_aware():
    req = _full_request()
    assert req.created_at.tzinfo is timezone.utc


def test_approval_request_metadata_defaults_empty():
    now = datetime.now(timezone.utc)
    req = ApprovalRequest(
        id="ap-1",
        run_id="r-1",
        tool_call_id="c-1",
        tool_name="terminal",
        reason=None,
        redacted_arguments={},
        arguments_hash="hash-empty",
        status=ApprovalStatus.PENDING,
        version=1,
        created_at=now,
        resolved_at=None,
        resolved_by=None,
    )
    assert req.metadata == {}


def test_approval_request_frozen():
    req = _full_request()
    with pytest.raises(FrozenInstanceError):
        req.status = ApprovalStatus.APPROVED  # type: ignore[misc]


# --- build_approval_request -------------------------------------------------


def test_build_approval_request_defaults():
    req = build_approval_request(
        run_id="r1", tool_call_id="c1", tool_name="terminal", reason="rm -rf"
    )
    assert req.run_id == "r1"
    assert req.tool_call_id == "c1"
    assert req.tool_name == "terminal"
    assert req.reason == "rm -rf"
    assert req.status is ApprovalStatus.PENDING
    assert req.version == 1
    assert req.resolved_at is None
    assert req.resolved_by is None
    assert req.redacted_arguments == {}
    assert req.metadata == {}
    assert req.created_at.tzinfo is timezone.utc


def test_build_approval_request_id_is_uuid4():
    req = build_approval_request(run_id="r1", tool_call_id="c1", tool_name="t")
    # uuid4 string form: 8-4-4-4-12 hex chars
    parts = req.id.split("-")
    assert len(parts) == 5
    assert len(req.id) == 36


def test_build_approval_request_copies_arguments():
    src = {"a": 1, "b": 2}
    req = build_approval_request(
        run_id="r1", tool_call_id="c1", tool_name="t", arguments=src
    )
    assert req.redacted_arguments == {"a": 1, "b": 2}
    # mutating the source after the fact does not leak into the record
    src["a"] = 999
    assert req.redacted_arguments == {"a": 1, "b": 2}


def test_build_approval_request_distinct_ids():
    a = build_approval_request(run_id="r1", tool_call_id="c1", tool_name="t")
    b = build_approval_request(run_id="r1", tool_call_id="c1", tool_name="t")
    assert a.id != b.id


# --- ApprovalStore Protocol -------------------------------------------------


class _StubStore:
    async def create(self, request): ...

    async def create_or_get_pending(
        self,
        *,
        run_id,
        tool_call_id,
        tool_name,
        reason,
        arguments,
        approval_id,
    ): ...

    async def get(self, approval_id): ...

    async def approve(self, approval_id, *, expected_version, resolved_by): ...

    async def reject(
        self, approval_id, *, expected_version, resolved_by, reason=None
    ): ...

    async def list_pending(self, run_id): ...

    async def list_for_run(self, run_id): ...


def test_approval_store_is_runtime_checkable():
    assert isinstance(_StubStore(), ApprovalStore)


def test_approval_store_rejects_non_implementor():
    class _Incomplete:
        async def create(self, request): ...

    assert not isinstance(_Incomplete(), ApprovalStore)


def test_approval_store_stub_methods_are_async():
    """A stub implementing all 6 async methods satisfies the Protocol; calling
    them returns a coroutine (sanity check on the async signature)."""
    import inspect

    stub = _StubStore()
    for method_name in (
        "create",
        "get",
        "approve",
        "reject",
        "list_pending",
        "list_for_run",
    ):
        assert inspect.iscoroutinefunction(getattr(stub, method_name)), method_name


# --- argument redaction + identity hash ----------------------


def test_build_approval_request_redacts_secret_arguments():
    # The REAL arguments never reach the persisted record: secret-keyed values
    # are masked in redacted_arguments, while the identity hash is computed
    # over the REAL arguments (so two calls differing only in a secret are
    # distinct calls, not a false dedupe hit).
    from linktools.ai.agent.approval import compute_arguments_hash

    raw = {"host": "prod.db", "password": "s3cr3t", "token": "abc", "cmd": "ls"}
    req = build_approval_request(
        run_id="r", tool_call_id="c", tool_name="deploy", arguments=raw
    )
    assert req.redacted_arguments["password"] == "***REDACTED***"
    assert req.redacted_arguments["token"] == "***REDACTED***"
    assert req.redacted_arguments["host"] == "prod.db"  # non-secret visible
    assert req.redacted_arguments["cmd"] == "ls"
    # Hash over the REAL arguments.
    assert req.arguments_hash == compute_arguments_hash("deploy", raw)
    # Two calls differing only in the secret value hash differently.
    other = build_approval_request(
        run_id="r",
        tool_call_id="c2",
        tool_name="deploy",
        arguments={"host": "prod.db", "password": "DIFFERENT", "cmd": "ls"},
    )
    assert other.arguments_hash != req.arguments_hash


def test_check_dedupe_conflict_uses_hash_not_raw():
    # Dedupe identity is the arguments hash; the redacted copy is never compared
    # (two calls redacting to the same mask but with different real secrets must
    # still conflict-detect as distinct -- covered above; here: same real args
    # do not conflict, different real args do).
    from linktools.ai.agent.approval import check_dedupe_conflict

    existing = build_approval_request(
        run_id="r", tool_call_id="c", tool_name="deploy",
        arguments={"password": "x", "host": "h"},
    )
    # Same args -> no conflict.
    check_dedupe_conflict(
        existing, tool_name="deploy",
        arguments={"password": "x", "host": "h"},
    )
    # Different args -> conflict (even though the redacted copy masks both).
    with pytest.raises(Exception):
        check_dedupe_conflict(
            existing, tool_name="deploy",
            arguments={"password": "y", "host": "h"},
        )
