#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume gate for ToolExecutor: when policy says REQUIRE_APPROVAL but the
approval_store already holds an APPROVED request matching
``(run_id, tool_call_id)``, ``check()`` MUST let the call through (return
without raising) instead of re-persisting a PENDING request and re-raising.
This is the resume case -- after an external ``approve()`` flips the request
to APPROVED, the model re-drives the same tool call (same tool_call_id), and
the executor's ``_already_approved`` guard recognizes it and short-circuits.

Covers three branches:
1. APPROVED matching (run_id, tool_call_id) -> ``check()`` returns, no raise.
2. No matching approval at all -> ``check()`` still raises
   ``RunPaused``.
3. Matching tool_call_id but status PENDING (not yet approved) -> ``check()``
   still raises (resume must wait for an actual approve())."""

import asyncio
import dataclasses

import pytest

from linktools.ai.agent.approval import (
    ApprovalRequest,
    ApprovalStatus,
    build_approval_request,
)
from linktools.ai.errors import RunPaused
from linktools.ai.policy.engine import (
    PolicyDecision,
    PolicyDecisionKind,
    PolicyEngine,
    ToolContext,
    ToolRequest,
)
from linktools.ai.tool.executor import ToolExecutor


class _Require:
    """Rule that always returns REQUIRE_APPROVAL."""

    async def evaluate(self, request, context):
        return PolicyDecision(
            kind=PolicyDecisionKind.REQUIRE_APPROVAL,
            rule_id="test",
            reason="needs approval",
        )


class _Store:
    """Dict-backed ApprovalStore implementing the full Protocol, including
    ``list_for_run`` (status-agnostic). Pre-seeded via ``seed_with``."""

    def __init__(self):
        self._by_id: "dict[str, ApprovalRequest]" = {}
        self.created_count = 0

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        self._by_id[request.id] = request
        self.created_count += 1
        return request

    async def get(self, approval_id: str) -> "ApprovalRequest | None":
        return self._by_id.get(approval_id)

    async def approve(
        self, approval_id: str, *, expected_version: int, resolved_by: str
    ) -> ApprovalRequest:
        raise NotImplementedError

    async def reject(
        self,
        approval_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        reason: "str | None" = None,
    ) -> ApprovalRequest:
        raise NotImplementedError

    async def list_pending(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        return tuple(
            r
            for r in self._by_id.values()
            if r.run_id == run_id and r.status == ApprovalStatus.PENDING
        )

    async def list_for_run(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        return tuple(r for r in self._by_id.values() if r.run_id == run_id)


def _approved_request(
    *,
    run_id: str,
    tool_call_id: str,
    tool_name: str = "rm_rf",
    reason: "str | None" = "needs approval",
) -> ApprovalRequest:
    """Build an APPROVED ApprovalRequest (status=APPROVED, version=2,
    resolved_at/resolved_by set) -- the fixture for the resume case.

    Mirrors what ``store.approve(...)`` would produce: starts from a PENDING
    ``build_approval_request`` (uuid id, fresh UTC timestamps, version=1) and
    flips the approval-relevant fields via ``dataclasses.replace``."""
    pending = build_approval_request(
        run_id=run_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        reason=reason,
        arguments={"path": "/"},
        descriptor_fingerprint="descriptor-v1",
        handler_revision="handler-v1",
        provider_revision="provider-v1",
        policy_revision="policy-v1",
        capability_revision="capability-v1",
        result_processor_revision="processor-v1",
    )
    return dataclasses.replace(
        pending,
        status=ApprovalStatus.APPROVED,
        version=2,
        resolved_by="approver",
    )


def _request() -> ToolRequest:
    return ToolRequest(tool_name="rm_rf", arguments={"path": "/"})


def _context(tool_call_id: "str | None" = "tcid-XYZ") -> ToolContext:
    return ToolContext(
        run_id="run-123",
        session_id="sess-456",
        tool_call_id=tool_call_id,
        metadata={"descriptor_fingerprint": "descriptor-v1",
                  "handler_revision": "handler-v1", "provider_revision": "provider-v1",
                  "policy_revision": "policy-v1", "capability_revision": "capability-v1",
                  "result_processor_revision": "processor-v1"},
    )


def test_check_allows_through_when_already_approved():
    """Branch 1: resume case -- APPROVED request matching
    (run_id, tool_call_id) is in the store, so ``check()`` returns without
    raising AND does not persist a new PENDING request."""
    store = _Store()
    approved = _approved_request(run_id="run-123", tool_call_id="tcid-XYZ")
    store._by_id[approved.id] = approved
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )

    async def _run():
        # Must NOT raise -- the resume gate recognizes the prior approval.
        await executor.check(_request(), _context(tool_call_id="tcid-XYZ"))

    asyncio.run(_run())

    # The resume path must not have persisted a fresh PENDING request -- the
    # store still holds exactly the one APPROVED fixture we seeded.
    assert store.created_count == 0
    assert len(store._by_id) == 1
    only = next(iter(store._by_id.values()))
    assert only.status is ApprovalStatus.APPROVED
    assert only.tool_call_id == "tcid-XYZ"


def test_check_raises_when_no_matching_approval():
    """Branch 2: no matching approval at all -> ``check()`` raises RunPaused.
    The executor only emits the signal; it does not persist (the caller does)."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )

    async def _run():
        await executor.check(_request(), _context(tool_call_id="tcid-OTHER"))

    with pytest.raises(RunPaused):
        asyncio.run(_run())

    # No prior approval seeded; the executor persists nothing.
    assert store.created_count == 0


def test_check_raises_when_matching_request_is_pending():
    """Branch 3: matching tool_call_id exists but status is PENDING (not yet
    approved) -> the resume gate does not fire, so ``check()`` raises
    ``RunPaused`` (resume must wait for an actual approve())."""
    store = _Store()
    pending = build_approval_request(
        run_id="run-123",
        tool_call_id="tcid-PENDING",
        tool_name="rm_rf",
        reason="needs approval",
    )
    store._by_id[pending.id] = pending
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )

    async def _run():
        await executor.check(_request(), _context(tool_call_id="tcid-PENDING"))

    with pytest.raises(RunPaused):
        asyncio.run(_run())

    # PENDING does not satisfy the resume gate; the executor persists nothing.
    assert store.created_count == 0


def test_check_raises_when_no_tool_call_id_in_context():
    """Branch 4: ``context.tool_call_id is None`` -> ``_already_approved``
    returns False early (no key to match on), so ``check()`` raises RunPaused.
    The uuid fallback mints a tool_call_id on the RunPaused signal; resume
    cannot apply because there was no stable id to match."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )

    async def _run():
        await executor.check(_request(), _context(tool_call_id=None))

    with pytest.raises(RunPaused):
        asyncio.run(_run())

    # No tool_call_id -> no resume -> the executor persists nothing.
    assert store.created_count == 0


def test_check_without_approval_store_raises_normally():
    """Branch 5: ``approval_store is None`` -> ``_already_approved`` returns
    False early; ``check()`` raises ``RunPaused`` exactly like
    the default-None path always has (no possible resume gate without a
    store)."""
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
    )

    async def _run():
        await executor.check(_request(), _context(tool_call_id="tcid-XYZ"))

    with pytest.raises(RunPaused):
        asyncio.run(_run())
