#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval-flow contract for ToolExecutor: when policy says REQUIRE_APPROVAL
and an approval_store is wired, the executor must persist a PENDING
ApprovalRequest, emit an ApprovalRequested event (if an event_store is
wired), and STILL raise ToolApprovalRequiredError so PolicyCapability
translates it into SkipToolExecution. Default-None (no stores wired)
preserves today's behavior identically."""
import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from linktools.ai.agent.approval import (
    ApprovalRequest,
    ApprovalStatus,
)
from linktools.ai.errors import ToolApprovalRequiredError
from linktools.ai.events.envelope import EventEnvelope
from linktools.ai.events.payloads import ApprovalRequested
from linktools.ai.events.store import EventPage
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


class _StubApprovalStore:
    """Dict-backed ApprovalStore: implements create/get/approve/reject/list_pending."""

    def __init__(self):
        self._by_id: "dict[str, ApprovalRequest]" = {}

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        self._by_id[request.id] = request
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

    def __len__(self) -> int:
        return len(self._by_id)


class _StubEventStore:
    """List-backed EventStore: records every appended envelope. Assigns the
    sequence itself (mirrors the real Protocol -- review doc §8.1)."""

    def __init__(self):
        self.events: "list[EventEnvelope]" = []

    async def append(
        self,
        *,
        stream_id: str,
        run_id: str,
        root_run_id: str,
        parent_run_id: "str | None",
        session_id: str,
        runnable_id: str,
        payload,
    ) -> EventEnvelope:
        sequence = sum(1 for e in self.events if e.stream_id == stream_id) + 1
        envelope = EventEnvelope(
            event_id=f"evt-{run_id}-{sequence}", stream_id=stream_id, sequence=sequence,
            occurred_at=datetime.now(timezone.utc), run_id=run_id,
            root_run_id=root_run_id, parent_run_id=parent_run_id,
            session_id=session_id, runnable_id=runnable_id, payload=payload,
        )
        self.events.append(envelope)
        return envelope

    async def list(
        self, stream_id: str, *, after_sequence: int = 0, limit: int = 100
    ) -> EventPage:
        items = tuple(
            e for e in self.events if e.stream_id == stream_id and e.sequence > after_sequence
        )
        return EventPage(items=items[:limit], cursor=None)


def _request() -> ToolRequest:
    return ToolRequest(tool_name="rm_rf", arguments={"path": "/"})


def _context() -> ToolContext:
    return ToolContext(run_id="run-123", session_id="sess-456")


def test_check_with_approval_store_persists_pending_request_and_raises():
    store = _StubApprovalStore()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )

    async def _run():
        await executor.check(_request(), _context())

    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(_run())

    # Exactly one PENDING ApprovalRequest was persisted.
    assert len(store) == 1
    pending = store._by_id[next(iter(store._by_id))]
    assert pending.status is ApprovalStatus.PENDING
    assert pending.tool_name == "rm_rf"
    assert pending.run_id == "run-123"
    assert pending.reason == "needs approval"
    assert dict(pending.arguments) == {"path": "/"}
    # build_approval_request mints a uuid; tool_call_id is a fresh uuid string.
    uuid.UUID(pending.tool_call_id)
    assert pending.version == 1


def test_check_with_event_store_also_emits_approval_requested_event():
    store = _StubApprovalStore()
    events = _StubEventStore()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
        event_store=events,
    )

    async def _run():
        await executor.check(_request(), _context())

    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(_run())

    assert len(events.events) == 1
    envelope = events.events[0]
    payload = envelope.payload
    assert isinstance(payload, ApprovalRequested)
    assert payload.tool_name == "rm_rf"
    assert payload.reason == "needs approval"
    # event's approval_id matches the persisted request's id
    assert len(store) == 1
    persisted = store._by_id[next(iter(store._by_id))]
    assert payload.approval_id == persisted.id
    # envelope routing fields
    assert envelope.run_id == "run-123"
    assert envelope.root_run_id == "run-123"
    assert envelope.parent_run_id is None
    assert envelope.session_id == "sess-456"
    assert envelope.runnable_id == "rm_rf"


def test_check_without_approval_store_raises_and_persists_nothing():
    # Default-None path: behavior is IDENTICAL to today.
    events = _StubEventStore()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        event_store=events,
    )

    async def _run():
        await executor.check(_request(), _context())

    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(_run())

    # No approval persisted (no store wired) AND no event emitted (event emission
    # is gated on approval_store, since there's no approval_id to reference).
    assert len(events.events) == 0


def test_execute_with_approval_store_raises_and_handler_not_called_request_persisted():
    store = _StubApprovalStore()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )
    ran = {"handler": False}

    async def _handler(**kwargs):
        ran["handler"] = True
        return "should-not-reach"

    async def _run():
        await executor.execute(_request(), _context(), _handler)

    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(_run())

    assert ran["handler"] is False
    assert len(store) == 1


def test_check_with_run_id_resolver_uses_custom_run_id_for_approval_and_event():
    """Test that run_id_resolver overrides context.run_id for both the persisted
    ApprovalRequest and the emitted ApprovalRequested event."""
    store = _StubApprovalStore()
    events = _StubEventStore()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
        event_store=events,
        run_id_resolver=lambda ctx: "resolved-run-99",
    )

    async def _run():
        await executor.check(_request(), _context())

    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(_run())

    # ApprovalRequest.run_id is the resolved run_id, NOT context.run_id
    assert len(store) == 1
    persisted = store._by_id[next(iter(store._by_id))]
    assert persisted.run_id == "resolved-run-99"
    assert persisted.run_id != "run-123"  # context.run_id was overridden

    # Event envelope run_id is also the resolved run_id
    assert len(events.events) == 1
    envelope = events.events[0]
    assert envelope.run_id == "resolved-run-99"
    assert envelope.run_id != "run-123"  # context.run_id was overridden
