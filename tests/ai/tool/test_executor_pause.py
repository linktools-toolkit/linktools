#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pause-mode contract for ToolExecutor: when ``pause_on_approval=True`` and
policy says REQUIRE_APPROVAL, ``check()`` must persist the PENDING
ApprovalRequest (Task 4's ``_record_approval``) and then raise
``RunPaused(run_id, approval_id)`` INSTEAD of ``ToolApprovalRequiredError``.

``RunPaused`` is a ``RunError`` (not a ``ToolError``), so
``PolicyCapability.before_tool_execute`` -- which only catches
``ToolDeniedError``/``ToolApprovalRequiredError`` -- does NOT translate it into
``SkipToolExecution``; it propagates out of pydantic-ai's tool-execution stack
to ``AgentRunner`` (Tasks 6-7), which checkpoints state, transitions the Run to
WAITING_APPROVAL, and stops.

Default ``pause_on_approval=False`` is byte-for-byte identical to today's
behavior (raises ``ToolApprovalRequiredError`` after ``_record_approval``)."""
import asyncio

import pytest

from linktools.ai.agent.approval import ApprovalRequest
from linktools.ai.errors import RunPaused, ToolApprovalRequiredError
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
            rule_id="t",
            reason="x",
        )


class _Store:
    """Dict-backed ApprovalStore implementing the full Protocol, including
    ``list_for_run`` (status-agnostic, consulted by the resume gate)."""

    def __init__(self):
        self.created: "list[ApprovalRequest]" = []

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        self.created.append(request)
        return request

    async def get(self, approval_id: str) -> "ApprovalRequest | None":
        return None

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
        return ()

    async def list_for_run(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        return ()


def test_pause_mode_raises_run_paused_with_ids():
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
        pause_on_approval=True,
    )

    async def _run():
        await executor.check(
            ToolRequest(tool_name="rm", arguments={}),
            ToolContext(run_id="r1", session_id="s1", tool_call_id="tc1"),
        )

    with pytest.raises(RunPaused) as exc_info:
        asyncio.run(_run())

    # RunPaused carries both ids AgentRunner needs to checkpoint + transition.
    assert exc_info.value.run_id == "r1"
    assert exc_info.value.approval_id == store.created[0].id
    # The persisted PENDING request is the source of truth for the pause UI.
    assert len(store.created) == 1
    assert store.created[0].id == exc_info.value.approval_id


def test_default_mode_still_raises_tool_approval_required_error():
    """Default ``pause_on_approval=False`` is byte-for-byte identical to today:
    still persists + still raises ``ToolApprovalRequiredError``."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )  # pause_on_approval=False default

    async def _run():
        await executor.check(
            ToolRequest(tool_name="rm", arguments={}),
            ToolContext(run_id="r1", session_id="s1", tool_call_id="tc1"),
        )

    with pytest.raises(ToolApprovalRequiredError):
        asyncio.run(_run())

    # The default-False path still persists the PENDING request (unchanged).
    assert len(store.created) == 1


def test_pause_mode_with_run_id_resolver_uses_resolved_run_id():
    """``RunPaused.run_id`` honors ``run_id_resolver`` exactly as
    ``_record_approval`` does -- so AgentRunner's checkpoint keys on the same
    resolved id the persisted ApprovalRequest uses."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
        pause_on_approval=True,
        run_id_resolver=lambda ctx: "resolved-run-99",
    )

    async def _run():
        await executor.check(
            ToolRequest(tool_name="rm", arguments={}),
            ToolContext(run_id="r1", session_id="s1", tool_call_id="tc1"),
        )

    with pytest.raises(RunPaused) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.run_id == "resolved-run-99"
    assert exc_info.value.run_id != "r1"  # context.run_id was overridden
    # Persisted request also carries the resolved run_id.
    assert store.created[0].run_id == "resolved-run-99"
