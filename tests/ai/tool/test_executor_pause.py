#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval contract for ToolExecutor: when policy says REQUIRE_APPROVAL,
``check()`` raises ``RunPaused`` carrying every field AgentRunner's suspension
handler needs to persist the ApprovalRequest itself -- the executor does not
persist anything. Persistence is deferred entirely to AgentRunner's pause
handler so it can share one UnitOfWork with the checkpoint save +
WAITING_APPROVAL transition + pause events (see
tests/ai/agent/test_runner_pause_atomic.py for the atomicity contract).

``RunPaused`` is a ``RunError`` (not a ``ToolError``), so
``PolicyCapability.before_tool_execute`` -- which only catches
``ToolDeniedError``/``ToolApprovalRequiredError`` -- does NOT translate it into
``SkipToolExecution``; it propagates out of pydantic-ai's tool-execution stack
to ``AgentRunner``."""

import asyncio

import pytest

from linktools.ai.agent.approval import ApprovalRequest
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

    async def create_or_get_pending(
        self,
        *,
        run_id,
        tool_call_id,
        tool_name,
        reason,
        arguments,
        approval_id,
    ) -> ApprovalRequest:
        raise NotImplementedError  # not exercised by ToolExecutor

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


def test_check_raises_run_paused_without_persisting():
    """REQUIRE_APPROVAL raises RunPaused carrying every field the suspension
    handler needs (tool_call_id/tool_name/reason/arguments) -- but the executor
    itself does NOT touch the ApprovalStore. Persistence is the caller's
    (AgentRunner's) responsibility."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )

    async def _run():
        await executor.check(
            ToolRequest(tool_name="rm", arguments={"path": "/tmp/x"}),
            ToolContext(run_id="r1", session_id="s1", tool_call_id="tc1"),
        )

    with pytest.raises(RunPaused) as exc_info:
        asyncio.run(_run())

    paused = exc_info.value
    assert paused.run_id == "r1"
    assert paused.approval_id  # minted, but not yet persisted anywhere
    assert paused.tool_call_id == "tc1"
    assert paused.tool_name == "rm"
    assert paused.reason == "x"
    assert paused.arguments == {"path": "/tmp/x"}
    # The executor must NOT have persisted anything -- that's AgentRunner's job.
    assert store.created == []


def test_check_with_run_id_resolver_uses_resolved_run_id():
    """``RunPaused.run_id`` honors ``run_id_resolver`` -- so AgentRunner's
    checkpoint and eventual ApprovalRequest both key on the resolved id."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
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
    assert store.created == []  # still nothing persisted by the executor


def test_check_mints_a_tool_call_id_when_context_carries_none():
    """When ToolContext has no tool_call_id, the executor mints a fresh uuid so
    RunPaused.tool_call_id is never None -- the suspension handler needs a
    stable key to persist under."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
    )

    async def _run():
        await executor.check(
            ToolRequest(tool_name="rm", arguments={}),
            ToolContext(run_id="r1", session_id="s1", tool_call_id=None),
        )

    with pytest.raises(RunPaused) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.tool_call_id is not None
