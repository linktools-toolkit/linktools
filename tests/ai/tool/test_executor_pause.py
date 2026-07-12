#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pause-mode contract for ToolExecutor: when ``pause_on_approval=True`` and
policy says REQUIRE_APPROVAL, ``check()`` raises ``RunPaused`` carrying every
field AgentRunner's suspension handler needs to persist the ApprovalRequest
ITSELF -- the executor no longer persists anything (review3 contract/scenario,
P0-6/G1). This is the key behavior change from the pre-Package-A contract:
previously the executor called ``_record_approval`` (persisting a PENDING
request) BEFORE raising; now persistence is deferred entirely to
AgentRunner's pause handler so it can share one UnitOfWork with the
checkpoint save + WAITING_APPROVAL transition + pause events (see
tests/ai/agent/test_runner_pause_atomic.py for the atomicity contract).

``RunPaused`` is a ``RunError`` (not a ``ToolError``), so
``PolicyCapability.before_tool_execute`` -- which only catches
``ToolDeniedError``/``ToolApprovalRequiredError`` -- does NOT translate it into
``SkipToolExecution``; it propagates out of pydantic-ai's tool-execution stack
to ``AgentRunner``.

Default ``pause_on_approval=False`` is unchanged: still persists via
``_record_approval`` before raising ``ToolApprovalRequiredError``."""

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
        raise NotImplementedError  # not exercised by ToolExecutor anymore

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


def test_pause_mode_raises_run_paused_without_persisting():
    """P0-6/G1: pause_on_approval=True raises RunPaused carrying every field
    the suspension handler needs (tool_call_id/tool_name/reason/arguments) --
    but the executor itself does NOT touch the ApprovalStore. Persistence is
    the caller's (AgentRunner's) responsibility now."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
        pause_on_approval=True,
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
    # The executor must NOT have persisted anything -- that's now
    # AgentRunner's suspension handler's job.
    assert store.created == []


def test_default_mode_still_raises_tool_approval_required_error():
    """Default ``pause_on_approval=False`` is unchanged: still persists +
    still raises ``ToolApprovalRequiredError``."""
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
    """``RunPaused.run_id`` honors ``run_id_resolver`` -- so AgentRunner's
    checkpoint and eventual ApprovalRequest both key on the resolved id."""
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
    assert store.created == []  # still nothing persisted by the executor


def test_pause_mode_mints_a_tool_call_id_when_context_carries_none():
    """When ToolContext has no tool_call_id (e.g. a test constructing it
    directly), the executor mints a fresh uuid so RunPaused.tool_call_id is
    never None -- the suspension handler needs a stable key to persist under."""
    store = _Store()
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)),
        approval_store=store,
        pause_on_approval=True,
    )

    async def _run():
        await executor.check(
            ToolRequest(tool_name="rm", arguments={}),
            ToolContext(run_id="r1", session_id="s1", tool_call_id=None),
        )

    with pytest.raises(RunPaused) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.tool_call_id is not None
