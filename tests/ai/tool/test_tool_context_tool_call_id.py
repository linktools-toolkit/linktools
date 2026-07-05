#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 3 contract: ToolContext carries ToolCallPart.tool_call_id end-to-end,
so the executor keys ApprovalRequest.tool_call_id on the SAME id pydantic-ai
uses in message history -- the linchpin of resume (a re-driven call after
approve() must find the matching approval). When PolicyCapability populates
the field from a real ToolCallPart, the executor uses it verbatim; when a
caller constructs ToolContext directly without it, the uuid fallback fires
(preserving test_executor_approval.py's behavior unchanged)."""
import asyncio
import uuid

from linktools.ai.errors import ToolApprovalRequiredError
from linktools.ai.policy.engine import (
    PolicyDecision,
    PolicyDecisionKind,
    PolicyEngine,
    ToolContext,
    ToolRequest,
)
from linktools.ai.tool.executor import ToolExecutor


class _Require:
    async def evaluate(self, request, context):
        return PolicyDecision(kind=PolicyDecisionKind.REQUIRE_APPROVAL, rule_id="t", reason="x")


class _Store:
    def __init__(self):
        self.created = []

    async def create(self, request):
        self.created.append(request)
        return request

    async def get(self, approval_id):
        return None

    async def approve(self, approval_id, *, expected_version, resolved_by):
        raise NotImplementedError

    async def reject(self, approval_id, *, expected_version, resolved_by, reason=None):
        raise NotImplementedError

    async def list_pending(self, run_id):
        return ()


def test_tool_context_carries_tool_call_id_field_with_default_none():
    assert ToolContext(run_id="r", session_id="s").tool_call_id is None


def test_executor_uses_context_tool_call_id_when_present():
    store = _Store()
    executor = ToolExecutor(policy=PolicyEngine(rules=(_Require(),)), approval_store=store)

    async def _run():
        await executor.check(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r", session_id="s", tool_call_id="tcid-XYZ"),
        )

    try:
        asyncio.run(_run())
    except ToolApprovalRequiredError:
        pass
    assert store.created[0].tool_call_id == "tcid-XYZ"


def test_executor_falls_back_to_uuid_when_tool_call_id_missing():
    store = _Store()
    executor = ToolExecutor(policy=PolicyEngine(rules=(_Require(),)), approval_store=store)

    async def _run():
        await executor.check(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r", session_id="s"),
        )

    try:
        asyncio.run(_run())
    except ToolApprovalRequiredError:
        pass
    # parses as a uuid (the fallback path)
    uuid.UUID(store.created[0].tool_call_id)
