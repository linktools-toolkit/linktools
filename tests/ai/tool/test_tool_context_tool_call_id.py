#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scenario contract: ToolContext carries ToolCallPart.tool_call_id end-to-end,
so the executor keys RunPaused.tool_call_id on the SAME id pydantic-ai uses in
message history -- the linchpin of resume (a re-driven call after approve()
must find the matching approval). When PolicyCapability populates the field
from a real ToolCallPart, the executor uses it verbatim; when a caller
constructs ToolContext directly without it, the uuid fallback fires."""

import asyncio
import uuid

import pytest

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
    async def evaluate(self, request, context):
        return PolicyDecision(
            kind=PolicyDecisionKind.REQUIRE_APPROVAL, rule_id="t", reason="x"
        )


class _Store:
    """ApprovalStore stub -- present so the resume gate has a store to query,
    though the single-path executor no longer persists on check()."""

    async def create(self, request):
        return request

    async def get(self, approval_id):
        return None

    async def approve(self, approval_id, *, expected_version, resolved_by):
        raise NotImplementedError

    async def reject(self, approval_id, *, expected_version, resolved_by, reason=None):
        raise NotImplementedError

    async def list_pending(self, run_id):
        return ()

    async def list_for_run(self, run_id):
        return ()


def test_tool_context_carries_tool_call_id_field_with_default_none():
    assert ToolContext(run_id="r", session_id="s").tool_call_id is None


def test_executor_uses_context_tool_call_id_when_present():
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)), approval_store=_Store()
    )

    async def _run():
        await executor.check(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r", session_id="s", tool_call_id="tcid-XYZ"),
        )

    with pytest.raises(RunPaused) as exc_info:
        asyncio.run(_run())
    assert exc_info.value.tool_call_id == "tcid-XYZ"


def test_executor_falls_back_to_uuid_when_tool_call_id_missing():
    executor = ToolExecutor(
        policy=PolicyEngine(rules=(_Require(),)), approval_store=_Store()
    )

    async def _run():
        await executor.check(
            ToolRequest(tool_name="t", arguments={}),
            ToolContext(run_id="r", session_id="s"),
        )

    with pytest.raises(RunPaused) as exc_info:
        asyncio.run(_run())
    # parses as a uuid (the fallback path)
    uuid.UUID(exc_info.value.tool_call_id)
