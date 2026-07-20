#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ManagedToolsetWrapper governance parity: a builtin-shaped toolset and an
MCP-shaped toolset must go through IDENTICAL governance when wrapped -- same
descriptor lookup, same security pipeline, same audit trail, same fail-closed
behavior for an undeclared tool name. Nothing in ManagedToolsetWrapper.call_tool
branches on the wrapped toolset's kind, so this pins that invariant down with a
shared spy instead of relying on code inspection alone."""

import pytest

from linktools.ai.governance.security.pipeline import (
    PipelineAction,
    PipelineDecision,
    ToolInvocationEvent,
    ToolResultEvent,
)
from linktools.ai.errors import ToolDeniedError
from linktools.ai.tool.models import ToolDescriptor
from linktools.ai.tool.pydantic import ManagedToolsetWrapper


class _Executor:
    """Minimal GovernedToolInvoker stand-in: runs the handler, then the adapter's
    result_processor (mirrors the real GovernedToolInvoker.execute contract)."""

    async def _is_approved_binding(self, run_id, call_id, *, binding):
        return False

    async def execute(self, request, context, handler, **kwargs):
        result = await handler(**request.arguments)
        processor = kwargs.get("result_processor")
        return result if processor is None else await processor(result)


class _SpyPipeline:
    """Records every before/after decision it is asked to make, tagged with
    which wrapped toolset the call came from (via the event's tool_name)."""

    def __init__(self) -> None:
        self.before_calls: "list[ToolInvocationEvent]" = []
        self.after_calls: "list[ToolResultEvent]" = []

    async def before_tool(self, event: ToolInvocationEvent) -> PipelineDecision:
        self.before_calls.append(event)
        return PipelineDecision(action=PipelineAction.ALLOW)

    async def after_tool(self, event: ToolResultEvent) -> PipelineDecision:
        self.after_calls.append(event)
        return PipelineDecision(action=PipelineAction.ALLOW)


class _FakeToolset:
    """Shaped like whatever AbstractToolset ManagedToolsetWrapper wraps --
    builtin FunctionToolset and an MCPToolset both only need call_tool for
    this seam. ``kind`` only labels which fake this is for the assertions."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.calls: "list[tuple]" = []

    async def call_tool(self, name, tool_args, ctx, tool):
        self.calls.append((name, dict(tool_args)))
        return {"echo": tool_args.get("x"), "from": self.kind}


def _descriptor(*, source: str, name: str = "shared_tool") -> ToolDescriptor:
    return ToolDescriptor(
        name=name, source=source, category="c", risk="low", mutating=False
    )


def _make_wrapper(wrapped: _FakeToolset, *, pipeline: _SpyPipeline) -> ManagedToolsetWrapper:
    # source deliberately differs per wrapped kind (builtin vs mcp) so a
    # regression that branched on descriptor.source, not just on the wrapped
    # toolset's runtime type, would also surface as an assertion failure.
    return ManagedToolsetWrapper(
        wrapped,
        descriptors={"shared_tool": _descriptor(source=wrapped.kind)},
        security_pipeline=pipeline,
        tool_executor=_Executor(),
    )


@pytest.mark.asyncio
async def test_builtin_and_mcp_toolsets_get_identical_governed_result():
    pipeline = _SpyPipeline()
    builtin = _FakeToolset("builtin")
    mcp = _FakeToolset("mcp")
    builtin_wrapper = _make_wrapper(builtin, pipeline=pipeline)
    mcp_wrapper = _make_wrapper(mcp, pipeline=pipeline)

    builtin_result = await builtin_wrapper.call_tool("shared_tool", {"x": 1}, None, None)
    mcp_result = await mcp_wrapper.call_tool("shared_tool", {"x": 1}, None, None)

    # Same governed shape back from the adapter regardless of which toolset
    # kind executed the call -- only the handler's own payload differs.
    assert builtin_result == {"echo": 1, "from": "builtin"}
    assert mcp_result == {"echo": 1, "from": "mcp"}
    # The underlying toolset actually ran (governance didn't short-circuit it).
    assert builtin.calls == [("shared_tool", {"x": 1})]
    assert mcp.calls == [("shared_tool", {"x": 1})]


@pytest.mark.asyncio
async def test_builtin_and_mcp_toolsets_produce_identical_audit_trail():
    pipeline = _SpyPipeline()
    builtin_wrapper = _make_wrapper(_FakeToolset("builtin"), pipeline=pipeline)
    mcp_wrapper = _make_wrapper(_FakeToolset("mcp"), pipeline=pipeline)

    await builtin_wrapper.call_tool("shared_tool", {"x": 1}, None, None)
    await mcp_wrapper.call_tool("shared_tool", {"x": 1}, None, None)

    assert len(pipeline.before_calls) == 2
    assert len(pipeline.after_calls) == 2
    # Same descriptor-driven fields on both before_tool events -- the wrapper
    # never lets the wrapped toolset's kind leak into the governance event.
    first, second = pipeline.before_calls
    assert first.tool_name == second.tool_name == "shared_tool"
    assert first.arguments == second.arguments == {"x": 1}
    assert first.risk == second.risk
    assert first.mutating == second.mutating


@pytest.mark.asyncio
async def test_builtin_and_mcp_toolsets_fail_closed_identically_for_undeclared_tool():
    pipeline = _SpyPipeline()
    builtin_wrapper = _make_wrapper(_FakeToolset("builtin"), pipeline=pipeline)
    mcp_wrapper = _make_wrapper(_FakeToolset("mcp"), pipeline=pipeline)

    with pytest.raises(ToolDeniedError):
        await builtin_wrapper.call_tool("not_registered", {}, None, None)
    with pytest.raises(ToolDeniedError):
        await mcp_wrapper.call_tool("not_registered", {}, None, None)

    # Fails closed BEFORE reaching the pipeline (no descriptor -> no adapter
    # constructed at all) -- identical for both toolset kinds.
    assert pipeline.before_calls == []
