#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture invariants for the simplified linktools.ai.

Each invariant is asserted against the TARGET architecture. Invariants that
the current code does not yet satisfy are marked ``xfail(strict=True)`` with a
reference to the phase that will make them pass -- strict means the moment the
invariant becomes true the test turns into an XPASS failure, forcing the
implementing phase to drop the marker. This keeps the gap visible instead of
hiding it behind a passing test.
"""

import asyncio
import dataclasses

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage
from linktools.ai.tool.models import ManagedToolDefinition, ToolContribution


def _model_fn(messages, info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='{"response": {"message": "ok"}}')])


def _registry():
    from linktools.ai.model.registry import ModelRegistry

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_model_fn))
    return registry


def _router():
    from linktools.ai.model.router import ModelRouter

    return ModelRouter(registry=_registry())


# --- Runtime has no `assemble` --------------------------------------------------
def test_invariant_runtime_has_no_assemble():
    assert not hasattr(Runtime, "assemble")


# --- Runtime exposes no public capability_assembler -----------------------------
def test_invariant_runtime_has_no_public_capability_assembler(tmp_path):
    runtime = Runtime.build(storage=FileStorage(root=tmp_path), model_router=_router())
    assert not hasattr(runtime, "capability_assembler")
    assert not hasattr(runtime, "assembler")


# --- CapabilityBundle carries no raw toolset ------------------------------------
def test_invariant_capability_bundle_has_no_raw_toolset():
    field_names = {f.name for f in dataclasses.fields(CapabilityBundle)}
    assert "toolsets" not in field_names


# --- ToolContribution carries only `tools` --------------------------------------
def test_invariant_tool_contribution_has_only_tools_field():
    field_names = [f.name for f in dataclasses.fields(ToolContribution)]
    assert field_names == ["tools"]


# --- AgentRunner requires an Assembler when tools are needed --------------------
def test_invariant_runner_requires_assembler_for_declared_tools(tmp_path):
    from linktools.ai.errors import RuntimeInitializationError

    runtime = Runtime.build(storage=FileStorage(root=tmp_path), model_router=_router())
    spec = AgentSpec(
        id="needs-tools",
        name="needs-tools",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(kind="builtin", name="file"),),
    )
    with pytest.raises(RuntimeInitializationError):
        asyncio.run(runtime.run(spec, "hi"))


def test_invariant_runner_empty_tools_does_not_require_assembler(tmp_path):
    runtime = Runtime.build(storage=FileStorage(root=tmp_path), model_router=_router())
    spec = AgentSpec(
        id="no-tools",
        name="no-tools",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(),
    )
    result = asyncio.run(runtime.run(spec, "hi"))
    assert result is not None


# --- MCPProvider returns ManagedToolDefinition ----------------------------------
def test_invariant_mcp_provider_returns_managed_tool_definitions():
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
    from linktools.ai.capability.provider import CapabilityContext
    from linktools.ai.capability.models import CapabilityRef
    from linktools.ai.mcp.provider import (
        MCPDiscoveryResult,
        MCPProvider,
        MCPToolInfo,
    )
    from linktools.ai.registry.mcp import MCPServerSpec

    class _FakeSpecProvider:
        async def list_ids(self):
            return ("search",)

        async def get(self, server_id):
            return MCPServerSpec(
                id="search", name="search", transport="stdio", command=("demo",)
            )

    class _FakeConnMgr:
        async def list_tools_result(self, spec):
            return MCPDiscoveryResult(
                tools=(MCPToolInfo(name="query"),),
                verified=True,
                connection_ref=object(),
            )

        async def call_tool(self, *, connection_ref, tool_name, arguments):
            return {"ok": True}

    provider = MCPProvider(
        mcp_provider=_FakeSpecProvider(), connection_manager=_FakeConnMgr()
    )
    context = CapabilityContext(
        agent_id="a", exposure_policy=CapabilityToolExposurePolicy()
    )
    bundle = asyncio.run(
        provider.resolve(CapabilityRef(kind="mcp", name="search"), context)
    )
    assert bundle.tool_contributions
    contrib = bundle.tool_contributions[0]
    assert contrib.tools
    assert isinstance(contrib.tools[0], ManagedToolDefinition)


# --- Runtime.inspect returns no handler ----------------------------------------
def test_invariant_inspect_returns_no_handler(tmp_path):
    runtime = Runtime.build(storage=FileStorage(root=tmp_path), model_router=_router())
    spec = AgentSpec(
        id="a",
        name="a",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
        tools=(),
    )
    inspection = asyncio.run(runtime.inspect(spec))
    fields = {f.name for f in dataclasses.fields(inspection)}
    assert "handler" not in fields
    assert not hasattr(inspection, "handler")


# --- ManagedToolAdapter delegates to ToolExecutor.execute ----------------------
def test_invariant_managed_adapter_delegates_to_executor_execute():
    from linktools.ai.tool.models import ToolDescriptor
    from linktools.ai.tool.managed import ManagedToolAdapter
    from linktools.ai.tool.policy import ResolvedToolPolicy

    class _FakeExecutor:
        def __init__(self):
            self.execute_called = False

        async def is_approved(self, run_id, call_id):
            return False

        async def execute(self, request, context, handler, **kwargs):
            self.execute_called = True
            return {"executed": True}

    descriptor = ToolDescriptor(
        name="t", source="test", category="misc", risk="low", mutating=False
    )
    executor = _FakeExecutor()
    adapter = ManagedToolAdapter(
        descriptor=descriptor,
        handler=lambda **kw: None,
        tool_executor=executor,
        baseline_policy=ResolvedToolPolicy(),
    )
    result = asyncio.run(adapter.invoke())
    assert executor.execute_called is True
    assert result == {"executed": True}
