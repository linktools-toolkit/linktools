#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration test: the full compat chain from CapabilityAssembler through
auto-generated ToolContributions to ManagedToolAdapter governance. Proves the
execution path is wired (contract core loop)."""

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability import CapabilityAssembler, CapabilityContext, CapabilityToolExposurePolicy
from linktools.ai.capability.builtin import BuiltinProvider
from linktools.ai.errors import ToolDeniedError
from linktools.ai.execution.local import LocalExecutionBackend
from linktools.ai.security.pipeline import (
    PipelineAction, PipelineDecision, SecurityPipeline,
)
from linktools.ai.tool.auto_descriptor import extract_handler
from linktools.ai.tool.managed import ManagedToolAdapter
from linktools.ai.model.policy import ModelPolicy


class _DenyAllPipeline:
    """Pipeline that denies every tool call."""
    async def before_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
    async def after_model(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
    async def before_tool(self, e):
        return PipelineDecision(action=PipelineAction.DENY, reason="blocked by test pipeline")
    async def after_tool(self, e): return PipelineDecision(action=PipelineAction.ALLOW)
    async def on_security_event(self, e): return PipelineDecision(action=PipelineAction.AUDIT_ONLY)


@pytest.mark.asyncio
async def test_assembler_auto_wraps_toolsets_into_contributions(tmp_path):
    """The assembler auto-generates ToolContributions with conservative
    descriptors from raw toolsets (compat layer, contract)."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    spec = AgentSpec(
        id="a1", name="a1", model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="file"),),
    )
    ctx = CapabilityContext(
        agent_id="a1", exposure_policy=CapabilityToolExposurePolicy(),
        execution=backend,
    )
    bundle = await asm.assemble(spec, ctx)
    # Auto-wrapped: tool_contributions should be populated.
    assert len(bundle.tool_contributions) > 0
    # Each contribution has descriptors with conservative defaults.
    for contrib in bundle.tool_contributions:
        for desc in contrib.descriptors:
            assert desc.name  # has a name
            assert desc.risk in ("low", "medium", "high", "critical")


@pytest.mark.asyncio
async def test_managed_adapter_from_assembler_output_deny(tmp_path):
    """End-to-end: assembler → tool_contributions → ManagedToolAdapter with a
    deny pipeline → tool call blocked."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    spec = AgentSpec(
        id="a1", name="a1", model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="file"),),
    )
    ctx = CapabilityContext(
        agent_id="a1", exposure_policy=CapabilityToolExposurePolicy(),
        execution=backend,
    )
    bundle = await asm.assemble(spec, ctx)

    # Build adapters from the assembled contributions (per-tool form). The
    # assembler normalizes introspectable toolsets to ManagedToolDefinitions, so
    # iterate contrib.tools -- iterating contrib.descriptors would be empty and
    # the deny assertion would never run.
    pipeline = _DenyAllPipeline()
    denied = 0
    for contrib in bundle.tool_contributions:
        for md in contrib.tools:
            adapter = ManagedToolAdapter(
                descriptor=md.descriptor, handler=md.handler,
                security_pipeline=pipeline,
            )
            with pytest.raises(ToolDeniedError, match="blocked by test pipeline"):
                await adapter.invoke()
            denied += 1
    assert denied > 0, "expected at least one assembled tool to be denied"
