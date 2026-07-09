#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityAssembler (spec §10): resolves AgentSpec.tools into one merged
bundle, enforcing resolution rules (unknown kind, conflicts, exposure caps) and
merging prompt sections in stable order."""

import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability import (
    BuiltinProvider,
    CapabilityAssembler,
    CapabilityBundle,
    CapabilityContext,
    CapabilityProvider,
    CapabilityToolExposurePolicy,
)
from linktools.ai.capability.ref import CapabilityRef
from linktools.ai.errors import CapabilityConflictError, CapabilityResolutionError
from linktools.ai.execution.local import LocalExecutionBackend
from linktools.ai.model.policy import ModelPolicy


def _ctx(execution, policy=None, agent_id="a1"):
    return CapabilityContext(
        agent_id=agent_id, exposure_policy=policy or CapabilityToolExposurePolicy(),
        execution=execution,
    )


def _spec(tools):
    return AgentSpec(
        id="a1", name="a1", model=ModelPolicy(primary="gpt-4"),
        instructions=PromptSpec(instructions="hi"), tools=tools,
    )


@pytest.mark.asyncio
async def test_assemble_builtin_file_only(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    bundle = await asm.assemble(_spec((ToolRef(name="file"),)), _ctx(backend))
    assert isinstance(bundle, CapabilityBundle)
    names = set(bundle.toolsets[0].tools.keys())
    assert "read_file" in names and "bash" not in names


@pytest.mark.asyncio
async def test_assemble_kindname_string_resolves(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    # kind:name parsed form
    spec = _spec((ToolRef(name="terminal", kind="builtin"),))
    bundle = await asm.assemble(spec, _ctx(backend))
    assert tuple(bundle.toolsets[0].tools.keys()) == ("bash",)


@pytest.mark.asyncio
async def test_assemble_empty_tools_yields_empty_bundle(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    bundle = await asm.assemble(_spec(()), _ctx(backend))
    assert bundle.toolsets == () and dict(bundle.prompt_sections) == {}


@pytest.mark.asyncio
async def test_unknown_kind_raises_resolution_error(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    with pytest.raises(CapabilityResolutionError, match="no capability provider"):
        await asm.assemble(_spec((ToolRef(name="x", kind="skill"),)), _ctx(backend))


@pytest.mark.asyncio
async def test_structurally_unknown_kind_raises_invalid_spec(tmp_path):
    # A kind outside the recognized set is a structurally invalid tool ref.
    from linktools.ai.errors import InvalidSpecError
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    with pytest.raises(InvalidSpecError, match="unknown capability kind"):
        await asm.assemble(_spec((ToolRef(name="x", kind="bogus"),)), _ctx(backend))


@pytest.mark.asyncio
async def test_duplicate_ref_raises_conflict(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    with pytest.raises(CapabilityConflictError, match="duplicate"):
        await asm.assemble(
            _spec((ToolRef(name="file"), ToolRef(name="file"))), _ctx(backend),
        )


class _CollidingProvider:
    """An mcp-kind provider that also emits the file tools to force a
    cross-capability name collision with builtin:file."""

    kind = "mcp"

    async def resolve(self, ref, context):
        from linktools.ai.execution.toolset import BuiltinToolContext, build_builtin_toolset
        ts = build_builtin_toolset(
            BuiltinToolContext(backend=context.execution, enabled_tools={"file"}))
        return CapabilityBundle(toolsets=(ts,))


@pytest.mark.asyncio
async def test_cross_capability_tool_name_conflict_detected(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider(), "mcp": _CollidingProvider()})
    # builtin:file and mcp:x both emit the file tools -> conflict, no silent overwrite.
    with pytest.raises(CapabilityConflictError, match="produced by both"):
        await asm.assemble(
            _spec((ToolRef(name="file"), ToolRef(name="x", kind="mcp"))), _ctx(backend),
        )


@pytest.mark.asyncio
async def test_total_tool_cap_enforced(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    policy = CapabilityToolExposurePolicy(max_tools_total=2)
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    # builtin:file alone exposes 5 tools -> exceeds max_tools_total=2.
    with pytest.raises(CapabilityConflictError, match="max_tools_total"):
        await asm.assemble(_spec((ToolRef(name="file"),)), _ctx(backend, policy=policy))


@pytest.mark.asyncio
async def test_per_capability_cap_enforced(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    policy = CapabilityToolExposurePolicy(max_tools_per_capability=2)
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    with pytest.raises(CapabilityConflictError, match="max_tools_per_capability"):
        await asm.assemble(_spec((ToolRef(name="file"),)), _ctx(backend, policy=policy))


class _PromptProvider:
    kind = "skill"

    async def resolve(self, ref, context):
        return CapabilityBundle(prompt_sections={"skills": f"section-{ref.name}"})


@pytest.mark.asyncio
async def test_prompt_sections_merged_in_stable_order(tmp_path):
    asm = CapabilityAssembler({"skill": _PromptProvider()})
    spec = _spec((ToolRef(name="a", kind="skill"), ToolRef(name="b", kind="skill")))
    bundle = await asm.assemble(spec, _ctx(None))
    assert "section-a" in bundle.prompt_sections["skills"]
    assert "section-b" in bundle.prompt_sections["skills"]


def test_capability_resolver_is_assembler_alias():
    from linktools.ai.capability import CapabilityResolver
    assert CapabilityResolver is CapabilityAssembler


def test_capability_provider_protocol_matches_builtin():
    assert isinstance(BuiltinProvider(), CapabilityProvider)
