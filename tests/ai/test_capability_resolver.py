#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CapabilityAssembler (contract): resolves AgentSpec.tools into one merged
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


def _contrib_names(bundle):
    """Descriptor names across all contributions, from the per-tool ``tools``
    form and/or the legacy ``descriptors`` tuple (the assembler normalizes
    introspectable contributions to the tools form)."""
    names = set()
    for c in bundle.tool_contributions:
        for md in getattr(c, "tools", ()):
            names.add(md.descriptor.name)
        for d in getattr(c, "descriptors", ()):
            names.add(d.name)
    return names


def _spec(tools):
    return AgentSpec(
        id="a1", name="a1", model=ModelPolicy(primary="gpt-4"),
        instructions=PromptSpec(instructions="hi"), tools=tools,
    )


@pytest.mark.asyncio
async def test_assemble_builtin_file_only(tmp_path):
    """Default policy (expose_execution_tools=False) exposes read-only file
    tools; write_file/batch_files/apply_patch stay hidden until execution
    tools are explicitly allowed."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    bundle = await asm.assemble(_spec((ToolRef(name="file"),)), _ctx(backend))
    assert isinstance(bundle, CapabilityBundle)
    names = _contrib_names(bundle)
    assert "read_file" in names and "bash" not in names and "write_file" not in names


@pytest.mark.asyncio
async def test_assemble_builtin_file_execution_tools_allowed_exposes_writes(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    policy = CapabilityToolExposurePolicy(expose_execution_tools=True)
    bundle = await asm.assemble(_spec((ToolRef(name="file"),)), _ctx(backend, policy=policy))
    names = _contrib_names(bundle)
    assert {"read_file", "write_file", "batch_files", "apply_patch"} <= names


@pytest.mark.asyncio
async def test_assemble_kindname_string_resolves(tmp_path):
    """builtin:terminal is mutating -- only reachable when execution tools
    are explicitly allowed by policy (contract: explicit builtin:terminal
    works when allowed, not unconditionally)."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    policy = CapabilityToolExposurePolicy(expose_execution_tools=True)
    # kind:name parsed form
    spec = _spec((ToolRef(name="terminal", kind="builtin"),))
    bundle = await asm.assemble(spec, _ctx(backend, policy=policy))
    assert _contrib_names(bundle) == {"bash"}


@pytest.mark.asyncio
async def test_assemble_kindname_terminal_hidden_by_default(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    spec = _spec((ToolRef(name="terminal", kind="builtin"),))
    bundle = await asm.assemble(spec, _ctx(backend))
    assert bundle.toolsets == () and bundle.tool_contributions == ()


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
async def test_unregistered_kind_raises_resolution_error_no_hardcoded_allowlist(tmp_path):
    """contract: validity is entirely provider-registration-driven -- there
    is no separate hardcoded kind allowlist. A completely made-up kind like
    "bogus" fails exactly like a recognized-but-unwired kind (e.g. "skill"
    with no SkillProvider registered): CapabilityResolutionError, not a
    distinct InvalidSpecError."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    with pytest.raises(CapabilityResolutionError, match="no capability provider"):
        await asm.assemble(_spec((ToolRef(name="x", kind="bogus"),)), _ctx(backend))


@pytest.mark.asyncio
async def test_register_rejects_duplicate_kind():
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    with pytest.raises(CapabilityConflictError, match="already registered"):
        asm.register(BuiltinProvider())


@pytest.mark.asyncio
async def test_replace_intentionally_overrides_existing_kind():
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    original = asm.providers["builtin"]
    replacement = BuiltinProvider()
    asm.replace(replacement)
    assert asm.providers["builtin"] is replacement
    assert asm.providers["builtin"] is not original


@pytest.mark.asyncio
async def test_register_accepts_a_new_kind():
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    asm.register(_PromptProvider())
    assert "skill" in asm.providers


def test_provider_kinds_reads_supported_kinds():
    """A provider declaring supported_kinds is recognized for every kind it
    owns -- no manual alias registration needed (the contract multi-kind model)."""
    from linktools.ai.capability.provider import provider_kinds
    from linktools.ai.package.capability_provider import PackageProvider

    kinds = provider_kinds(PackageProvider())
    assert kinds == frozenset({"package", "package-resource", "package-entrypoint"})


@pytest.mark.asyncio
async def test_register_multi_kind_provider_under_all_kinds():
    class _Multi:
        kind = "alpha"
        supported_kinds = frozenset({"alpha", "beta", "gamma"})

        async def resolve(self, ref, context):
            return CapabilityBundle()

    asm = CapabilityAssembler({})
    asm.register(_Multi())
    assert {"alpha", "beta", "gamma"} <= set(asm.providers)


@pytest.mark.asyncio
async def test_descriptor_without_matching_handler_fails_at_assembly(tmp_path):
    """A descriptor that names a tool not present in an introspectable toolset
    is rejected at assembly (not left to fail-closed at call time)."""
    from linktools.ai.security.descriptor import ToolDescriptor
    from linktools.ai.tool.contribution import ToolContribution

    class _Mismatch:
        kind = "mcp"

        async def resolve(self, ref, context):
            from pydantic_ai.toolsets import FunctionToolset
            ts = FunctionToolset()

            async def real_tool(): return "ok"
            ts.add_function(real_tool)
            # Declare a descriptor for a tool that does NOT exist in the toolset.
            return CapabilityBundle(
                toolsets=(ts,),
                tool_contributions=(ToolContribution(
                    toolset=ts,
                    descriptors=(ToolDescriptor(
                        name="ghost_tool", source="mcp", category="mcp-write",
                        risk="high", mutating=True),),
                ),),
            )

    asm = CapabilityAssembler({"mcp": _Mismatch()})
    with pytest.raises(CapabilityResolutionError, match="ManagedToolDefinition"):
        await asm.assemble(_spec((ToolRef(name="x", kind="mcp"),)), _ctx(None))


@pytest.mark.asyncio
async def test_introspectable_contribution_populates_per_tool_definitions(tmp_path):
    """An introspectable toolset contribution is upgraded to the per-tool
    ManagedToolDefinition form at assembly -- each tool carries its own explicit
    descriptor + extractable handler (the contract per-tool model, exercised)."""
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    bundle = await asm.assemble(_spec((ToolRef(name="file-read"),)), _ctx(backend))
    contrib = bundle.tool_contributions[0]
    assert contrib.tools, "per-tool ManagedToolDefinitions must be populated"
    # Each definition has a real handler resolved from the toolset.
    for md in contrib.tools:
        assert md.descriptor.name in {"list_dir", "read_file"}
        assert callable(md.handler)


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
        from linktools.ai.security.descriptor import ToolDescriptor
        from linktools.ai.tool.legacy import LegacyToolsetAdapter
        ts = build_builtin_toolset(
            BuiltinToolContext(backend=context.execution, enabled_tools={"file"}))
        descriptors = tuple(ToolDescriptor(
            name=name, source="mcp", category="custom", risk="high", mutating=True,
            capability_kind="mcp", capability_name=ref.name,
        ) for name in ("list_dir", "read_file", "write_file", "batch_files", "apply_patch"))
        return CapabilityBundle(toolsets=(LegacyToolsetAdapter(ts, descriptors),))


@pytest.mark.asyncio
async def test_cross_capability_tool_name_conflict_detected(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    asm = CapabilityAssembler({"builtin": BuiltinProvider(), "mcp": _CollidingProvider()})
    # Allow execution tools so both providers' full tool sets (including the
    # auto-generated conservative mcp descriptors) are exposed and can collide.
    policy = CapabilityToolExposurePolicy(expose_execution_tools=True)
    # builtin:file and mcp:x both emit the file tools -> conflict, no silent overwrite.
    with pytest.raises(CapabilityConflictError, match="produced by both"):
        await asm.assemble(
            _spec((ToolRef(name="file"), ToolRef(name="x", kind="mcp"))), _ctx(backend, policy=policy),
        )


@pytest.mark.asyncio
async def test_total_tool_cap_enforced(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    policy = CapabilityToolExposurePolicy(max_tools_total=2, expose_execution_tools=True)
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    # builtin:file alone exposes 5 tools -> exceeds max_tools_total=2.
    with pytest.raises(CapabilityConflictError, match="max_tools_total"):
        await asm.assemble(_spec((ToolRef(name="file"),)), _ctx(backend, policy=policy))


@pytest.mark.asyncio
async def test_per_capability_cap_enforced(tmp_path):
    backend = LocalExecutionBackend(runtime_dir=str(tmp_path))
    policy = CapabilityToolExposurePolicy(max_tools_per_capability=2, expose_execution_tools=True)
    asm = CapabilityAssembler({"builtin": BuiltinProvider()})
    with pytest.raises(CapabilityConflictError, match="max_tools_per_capability"):
        await asm.assemble(_spec((ToolRef(name="file"),)), _ctx(backend, policy=policy))


class _PromptProvider:
    kind = "skill"

    async def resolve(self, ref, context):
        return CapabilityBundle(prompt_sections={"skills": f"section-{ref.name}"})


class _PipelineProvider:
    """Declares a SecurityPipeline in its bundle -- a stable CapabilityBundle
    field that must be collected, not silently dropped."""
    kind = "skill"

    def __init__(self):
        self.pipeline = object()

    async def resolve(self, ref, context):
        return CapabilityBundle(pipelines=(self.pipeline,))


@pytest.mark.asyncio
async def test_declared_pipelines_are_merged_into_bundle(tmp_path):
    """A capability declaring pipelines must have them collected into the
    assembled bundle (a stable field never silently dropped)."""
    asm = CapabilityAssembler({"skill": _PipelineProvider()})
    spec = _spec((ToolRef(name="a", kind="skill"), ToolRef(name="b", kind="skill")))
    bundle = await asm.assemble(spec, _ctx(None))
    assert len(bundle.pipelines) == 2


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
