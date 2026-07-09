#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime capability wiring (spec §9/§17.9): ProviderBundle + options types,
provider/expanded-param mixing rejection, custom providers feed the assembler,
and async-context close releases MCP connections."""

import pytest

from linktools.ai.capability import CapabilityRuntimeOptions, CapabilityToolExposurePolicy
from linktools.ai.providers import ProviderBundle
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage


def _runtime(tmp_path, **kw):
    from linktools.ai.model.router import ModelRouter
    return Runtime.build(storage=FileStorage(root=tmp_path),
                         model_router=ModelRouter(), **kw)


def test_provider_bundle_defaults_and_empty():
    assert ProviderBundle().is_empty()
    b = ProviderBundle(skills=object())
    assert not b.is_empty()
    assert b.skills is not None


def test_capability_runtime_options_defaults():
    o = CapabilityRuntimeOptions()
    assert isinstance(o.tool_exposure, CapabilityToolExposurePolicy)
    assert o.allow_mcp_wildcard is False
    assert o.session_window_policy is None


def test_build_rejects_mixing_providers_and_expanded(tmp_path):
    bundle = ProviderBundle(skills=object())
    with pytest.raises(ValueError, match="not both"):
        _runtime(tmp_path, providers=bundle, skills=object())


def test_build_accepts_providers_bundle(tmp_path):
    bundle = ProviderBundle()  # empty bundle is fine
    rt = _runtime(tmp_path, providers=bundle)
    assert rt is not None
    # No mcp provider -> no connection manager.
    assert rt._mcp_connection_manager is None


def test_build_accepts_expanded_params(tmp_path):
    rt = _runtime(tmp_path, skills=object())
    assert rt is not None


@pytest.mark.asyncio
async def test_assemble_empty_without_providers(tmp_path):
    from linktools.ai.agent.spec import AgentSpec, PromptSpec
    from linktools.ai.model.policy import ModelPolicy
    rt = _runtime(tmp_path)
    spec = AgentSpec(id="a", name="a", model=ModelPolicy(primary="m"),
                     instructions=PromptSpec(instructions="hi"))
    bundle = await rt.assemble(spec, execution=None)
    assert bundle.toolsets == ()


@pytest.mark.asyncio
async def test_package_resource_ref_resolves_through_runtime(tmp_path):
    # Agent #2 defect #1: package-resource / package-entrypoint refs must resolve
    # (PackageProvider registered under all three kinds).
    from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.package.provider import DirectoryPackageResourceProvider
    from linktools.ai.package.resolver import DirectoryEntrypointResolver

    root = tmp_path / "skill-creator"
    (root / "agents").mkdir(parents=True)
    (root / "SKILL.md").write_text("# s", encoding="utf-8")
    (root / "agents" / "grader.md").write_text(
        "---\nname: grader\nmodel:\n  primary: gpt-4o\n---\nGrade.\n", encoding="utf-8")
    rp = DirectoryPackageResourceProvider({"skill-creator": root})
    er = DirectoryEntrypointResolver({"skill-creator": root})
    # entrypoints is bundle-only (spec §9.1 has no expanded entrypoints param).
    rt = Runtime.build(storage=FileStorage(root=tmp_path),
                       providers=ProviderBundle(package_resources=rp, entrypoints=er))

    for kind in ("package-resource", "package-entrypoint"):
        spec = AgentSpec(
            id="a", name="a", model=ModelPolicy(primary="m"),
            instructions=PromptSpec(instructions="hi"),
            tools=(ToolRef(name="skill-creator", kind=kind),),
        )
        bundle = await rt.assemble(spec, execution=None)
        assert len(bundle.toolsets) == 1, f"{kind} ref did not resolve"


@pytest.mark.asyncio
async def test_allow_mcp_wildcard_build_param_is_honored(tmp_path):
    # Agent #2 defect #2: Runtime.build(allow_mcp_wildcard=True) must enable mcp:*.
    from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
    from linktools.ai.errors import CapabilityResolutionError
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.registry.mcp import MCPServerSpec

    class _McpSrc:
        async def list_ids(self):
            return ("risk",)
        async def get(self, sid):
            return MCPServerSpec(id=sid, name=sid, transport="stdio",
                                 command_or_url="python -m r", command=("python", "-m", "r"))

    spec = AgentSpec(id="a", name="a", model=ModelPolicy(primary="m"),
                     instructions=PromptSpec(instructions="hi"),
                     tools=(ToolRef(name="*", kind="mcp"),))

    # Off: the gate denies mcp:* at assemble time (before any connection).
    rt_off = Runtime.build(storage=FileStorage(root=tmp_path), mcp_servers=_McpSrc())
    assert rt_off._options.allow_mcp_wildcard is False
    with pytest.raises(CapabilityResolutionError, match="allow_mcp_wildcard"):
        await rt_off.assemble(spec, execution=None)

    # On: the build flag is folded into options (live connection is exercised
    # separately via MCPProvider + a fake manager in test_mcp_provider.py).
    rt_on = Runtime.build(storage=FileStorage(root=tmp_path / "on"),
                          mcp_servers=_McpSrc(), allow_mcp_wildcard=True)
    assert rt_on._options.allow_mcp_wildcard is True


@pytest.mark.asyncio
async def test_custom_skill_provider_wires_assembler(tmp_path):
    from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
    from linktools.ai.model.policy import ModelPolicy

    class _SkillSpec:
        def __init__(self, i, n, d, instr):
            self.id = i; self.name = n; self.description = d; self.instructions = instr
            self.metadata = {}

    class _SkillSrc:
        async def list_ids(self):
            return ("sql",)
        async def get(self, sid):
            return _SkillSpec("sql", "sql", "SQL analysis", "FULL SQL")

    rt = Runtime.build(storage=FileStorage(root=tmp_path), skills=_SkillSrc())
    spec = AgentSpec(
        id="a", name="a", model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="*", kind="skill"),),
    )
    bundle = await rt.assemble(spec, execution=None)
    # Skill catalog prompt injected + list_skills/read_skill exposed.
    assert "sql" in bundle.prompt_sections.get("skills", "")
    names = set(bundle.toolsets[0].tools.keys())
    assert {"list_skills", "read_skills"} <= names or {"list_skills", "read_skill"} <= names


@pytest.mark.asyncio
async def test_runtime_async_context_manager_closes_mcp(tmp_path):
    # An MCP provider wires a connection manager that aclose() must release.
    class _McpSrc:
        async def list_ids(self):
            return ()
        async def get(self, sid):
            raise KeyError(sid)

    rt = Runtime.build(storage=FileStorage(root=tmp_path), mcp_servers=_McpSrc())
    assert rt._mcp_connection_manager is not None
    closed = {"v": False}

    async def _close():
        closed["v"] = True
    rt._mcp_connection_manager.close = _close

    async with rt:
        pass
    assert closed["v"] is True
    assert rt._mcp_connection_manager is None  # aclose clears it (idempotent)


@pytest.mark.asyncio
async def test_provider_bundle_from_resources_builds_registries(tmp_path):
    # spec §17.5: ProviderBundle.from_resources constructs the default
    # Spec-backed registries from a resource store + prefixes.
    from linktools.ai.providers import ProviderBundle, ProviderPrefixes

    class _Store:
        def __init__(self, files):
            self._files = files

        async def get(self, path):
            from types import SimpleNamespace
            if path not in self._files:
                return None
            return SimpleNamespace(content=self._files[path])

        async def revision(self):
            return 1

    store = _Store({
        "specs/agents/writer.md": "---\nname: writer\nmodel:\n  primary: gpt-4o\n---\nhi\n",
        "specs/skills/sql.md": "---\nname: sql\n---\nx\n",
    })
    bundle = ProviderBundle.from_resources(store, prefixes=ProviderPrefixes())
    assert bundle.agents is not None
    assert bundle.skills is not None
    assert bundle.mcp_servers is not None
    assert bundle.tool_policies is not None


@pytest.mark.asyncio
async def test_runtime_resolve_agent_and_swarm_via_providers(tmp_path):
    # spec §9.4 #4: bundle.agents / bundle.swarms are consumed by by-id lookups.
    from linktools.ai.agent.spec import AgentSpec, PromptSpec
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.providers import ProviderBundle

    class _AgentSrc:
        async def list_ids(self):
            return ("reviewer",)
        async def get(self, aid):
            if aid != "reviewer":
                raise KeyError(aid)
            return AgentSpec(id=aid, name=aid, model=ModelPolicy(primary="m"),
                             instructions=PromptSpec(instructions="hi"))

    class _SwarmSrc:
        async def list_ids(self):
            return ()
        async def get(self, sid):
            raise KeyError(sid)

    rt = Runtime.build(storage=FileStorage(root=tmp_path),
                       providers=ProviderBundle(agents=_AgentSrc(), swarms=_SwarmSrc()))
    agent = await rt.resolve_agent("reviewer")
    assert agent.id == "reviewer"
    from linktools.ai.errors import SwarmError
    # No swarm provider -> resolve_swarm raises SwarmError (configured here but empty).
    rt2 = Runtime.build(storage=FileStorage(root=tmp_path / "x"))
    with pytest.raises(SwarmError):
        await rt2.resolve_swarm("anything")
