#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime capability wiring (contract/contract): ProviderBundle + options types,
provider/expanded-param mixing rejection, custom providers feed the assembler,
and async-context close releases MCP connections."""

import pytest

from linktools.ai.capability import CapabilityRuntimeOptions
from linktools.ai.providers import ProviderBundle
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage


def _runtime(tmp_path, **kw):
    from linktools.ai.model.router import ModelRouter

    return Runtime.build(
        storage=FileStorage(root=tmp_path), model_router=ModelRouter(), **kw
    )


def test_provider_bundle_defaults_and_empty():
    assert ProviderBundle().is_empty()
    b = ProviderBundle(skills=object())
    assert not b.is_empty()
    assert b.skills is not None


def test_capability_runtime_options_defaults():
    o = CapabilityRuntimeOptions()
    assert o.tool_exposure is None
    assert o.allow_mcp_wildcard is False
    assert o.session_window_policy is None


def test_build_rejects_expanded_provider_params(tmp_path):
    # Expanded provider params (agents/skills/mcp_servers/...) are gone;
    # providers must come via a ProviderBundle. Passing one raises TypeError.
    with pytest.raises(TypeError):
        _runtime(tmp_path, skills=object())


def test_build_accepts_providers_bundle(tmp_path):
    bundle = ProviderBundle()  # empty bundle is fine
    rt = _runtime(tmp_path, providers=bundle)
    assert rt is not None
    # No mcp provider -> no connection manager.
    assert rt._mcp_connection_manager is None


@pytest.mark.asyncio
async def test_assemble_empty_without_providers(tmp_path):
    from linktools.ai.agent.spec import AgentSpec, PromptSpec
    from linktools.ai.model.policy import ModelPolicy

    rt = _runtime(tmp_path)
    spec = AgentSpec(
        id="a",
        name="a",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
    )
    inspection = await rt.inspect(spec, execution=None)
    assert inspection.tools == ()


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
        "---\nname: grader\nmodel:\n  primary: gpt-4o\n---\nGrade.\n", encoding="utf-8"
    )
    rp = DirectoryPackageResourceProvider({"skill-creator": root})
    er = DirectoryEntrypointResolver({"skill-creator": root})
    # entrypoints is bundle-only (contract has no expanded entrypoints param).
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        providers=ProviderBundle(package_resources=rp, entrypoints=er),
    )

    for kind in ("package-resource", "package-entrypoint"):
        spec = AgentSpec(
            id="a",
            name="a",
            model=ModelPolicy(primary="m"),
            instructions=PromptSpec(instructions="hi"),
            tools=(ToolRef(name="skill-creator", kind=kind),),
        )
        inspection = await rt.inspect(spec, execution=None)
        # The ref resolved to a non-empty contribution (tools form for
        # introspectable toolsets). package-entrypoint with no expose_call_tool
        # contributes only a discovery tool; package-resource contributes read tools.
        assert inspection.tools, f"{kind} ref did not resolve"


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
            return MCPServerSpec(
                id=sid, name=sid, transport="stdio", command=("python", "-m", "r")
            )

    spec = AgentSpec(
        id="a",
        name="a",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="*", kind="mcp"),),
    )

    # Off: the gate denies mcp:* at assemble time (before any connection).
    rt_off = Runtime.build(
        storage=FileStorage(root=tmp_path),
        providers=ProviderBundle(mcp_servers=_McpSrc()),
    )
    assert rt_off._options.allow_mcp_wildcard is False
    with pytest.raises(CapabilityResolutionError, match="allow_mcp_wildcard"):
        await rt_off.inspect(spec, execution=None)

    # On: the build flag is folded into options (live connection is exercised
    # separately via MCPProvider + a fake manager in test_mcp_provider.py).
    rt_on = Runtime.build(
        storage=FileStorage(root=tmp_path / "on"),
        providers=ProviderBundle(mcp_servers=_McpSrc()),
        allow_mcp_wildcard=True,
    )
    assert rt_on._options.allow_mcp_wildcard is True


@pytest.mark.asyncio
async def test_mcp_tool_runs_through_runtime_to_connection_manager(tmp_path):
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
    from linktools.ai.mcp.client import MCPConnectionRef
    from linktools.ai.mcp.provider import MCPDiscoveryResult, MCPToolInfo
    from linktools.ai.capability.models import CapabilityRuntimeOptions
    from linktools.ai.capability.exposure import CapabilityToolExposurePolicy
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.registry import ModelRegistry
    from linktools.ai.model.router import ModelRouter
    from linktools.ai.registry.mcp import MCPServerSpec

    class _McpSrc:
        async def list_ids(self):
            return ("srv",)

        async def get(self, sid):
            return MCPServerSpec(
                id=sid, name=sid, transport="stdio", command=("python", "-m", "fake")
            )

    class _Manager:
        def __init__(self):
            self.calls = []
            self.ref = MCPConnectionRef("srv", "fingerprint-a")

        async def list_tools_result(self, spec):
            return MCPDiscoveryResult(
                tools=(
                    MCPToolInfo(
                        "echo",
                        {"type": "object", "properties": {"value": {"type": "string"}}},
                    ),
                ),
                verified=True,
                connection_ref=self.ref,
            )

        async def call_tool(self, *, connection_ref, tool_name, arguments):
            self.calls.append((connection_ref, tool_name, arguments))
            return {"echo": arguments["value"]}

    calls = []

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        if not calls:
            calls.append(True)
            function_tools = getattr(info, "function_tools", ())
            if isinstance(function_tools, dict):
                tool_name = next(iter(function_tools))
            else:
                tool_name = getattr(function_tools[0], "name", "srv.echo")
            return ModelResponse(
                parts=[ToolCallPart(tool_name=tool_name, args={"value": "hello"})]
            )
        return ModelResponse(parts=[TextPart(content="done")])

    registry = ModelRegistry()
    registry.register("m", model=FunctionModel(model_fn))
    manager = _Manager()
    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=registry),
        providers=ProviderBundle(mcp_servers=_McpSrc()),
        mcp_connection_manager=manager,
        options=CapabilityRuntimeOptions(
            tool_exposure=CapabilityToolExposurePolicy(expose_execution_tools=True)
        ),
    )
    spec = AgentSpec(
        id="mcp-e2e",
        name="mcp-e2e",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="srv", kind="mcp"),),
    )
    result = await rt.run(spec, "call echo")
    assert "done" in str(result.output)
    assert manager.calls == [(manager.ref, "echo", {"value": "hello"})]


@pytest.mark.asyncio
async def test_custom_skill_provider_wires_assembler(tmp_path):
    from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
    from linktools.ai.model.policy import ModelPolicy

    class _SkillSpec:
        def __init__(self, i, n, d, instr):
            self.id = i
            self.name = n
            self.description = d
            self.instructions = instr
            self.metadata = {}

    class _SkillSrc:
        async def list_ids(self):
            return ("sql",)

        async def get(self, sid):
            return _SkillSpec("sql", "sql", "SQL analysis", "FULL SQL")

    rt = Runtime.build(
        storage=FileStorage(root=tmp_path), providers=ProviderBundle(skills=_SkillSrc())
    )
    spec = AgentSpec(
        id="a",
        name="a",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(name="*", kind="skill"),),
    )
    inspection = await rt.inspect(spec, execution=None)
    # Skill catalog prompt injected + list_skills/read_skill exposed.
    assert "sql" in inspection.prompt_sections.get("skills", "")
    names = {tool.name for tool in inspection.tools}
    assert {"list_skills", "read_skill"} <= names


@pytest.mark.asyncio
async def test_runtime_async_context_manager_closes_mcp(tmp_path):
    # An MCP provider wires a connection manager that aclose() must release.
    class _McpSrc:
        async def list_ids(self):
            return ()

        async def get(self, sid):
            raise KeyError(sid)

    closed = {"v": False}

    class _Manager:
        async def close(self):
            closed["v"] = True

    rt = Runtime.build(
        storage=FileStorage(root=tmp_path),
        providers=ProviderBundle(mcp_servers=_McpSrc()),
        mcp_connection_manager=_Manager(),
    )
    async with rt:
        pass
    assert closed["v"] is True


@pytest.mark.asyncio
async def test_provider_bundle_from_resources_builds_registries(tmp_path):
    # contract: ProviderBundle.from_resources constructs the default
    # Spec-backed registries from a resource store + prefixes.
    from linktools.ai.providers import ProviderBundle
    from linktools.ai.providers.bundle import ProviderPrefixes

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

    store = _Store(
        {
            "specs/agents/writer.md": "---\nname: writer\nmodel:\n  primary: gpt-4o\n---\nhi\n",
            "specs/skills/sql.md": "---\nname: sql\n---\nx\n",
        }
    )
    bundle = ProviderBundle.from_resources(store, prefixes=ProviderPrefixes())
    assert bundle.agents is not None
    assert bundle.skills is not None
    assert bundle.mcp_servers is not None
    assert bundle.tool_policies is not None


@pytest.mark.asyncio
async def test_runtime_resolve_agent_and_swarm_via_providers(tmp_path):
    # contract #4: bundle.agents / bundle.swarms are consumed by by-id lookups.
    from linktools.ai.agent.spec import AgentSpec, PromptSpec
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.providers import ProviderBundle

    class _AgentSrc:
        async def list_ids(self):
            return ("reviewer",)

        async def get(self, aid):
            if aid != "reviewer":
                raise KeyError(aid)
            return AgentSpec(
                id=aid,
                name=aid,
                model=ModelPolicy(primary="m"),
                instructions=PromptSpec(instructions="hi"),
            )

    class _SwarmSrc:
        async def list_ids(self):
            return ()

        async def get(self, sid):
            raise KeyError(sid)

    bundle = ProviderBundle(agents=_AgentSrc(), swarms=_SwarmSrc())
    Runtime.build(storage=FileStorage(root=tmp_path), providers=bundle)
    # By-id resolution is the caller's responsibility via the bundle directly
    # (Runtime no longer exposes resolve_agent/resolve_swarm).
    agent = await bundle.agents.get("reviewer")
    assert agent.id == "reviewer"
    # No swarm provider configured -> bundle.swarms is None (caller must guard).
    bundle2 = ProviderBundle()
    assert bundle2.swarms is None
    assert bundle2.agents is None
