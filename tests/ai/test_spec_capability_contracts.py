#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declaration, capability contract, and runtime-leaf contracts."""

import asyncio
import hashlib
import json
import re
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai import Tool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

import linktools.ai.agent._compiler as agent_compiler
import linktools.ai.runtime._mcp as mcp_runtime
from linktools.ai.agent import AgentCompiler
from linktools.ai.capability import (
    AssetSkillSource,
    CapabilityContribution,
    CapabilityGroup,
    SkillCapability,
    SkillResource,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
    tool_metadata,
    validate_tool_metadata,
)
from linktools.ai.asset import (
    AssetKey,
    AssetStore,
    AssetVersionRef,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._harness_memory import select_harness_memory_tools
from linktools.ai.runtime._mcp import (
    _MCPModelToolset,
    _MCPBinding,
    _bound_resource_versions,
    prepare_mcp_projections,
)
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    BoundaryToolset,
)
from linktools.ai.runtime.state import (
    RuntimeDomain,
    runtime_domain_uses_object_store,
)
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecAdapter,
    AgentSpecCodec,
    AgentUsageLimits,
    MCPServerSpec,
    MCPServerSpecAdapter,
    MCPServerSpecCodec,
    SkillSpec,
    SkillSpecAdapter,
    SkillSpecCodec,
    canonical_selectors,
    capability_ref_payload,
    mcp_server_selector,
    mcp_tool_selector,
    parse_mcp_tool_selector,
)
from linktools.ai.storage import (
    StorageEntryRevision,
    StorageOverlay,
)
from linktools.ai.workspace import SandboxResourcePath


def _expected_mcp_tool_name(server_id: str, tool_name: str) -> str:
    server_token = canonical_sha256(
        {
            "version": 1,
            "kind": "mcp-server-name",
            "server_id": server_id,
        }
    )[:24]
    tool_token = canonical_sha256(
        {
            "version": 1,
            "kind": "mcp-tool-name",
            "server_id": server_id,
            "tool_name": tool_name,
        }
    )[:24]
    return f"mcp__{server_token}__{tool_token}"


def test_skill_wildcard_allows_preload_outside_explicit_requirements() -> None:
    spec = AgentSpec(
        "agent",
        allow_skills=("*", "required"),
        preload_skills=("preloaded",),
    )
    assert spec.allow_skills == ("*", "required")
    assert spec.preload_skills == ("preloaded",)


def test_skill_markdown_accepts_string_revision_without_exposing_new_fields() -> None:
    adapter = SkillSpecAdapter()
    content = (
        "---\n"
        "name: review\n"
        "description: Review changes.\n"
        "metadata:\n"
        '  linktools-revision: "2"\n'
        "---\n"
        "Review changes.\n"
    )
    spec = adapter.decode_markdown(content.encode())
    assert spec.revision == 2
    assert "linktools-revision" not in spec.metadata
    assert adapter.encode_markdown(spec) == content.encode()


def test_mcp_shared_config_normalizes_supported_transports() -> None:
    servers = MCPServerSpecAdapter().decode_config(
        json.dumps(
            {
                "mcpServers": {
                    "local": {
                        "command": "python",
                        "args": ["-m", "server"],
                        "env": {"MODE": "readonly"},
                        "future": {"kept-open": True},
                    },
                    "remote": {
                        "url": "https://example.test/mcp",
                        "headers": {"X-Tenant": "tenant"},
                    },
                    "legacy": {
                        "type": "sse",
                        "url": "https://example.test/sse",
                    },
                }
            }
        ).encode()
    )
    by_id = {server.id: server for server in servers}
    assert by_id["local"].transport == "stdio"
    assert dict(by_id["local"].env) == {"MODE": "readonly"}
    assert by_id["remote"].transport == "streamable-http"
    assert by_id["legacy"].transport == "sse"


@pytest.mark.parametrize(
    "payload",
    (
        {"command": "python", "url": "https://example.test/mcp"},
        {"type": "stdio", "url": "https://example.test/mcp"},
        {"type": "http", "command": "python"},
        {"type": "unknown", "url": "https://example.test/mcp"},
    ),
)
def test_mcp_shared_config_rejects_transport_conflicts(
    payload: dict[str, object],
) -> None:
    with pytest.raises(AIError) as error:
        MCPServerSpecAdapter().decode_config(
            json.dumps({"mcpServers": {"server": payload}}).encode()
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_mcp_wire_round_trips_resource_backed_stdio() -> None:
    codec = MCPServerSpecCodec()
    server = MCPServerSpec(
        "server",
        "python",
        ("resource:script.py",),
        AssetKey("mcp", "server"),
        env={"MODE": "readonly"},
        revision=2,
    )

    restored = codec.decode(codec.encode(server))

    assert restored.id == server.id
    assert restored.revision == server.revision
    assert restored.transport == "stdio"
    assert restored.command == "python"
    assert restored.args == ("resource:script.py",)
    assert restored.resource == AssetKey("mcp", "server")
    assert dict(restored.env) == {"MODE": "readonly"}


def test_mcp_wire_rejects_resource_on_remote_transport() -> None:
    codec = MCPServerSpecCodec()
    payload = codec.to_wire_payload(
        MCPServerSpec(
            "remote",
            transport="streamable-http",
            url="https://example.test/mcp",
        )
    )
    payload["resource"] = {"kind": "mcp", "id": "remote"}

    with pytest.raises(AIError) as error:
        codec.from_payload(payload)

    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_mcp_durable_contract_excludes_connection_values() -> None:
    codec = MCPServerSpecCodec()
    first = MCPServerSpec(
        "remote",
        transport="streamable-http",
        url="https://first.example/mcp",
        headers={"Authorization": "secret-a"},
    )
    second = MCPServerSpec(
        "remote",
        transport="streamable-http",
        url="https://second.example/mcp",
        headers={"Authorization": "secret-b"},
    )
    assert codec.to_contract_payload(first) == codec.to_contract_payload(second)
    payload = codec.to_binding_payload(
        first,
        None,
        execution_policy={"version": 1, "boundary": "host-network"},
    )
    encoded = json.dumps(payload)
    assert "first.example" not in encoded
    assert "secret-a" not in encoded
    assert codec.decode_binding_payload(payload, declaration=second) is None


def test_mcp_selectors_round_trip_logical_names() -> None:
    server_id = "审计/security"
    tool_name = "scan:files/%*__v2"
    selector = mcp_tool_selector(server_id, tool_name)
    assert selector == (
        "mcp:%E5%AE%A1%E8%AE%A1%2Fsecurity:"
        "scan%3Afiles%2F%25%2A__v2"
    )
    assert parse_mcp_tool_selector(selector) == (server_id, tool_name)
    lowercase_hex = re.sub(
        r"%[0-9A-F]{2}", lambda match: match.group().lower(), selector
    )
    assert parse_mcp_tool_selector(lowercase_hex) == (server_id, tool_name)
    assert canonical_selectors(
        (lowercase_hex,), field_name="allow_tools", mcp=True
    ) == (selector,)

    wildcard = mcp_server_selector(server_id)
    assert parse_mcp_tool_selector(wildcard) == (server_id, None)
    assert mcp_tool_selector(server_id, "*").endswith(":%2A")
    assert parse_mcp_tool_selector(mcp_tool_selector(server_id, "*")) == (
        server_id,
        "*",
    )
    assert canonical_selectors(
        (wildcard, selector), field_name="allow_tools", mcp=True
    ) == tuple(sorted((wildcard, selector)))
    assert canonical_selectors(
        ("*", selector), field_name="allow_tools", mcp=True
    ) == ("*", selector)
    model_name = _expected_mcp_tool_name(server_id, tool_name)
    assert len(model_name) == 55
    assert model_name.isascii()


@pytest.mark.parametrize(
    "selector",
    (
        "mcp::tool",
        "mcp:server:",
        "mcp:server:tool:extra",
        "mcp:server:unescaped/name",
        "mcp:server:bad%",
        "mcp:server:%FF",
        "mcp:server:bad%2G",
    ),
)
def test_mcp_selector_rejects_malformed_components(selector: str) -> None:
    with pytest.raises(AIError) as error:
        parse_mcp_tool_selector(selector)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_mcp_legacy_selector_is_not_accepted() -> None:
    with pytest.raises(AIError) as error:
        canonical_selectors(
            ("mcp__legacy__tool",),
            field_name="allow_tools",
            mcp=True,
        )
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_mcp_model_tool_mapping_calls_the_original_identity() -> None:
    wrapped = _MCPModelToolset(
        FunctionToolset(
            [Tool(_business, name="scan:file", description="Scan the input.")]
        ),
        "security/audit",
        frozenset({"scan:file"}),
    )
    context = _context()
    tools = await wrapped.get_tools(context)
    model_name = _expected_mcp_tool_name("security/audit", "scan:file")
    assert tuple(tools) == (model_name,)
    assert tools[model_name].tool_def.description == (
        '[MCP identity: {"server_id":"security/audit",'
        '"tool_name":"scan:file"}]\nScan the input.'
    )
    assert await wrapped.call_tool(
        model_name,
        {"value": "request"},
        context,
        tools[model_name],
    ) == "request"


@pytest.mark.asyncio
async def test_mcp_global_wildcard_still_requires_explicit_tool() -> None:
    wrapped = _MCPModelToolset(
        FunctionToolset([Tool(_business, name="available")]),
        "server",
        None,
        frozenset({"missing"}),
    )
    with pytest.raises(AIError) as error:
        await wrapped.get_tools(_context())
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_mcp_server_token_collision_fails_at_compiler_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = agent_compiler.canonical_sha256

    def collide(value: object) -> str:
        if isinstance(value, dict) and value.get("kind") == "mcp-server-name":
            return "a" * 64
        return original(value)  # type: ignore[arg-type]

    monkeypatch.setattr(agent_compiler, "canonical_sha256", collide)
    first = MCPServerSpec("first", "python")
    second = MCPServerSpec("second", "python")
    candidates = (
        CapabilityContribution.from_declaration(first),
        CapabilityContribution.from_declaration(second),
    )
    with pytest.raises(AIError) as error:
        AgentCompiler(
            model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
            candidates=candidates,
            agents={"agent": AgentSpec("agent")},
        )
    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


@pytest.mark.asyncio
async def test_mcp_model_tool_collision_fails_before_exposure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mcp_runtime,
        "_model_tool_name",
        lambda _server, _tool: "mcp__same__token",
    )
    wrapped = _MCPModelToolset(
        FunctionToolset(
            [
                Tool(_business, name="first"),
                Tool(_business, name="second"),
            ]
        ),
        "server",
        None,
    )
    with pytest.raises(AIError) as error:
        await wrapped.get_tools(_context())
    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


def test_skill_contract_round_trips_asset_version_refs() -> None:
    asset = AssetVersionRef(
        AssetKey("skill", "review/guide.md"),
        "asset-source",
        StorageEntryRevision(3),
        "a" * 64,
        7,
    )
    definition = SkillDefinition(
        SkillSpec("review", "instructions"),
        SkillSourceRef(
            "application",
            "review",
            (SkillResource("guide.md", asset, 0o111),),
        ),
    )

    contract = definition.contract
    source = contract["source"]
    assert isinstance(source, dict)
    versions = source["resources"]
    assert isinstance(versions, list)
    assert versions[0]["asset"] == asset.to_payload()
    assert versions[0]["executable_bits"] == 0o111

    restored = SkillDefinition.from_contract(contract)
    assert restored == definition


def test_skill_contract_rejects_malformed_asset_version_ref() -> None:
    with pytest.raises(AIError) as error:
        SkillDefinition.from_contract(
            {
                "version": 1,
                "id": "review",
                "content": "instructions",
                "source": {
                    "source_id": "application",
                    "root": "review",
                    "resources": [
                        {
                            "path": "guide.md",
                            "asset": {
                                "version": 1,
                                "kind": "skill",
                                "id": "review/guide.md",
                                "layer_id": "",
                                "revision": 1,
                                "etag": "a" * 64,
                                "size": 1,
                            },
                            "executable_bits": 0,
                        }
                    ],
                },
            }
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_asset_skill_source_rejects_mismatched_source_ref() -> None:
    store = AssetStore(StorageOverlay(InMemoryAssetBackend()))
    ref = SkillSourceRef("other", "review")
    source = AssetSkillSource("application", store)
    with pytest.raises(AIError) as error:
        await source.inspect(ref)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_memory_owner_selects_only_its_declared_tools() -> None:
    assert select_harness_memory_tools(("write_plan",)) == ()
    assert select_harness_memory_tools(("read_memory",)) == ("read_memory",)
    assert select_harness_memory_tools(("*",)) == (
        "delete_memory",
        "read_memory",
        "search_memory",
        "write_memory",
    )


def test_tool_metadata_preserves_upstream_values() -> None:
    metadata = tool_metadata(
        base={"upstream": "retained"},
        effect_policy="replay_safe",
        plan_safe=True,
        tool_class="business",
    )
    assert metadata["upstream"] == "retained"
    assert metadata["linktools.ai.effect_policy"] == "replay_safe"
    assert metadata["linktools.ai.plan_safe"] is True
    assert metadata["linktools.ai.tool_class"] == "business"


@pytest.mark.parametrize(
    "metadata",
    (
        {"linktools.ai.effect_policy": "unknown"},
        {"linktools.ai.plan_safe": 1},
        {"linktools.ai.tool_class": "filesystem"},
        {"linktools.ai.path_fields": {"path"}},
        {"linktools.ai.path_fields": ["path", "path"]},
        {"linktools.ai.context_dedupe": "legacy"},
    ),
)
def test_invalid_tool_metadata_fails_without_fallback(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(AIError) as error:
        validate_tool_metadata(metadata)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_business_tool_rejects_reserved_mcp_transport_prefix() -> None:
    async def reserved(_ctx: RunContext[None]) -> str:
        return "ok"

    group = CapabilityGroup[None]("business")
    with pytest.raises(AIError) as error:
        group.tool(reserved, name="mcp__reserved")
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_business_tool_metadata_is_captured_in_tool_contract() -> None:
    async def business_tool(
        _ctx: RunContext[None],
        value: str,
    ) -> str:
        return value

    group = CapabilityGroup[None]("business")
    tool = group.tool(
        business_tool,
        effect_policy="replay_safe",
        plan_safe=True,
    )
    candidate = (await group.capture()).contributions[0]
    assert tool.tool_def.metadata == {
        "linktools.ai.effect_policy": "replay_safe",
        "linktools.ai.plan_safe": True,
        "linktools.ai.tool_class": "business",
    }
    assert candidate.contract["metadata"] == tool.tool_def.metadata
    assert "config" not in candidate.contract


@pytest.mark.asyncio
async def test_business_tool_adapter_preserves_sync_execution_and_ctx_keyword() -> None:
    release = threading.Event()

    def blocking_tool(_context: object, ctx: str) -> str:
        return ctx if release.wait(0.2) else "blocked"

    group = CapabilityGroup[None]("business")
    tool = group.tool(blocking_tool, effect_policy="none")
    toolset = FunctionToolset([tool])
    context = _context()
    tools = await toolset.get_tools(context)

    async def release_tool() -> None:
        await asyncio.sleep(0)
        release.set()

    release_task = asyncio.create_task(release_tool())
    result = await toolset.call_tool(
        tool.name,
        {"ctx": "business-value"},
        context,
        tools[tool.name],
    )
    await release_task

    assert result == "business-value"


@pytest.mark.asyncio
async def test_business_tool_adapter_preserves_ctx_keyword_for_async_tools() -> None:
    async def business_tool(_context: object, ctx: str) -> str:
        return ctx

    group = CapabilityGroup[None]("business")
    tool = group.tool(business_tool, effect_policy="none")
    toolset = FunctionToolset([tool])
    context = _context()
    tools = await toolset.get_tools(context)

    assert await toolset.call_tool(
        tool.name,
        {"ctx": "business-value"},
        context,
        tools[tool.name],
    ) == "business-value"


def test_runtime_domain_object_store_trait_has_one_owner() -> None:
    object_domains = {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.TASK,
        RuntimeDomain.RECOVERY,
    }
    assert {
        domain
        for domain in RuntimeDomain
        if runtime_domain_uses_object_store(domain)
    } == object_domains
    assert not runtime_domain_uses_object_store(RuntimeDomain.EVALUATION)


def test_agent_spec_codec_rejects_invalid_v1_payload() -> None:
    with pytest.raises(AIError) as error:
        AgentSpecCodec().decode(
            json.dumps(
                {
                    "version": 1,
                    "id": "agent",
                    "model": "model",
                    "planning": "yes",
                }
            ).encode()
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_declaration_codecs_ignore_unrelated_author_fields() -> None:
    agent = AgentSpecAdapter().from_mapping(
        {
            "version": 1,
            "id": "agent",
            "model": "route",
            "planning": True,
            "future_metadata": {"future": True},
        },
        logical_id="agent",
    )
    assert agent.model == "route"
    assert agent.planning is False

    skill = SkillSpecAdapter().from_mapping(
        {
            "version": 1,
            "id": "skill",
            "content": "skill content",
            "future_metadata": {"future": True},
        }
    )
    assert skill == SkillSpec("skill", "skill content")

    server = MCPServerSpecAdapter().decode_json(
        json.dumps(
            {
                "version": 1,
                "id": "mcp",
                "command": "echo",
                "future_metadata": {"future": True},
            }
        ).encode(),
    )
    assert server == MCPServerSpec("mcp", "echo")


def test_mcp_package_resource_ignores_unrelated_fields() -> None:
    server = MCPServerSpecAdapter().decode_json(
        json.dumps(
            {
                "version": 1,
                "command": "python",
                "resource": {
                    "kind": "mcp",
                    "id": "server",
                    "future": {"enabled": True},
                },
            }
        ).encode(),
        package_id="server",
    )

    assert server.resource == AssetKey("mcp", "server")


def test_mcp_non_package_author_resource_field_has_no_runtime_semantics() -> None:
    server = MCPServerSpecAdapter().decode_json(
        json.dumps(
            {
                "version": 1,
                "id": "server",
                "command": "python",
                "resource": {
                    "kind": "mcp",
                    "id": "server/assets",
                    "future": {"enabled": True},
                },
            }
        ).encode(),
    )

    assert server.resource is None


def test_durable_spec_readers_ignore_additive_fields() -> None:
    agent_codec = AgentSpecCodec()
    agent = AgentSpec("agent")
    agent_payload = agent_codec.to_wire_payload(agent)
    agent_payload["future_note"] = {"category": "display"}
    assert agent_codec.from_payload(agent_payload) == agent

    skill_codec = SkillSpecCodec()
    skill = SkillSpec("skill", "instructions")
    skill_payload = skill_codec.to_wire_payload(skill)
    skill_payload["future_note"] = {"category": "display"}
    assert skill_codec.from_payload(skill_payload) == skill

    mcp_codec = MCPServerSpecCodec()
    server = MCPServerSpec("server", "command")
    mcp_payload = mcp_codec.to_wire_payload(server)
    mcp_payload["future_note"] = {"category": "display"}
    assert mcp_codec.from_payload(mcp_payload) == server

    binding_payload = mcp_codec.to_binding_payload(
        server,
        None,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    binding_payload["future_note"] = {"category": "display"}
    assert (
        mcp_codec.decode_binding_payload(
            binding_payload,
            declaration=server,
        )
        is None
    )


def test_mcp_execution_policy_rejects_non_string_workspace_access() -> None:
    codec = MCPServerSpecCodec()
    server = MCPServerSpec("server", "command")
    payload = codec.to_binding_payload(
        server,
        None,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    payload["execution_policy"] = {
        "version": 1,
        "boundary": "workspace-stdio",
        "workspace_access": [],
        "hidden_paths": [],
        "network": "isolated",
    }

    with pytest.raises(AIError) as error:
        codec.decode_binding_payload(payload, declaration=server)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_mcp_execution_resource_contract_rejects_missing_versions() -> None:
    codec = MCPServerSpecCodec()
    server = MCPServerSpec(
        "mcp",
        "server",
        ("resource:script.py",),
        AssetKey("mcp", "server"),
    )
    payload = codec.to_contract_payload(server)
    payload["execution_policy"] = {"version": 1, "boundary": "host-stdio"}

    with pytest.raises(AIError) as error:
        codec.decode_binding_payload(payload, declaration=server)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_mcp_resource_versions_are_locator_only_for_named_identity() -> None:
    codec = MCPServerSpecCodec()
    server = MCPServerSpec(
        "mcp",
        "server",
        ("resource:script.py",),
        AssetKey("mcp", "server"),
    )
    key = AssetKey("mcp", "server/script.py")
    first_ref = AssetVersionRef(
        key,
        "source-a",
        StorageEntryRevision(1),
        "a" * 64,
        1,
    )
    second_ref = AssetVersionRef(
        key,
        "source-b",
        StorageEntryRevision(9),
        "a" * 64,
        2,
    )

    first = codec.to_binding_payload(
        server,
        (first_ref,),
        asset_source_id="group-a",
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    second = codec.to_binding_payload(
        server,
        (second_ref,),
        asset_source_id="group-b",
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )

    assert first["args"] == ["resource:script.py"]
    assert first["source_id"] == "group-a"
    versions = codec.decode_binding_payload(first, declaration=server)
    assert versions == (first_ref,)
    assert capability_ref_payload("mcp", server.id, first) == (
        capability_ref_payload("mcp", server.id, second)
    )
    with pytest.raises(AIError) as raised:
        codec.from_payload(first)
    assert raised.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


async def _capture_mcp_resource_versions(
    store: AssetStore,
    server: MCPServerSpec,
) -> tuple[AssetVersionRef, ...]:
    capture = await CapabilityGroup("application", assets=store).capture()
    contribution = next(
        item
        for item in capture.contributions
        if item.kind == "mcp" and item.id == server.id
    )
    declaration = contribution.value
    assert isinstance(declaration, MCPServerSpec)
    versions = MCPServerSpecCodec().decode_binding_payload(
        contribution.contract,
        declaration=declaration,
    )
    assert versions is not None
    return versions


@pytest.mark.asyncio
async def test_mcp_resource_capture_is_stable_and_includes_binary_files() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    root = AssetKey("mcp", "server")
    server = MCPServerSpec("server", "python", (), root)
    codec = MCPServerSpecCodec()
    binary = b"\x00\xffresource"
    try:
        await store.put(
            AssetKey("mcp", "server/mcp.json"),
            codec.encode(server),
        )
        await store.put(AssetKey("mcp", "server/data.bin"), binary)
        await store.put(AssetKey("mcp", "server/nested/guide.md"), b"guide")
        first_versions = await _capture_mcp_resource_versions(store, server)
        await store.put(
            AssetKey("agent", "unrelated"),
            AgentSpecCodec().encode(AgentSpec("unrelated")),
        )
        second_versions = await _capture_mcp_resource_versions(store, server)

        assert first_versions == second_versions
        assert {
            item.key.id: item.etag
            for item in first_versions
        } == {
            "server/data.bin": hashlib.sha256(binary).hexdigest(),
            "server/nested/guide.md": hashlib.sha256(b"guide").hexdigest(),
        }
        empty_server = MCPServerSpec(
            "empty",
            "python",
            (),
            AssetKey("mcp", "empty"),
        )
        await store.put(
            AssetKey("mcp", "empty/mcp.json"),
            codec.encode(empty_server),
        )
        empty_versions = await _capture_mcp_resource_versions(store, empty_server)
        assert empty_versions == ()
    finally:
        await store.close()


class _RacingMCPAssetStore(AssetStore):
    def __init__(self, backend: InMemoryAssetBackend) -> None:
        super().__init__(StorageOverlay(backend, writer=backend))
        self._raced = False

    async def resolve_versions(
        self,
        keys: Sequence[AssetKey],
    ) -> tuple[AssetVersionRef, ...]:
        if not self._raced:
            self._raced = True
            await self.put(keys[0], b"changed")
        return await super().resolve_versions(keys)


@pytest.mark.asyncio
async def test_mcp_resource_resolution_rejects_selected_asset_version_race() -> None:
    backend = InMemoryAssetBackend()
    store = _RacingMCPAssetStore(backend)
    await store.initialize()
    server = MCPServerSpec("server", "python", (), AssetKey("mcp", "server"))
    await store.put(
        AssetKey("mcp", "server/mcp.json"),
        MCPServerSpecCodec().encode(server),
    )
    await store.put(AssetKey("mcp", "server/tool.py"), b"original")
    try:
        with pytest.raises(AIError) as error:
            await _capture_mcp_resource_versions(store, server)
        assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mcp_resource_resolution_excludes_declaration_files() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        root = AssetKey("mcp", "server")
        server = MCPServerSpec("server", "python", (), root)
        declaration = AssetKey("mcp", "server/mcp.json")
        resource = AssetKey("mcp", "server/tool.py")
        await store.put(declaration, MCPServerSpecCodec().encode(server))
        await store.put(resource, b"resource")

        versions = await _capture_mcp_resource_versions(store, server)

        assert tuple(item.key for item in versions) == (resource,)
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "paths",
    (
        ("a/./b.py",),
        ("a//b.py",),
        ("../escape.py",),
        ("./tool.py",),
        ("tool.py/",),
        ("C:/tool.py",),
        ("file:tool.py",),
        ("virtual:tool.py",),
        ("lib", "lib/helper.py"),
    ),
)
@pytest.mark.parametrize("boundary", ("resolution", "binding"))
async def test_mcp_resource_versions_reject_invalid_tree(
    paths: tuple[str, ...],
    boundary: str,
) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        root = AssetKey("mcp", "server/assets")
        server = MCPServerSpec("server/assets", "python", (), root)
        keys = tuple(AssetKey("mcp", f"server/assets/{path}") for path in paths)
        await store.put(
            AssetKey("mcp", "server/assets/mcp.json"),
            MCPServerSpecCodec().encode(server),
        )
        for key in keys:
            await store.put(key, b"data")
        with pytest.raises(AIError) as raised:
            if boundary == "resolution":
                await _capture_mcp_resource_versions(store, server)
            else:
                versions = await store.resolve_versions(keys)
                _bound_resource_versions(
                    server,
                    _MCPBinding(
                        versions,
                        "application",
                        {"version": 1, "boundary": "host-stdio"},
                    ),
                )
        assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mcp_resource_prefix_is_literal_without_resource() -> None:
    server = MCPServerSpec(
        "server",
        "python",
        ("resource:literal-value",),
    )
    binding = _MCPBinding(
        None,
        None,
        {"version": 1, "boundary": "host-stdio"},
    )

    projections = await prepare_mcp_projections(
        (server,),
        {server.id: binding},
        asset_readers={},
        sandboxed=False,
    )

    assert projections[server.id].args == ("resource:literal-value",)
    assert projections[server.id].resources == ()


@pytest.mark.asyncio
async def test_mcp_resource_path_requires_local_asset_files() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        resource = AssetKey("mcp", "server/assets/script.py")
        helper = AssetKey("mcp", "server/assets/lib/helper.py")
        root = AssetKey("mcp", "server/assets")
        server = MCPServerSpec(
            "server/assets",
            "python",
            ("resource:script.py",),
            root,
        )
        await store.put(
            AssetKey("mcp", "server/assets/mcp.json"),
            MCPServerSpecCodec().encode(server),
        )
        await store.put(resource, b"print('versioned')\n")
        await store.put(helper, b"VALUE = 42\n")
        versions = await _capture_mcp_resource_versions(store, server)
        with pytest.raises(AIError) as raised:
            await prepare_mcp_projections(
                (server,),
                {
                    server.id: _MCPBinding(
                        versions,
                        "application",
                        {"version": 1, "boundary": "host-stdio"},
                    )
                },
                asset_readers={"application": store},
                sandboxed=False,
            )
        assert raised.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        assert raised.value.safe_details == {
            "kind": "mcp_local_resource",
            "server_id": server.id,
        }
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mcp_resource_paths_use_original_local_files(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    script = root / "mcp" / "server" / "assets" / "script.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ready')\n", encoding="utf-8")
    resource = AssetKey("mcp", "server/assets")
    server = MCPServerSpec(
        "server/assets", "python", ("resource:script.py",), resource
    )
    declaration = root / "mcp" / "server" / "assets" / "mcp.json"
    declaration.parent.mkdir(parents=True, exist_ok=True)
    declaration.write_bytes(MCPServerSpecCodec().encode(server))
    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=PrefixAssetPathAdapter({"mcp": "mcp"}),
                kinds=("mcp",),
            )
        )
    )
    await store.initialize()
    try:
        versions = await _capture_mcp_resource_versions(store, server)
        binding = _MCPBinding(
            versions,
            "application",
            {"version": 1, "boundary": "host-stdio"},
        )
        host = await prepare_mcp_projections(
            (server,),
            {server.id: binding},
            asset_readers={"application": store},
            sandboxed=False,
        )
        assert host[server.id].args == (str(script.resolve()),)
        assert host[server.id].resources == ()

        sandboxed = await prepare_mcp_projections(
            (server,),
            {server.id: binding},
            asset_readers={"application": store},
            sandboxed=True,
        )
        assert sandboxed[server.id].args == (
            SandboxResourcePath(server.id, "script.py"),
        )
        assert sandboxed[server.id].resources[0].files == {
            "script.py": script.resolve()
        }
    finally:
        await store.close()


def test_agent_authoring_ignores_runtime_only_usage_limits() -> None:
    spec = AgentSpecAdapter().from_mapping(
        {
            "version": 1,
            "id": "agent",
            "usage_limits": {
                "model_requests": 1,
                "future_limit": {"unit": "request"},
            },
        },
        logical_id="agent",
    )
    assert spec.usage_limits is None


def test_durable_usage_limits_ignore_additive_fields() -> None:
    spec = AgentSpecCodec().from_payload(
        {
            "version": 1,
            "id": "agent",
            "usage_limits": {
                "model_requests": 1,
                "future_limit": {"unit": "request"},
            },
        }
    )
    assert spec.usage_limits == AgentUsageLimits(model_requests=1)


def test_spec_constructors_reject_invalid_values() -> None:
    with pytest.raises(TypeError):
        AgentUsageLimits(model_requests=True)
    with pytest.raises(ValueError):
        AgentUsageLimits()
    with pytest.raises(ValueError):
        AgentSpec("")
    with pytest.raises(TypeError):
        AgentSpec("agent", instructions=(1,))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SkillSpec("", "content")
    with pytest.raises(TypeError):
        SkillSpec("skill", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", "")
    with pytest.raises(ValueError):
        AgentSpec("agent", revision=0)
    with pytest.raises(ValueError):
        SkillSpec("skill", "content", revision=0)
    with pytest.raises(ValueError):
        MCPServerSpec("mcp", "command", revision=0)


@pytest.mark.asyncio
async def test_skill_capability_lists_skills_and_rejects_duplicate_ids() -> None:
    capability = SkillCapability(
        (
            SkillDefinition(SkillSpec("z", "z skill")),
            SkillDefinition(SkillSpec("a", "a skill")),
        ),
        SkillSourceRegistry(),
    )
    assert [item["id"] for item in await capability.list_skills()] == ["a", "z"]
    assert capability.get_toolset().id == "linktools.ai.skills"
    with pytest.raises(AIError) as error:
        SkillCapability(
            (
                SkillDefinition(SkillSpec("same", "one")),
                SkillDefinition(SkillSpec("same", "two")),
            ),
            SkillSourceRegistry(),
        )
    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT


async def _business(value: str) -> str:
    return value


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_runtime_tool_boundary_requires_a_descriptor_for_every_leaf() -> None:
    boundary = BoundaryToolset(
        (
            FunctionToolset(
                [
                    Tool(
                        _business,
                        metadata=tool_metadata(
                            effect_policy="none",
                            tool_class="business",
                        ),
                    )
                ]
            ),
        ),
        {
            "_business": ManagedToolDescriptor(
                effect_owner="none",
                effect_policy="none",
                tool_class="business",
            )
        },
        id="business",
    )
    context = _context()
    tools = await boundary.get_tools(context)
    assert await boundary.call_tool(
        "_business", {"value": "ok"}, context, tools["_business"]
    ) == "ok"
    unknown = BoundaryToolset(
        (FunctionToolset([_business]),),
        {},
        id="invalid",
    )
    with pytest.raises(AIError) as error:
        await unknown.get_tools(context)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_runtime_tool_boundary_does_not_rewrite_explicit_descriptor() -> None:
    boundary = BoundaryToolset(
        (
            FunctionToolset(
                [
                    Tool(
                        _business,
                        metadata={"linktools.ai.effect_policy": "invalid"},
                    )
                ]
            ),
        ),
        {
            "_business": ManagedToolDescriptor(
                effect_owner="none",
                effect_policy="none",
                tool_class="business",
            )
        },
        id="business",
    )
    context = _context()
    tools = await boundary.get_tools(context)
    assert await boundary.call_tool(
        "_business", {"value": "ok"}, context, tools["_business"]
    ) == "ok"
