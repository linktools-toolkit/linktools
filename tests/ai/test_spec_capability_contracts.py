#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declaration, capability semantic, and runtime-leaf contracts."""

import hashlib
import json
import re
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
    CapabilityContribution,
    CapabilityGroup,
    AssetVersionSkillResourceSource,
    SkillCapability,
    SkillResourceVersion,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
    tool_semantic_metadata,
    validate_tool_semantic_metadata,
)
from linktools.ai.asset import (
    AssetKey,
    AssetStore,
    AssetVersionRef,
    InMemoryAssetBackend,
)
from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._binding_resolver import _resolve_mcp_resource_versions
from linktools.ai.runtime._harness_memory import select_harness_memory_tools
from linktools.ai.runtime._mcp import (
    _MCPModelToolset,
    _MCPResourceBinding,
    _materialize_resource_versions,
)
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.runtime.state import (
    RuntimeDomain,
    runtime_domain_uses_object_store,
)
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecCodec,
    AgentUsageLimits,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillSpec,
    SkillSpecCodec,
    canonical_selectors,
    capability_identity_payload,
    mcp_server_selector,
    mcp_tool_selector,
    parse_mcp_tool_selector,
)
from linktools.ai.storage import (
    StorageEntryRevision,
    StorageOverlay,
)


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
            model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
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
        SkillSourceRef("application", "review").with_asset_versions(
            (SkillResourceVersion("guide.md", asset),),
            "b" * 64,
            sandbox_materialize=False,
        ),
    )

    contract = definition.semantic_contract
    source = contract["source"]
    assert isinstance(source, dict)
    versions = source["resource_versions"]
    assert isinstance(versions, list)
    assert versions[0]["asset"] == asset.to_payload()

    restored = SkillDefinition.from_semantic_contract(contract)
    assert restored == definition


def test_skill_contract_rejects_malformed_asset_version_ref() -> None:
    with pytest.raises(AIError) as error:
        SkillDefinition.from_semantic_contract(
            {
                "version": 1,
                "id": "review",
                "content": "instructions",
                "source": {
                    "source_id": "application",
                    "root": "review",
                    "resource_versions": [
                        {
                            "path": "guide.md",
                            "asset": {
                                "version": 1,
                                "kind": "skill",
                                "id": "review/guide.md",
                                "source_id": "",
                                "revision": 1,
                                "etag": "a" * 64,
                                "size": 1,
                            },
                            "executable_bits": 0,
                        }
                    ],
                    "sandbox_materialize": False,
                    "resource_semantic_digest": "b" * 64,
                },
            }
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_asset_version_skill_source_rejects_mismatched_source_ref() -> None:
    store = AssetStore(StorageOverlay(InMemoryAssetBackend()))
    ref = SkillSourceRef("other", "review").with_asset_versions(
        (),
        canonical_sha256(
            {
                "version": 1,
                "kind": "skill-resource-semantics",
                "sandbox_materialize": False,
                "files": [],
            }
        ),
        sandbox_materialize=False,
    )
    with pytest.raises(AIError) as error:
        AssetVersionSkillResourceSource(
            "application",
            {"review": ref},
            store,
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_memory_owner_selects_only_its_declared_tools() -> None:
    assert select_harness_memory_tools(("write_plan",)) == ()
    assert select_harness_memory_tools(("read_memory",)) == ("read_memory",)
    assert select_harness_memory_tools(("*",)) == (
        "delete_memory",
        "read_memory",
        "search_memory",
        "write_memory",
    )


def test_tool_semantic_metadata_preserves_upstream_values() -> None:
    metadata = tool_semantic_metadata(
        base={"upstream": "retained"},
        effect="replay_safe",
        plan_safe=True,
        tool_class="business",
    )
    assert metadata["upstream"] == "retained"
    assert metadata["linktools.ai.effect"] == "replay_safe"
    assert metadata["linktools.ai.plan_safe"] is True
    assert metadata["linktools.ai.tool_class"] == "business"


@pytest.mark.parametrize(
    "metadata",
    (
        {"linktools.ai.effect": "unknown"},
        {"linktools.ai.plan_safe": 1},
        {"linktools.ai.tool_class": "filesystem"},
        {"linktools.ai.path_fields": {"path"}},
        {"linktools.ai.path_fields": ["path", "path"]},
        {"linktools.ai.context_dedupe": "legacy"},
    ),
)
def test_invalid_tool_semantics_fail_without_fallback(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(AIError) as error:
        validate_tool_semantic_metadata(metadata)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_business_tool_rejects_reserved_mcp_transport_prefix() -> None:
    async def reserved(_ctx: RunContext[None]) -> str:
        return "ok"

    group = CapabilityGroup[None]("business")
    with pytest.raises(AIError) as error:
        group.tool(reserved, name="mcp__reserved")
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_business_tool_semantics_are_captured_in_tool_metadata() -> None:
    async def business_tool(
        _ctx: RunContext[None],
        value: str,
    ) -> str:
        return value

    group = CapabilityGroup[None]("business")
    tool = group.tool(
        business_tool,
        effect="replay_safe",
        plan_safe=True,
    )
    candidate = (await group.snapshot()).contributions[0]
    assert tool.tool_def.metadata == {
        "linktools.ai.effect": "replay_safe",
        "linktools.ai.plan_safe": True,
        "linktools.ai.tool_class": "business",
    }
    assert candidate.semantic_contract["metadata"] == tool.tool_def.metadata
    assert "config" not in candidate.semantic_contract


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


def test_declaration_codecs_reject_unknown_author_fields() -> None:
    agent_payload = {
        "version": 1,
        "id": "agent",
        "future_metadata": {"future": True},
    }
    with pytest.raises(AIError) as error:
        AgentSpecCodec().from_author_payload(agent_payload)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    skill_payload = {
        "version": 1,
        "id": "skill",
        "content": "skill content",
        "future_metadata": {"future": True},
    }
    with pytest.raises(AIError) as error:
        SkillSpecCodec().from_author_payload(skill_payload)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    mcp_payload = {
        "version": 1,
        "id": "mcp",
        "command": "echo",
        "future_metadata": {"future": True},
    }
    with pytest.raises(AIError) as error:
        MCPServerSpecCodec().decode_author(
            json.dumps(mcp_payload).encode(),
            format="json",
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_durable_spec_readers_ignore_additive_fields() -> None:
    agent_codec = AgentSpecCodec()
    agent = AgentSpec("agent")
    agent_payload = agent_codec.to_payload(agent)
    agent_payload["future_note"] = {"category": "display"}
    assert agent_codec.from_payload(agent_payload) == agent

    skill_codec = SkillSpecCodec()
    skill = SkillSpec("skill", "instructions")
    skill_payload = skill_codec.to_payload(skill)
    skill_payload["future_note"] = {"category": "display"}
    assert skill_codec.from_payload(skill_payload) == skill

    mcp_codec = MCPServerSpecCodec()
    server = MCPServerSpec("server", "command")
    mcp_payload = mcp_codec.to_payload(server)
    mcp_payload["future_note"] = {"category": "display"}
    assert mcp_codec.from_payload(mcp_payload) == server

    execution_payload = mcp_codec.to_execution_payload(
        server,
        None,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    execution_payload["future_note"] = {"category": "display"}
    assert mcp_codec.from_execution_payload(execution_payload) == (server, None)


def test_mcp_execution_resource_contract_rejects_missing_versions() -> None:
    codec = MCPServerSpecCodec()
    server = MCPServerSpec(
        "mcp",
        "server",
        ("resource:script.py",),
        AssetKey("mcp", "server"),
    )
    payload = codec.to_payload(server)
    payload["execution_policy"] = {"version": 1, "boundary": "host-stdio"}

    with pytest.raises(AIError) as error:
        codec.from_execution_payload(payload)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_mcp_resource_versions_are_locator_only_for_semantic_identity() -> None:
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

    first = codec.to_execution_payload(
        server,
        (first_ref,),
        resource_source_id="group-a",
        resource_semantic_digest="d" * 64,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    second = codec.to_execution_payload(
        server,
        (second_ref,),
        resource_source_id="group-b",
        resource_semantic_digest="d" * 64,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )

    assert first["args"] == ["resource:script.py"]
    assert first["resource_source_id"] == "group-a"
    restored, versions = codec.from_execution_payload(first)
    assert restored == server
    assert versions == (first_ref,)
    assert capability_identity_payload("mcp", server.id, first) == (
        capability_identity_payload("mcp", server.id, second)
    )
    with pytest.raises(AIError) as raised:
        codec.from_payload(first)
    assert raised.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.asyncio
async def test_skill_resource_digest_tracks_behavior_not_asset_locator() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        key = AssetKey("skill", "review/scripts/run.bin")
        expected_digest = canonical_sha256(
            {
                "version": 1,
                "kind": "skill-resource-semantics",
                "sandbox_materialize": True,
                "files": [
                    {
                        "path": "scripts/run.bin",
                        "sha256": "a" * 64,
                        "executable_bits": 0o111,
                    }
                ],
            }
        )
        original = SkillSourceRef("application", "review").with_asset_versions(
            (
                SkillResourceVersion(
                    "scripts/run.bin",
                    AssetVersionRef(
                        key,
                        "source-a",
                        StorageEntryRevision(1),
                        "a" * 64,
                        1,
                    ),
                    0o111,
                ),
            ),
            expected_digest,
            sandbox_materialize=True,
        )

        async def digest(ref: SkillSourceRef) -> str:
            source = AssetVersionSkillResourceSource(
                "application",
                {"review": ref},
                store,
            )
            return await source.semantic_digest("review")

        first = await digest(original)
        relocated = SkillSourceRef("application", "review").with_asset_versions(
            (
                SkillResourceVersion(
                    "scripts/run.bin",
                    AssetVersionRef(
                        key,
                        "source-b",
                        StorageEntryRevision(9),
                        "a" * 64,
                        99,
                    ),
                    0o111,
                ),
            ),
            first,
            sandbox_materialize=True,
        )
        non_executable = SkillSourceRef("application", "review").with_asset_versions(
            (
                SkillResourceVersion(
                    "scripts/run.bin",
                    relocated.resource_versions[0].asset,
                    0,
                ),
            ),
            "0" * 64,
            sandbox_materialize=True,
        )
        not_materialized = SkillSourceRef("application", "review").with_asset_versions(
            relocated.resource_versions,
            "0" * 64,
            sandbox_materialize=False,
        )
        changed = SkillSourceRef("application", "review").with_asset_versions(
            (
                SkillResourceVersion(
                    "scripts/run.bin",
                    AssetVersionRef(
                        key,
                        "source-a",
                        StorageEntryRevision(2),
                        "b" * 64,
                        1,
                    ),
                    0o111,
                ),
            ),
            "0" * 64,
            sandbox_materialize=True,
        )

        assert first == await digest(relocated)
        with pytest.raises(AIError):
            await digest(non_executable)
        with pytest.raises(AIError):
            await digest(not_materialized)
        with pytest.raises(AIError):
            await digest(changed)
        assert first == expected_digest
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mcp_resource_digest_is_stable_and_includes_binary_files() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    root = AssetKey("mcp", "server")
    binary = b"\x00\xffresource"
    try:
        await store.put(AssetKey("mcp", "server/data.bin"), binary)
        await store.put(AssetKey("mcp", "server/nested/guide.md"), b"guide")
        first_versions, first_digest = await _resolve_mcp_resource_versions(
            store,
            root,
            (),
        )
        await store.put(AssetKey("agent", "unrelated"), b"unrelated")
        second_versions, second_digest = await _resolve_mcp_resource_versions(
            store,
            root,
            (),
        )

        assert first_versions == second_versions
        assert first_digest == second_digest
        assert first_digest == canonical_sha256(
            {
                "version": 1,
                "kind": "mcp-resource-semantics",
                "files": [
                    {
                        "path": "data.bin",
                        "sha256": hashlib.sha256(binary).hexdigest(),
                    },
                    {
                        "path": "nested/guide.md",
                        "sha256": hashlib.sha256(b"guide").hexdigest(),
                    },
                ],
            }
        )
        empty_versions, empty_digest = await _resolve_mcp_resource_versions(
            store,
            AssetKey("mcp", "empty"),
            (),
        )
        assert empty_versions == ()
        assert empty_digest == canonical_sha256(
            {
                "version": 1,
                "kind": "mcp-resource-semantics",
                "files": [],
            }
        )
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
    await store.put(AssetKey("mcp", "server/tool.py"), b"original")
    try:
        with pytest.raises(AIError) as error:
            await _resolve_mcp_resource_versions(
                store,
                AssetKey("mcp", "server"),
                (),
            )
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
        declaration = AssetKey("mcp", "server/mcp.json")
        resource = AssetKey("mcp", "server/tool.py")
        await store.put(declaration, b"declaration")
        await store.put(resource, b"resource")

        versions, _digest = await _resolve_mcp_resource_versions(store, root, ())

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
@pytest.mark.parametrize("boundary", ("resolution", "materialization"))
async def test_mcp_resource_versions_reject_unmaterializable_tree(
    paths: tuple[str, ...],
    boundary: str,
) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        root = AssetKey("mcp", "server/assets")
        keys = tuple(AssetKey("mcp", f"server/assets/{path}") for path in paths)
        for key in keys:
            await store.put(key, b"data")
        with pytest.raises(AIError) as raised:
            if boundary == "resolution":
                await _resolve_mcp_resource_versions(store, root, ())
            else:
                versions = await store.resolve_versions(keys)
                await _materialize_resource_versions(
                    MCPServerSpec("server", "python", (), root),
                    _MCPResourceBinding(
                        versions,
                        "application",
                        "a" * 64,
                        {"version": 1, "boundary": "host-stdio"},
                    ),
                    store,
                )
        assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mcp_resource_versions_materialize_deleted_current_assets() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    directory = None
    try:
        resource = AssetKey("mcp", "server/assets/script.py")
        helper = AssetKey("mcp", "server/assets/lib/helper.py")
        root = AssetKey("mcp", "server/assets")
        await store.put(resource, b"print('versioned')\n")
        await store.put(helper, b"VALUE = 42\n")
        versions, digest = await _resolve_mcp_resource_versions(
            store,
            root,
            ("resource:script.py",),
        )
        await store.delete(resource)
        await store.delete(helper)

        server = MCPServerSpec(
            "foo/bar",
            "python",
            ("resource:script.py",),
            root,
        )
        directory = await _materialize_resource_versions(
            server,
            _MCPResourceBinding(
                versions,
                "application",
                digest,
                {"version": 1, "boundary": "host-stdio"},
            ),
            store,
        )

        assert (Path(directory.name) / "script.py").read_bytes() == (
            b"print('versioned')\n"
        )
        assert (Path(directory.name) / "lib/helper.py").read_bytes() == b"VALUE = 42\n"
    finally:
        if directory is not None:
            directory.cleanup()
        await store.close()


def test_agent_spec_codec_rejects_unknown_usage_limit_fields() -> None:
    payload = {
        "version": 1,
        "id": "agent",
        "usage_limits": {
            "model_requests": 1,
            "future_limit": {"unit": "request"},
        },
    }
    with pytest.raises(AIError) as error:
        AgentSpecCodec().from_author_payload(payload)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


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
    boundary = RuntimeToolBoundaryToolset(
        (
            FunctionToolset(
                [
                    Tool(
                        _business,
                        metadata=tool_semantic_metadata(
                            effect="none",
                            tool_class="business",
                        ),
                    )
                ]
            ),
        ),
        {
            "_business": ManagedToolDescriptor(
                effect_owner="none",
                effect="none",
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
    unknown = RuntimeToolBoundaryToolset(
        (FunctionToolset([_business]),),
        {},
        id="invalid",
    )
    with pytest.raises(AIError) as error:
        await unknown.get_tools(context)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_runtime_tool_boundary_does_not_rewrite_explicit_descriptor() -> None:
    boundary = RuntimeToolBoundaryToolset(
        (
            FunctionToolset(
                [
                    Tool(
                        _business,
                        metadata={"linktools.ai.effect": "invalid"},
                    )
                ]
            ),
        ),
        {
            "_business": ManagedToolDescriptor(
                effect_owner="none",
                effect="none",
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
