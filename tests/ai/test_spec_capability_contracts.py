#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declaration, capability semantic, and runtime-leaf contracts."""

import hashlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai import Tool
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

import linktools.ai.runtime._mcp as mcp_runtime
from linktools.ai.capability import (
    CapabilityGroup,
    FrozenSkillResourceSource,
    SkillCapability,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
    tool_semantic_metadata,
    validate_tool_semantic_metadata,
)
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.core import canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._binding_freeze import _snapshot_mcp_resources
from linktools.ai.runtime._harness_memory import select_harness_memory_tools
from linktools.ai.runtime._mcp import (
    _MCPModelToolset,
    _FrozenMCPResources,
    _materialize_resource_snapshot,
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
    InMemoryObjectStore,
    ObjectRef,
    StorageOverlay,
    StorageRevision,
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


async def _put_object(
    store: InMemoryObjectStore,
    key: str,
    value: bytes,
) -> ObjectRef:
    async def chunks() -> AsyncIterator[bytes]:
        yield value

    stat = await store.put(
        key,
        chunks(),
        expected_size=len(value),
        expected_digest=hashlib.sha256(value).hexdigest(),
    )
    return ObjectRef(store.store_id, stat.key, stat.digest, stat.size)


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


def test_skill_snapshot_contract_preserves_object_store_owner() -> None:
    reference = ObjectRef("execution", "snapshot", "a" * 64, 1)
    definition = SkillDefinition(
        SkillSpec("review", "instructions"),
        SkillSourceRef(
            "application",
            "review",
            reference,
            "b" * 64,
        ),
    )

    contract = definition.semantic_contract
    source = contract["source"]
    assert isinstance(source, dict)
    snapshot = source["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["store_id"] == "execution"
    restored = SkillDefinition.from_semantic_contract(contract)
    assert restored.source_ref is not None
    assert restored.source_ref.snapshot == reference


def test_frozen_skill_source_rejects_wrong_object_store_owner() -> None:
    with pytest.raises(AIError) as error:
        FrozenSkillResourceSource(
            "application",
            {"review": ObjectRef("owner", "snapshot", "a" * 64, 1)},
            InMemoryObjectStore("other"),
        )
    assert error.value.code is ErrorCode.STORAGE_OWNER_MISMATCH


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


@pytest.mark.asyncio
async def test_business_tool_semantics_are_frozen_in_tool_metadata() -> None:
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
    candidate = (await group.freeze()).contributions[0]
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

    frozen_payload = mcp_codec.to_frozen_payload(
        server,
        None,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    frozen_payload["future_note"] = {"category": "display"}
    assert mcp_codec.from_frozen_payload(frozen_payload) == (server, None)


def test_mcp_resource_snapshot_is_runtime_owned_and_locator_is_not_semantic() -> None:
    codec = MCPServerSpecCodec()
    server = MCPServerSpec(
        "mcp",
        "server",
        ("resource:script.py",),
        AssetKey("mcp", "server"),
    )
    declaration = codec.to_payload(server)
    assert declaration["version"] == 1
    assert codec.from_payload(declaration) == server

    first = codec.to_frozen_payload(
        server,
        ObjectRef("runtime", "v1/asset-snapshot/one", "a" * 64, 1),
        resource_semantic_digest="d" * 64,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    second = codec.to_frozen_payload(
        server,
        ObjectRef("other", "v1/asset-snapshot/two", "a" * 64, 1),
        resource_semantic_digest="d" * 64,
        execution_policy={"version": 1, "boundary": "host-stdio"},
    )
    assert first["version"] == 1
    assert first["args"] is None
    assert first["frozen_args"] == ["resource:script.py"]
    restored, reference = codec.from_frozen_payload(first)
    assert restored == server
    assert reference == ObjectRef(
        "runtime", "v1/asset-snapshot/one", "a" * 64, 1
    )
    assert capability_identity_payload("mcp", server.id, first) == (
        capability_identity_payload("mcp", server.id, second)
    )
    with pytest.raises(AIError) as raised:
        codec.from_payload(first)
    assert raised.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.asyncio
async def test_skill_resource_digest_tracks_behavior_not_manifest_metadata() -> None:
    objects = InMemoryObjectStore("runtime")
    content = b"\x00\xffbinary"
    content_digest = hashlib.sha256(content).hexdigest()
    await _put_object(objects, "resources/script", content)

    def manifest(
        *,
        revision: str,
        mode: int = 0o111,
        sandbox_materialize: bool = True,
        body: bytes = content,
        content_key: str = "resources/script",
        extra: bool = False,
    ) -> dict[str, object]:
        resource_digest = hashlib.sha256(body).hexdigest()
        payload: dict[str, object] = {
            "kind": "skill-source-snapshot",
            "format_version": 1,
            "source_id": "application",
            "root": "review",
            "revision": revision,
            "sandbox_materialize": sandbox_materialize,
            "resources": [
                {
                    "path": "scripts/run.bin",
                    "mode": mode,
                    "content": {
                        "key": content_key,
                        "digest": resource_digest,
                        "size": len(body),
                    },
                }
            ],
        }
        if extra:
            payload["storage_metadata"] = {"writer": "fixture"}
        return payload

    original = await _put_object(
        objects,
        "manifests/original",
        canonical_json_bytes(manifest(revision="revision-1")),
    )
    revised = await _put_object(
        objects,
        "manifests/revised",
        canonical_json_bytes(
            manifest(revision="revision-2", extra=True)
        ),
    )
    non_executable = await _put_object(
        objects,
        "manifests/non-executable",
        canonical_json_bytes(
            manifest(revision="revision-2", mode=0)
        ),
    )
    not_materialized = await _put_object(
        objects,
        "manifests/not-materialized",
        canonical_json_bytes(
            manifest(
                revision="revision-2",
                sandbox_materialize=False,
            )
        ),
    )
    await _put_object(objects, "resources/changed", b"changed")
    changed = await _put_object(
        objects,
        "manifests/changed",
        canonical_json_bytes(
            manifest(
                revision="revision-2",
                body=b"changed",
                content_key="resources/changed",
            )
        ),
    )

    async def digest(reference: ObjectRef) -> str:
        source = FrozenSkillResourceSource(
            "application",
            {"review": reference},
            objects,
        )
        return await source.semantic_digest("review")

    first = await digest(original)
    assert first == await digest(revised)
    assert first != await digest(non_executable)
    assert first != await digest(not_materialized)
    assert first != await digest(changed)
    assert first == canonical_sha256(
        {
            "version": 1,
            "kind": "skill-resource-semantics",
            "sandbox_materialize": True,
            "files": [
                {
                    "path": "scripts/run.bin",
                    "sha256": content_digest,
                    "executable_bits": 0o111,
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_mcp_resource_digest_is_stable_and_includes_binary_files() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    objects = InMemoryObjectStore("runtime")
    root = AssetKey("mcp", "server")
    binary = b"\x00\xffresource"
    try:
        await store.put(AssetKey("mcp", "server/data.bin"), binary)
        await store.put(AssetKey("mcp", "server/nested/guide.md"), b"guide")
        first_ref, first_digest = await _snapshot_mcp_resources(
            store,
            root,
            (),
            object_store=objects,
        )
        await store.put(AssetKey("agent", "unrelated"), b"unrelated")
        second_ref, second_digest = await _snapshot_mcp_resources(
            store,
            root,
            (),
            object_store=objects,
        )

        assert first_ref != second_ref
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
        _, empty_digest = await _snapshot_mcp_resources(
            store,
            AssetKey("mcp", "empty"),
            (),
            object_store=objects,
        )
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

    async def snapshot(
        self,
        keys: Sequence[AssetKey],
        *,
        object_store: InMemoryObjectStore,
        expected_revision: StorageRevision | None = None,
    ) -> ObjectRef:
        if not self._raced:
            self._raced = True
            await self.put(AssetKey("mcp", "unrelated"), b"changed")
        return await super().snapshot(
            keys,
            object_store=object_store,
            expected_revision=expected_revision,
        )


@pytest.mark.asyncio
async def test_mcp_resource_freeze_rejects_a_source_revision_race() -> None:
    backend = InMemoryAssetBackend()
    store = _RacingMCPAssetStore(backend)
    await store.initialize()
    await store.put(AssetKey("mcp", "server/tool.py"), b"print('ok')")
    try:
        with pytest.raises(AIError) as error:
            await _snapshot_mcp_resources(
                store,
                AssetKey("mcp", "server"),
                (),
                object_store=InMemoryObjectStore("runtime"),
            )
        assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "paths", (("a/./b.py",), ("a//b.py",), ("../escape.py",), ("lib", "lib/helper.py"))
)
@pytest.mark.parametrize("boundary", ("snapshot", "materialization"))
async def test_mcp_resource_snapshot_rejects_unmaterializable_tree(
    paths: tuple[str, ...],
    boundary: str,
) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        objects = InMemoryObjectStore("runtime")
        root = AssetKey("mcp", "server/assets")
        keys = tuple(AssetKey("mcp", f"server/assets/{path}") for path in paths)
        for key in keys:
            await store.put(key, b"data")
        reference = (
            await store.snapshot(keys, object_store=objects)
            if boundary == "materialization"
            else None
        )
        digest = "a" * 64
        with pytest.raises(AIError) as raised:
            if boundary == "snapshot":
                await _snapshot_mcp_resources(store, root, (), object_store=objects)
            else:
                await _materialize_resource_snapshot(
                    MCPServerSpec("server", "python", (), root),
                    _FrozenMCPResources(
                        reference,
                        digest,
                        {"version": 1, "boundary": "host-stdio"},
                    ),
                    objects,
                )
        assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mcp_resource_snapshot_materializes_deleted_source_bytes() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    objects = InMemoryObjectStore("runtime")
    await store.initialize()
    directory = None
    try:
        resource = AssetKey("mcp", "server/assets/script.py")
        root = AssetKey("mcp", "server/assets")
        await store.put(resource, b"print('frozen')\n")
        helper = AssetKey("mcp", "server/assets/lib/helper.py")
        await store.put(helper, b"VALUE = 42\n")
        reference, digest = await _snapshot_mcp_resources(
            store,
            root,
            ("resource:script.py",),
            object_store=objects,
        )
        await store.delete(resource)
        await store.delete(helper)

        server = MCPServerSpec(
            "foo/bar",
            "python",
            ("resource:script.py",),
            root,
        )
        directory = await _materialize_resource_snapshot(
            server,
            _FrozenMCPResources(
                reference,
                digest,
                {"version": 1, "boundary": "host-stdio"},
            ),
            objects,
        )

        assert (Path(directory.name) / "script.py").read_bytes() == (
            b"print('frozen')\n"
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
