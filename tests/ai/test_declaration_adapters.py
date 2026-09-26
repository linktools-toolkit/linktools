#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authoring adapters and declaration discovery contracts."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import (
    AssetKey,
    AssetStore,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
    SqlAssetBackend,
)
from linktools.ai.capability import (
    AgentDeclarationLoader,
    CapabilityContribution,
    CapabilityGroup,
    CapabilityLoadContext,
    SkillDefinition,
    SkillResourceVersion,
    SkillSourceRef,
)
from linktools.ai.core import validate_logical_id as validate_core_logical_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.spec import (
    AgentMarkdownSpecCodec,
    AgentSpec,
    AgentSpecCodec,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillMarkdownSpecCodec,
    SkillSpec,
    SkillSpecCodec,
    SubagentRef,
    validate_logical_id as validate_spec_logical_id,
)
from linktools.ai.storage import StorageOverlay


@pytest.mark.parametrize(
    "logical_id",
    (
        "a" * 63,
        "a" * 64,
        "界" * 64,
        "security/audit",
        "é",
        "e\u0301",
    ),
)
def test_logical_ids_are_shared_and_preserved_without_normalization(
    logical_id: str,
) -> None:
    assert validate_core_logical_id(logical_id) == logical_id
    assert validate_spec_logical_id(logical_id) == logical_id
    assert AgentSpec(logical_id).id == logical_id
    assert SkillSpec(logical_id, "skill").id == logical_id
    assert SubagentRef("agent", logical_id).id == logical_id
    assert AgentSpecCodec().decode(
        AgentSpecCodec().encode(AgentSpec(logical_id))
    ).id == logical_id


@pytest.mark.parametrize(
    "logical_id",
    (
        "",
        "a" * 65,
        "/absolute",
        "trailing/",
        "double//segment",
        "dot/./segment",
        "dotdot/../segment",
        "back\\slash",
        "wild*card",
        "control\ncharacter",
        "unpaired\ud800",
    ),
)
def test_logical_id_grammar_rejects_invalid_values_consistently(
    logical_id: str,
) -> None:
    with pytest.raises(ValueError):
        validate_core_logical_id(logical_id)
    with pytest.raises(ValueError):
        validate_spec_logical_id(logical_id)
    with pytest.raises(ValueError):
        AgentSpec(logical_id)
    with pytest.raises(ValueError):
        SkillSpec(logical_id, "skill")
    with pytest.raises(AIError) as error:
        SubagentRef("agent", logical_id)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_agent_markdown_preserves_plain_prompt_bom_and_crlf_body() -> None:
    codec = AgentMarkdownSpecCodec()

    assert codec.parse(b"") == {"system_prompt": ""}
    plain = b"\xef\xbb\xbf---\nnot frontmatter\n"
    assert codec.parse(plain) == {"system_prompt": plain.decode("utf-8")}
    assert codec.parse(b"prompt\n---\n") == {
        "system_prompt": "prompt\n---\n"
    }

    payload = codec.parse(
        b"---\r\nmodel: test\r\nplanning: false\r\n---\r\n\r\nbody \r\n"
    )
    assert payload == {
        "model": "test",
        "planning": False,
        "system_prompt": "\r\nbody \r\n",
    }
    assert "version" not in payload


@pytest.mark.parametrize(
    "document",
    (
        b"---\nmodel: test\n",
        b"---\nnull\n---\nbody",
        b"---\nmodel: first\nmodel: second\n---\nbody",
        b"---\nallow_tools: []\nallow-tools: []\n---\nbody",
        b"---\n<<: {model: inherited}\n---\nbody",
        b"---\nvalue: !!python/object/apply:os.system ['true']\n---\nbody",
    ),
)
def test_agent_markdown_rejects_invalid_frontmatter(document: bytes) -> None:
    with pytest.raises(AIError) as error:
        AgentMarkdownSpecCodec().parse(document)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_agent_markdown_resolves_only_stable_author_defaults() -> None:
    codec = AgentMarkdownSpecCodec()
    defaults = {
        "model": "worker-model",
        "allow_tools": ["tool"],
        "planning": True,
        "tool_retries": 7,
        "usage_limits": {"model_requests": 100},
    }
    spec = codec.decode(
        b"---\nallow-tools: []\nplanning: true\n"
        b"tool-retries: 1\nusage-limits: {model-requests: 2}\n---\nbody",
        logical_id="team/worker",
        defaults=defaults,
    )

    assert spec.id == "team/worker"
    assert spec.model == "worker-model"
    assert spec.allow_tools == ()
    assert spec.planning is False
    assert spec.tool_retries == AgentSpec.DEFAULT_TOOL_RETRIES
    assert spec.usage_limits is None


def test_skill_markdown_ignores_unrelated_frontmatter_fields() -> None:
    spec = SkillMarkdownSpecCodec().decode(
        b"---\nname: review\ndescription: Review changes.\n"
        b"compatibility: [future]\nlicense: {future: true}\n"
        b"allowed-tools: [future]\ncustom-field: {nested: true}\n"
        b"---\nReview changes.\n"
    )

    assert spec.id == "review"
    assert spec.description == "Review changes."


def test_agent_markdown_metadata_round_trips_without_changing_identity() -> None:
    document = (
        b"---\nmetadata:\n  author: Mei\n  version: 2\n"
        b"  flags: [true, null, 1.5]\n  options: {enabled: false}\n"
        b"---\nReview requests.\n"
    )
    codec = AgentMarkdownSpecCodec()
    spec = codec.decode(document, logical_id="worker")
    expected = {
        "author": "Mei",
        "version": 2,
        "flags": [True, None, 1.5],
        "options": {"enabled": False},
    }
    assert dict(spec.metadata) == expected
    flags = spec.metadata["flags"]
    assert isinstance(flags, list)
    flags.append("changed")
    assert dict(spec.metadata) == expected

    wire = AgentSpecCodec().encode(spec)
    assert AgentSpecCodec().decode(wire) == spec
    assert json.loads(wire)["metadata"] == expected
    changed_metadata = codec.decode(
        document.replace(b"version: 2", b"version: 3"),
        logical_id="worker",
    )
    changed_prompt = codec.decode(
        document.replace(b"Review requests.", b"Review carefully."),
        logical_id="worker",
    )
    identity = CapabilityContribution.from_declaration(spec)
    metadata_identity = CapabilityContribution.from_declaration(changed_metadata)
    prompt_identity = CapabilityContribution.from_declaration(changed_prompt)
    assert (identity.kind, identity.id, identity.revision) == (
        metadata_identity.kind,
        metadata_identity.id,
        metadata_identity.revision,
    )
    assert (identity.kind, identity.id, identity.revision) == (
        prompt_identity.kind,
        prompt_identity.id,
        prompt_identity.revision,
    )


def test_agent_markdown_rejects_metadata_that_is_not_a_json_map() -> None:
    with pytest.raises(AIError) as error:
        AgentMarkdownSpecCodec().decode(
            b"---\nmetadata: [author, Mei]\n---\nPrompt.\n",
            logical_id="worker",
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_agent_markdown_ignores_unrelated_author_fields() -> None:
    codec = AgentMarkdownSpecCodec()

    versioned = codec.from_payload(
        {"system_prompt": "", "version": 1, "future_field": True},
        logical_id="worker",
    )
    assert versioned == AgentSpec("worker")

    explicit = codec.from_payload(
        {
            "system_prompt": "",
            "model": "explicit",
            "planning": True,
            "future_field": {"future": True},
        },
        logical_id="worker",
        defaults={
            "model": "configured",
            "tool_retries": 1,
            "future_default": True,
        },
    )
    assert explicit.model == "explicit"
    assert explicit.planning is False
    assert explicit.tool_retries == AgentSpec.DEFAULT_TOOL_RETRIES


@pytest.mark.asyncio
async def test_agent_declaration_loader_freezes_custom_kind_defaults() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    defaults = {"model": "configured", "tool_retries": 4}
    loader = AgentDeclarationLoader("worker", defaults)
    defaults["model"] = "changed"
    defaults["tool_retries"] = 40
    await store.put(
        AssetKey("worker", "team/AGENT.md"),
        b"---\n{}\n---\nworker prompt",
    )
    group = CapabilityGroup("application", assets=store)
    group.loader("worker", loader)

    snapshot = await group.capture()

    assert len(snapshot.contributions) == 1
    spec = snapshot.contributions[0].value
    assert isinstance(spec, AgentSpec)
    assert spec.id == "team"
    assert spec.model == "configured"
    assert spec.tool_retries == AgentSpec.DEFAULT_TOOL_RETRIES
    await store.close()


@pytest.mark.asyncio
async def test_custom_agent_loader_consumes_business_fields_with_public_parser() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    await store.put(
        AssetKey("worker", "security/audit/AGENT.md"),
        b"---\nmodel: test\nworker-mode: isolated\n---\nworker prompt",
    )
    codec = AgentMarkdownSpecCodec()

    class WorkerLoader:
        source_kind = "worker"

        async def load(
            self,
            context: CapabilityLoadContext,
        ) -> "Sequence[AgentSpec]":
            entry = context.list(kind=self.source_kind)[0]
            payload = codec.parse(await context.read(entry.key))
            assert payload.pop("worker-mode") == "isolated"
            return (
                codec.from_payload(
                    payload,
                    logical_id="security/audit",
                ),
            )

    group = CapabilityGroup("application", assets=store)
    group.loader("worker", WorkerLoader())

    snapshot = await group.capture()

    assert [item.id for item in snapshot.contributions] == ["security/audit"]
    assert isinstance(snapshot.contributions[0].value, AgentSpec)
    await store.close()


@pytest.mark.asyncio
async def test_custom_skill_loader_keeps_captured_resource_versions() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    resource = AssetKey("worker", "audit/helper.py")
    await store.put(resource, b"print('audit')\n")

    class WorkerLoader:
        source_kind = "worker"

        async def load(
            self,
            context: CapabilityLoadContext,
        ) -> "Sequence[SkillDefinition]":
            (ref,) = context.bind_versions((resource,))
            return (
                SkillDefinition(
                    SkillSpec("audit", "audit instructions"),
                    SkillSourceRef(
                        context.group_id,
                        "audit",
                        (SkillResourceVersion("helper.py", ref),),
                    ),
                ),
            )

    group = CapabilityGroup("application", assets=store)
    group.loader("worker", WorkerLoader())
    try:
        capture = await group.capture()
        assert len(capture.contributions) == 1
        skill = capture.contributions[0].value
        assert isinstance(skill, SkillDefinition)
        assert skill.source_ref is not None
        assert len(skill.source_ref.resource_versions) == 1
        reader = capture.asset_reader
        assert reader is not None
        ref = skill.source_ref.resource_versions[0].asset
        assert await reader.read_versions((ref,)) == (b"print('audit')\n",)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_custom_mcp_loader_binds_resource_versions() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    resource = AssetKey("worker", "server/script.py")
    await store.put(resource, b"original")

    class WorkerLoader:
        source_kind = "worker"

        async def load(
            self,
            _context: CapabilityLoadContext,
        ) -> "Sequence[MCPServerSpec]":
            return (
                MCPServerSpec(
                    "server",
                    "python",
                    ("resource:script.py",),
                    AssetKey("worker", "server"),
                ),
            )

    group = CapabilityGroup("application", assets=store)
    group.loader("worker", WorkerLoader())
    try:
        capture = await group.capture()
        assert len(capture.contributions) == 1
        contribution = capture.contributions[0]
        assert contribution.kind == "mcp"
        server = contribution.value
        assert isinstance(server, MCPServerSpec)
        versions = MCPServerSpecCodec().decode_execution_payload(
            contribution.contract,
            declaration=server,
        )
        assert server.resource == AssetKey("worker", "server")
        assert contribution.contract["asset_source_id"] == "application"
        assert versions is not None
        assert len(versions) == 1

        await store.put(resource, b"changed")
        reader = capture.asset_reader
        assert reader is not None
        assert await reader.read_versions(versions) == (b"original",)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_capture_rejects_asset_change_during_loader() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    declaration = AssetKey("worker", "team/AGENT.md")
    late = AssetKey("audit", "late")
    await store.put(declaration, b"worker")

    class WorkerLoader:
        source_kind = "worker"

        async def load(
            self,
            context: CapabilityLoadContext,
        ) -> "Sequence[AgentSpec]":
            assert await context.read(declaration) == b"worker"
            await store.put(late, b"late")
            return (AgentSpec("team"),)

    group = CapabilityGroup("application", assets=store)
    group.loader("worker", WorkerLoader())
    try:
        with pytest.raises(AIError) as error:
            await group.capture()
        assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_capture_reports_directory_file_change_as_snapshot_conflict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "captured-assets"
    target = root / "worker" / "team"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"first")
    backend = DirectoryAssetBackend(str(root), kinds=("worker",))
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    try:
        context = await CapabilityLoadContext.capture("application", store)
        target.write_bytes(b"second")

        with pytest.raises(AIError) as error:
            await context.read(AssetKey("worker", "team"))
        assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_snapshot_local_paths_ignore_unrelated_directory_revision(
    tmp_path: Path,
) -> None:
    root = tmp_path / "snapshot-assets"
    key = AssetKey("audit", "value.bin")
    target = root / "audit" / "value.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"value")
    backend = DirectoryAssetBackend(str(root), kinds=("audit",))
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    try:
        snapshot = await CapabilityGroup("application", assets=store).capture()
        reader = snapshot.asset_reader
        assert reader is not None
        refs = await reader.resolve_versions((key,))

        (root / "audit" / "late.bin").write_bytes(b"late")
        await store.current_revision()

        assert await reader.local_paths((key,)) == (target.resolve(),)
        assert await reader.read_versions(refs) == (b"value",)
        with pytest.raises(AIError) as error:
            await snapshot.verify_source_revision()
        assert error.value.code is ErrorCode.SNAPSHOT_CONFLICT
    finally:
        await store.close()


def test_agent_json_and_markdown_and_mcp_json_and_yaml_converge() -> None:
    agent_json = AgentSpecCodec().decode(
        json.dumps(
            {
                "version": 1,
                "id": "security/audit",
                "model": "test",
                "system_prompt": "prompt",
                "allow_tools": ["lookup"],
            }
        ).encode()
    )
    agent_markdown = AgentMarkdownSpecCodec().decode(
        b"---\nmodel: test\nallow-tools: [lookup]\n---\nprompt",
        logical_id="security/audit",
    )
    assert agent_json == agent_markdown

    mcp_json = b'{"version":1,"id":"server","command":"python",' b'"args":["-m","server"]}'
    mcp_yaml = b"version: 1\nid: server\ncommand: python\nargs: [-m, server]\n"
    codec = MCPServerSpecCodec()
    assert codec.decode_author(mcp_json, format="json") == codec.decode_author(
        mcp_yaml,
        format="yaml",
    )


def _declarations() -> dict[AssetKey, bytes]:
    return {
        AssetKey("agent", "security/audit/AGENT.md"): (
            b"---\r\nmodel: test\r\n---\r\naudit prompt"
        ),
        AssetKey("agent", "security/audit/notes.bin"): b"\x00\xff",
        AssetKey("mcp", "security/audit/mcp.yaml"): (
            b"version: 1\ncommand: python\nargs: [resource:data.bin]\n"
        ),
        AssetKey("mcp", "security/audit/data.bin"): b"\x00\xff",
        AssetKey("skill", "audit/SKILL.md"): (
            b"---\nname: audit\ndescription: Audit changes.\n---\n"
            b"audit instructions"
        ),
    }


@pytest.mark.asyncio
async def test_declaration_loaders_are_backend_agnostic(
    tmp_path: Path,
) -> None:
    values = _declarations()
    stores: list[tuple[AssetStore, object | None]] = []

    memory = InMemoryAssetBackend()
    memory_store = AssetStore(StorageOverlay(memory, writer=memory))
    await memory_store.initialize()
    for key, payload in values.items():
        await memory_store.put(key, payload)
    stores.append((memory_store, None))

    root = tmp_path / "directory-assets"
    prefixes = {"agent": "agents", "mcp": "mcp", "skill": "skills"}
    for key, payload in values.items():
        target = root / prefixes[key.kind] / key.id
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    directory = DirectoryAssetBackend(
        str(root),
        path_adapter=PrefixAssetPathAdapter(prefixes),
        kinds=tuple(prefixes),
    )
    directory_store = AssetStore(StorageOverlay(directory))
    await directory_store.initialize()
    stores.append((directory_store, None))

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    sql = SqlAssetBackend(engine, namespace="declarations")
    sql_store = AssetStore(StorageOverlay(sql, writer=sql))
    await sql_store.initialize()
    for key, payload in values.items():
        await sql_store.put(key, payload)
    stores.append((sql_store, engine))

    try:
        results = []
        for store, _engine in stores:
            snapshot = await CapabilityGroup("assets", assets=store).capture()
            results.append(
                tuple(
                    (item.kind, item.id, item.value)
                    for item in snapshot.contributions
                )
            )
        assert results[0] == results[1] == results[2]
        assert results[0][0][:2] == ("agent", "security/audit")
        mcp = results[0][1][2]
        assert isinstance(mcp, MCPServerSpec)
        assert mcp.resource == AssetKey("mcp", "security/audit")
        assert results[0][2][:2] == ("skill", "audit")
    finally:
        for store, _engine in stores:
            await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_shared_mcp_manifest_loads_multiple_servers() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(
            AssetKey("mcp", "shared.json"),
            json.dumps(
                {
                    "mcpServers": {
                        "local": {"command": "python"},
                        "remote": {
                            "type": "http",
                            "url": "https://example.test/mcp",
                        },
                    }
                }
            ).encode(),
        )

        capture = await CapabilityGroup("application", assets=store).capture()
        servers = {
            item.id: item.value
            for item in capture.contributions
            if item.kind == "mcp"
        }

        assert tuple(sorted(servers)) == ("local", "remote")
        assert isinstance(servers["local"], MCPServerSpec)
        assert isinstance(servers["remote"], MCPServerSpec)
        assert servers["local"].transport == "stdio"
        assert servers["remote"].transport == "streamable-http"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_mcp_package_resource_reserves_declaration_filename() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(
            AssetKey("mcp", "server/mcp.json"),
            b'{"command":"python","args":["resource:mcp.json"]}',
        )
        with pytest.raises(AIError) as error:
            await CapabilityGroup("application", assets=store).capture()
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_builtin_loader_rejects_multiple_mcp_package_declarations() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    for name in ("mcp.json", "mcp.yaml"):
        await store.put(
            AssetKey("mcp", f"server/{name}"),
            b"version: 1\ncommand: python\n",
        )
    with pytest.raises(AIError) as error:
        await CapabilityGroup("application", assets=store).capture()
    assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT
    await store.close()
