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
        b"---\nversion: 1\n---\nbody",
        b"---\nsystem-prompt: hidden\n---\nbody",
        b"---\n<<: {model: inherited}\n---\nbody",
        b"---\nvalue: !!python/object/apply:os.system ['true']\n---\nbody",
    ),
)
def test_agent_markdown_rejects_invalid_frontmatter(document: bytes) -> None:
    with pytest.raises(AIError) as error:
        AgentMarkdownSpecCodec().parse(document)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_agent_markdown_resolves_defaults_without_truthiness_or_deep_merge() -> None:
    codec = AgentMarkdownSpecCodec()
    defaults = {
        "model": "worker-model",
        "tool_retries": 7,
        "allow_tools": ["tool"],
        "planning": True,
        "usage_limits": {"model_requests": 100, "tool_calls": 50},
    }
    spec = codec.decode(
        b"---\nallow-tools: []\nplanning: false\n"
        b"usage-limits: {model-requests: 2}\n---\nbody",
        logical_id="team/worker",
        defaults=defaults,
    )

    assert spec.id == "team/worker"
    assert spec.model == "worker-model"
    assert spec.allow_tools == ()
    assert spec.planning is False
    assert spec.tool_retries == 7
    assert spec.usage_limits is not None
    assert spec.usage_limits.model_requests == 2
    assert spec.usage_limits.tool_calls is None

    closed_limits = codec.from_payload(
        {"system_prompt": "", "usage_limits": None},
        logical_id="worker",
        defaults={"usage_limits": {"tool_calls": 10}},
    )
    assert closed_limits.usage_limits is None


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
    assert (
        CapabilityContribution.from_declaration(spec).fingerprint
        == CapabilityContribution.from_declaration(changed_metadata).fingerprint
    )
    assert (
        CapabilityContribution.from_declaration(spec).fingerprint
        != CapabilityContribution.from_declaration(changed_prompt).fingerprint
    )


def test_agent_markdown_rejects_metadata_that_is_not_a_json_map() -> None:
    with pytest.raises(AIError) as error:
        AgentMarkdownSpecCodec().decode(
            b"---\nmetadata: [author, Mei]\n---\nPrompt.\n",
            logical_id="worker",
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


def test_agent_markdown_rejects_version_unknown_fields_and_invalid_defaults() -> None:
    codec = AgentMarkdownSpecCodec()
    with pytest.raises(AIError) as version:
        codec.from_payload(
            {"system_prompt": "", "version": 1},
            logical_id="worker",
        )
    assert version.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    with pytest.raises(AIError) as unknown:
        codec.from_payload(
            {"system_prompt": "", "future_field": True},
            logical_id="worker",
        )
    assert unknown.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID

    with pytest.raises(AIError) as defaults:
        codec.from_payload(
            {"system_prompt": "", "model": "explicit"},
            logical_id="worker",
            defaults={"future_field": True},
        )
    assert defaults.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


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

    snapshot = await group.snapshot()

    assert len(snapshot.contributions) == 1
    spec = snapshot.contributions[0].value
    assert isinstance(spec, AgentSpec)
    assert spec.id == "team"
    assert spec.model == "configured"
    assert spec.tool_retries == 4
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

    snapshot = await group.snapshot()

    assert [item.id for item in snapshot.contributions] == ["security/audit"]
    assert isinstance(snapshot.contributions[0].value, AgentSpec)
    await store.close()


@pytest.mark.asyncio
async def test_snapshot_ignores_unrelated_asset_added_during_loader() -> None:
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
        snapshot = await group.snapshot()
        reader = snapshot.asset_reader
        assert reader is not None
        assert late not in {
            info.key
            for info in await reader.metadata_snapshot()
        }
        assert snapshot.source_revision == await store.current_revision()
        await snapshot.verify_source_revision()
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
        snapshot = await CapabilityGroup("application", assets=store).snapshot()
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
        AssetKey("skill", "audit"): SkillSpecCodec().encode(
            SkillSpec("audit", "audit instructions")
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
            snapshot = await CapabilityGroup("assets", assets=store).snapshot()
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
        assert mcp.resource_root == AssetKey("mcp", "security/audit")
        assert results[0][2][:2] == ("skill", "audit")
    finally:
        for store, _engine in stores:
            await store.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_explicit_mcp_resource_root_reserves_declaration_filenames() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        server = MCPServerSpec(
            "server",
            "python",
            ("resource:mcp.json",),
            AssetKey("mcp", "server/assets"),
        )
        await store.put(
            AssetKey("mcp", "server"),
            MCPServerSpecCodec().encode(server),
        )
        await store.put(
            AssetKey("mcp", "server/assets/mcp.json"),
            b"resource-like declaration",
        )

        with pytest.raises(AIError) as error:
            await CapabilityGroup("application", assets=store).snapshot()

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
        await CapabilityGroup("application", assets=store).snapshot()
    assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT
    await store.close()
