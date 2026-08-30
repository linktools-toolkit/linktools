#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic discovery ignores environment noise without restricting direct reads."""

from pathlib import Path

import pytest

from linktools.ai.asset import (
    AssetKey,
    AssetStore,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import (
    AssetSkillResourceSource,
    LocalSkillResourceSource,
    SkillCapability,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
)
from linktools.ai.runtime._factory import _default_workspace_store
from linktools.ai.spec import SkillSpec
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import LocalRuleCatalog, Workspace, WorkspacePolicy


@pytest.mark.asyncio
async def test_workspace_declaration_discovery_ignores_noise_without_restricting_ids(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path)
    root = workspace.storage_root
    (root / "agents" / "__pycache__").mkdir(parents=True)
    (root / "agents" / "nested").mkdir()
    (root / "mcp" / ".cache").mkdir(parents=True)
    (root / "mcp" / "nested").mkdir()
    (root / "skills" / "review").mkdir(parents=True)

    (root / "agents" / "review").write_text("agent", encoding="utf-8")
    (root / "agents" / ".DS_Store").write_bytes(b"noise")
    (root / "agents" / "__pycache__" / "review.PYC").write_bytes(b"\xff")
    (root / "agents" / "nested" / "worker").write_text("nested", encoding="utf-8")
    (root / "mcp" / "server").write_text("mcp", encoding="utf-8")
    (root / "mcp" / ".cache" / "ignored").write_text("hidden", encoding="utf-8")
    (root / "mcp" / "nested" / "server").write_text("nested", encoding="utf-8")
    (root / "skills" / "review" / "SKILL.md").write_text("skill", encoding="utf-8")
    (root / "skills" / "Thumbs.DB").write_bytes(b"noise")

    store, backend = _default_workspace_store(workspace)
    await store.initialize()
    try:
        page = await store.list_info(limit=200)
        assert page.next_cursor is None
        assert {item.key for item in page.items} == {
            AssetKey("agent", "review"),
            AssetKey("agent", "nested/worker"),
            AssetKey("mcp", "server"),
            AssetKey("mcp", "nested/server"),
            AssetKey("skill", "review/SKILL.md"),
        }
    finally:
        await store.close()
        await backend.close()


@pytest.mark.asyncio
async def test_directory_asset_ignore_prunes_before_following_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    external = tmp_path / "external"
    (root / "agents").mkdir(parents=True)
    (external / "nested").mkdir(parents=True)
    (external / "nested" / "ignored").write_text("ignored", encoding="utf-8")
    try:
        (root / "agents" / ".cache").symlink_to(external, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")

    visited: list[str] = []

    def ignore(path: str) -> bool:
        visited.append(path)
        return path.startswith(".")

    backend = DirectoryAssetBackend(
        str(root),
        path_adapter=PrefixAssetPathAdapter({"agent": "agents"}),
        kinds=("agent",),
        follow_external_symlinks=True,
        ignore_paths=ignore,
    )
    await backend.initialize()
    try:
        loaded = await backend.load_metadata(None)
        assert loaded.changes == ()
        assert ".cache" in visited
        assert not any(path.startswith(".cache/") for path in visited)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_local_skill_resource_discovery_ignores_noise_but_explicit_read_works(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    package = skills_root / "review"
    (package / "references").mkdir(parents=True)
    (package / "scripts" / "__pycache__").mkdir(parents=True)
    (package / "assets").mkdir()
    (package / "SKILL.md").write_text("skill", encoding="utf-8")
    (package / "references" / "rules.md").write_text("rules", encoding="utf-8")
    (package / "assets" / "payload.bin").write_bytes(b"\xff\xfe")
    (package / ".hidden.md").write_text("hidden", encoding="utf-8")
    (package / "scripts" / "__pycache__" / "helper.pyc").write_bytes(b"\xff")
    (package / "scripts" / "helper.PYO").write_bytes(b"\xff")
    (package / "Thumbs.DB").write_bytes(b"noise")
    (package / "__MACOSX").mkdir()
    (package / "__MACOSX" / "metadata").write_bytes(b"noise")

    source = LocalSkillResourceSource("local", skills_root)
    capability = SkillCapability(
        (
            SkillDefinition(
                SkillSpec("review", "pinned"),
                SkillSourceRef("local", "review"),
            ),
        ),
        SkillSourceRegistry((source,)),
    )
    root = await capability.load_skill("review")

    assert root["resources"] == ["assets/payload.bin", "references/rules.md"]
    assert await capability.load_skill("review", ".hidden.md") == {
        "id": "review",
        "path": ".hidden.md",
        "content": "hidden",
    }
    assert await source.read("review", "scripts/helper.PYO") == b"\xff"


@pytest.mark.asyncio
async def test_virtual_skill_resource_discovery_applies_the_same_noise_policy() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("skill", "review/SKILL.md"), b"skill")
        await store.put(AssetKey("skill", "review/references/rules.md"), b"rules")
        await store.put(AssetKey("skill", "review/assets/payload.bin"), b"\xff\xfe")
        await store.put(AssetKey("skill", "review/.hidden.md"), b"hidden")
        await store.put(AssetKey("skill", "review/__pycache__/helper.pyc"), b"\xff")
        await store.put(AssetKey("skill", "review/scripts/helper.PYC"), b"\xff")
        await store.put(AssetKey("skill", "review/Desktop.INI"), b"noise")
        await store.put(AssetKey("skill", "review/__MACOSX/metadata"), b"noise")

        source = AssetSkillResourceSource("virtual", store)
        view = await source.inspect("review")

        assert view.resources == ("assets/payload.bin", "references/rules.md")
        assert await source.read("review", ".hidden.md") == b"hidden"
        assert await source.read("review", "scripts/helper.PYC") == b"\xff"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rule_catalog_ignores_hidden_and_cache_discovery_paths(tmp_path: Path) -> None:
    rules = tmp_path / ".linktools" / "rules"
    (rules / "nested").mkdir(parents=True)
    (rules / ".cache").mkdir()
    (rules / "__pycache__").mkdir()
    (rules / "__MACOSX").mkdir()
    (rules / "active.md").write_text("active", encoding="utf-8")
    (rules / "nested" / "visible.md").write_text("visible", encoding="utf-8")
    (rules / ".hidden.md").write_bytes(b"\xff")
    (rules / ".cache" / "invalid.md").write_bytes(b"\xff")
    (rules / "__pycache__" / "invalid.md").write_bytes(b"\xff")
    (rules / "__MACOSX" / "invalid.md").write_bytes(b"\xff")

    catalog = await LocalRuleCatalog.load(tmp_path, WorkspacePolicy())

    assert tuple(document.source for document in catalog.documents) == (
        "rule:active",
        "rule:nested/visible",
    )
