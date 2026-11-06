#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Editable Skill discovery reads current files without replacing other sources."""

from pathlib import Path

import pytest

from linktools.ai.asset import AssetKey, AssetStore, DirectoryAssetBackend, InMemoryAssetBackend, PrefixAssetPathAdapter
from linktools.ai.capability import AssetSkillSource, CapabilityLoadContext, SkillDefinition
from linktools.ai.capability._declaration import BuiltinDeclarationLoader
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import StorageOverlay


def _declaration(content: str) -> str:
    return f"---\nname: guide\ndescription: Guide\n---\n{content}\n"


@pytest.mark.asyncio
async def test_live_skill_source_discovers_new_packages_and_reads_edits(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    root.mkdir()
    store = AssetStore(StorageOverlay(DirectoryAssetBackend(
        str(tmp_path), path_adapter=PrefixAssetPathAdapter({"skill": "skills"}), kinds=("skill",),
    )))
    loader = BuiltinDeclarationLoader("skill")

    async def discover() -> tuple[SkillDefinition, ...]:
        context = await CapabilityLoadContext.read_current("skills", store)
        definitions = await loader.load(context)
        assert all(isinstance(item, SkillDefinition) for item in definitions)
        return tuple(definitions)

    try:
        await store.initialize()
        source = AssetSkillSource("skills", store, discover=discover)
        assert source.dynamic
        assert await source.discover() == ()
        package = root / "guide"
        package.mkdir()
        (package / "SKILL.md").write_text(_declaration("Initial instructions"))
        guide = package / "guide.txt"
        guide.write_text("Initial resource")
        first, = await source.discover()
        assert "Initial instructions" in first.model_content
        assert first.source_ref is not None
        assert await source.read(first.source_ref, "guide.txt") == b"Initial resource"

        (package / "SKILL.md").write_text(_declaration("Updated instructions"))
        (package / "new.txt").write_text("New resource")
        guide.write_text("Updated resource")
        current, = await source.discover()
        assert "Updated instructions" in current.model_content
        assert current.source_ref is not None
        assert (await source.inspect(current.source_ref)).resources == ("guide.txt", "new.txt")
        assert await source.read(current.source_ref, "new.txt") == b"New resource"
        assert await source.read(current.source_ref, "guide.txt") == b"Updated resource"
        with pytest.raises(AIError) as changed:
            await source.read(first.source_ref, "guide.txt")
        assert changed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert "Initial instructions" in first.model_content
        with pytest.raises(AIError) as traversal:
            await source.read(current.source_ref, "../outside.txt")
        assert traversal.value.code is ErrorCode.REQUEST_FIELD_INVALID
        guide.unlink()
        with pytest.raises(AIError) as missing:
            await source.read(current.source_ref, "guide.txt")
        assert missing.value.code is ErrorCode.ASSET_VERSION_NOT_FOUND
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_current_skill_context_does_not_require_global_revision_to_stay_fixed(tmp_path: Path) -> None:
    package = tmp_path / "skills" / "guide"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text(_declaration("Instructions"))
    store = AssetStore(StorageOverlay(DirectoryAssetBackend(
        str(tmp_path), path_adapter=PrefixAssetPathAdapter({"skill": "skills"}), kinds=("skill",),
    )))
    try:
        await store.initialize()
        context = await CapabilityLoadContext.read_current("skills", store)
        other = tmp_path / "skills" / "other"
        other.mkdir()
        (other / "SKILL.md").write_text("---\nname: other\ndescription: Other\n---\nOther instructions")
        loaded = await BuiltinDeclarationLoader("skill").load(context)
        assert tuple(item.id for item in loaded) == ("guide",)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_current_loader_reader_uses_one_calls_versioned_inputs():
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        key = AssetKey("template", "instructions.txt")
        await store.put(key, b"First")
        context = await CapabilityLoadContext.read_current("skills", store)
        await store.put(key, b"Second")
        assert await context.asset_reader.get(key) == b"First"
        assert await context.read(key) == b"First"
        next_context = await CapabilityLoadContext.read_current("skills", store)
        assert await next_context.asset_reader.get(key) == b"Second"
    finally:
        await store.close()
