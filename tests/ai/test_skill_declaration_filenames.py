#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill filename aliases preserve package identity and resource contracts."""

from collections.abc import Sequence
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
    AssetSkillSource,
    CapabilityGroup,
    LocalSkillSource,
    SkillCapability,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.storage import InMemoryObjectStore, StorageOverlay

from .test_asset_backend_conformance import asset_store as asset_store


_DECLARATION = b"---\nname: review\ndescription: Review code.\n---\nOriginal instructions."
_ROOT = "Team/review"


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ("SKILL.md", "skill.md"))
async def test_skill_aliases_capture_exact_keys_and_frozen_resources(
    asset_store: AssetStore, filename: str,
) -> None:
    declaration = AssetKey("skill", f"{_ROOT}/{filename}")
    resource = AssetKey("skill", f"{_ROOT}/Guide.md")
    await asset_store.put(declaration, _DECLARATION)
    await asset_store.put(resource, b"original guide")
    capture = await CapabilityGroup("skills", assets=asset_store).capture()
    definition = capture.contributions[0].value
    assert isinstance(definition, SkillDefinition)
    assert definition.id == _ROOT
    assert definition.source_ref is not None
    assert [item.asset.key for item in definition.source_ref.resources] == [resource]
    reader = capture.asset_reader
    assert reader is not None
    assert (await reader.resolve_versions((declaration,)))[0].key == declaration

    objects = InMemoryObjectStore()
    snapshot = await asset_store.snapshot((declaration, resource), object_store=objects)
    await asset_store.put(declaration, _DECLARATION.replace(b"Original", b"Changed"))
    await asset_store.put(resource, b"changed guide")
    skill = SkillCapability(
        (SkillDefinition.from_contract(definition.contract),),
        SkillSourceRegistry((AssetSkillSource("skills", reader),)),
    )
    loaded = await skill.load_skill(_ROOT)
    assert loaded["resources"] == ["Guide.md"]
    assert "Original instructions." in loaded["instructions"]
    assert (await skill.load_skill(_ROOT, "Guide.md"))["content"] == "original guide"
    for reserved in ("SKILL.md", "skill.md"):
        with pytest.raises(AIError) as error:
            await skill.load_skill(_ROOT, reserved)
        assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID
    with pytest.raises(AIError) as error:
        await skill.load_skill(_ROOT, "guide.md")
    assert error.value.code is ErrorCode.ASSET_NOT_FOUND

    restored = AssetStore.from_snapshot(snapshot, object_store=objects)
    await restored.initialize()
    try:
        restored_capture = await CapabilityGroup("skills", assets=restored).capture()
        restored_definition = restored_capture.contributions[0].value
        assert isinstance(restored_definition, SkillDefinition)
        assert restored_definition.spec == definition.spec
        assert await restored.get(declaration) == _DECLARATION
    finally:
        await restored.close()


class _SuffixedSkillPathAdapter:
    def validate(self, kinds: Sequence[str]) -> None:
        assert tuple(kinds) == ("skill",)

    def root_path(self, kind: str) -> str:
        assert kind == "skill"
        return "imported"

    def to_path(self, key: AssetKey) -> str:
        return f"imported/{key.id}.asset"

    def from_path(self, path: str) -> AssetKey | None:
        if path.startswith("imported/") and path.endswith(".asset"):
            return AssetKey("skill", path[len("imported/"):-len(".asset")])
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ("SKILL.md", "skill.md"))
@pytest.mark.parametrize("mapped", (False, True), ids=("prefix", "mapped"))
async def test_directory_skill_aliases_use_logical_declaration_paths(
    tmp_path: Path, filename: str, mapped: bool,
) -> None:
    adapter = _SuffixedSkillPathAdapter() if mapped else PrefixAssetPathAdapter({"skill": "imported"})
    for key, content in (
        (AssetKey("skill", f"{_ROOT}/{filename}"), _DECLARATION),
        (AssetKey("skill", f"{_ROOT}/Guide.md"), b"guide"),
    ):
        path = tmp_path / adapter.to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    backend = DirectoryAssetBackend(str(tmp_path), path_adapter=adapter, kinds=("skill",))
    store = AssetStore(StorageOverlay(backend))
    await store.initialize()
    try:
        capture = await CapabilityGroup("skills", assets=store).capture()
        definition = capture.contributions[0].value
        assert isinstance(definition, SkillDefinition)
        assert definition.source_ref is not None
        reader = capture.asset_reader
        assert reader is not None
        assert await reader.get(AssetKey("skill", f"{_ROOT}/{filename}")) == _DECLARATION
        source = AssetSkillSource("skills", reader)
        view = await source.inspect(definition.source_ref)
        assert view.resources == ("Guide.md",)
        assert view.location.kind == ("virtual" if mapped else "local")
        assert await source.read(definition.source_ref, "Guide.md") == b"guide"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_local_skill_source_excludes_both_root_declaration_names(tmp_path: Path) -> None:
    package = tmp_path / "review"
    package.mkdir()
    for filename in ("SKILL.md", "skill.md", "Guide.md"):
        (package / filename).write_text("content", encoding="utf-8")
    source = LocalSkillSource("local", tmp_path)
    view = await source.inspect(SkillSourceRef("local", "review"))
    assert view.resources == ("Guide.md",)


@pytest.mark.asyncio
@pytest.mark.parametrize("second", ("review/skill.md", "review/nested/skill.md"))
async def test_skill_aliases_preserve_ambiguous_package_rejection(second: str) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("skill", "review/SKILL.md"), _DECLARATION)
        await store.put(AssetKey("skill", second), _DECLARATION)
        with pytest.raises(AIError) as error:
            await CapabilityGroup("skills", assets=store).capture()
        assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_filename_aliases_do_not_case_fold_other_declarations_or_flat_paths() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        for kind, identifier in (
            ("skill", "review/Skill.md"),
            ("skill", "other/SKILL.MD"),
            ("skill", "skill.md"),
            ("agent", "review/agent.md"),
            ("mcp", "review/MCP.json"),
            ("mcp", "other/mcp.YAML"),
            ("rule", "review.MD"),
        ):
            await store.put(AssetKey(kind, identifier), b"not a declaration")
        capture = await CapabilityGroup("skills", assets=store).capture()
        assert capture.contributions == ()
        assert capture.instructions.documents == ()
    finally:
        await store.close()
