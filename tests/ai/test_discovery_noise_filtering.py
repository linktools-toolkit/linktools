#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic discovery ignores environment noise without restricting direct reads."""

import os
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
    SkillResource,
    SkillCapability,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
)
from linktools.ai.core import DEFAULT_DISCOVERY_POLICY
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import SkillSpec
from linktools.ai.storage import StorageLayer, StorageOverlay
from linktools.ai.workspace import SandboxResource, WorkspacePolicy


class _SuffixSkillPathAdapter:
    def validate(self, kinds: "Sequence[str]") -> None:
        if tuple(kinds) != ("skill",):
            raise ValueError("unexpected kinds")

    def root_path(self, kind: str) -> str:
        if kind != "skill":
            raise ValueError("unexpected kind")
        return "skills"

    def to_path(self, key: AssetKey) -> str:
        return f"skills/{key.id}.asset"

    def from_path(self, path: str) -> "AssetKey | None":
        prefix = "skills/"
        suffix = ".asset"
        if not path.startswith(prefix) or not path.endswith(suffix):
            return None
        return AssetKey("skill", path[len(prefix) : -len(suffix)])


async def _capture_skill_source_ref(
    store: AssetStore,
    source_id: str,
    root: str,
) -> SkillSourceRef:
    prefix = f"{root}/"
    metadata = await store.capture_metadata()
    selected = tuple(
        (info.key, info.key.id[len(prefix) :])
        for info in metadata
        if info.key.kind == "skill"
        and info.key.id.startswith(prefix)
        and info.key.id[len(prefix) :] != "SKILL.md"
    )
    keys = tuple(key for key, _relative in selected)
    refs = await store.resolve_versions(keys)
    paths = await store.local_paths(keys)
    versions = tuple(
        SkillResource(
            relative,
            ref,
            0 if path is None else path.stat().st_mode & 0o111,
        )
        for (_key, relative), ref, path in zip(
            selected,
            refs,
            paths,
            strict=True,
        )
    )
    return SkillSourceRef(source_id, root, versions)


@pytest.mark.asyncio
async def test_directory_declaration_assets_ignore_noise_without_restricting_ids(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".linktools"
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

    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=PrefixAssetPathAdapter(
                    {"agent": "agents", "skill": "skills", "mcp": "mcp"}
                ),
                kinds=("agent", "skill", "mcp"),
                follow_external_symlinks=True,
                ignore_paths=DEFAULT_DISCOVERY_POLICY.ignores,
            )
        )
    )
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

    source = LocalSkillSource("local", skills_root)
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
    assert await source.read(
        SkillSourceRef("local", "review"), "scripts/helper.PYO"
    ) == b"\xff"


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

        source = AssetSkillSource("virtual", store)
        source_ref = await _capture_skill_source_ref(store, "virtual", "review")
        view = await source.inspect(source_ref)

        assert view.resources == ("assets/payload.bin", "references/rules.md")
        assert await source.read(source_ref, ".hidden.md") == b"hidden"
        assert await source.read(source_ref, "scripts/helper.PYC") == b"\xff"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_directory_asset_skill_preserves_local_path_and_executable_mode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    package = root / "skills" / "review"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("skill", encoding="utf-8")
    script = package / "run.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(script, 0o755)

    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
                kinds=("skill",),
            )
        )
    )
    await store.initialize()
    try:
        source_ref = await _capture_skill_source_ref(store, "application", "review")
        source = AssetSkillSource("application", store)
        view = await source.inspect(source_ref)

        assert view.location.kind == "local"
        assert Path(view.location.path) == package.resolve()
        assert view.resources == ("run.sh",)
        assert script.stat().st_mode & 0o111 == 0o111
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sandbox_uses_original_pinned_asset_files(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    package = root / "skills" / "review"
    package.mkdir(parents=True)
    script = package / "run.sh"
    script.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    os.chmod(script, 0o755)
    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
                kinds=("skill",),
            )
        )
    )
    await store.initialize()
    try:
        source_ref = await _capture_skill_source_ref(store, "application", "review")
        files = {item.path: item.asset for item in source_ref.resources}
        modes = {
            item.path: item.executable_bits for item in source_ref.resources
        }
        resource = await SandboxResource.from_asset_versions(
            "review", store, files, executable_bits=modes
        )
        assert resource is not None
        assert resource.source == package.resolve()
        assert resource.files == {"run.sh": script.resolve()}

        script.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")
        with pytest.raises(AIError) as error:
            await SandboxResource.from_asset_versions(
                "review", store, files, executable_bits=modes
            )
        assert error.value.code in {
            ErrorCode.SNAPSHOT_CONFLICT,
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        }
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sandbox_keeps_memory_assets_virtual() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        key = AssetKey("skill", "review/run.sh")
        await store.put(key, b"#!/bin/sh\n")
        version = (await store.resolve_versions((key,)))[0]
        assert await SandboxResource.from_asset_versions(
            "review", store, {"run.sh": version}
        ) is None
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_directory_asset_skill_local_path_does_not_require_skill_markdown(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    package = root / "skills" / "review"
    (package / "scripts").mkdir(parents=True)
    (package / "manifest.yaml").write_text("name: review\n", encoding="utf-8")
    (package / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
                kinds=("skill",),
            )
        )
    )
    await store.initialize()
    try:
        source_ref = await _capture_skill_source_ref(store, "application", "review")
        source = AssetSkillSource("application", store)
        view = await source.inspect(source_ref)

        assert view.location.kind == "local"
        assert Path(view.location.path) == package.resolve()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_directory_asset_skill_with_remapped_paths_is_virtual(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    mapped = root / "skills" / "review"
    mapped.mkdir(parents=True)
    (mapped / "manifest.yaml.asset").write_text("name: review\n", encoding="utf-8")
    (mapped / "run.sh.asset").write_text("#!/bin/sh\n", encoding="utf-8")

    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=_SuffixSkillPathAdapter(),
                kinds=("skill",),
            )
        )
    )
    await store.initialize()
    try:
        source_ref = await _capture_skill_source_ref(store, "application", "review")
        source = AssetSkillSource("application", store)
        view = await source.inspect(source_ref)

        assert view.location.kind == "virtual"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_skill_uses_virtual_location_when_overlay_mixes_origins(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    package = root / "skills" / "review"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("skill", encoding="utf-8")
    (package / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")

    primary = InMemoryAssetBackend()
    await primary.put(AssetKey("skill", "review/run.sh"), b"override")
    directory = DirectoryAssetBackend(
        str(root),
        path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
        kinds=("skill",),
    )
    store = AssetStore(
        StorageOverlay(
            primary,
            layers=(StorageLayer("directory", directory),),
        )
    )
    await store.initialize()
    try:
        source_ref = await _capture_skill_source_ref(store, "application", "review")
        source = AssetSkillSource("application", store)
        view = await source.inspect(source_ref)

        assert view.location.kind == "virtual"
        assert view.resources == ("run.sh",)
        assert await source.read(source_ref, "run.sh") == b"override"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_virtual_skill_versions_ignore_unrelated_asset_changes() -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("skill", "review/references/rules.md"), b"rules")
        first = await _capture_skill_source_ref(store, "virtual", "review")

        await store.put(AssetKey("agent", "unrelated"), b"agent")
        await store.put(AssetKey("skill", "other/reference.md"), b"other")
        assert await _capture_skill_source_ref(store, "virtual", "review") == first

        await store.put(
            AssetKey("skill", "review/references/rules.md"),
            b"changed",
        )
        assert await _capture_skill_source_ref(store, "virtual", "review") != first
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_asset_skill_source_captures_executable_mode(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    package = root / "skills" / "review"
    package.mkdir(parents=True)
    script = package / "run.sh"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(script, 0o644)
    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(root),
                path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
                kinds=("skill",),
            )
        )
    )
    await store.initialize()
    try:
        first = await _capture_skill_source_ref(store, "application", "review")
        assert first.resources[0].executable_bits == 0

        os.chmod(script, 0o755)
        second = await _capture_skill_source_ref(store, "application", "review")
        assert second.resources[0].executable_bits == 0o111
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rule_catalog_ignores_hidden_and_cache_discovery_paths(tmp_path: Path) -> None:
    rules = tmp_path / "assets" / "rules"
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

    store = AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(tmp_path / "assets"),
                path_adapter=PrefixAssetPathAdapter({"rule": "rules"}),
                kinds=("rule",),
            )
        )
    )
    await store.initialize()
    try:
        capture = await CapabilityGroup("rules", assets=store).capture()
    finally:
        await store.close()

    assert tuple(document.source for document in capture.instructions.documents) == (
        "rule:active",
        "rule:nested/visible",
    )
