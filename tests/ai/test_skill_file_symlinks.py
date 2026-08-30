#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local declaration aliases support symlinks without weakening resource containment."""

from pathlib import Path

import pytest

from linktools.ai.asset import (
    AssetKey,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import CapabilityGroup, LocalSkillResourceSource
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._factory import _default_workspace_store
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecCodec,
    MCPServerSpec,
    MCPServerSpecCodec,
)
from linktools.ai.workspace import Workspace


def _symlink(target: Path, link: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"symlinks are unavailable: {error}")


@pytest.mark.asyncio
async def test_local_skill_resource_file_symlink_is_discovered_and_read(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    package = skills_root / "review"
    shared = package / "shared"
    package.mkdir(parents=True)
    shared.mkdir()
    target = shared / "guide.md"
    target.write_text("shared guide", encoding="utf-8")
    _symlink(target, package / "guide.md")

    source = LocalSkillResourceSource("local", skills_root)

    view = await source.inspect("review")

    assert view.resources == ("guide.md", "shared/guide.md")
    assert await source.read("review", "guide.md") == b"shared guide"


@pytest.mark.asyncio
async def test_local_skill_contained_directory_symlink_is_discovered(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    package = skills_root / "review"
    hidden = package / ".shared"
    package.mkdir(parents=True)
    hidden.mkdir()
    (package / "SKILL.md").write_text("skill", encoding="utf-8")
    (hidden / "guide.md").write_text("guide", encoding="utf-8")
    _symlink(hidden, package / "references", directory=True)

    source = LocalSkillResourceSource("local", skills_root)
    view = await source.inspect("review")

    assert view.resources == ("references/guide.md",)
    assert await source.read("review", "references/guide.md") == b"guide"


@pytest.mark.asyncio
async def test_local_skill_package_directory_symlink_can_target_outside_source_root(
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "claude" / "skills"
    external_package = tmp_path / "ccswitch" / "skills" / "review"
    skills_root.mkdir(parents=True)
    external_package.mkdir(parents=True)
    (external_package / "SKILL.md").write_text("skill", encoding="utf-8")
    (external_package / "guide.md").write_text("guide", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "nested.md").write_text("outside", encoding="utf-8")
    _symlink(outside, external_package / "outside-link")
    _symlink(outside_dir, external_package / "outside-dir-link", directory=True)
    _symlink(external_package, skills_root / "review", directory=True)

    source = LocalSkillResourceSource("local", skills_root)
    view = await source.inspect("review")

    assert Path(view.location.path) == external_package.resolve()
    assert view.resources == ("guide.md",)
    assert await source.read("review", "guide.md") == b"guide"
    with pytest.raises(AIError) as file_error:
        await source.read("review", "outside-link")
    assert file_error.value.code is ErrorCode.ASSET_PATH_OUTSIDE_ROOT
    with pytest.raises(AIError) as directory_error:
        await source.read("review", "outside-dir-link/nested.md")
    assert directory_error.value.code is ErrorCode.ASSET_PATH_OUTSIDE_ROOT


@pytest.mark.asyncio
async def test_local_skill_symlink_loops_use_stable_errors(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir()
    _symlink(Path("b"), skills_root / "a", directory=True)
    _symlink(Path("a"), skills_root / "b", directory=True)

    source = LocalSkillResourceSource("local", skills_root)
    with pytest.raises(AIError) as package_error:
        await source.inspect("a")
    assert package_error.value.code is ErrorCode.ASSET_NOT_FOUND

    package = skills_root / "review"
    package.mkdir()
    (package / "SKILL.md").write_text("skill", encoding="utf-8")
    _symlink(Path("loop-b"), package / "loop-a")
    _symlink(Path("loop-a"), package / "loop-b")
    _symlink(package, package / "loop-dir", directory=True)

    view = await source.inspect("review")
    assert view.resources == ()
    with pytest.raises(AIError) as resource_error:
        await source.read("review", "loop-a")
    assert resource_error.value.code is ErrorCode.ASSET_NOT_FOUND


@pytest.mark.asyncio
async def test_directory_asset_backend_does_not_follow_symlinks_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    external = tmp_path / "external"
    (root / "agents").mkdir(parents=True)
    external.mkdir()
    target = external / "review"
    target.write_text("agent", encoding="utf-8")
    _symlink(target, root / "agents" / "review")

    backend = DirectoryAssetBackend(
        str(root),
        path_adapter=PrefixAssetPathAdapter({"agent": "agents"}),
        kinds=("agent",),
    )
    await backend.initialize()
    try:
        loaded = await backend.load_metadata(None)
        assert loaded.changes == ()
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_workspace_declaration_symlinks_freeze_valid_external_declarations(
    tmp_path: Path,
) -> None:
    workspace = Workspace.load(tmp_path / "workspace")
    storage_root = workspace.storage_root
    external = tmp_path / "shared-declarations"
    external_skill = external / "skills" / "review"
    external_skill.mkdir(parents=True)
    (external_skill / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review changes\n---\n\nReview changes.\n",
        encoding="utf-8",
    )
    agent = external / "agent"
    mcp = external / "mcp"
    agent.write_bytes(AgentSpecCodec().encode(AgentSpec("review")))
    mcp.write_bytes(MCPServerSpecCodec().encode(MCPServerSpec("server", "python")))

    (storage_root / "agents").mkdir(parents=True)
    (storage_root / "mcp").mkdir()
    (storage_root / "skills").mkdir()
    _symlink(agent, storage_root / "agents" / "review")
    _symlink(mcp, storage_root / "mcp" / "server")
    _symlink(external_skill, storage_root / "skills" / "review", directory=True)
    _symlink(storage_root / "skills", storage_root / "skills" / "loop", directory=True)

    store, backend = _default_workspace_store(workspace)
    await store.initialize()
    try:
        group: CapabilityGroup[object] = CapabilityGroup.from_store(
            "workspace",
            store,
            skill_source=LocalSkillResourceSource(
                "workspace",
                storage_root / "skills",
            ),
        )
        frozen = await group.freeze()
        assert {(item.kind, item.id) for item in frozen} == {
            ("agent", "review"),
            ("mcp", "server"),
            ("skill", "review"),
        }
        assert await store.get(AssetKey("agent", "review")) == agent.read_bytes()
        assert await store.get(AssetKey("mcp", "server")) == mcp.read_bytes()
    finally:
        await store.close()
        await backend.close()
