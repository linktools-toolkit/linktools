from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._repository_instructions import _discover_repository_overlay
from linktools.ai.spec import RepositoryInstructions
from linktools.ai.workspace import WorkspacePolicy
from linktools.ai.workspace._instructions import WorkspaceInstructionResolver


def _instructions(root: Path, scope: str, content: str) -> None:
    directory = root / scope
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "AGENTS.md").write_text(content, encoding="utf-8")


@pytest.mark.asyncio
async def test_discovery_skips_oversized_document_and_keeps_whole_fitting_nested_document(tmp_path: Path) -> None:
    _instructions(tmp_path, ".", "root\n")
    _instructions(tmp_path, "large", "x" * 8192)
    _instructions(tmp_path, "large/child", "nested 整份保留\n")
    resolver = WorkspaceInstructionResolver(
        tmp_path, WorkspacePolicy(max_repository_instruction_bytes=2048), RepositoryInstructions(),
    )
    initial = await resolver.resolve(".")
    with pytest.raises(AIError) as captured:
        await resolver.resolve("large/child/file.txt")
    assert captured.value.code is ErrorCode.PROMPT_TOO_LARGE

    overlay = await _discover_repository_overlay(resolver, initial, None, ("large/child/file.txt",))
    assert overlay is not None
    assert [(document.scope, document.content) for document in overlay.documents] == [
        ("large/child", "nested 整份保留\n"),
    ]


@pytest.mark.asyncio
async def test_discovery_shares_document_and_rendered_byte_limits_across_paths(tmp_path: Path) -> None:
    for scope in (".", "first", "second"):
        _instructions(tmp_path, scope, f"{scope} 内容\n")
    unlimited = WorkspaceInstructionResolver(tmp_path, WorkspacePolicy(), RepositoryInstructions())
    fit = await unlimited.resolve("first/file.txt")
    for policy in (
        WorkspacePolicy(max_repository_instruction_documents=2),
        WorkspacePolicy(max_repository_instruction_bytes=len(fit.render().encode("utf-8"))),
    ):
        resolver = WorkspaceInstructionResolver(tmp_path, policy, RepositoryInstructions())
        initial = await resolver.resolve(".")
        overlay = await _discover_repository_overlay(resolver, initial, None, ("first/file.txt", "second/file.txt"))
        assert overlay is not None
        assert [document.scope for document in overlay.documents] == ["first"]
        assert await _discover_repository_overlay(resolver, initial, overlay, ("second/file.txt",)) is None


@pytest.mark.asyncio
async def test_discovery_budget_cannot_hide_an_outside_root_file(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    _instructions(root, ".", "root\n")
    outside = tmp_path / "outside.md"
    outside.write_text("x" * 8192, encoding="utf-8")
    directory = root / "nested"
    directory.mkdir()
    (directory / "AGENTS.md").symlink_to(outside)
    resolver = WorkspaceInstructionResolver(
        root, WorkspacePolicy(max_repository_instruction_documents=1, max_repository_instruction_bytes=2048),
        RepositoryInstructions(),
    )
    initial = await resolver.resolve(".")
    with pytest.raises(AIError) as captured:
        await _discover_repository_overlay(resolver, initial, None, ("nested/file.txt",))
    assert captured.value.code is ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT


@pytest.mark.asyncio
async def test_discovery_skips_documents_that_do_not_fit_a_small_persisted_overlay(tmp_path: Path) -> None:
    _instructions(tmp_path, ".", "root\n")
    _instructions(tmp_path, "large", "x" * 100)
    _instructions(tmp_path, "large/nested", "nested\n")
    resolver = WorkspaceInstructionResolver(tmp_path, WorkspacePolicy(), RepositoryInstructions())
    initial = await resolver.resolve(".")
    overlay = await _discover_repository_overlay(
        resolver, initial, None, ("large/nested/file.txt",),
        fits_overlay=lambda bundle: sum(len(document.content) for document in bundle.documents) <= 20,
    )
    assert overlay is not None
    assert [document.scope for document in overlay.documents] == ["large/nested"]
