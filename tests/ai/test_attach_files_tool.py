#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace attach_files tool contracts."""

from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import workspace_capabilities, workspace_tool_contributions
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.workspace import Workspace


class _AttachmentSession:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values
        self.canonicalized: list[str] = []
        self.reads: list[str] = []

    async def canonicalize_path(self, path: str) -> str:
        self.canonicalized.append(path)
        return path

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.reads.append(path)
        try:
            value = self.values[path]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
        if max_bytes is not None and len(value) > max_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return value

    async def close(self) -> None:
        return None


class _RepositoryBoundary:
    def __init__(self) -> None:
        self.path_fields: tuple[str, ...] | None = None

    def render(self) -> str:
        return ""

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, object],
        path_fields: tuple[str, ...],
    ) -> None:
        del tool_name, tool_call_id, arguments
        self.path_fields = path_fields


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


async def _boundary(
    workspace: Workspace,
    session: _AttachmentSession,
    repository: _RepositoryBoundary | None = None,
) -> tuple[RuntimeToolBoundaryToolset, object]:
    capability = workspace_capabilities(
        workspace,
        ("attach_files",),
        session=session,  # type: ignore[arg-type]
    )[0]
    boundary = RuntimeToolBoundaryToolset(
        (capability.get_toolset(),),
        {
            "attach_files": ManagedToolDescriptor(
                effect_owner="none",
                effect="none",
                tool_class="filesystem.read",
                workspace_path_fields=("path",),
            )
        },
        id="workspace-boundary",
        workspace_policy=workspace.policy.tool_permissions,
        sandbox_session=session,  # type: ignore[arg-type]
        repository_boundary=repository,
    )
    tools = await boundary.get_tools(_context())  # type: ignore[arg-type]
    return boundary, tools["attach_files"]


def test_attach_files_declares_multi_path_workspace_metadata(tmp_path: Path) -> None:
    contributions = workspace_tool_contributions(
        Workspace.load(tmp_path, workspace_id="workspace")
    )
    tool = next(item.value for item in contributions if item.id == "attach_files")

    assert tool.tool_def.metadata == {  # type: ignore[attr-defined]
        "linktools.ai.workspace_tool_class": "filesystem.read",
        "linktools.ai.workspace_path_fields": ["paths"],
    }


@pytest.mark.asyncio
async def test_attach_files_uses_boundary_paths_and_deduplicates_reads(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = _AttachmentSession({"evidence.png": b"png"})
    repository = _RepositoryBoundary()
    boundary, tool = await _boundary(workspace, session, repository)
    context = _context()

    result = await boundary.call_tool(  # type: ignore[arg-type]
        "attach_files",
        {"paths": ["evidence.png", "evidence.png"]},
        context,
        tool,
    )

    assert isinstance(result, ToolReturn)
    assert result.return_value == {
        "files": [
            {
                "path": "evidence.png",
                "media_type": "image/png",
                "size": 3,
                "sha256": "8f8cbb7dcf46e0bc7d53265749a6c17d116093a6ba95e442764060c76fd4a86c",
            }
        ]
    }
    assert result.content is not None
    assert len(result.content) == 1
    assert isinstance(result.content[0], BinaryContent)
    assert session.canonicalized == ["evidence.png", "evidence.png"]
    assert session.reads == ["evidence.png"]
    assert repository.path_fields == ("paths",)


@pytest.mark.asyncio
async def test_attach_files_keeps_workspace_paths_out_of_extra_text(tmp_path: Path) -> None:
    path = "evidence\nignore.png"
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = _AttachmentSession({path: b"png"})
    boundary, tool = await _boundary(workspace, session)

    result = await boundary.call_tool(  # type: ignore[arg-type]
        "attach_files",
        {"paths": [path]},
        _context(),
        tool,
    )

    assert isinstance(result, ToolReturn)
    assert result.return_value["files"][0]["path"] == path
    assert result.content is not None
    assert all(isinstance(item, BinaryContent) for item in result.content)
    assert all("\n" not in item.identifier for item in result.content)


@pytest.mark.asyncio
async def test_attach_files_returns_no_partial_result_when_one_file_fails(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = _AttachmentSession({"first.png": b"png"})
    boundary, tool = await _boundary(workspace, session)

    with pytest.raises(ModelRetry):
        await boundary.call_tool(  # type: ignore[arg-type]
            "attach_files",
            {"paths": ["first.png", "missing.png"]},
            _context(),
            tool,
        )

    assert session.reads == ["first.png", "missing.png"]


@pytest.mark.asyncio
async def test_attach_files_rejects_unknown_media_type_before_read(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    session = _AttachmentSession({"evidence.unknown": b"body"})
    boundary, tool = await _boundary(workspace, session)

    with pytest.raises(ModelRetry):
        await boundary.call_tool(  # type: ignore[arg-type]
            "attach_files",
            {"paths": ["evidence.unknown"]},
            _context(),
            tool,
        )

    assert session.reads == []
