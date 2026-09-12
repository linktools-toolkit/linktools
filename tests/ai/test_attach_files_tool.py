#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace attach_files tool contracts."""

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import workspace_capabilities, workspace_tool_contributions
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
        value = self.values[path]
        if max_bytes is not None and len(value) > max_bytes:
            raise ValueError("test fixture exceeds max_bytes")
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
    capability = workspace_capabilities(
        workspace,
        ("attach_files",),
        session=session,  # type: ignore[arg-type]
    )[0]
    toolset = capability.get_toolset()
    repository = _RepositoryBoundary()
    boundary = RuntimeToolBoundaryToolset(
        (toolset,),
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
    context = _context()
    tools = await boundary.get_tools(context)  # type: ignore[arg-type]

    result = await boundary.call_tool(  # type: ignore[arg-type]
        "attach_files",
        {"paths": ["evidence.png", "evidence.png"]},
        context,
        tools["attach_files"],
    )

    assert isinstance(result, ToolReturn)
    assert result.return_value == {
        "files": [
            {
                "path": "evidence.png",
                "media_type": "image/png",
                "size": 3,
            }
        ]
    }
    assert result.content is not None
    assert "Workspace file: evidence.png" in result.content
    assert sum(isinstance(item, BinaryContent) for item in result.content) == 1
    assert session.canonicalized == ["evidence.png", "evidence.png"]
    assert session.reads == ["evidence.png"]
    assert repository.path_fields == ("paths",)
