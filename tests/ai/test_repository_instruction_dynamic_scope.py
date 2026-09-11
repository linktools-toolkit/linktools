#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-instruction checks at the final runtime tool boundary."""

from typing import Any

import pytest
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RepositoryInstructionBoundary,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.workspace import (
    WorkspaceToolPermissionPolicy,
)


class _Session:
    async def canonicalize_path(self, path: str) -> str:
        return "." if path == "" else path


class _Boundary:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def render(self) -> str:
        return "repository instructions"

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        path_fields: tuple[str, ...],
    ) -> None:
        self.calls.append(
            {
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments,
                "path_fields": path_fields,
            }
        )
        if self.fail:
            raise ToolFailed("repository instructions changed")


async def _read_file(path: str) -> str:
    return path


async def _list_directory(path: str = ".") -> str:
    return path


def _context(call_id: str = "call") -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id=call_id,
    )


def _boundary(
    toolset: FunctionToolset[None],
    name: str,
    repository: RepositoryInstructionBoundary,
    *,
    policy: WorkspaceToolPermissionPolicy | None = None,
    path_fields: tuple[str, ...] = ("path",),
) -> RuntimeToolBoundaryToolset:
    return RuntimeToolBoundaryToolset(
        (toolset,),
        {
            name: ManagedToolDescriptor(
                effect_owner="none",
                effect="none",
                tool_class="filesystem.read",
                workspace_path_fields=path_fields,
            )
        },
        id="workspace",
        sandbox_session=_Session(),  # type: ignore[arg-type]
        workspace_policy=(
            None
            if policy is None
            else policy
        ),
        repository_boundary=repository,
    )


@pytest.mark.asyncio
async def test_repository_instruction_check_precedes_ask_permission() -> None:
    repository = _Boundary(fail=True)
    toolset = _boundary(
        FunctionToolset([_read_file]),
        "_read_file",
        repository,
        policy=WorkspaceToolPermissionPolicy(default="ask"),
    )
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolFailed, match="repository instructions changed"):
        await toolset.call_tool(
            "_read_file",
            {"path": "pkg/file.txt"},
            context,
            tools["_read_file"],
        )

    assert len(repository.calls) == 1
    assert repository.calls[0]["arguments"] == {"path": "pkg/file.txt"}


@pytest.mark.asyncio
async def test_repository_instruction_refresh_can_fence_one_model_call() -> None:
    repository = _Boundary(fail=True)
    toolset = _boundary(FunctionToolset([_read_file]), "_read_file", repository)
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(ToolFailed):
        await toolset.call_tool(
            "_read_file",
            {"path": "pkg/file.txt"},
            context,
            tools["_read_file"],
        )
    assert repository.calls[0]["tool_call_id"] == "call"


@pytest.mark.asyncio
async def test_empty_workspace_path_is_normalized_before_instruction_check() -> None:
    repository = _Boundary()
    toolset = _boundary(
        FunctionToolset([_list_directory]),
        "_list_directory",
        repository,
    )
    context = _context()
    tools = await toolset.get_tools(context)

    result = await toolset.call_tool(
        "_list_directory",
        {"path": ""},
        context,
        tools["_list_directory"],
    )

    assert result == "."
    assert repository.calls[0]["arguments"] == {"path": "."}


@pytest.mark.asyncio
async def test_approved_call_reaches_tool_after_instruction_check() -> None:
    repository = _Boundary()
    toolset = _boundary(
        FunctionToolset([_read_file]),
        "_read_file",
        repository,
        policy=WorkspaceToolPermissionPolicy(default="ask"),
    )
    context = _context()
    context.tool_call_approved = True
    tools = await toolset.get_tools(context)

    assert (
        await toolset.call_tool(
            "_read_file",
            {"path": "pkg/file.txt"},
            context,
            tools["_read_file"],
        )
        == "pkg/file.txt"
    )
    assert repository.calls
