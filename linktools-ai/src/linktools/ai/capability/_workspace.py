#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness workspace capability adaptation and stable tool projection."""

from collections.abc import Sequence
from typing import Any

from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell

from ..workspace import Workspace
from ._context import RunContext
from ._group import (
    CapabilityContribution,
    capability_fingerprint,
    contribution_semantic_contract,
)

WORKSPACE_FILESYSTEM_TOOL_NAMES = (
    "create_directory",
    "edit_file",
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
    "write_file",
)
WORKSPACE_FILESYSTEM_READ_TOOL_NAMES = (
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
)
WORKSPACE_SHELL_TOOL_NAMES = (
    "check_command",
    "run_command",
    "start_command",
    "stop_command",
)
_WORKSPACE_METADATA_KEY = "linktools.ai.workspace_tool_class"


class _WorkspaceFileSystemToolset(WrapperToolset[RunContext[object]]):
    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: PydanticRunContext[RunContext[object]],
        tool: ToolsetTool[RunContext[object]],
    ) -> Any:
        try:
            return await super().call_tool(name, tool_args, ctx, tool)
        except ModelRetry as error:
            raise ToolFailed(error.message) from error


def workspace_tool_contributions(
    workspace: Workspace,
) -> "tuple[CapabilityContribution[object], ...]":
    """Project Harness public tool methods into stable compiler candidates."""
    filesystem = FileSystem[RunContext[object]](root_dir=workspace.root).get_toolset()
    shell = Shell[RunContext[object]](
        cwd=workspace.root,
        denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
    ).get_toolset()
    result = []
    for name in WORKSPACE_FILESYSTEM_TOOL_NAMES:
        tool_class = "filesystem.read" if name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES else "filesystem.write"
        function = {
            "create_directory": filesystem.create_directory,
            "edit_file": filesystem.edit_file,
            "file_info": filesystem.file_info,
            "find_files": filesystem.find_files,
            "list_directory": filesystem.list_directory,
            "read_file": filesystem.read_file,
            "search_files": filesystem.search_files,
            "write_file": filesystem.write_file,
        }[name]
        result.append(_project_tool(function, name, tool_class))
    for name in WORKSPACE_SHELL_TOOL_NAMES:
        function = {
            "check_command": shell.check_command,
            "run_command": shell.run_command,
            "start_command": shell.start_command,
            "stop_command": shell.stop_command,
        }[name]
        result.append(_project_tool(function, name, "shell"))
    return tuple(result)


def workspace_capabilities(
    workspace: Workspace,
    selected_tool_names: Sequence[str],
) -> "tuple[AbstractCapability[RunContext[object]], ...]":
    """Materialize only the exact workspace tools selected by AgentCompiler."""
    selected = frozenset(selected_tool_names)
    unknown = selected.difference(
        (*WORKSPACE_FILESYSTEM_TOOL_NAMES, *WORKSPACE_SHELL_TOOL_NAMES)
    )
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    values: list[AbstractCapability[RunContext[object]]] = []
    filesystem_names = selected.intersection(WORKSPACE_FILESYSTEM_TOOL_NAMES)
    if filesystem_names:
        toolset = _WorkspaceFileSystemToolset(
            FileSystem[RunContext[object]](root_dir=workspace.root).get_toolset()
        ).filtered(
            lambda _ctx, definition: definition.name in filesystem_names
        )
        values.append(Toolset(toolset, id="workspace-filesystem"))
    shell_names = selected.intersection(WORKSPACE_SHELL_TOOL_NAMES)
    if shell_names:
        toolset = Shell[RunContext[object]](
            cwd=workspace.root,
            denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
        ).get_toolset().filtered(
            lambda _ctx, definition: definition.name in shell_names
        )
        values.append(Toolset(toolset, id="workspace-shell"))
    return tuple(values)


def workspace_tool_class(tool: Tool) -> "str | None":
    metadata = tool.tool_def.metadata or {}
    value = metadata.get(_WORKSPACE_METADATA_KEY)
    return value if isinstance(value, str) else None


def _project_tool(function: object, name: str, tool_class: str) -> CapabilityContribution[object]:
    tool = Tool(
        function,  # type: ignore[arg-type]
        takes_ctx=False,
        name=name,
        metadata={_WORKSPACE_METADATA_KEY: tool_class},
    )
    semantic = contribution_semantic_contract("tool", name, tool)
    return CapabilityContribution(
        "tool",
        name,
        capability_fingerprint("tool", name, semantic),
        tool,
    )


__all__ = [
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "workspace_capabilities",
    "workspace_tool_class",
    "workspace_tool_contributions",
]
