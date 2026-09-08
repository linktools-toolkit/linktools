#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt one opened workspace session to Pydantic AI tools."""

from collections.abc import Sequence
from typing import Any, cast

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from ..errors import AIError, ErrorCode
from ..workspace import SandboxSession, Workspace
from ._context import AgentContext
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
_WORKSPACE_TOOL_NAMES = (
    *WORKSPACE_FILESYSTEM_TOOL_NAMES,
    *WORKSPACE_SHELL_TOOL_NAMES,
)
_WORKSPACE_TOOL_CLASSES = {
    **{
        name: "filesystem.read"
        for name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
    },
    **{
        name: "filesystem.write"
        for name in WORKSPACE_FILESYSTEM_TOOL_NAMES
        if name not in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
    },
    **{name: "shell" for name in WORKSPACE_SHELL_TOOL_NAMES},
}
_WORKSPACE_METADATA_KEY = "linktools.ai.workspace_tool_class"
_WORKSPACE_SANDBOX_CAPABILITY_ID = "workspace-sandbox"
_logger = environ.get_logger("ai.capability.workspace")


class _WorkspaceToolSurface:
    def __init__(self, session: SandboxSession | None) -> None:
        self._session = session

    def _require_session(self) -> SandboxSession:
        if self._session is None:
            raise RuntimeError("workspace sandbox session is not open")
        return self._session

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        return await self._require_session().read_file(
            path,
            offset=offset,
            limit=limit,
        )

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        return await self._require_session().write_file(
            path,
            content,
            expected_hash=expected_hash,
        )

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        return await self._require_session().edit_file(
            path,
            old_text,
            new_text,
            expected_hash=expected_hash,
        )

    async def list_directory(self, path: str = ".") -> str:
        return await self._require_session().list_directory(path)

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: str | None = None,
    ) -> str:
        return await self._require_session().search_files(
            pattern,
            path=path,
            include_glob=include_glob,
        )

    async def find_files(self, pattern: str, *, path: str = ".") -> str:
        return await self._require_session().find_files(pattern, path=path)

    async def create_directory(self, path: str) -> str:
        return await self._require_session().create_directory(path)

    async def file_info(self, path: str) -> str:
        return await self._require_session().file_info(path)

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        return await self._require_session().run_command(
            command,
            timeout_seconds=timeout_seconds,
        )

    async def start_command(self, command: str) -> str:
        return await self._require_session().start_command(command)

    async def check_command(self, command_id: str) -> str:
        return await self._require_session().check_command(command_id)

    async def stop_command(self, command_id: str) -> str:
        return await self._require_session().stop_command(command_id)


class _WorkspaceSandboxToolset(FunctionToolset[AgentContext[object]]):
    def __init__(
        self,
        selected_tool_names: tuple[str, ...],
        session: SandboxSession | None,
    ) -> None:
        super().__init__(id=_WORKSPACE_SANDBOX_CAPABILITY_ID)
        surface = _WorkspaceToolSurface(session)
        for name in selected_tool_names:
            self.add_tool(_workspace_tool(surface, name))


class _WorkspaceCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        selected_tool_names: tuple[str, ...],
        session: SandboxSession | None,
    ) -> None:
        self.id = _WORKSPACE_SANDBOX_CAPABILITY_ID
        self._selected_tool_names = selected_tool_names
        self._session = session

    def get_toolset(self) -> _WorkspaceSandboxToolset:
        return _WorkspaceSandboxToolset(self._selected_tool_names, self._session)


def workspace_tool_contributions(
    workspace: Workspace,
) -> tuple[CapabilityContribution[object], ...]:
    """Return the stable workspace tool definitions used by the compiler."""
    del workspace
    surface = _WorkspaceToolSurface(None)
    result: list[CapabilityContribution[object]] = []
    for name in _WORKSPACE_TOOL_NAMES:
        tool = _workspace_tool(surface, name)
        semantic = contribution_semantic_contract("tool", name, tool)
        result.append(
            CapabilityContribution(
                "tool",
                name,
                capability_fingerprint("tool", name, semantic),
                tool,
            )
        )
    return tuple(result)


def workspace_capabilities(
    workspace: Workspace,
    selected_tool_names: Sequence[str],
    *,
    session: SandboxSession | None = None,
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Adapt selected tools to a caller-owned, already-opened session."""
    del workspace
    selected = frozenset(selected_tool_names)
    unknown = selected.difference(_WORKSPACE_TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    if not selected:
        return ()
    if session is None:
        raise AIError(ErrorCode.SANDBOX_SESSION_CLOSED)
    ordered = tuple(name for name in _WORKSPACE_TOOL_NAMES if name in selected)
    _logger.debug(
        "workspace capability materialized: tools=%s session_open=%s",
        ordered,
        session is not None,
    )
    return (_WorkspaceCapability(ordered, session),)


def workspace_tool_class(tool: Tool[Any]) -> str | None:
    if not isinstance(tool, Tool):
        return None
    expected = _WORKSPACE_TOOL_CLASSES.get(tool.name)
    if expected is None:
        return None
    metadata = tool.tool_def.metadata or {}
    if metadata.get(_WORKSPACE_METADATA_KEY) != expected:
        return None
    owner = getattr(tool.function, "__self__", None)
    if type(owner) is not _WorkspaceToolSurface:
        return None
    if getattr(tool.function, "__name__", None) != tool.name:
        return None
    return expected


def _workspace_tool(surface: _WorkspaceToolSurface, name: str) -> Tool[Any]:
    tool_class = (
        "filesystem.read"
        if name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        else "filesystem.write"
        if name in WORKSPACE_FILESYSTEM_TOOL_NAMES
        else "shell"
    )
    return Tool(
        cast(Any, getattr(surface, name)),
        takes_ctx=False,
        name=name,
        metadata={_WORKSPACE_METADATA_KEY: tool_class},
    )


__all__ = [
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "workspace_capabilities",
    "workspace_tool_class",
    "workspace_tool_contributions",
]
