#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace tool semantics and Sandbox-backed runtime adaptation."""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from linktools.core import environ
from pydantic_ai import RunContext as PydanticRunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell, ShellToolset

from ..workspace import Sandbox, SandboxSession, Workspace
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
_WORKSPACE_TOOL_NAMES = (*WORKSPACE_FILESYSTEM_TOOL_NAMES, *WORKSPACE_SHELL_TOOL_NAMES)
_WORKSPACE_METADATA_KEY = "linktools.ai.workspace_tool_class"
_WORKSPACE_SANDBOX_CAPABILITY_ID = "workspace-sandbox"
_logger = environ.get_logger("ai.capability.workspace")


class _LocalSandbox:
    def __init__(self, root: Path) -> None:
        self._root = root

    async def open(self) -> SandboxSession:
        return _LocalSandboxSession(self._root)


class _LocalSandboxSession:
    def __init__(self, root: Path) -> None:
        self._filesystem = cast(
            "FileSystemToolset[RunContext[object]]",
            FileSystem[RunContext[object]](root_dir=root).get_toolset(),
        )
        self._shell = cast(
            "ShellToolset[RunContext[object]]",
            Shell[RunContext[object]](
                cwd=root,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ).get_toolset(),
        )

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: "int | None" = None,
    ) -> str:
        return await self._filesystem.read_file(path, offset=offset, limit=limit)

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str:
        return await self._filesystem.write_file(path, content, expected_hash=expected_hash)

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str:
        return await self._filesystem.edit_file(
            path,
            old_text,
            new_text,
            expected_hash=expected_hash,
        )

    async def list_directory(self, path: str = ".") -> str:
        return await self._filesystem.list_directory(path)

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: "str | None" = None,
    ) -> str:
        return await self._filesystem.search_files(
            pattern,
            path=path,
            include_glob=include_glob,
        )

    async def find_files(
        self,
        pattern: str,
        *,
        path: str = ".",
    ) -> str:
        return await self._filesystem.find_files(pattern, path=path)

    async def create_directory(self, path: str) -> str:
        return await self._filesystem.create_directory(path)

    async def file_info(self, path: str) -> str:
        return await self._filesystem.file_info(path)

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: "float | None" = None,
    ) -> str:
        return await self._shell.run_command(command, timeout_seconds=timeout_seconds)

    async def start_command(self, command: str) -> str:
        return await self._shell.start_command(command)

    async def check_command(self, command_id: str) -> str:
        return await self._shell.check_command(command_id)

    async def stop_command(self, command_id: str) -> str:
        return await self._shell.stop_command(command_id)

    async def close(self) -> None:
        await self._shell.__aexit__(None, None, None)


class _WorkspaceToolSurface:
    def __init__(self, session: "SandboxSession | None") -> None:
        self._session = session

    def _require_session(self) -> SandboxSession:
        if self._session is None:
            raise RuntimeError("workspace tool surface is not bound to a SandboxSession")
        return self._session

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: "int | None" = None,
    ) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        return await self._require_session().read_file(path, offset=offset, limit=limit)

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str:
        """Create or overwrite a file with conflict detection.

        Args:
            path: File path relative to the root directory.
            content: The text content to write.
            expected_hash: If provided, the write is rejected when the file exists
                and its current hash doesn't match (optimistic concurrency).

        Returns:
            Confirmation message with new hash.
        """
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
        expected_hash: "str | None" = None,
    ) -> str:
        """Edit a file by exact string replacement with conflict detection.

        The old_text must appear exactly once in the file. Include surrounding
        context lines to ensure uniqueness.

        Args:
            path: File path relative to the root directory.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash for subsequent operations.
        """
        return await self._require_session().edit_file(
            path,
            old_text,
            new_text,
            expected_hash=expected_hash,
        )

    async def list_directory(self, path: str = ".") -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        return await self._require_session().list_directory(path)

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: "str | None" = None,
    ) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').

        Returns:
            str: Matching lines formatted as file:line_number:text.
        """
        return await self._require_session().search_files(
            pattern,
            path=path,
            include_glob=include_glob,
        )

    async def find_files(
        self,
        pattern: str,
        *,
        path: str = ".",
    ) -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Directory to search in, relative to the root directory.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        return await self._require_session().find_files(pattern, path=path)

    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            Confirmation message.
        """
        return await self._require_session().create_directory(path)

    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        return await self._require_session().file_info(path)

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: "float | None" = None,
    ) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        return await self._require_session().run_command(
            command,
            timeout_seconds=timeout_seconds,
        )

    async def start_command(self, command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Callers MUST call `stop_command(command_id)` when done to terminate the
        process and clean up temporary output files.

        Args:
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        return await self._require_session().start_command(command)

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        return await self._require_session().check_command(command_id)

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        return await self._require_session().stop_command(command_id)


class _WorkspaceSandboxToolset(FunctionToolset[RunContext[object]]):
    def __init__(
        self,
        sandbox: Sandbox,
        selected_tool_names: tuple[str, ...],
        *,
        session: "SandboxSession | None" = None,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._selected_tool_names = selected_tool_names
        self._session = session
        surface = _WorkspaceToolSurface(session)
        for name in selected_tool_names:
            self.add_tool(_workspace_tool(surface, name))

    async def for_run(
        self,
        ctx: "PydanticRunContext[RunContext[object]]",
    ) -> "_WorkspaceSandboxToolset":
        del ctx
        session = await self._sandbox.open()
        return _WorkspaceSandboxToolset(
            self._sandbox,
            self._selected_tool_names,
            session=session,
        )

    async def __aexit__(self, *args: Any) -> "bool | None":
        if self._session is None:
            return None
        primary_error = args[1] if len(args) > 1 and isinstance(args[1], BaseException) else None
        close_task = asyncio.create_task(self._session.close(), name="workspace-sandbox-close")
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            try:
                await close_task
            except BaseException:  # noqa: BLE001
                _logger.exception("workspace sandbox cleanup failed during cancellation")
            raise
        except BaseException:  # noqa: BLE001
            if primary_error is None:
                raise
            _logger.exception("workspace sandbox cleanup failed after run failure")
        return None


def workspace_tool_contributions(
    workspace: Workspace,
) -> "tuple[CapabilityContribution[object], ...]":
    """Return LinkTools-owned stable workspace tool compiler candidates."""
    del workspace
    surface = _WorkspaceToolSurface(None)
    result = []
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
) -> "tuple[AbstractCapability[RunContext[object]], ...]":
    """Materialize the selected workspace tools through one per-run SandboxSession."""
    selected = frozenset(selected_tool_names)
    unknown = selected.difference(_WORKSPACE_TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    if not selected:
        return ()
    ordered = tuple(name for name in _WORKSPACE_TOOL_NAMES if name in selected)
    sandbox = workspace.sandbox if workspace.sandbox is not None else _LocalSandbox(workspace.root)
    toolset = _WorkspaceSandboxToolset(sandbox, ordered)
    return (Toolset(toolset, id=_WORKSPACE_SANDBOX_CAPABILITY_ID),)


def workspace_tool_class(tool: Tool) -> "str | None":
    metadata = tool.tool_def.metadata or {}
    value = metadata.get(_WORKSPACE_METADATA_KEY)
    return value if isinstance(value, str) else None


def _workspace_tool(surface: _WorkspaceToolSurface, name: str) -> Tool:
    tool_class = (
        "filesystem.read"
        if name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        else "filesystem.write"
        if name in WORKSPACE_FILESYSTEM_TOOL_NAMES
        else "shell"
    )
    function = getattr(surface, name)
    return Tool(
        function,
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
