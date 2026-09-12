#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt one opened workspace session to Pydantic AI tools."""

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets import FunctionToolset

from ..errors import AIError, ErrorCode
from ..workspace import (
    LocalSandbox,
    Sandbox,
    SandboxSession,
    Workspace,
    normalize_workspace_path,
)
from ._context import AgentContext
from ._group import (
    CapabilityContribution,
    capability_fingerprint,
    contribution_semantic_contract,
)

_BASE_WORKSPACE_FILESYSTEM_TOOL_NAMES = (
    "create_directory",
    "edit_file",
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
    "write_file",
)
_BASE_WORKSPACE_FILESYSTEM_READ_TOOL_NAMES = (
    "file_info",
    "find_files",
    "list_directory",
    "read_file",
    "search_files",
)
WORKSPACE_FILESYSTEM_TOOL_NAMES = (
    *_BASE_WORKSPACE_FILESYSTEM_TOOL_NAMES,
)
WORKSPACE_FILESYSTEM_READ_TOOL_NAMES = (
    *_BASE_WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
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
_WORKSPACE_PATH_FIELDS_KEY = "linktools.ai.workspace_path_fields"
_WORKSPACE_SANDBOX_CAPABILITY_ID = "workspace-sandbox"
_MODEL_CORRECTABLE_ERRORS = {
    ErrorCode.REQUEST_FIELD_INVALID,
    ErrorCode.STORAGE_NOT_FOUND,
    ErrorCode.STORAGE_CONFLICT,
    ErrorCode.AUTHORIZATION_DENIED,
}
_MODEL_ERROR_MESSAGES = {
    ErrorCode.REQUEST_FIELD_INVALID: (
        "The workspace tool arguments or target are invalid. Correct them and retry."
    ),
    ErrorCode.STORAGE_NOT_FOUND: (
        "The requested workspace path does not exist, or its parent directory is missing. "
        "Correct the path and retry."
    ),
    ErrorCode.STORAGE_CONFLICT: (
        "The workspace changed since it was read. Read the target again and retry with "
        "the current hash."
    ),
    ErrorCode.AUTHORIZATION_DENIED: (
        "The requested workspace path or command is not allowed. Choose an allowed target "
        "or command and retry."
    ),
}
_logger = environ.get_logger("ai.capability.workspace")


class WorkspaceAccess:
    """Own one lazy SandboxSession for durable path and byte access."""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        root: Path,
        session: SandboxSession | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._root = root
        self._session = session
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    def for_workspace(cls, workspace: Workspace) -> "WorkspaceAccess":
        sandbox = workspace.sandbox if workspace.sandbox is not None else LocalSandbox()
        return cls(sandbox, root=workspace.root)

    async def _ensure_session(self) -> SandboxSession:
        async with self._lock:
            if self._closed:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if self._session is None:
                self._session = await self._sandbox.open(root=self._root)
            return self._session

    async def canonicalize_path(self, path: str) -> str:
        session = await self._ensure_session()
        return normalize_workspace_path(await session.canonicalize_path(path))

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> bytes:
        session = await self._ensure_session()
        return await session.read_bytes(path, max_bytes=max_bytes)

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
            self._session = None
        if session is not None:
            await session.close()


def workspace_tool_path_fields(tool: Tool[Any]) -> tuple[str, ...]:
    metadata = tool.tool_def.metadata or {}
    return workspace_tool_path_fields_from_metadata(metadata)


def workspace_tool_path_fields_from_metadata(
    metadata: Mapping[str, object] | None,
) -> tuple[str, ...]:
    value = None if metadata is None else metadata.get(_WORKSPACE_PATH_FIELDS_KEY)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def workspace_tool_path_metadata(fields: Sequence[str]) -> dict[str, object]:
    values = tuple(fields)
    if not values or any(not isinstance(field, str) or not field for field in values):
        raise ValueError("workspace path fields must be non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError("workspace path fields must be unique")
    return {_WORKSPACE_PATH_FIELDS_KEY: list(values)}


class _WorkspaceToolSurface:
    def __init__(self, session: SandboxSession | None) -> None:
        self._session = session

    def _require_session(self) -> SandboxSession:
        if self._session is None:
            raise RuntimeError("workspace sandbox session is not open")
        return self._session

    @staticmethod
    async def _call(operation: Awaitable[str]) -> str:
        try:
            return await operation
        except AIError as error:
            if error.code not in _MODEL_CORRECTABLE_ERRORS:
                raise
            raise ModelRetry(_MODEL_ERROR_MESSAGES[error.code]) from error

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> str:
        """Read a text file with line numbers.

        Args:
            path: File path relative to the root directory.
            offset: Zero-based line offset to start reading from.
            limit: Maximum number of lines to return (default: 2000).

        Returns:
            File content with line numbers, plus metadata header.
        """
        return await self._call(
            self._require_session().read_file(path, offset=offset, limit=limit)
        )

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: str | None = None,
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
        return await self._call(
            self._require_session().write_file(
                path,
                content,
                expected_hash=expected_hash,
            )
        )

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: str | None = None,
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
        return await self._call(
            self._require_session().edit_file(
                path,
                old_text,
                new_text,
                expected_hash=expected_hash,
            )
        )

    async def list_directory(self, path: str = ".") -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        return await self._call(self._require_session().list_directory(path))

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: str | None = None,
    ) -> str:
        """Search file contents using a regular expression.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in, relative to the root directory.
            include_glob: If provided, only search files matching this glob (e.g. '*.py').

        Returns:
            str: Matching lines formatted as file:line_number:text.
        """
        return await self._call(
            self._require_session().search_files(
                pattern,
                path=path,
                include_glob=include_glob,
            )
        )

    async def find_files(self, pattern: str, *, path: str = ".") -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Directory to search in, relative to the root directory.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        return await self._call(
            self._require_session().find_files(pattern, path=path)
        )

    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            Confirmation message.
        """
        return await self._call(self._require_session().create_directory(path))

    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        return await self._call(self._require_session().file_info(path))

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        return await self._call(
            self._require_session().run_command(
                command,
                timeout_seconds=timeout_seconds,
            )
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
        return await self._call(self._require_session().start_command(command))

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        return await self._call(self._require_session().check_command(command_id))

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        return await self._call(self._require_session().stop_command(command_id))


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
    return expected


def _workspace_tool(surface: _WorkspaceToolSurface, name: str) -> Tool[Any]:
    tool_class = (
        "filesystem.read"
        if name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        else "filesystem.write"
        if name in WORKSPACE_FILESYSTEM_TOOL_NAMES
        else "shell"
    )
    metadata: dict[str, object] = {_WORKSPACE_METADATA_KEY: tool_class}
    if tool_class in {"filesystem.read", "filesystem.write"}:
        metadata.update(workspace_tool_path_metadata(("path",)))
    return Tool(
        cast(Any, getattr(surface, name)),
        takes_ctx=False,
        name=name,
        metadata=metadata,
    )


__all__ = [
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "WorkspaceAccess",
    "workspace_capabilities",
    "workspace_tool_class",
    "workspace_tool_path_fields",
    "workspace_tool_path_fields_from_metadata",
    "workspace_tool_path_metadata",
    "workspace_tool_contributions",
]
