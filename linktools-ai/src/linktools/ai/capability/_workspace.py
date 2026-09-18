#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt one opened workspace session to Pydantic AI tools."""

import asyncio
import hashlib
import mimetypes
from collections.abc import Awaitable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent, InstructionPart, ToolReturn, UserContent
from pydantic_ai.toolsets import FunctionToolset

from ..core import PromptLimits
from ..errors import AIError, ErrorCode
from ..workspace import (
    LocalSandbox,
    Sandbox,
    SandboxOperationRejected,
    SandboxSession,
    Workspace,
    WorkspacePolicy,
    normalize_workspace_path,
)
from ._context import AgentContext
from ._tool_signal import ToolCallRetry
from ._tool_semantic import tool_effect_from_metadata, tool_semantic_metadata

_ResultT = TypeVar("_ResultT")

_WORKSPACE_TOOL_DECLARATIONS: dict[str, Mapping[str, object]] = {
    "attach_files": tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("paths",),
    ),
    "create_directory": tool_semantic_metadata(
        effect="non_replay_safe",
        tool_class="filesystem.write",
        path_fields=("path",),
    ),
    "edit_file": tool_semantic_metadata(
        effect="non_replay_safe",
        tool_class="filesystem.write",
        path_fields=("path",),
    ),
    "file_info": tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    ),
    "find_files": tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    ),
    "list_directory": tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    ),
    "read_file": tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
        context_dedupe="workspace_file_read_v1",
    ),
    "search_files": tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    ),
    "write_file": tool_semantic_metadata(
        effect="non_replay_safe",
        tool_class="filesystem.write",
        path_fields=("path",),
    ),
    "check_command": tool_semantic_metadata(
        effect="none",
        plan_safe=True,
        tool_class="shell",
    ),
    "run_command": tool_semantic_metadata(
        effect="non_replay_safe",
        tool_class="shell",
    ),
    "start_command": tool_semantic_metadata(
        effect="non_replay_safe",
        tool_class="shell",
    ),
    "stop_command": tool_semantic_metadata(
        effect="non_replay_safe",
        tool_class="shell",
    ),
}
_EFFECTFUL_WORKSPACE_TOOLS = frozenset(
    {
        name
        for name, metadata in _WORKSPACE_TOOL_DECLARATIONS.items()
        if tool_effect_from_metadata(metadata, require=True) != "none"
    }
)
_WORKSPACE_SANDBOX_CAPABILITY_ID = "workspace-sandbox"
_WORKSPACE_INSTRUCTIONS = InstructionPart(
    content=(
        "Workspace file-tool paths are relative to the logical Workspace root unless "
        "the tool contract says otherwise. Prefer workspace-relative paths and do not "
        "infer host absolute paths. A visible Workspace tool is not automatically "
        "authorized: follow approval requirements and do not switch tools to bypass a "
        "denied or restricted operation."
    ),
    name="workspace",
    dynamic=False,
)
_logger = environ.get_logger("ai.capability.workspace")
_INVALID_WORKSPACE_REQUEST = (
    "The workspace tool arguments or target are invalid. Correct them and retry."
)
_MISSING_WORKSPACE_TARGET = (
    "The requested workspace path does not exist, or its parent directory is missing. "
    "Correct the path and retry."
)
_WORKSPACE_CONFLICT = (
    "The workspace changed since it was read. Read the target again and retry with "
    "the current hash."
)
_DISALLOWED_WORKSPACE_TARGET = (
    "The requested workspace path or command is not allowed. Choose an allowed target "
    "or command and retry."
)
_LARGE_WORKSPACE_REQUEST = (
    "The workspace tool arguments are too large. Use a smaller request and retry."
)
_TOO_MANY_WORKSPACE_COMMANDS = (
    "Too many workspace operations are pending. Reuse, inspect, or stop existing "
    "operations before starting another one."
)
_UNSUPPORTED_IMAGE_INPUT = (
    "The current model does not support image attachments. "
    "Use a non-image input or another approach."
)
_UNRECOGNIZED_ATTACHMENT_TYPE = (
    "A requested attachment has no recognized media type. Use a workspace file with "
    "a recognized file extension or inspect it with another tool."
)
_INVALID_UTF8_WORKSPACE_CONTENT = (
    "The requested workspace content is not valid UTF-8 text. Use a binary-capable "
    "approach or choose a UTF-8 text target."
)
_INVALID_WORKSPACE_REQUESTS = {
    "attach_files": (
        "The attachment request is invalid. Provide at least one existing workspace "
        "file with a recognized type and reduce the number or total size of files "
        "before retrying."
    ),
    "read_file": (
        "The read_file arguments are invalid. Use a non-negative offset and a positive "
        "limit; if the offset is past the file, retry from an earlier offset."
    ),
    "write_file": (
        "The write_file arguments are invalid. Use a writable workspace path and valid "
        "UTF-8 text; if expected_hash is supplied, use the current file hash."
    ),
    "edit_file": (
        "The edit_file arguments are invalid. Read the target and retry with non-empty "
        "old_text that matches exactly once, valid UTF-8 replacement text, and the "
        "current hash if supplied."
    ),
    "list_directory": (
        "The list_directory target is invalid. Use a valid workspace directory path "
        "and retry."
    ),
    "search_files": (
        "The search_files arguments are invalid. Use a valid regular expression, a "
        "valid workspace path, and a relative include_glob when provided."
    ),
    "find_files": (
        "The find_files arguments are invalid. Use a relative glob pattern and a valid "
        "workspace directory path."
    ),
    "create_directory": (
        "The create_directory target is invalid. Use an allowed writable workspace "
        "path and retry."
    ),
    "file_info": (
        "The file_info target is invalid. Use a workspace path that resolves to a "
        "regular file or directory."
    ),
    "run_command": (
        "The run_command arguments are invalid. Use a permitted non-interactive "
        "command and a valid timeout, then retry."
    ),
    "start_command": (
        "The start_command arguments are invalid. Use a permitted non-interactive "
        "command and retry."
    ),
    "check_command": (
        "The command id is invalid or no longer active. Use an active command id "
        "returned by start_command."
    ),
    "stop_command": (
        "The command id is invalid or no longer active. Use an active command id "
        "returned by start_command."
    ),
}


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


class _WorkspaceToolSurface:
    def __init__(
        self,
        session: SandboxSession | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        vision: bool = True,
    ) -> None:
        self._session = session
        self._policy = policy
        self._limits = limits
        self._vision = vision
        self._mime = mimetypes.MimeTypes(filenames=())

    def _require_session(self) -> SandboxSession:
        if self._session is None:
            raise RuntimeError("workspace sandbox session is not open")
        return self._session

    @staticmethod
    async def _call(
        operation: Awaitable[_ResultT],
        *,
        name: str,
        rejected_codes: frozenset[ErrorCode],
        rejected_message: str,
    ) -> _ResultT:
        try:
            return await operation
        except SandboxOperationRejected as error:
            if error.code not in rejected_codes:
                raise
            raise _workspace_tool_rejected(name, error, rejected_message) from error
        except AIError as error:
            if name in _EFFECTFUL_WORKSPACE_TOOLS or error.code not in rejected_codes:
                raise
            raise _workspace_tool_rejected(name, error, rejected_message) from error

    async def attach_files(self, paths: list[str]) -> ToolReturn:
        """Attach Workspace files to the next model request.

        Args:
            paths: File paths relative to the root directory.

        Returns:
            Lightweight file metadata plus the file content for the next model request.
        """
        return await self._call(
            self._attach_files(paths),
            name="attach_files",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

    async def _attach_files(self, paths: list[str]) -> ToolReturn:
        if not paths:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        unique_paths = tuple(dict.fromkeys(paths))
        if len(unique_paths) > self._limits.max_binary_input_parts:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        session = self._require_session()
        total_bytes = 0
        metadata: list[dict[str, object]] = []
        content: list[UserContent] = []
        for path in unique_paths:
            media_type, _ = self._mime.guess_type(path, strict=False)
            if not media_type:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={
                        "field": "paths",
                        "reason": "media_type_unknown",
                    },
                )
            if not self._vision and media_type.lower().startswith("image/"):
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={
                        "field": "paths",
                        "reason": "image_input_not_supported",
                    },
                )
            remaining = self._limits.max_binary_input_bytes - total_bytes
            if remaining < 0:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            body = await session.read_bytes(path, max_bytes=remaining)
            total_bytes += len(body)
            if total_bytes > self._limits.max_binary_input_bytes:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            metadata.append(
                {
                    "path": path,
                    "media_type": media_type,
                    "size": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
            content.append(
                BinaryContent(
                    data=body,
                    media_type=media_type,
                )
            )
        return ToolReturn(
            return_value={"files": metadata},
            content=content,
        )

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
            self._require_session().read_file(path, offset=offset, limit=limit),
            name="read_file",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
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
        rejected_codes = {
            ErrorCode.REQUEST_FIELD_INVALID,
            ErrorCode.STORAGE_NOT_FOUND,
            ErrorCode.AUTHORIZATION_DENIED,
            ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
        }
        if expected_hash is not None:
            rejected_codes.add(ErrorCode.STORAGE_CONFLICT)
        return await self._call(
            self._require_session().write_file(
                path,
                content,
                expected_hash=expected_hash,
            ),
            name="write_file",
            rejected_codes=frozenset(rejected_codes),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
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
        rejected_codes = {
            ErrorCode.REQUEST_FIELD_INVALID,
            ErrorCode.STORAGE_NOT_FOUND,
            ErrorCode.AUTHORIZATION_DENIED,
            ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
        }
        if expected_hash is not None:
            rejected_codes.add(ErrorCode.STORAGE_CONFLICT)
        return await self._call(
            self._require_session().edit_file(
                path,
                old_text,
                new_text,
                expected_hash=expected_hash,
            ),
            name="edit_file",
            rejected_codes=frozenset(rejected_codes),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

    async def list_directory(self, path: str = ".") -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        return await self._call(
            self._require_session().list_directory(path),
            name="list_directory",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

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
            ),
            name="search_files",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
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
            self._require_session().find_files(pattern, path=path),
            name="find_files",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            Confirmation message.
        """
        return await self._call(
            self._require_session().create_directory(path),
            name="create_directory",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        return await self._call(
            self._require_session().file_info(path),
            name="file_info",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

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
            ),
            name="run_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
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
        return await self._call(
            self._require_session().start_command(command),
            name="start_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                    ErrorCode.TOO_MANY_PENDING_OPERATIONS,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        return await self._call(
            self._require_session().check_command(command_id),
            name="check_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        return await self._call(
            self._require_session().stop_command(command_id),
            name="stop_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                }
            ),
            rejected_message=_INVALID_WORKSPACE_REQUEST,
        )


class _WorkspaceSandboxToolset(FunctionToolset[AgentContext[object]]):
    def __init__(
        self,
        selected_tool_names: tuple[str, ...],
        session: SandboxSession | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        vision: bool,
    ) -> None:
        super().__init__(id=_WORKSPACE_SANDBOX_CAPABILITY_ID)
        surface = _WorkspaceToolSurface(
            session,
            policy,
            limits,
            vision=vision,
        )
        for name in selected_tool_names:
            self.add_tool(
                _workspace_tool(
                    surface,
                    name,
                    _WORKSPACE_TOOL_DECLARATIONS[name],
                )
            )


class _WorkspaceCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        selected_tool_names: tuple[str, ...],
        session: SandboxSession | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        vision: bool,
    ) -> None:
        self.id = _WORKSPACE_SANDBOX_CAPABILITY_ID
        self._selected_tool_names = selected_tool_names
        self._session = session
        self._policy = policy
        self._limits = limits
        self._vision = vision

    def get_instructions(self) -> InstructionPart:
        return _WORKSPACE_INSTRUCTIONS

    def get_toolset(self) -> _WorkspaceSandboxToolset:
        return _WorkspaceSandboxToolset(
            self._selected_tool_names,
            self._session,
            self._policy,
            self._limits,
            vision=self._vision,
        )


def _workspace_tool_definitions(workspace: Workspace) -> tuple[Tool[Any], ...]:
    """Return stable Workspace tool definitions before execution materialization."""
    surface = _WorkspaceToolSurface(None, workspace.policy, PromptLimits())
    return tuple(
        _workspace_tool(surface, name, metadata)
        for name, metadata in _WORKSPACE_TOOL_DECLARATIONS.items()
    )


def workspace_capabilities(
    workspace: Workspace,
    selected_tool_names: Sequence[str],
    *,
    limits: "PromptLimits | None" = None,
    session: SandboxSession | None = None,
    vision: bool = True,
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Adapt selected tools to a caller-owned, already-opened session."""
    selected_limits = PromptLimits() if limits is None else limits
    if not isinstance(selected_limits, PromptLimits):
        raise TypeError("limits must be PromptLimits")
    selected = frozenset(selected_tool_names)
    unknown = selected.difference(_WORKSPACE_TOOL_DECLARATIONS)
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    if not selected:
        return ()
    if session is None:
        raise AIError(ErrorCode.SANDBOX_SESSION_CLOSED)
    ordered = tuple(
        name for name in _WORKSPACE_TOOL_DECLARATIONS if name in selected
    )
    _logger.debug(
        "workspace capability materialized: tools=%s session_open=%s",
        ordered,
        session is not None,
    )
    return (
        _WorkspaceCapability(
            ordered,
            session,
            workspace.policy,
            selected_limits,
            vision=vision,
        ),
    )


def _workspace_tool(
    surface: _WorkspaceToolSurface,
    name: str,
    metadata: Mapping[str, object],
) -> Tool[Any]:
    return Tool(
        cast(Any, getattr(surface, name)),
        takes_ctx=False,
        name=name,
        metadata=dict(metadata),
    )


def _workspace_tool_rejected(
    name: str,
    error: AIError,
    default_message: str,
) -> ToolCallRetry:
    reason = error.safe_details.get("reason")
    if name == "attach_files" and reason == "image_input_not_supported":
        message = _UNSUPPORTED_IMAGE_INPUT
    elif name == "attach_files" and reason == "media_type_unknown":
        message = _UNRECOGNIZED_ATTACHMENT_TYPE
    elif reason == "invalid_utf8":
        message = _INVALID_UTF8_WORKSPACE_CONTENT
    elif error.code is ErrorCode.STORAGE_NOT_FOUND:
        message = _MISSING_WORKSPACE_TARGET
    elif error.code is ErrorCode.STORAGE_CONFLICT:
        message = _WORKSPACE_CONFLICT
    elif error.code is ErrorCode.AUTHORIZATION_DENIED:
        message = _DISALLOWED_WORKSPACE_TARGET
    elif error.code is ErrorCode.TOOL_ARGUMENTS_TOO_LARGE:
        message = _LARGE_WORKSPACE_REQUEST
    elif error.code is ErrorCode.TOO_MANY_PENDING_OPERATIONS:
        message = _TOO_MANY_WORKSPACE_COMMANDS
    else:
        message = _INVALID_WORKSPACE_REQUESTS.get(name, default_message)
    _logger.debug(
        "workspace tool call retry: operation=%s code=%s",
        name,
        error.code.value,
    )
    return ToolCallRetry(message)


__all__ = [
    "WorkspaceAccess",
    "workspace_capabilities",
]
