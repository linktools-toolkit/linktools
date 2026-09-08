#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace tool semantics and Sandbox-backed runtime adaptation."""

import asyncio
import stat
import sys
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability, Toolset
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS, Shell

from ..errors import AIError, ErrorCode
from ..workspace import Sandbox, SandboxSession, Workspace
from ._context import AgentContext
from ._group import (
    CapabilityContribution,
    capability_fingerprint,
    contribution_semantic_contract,
)

if TYPE_CHECKING:
    from pydantic_ai import RunContext as PydanticRunContext
    from pydantic_ai.toolsets import ToolsetTool
    from pydantic_ai_harness.filesystem import FileSystemToolset
    from pydantic_ai_harness.shell import ShellToolset

AttachmentReader = Callable[["WorkspaceAccess", str], Awaitable[dict[str, Any]]]
SessionOpener = Callable[[], Awaitable[SandboxSession]]

WORKSPACE_FILESYSTEM_TOOL_NAMES = (
    "create_directory",
    "edit_file",
    "file_info",
    "find_files",
    "list_directory",
    "read_attachment",
    "read_file",
    "search_files",
    "write_file",
)
WORKSPACE_FILESYSTEM_READ_TOOL_NAMES = (
    "file_info",
    "find_files",
    "list_directory",
    "read_attachment",
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

    async def _open_for_run(
        self,
        ctx: "PydanticRunContext[AgentContext[object]]",
    ) -> SandboxSession:
        session = _LocalSandboxSession(self._root)
        await session._bind_run(ctx)
        return session


class _LocalSandboxSession:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._filesystem = cast(
            "FileSystemToolset[AgentContext[object]]",
            FileSystem[AgentContext[object]](root_dir=root).get_toolset(),
        )
        self._shell = cast(
            "ShellToolset[AgentContext[object]]",
            Shell[AgentContext[object]](
                cwd=root,
                denied_env_patterns=LLM_API_KEY_ENV_PATTERNS,
            ).get_toolset(),
        )
        self._shell_context: "PydanticRunContext[AgentContext[object]] | None" = None
        self._shell_tools: "dict[str, ToolsetTool[AgentContext[object]]]" = {}
        self._stack: AsyncExitStack | None = None

    async def _bind_run(
        self,
        ctx: "PydanticRunContext[AgentContext[object]]",
    ) -> None:
        stack = AsyncExitStack()
        try:
            filesystem = cast(
                "FileSystemToolset[AgentContext[object]]",
                await self._filesystem.for_run(ctx),
            )
            self._filesystem = cast(
                "FileSystemToolset[AgentContext[object]]",
                await stack.enter_async_context(filesystem),
            )
            shell = cast(
                "ShellToolset[AgentContext[object]]",
                await self._shell.for_run(ctx),
            )
            self._shell = cast(
                "ShellToolset[AgentContext[object]]",
                await stack.enter_async_context(shell),
            )
            self._shell_context = ctx
            self._shell_tools = await self._shell.get_tools(ctx)
        except BaseException:
            await stack.__aexit__(*sys.exc_info())
            raise
        self._stack = stack

    async def _call_shell(self, name: str, args: dict[str, Any]) -> str:
        ctx = cast(
            "PydanticRunContext[AgentContext[object]]",
            self._shell_context,
        )
        result = await self._shell.call_tool(
            name,
            args,
            ctx,
            self._shell_tools[name],
        )
        return cast(str, result)

    async def read_bytes(self, path: str) -> bytes:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        relative = Path(path)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        current = self._root
        try:
            for part in relative.parts:
                current = current / part
                info = current.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            resolved = current.resolve(strict=True)
            resolved.relative_to(self._root)
            if not stat.S_ISREG(resolved.stat().st_mode):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return resolved.read_bytes()
        except AIError:
            raise
        except (FileNotFoundError, OSError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error

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
        return await self._filesystem.write_file(
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
        return await self._call_shell(
            "run_command",
            {"command": command, "timeout_seconds": timeout_seconds},
        )

    async def start_command(self, command: str) -> str:
        return await self._call_shell("start_command", {"command": command})

    async def check_command(self, command_id: str) -> str:
        return await self._call_shell("check_command", {"command_id": command_id})

    async def stop_command(self, command_id: str) -> str:
        return await self._call_shell("stop_command", {"command_id": command_id})

    async def close(self) -> None:
        stack = self._stack
        self._stack = None
        if stack is None:
            await self._shell.__aexit__(None, None, None)
            return
        await stack.aclose()


class WorkspaceAccess:
    """Lazily own one SandboxSession without exposing it across module boundaries."""

    def __init__(
        self,
        sandbox: Sandbox,
        *,
        session: "SandboxSession | None" = None,
        opener: "SessionOpener | None" = None,
    ) -> None:
        if session is not None and opener is not None:
            raise ValueError("WorkspaceAccess accepts either session or opener")
        self._sandbox = sandbox
        self._session = session
        self._opener = opener or sandbox.open
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    def for_workspace(cls, workspace: Workspace) -> "WorkspaceAccess":
        sandbox = (
            workspace.sandbox
            if workspace.sandbox is not None
            else _LocalSandbox(workspace.root)
        )
        return cls(sandbox)

    async def _ensure_session(self) -> SandboxSession:
        async with self._lock:
            if self._closed:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            session = self._session
            if session is None:
                session = await self._opener()
                self._session = session
            return session

    async def read_bytes(self, path: str) -> bytes:
        session = await self._ensure_session()
        try:
            return await session.read_bytes(path)
        except AttributeError as error:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE) from error

    async def close(self) -> None:
        await self._close(None)

    async def _session_for_tools(self) -> SandboxSession:
        return await self._ensure_session()

    async def _close(self, primary_error: "BaseException | None") -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            session = self._session
            self._session = None
        if session is None:
            return
        close_task = asyncio.create_task(session.close(), name="workspace-sandbox-close")
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            if close_task.cancelled():
                if primary_error is None:
                    raise
                _logger.exception("workspace sandbox cleanup failed after run failure")
                return
            try:
                await close_task
            except BaseException:  # noqa: BLE001
                _logger.exception("workspace sandbox cleanup failed during cancellation")
            raise
        except BaseException:  # noqa: BLE001
            if primary_error is None:
                raise
            _logger.exception("workspace sandbox cleanup failed after run failure")


class _WorkspaceToolSurface:
    def __init__(
        self,
        access: "WorkspaceAccess | None",
        attachment_reader: "AttachmentReader | None" = None,
    ) -> None:
        self._access = access
        self._attachment_reader = attachment_reader

    async def _require_session(self) -> SandboxSession:
        if self._access is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._access._session_for_tools()

    async def read_attachment(self, path: str) -> dict[str, Any]:
        """Read an authorized attachment and make it available to the next model step."""
        if self._access is None or self._attachment_reader is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self._attachment_reader(self._access, path)

    async def read_file(
        self,
        path: str,
        *,
        offset: int = 0,
        limit: "int | None" = None,
    ) -> str:
        """Read a text file with line numbers."""
        session = await self._require_session()
        return await session.read_file(path, offset=offset, limit=limit)

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str:
        """Create or overwrite a file with conflict detection."""
        session = await self._require_session()
        return await session.write_file(path, content, expected_hash=expected_hash)

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: "str | None" = None,
    ) -> str:
        """Edit a file by exact string replacement with conflict detection."""
        session = await self._require_session()
        return await session.edit_file(
            path,
            old_text,
            new_text,
            expected_hash=expected_hash,
        )

    async def list_directory(self, path: str = ".") -> str:
        """List the contents of a directory."""
        session = await self._require_session()
        return await session.list_directory(path)

    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: "str | None" = None,
    ) -> str:
        """Search file contents using a regular expression."""
        session = await self._require_session()
        return await session.search_files(
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
        """Find files by glob pattern."""
        session = await self._require_session()
        return await session.find_files(pattern, path=path)

    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents."""
        session = await self._require_session()
        return await session.create_directory(path)

    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory."""
        session = await self._require_session()
        return await session.file_info(path)

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: "float | None" = None,
    ) -> str:
        """Execute a shell command and return its output."""
        session = await self._require_session()
        return await session.run_command(command, timeout_seconds=timeout_seconds)

    async def start_command(self, command: str) -> str:
        """Start a long-running command in the background."""
        session = await self._require_session()
        return await session.start_command(command)

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command."""
        session = await self._require_session()
        return await session.check_command(command_id)

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output."""
        session = await self._require_session()
        return await session.stop_command(command_id)


class _WorkspaceSandboxToolset(FunctionToolset[AgentContext[object]]):
    def __init__(
        self,
        sandbox: Sandbox,
        selected_tool_names: tuple[str, ...],
        *,
        access: "WorkspaceAccess | None" = None,
        attachment_reader: "AttachmentReader | None" = None,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._selected_tool_names = selected_tool_names
        self._access = access
        self._attachment_reader = attachment_reader
        surface = _WorkspaceToolSurface(access, attachment_reader)
        for name in selected_tool_names:
            self.add_tool(_workspace_tool(surface, name))

    async def for_run(
        self,
        ctx: "PydanticRunContext[AgentContext[object]]",
    ) -> "_WorkspaceSandboxToolset":
        async def open_session() -> SandboxSession:
            if isinstance(self._sandbox, _LocalSandbox):
                return await self._sandbox._open_for_run(ctx)
            return await self._sandbox.open()

        access = WorkspaceAccess(self._sandbox, opener=open_session)
        return _WorkspaceSandboxToolset(
            self._sandbox,
            self._selected_tool_names,
            access=access,
            attachment_reader=self._attachment_reader,
        )

    async def __aexit__(self, *args: Any) -> "bool | None":
        access = self._access
        if access is None:
            return None
        primary_error = (
            args[1]
            if len(args) > 1 and isinstance(args[1], BaseException)
            else None
        )
        await access._close(primary_error)
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
    *,
    attachment_reader: "AttachmentReader | None" = None,
) -> "tuple[AbstractCapability[AgentContext[object]], ...]":
    """Materialize selected workspace tools through one lazy per-run SandboxSession."""
    selected = frozenset(selected_tool_names)
    unknown = selected.difference(_WORKSPACE_TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    if not selected:
        return ()
    ordered = tuple(name for name in _WORKSPACE_TOOL_NAMES if name in selected)
    sandbox = (
        workspace.sandbox
        if workspace.sandbox is not None
        else _LocalSandbox(workspace.root)
    )
    toolset = _WorkspaceSandboxToolset(
        sandbox,
        ordered,
        attachment_reader=attachment_reader,
    )
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
    "WorkspaceAccess",
    "workspace_capabilities",
    "workspace_tool_class",
    "workspace_tool_contributions",
]
