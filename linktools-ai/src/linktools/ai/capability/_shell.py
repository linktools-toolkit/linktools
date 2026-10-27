#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command tools borrowing the workspace provider's process ownership."""

from collections.abc import Awaitable, Callable, Sequence
from copy import copy
from typing import Any, TypeVar

from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from ..errors import AIError, ErrorCode
from ..workspace import SupportsCommandJobs, ToolPermissionPolicy, WorkspaceOperationRejected
from ._tool_metadata import tool_metadata
from ._tool_signal import ToolCallRetry
from ._workspace_policy import workspace_policy_toolset

_ResultT = TypeVar("_ResultT")
_TOOL_NAMES = ("check_command", "run_command", "start_command", "stop_command")
_REJECTED_MESSAGES = {
    ErrorCode.REQUEST_FIELD_INVALID: "The command or command id is invalid. Use a valid command and retry.",
    ErrorCode.STORAGE_NOT_FOUND: "The command id is unknown or no longer active. Use a command id from this workspace.",
    ErrorCode.AUTHORIZATION_DENIED: "The requested command is not allowed. Choose an allowed command and retry.",
    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE: "The command arguments are too large. Use a smaller request and retry.",
    ErrorCode.TOO_MANY_PENDING_OPERATIONS: "Too many commands are pending. Check or stop an existing command first.",
}


class _ShellTools:
    def __init__(self, backend: SupportsCommandJobs | None) -> None:
        self.backend = backend

    def _require_backend(self) -> SupportsCommandJobs:
        if self.backend is None:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        return self.backend

    async def _call(
        self, operation: Awaitable[_ResultT], *, name: str, rejected_codes: frozenset[ErrorCode],
    ) -> _ResultT:
        try:
            return await operation
        except WorkspaceOperationRejected as error:
            if error.code not in rejected_codes:
                raise
            raise ToolCallRetry(_REJECTED_MESSAGES[error.code]) from error
        except AIError as error:
            if name != "check_command" or error.code not in rejected_codes:
                raise
            raise ToolCallRetry(_REJECTED_MESSAGES[error.code]) from error

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
            self._require_backend().run_command(
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
            self._require_backend().start_command(command),
            name="start_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                    ErrorCode.TOO_MANY_PENDING_OPERATIONS,
                }
            ),
        )

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        return await self._call(
            self._require_backend().check_command(command_id),
            name="check_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                }
            ),
        )

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        return await self._call(
            self._require_backend().stop_command(command_id),
            name="stop_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                }
            ),
        )


class ShellCapability(AbstractCapability[Any]):
    """Expose command tools; the supplied native workspace remains the sole owner."""

    id = "runtime.shell"

    def __init__(self) -> None:
        self._bound = False
        self._toolset: AbstractToolset[Any] | None = None

    def tool_definitions(self) -> tuple[Tool[Any], ...]:
        return _tools(None, _TOOL_NAMES)

    def bind(
        self, selected_tool_names: Sequence[str], *, backend: SupportsCommandJobs | None,
        policy: ToolPermissionPolicy | None = None,
        wrap_toolset: Callable[[AbstractToolset[Any]], AbstractToolset[Any]] | None = None,
    ) -> "ShellCapability":
        selected = frozenset(selected_tool_names)
        if selected.difference(_TOOL_NAMES):
            raise ValueError("unknown shell tools")
        if selected and not isinstance(backend, SupportsCommandJobs):
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE,
                          safe_details={"reason": "workspace_command_jobs_required"})
        bound = copy(self)
        bound._bound = True
        bound._toolset = None
        if selected:
            toolset = FunctionToolset(_tools(backend, tuple(name for name in _TOOL_NAMES if name in selected)),
                                     id="shell")
            bound._toolset = (wrap_toolset(toolset) if wrap_toolset is not None
                              else workspace_policy_toolset(toolset, policy or ToolPermissionPolicy()))
        return bound

    def get_toolset(self) -> AbstractToolset[Any] | None:
        if not self._bound:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._toolset


def _tools(backend: SupportsCommandJobs | None, names: Sequence[str]) -> tuple[Tool[Any], ...]:
    surface = _ShellTools(backend)
    return tuple(Tool(
        getattr(surface, name), takes_ctx=False, name=name,
        metadata=tool_metadata(tool_class="shell", effect_policy="none" if name == "check_command" else "non_replay_safe",
                               plan_safe=name == "check_command"),
    ) for name in names)


__all__ = ["ShellCapability"]
