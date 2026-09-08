#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public workspace capability adapters shared with Runtime integrations."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability, Toolset

from ..errors import AIError, ErrorCode
from ..workspace import Workspace
from ._context import AgentContext
from ._workspace import (
    AttachmentReader,
    WorkspaceAccess,
    _LocalSandbox,
    _WORKSPACE_SANDBOX_CAPABILITY_ID,
    _WORKSPACE_TOOL_NAMES,
    _WorkspaceSandboxToolset,
)

if TYPE_CHECKING:
    from pydantic_ai import RunContext as PydanticRunContext

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
ATTACHMENT_TOOL_NAMES = ("read_attachment",)


class _SharedWorkspaceSandboxToolset(_WorkspaceSandboxToolset):
    """Materialize workspace tools around one Runtime-owned WorkspaceAccess."""

    async def for_run(
        self,
        ctx: "PydanticRunContext[AgentContext[object]]",
    ) -> "_SharedWorkspaceSandboxToolset":
        access = self._access
        if access is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        current = access._run_context
        if current is None:
            access._run_context = ctx
        elif current is not ctx:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _SharedWorkspaceSandboxToolset(
            self._sandbox,
            self._selected_tool_names,
            access=access,
            attachment_reader=self._attachment_reader,
        )


def workspace_capabilities_with_access(
    workspace: Workspace,
    selected_tool_names: Sequence[str],
    *,
    access: WorkspaceAccess,
    attachment_reader: AttachmentReader | None = None,
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Materialize selected workspace tools using one caller-owned lazy access."""
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace must be Workspace")
    if not isinstance(access, WorkspaceAccess):
        raise TypeError("access must be WorkspaceAccess")
    selected = frozenset(selected_tool_names)
    unknown = selected.difference(_WORKSPACE_TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    if not selected:
        return ()
    if "read_attachment" in selected and attachment_reader is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    ordered = tuple(name for name in _WORKSPACE_TOOL_NAMES if name in selected)
    sandbox = (
        workspace.sandbox
        if workspace.sandbox is not None
        else _LocalSandbox(workspace.root)
    )
    toolset = _SharedWorkspaceSandboxToolset(
        sandbox,
        ordered,
        access=access,
        attachment_reader=attachment_reader,
    )
    return (Toolset(toolset, id=_WORKSPACE_SANDBOX_CAPABILITY_ID),)


__all__ = [
    "ATTACHMENT_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "workspace_capabilities_with_access",
]
