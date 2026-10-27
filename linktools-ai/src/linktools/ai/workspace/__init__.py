#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace identity, discovery, policy, instructions, and sandbox contracts."""

from ._access import WorkspaceAccess
from ._instructions import WorkspaceInstructionResolver
from ._root import (
    PermissionDecision,
    ToolPermissionRule,
    Workspace,
    WorkspacePolicy,
    ToolPermissionPolicy,
    validate_workspace_path,
)
from ._local_sandbox import LocalWorkspaceBackend
from ._bubblewrap import BubblewrapWorkspaceBackend
from ._sandbox import (
    WorkspaceReadPolicy,
    SupportsFileTools, SupportsStdio, SupportsResources, SupportsClose, SupportsCommandJobs,
    WorkspaceOperationRejected,
    WorkspaceResource,
    WorkspaceResourcePath,
    WorkspaceStdioProcess,
    normalize_workspace_input_path,
)

__all__ = [
    "SupportsFileTools", "SupportsStdio", "SupportsResources", "SupportsClose", "SupportsCommandJobs",
    "WorkspaceReadPolicy",
    "BubblewrapWorkspaceBackend",
    "WorkspaceInstructionResolver",
    "PermissionDecision",
    "WorkspaceOperationRejected",
    "WorkspaceResource",
    "WorkspaceResourcePath",
    "WorkspaceStdioProcess",
    "LocalWorkspaceBackend",
    "ToolPermissionRule",
    "Workspace",
    "WorkspaceAccess",
    "WorkspacePolicy",
    "ToolPermissionPolicy",
    "normalize_workspace_input_path",
    "validate_workspace_path",
]
