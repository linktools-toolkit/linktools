#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WorkspaceManager: creates/resolves/cleans up the physical execution directory
for a Run, keyed by run_id. AgentSpec/CompiledAgent/RunContext hold only a
WorkspaceRef, never a raw Path -- only WorkspaceManager.resolve() exposes one,
and only at the point of tool execution (spec section 29).

``WorkspaceManager`` is the Protocol every backend satisfies; ``LocalWorkspaceManager``
is the local-filesystem implementation. A container-backed implementation can
satisfy the same Protocol without touching the local disk."""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..run.context import RunContext


@dataclass(frozen=True, slots=True)
class WorkspaceRef:
    id: str
    run_id: str
    tenant_id: "str | None"


@dataclass(frozen=True, slots=True)
class ExecutionWorkspace:
    ref: WorkspaceRef
    root: Path


@runtime_checkable
class WorkspaceManager(Protocol):
    """Resolve a RunContext/WorkspaceRef to a physical execution directory.

    Implementations must confine path exposure to ``resolve()`` -- callers
    receive a WorkspaceRef, never a raw Path, until the moment of tool
    execution."""

    async def create(self, run: RunContext) -> WorkspaceRef: ...

    async def resolve(self, workspace: WorkspaceRef) -> ExecutionWorkspace: ...

    async def cleanup(self, workspace: WorkspaceRef) -> None: ...


class LocalWorkspaceManager:
    """Local-filesystem WorkspaceManager: one directory per run_id under ``root``."""

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    async def create(self, run: RunContext) -> WorkspaceRef:
        ref = WorkspaceRef(id=run.run_id, run_id=run.run_id, tenant_id=run.tenant_id)
        (self._root / run.run_id).mkdir(parents=True, exist_ok=True)
        return ref

    async def resolve(self, workspace: WorkspaceRef) -> ExecutionWorkspace:
        path = self._root / workspace.run_id
        path.mkdir(parents=True, exist_ok=True)
        return ExecutionWorkspace(ref=workspace, root=path)

    async def cleanup(self, workspace: WorkspaceRef) -> None:
        path = self._root / workspace.run_id
        if path.exists():
            shutil.rmtree(path)
