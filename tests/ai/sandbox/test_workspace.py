#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/execution/test_workspace.py"""

import pytest

from linktools.ai.sandbox.workspace import (
    Workspace,
    LocalWorkspaceManager,
    WorkspaceManager,
    WorkspaceRef,
)
from linktools.ai.run.context import RunContext
from linktools.ai.run.models import RunnableType


def _run_context(run_id="run-1") -> RunContext:
    return RunContext(
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id="session-1",
        runnable_id="agent-1",
        runnable_type=RunnableType.AGENT,
        user_id=None,
        tenant_id="tenant-1",
        workspace=None,
    )


def test_local_manager_satisfies_protocol(tmp_path):
    manager = LocalWorkspaceManager(root=tmp_path)
    assert isinstance(manager, WorkspaceManager)


@pytest.mark.asyncio
async def test_create_returns_workspace_ref(tmp_path):
    manager = LocalWorkspaceManager(root=tmp_path)
    ref = await manager.create(_run_context())
    assert isinstance(ref, WorkspaceRef)
    assert ref.run_id == "run-1"
    assert ref.tenant_id == "tenant-1"


@pytest.mark.asyncio
async def test_resolve_returns_existing_directory(tmp_path):
    manager = LocalWorkspaceManager(root=tmp_path)
    ref = await manager.create(_run_context())
    workspace = await manager.resolve(ref)
    assert isinstance(workspace, Workspace)
    assert workspace.root.exists()
    assert workspace.root.is_dir()


@pytest.mark.asyncio
async def test_cleanup_removes_directory(tmp_path):
    manager = LocalWorkspaceManager(root=tmp_path)
    ref = await manager.create(_run_context())
    workspace = await manager.resolve(ref)
    await manager.cleanup(ref)
    assert not workspace.root.exists()


@pytest.mark.asyncio
async def test_two_runs_get_isolated_workspaces(tmp_path):
    manager = LocalWorkspaceManager(root=tmp_path)
    ref_a = await manager.create(_run_context(run_id="run-a"))
    ref_b = await manager.create(_run_context(run_id="run-b"))
    workspace_a = await manager.resolve(ref_a)
    workspace_b = await manager.resolve(ref_b)
    assert workspace_a.root != workspace_b.root
