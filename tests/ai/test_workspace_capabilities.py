#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace capability projection and materialization contracts."""

from pathlib import Path

import pytest
from linktools.ai.capability import (
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
    workspace_capabilities,
    workspace_tool_class,
    workspace_tool_contributions,
)
from linktools.ai.workspace import Workspace


def test_workspace_tool_contributions_are_stable_and_classified(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)
    contributions = workspace_tool_contributions(workspace)

    assert tuple(item.id for item in contributions) == (
        *WORKSPACE_FILESYSTEM_TOOL_NAMES,
        *WORKSPACE_SHELL_TOOL_NAMES,
    )
    assert all(item.kind == "tool" for item in contributions)
    assert all(len(item.fingerprint) == 64 for item in contributions)
    assert tuple(workspace_tool_class(item.value) for item in contributions) == tuple(
        "filesystem.read"
        if name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        else "filesystem.write"
        for name in WORKSPACE_FILESYSTEM_TOOL_NAMES
    ) + tuple("shell" for _ in WORKSPACE_SHELL_TOOL_NAMES)
    assert tuple(item.fingerprint for item in contributions) == tuple(
        item.fingerprint for item in workspace_tool_contributions(workspace)
    )


def test_workspace_capabilities_materialize_only_selected_tool_groups(tmp_path: Path) -> None:
    workspace = Workspace.load(tmp_path)

    capabilities = workspace_capabilities(
        workspace,
        ("read_file", "run_command"),
    )

    assert tuple(capability.id for capability in capabilities) == (
        "workspace-filesystem",
        "workspace-shell",
    )
    assert workspace_capabilities(workspace, ()) == ()


def test_workspace_capabilities_reject_unknown_tool_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown workspace tools"):
        workspace_capabilities(Workspace.load(tmp_path), ("missing_tool",))
