#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace-local Agent Skills and sub-agent discovery."""

from pathlib import Path

import pytest
from linktools.ai.workspace import build_workspace_capability_grants, build_workspace_tool_map


def test_workspace_grants_are_stable_and_root_scoped(tmp_path: Path) -> None:
    grants = build_workspace_capability_grants(tmp_path)
    assert tuple(grant.provider for grant in grants) == ("application", "application")
    assert tuple(grant.id for grant in grants) == ("workspace-filesystem", "workspace-shell")
    assert grants == build_workspace_capability_grants(tmp_path)


@pytest.mark.asyncio
async def test_workspace_tools_enforce_project_root(tmp_path: Path) -> None:
    tools = build_workspace_tool_map(tmp_path)
    result = await tools["read_file"](path="../outside")
    assert result == {"error": "PATH_OUTSIDE_ROOT"}
