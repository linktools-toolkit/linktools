
"""Workspace-local Agent Skills and sub-agent discovery."""

from pathlib import Path

import pytest
from linktools.ai.workspace._tools import (
    build_workspace_capabilities,
    build_workspace_tool_map,
)


def test_workspace_capabilities_are_stable_and_root_scoped(tmp_path: Path) -> None:
    capabilities = build_workspace_capabilities(tmp_path)
    assert tuple(capability.provider for capability in capabilities) == ("runtime", "runtime")
    assert tuple(capability.id for capability in capabilities) == ("workspace-filesystem", "workspace-shell")
    assert capabilities == build_workspace_capabilities(tmp_path)


@pytest.mark.asyncio
async def test_workspace_tools_enforce_project_root(tmp_path: Path) -> None:
    tools = build_workspace_tool_map(tmp_path)
    result = await tools["read_file"](path="../outside")
    assert result == {"error": "PATH_OUTSIDE_ROOT"}
