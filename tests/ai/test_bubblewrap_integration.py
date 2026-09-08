#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Linux Bubblewrap acceptance coverage."""

import os
from pathlib import Path

import pytest

from linktools.ai.workspace import BubblewrapSandbox, SandboxResource

pytestmark = pytest.mark.asyncio


async def test_bubblewrap_session_is_real_and_shares_workspace(tmp_path: Path) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    hidden = workspace_root / ".linktools"
    hidden.mkdir()
    (hidden / "host-control.txt").write_text("host-only", encoding="utf-8")
    skill_root = tmp_path / "skill"
    skill_root.mkdir()
    (skill_root / "resource.txt").write_text("read-only", encoding="utf-8")

    sandbox = BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    )
    session = await sandbox.open(
        root=workspace_root,
        resources=(SandboxResource("skill-resource", skill_root),),
    )
    try:
        command_result = await session.run_command(
            "printf 'created-by-worker' > worker.txt"
        )
        assert "status: exited" in command_result
        assert (workspace_root / "worker.txt").read_text(encoding="utf-8") == (
            "created-by-worker"
        )

        resource_path = session.resource_path("skill-resource")
        resource_result = await session.run_command(
            f"test -r '{resource_path}/resource.txt' && "
            f"! printf blocked >> '{resource_path}/resource.txt'"
        )
        assert "exit_code: 0" in resource_result

        hidden_result = await session.run_command(
            "test ! -e /workspace/.linktools/host-control.txt"
        )
        assert "exit_code: 0" in hidden_result
    finally:
        await session.close()
