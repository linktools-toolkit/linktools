#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-platform LocalWorkspaceBackend acceptance coverage."""

import os
import shlex
import sys
from pathlib import Path

import pytest

from linktools.ai.errors import ErrorCode
from linktools.ai.workspace import LocalWorkspaceBackend, WorkspaceOperationRejected

pytestmark = pytest.mark.asyncio


@pytest.mark.skipif(os.name != "posix", reason="Command execution requires POSIX")
async def test_local_sandbox_runs_python_in_shared_workspace(tmp_path: Path) -> None:
    marker = tmp_path / "platform-marker.txt"
    script = tmp_path / "platform-runner.py"
    script.write_text(
        "from pathlib import Path\n"
        "Path('platform-marker.txt').write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    command = shlex.join([sys.executable, script.name])
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.run_command(command, timeout_seconds=10)
    finally:
        await session.close()

    assert "exit_code: 0" in result
    assert marker.read_text(encoding="utf-8") == "ok"


@pytest.mark.skipif(os.name == "posix", reason="Requires an actual non-POSIX Python runtime")
@pytest.mark.parametrize("stdio", (False, True))
async def test_local_sandbox_rejects_non_posix_execution(tmp_path: Path, stdio: bool) -> None:
    backend = LocalWorkspaceBackend(tmp_path)
    try:
        with pytest.raises(WorkspaceOperationRejected) as raised:
            if stdio:
                await backend.open_stdio_process(sys.executable, ("-c", "pass"))
            else:
                await backend.run_command("echo unsupported")
        assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
        assert raised.value.safe_details["reason"] == "posix_execution_required"
    finally:
        await backend.close()
