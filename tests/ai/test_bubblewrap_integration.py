#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real Linux Bubblewrap acceptance coverage."""

import asyncio
import os
import re
import shlex
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
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


async def test_bubblewrap_blocks_external_network_but_keeps_loopback(
    tmp_path: Path,
) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    session = await BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    ).open(root=workspace_root)
    try:
        blocked_code = (
            "import socket,sys; "
            "s=socket.socket(); s.settimeout(1); "
            "\ntry: s.connect(('1.1.1.1', 53))\n"
            "except OSError: sys.exit(0)\n"
            "sys.exit(1)"
        )
        blocked = await session.run_command(
            f"/usr/bin/python3 -c {shlex.quote(blocked_code)}"
        )
        assert "exit_code: 0" in blocked

        loopback_code = (
            "import socket; "
            "server=socket.socket(); server.bind(('127.0.0.1', 0)); "
            "server.listen(1); port=server.getsockname()[1]; "
            "client=socket.socket(); client.connect(('127.0.0.1', port)); "
            "conn,_=server.accept(); client.sendall(b'ok'); "
            "assert conn.recv(2) == b'ok'"
        )
        loopback = await session.run_command(
            f"/usr/bin/python3 -c {shlex.quote(loopback_code)}"
        )
        assert "exit_code: 0" in loopback
    finally:
        await session.close()


async def test_bubblewrap_close_reaps_background_command(tmp_path: Path) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    marker = workspace_root / "background-marker.txt"
    code = (
        "import pathlib,time; time.sleep(2); "
        "pathlib.Path('background-marker.txt').write_text('alive', encoding='utf-8')"
    )
    session = await BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    ).open(root=workspace_root)
    started = await session.start_command(
        f"/usr/bin/python3 -c {shlex.quote(code)}"
    )
    assert re.search(r"command_id: [A-Za-z0-9._-]+", started)
    await session.close()
    await asyncio.sleep(2.2)
    assert not marker.exists()


async def test_bubblewrap_guardian_loss_is_observable(tmp_path: Path) -> None:
    runtime_root_value = os.environ.get("LINKTOOLS_BWRAP_RUNTIME_ROOT")
    executable_value = os.environ.get("LINKTOOLS_BWRAP_EXECUTABLE")
    required = os.environ.get("LINKTOOLS_REQUIRE_BUBBLEWRAP") == "1"
    if not runtime_root_value or not executable_value:
        if required:
            pytest.fail("the required Bubblewrap acceptance environment is missing")
        pytest.skip("Bubblewrap acceptance rootfs is not configured")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    session = await BubblewrapSandbox(
        runtime_root=Path(runtime_root_value),
        bwrap_executable=Path(executable_value),
    ).open(root=workspace_root)
    process = session._process
    process.kill()
    await process.wait()
    await asyncio.sleep(0)
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("true")
        assert raised.value.code is ErrorCode.SANDBOX_SESSION_LOST
    finally:
        await session.close()
