#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native LocalSandbox command contract."""

import asyncio
import hashlib
import shlex
import sys
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace._local_sandbox import _LocalSandboxSession
from linktools.ai.workspace import LocalSandbox, SandboxResource

pytestmark = pytest.mark.asyncio


async def test_local_sandbox_bounds_command_output(tmp_path: Path) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    command = f"{sys.executable} -c \"print('x' * 60000)\""
    try:
        result = await session.run_command(command, timeout_seconds=None)
    finally:
        await session.close()

    assert isinstance(result, str)
    assert len(result) <= 50_000


async def test_local_sandbox_reaps_background_process_after_shell_exit(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "background-marker"
    code = (
        "import pathlib,time; time.sleep(2); "
        f"pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    command = (
        f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} &"
    )
    session = await LocalSandbox().open(root=tmp_path)
    try:
        await session.run_command(command, timeout_seconds=5)
        await asyncio.sleep(2.2)
    finally:
        await session.close()

    assert not marker.exists()


async def test_local_sandbox_rejects_remote_commands(tmp_path: Path) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("ssh example.invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_bubblewrap_worker_mode_does_not_use_local_command_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_local_policy(command: str) -> None:
        raise AssertionError(f"local policy was applied to {command}")

    monkeypatch.setattr(
        "linktools.ai.workspace._local_sandbox._validate_command",
        reject_local_policy,
    )
    session = _LocalSandboxSession(
        tmp_path,
        (),
        lock_root=tmp_path / "locks",
        enforce_command_policy=False,
    )
    try:
        result = await session.run_command("printf sandbox")
    finally:
        await session.close()

    assert "exit_code: 0" in result


async def test_bubblewrap_worker_mode_still_rejects_invalid_command_input(
    tmp_path: Path,
) -> None:
    session = _LocalSandboxSession(
        tmp_path,
        (),
        lock_root=tmp_path / "locks",
        enforce_command_policy=False,
    )
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("printf\x00invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_bubblewrap_worker_mode_rejects_invalid_background_command(
    tmp_path: Path,
) -> None:
    session = _LocalSandboxSession(
        tmp_path,
        (),
        lock_root=tmp_path / "locks",
        enforce_command_policy=False,
    )
    try:
        with pytest.raises(AIError) as raised:
            await session.start_command("printf\x00invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_local_sandbox_reads_empty_file_with_complete_hash(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty.txt").write_bytes(b"")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.read_file("empty.txt")
    finally:
        await session.close()

    assert f"hash: {hashlib.sha256(b'').hexdigest()}" in result
    assert "lines: 0/0" in result


async def test_local_sandbox_allows_explicit_skill_resource_under_storage(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".linktools" / "skills" / "review"
    source.mkdir(parents=True)
    session = await LocalSandbox().open(
        root=tmp_path,
        resources=(SandboxResource("resource", source),),
    )
    try:
        assert session.resource_path("resource") == str(source.resolve())
    finally:
        await session.close()
