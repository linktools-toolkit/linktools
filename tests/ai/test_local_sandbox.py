#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native LocalSandbox command contract."""

import asyncio
import hashlib
import os
import shlex
import sys
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace._local_sandbox import _LocalSandboxSession
from linktools.ai.workspace import LocalSandbox, SandboxResource

pytestmark = pytest.mark.asyncio


def _python_command(code: str) -> str:
    if os.name == "nt":
        return f'"{sys.executable}" -c "{code}"'
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


async def test_local_sandbox_bounds_command_output(tmp_path: Path) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    command = _python_command("print('x' * 60000)")
    try:
        result = await session.run_command(command, timeout_seconds=None)
    finally:
        await session.close()

    assert isinstance(result, str)
    assert len(result) <= 50_000


async def test_local_sandbox_preserves_stdout_and_stderr(tmp_path: Path) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    command = _python_command(
        "import sys;print('stdout-value');print('stderr-value',file=sys.stderr)"
    )
    try:
        result = await session.run_command(command, timeout_seconds=5)
    finally:
        await session.close()

    assert "[stdout]\nstdout-value" in result
    assert "[stderr]\nstderr-value" in result


async def test_local_sandbox_truncation_preserves_stderr_and_status(
    tmp_path: Path,
) -> None:
    session = await LocalSandbox().open(root=tmp_path)
    command = _python_command(
        "import sys;sys.stdout.write('x' * 60000);sys.stderr.write('stderr-tail')"
    )
    try:
        result = await session.run_command(command, timeout_seconds=5)
    finally:
        await session.close()

    assert len(result) <= 50_000
    assert "command_id:" in result
    assert "status: exited" in result
    assert "exit_code: 0" in result
    assert "[stdout]" in result
    assert "[stderr]\nstderr-tail" in result
    assert result.endswith("[output incomplete]")


async def test_local_sandbox_reaps_background_process_after_shell_exit(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "background-marker"
    code = (
        "import pathlib,time; time.sleep(2); "
        f"pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)} &"
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


async def test_worker_session_uses_shared_command_policy(tmp_path: Path) -> None:
    session = _LocalSandboxSession(
        tmp_path,
        (),
        lock_root=tmp_path / "locks",
    )
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("ssh example.invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_worker_session_rejects_invalid_command_input(tmp_path: Path) -> None:
    session = _LocalSandboxSession(
        tmp_path,
        (),
        lock_root=tmp_path / "locks",
    )
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("printf\x00invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_worker_session_rejects_invalid_background_command(tmp_path: Path) -> None:
    session = _LocalSandboxSession(
        tmp_path,
        (),
        lock_root=tmp_path / "locks",
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


async def test_local_sandbox_search_continues_after_large_file(tmp_path: Path) -> None:
    (tmp_path / "a-large.txt").write_text("x" * 70_000, encoding="utf-8")
    (tmp_path / "b-target.txt").write_text("needle\n", encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.search_files("needle")
    finally:
        await session.close()

    assert "b-target.txt:1:needle" in result


async def test_local_sandbox_search_reads_beyond_old_file_prefix(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "x" * 70_000 + "\nneedle-after-prefix\n",
        encoding="utf-8",
    )
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.search_files("needle-after-prefix")
    finally:
        await session.close()

    assert "large.txt:2:needle-after-prefix" in result


async def test_local_sandbox_search_accepts_single_file_path(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("needle\n", encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.search_files("needle", path="target.txt")
    finally:
        await session.close()

    assert result == "target.txt:1:needle"


async def test_local_sandbox_read_reports_binary_file(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n\x00payload"
    (tmp_path / "image.bin").write_bytes(payload)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.read_file("image.bin")
    finally:
        await session.close()

    assert result == f"[Binary file: {len(payload)} bytes. Use a binary-aware tool to inspect.]"


async def test_local_sandbox_read_replaces_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "mixed.txt").write_bytes(b"before-\xff-after\n")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.read_file("mixed.txt")
    finally:
        await session.close()

    assert "before-\ufffd-after" in result


async def test_local_sandbox_listing_includes_regular_file_size(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_bytes(b"abc")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.list_directory()
    finally:
        await session.close()

    assert "sample.txt  (3 bytes)" in result


async def test_local_sandbox_file_info_includes_text_metadata(tmp_path: Path) -> None:
    content = "first\nsecond\n"
    (tmp_path / "sample.txt").write_text(content, encoding="utf-8")
    session = await LocalSandbox().open(root=tmp_path)
    try:
        result = await session.file_info("sample.txt")
    finally:
        await session.close()

    assert "binary: false" in result
    assert "lines: 2" in result
    assert f"hash: {hashlib.sha256(content.encode()).hexdigest()}" in result
