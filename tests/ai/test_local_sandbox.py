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
from linktools.ai.workspace import (
    LocalSandbox,
    ReadOnlySandboxPolicy,
    SandboxOperationRejected,
    SandboxResource,
)

pytestmark = pytest.mark.asyncio


async def test_file_info_does_not_authorize_a_file_as_a_directory(tmp_path: Path) -> None:
    (tmp_path / "secret").write_text("private contents", encoding="utf-8")
    policy = ReadOnlySandboxPolicy(("secret/allowed.txt",))
    session = await LocalSandbox(read_policy=policy).open(root=tmp_path)
    try:
        with pytest.raises(AIError) as error:
            await session.file_info("secret")
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await session.close()


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


async def test_local_sandbox_read_policy_filters_descendants_and_writes(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "visible.py").write_text("print('ok')\n", encoding="utf-8")
    (allowed / "hidden.txt").write_text("secret\n", encoding="utf-8")
    policy = ReadOnlySandboxPolicy(("allowed/*.py",))
    session = await LocalSandbox(read_policy=policy).open(root=tmp_path)
    try:
        assert "visible.py" in await session.list_directory("allowed")
        assert "hidden.txt" not in await session.list_directory("allowed")
        assert "type: directory" in await session.file_info("allowed")
        assert "print('ok')" in await session.read_file("allowed/visible.py")
        with pytest.raises(AIError) as denied:
            await session.read_file("allowed/hidden.txt")
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(SandboxOperationRejected) as rejected:
            await session.write_file("allowed/new.py", "blocked")
        assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await session.close()


async def test_local_sandbox_read_policy_filters_before_listing_limit(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "visible.py").write_text("ok\n", encoding="utf-8")
    for index in range(1_001):
        (allowed / f"hidden-{index:04d}.txt").write_text(
            "secret\n",
            encoding="utf-8",
        )
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("allowed/*.py",)),
    ).open(root=tmp_path)
    try:
        result = await session.list_directory("allowed")
    finally:
        await session.close()

    assert result == "visible.py  (3 bytes)"


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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux process groups")
@pytest.mark.parametrize("operation", ("run", "start", "cancel_spawn"))
@pytest.mark.parametrize("inherit_output", (False, True))
async def test_local_sandbox_reaps_descendants_when_shell_is_already_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    inherit_output: bool,
) -> None:
    import signal

    ready = tmp_path / "child.pid"
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    redirect = "" if inherit_output else " >/dev/null 2>&1"
    command = f"{_python_command(code)}{redirect} &"
    original = asyncio.create_subprocess_exec
    reaped = asyncio.Event()
    release = asyncio.Event()
    command_task: asyncio.Task[str] | None = None

    async def create_after_reaping(
        *args: object, **kwargs: object
    ) -> asyncio.subprocess.Process:
        process = await original(*args, **kwargs)

        async def wait_until_reaped() -> None:
            while process.returncode is None or not ready.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_until_reaped(), 10)
        assert not Path(f"/proc/{process.pid}").exists()
        reaped.set()
        if operation == "cancel_spawn":
            await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_after_reaping)
    session = await LocalSandbox().open(root=tmp_path)
    try:
        if operation == "cancel_spawn":
            command_task = asyncio.create_task(session.run_command(command))
            await asyncio.wait_for(reaped.wait(), 10)
            command_task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await command_task
        elif operation == "run":
            result = await session.run_command(command, timeout_seconds=5)
            assert "status: exited" in result
        else:
            started = await session.start_command(command)
            command_id = started.splitlines()[0].split(": ", 1)[1]

            async def wait_until_exited() -> None:
                while "status: running" in await session.check_command(command_id):
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_until_exited(), 10)
        child_id = int(ready.read_text())
        stat = Path(f"/proc/{child_id}/stat")
        if stat.exists():
            assert stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
    finally:
        release.set()
        if command_task is not None and not command_task.done():
            command_task.cancel()
            await asyncio.gather(command_task, return_exceptions=True)
        if ready.exists():
            try:
                os.kill(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        await session.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux child reaping")
@pytest.mark.parametrize("owned", (False, True))
async def test_reaped_child_notification_keeps_owned_group_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    owned: bool,
) -> None:
    from types import SimpleNamespace

    from linktools.ai.workspace import _local_process

    state = SimpleNamespace(process=SimpleNamespace(pid=123456, returncode=None))

    def already_reaped(*args: object) -> None:
        raise ChildProcessError("asyncio reaped the shell before returncode delivery")

    monkeypatch.setattr(_local_process.os, "waitid", already_reaped)
    monkeypatch.setattr(_local_process, "_process_group_is_owned", lambda _: owned)
    assert await _local_process._observe_process_exit(state) is owned
