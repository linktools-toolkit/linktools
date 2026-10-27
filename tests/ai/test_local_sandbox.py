#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native LocalWorkspaceBackend command contract."""

import asyncio
import hashlib
import os
import shlex
import sys
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import (
    LocalWorkspaceBackend,
    WorkspaceReadPolicy,
    WorkspaceOperationRejected,
    WorkspaceResource,
)

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("allow_stdio", (False, True))
async def test_canonicalize_path_does_not_apply_read_authorization(
    tmp_path: Path,
    allow_stdio: bool,
) -> None:
    (tmp_path / "allowed").mkdir()
    (tmp_path / "allowed" / "visible.py").write_text("ok", encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("allowed/*.py",)), allow_host_stdio_with_read_policy=allow_stdio)
    await session.working_dir()
    try:
        assert await session.canonicalize_path("allowed") == "allowed"
        assert "visible.py" in await session.list_files("allowed")
    finally:
        await session.close()


@pytest.mark.parametrize("allow_stdio", (False, True))
async def test_inspection_does_not_authorize_a_file_as_a_directory(
    tmp_path: Path,
    allow_stdio: bool,
) -> None:
    (tmp_path / "secret").write_text("private contents", encoding="utf-8")
    policy = WorkspaceReadPolicy(("secret/allowed.txt",))
    session = LocalWorkspaceBackend(tmp_path, read_policy=policy, allow_host_stdio_with_read_policy=allow_stdio)
    await session.working_dir()
    try:
        with pytest.raises(AIError) as error:
            await session.inspect_file("secret")
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await session.close()


@pytest.mark.parametrize("allow_stdio", (False, True))
async def test_read_only_policy_hides_unauthorized_resource_root(
    tmp_path: Path,
    allow_stdio: bool,
) -> None:
    resource = tmp_path / "resource"
    resource.mkdir()
    session = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("allowed.txt",)), allow_host_stdio_with_read_policy=allow_stdio, resources=(WorkspaceResource("resource", resource),))
    await session.working_dir()
    try:
        assert session.resource_path("resource") is None
    finally:
        await session.close()


@pytest.mark.parametrize("allow_stdio", (False, True))
async def test_read_only_policy_exposes_authorized_resource_root(
    tmp_path: Path,
    allow_stdio: bool,
) -> None:
    resource = tmp_path / "resource"
    resource.mkdir()
    session = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(
            ("allowed.txt",),
            {"resource": ("resource.txt",)},
        ), allow_host_stdio_with_read_policy=allow_stdio, resources=(WorkspaceResource("resource", resource),))
    await session.working_dir()
    try:
        assert session.resource_path("resource") == str(resource.resolve())
    finally:
        await session.close()


async def test_local_sandbox_keeps_non_native_resource_virtual(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "run.py"
    source.write_text("print('ready')\n", encoding="utf-8")

    session = LocalWorkspaceBackend(workspace, resources=(WorkspaceResource("skill", files={"script.py": source}),))
    await session.working_dir()
    try:
        assert session.resource_path("skill") is None
    finally:
        await session.close()


def _python_command(code: str) -> str:
    if os.name == "nt":
        return f'"{sys.executable}" -c "{code}"'
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


async def test_read_only_policy_rejects_command_status(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("**",)))
    await session.working_dir()
    try:
        with pytest.raises(WorkspaceOperationRejected) as raised:
            await session.check_command("command")
        assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await session.close()


async def test_local_sandbox_bounds_command_output(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    command = _python_command("print('x' * 60000)")
    try:
        result = await session.run_command(command, timeout_seconds=None)
    finally:
        await session.close()

    assert isinstance(result, str)
    assert len(result) <= 50_000


async def test_local_sandbox_preserves_stdout_and_stderr(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
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
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
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
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        await session.run_command(command, timeout_seconds=5)
        await asyncio.sleep(2.2)
    finally:
        await session.close()

    assert not marker.exists()


async def test_local_sandbox_rejects_remote_commands(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("ssh example.invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_worker_session_uses_shared_command_policy(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("ssh example.invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_worker_session_rejects_invalid_command_input(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        with pytest.raises(AIError) as raised:
            await session.run_command("printf\x00invalid")
    finally:
        await session.close()

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


async def test_worker_session_rejects_invalid_background_command(tmp_path: Path) -> None:
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
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
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.read_text("empty.txt", full_metadata=True)
    finally:
        await session.close()

    assert result["hash"] == hashlib.sha256(b"").hexdigest()
    assert result["total_lines"] == 0 and result["content"] == ""


async def test_local_sandbox_allows_explicit_skill_resource_under_storage(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".linktools" / "skills" / "review"
    source.mkdir(parents=True)
    session = LocalWorkspaceBackend(tmp_path, resources=(WorkspaceResource("resource", source),))
    await session.working_dir()
    try:
        assert session.resource_path("resource") == str(source.resolve())
    finally:
        await session.close()


async def test_local_sandbox_search_continues_after_large_file(tmp_path: Path) -> None:
    (tmp_path / "a-large.txt").write_text("x" * 70_000, encoding="utf-8")
    (tmp_path / "b-target.txt").write_text("needle\n", encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.search_text("needle")
    finally:
        await session.close()

    assert "b-target.txt:1:needle" in result


async def test_local_sandbox_search_reads_beyond_old_file_prefix(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text(
        "x" * 70_000 + "\nneedle-after-prefix\n",
        encoding="utf-8",
    )
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.search_text("needle-after-prefix")
    finally:
        await session.close()

    assert "large.txt:2:needle-after-prefix" in result


async def test_local_sandbox_search_accepts_single_file_path(tmp_path: Path) -> None:
    (tmp_path / "target.txt").write_text("needle\n", encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.search_text("needle", path="target.txt")
    finally:
        await session.close()

    assert result == "target.txt:1:needle"


async def test_local_sandbox_read_reports_binary_file(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\n\x00payload"
    (tmp_path / "image.bin").write_bytes(payload)
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.inspect_file("image.bin")
    finally:
        await session.close()

    assert result["is_text"] is False
    assert result["size"] == len(payload)


async def test_local_sandbox_read_replaces_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "mixed.txt").write_bytes(b"before-\xff-after\n")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.read_text("mixed.txt")
    finally:
        await session.close()

    assert "before-\ufffd-after" in result["content"]
    assert result["lossy"] is True


async def test_local_sandbox_listing_includes_regular_file_size(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_bytes(b"abc")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.list_files()
    finally:
        await session.close()

    assert "sample.txt  file  3" in result


@pytest.mark.parametrize("allow_stdio", (False, True))
async def test_local_sandbox_read_policy_filters_descendants_and_writes(
    tmp_path: Path,
    allow_stdio: bool,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "visible.py").write_text("print('ok')\n", encoding="utf-8")
    (allowed / "hidden.txt").write_text("secret\n", encoding="utf-8")
    policy = WorkspaceReadPolicy(("allowed/*.py",))
    session = LocalWorkspaceBackend(tmp_path, read_policy=policy, allow_host_stdio_with_read_policy=allow_stdio)
    await session.working_dir()
    try:
        assert "visible.py" in await session.list_files("allowed")
        assert "hidden.txt" not in await session.list_files("allowed")
        assert (await session.stat("allowed")).is_dir
        assert "print('ok')" in (await session.read_text("allowed/visible.py"))["content"]
        with pytest.raises(AIError) as denied:
            await session.read_text("allowed/hidden.txt")
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        with pytest.raises(WorkspaceOperationRejected) as rejected:
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
    session = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("allowed/*.py",)))
    await session.working_dir()
    try:
        result = await session.list_files("allowed")
    finally:
        await session.close()

    assert "allowed/visible.py  file  3" in result
    assert "hidden" not in result


async def test_local_sandbox_full_text_metadata_includes_hash_and_lines(tmp_path: Path) -> None:
    content = "first\nsecond\n"
    (tmp_path / "sample.txt").write_text(content, encoding="utf-8")
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
    try:
        result = await session.read_text("sample.txt", full_metadata=True)
    finally:
        await session.close()

    assert result["content"] == content
    assert result["total_lines"] == 2
    assert result["hash"] == hashlib.sha256(content.encode()).hexdigest()


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
    # Publish readiness only after the PID is complete; cleanup may kill the child immediately.
    code = (
        "import os,pathlib,time; "
        f"ready = pathlib.Path({str(ready)!r}); "
        "pending = ready.with_suffix('.tmp'); "
        "pending.write_text(str(os.getpid())); pending.replace(ready); "
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
    session = LocalWorkspaceBackend(tmp_path)
    await session.working_dir()
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
