#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The native workspace borrows the sandbox's file and process authority."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.workspaces import (
    SupportsCommands, SupportsFilesystem, SupportsRealpath, Workspace,
    WorkspaceOutputLimitError, WorkspaceTimeoutError, WorkspaceUnavailableError,
)

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import LocalWorkspaceBackend, WorkspaceReadPolicy


def test_native_workspace_shares_bytes_and_session_identity(tmp_path: Path) -> None:
    async def run() -> None:
        session = LocalWorkspaceBackend(tmp_path)
        backend = session
        workspace = Workspace(backend)
        assert isinstance(backend, (SupportsCommands, SupportsFilesystem, SupportsRealpath))
        assert backend.ref is None
        await backend.working_dir()
        assert backend.ref is not None
        try:
            await workspace.write_bytes("nested/file.bin", b"\x00\xffimage")
            assert await session.read_bytes("nested/file.bin") == b"\x00\xffimage"
            assert await workspace.read_bytes("nested/file.bin") == b"\x00\xffimage"
            assert (await workspace.stat("nested/file.bin")).size == 7
            assert [item.name for item in await workspace.list_dir("nested")] == ["file.bin"]
            assert await workspace.realpath("nested/../nested/file.bin") == (tmp_path / "nested/file.bin").as_posix()
            await workspace.remove("nested")
            assert not await workspace.exists("nested")
            with pytest.raises(ValueError):
                await workspace.remove(".")
        finally:
            await session.close()
        with pytest.raises(WorkspaceUnavailableError):
            await workspace.exists(".")
    asyncio.run(run())


def test_native_workspace_does_not_bypass_file_authority(tmp_path: Path) -> None:
    async def run() -> None:
        (tmp_path / "allowed.txt").write_text("yes")
        (tmp_path / "hidden.txt").write_text("no")
        session = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(readable_paths=("allowed.txt",)))
        workspace = Workspace(session)
        try:
            assert await workspace.read_bytes("allowed.txt") == b"yes"
            with pytest.raises(PermissionError):
                await workspace.read_bytes("hidden.txt")
            with pytest.raises(PermissionError):
                await workspace.write_bytes("allowed.txt", b"changed")
            with pytest.raises(PermissionError):
                await workspace.run("printf bypass", shell=True)
        finally:
            await session.close()
    asyncio.run(run())


def test_native_workspace_rejects_escape_and_preserves_symlink_remove(tmp_path: Path) -> None:
    async def run() -> None:
        (tmp_path / "target").write_bytes(b"safe")
        (tmp_path / "link").symlink_to("target")
        session = LocalWorkspaceBackend(tmp_path)
        workspace = Workspace(session)
        try:
            assert await workspace.read_bytes("link") == b"safe"
            await workspace.remove("link")
            assert (tmp_path / "target").read_bytes() == b"safe"
            assert not (tmp_path / "link").exists()
            with pytest.raises(PermissionError):
                await workspace.read_bytes("../outside")
            with pytest.raises(FileNotFoundError):
                await workspace.stat("missing")
        finally:
            await session.close()
    asyncio.run(run())


def test_native_command_result_is_structured_and_environment_is_explicit(tmp_path: Path) -> None:
    async def run() -> None:
        session = LocalWorkspaceBackend(tmp_path)
        backend = session
        try:
            result = await backend.run("printf '%s' \"$EXAMPLE\"; printf error >&2; exit 7", shell=True, env={"EXAMPLE": "value"})
            assert (result.exit_code, result.stdout, result.stderr) == (7, "value", "error")
            assert (await backend.run(["printf", "%s", "a b"])).stdout == "a b"
            assert (await backend.run(["linktools-missing-executable"])).exit_code == 127
            with pytest.raises(WorkspaceTimeoutError):
                await backend.run("sleep 5", shell=True, timeout=0.01)
            with pytest.raises(TypeError):
                await backend.run("true")
            with pytest.raises(ValueError):
                await backend.run([], timeout=1)
        finally:
            await session.close()
    asyncio.run(run())


def test_native_output_cap_is_explicit_not_silent_truncation(tmp_path: Path) -> None:
    async def run() -> None:
        session = LocalWorkspaceBackend(tmp_path)
        try:
            with pytest.raises(WorkspaceOutputLimitError):
                await session.run("head -c 60000 /dev/zero", shell=True)
        finally:
            await session.close()
    asyncio.run(run())


def test_native_command_unknown_effect_error_remains_unknown() -> None:
    class Session(LocalWorkspaceBackend):
        async def execute_command(self, command: str, **kwargs: object) -> object:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
    async def run() -> None:
        with pytest.raises(AIError):
            await Session(Path(".")).run("touch changed", shell=True)
    asyncio.run(run())


def test_worker_protocol_preserves_binary_files_and_structured_commands(tmp_path: Path) -> None:
    import base64
    import json

    from linktools.ai.workspace._sandbox_protocol import validate_request_params
    from linktools.ai.workspace.sandbox_worker import _dispatch

    async def run() -> None:
        session = LocalWorkspaceBackend(tmp_path)
        try:
            params = {"operation": "write", "path": "raw.bin", "data": base64.b64encode(b"\x00\xff").decode("ascii")}
            validate_request_params("raw_filesystem", params)
            assert json.loads(await _dispatch(session, "raw_filesystem", params)) == {}
            assert await session.read_bytes("raw.bin") == b"\x00\xff"
            params = {"command": "printf '%s' \"$MESSAGE\"; exit 4", "environment": {"MESSAGE": "hello"}, "timeout_seconds": 1}
            validate_request_params("execute_command", params)
            result = json.loads(await _dispatch(session, "execute_command", params))
            assert result == {"exit_code": 4, "stdout": "hello", "stderr": "", "timed_out": False, "output_truncated": False}
        finally:
            await session.close()
    asyncio.run(run())


@pytest.mark.parametrize("restricted", [False, True])
def test_native_resource_reads_intersect_membership_and_policy(tmp_path: Path, restricted: bool) -> None:
    from linktools.ai.workspace import WorkspaceResource

    async def run() -> None:
        root = tmp_path / "workspace"
        resource_root = tmp_path / "resource"
        root.mkdir()
        resource_root.mkdir()
        granted = resource_root / "granted.txt"
        denied = resource_root / "denied.txt"
        granted.write_text("granted")
        denied.write_text("denied")
        resource = WorkspaceResource("docs", resource_root, files={"granted.txt": granted})
        policy = WorkspaceReadPolicy(("**",), resource_paths={"docs": ("**",)}) if restricted else None
        backend = LocalWorkspaceBackend(root, resources=(resource,), read_policy=policy)
        workspace = Workspace(backend)
        try:
            assert await workspace.read_bytes(granted.as_posix()) == b"granted"
            with pytest.raises(PermissionError):
                await workspace.read_bytes(denied.as_posix())
            with pytest.raises(PermissionError):
                await workspace.write_bytes(granted.as_posix(), b"changed")
        finally:
            await backend.close()
    asyncio.run(run())


def test_failed_lazy_environment_keeps_identity_and_close_does_not_create(tmp_path: Path) -> None:
    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path / "missing")
        assert backend.ref is None
        with pytest.raises(AIError):
            await backend.working_dir()
        ref = backend.ref
        assert ref is not None
        with pytest.raises(WorkspaceUnavailableError):
            await backend.working_dir()
        assert backend.ref == ref
        await backend.close()
        assert not (tmp_path / "missing").exists()
    asyncio.run(run())


def test_command_timeout_keeps_failed_cleanup_owned_for_close(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from linktools.ai.workspace import _local_sandbox

    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        await backend.working_dir()
        wait = asyncio.create_task(asyncio.sleep(60))
        process = SimpleNamespace(command_id="owned", wait_task=wait)
        attempts = 0

        async def start(command: str, **kwargs: object) -> object:
            backend._processes["owned"] = process
            return process

        async def stop(process: object, **kwargs: object) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)

        monkeypatch.setattr(backend, "_start_process", start)
        monkeypatch.setattr(_local_sandbox, "_stop_process", stop)
        try:
            with pytest.raises(AIError):
                await backend.execute_command("sleep", timeout_seconds=0.001)
            assert "owned" in backend._processes
            await backend.close()
            assert attempts == 2
            assert not backend._processes
        finally:
            wait.cancel()
            await asyncio.gather(wait, return_exceptions=True)
    asyncio.run(run())


def test_failed_bubblewrap_handshake_retains_process_and_lock_until_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace
    from linktools.ai.workspace import BubblewrapWorkspaceBackend, _bubblewrap

    async def run() -> None:
        class Stdin:
            def is_closing(self) -> bool:
                return False
            def close(self) -> None:
                pass

        process = SimpleNamespace(stdin=Stdin(), returncode=None)
        (tmp_path / ".linktools").mkdir()
        backend = BubblewrapWorkspaceBackend(tmp_path, runtime_root=tmp_path,
                                            bwrap_executable=tmp_path / "bwrap",
                                            read_policy=WorkspaceReadPolicy(()))
        monkeypatch.setattr(_bubblewrap, "_resolve_directory", lambda path: path)
        monkeypatch.setattr(_bubblewrap, "_resolve_executable", lambda path: path)
        monkeypatch.setattr(_bubblewrap, "_validate_rootfs", lambda *args: None)
        monkeypatch.setattr(_bubblewrap, "_validate_bwrap", lambda *args: None)
        monkeypatch.setattr(_bubblewrap, "_open_runtime_pidfd", lambda: 12345)
        monkeypatch.setattr(_bubblewrap, "_close_fd", lambda fd: None)
        monkeypatch.setattr(_bubblewrap, "_guardian_config", lambda **kwargs: {})

        async def spawn(*args: object) -> tuple[object, int]:
            return process, -1
        async def handshake() -> None:
            raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)
        async def abort(*args: object, **kwargs: object) -> None:
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)
        waits = 0
        async def wait(*args: object, **kwargs: object) -> None:
            nonlocal waits
            waits += 1
            assert backend._lock_root.exists()
            process.returncode = 0

        monkeypatch.setattr(_bubblewrap, "_spawn_guardian", spawn)
        monkeypatch.setattr(backend, "_open_handshake", handshake)
        monkeypatch.setattr(_bubblewrap, "_abort_guardian", abort)
        monkeypatch.setattr(_bubblewrap, "_wait_guardian", wait)
        with pytest.raises(AIError):
            await backend.working_dir()
        ref, lock = backend.ref, backend._lock_root
        assert ref is not None and lock.exists()
        assert backend._process is process
        await backend.close()
        assert waits == 1 and not lock.exists()
        assert backend.ref == ref
    asyncio.run(run())


def test_provider_construction_and_unused_close_do_not_touch_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.workspace import BubblewrapWorkspaceBackend

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("construction must not inspect or create filesystem state")

    async def run() -> None:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "resolve", forbidden)
            patch.setattr(Path, "mkdir", forbidden)
            providers = (LocalWorkspaceBackend(tmp_path), BubblewrapWorkspaceBackend(
                tmp_path, runtime_root=tmp_path, bwrap_executable=tmp_path / "bwrap",
            ))
            for backend in providers:
                assert backend.ref is None
                await backend.close()
                assert backend.ref is None
    asyncio.run(run())


def test_same_root_environments_do_not_share_command_job_ownership(tmp_path: Path) -> None:
    async def run() -> None:
        first, second = LocalWorkspaceBackend(tmp_path), LocalWorkspaceBackend(tmp_path)
        try:
            started = await first.start_command("sleep 30")
            command_id = started.split("command_id: ", 1)[1].splitlines()[0]
            with pytest.raises(AIError):
                await second.stop_command(command_id)
            assert "status: running" in await first.check_command(command_id)
            assert first.ref != second.ref
            await first.close()
            assert await second.working_dir() == tmp_path.as_posix()
        finally:
            await first.close()
            await second.close()
    asyncio.run(run())
