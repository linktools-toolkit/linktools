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


@pytest.mark.parametrize("restricted", [False, True])
def test_resource_ancestor_symlink_preserves_grant_boundaries(tmp_path: Path, restricted: bool) -> None:
    from linktools.ai.workspace import WorkspaceResource

    async def run() -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        resource_root = tmp_path / "real" / "assets"
        resource_root.mkdir(parents=True)
        granted = resource_root / "granted.txt"
        denied = resource_root / "denied.txt"
        policy_denied = resource_root / "policy-denied.txt"
        granted.write_text("granted")
        denied.write_text("denied")
        policy_denied.write_text("policy-denied")
        alias = tmp_path / "alias"
        alias.symlink_to(resource_root.parent, target_is_directory=True)
        resource = WorkspaceResource("docs", alias / "assets", files={
            "granted.txt": granted, "policy-denied.txt": policy_denied,
        })
        policy = WorkspaceReadPolicy((), resource_paths={"docs": ("granted.txt",)}) if restricted else None
        backend = LocalWorkspaceBackend(root, resources=(resource,), read_policy=policy)
        try:
            await backend.working_dir()
            resource_path = backend.resource_path("docs")
            assert resource_path is not None
            assert await backend.read_bytes(resource_path + "/granted.txt") == b"granted"
            with pytest.raises(PermissionError):
                await backend.read_bytes(resource_path + "/denied.txt")
            if restricted:
                with pytest.raises(PermissionError):
                    await backend.read_bytes(resource_path + "/policy-denied.txt")
            else:
                assert await backend.read_bytes(resource_path + "/policy-denied.txt") == b"policy-denied"
        finally:
            await backend.close()
    asyncio.run(run())


def test_read_bytes_invalid_path_retains_standard_error(tmp_path: Path) -> None:
    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        try:
            with pytest.raises(AIError) as raised:
                await backend.read_bytes(None)
            assert raised.value.code == ErrorCode.REQUEST_FIELD_INVALID
        finally:
            await backend.close()
    asyncio.run(run())


def test_write_creates_only_necessary_parent_directories(tmp_path: Path) -> None:
    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        try:
            await backend.write_file("new/nested/file.txt", "hello")
            assert await backend.read_bytes("new/nested/file.txt") == b"hello"
            with pytest.raises(AIError) as conflict:
                await backend.write_file("rejected/nested/file.txt", "hello", expected_hash="0" * 64)
            assert conflict.value.code == ErrorCode.STORAGE_CONFLICT
            assert not (tmp_path / "rejected").exists()
            with pytest.raises(AIError):
                await backend.write_file("edit/nested/file.txt", "new", old_text="old")
            assert not (tmp_path / "edit").exists()
        finally:
            await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize("restricted", [False, True])
def test_write_parent_creation_preserves_path_authority(tmp_path: Path, restricted: bool) -> None:
    async def run() -> None:
        root = tmp_path / "workspace"
        outside = tmp_path / "outside"
        root.mkdir()
        outside.mkdir()
        (root / "link").symlink_to(outside, target_is_directory=True)
        policy = WorkspaceReadPolicy(("**",)) if restricted else None
        backend = LocalWorkspaceBackend(root, read_policy=policy)
        try:
            for path in ("link/new/file.txt", "../outside/new/file.txt", ".git/new/file.txt", "secrets/new/file.txt"):
                with pytest.raises(AIError):
                    await backend.write_file(path, "blocked")
            assert not (outside / "new").exists()
            assert not (root / ".git").exists()
            assert not (root / "secrets").exists()
            if restricted:
                with pytest.raises(AIError):
                    await backend.write_file("ordinary/new/file.txt", "blocked")
                assert not (root / "ordinary").exists()
        finally:
            await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["empty", "foreign_content", "replacement", "cancellation"])
def test_failed_write_rolls_back_only_owned_empty_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    from linktools.ai.workspace import _local_sandbox

    def fail_write(root, path, write):
        if failure == "foreign_content":
            (path.parent / "foreign.txt").write_text("keep")
        elif failure == "replacement":
            path.parent.rename(path.parent.with_name("owned-moved"))
            path.parent.mkdir()
        elif failure == "cancellation":
            raise asyncio.CancelledError()
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(_local_sandbox, "_atomic_stream_write", fail_write)

    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        try:
            expected_error = asyncio.CancelledError if failure == "cancellation" else AIError
            with pytest.raises(expected_error):
                await backend.write_file("new/nested/file.txt", "hello")
            assert not (tmp_path / "new/nested/file.txt").exists()
            if failure == "foreign_content":
                assert (tmp_path / "new/nested/foreign.txt").read_text() == "keep"
            elif failure == "replacement":
                assert (tmp_path / "new/nested").is_dir()
                assert (tmp_path / "new/owned-moved").is_dir()
            else:
                assert not (tmp_path / "new").exists()
        finally:
            await backend.close()
    asyncio.run(run())


def test_worker_write_supports_creation_and_exact_replacement(tmp_path: Path) -> None:
    from linktools.ai.workspace._sandbox_protocol import validate_request_params
    from linktools.ai.workspace.sandbox_worker import _dispatch

    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        try:
            create = {"path": "new/nested/report.txt", "content": "before value", "old_text": None}
            validate_request_params("write_file", create)
            await _dispatch(backend, "write_file", create)
            replace = {"path": create["path"], "content": "after", "old_text": "before"}
            validate_request_params("write_file", replace)
            await _dispatch(backend, "write_file", replace)
            assert await backend.read_bytes(create["path"]) == b"after value"
            with pytest.raises(AIError) as invalid:
                await _dispatch(backend, "write_file", {**replace, "old_text": ""})
            assert invalid.value.code == ErrorCode.REQUEST_FIELD_INVALID
            assert await backend.read_bytes(create["path"]) == b"after value"
        finally:
            await backend.close()
    asyncio.run(run())


@pytest.mark.parametrize("owner_kind", ("local", "bubblewrap", "local_stdio", "bubblewrap_stdio"))
@pytest.mark.parametrize("cancel_caller", (False, True))
def test_failed_workspace_cleanup_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner_kind: str, cancel_caller: bool,
) -> None:
    from linktools.ai.workspace import _bubblewrap, _local_sandbox

    async def run() -> None:
        # Fault injection isolates close-task ownership from OS process teardown.
        owner_type = {
            "local": _local_sandbox.LocalWorkspaceBackend,
            "bubblewrap": _bubblewrap.BubblewrapWorkspaceBackend,
            "local_stdio": _local_sandbox._LocalStdioProcess,
            "bubblewrap_stdio": _bubblewrap._BubblewrapStdioProcess,
        }[owner_kind]
        owner = object.__new__(owner_type)
        owner._state = "OPEN"
        owner._state_lock = asyncio.Lock()
        owner._lock = asyncio.Lock()
        owner._initialize_lock = asyncio.Lock()
        owner._process = object()
        owner._close_task = None
        started, release = asyncio.Event(), asyncio.Event()
        owned = {"process"}
        failure = AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)
        fail = True

        async def cleanup() -> None:
            started.set()
            await release.wait()
            if fail:
                raise failure
            owned.clear()
            owner._state = "CLOSED"

        monkeypatch.setattr(owner, "_close_impl", cleanup)
        first = asyncio.create_task(owner.close())
        await started.wait()
        follower = asyncio.create_task(owner.close())
        await asyncio.sleep(0)
        if cancel_caller:
            first.cancel()
            await asyncio.sleep(0)
        release.set()
        error_type = asyncio.CancelledError if cancel_caller else AIError
        with pytest.raises(error_type) as raised:
            await first
        if cancel_caller:
            assert raised.value.__cause__ is failure
        else:
            assert raised.value is failure
        with pytest.raises(AIError) as follower_error:
            await follower
        assert follower_error.value is failure
        assert owned == {"process"}
        fail = False
        await owner.close()
        assert not owned
        await owner.close()

    asyncio.run(run())


def test_local_provider_retries_retained_stdio_cleanup(tmp_path: Path) -> None:
    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path)
        await backend.working_dir()
        failed = True

        class Handle:
            async def close(self) -> None:
                if failed:
                    raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)
                backend._stdio_processes.discard(self)

        handle = Handle()
        backend._stdio_processes.add(handle)
        with pytest.raises(AIError):
            await backend.close()
        assert handle in backend._stdio_processes
        failed = False
        await backend.close()
        assert not backend._stdio_processes

    asyncio.run(run())
