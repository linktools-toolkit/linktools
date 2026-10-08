#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trusted host stdio shares restricted file sessions without widening policy."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import (
    LocalSandbox,
    ReadOnlySandboxPolicy,
    SandboxOperationRejected,
    SandboxResource,
)
from linktools.ai.workspace import _local_sandbox


@pytest.fixture
def subprocess_double(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, AsyncMock, AsyncMock]:
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.pid = 123456
    process.returncode = None
    process.stdin = MagicMock(spec=asyncio.StreamWriter)
    process.stdin.is_closing.return_value = False
    process.stdout = AsyncMock(spec=asyncio.StreamReader)
    process.stderr = AsyncMock(spec=asyncio.StreamReader)
    process.stderr.read.return_value = b""

    async def terminate(*args: object) -> None:
        process.returncode = -15

    spawn = AsyncMock(return_value=process)
    stop = AsyncMock(side_effect=terminate)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(_local_sandbox, "_terminate_unregistered_process", stop)
    monkeypatch.setattr(_local_sandbox._WindowsJob, "create", MagicMock())
    return process, spawn, stop


@pytest.mark.parametrize("option", (None, 0, 1, "true", (), {}))
def test_host_stdio_permission_requires_boolean(option: object) -> None:
    with pytest.raises(TypeError, match="allow_host_stdio_with_read_policy must be bool"):
        LocalSandbox(allow_host_stdio_with_read_policy=option)  # type: ignore[arg-type]


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted", (False, True))
@pytest.mark.parametrize("option", (None, False, True), ids=("default", "false", "true"))
async def test_host_stdio_advertisement_matches_process_permission(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
    restricted: bool,
    option: bool | None,
) -> None:
    process, spawn, _stop = subprocess_double
    sandbox = LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)) if restricted else None,
        **({} if option is None else {"allow_host_stdio_with_read_policy": option}),
    )
    session = await sandbox.open(root=tmp_path)
    try:
        if restricted and option is not True:
            with pytest.raises(AIError) as advertised:
                sandbox.stdio_execution_policy()
            with pytest.raises(AIError) as launched:
                await session.open_stdio_process("trusted-server")
            assert advertised.value.code is launched.value.code is ErrorCode.SANDBOX_UNAVAILABLE
            spawn.assert_not_awaited()
        else:
            assert sandbox.stdio_execution_policy() == {"version": 1, "boundary": "host-stdio"}
            await session.open_stdio_process("trusted-server")
            assert process.returncode is None
    finally:
        await session.close()
    if spawn.called:
        assert process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_stdio", (False, True))
@pytest.mark.parametrize(
    ("operation", "args"),
    (
        ("write_file", ("visible.txt", "changed")),
        ("edit_file", ("visible.txt", "visible", "changed")),
        ("create_directory", ("new",)),
        ("run_command", ("echo denied",)),
        ("start_command", ("echo denied",)),
        ("check_command", ("command",)),
        ("stop_command", ("command",)),
    ),
)
async def test_host_stdio_permission_does_not_allow_commands_or_file_mutation(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
    allow_stdio: bool,
    operation: str,
    args: tuple[str, ...],
) -> None:
    _process, spawn, _stop = subprocess_double
    (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=allow_stdio,
    ).open(root=tmp_path)
    try:
        with pytest.raises(SandboxOperationRejected) as denied:
            await getattr(session, operation)(*args)
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await session.read_bytes("visible.txt") == b"visible"
        assert not (tmp_path / "new").exists()
        spawn.assert_not_awaited()
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_host_stdio_preserves_resource_cwd_and_environment(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    _process, spawn, _stop = subprocess_double
    package = tmp_path / "package"
    package.mkdir()
    resource = SandboxResource("package", package)
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ).open(root=tmp_path, resources=(resource,))
    try:
        assert session.resource_path("package") is None
        with pytest.raises(AIError) as denied:
            await session.open_stdio_process("trusted-server", cwd_resource_id="package")
        assert denied.value.code is ErrorCode.REQUEST_FIELD_INVALID
        spawn.assert_not_awaited()
        await session.open_stdio_process(
            "trusted-server",
            ("server.py", ""),
            resources=(resource,),
            cwd_resource_id="package",
            environment={"PWD": "/caller", "EMPTY": ""},
        )
        assert spawn.call_args.args == ("trusted-server", "server.py", "")
        assert spawn.call_args.kwargs["cwd"] == str(package.resolve())
        assert spawn.call_args.kwargs["env"]["PWD"] == str(package.resolve())
        assert spawn.call_args.kwargs["env"]["EMPTY"] == ""
        assert session.resource_path("package") is None
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_failed_host_stdio_start_keeps_restricted_session_usable(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    process, spawn, _stop = subprocess_double
    failure = OSError("could not start")
    spawn.side_effect = failure
    (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ).open(root=tmp_path)
    try:
        with pytest.raises(AIError) as raised:
            await session.open_stdio_process("trusted-server")
        assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
        assert raised.value.__cause__ is failure
        assert await session.read_bytes("visible.txt") == b"visible"
        spawn.side_effect = None
        await session.open_stdio_process("trusted-server")
    finally:
        await session.close()
    assert process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_start", (False, True))
async def test_closing_restricted_session_owns_inflight_host_stdio_start(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
    cancel_start: bool,
) -> None:
    process, spawn, _stop = subprocess_double
    entered, release = asyncio.Event(), asyncio.Event()

    async def start(*args: object, **kwargs: object) -> MagicMock:
        entered.set()
        await release.wait()
        return process

    spawn.side_effect = start
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ).open(root=tmp_path)
    opening = asyncio.create_task(session.open_stdio_process("trusted-server"))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if cancel_start:
            opening.cancel()
            await asyncio.sleep(0)
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
        with pytest.raises(AIError) as reading:
            await session.read_bytes("visible.txt")
        assert reading.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
        with pytest.raises(AIError) as launching:
            await session.open_stdio_process("trusted-server")
        assert launching.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel_start else AIError) as raised:
            await opening
        if not cancel_start:
            assert raised.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
        await asyncio.wait_for(closing, 2)
        assert process.returncode is not None
    finally:
        release.set()
        await asyncio.gather(opening, return_exceptions=True)
        await session.close()
        if closing is not None:
            await closing


@pytest.mark.asyncio
async def test_cancelled_restricted_session_close_finishes_host_stdio_cleanup(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    process, _spawn, stop = subprocess_double
    entered, release = asyncio.Event(), asyncio.Event()

    async def terminate(*args: object) -> None:
        entered.set()
        await release.wait()
        process.returncode = -15

    stop.side_effect = terminate
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ).open(root=tmp_path)
    handle = await session.open_stdio_process("trusted-server")
    closing = asyncio.create_task(session.close())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert process.returncode is not None
        with pytest.raises(AIError) as raised:
            await handle.read_stdout()
        assert raised.value.code is ErrorCode.SANDBOX_SESSION_CLOSED
    finally:
        release.set()
        await asyncio.gather(closing, return_exceptions=True)
        await session.close()


@pytest.mark.asyncio
async def test_host_stdio_permission_does_not_leak_between_sessions(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    policy = ReadOnlySandboxPolicy(("visible.txt",))
    enabled = LocalSandbox(read_policy=policy, allow_host_stdio_with_read_policy=True)
    restricted = LocalSandbox(read_policy=policy)
    enabled_session = await enabled.open(root=tmp_path)
    restricted_session = await restricted.open(root=tmp_path)
    try:
        await enabled_session.open_stdio_process("trusted-server")
        with pytest.raises(AIError) as advertised:
            restricted.stdio_execution_policy()
        with pytest.raises(AIError) as launched:
            await restricted_session.open_stdio_process("trusted-server")
        assert advertised.value.code is launched.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    finally:
        await enabled_session.close()
        await restricted_session.close()


@pytest.mark.asyncio
async def test_failed_host_stdio_process_close_retains_ownership_until_retry(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    process, _spawn, stop = subprocess_double
    terminate = stop.side_effect
    failure = RuntimeError("process cleanup failed")
    stop.side_effect = failure
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ).open(root=tmp_path)
    try:
        handle = await session.open_stdio_process("trusted-server")
        with pytest.raises(AIError) as raised:
            await handle.close()
        assert raised.value.code is ErrorCode.SANDBOX_CLEANUP_FAILED
        assert raised.value.__cause__ is failure
        assert session.managed_process_ids() == frozenset({process.pid})
    finally:
        stop.side_effect = terminate
        await session.close()
    assert process.returncode is not None
    assert session.managed_process_ids() == frozenset()


@pytest.mark.asyncio
async def test_pending_host_stdio_cleanup_failure_retains_ownership_and_cause(
    tmp_path: Path,
    subprocess_double: tuple[MagicMock, AsyncMock, AsyncMock],
) -> None:
    process, spawn, stop = subprocess_double
    entered, release = asyncio.Event(), asyncio.Event()
    failure = RuntimeError("process cleanup failed")

    async def start(*args: object, **kwargs: object) -> MagicMock:
        entered.set()
        await release.wait()
        return process

    spawn.side_effect = start
    stop.side_effect = failure
    session = await LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ).open(root=tmp_path)
    opening = asyncio.create_task(session.open_stdio_process("trusted-server"))
    await asyncio.wait_for(entered.wait(), 2)
    opening.cancel()
    await asyncio.sleep(0)
    release.set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await opening
    assert cancelled.value.__cause__ is failure
    assert process.returncode is None
    assert session.managed_process_ids() == frozenset({process.pid})
    with pytest.raises(AIError) as closing:
        await session.close()
    assert closing.value.code is ErrorCode.SANDBOX_CLEANUP_FAILED
    assert closing.value.__cause__ is failure
    assert session.managed_process_ids() == frozenset({process.pid})
    with pytest.raises(AIError) as reading:
        await session.read_bytes("visible.txt")
    assert reading.value.code is ErrorCode.SANDBOX_SESSION_LOST
    with pytest.raises(AIError) as repeated:
        await session.close()
    assert repeated.value is closing.value
