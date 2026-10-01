#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
from pathlib import Path

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import (
    DisabledSandbox,
    SandboxResource,
    SandboxSession,
    LocalSandbox,
    Workspace,
    WorkspaceAccess,
)


class _ByteSession:
    def __init__(self, values: dict[str, bytes]) -> None:
        self._values = values
        self.reads: list[str] = []
        self.closed = 0

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.reads.append(path)
        value = self._values[path]
        if max_bytes is not None and len(value) > max_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return value

    async def close(self) -> None:
        self.closed += 1


class _ByteSandbox:
    def __init__(self, session: _ByteSession) -> None:
        self._session = session
        self.opens = 0

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        del root, resources
        self.opens += 1
        return self._session  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_workspace_access_lazily_opens_one_custom_session(tmp_path: Path) -> None:
    session = _ByteSession({"a.bin": b"a", "b.bin": b"b"})
    sandbox = _ByteSandbox(session)
    access = WorkspaceAccess(sandbox, root=tmp_path)

    assert sandbox.opens == 0
    assert await access.read_bytes("a.bin") == b"a"
    assert await access.read_bytes("b.bin") == b"b"
    assert sandbox.opens == 1
    assert session.reads == ["a.bin", "b.bin"]

    await access.close()
    await access.close()
    assert session.closed == 1

    with pytest.raises(AIError) as raised:
        await access.read_bytes("a.bin")
    assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_workspace_access_retries_failed_session_cleanup(tmp_path: Path) -> None:
    class FailingSession(_ByteSession):
        async def close(self) -> None:
            self.closed += 1
            if self.closed == 1:
                raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)

    session = FailingSession({"a": b"a"})
    access = WorkspaceAccess(_ByteSandbox(session), root=tmp_path)
    assert await access.read_bytes("a") == b"a"
    with pytest.raises(AIError):
        await access.close()
    await access.close()
    assert session.closed == 2
    with pytest.raises(AIError) as error:
        await access.read_bytes("a")
    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_workspace_access_close_calls_share_cleanup_completion(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class WaitingSession(_ByteSession):
        async def close(self) -> None:
            self.closed += 1
            entered.set()
            await release.wait()

    session = WaitingSession({"a": b"a"})
    access = WorkspaceAccess(_ByteSandbox(session), root=tmp_path)
    assert await access.read_bytes("a") == b"a"
    first = asyncio.create_task(access.close())
    await entered.wait()
    second = asyncio.create_task(access.close())
    try:
        await asyncio.sleep(0)
        assert not second.done()
    finally:
        release.set()
        await asyncio.gather(first, second)
    assert session.closed == 1


@pytest.mark.asyncio
async def test_workspace_access_uses_local_workspace_boundary(tmp_path: Path) -> None:
    (tmp_path / "evidence.bin").write_bytes(b"evidence")
    access = WorkspaceAccess.for_workspace(
        Workspace.load(tmp_path),
        sandbox=LocalSandbox(),
    )
    try:
        assert await access.read_bytes("evidence.bin") == b"evidence"
    finally:
        await access.close()


@pytest.mark.asyncio
async def test_workspace_access_does_not_fallback_from_disabled_sandbox(tmp_path: Path) -> None:
    access = WorkspaceAccess.for_workspace(
        Workspace.load(tmp_path),
        sandbox=DisabledSandbox(),
    )
    with pytest.raises(AIError) as raised:
        await access.read_bytes("evidence.bin")
    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    await access.close()
