#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquired MCP processes keep their resources until cleanup is confirmed."""

import asyncio
from collections.abc import Coroutine, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from pydantic_ai.models.test import TestModel

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import AssetCapability
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime import _mcp_transport
from linktools.ai.runtime._mcp_transport import _WorkspaceMCPTransport
from linktools.ai.spec import MCPServerSpec, mcp_server_selector
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import LocalWorkspaceBackend, WorkspaceResource, WorkspaceResourcePath


class _Process:
    def __init__(self) -> None:
        self.closed = False
        self.events: list[str] = []
        self.read_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_release: asyncio.Event | None = None
        self.close_failures = 0
        self.allow_close = True
        self.stdin_error: BaseException | None = None
        self.resource_root: Path | None = None
        self.resource_observations: list[tuple[bool, bool]] = []

    async def write_stdin(self, data: bytes) -> None:
        raise AssertionError("This process must not receive MCP traffic")

    async def read_stdout(self, max_bytes: int = 65536) -> bytes:
        self.read_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.events.append("reader_stopped")
        raise AssertionError("The reader must be canceled")

    async def close_stdin(self) -> None:
        self.events.append("stdin_closed")
        if self.stdin_error is not None:
            raise self.stdin_error

    async def close(self) -> None:
        if self.closed:
            return
        self.events.append("process_close_attempt")
        if self.resource_root is not None:
            self.resource_observations.append((self.closed, self.resource_root.exists()))
        self.close_started.set()
        if self.close_failures or not self.allow_close:
            self.close_failures = max(0, self.close_failures - 1)
            raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)
        if self.close_release is not None:
            await self.close_release.wait()
        self.closed = True
        self.events.append("process_closed")


class _Backend:
    def __init__(self, process: _Process) -> None:
        self.process = process

    def stdio_execution_policy(self) -> Mapping[str, JsonValue]:
        return {"version": 1, "boundary": "host-stdio"}

    async def open_stdio_process(
        self,
        command: str,
        args: Sequence[str | WorkspaceResourcePath] = (),
        *,
        resources: Sequence[WorkspaceResource] = (),
        environment: Mapping[str, str] | None = None,
        cwd_resource_id: str | None = None,
    ) -> _Process:
        return self.process


def _transport(process: _Process) -> _WorkspaceMCPTransport:
    return _WorkspaceMCPTransport(_Backend(process), "not-launched", (), ())


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("construct", "enter", "worker_start"))
async def test_mcp_acquired_process_closes_on_session_setup_failure(
    monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    process = _Process()
    transport = _transport(process)
    failure = AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    def fail_construct(*args: object, **kwargs: object) -> ClientSession:
        raise failure

    async def fail_enter(self: ClientSession) -> ClientSession:
        raise failure

    def fail_start(*args: object) -> Coroutine[Any, Any, None]:
        raise failure

    if phase == "construct":
        monkeypatch.setattr(_mcp_transport, "ClientSession", fail_construct)
    elif phase == "enter":
        monkeypatch.setattr(ClientSession, "__aenter__", fail_enter)
    else:
        monkeypatch.setattr(transport, "_write_worker", fail_start)

    with pytest.raises(AIError) as raised:
        async with asyncio.timeout(2), transport.connect_session():
            pytest.fail("Session setup must fail")
    assert raised.value is failure
    assert process.closed
    assert process.events.index("stdin_closed") < process.events.index("process_closed")
    if phase in {"construct", "enter"}:
        assert not process.read_started.is_set()
    await transport.close()
    assert process.events.count("process_close_attempt") == 1


@pytest.mark.asyncio
async def test_mcp_acquired_process_cancellation_before_session_entry_closes_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()
    transport = _transport(process)
    entering = asyncio.Event()

    async def suspend_enter(self: ClientSession) -> ClientSession:
        entering.set()
        await asyncio.Event().wait()
        raise AssertionError("Session entry must be canceled")

    monkeypatch.setattr(ClientSession, "__aenter__", suspend_enter)

    async def connect() -> None:
        async with transport.connect_session():
            pytest.fail("Session entry must be canceled")

    task = asyncio.create_task(connect())
    await asyncio.wait_for(entering.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert process.closed
    assert not process.read_started.is_set()
    await transport.close()
    assert process.events.count("process_close_attempt") == 1


@pytest.mark.asyncio
async def test_mcp_acquired_process_ordinary_exit_joins_reader_before_close() -> None:
    process = _Process()
    transport = _transport(process)
    async with asyncio.timeout(2), transport.connect_session():
        await process.read_started.wait()
    assert process.closed
    assert process.events.index("reader_stopped") < process.events.index("process_closed")
    await transport.close()
    assert process.events.count("process_close_attempt") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("entry", "exit"))
async def test_mcp_acquired_process_failed_close_remains_retryable(
    monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    process = _Process()
    process.close_failures = 1
    transport = _transport(process)
    primary = RuntimeError("session entry failed")

    async def fail_enter(self: ClientSession) -> ClientSession:
        raise primary

    if phase == "entry":
        monkeypatch.setattr(ClientSession, "__aenter__", fail_enter)
    with pytest.raises(RuntimeError if phase == "entry" else AIError) as raised:
        async with asyncio.timeout(2), transport.connect_session():
            await process.read_started.wait()
    if phase == "entry":
        assert raised.value is primary
        assert isinstance(raised.value.__cause__, AIError)
        assert raised.value.__cause__.code is ErrorCode.SANDBOX_CLEANUP_FAILED
    else:
        assert raised.value.code is ErrorCode.SANDBOX_CLEANUP_FAILED
    assert not process.closed
    await transport.close()
    assert process.closed
    assert process.events.count("process_close_attempt") == 2
    await transport.close()
    assert process.events.count("process_close_attempt") == 2


@pytest.mark.asyncio
async def test_mcp_acquired_process_stdin_failure_does_not_skip_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()
    process.stdin_error = RuntimeError("stdin close failed")
    transport = _transport(process)
    primary = ValueError("session entry failed")

    async def fail_enter(self: ClientSession) -> ClientSession:
        raise primary

    monkeypatch.setattr(ClientSession, "__aenter__", fail_enter)
    with pytest.raises(ValueError) as raised:
        async with transport.connect_session():
            pytest.fail("Session entry must fail")
    assert raised.value is primary
    assert raised.value.__cause__ is process.stdin_error
    assert process.closed
    await transport.close()
    assert process.events.count("process_close_attempt") == 1


@pytest.mark.asyncio
async def test_mcp_acquired_process_interrupted_close_retains_retryable_owner() -> None:
    process = _Process()
    process.close_release = asyncio.Event()
    transport = _transport(process)

    async def connect() -> None:
        async with transport.connect_session():
            await process.read_started.wait()

    task = asyncio.create_task(connect())
    await asyncio.wait_for(process.close_started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert not process.closed
    assert "reader_stopped" in process.events
    process.close_release.set()
    await transport.close()
    assert process.closed
    assert process.events.count("process_close_attempt") == 2


class _ModelBinding:
    provider = "test"
    model_identity = "test:mcp-cleanup"
    contract: Mapping[str, JsonValue] = {"provider": "test", "model": "mcp-cleanup"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="unreachable")


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_until_workspace_close", (False, True))
async def test_mcp_acquired_process_runtime_cancel_keeps_resources_until_closed(
    monkeypatch: pytest.MonkeyPatch, fail_until_workspace_close: bool,
) -> None:
    process = _Process()
    process.allow_close = not fail_until_workspace_close
    entering = asyncio.Event()
    workspace_observations: list[tuple[bool, bool]] = []

    class Workspace(LocalWorkspaceBackend):
        async def open_stdio_process(
            self,
            command: str,
            args: Sequence[str | WorkspaceResourcePath] = (),
            *,
            resources: Sequence[WorkspaceResource] = (),
            environment: Mapping[str, str] | None = None,
            cwd_resource_id: str | None = None,
        ) -> _Process:
            process.resource_root = resources[0].source
            assert process.resource_root is not None and process.resource_root.exists()
            return process

        async def close(self) -> None:
            assert process.resource_root is not None
            workspace_observations.append((process.closed, process.resource_root.exists()))
            process.allow_close = True
            await process.close()
            await super().close()

    async def suspend_enter(self: ClientSession) -> ClientSession:
        entering.set()
        await asyncio.Event().wait()
        raise AssertionError("Session entry must be canceled")

    monkeypatch.setattr(ClientSession, "__aenter__", suspend_enter)
    backend = InMemoryAssetBackend()
    assets = AssetStore(StorageOverlay(backend, writer=backend))
    await assets.initialize()
    await assets.put(AssetKey("mcp", "server/server.py"), b"Resource only. Never executed.")
    features = RuntimeFeatures("mcp-cleanup", workspace_factory=Workspace)
    features.model("default", _ModelBinding())
    features.capability(AssetCapability(assets), scope="runtime")
    features.mcp(MCPServerSpec(
        "server", "not-launched", ("resource:server.py",), AssetKey("mcp", "server"),
    ))
    features.agent("default", allow_tools=(mcp_server_selector("server"),))
    async with Persistence.open("mcp-cleanup", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get("default").start("prepare tools")
            await asyncio.wait_for(entering.wait(), 2)
            cancellation = await execution.cancel(idempotency_key="cancel-during-entry")
            result = (await execution.wait(timeout_seconds=5)).result
            assert cancellation.cancelled
            assert result is not None
            assert result.status is ExecutionStatus.CANCELLED
            assert result.error_code == ErrorCode.EXECUTION_CANCELLED.value
    assert process.closed
    assert workspace_observations
    assert all(closed or exists for closed, exists in workspace_observations)
    assert process.resource_observations and all(exists for _, exists in process.resource_observations)
    assert process.resource_root is not None and not process.resource_root.exists()
    if fail_until_workspace_close:
        assert workspace_observations[0] == (False, True)
        assert process.events.count("process_close_attempt") >= 2
    else:
        assert workspace_observations[0][0]
