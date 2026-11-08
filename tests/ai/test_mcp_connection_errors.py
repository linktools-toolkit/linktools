#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public MCP connection classification preserves primary failure semantics."""

import asyncio
from collections.abc import Mapping, Sequence

import pytest
from mcp import ClientSession
from pydantic_ai.mcp import MCPToolset as SDKMCPToolset

from linktools.ai.capability.mcp import MCPToolset
from linktools.ai.core import JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._mcp_transport import _WorkspaceMCPTransport
from linktools.ai.spec import MCPServerSpec
from linktools.ai.workspace import WorkspaceResource, WorkspaceResourcePath


class _Process:
    def __init__(self) -> None:
        self.failure: AIError | None = AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)
        self.closed = False

    async def write_stdin(self, data: bytes) -> None:
        raise AssertionError("Entry must fail before message traffic")

    async def read_stdout(self, max_bytes: int = 65536) -> bytes:
        raise AssertionError("Entry must fail before message traffic")

    async def close_stdin(self) -> None:
        pass

    async def close(self) -> None:
        if self.failure is not None:
            raise self.failure
        self.closed = True


class _Backend:
    def __init__(self, process: _Process) -> None:
        self.process = process

    def stdio_execution_policy(self) -> Mapping[str, JsonValue]:
        return {"version": 1, "boundary": "host-stdio"}

    async def open_stdio_process(
        self, command: str, args: Sequence[str | WorkspaceResourcePath] = (), *,
        resources: Sequence[WorkspaceResource] = (), environment: Mapping[str, str] | None = None,
        cwd_resource_id: str | None = None,
    ) -> _Process:
        return self.process


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("internal", "typed", "wrapper_prefix"))
async def test_mcp_connection_primary_is_not_replaced_by_cleanup(
    monkeypatch: pytest.MonkeyPatch, failure_kind: str,
) -> None:
    process = _Process()
    cleanup = process.failure
    primary: BaseException
    if failure_kind == "typed":
        primary = AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    elif failure_kind == "wrapper_prefix":
        primary = RuntimeError("Client failed to connect: internal state invariant")
    else:
        primary = RuntimeError("session entry internal invariant")

    async def fail_enter(self: ClientSession) -> ClientSession:
        raise primary

    monkeypatch.setattr(ClientSession, "__aenter__", fail_enter)
    transport = _WorkspaceMCPTransport(_Backend(process), "not-launched", (), ())
    toolset = MCPToolset(transport, server=MCPServerSpec("server", "not-launched"))
    try:
        with pytest.raises(AIError if failure_kind == "typed" else RuntimeError) as raised:
            await toolset.__aenter__()
        assert raised.value is not cleanup
        if failure_kind == "typed":
            assert raised.value is primary
        else:
            assert str(raised.value) == f"Client failed to connect: {primary}"
            assert raised.value.__cause__ is primary
        assert primary.__cause__ is cleanup
    finally:
        process.failure = None
        await transport.close()
        assert process.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("io", "typed", "negotiation", "internal"))
async def test_mcp_connection_keeps_sdk_wrapper_classification(
    monkeypatch: pytest.MonkeyPatch, failure_kind: str,
) -> None:
    process = _Process()
    process.failure = None
    primary: BaseException
    if failure_kind == "io":
        primary = OSError("controlled connection failure")
    elif failure_kind == "typed":
        primary = AIError(ErrorCode.CAPABILITY_CONFLICT)
    elif failure_kind == "negotiation":
        primary = RuntimeError("Failed to initialize server session")
    else:
        primary = RuntimeError("session entry internal invariant")

    async def fail_enter(self: ClientSession) -> ClientSession:
        raise primary

    monkeypatch.setattr(ClientSession, "__aenter__", fail_enter)
    transport = _WorkspaceMCPTransport(_Backend(process), "not-launched", (), ())
    toolset = MCPToolset(transport, server=MCPServerSpec("server", "not-launched"))
    with pytest.raises(RuntimeError if failure_kind == "internal" else AIError) as raised:
        await toolset.__aenter__()
    if failure_kind in {"io", "negotiation"}:
        assert isinstance(raised.value, AIError)
        assert raised.value.code is ErrorCode.MCP_CONNECTION_FAILED
        assert raised.value.__cause__ is None
        assert raised.value.safe_details == {
            "server_id": "server", "transport": "stdio", "phase": "connect",
        }
    elif failure_kind == "typed":
        assert raised.value is primary
    else:
        assert raised.value.__cause__ is primary
    assert process.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("message", (
    "Failed to initialize server session",
    "Server session was closed unexpectedly",
    "Unsupported protocol version from the server: private-value",
))
async def test_mcp_connection_negotiation_failure_remains_primary_after_failed_cleanup(
    monkeypatch: pytest.MonkeyPatch, message: str,
) -> None:
    process = _Process()
    primary = RuntimeError(message)
    cleanup = process.failure

    async def fail_enter(self: ClientSession) -> ClientSession:
        raise primary

    monkeypatch.setattr(ClientSession, "__aenter__", fail_enter)
    transport = _WorkspaceMCPTransport(_Backend(process), "not-launched", (), ())
    toolset = MCPToolset(transport, server=MCPServerSpec("server", "not-launched"))
    try:
        with pytest.raises(AIError) as raised:
            await toolset.__aenter__()
        assert raised.value.code is ErrorCode.MCP_CONNECTION_FAILED
        assert raised.value is not cleanup
        assert primary.__cause__ is cleanup
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is not None
        assert raised.value.__context__.__cause__ is primary
        assert "private-value" not in str(raised.value)
    finally:
        process.failure = None
        await transport.close()
        assert process.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("__aenter__", "list_tools"))
async def test_mcp_connection_boundary_preserves_sdk_wrapped_cancellation(
    monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    cancelled = asyncio.CancelledError("cancelled connection")

    async def fail(self: SDKMCPToolset[object]) -> None:
        raise RuntimeError(f"Client failed to connect: {cancelled}") from cancelled

    monkeypatch.setattr(SDKMCPToolset, method, fail)
    process = _Process()
    transport = _WorkspaceMCPTransport(_Backend(process), "not-launched", (), ())
    toolset = MCPToolset(transport, server=MCPServerSpec("server", "not-launched"))
    with pytest.raises(asyncio.CancelledError) as raised:
        if method == "__aenter__":
            await toolset.__aenter__()
        else:
            await toolset.list_tools()
    assert raised.value is cancelled
    await toolset.close_resources()
