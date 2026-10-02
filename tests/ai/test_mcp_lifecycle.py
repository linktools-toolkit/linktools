#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP ownership, cancellation, and safe boundary failure contracts."""

import asyncio

import pytest
from pydantic_ai.mcp import MCPToolset

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import _with_cleanup_diagnostic
from linktools.ai.runtime._mcp import (
    _MCPCapability,
    _MCPDiscoveryToolset,
    close_mcp_resources,
)
from linktools.ai.runtime._mcp_transport import _create_mcp_transport
from linktools.ai.spec import MCPServerSpec


def _discovery() -> _MCPDiscoveryToolset:
    server = MCPServerSpec(
        "remote", transport="streamable-http", url="https://example.invalid/mcp",
    )
    transport = _create_mcp_transport(
        server, sandboxed=False, sandbox_session=None, host_cwd=None,
    )
    return _MCPDiscoveryToolset(transport, server=server)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", (False, True))
async def test_cancellation_during_final_cleanup_finishes_every_owned_resource(
    fail_close: bool,
) -> None:
    started, release = asyncio.Event(), asyncio.Event()
    closed: list[str] = []

    class First:
        async def close_resources(self) -> None:
            started.set()
            await release.wait()
            closed.append("first")
            if fail_close:
                raise RuntimeError("private cleanup details")

    class Second:
        async def close_resources(self) -> None:
            closed.append("second")

    task = asyncio.create_task(close_mcp_resources((
        _MCPCapability("first", object(), First()),
        _MCPCapability("second", object(), Second()),
    )))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(task, 1)
    assert closed == ["first", "second"]
    if fail_close:
        cleanup_error: BaseException | None = raised.value
        while cleanup_error is not None and not isinstance(cleanup_error, AIError):
            cleanup_error = cleanup_error.__cause__ or cleanup_error.__context__
        assert isinstance(cleanup_error, AIError)
        assert cleanup_error.code is ErrorCode.MCP_CLEANUP_FAILED
        assert "private" not in str(cleanup_error)


@pytest.mark.asyncio
async def test_cleanup_keeps_sandbox_failure_and_closes_remaining_connections() -> None:
    failure = AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)
    closed: list[str] = []

    class Failed:
        async def close_resources(self) -> None:
            raise failure

    class Remaining:
        async def close_resources(self) -> None:
            closed.append("remaining")

    with pytest.raises(AIError) as raised:
        await close_mcp_resources((
            _MCPCapability("failed", object(), Failed()),
            _MCPCapability("remaining", object(), Remaining()),
        ))
    assert raised.value is failure
    assert closed == ["remaining"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", (
    RuntimeError("internal state invariant"),
    ValueError("implementation bug"),
    AIError(ErrorCode.CAPABILITY_CONFLICT),
    asyncio.CancelledError(),
))
async def test_connection_boundary_preserves_internal_typed_and_cancelled_errors(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException,
) -> None:
    async def fail_enter(self: MCPToolset) -> None:
        raise failure

    monkeypatch.setattr(MCPToolset, "__aenter__", fail_enter)
    toolset = _discovery()
    with pytest.raises(type(failure)) as raised:
        await toolset.__aenter__()
    assert raised.value is failure
    await toolset.close_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", (
    "Failed to initialize server session",
    "Unsupported protocol version from the server: private-value",
))
async def test_sdk_untyped_negotiation_failures_are_safe_connection_errors(
    monkeypatch: pytest.MonkeyPatch, message: str,
) -> None:
    async def fail_enter(self: MCPToolset) -> None:
        raise RuntimeError(message)

    monkeypatch.setattr(MCPToolset, "__aenter__", fail_enter)
    toolset = _discovery()
    with pytest.raises(AIError) as raised:
        await toolset.__aenter__()
    assert raised.value.code is ErrorCode.MCP_CONNECTION_FAILED
    assert raised.value.safe_details == {
        "server_id": "remote", "transport": "streamable-http", "phase": "connect",
    }
    assert raised.value.diagnostics is None
    assert raised.value.retryable is False
    assert "private-value" not in str(raised.value)
    await toolset.close_resources()


@pytest.mark.parametrize("code", (
    ErrorCode.MCP_CLEANUP_FAILED, ErrorCode.SANDBOX_CLEANUP_FAILED,
))
def test_execution_keeps_primary_error_and_secondary_cleanup_code(code: ErrorCode) -> None:
    primary = AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
    primary.__cause__ = AIError(code)
    result = _with_cleanup_diagnostic(primary, primary)
    assert result.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert result.safe_details == {"secondary_error_code": code.value}
    assert result.retryable is False


@pytest.mark.asyncio
async def test_wrapper_exit_cleanup_does_not_replace_primary_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock
    from linktools.ai.runtime._agent_executor import _close_mcp_resources

    async def fail_exit(self: MCPToolset, *args: object) -> None:
        raise RuntimeError("private cleanup response")

    monkeypatch.setattr(MCPToolset, "__aexit__", fail_exit)
    toolset = _discovery()
    close = AsyncMock()
    monkeypatch.setattr(toolset.client, "close", close)
    primary = AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
    with pytest.raises(AIError) as raised:
        try:
            raise primary
        finally:
            await toolset.__aexit__(None, None, None)
    assert raised.value is primary
    capability = _MCPCapability("remote", object(), toolset)
    with pytest.raises(AIError) as raised:
        await _close_mcp_resources((capability,), primary)
    assert raised.value is primary
    assert raised.value.__cause__.code is ErrorCode.MCP_CLEANUP_FAILED
    assert close.await_count == 1
    assert "private" not in str(raised.value.__cause__)


@pytest.mark.asyncio
async def test_inner_toolset_exit_does_not_swallow_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def cancel_exit(self: MCPToolset, *args: object) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(MCPToolset, "__aexit__", cancel_exit)
    toolset = _discovery()
    with pytest.raises(asyncio.CancelledError):
        await toolset.__aexit__(None, None, None)
    await toolset.close_resources()


@pytest.mark.asyncio
@pytest.mark.parametrize("session_failure", (False, True))
async def test_asset_projection_cleanup_follows_proven_process_shutdown(
    session_failure: bool,
) -> None:
    from linktools.ai.runtime._agent_executor import _cleanup_agent_run_resources

    closed: list[str] = []
    class Session:
        async def close(self) -> None:
            closed.append("session")
            if session_failure:
                raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED)

    class Materializer:
        async def close(self) -> None:
            closed.append("assets")

    if session_failure:
        with pytest.raises(AIError) as raised:
            await _cleanup_agent_run_resources(Session(), None, materializer=Materializer())
        assert raised.value.code is ErrorCode.SANDBOX_CLEANUP_FAILED
        assert closed == ["session"]
    else:
        await _cleanup_agent_run_resources(Session(), None, materializer=Materializer())
        assert closed == ["session", "assets"]


@pytest.mark.asyncio
async def test_unconfirmed_host_process_cleanup_retains_asset_projection() -> None:
    from linktools.ai.runtime._agent_executor import _cleanup_agent_run_resources

    class Materializer:
        async def close(self) -> None:
            pytest.fail("resources may still be in use")

    primary = AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
    primary.__cause__ = AIError(ErrorCode.MCP_CLEANUP_FAILED)
    cancellation = asyncio.CancelledError()
    cancellation.__context__ = primary
    await _cleanup_agent_run_resources(None, cancellation, materializer=Materializer())


@pytest.mark.asyncio
async def test_asset_cleanup_fault_survives_safe_primary_error_serialization() -> None:
    from linktools.ai.runtime._agent_executor import _cleanup_agent_run_resources

    cleanup = AIError(
        ErrorCode.STORAGE_UNAVAILABLE,
        safe_details={"phase": "asset_materialization_cleanup"},
    )
    class Materializer:
        async def close(self) -> None:
            raise cleanup

    primary = AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
    with pytest.raises(AIError) as raised:
        await _cleanup_agent_run_resources(None, primary, materializer=Materializer())
    assert raised.value is primary
    result = _with_cleanup_diagnostic(primary, primary).to_safe_error(operation_id="run")
    assert result.code == ErrorCode.TOOL_EFFECT_UNKNOWN.value
    assert result.safe_details == {"secondary_error_code": ErrorCode.STORAGE_UNAVAILABLE.value}
