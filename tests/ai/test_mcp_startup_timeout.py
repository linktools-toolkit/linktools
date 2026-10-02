#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit MCP startup deadlines include the pre-session SSE exchange."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from linktools.ai.errors import AIError, ErrorCode

from .test_mcp_remote import _serve, _spec, _toolsets, _wait_closed


@pytest.mark.asyncio
async def test_sse_startup_timeout_closes_invalid_endpoint_stream_and_client_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    async with _serve("sse") as remote:
        original_app = remote.app
        stream_started = asyncio.Event()
        stream_closed = asyncio.Event()
        keep_stream_open = asyncio.Event()

        async def events() -> AsyncIterator[bytes]:
            try:
                stream_started.set()
                yield (
                    b"event: endpoint\n"
                    b"data: https://different.invalid/messages?private-endpoint-token\n\n"
                )
                await keep_stream_open.wait()
            finally:
                stream_closed.set()

        async def invalid_endpoint(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http" and scope["method"] == "GET":
                await StreamingResponse(events(), media_type="text/event-stream")(
                    scope, receive, send,
                )
                return
            await original_app(scope, receive, send)

        remote.app = invalid_endpoint
        baseline_tasks = asyncio.all_tasks()

        async def connect() -> None:
            async with _toolsets(_spec(remote, init_timeout=1)):
                pytest.fail("invalid SSE endpoint must not establish a session")

        try:
            with pytest.raises(AIError) as raised:
                await asyncio.wait_for(connect(), timeout=5)
            error = raised.value
            assert error.code is ErrorCode.MCP_CONNECTION_FAILED
            assert error.safe_details == {
                "server_id": "remote", "transport": "sse", "phase": "connect",
            }
            assert "private-endpoint-token" not in str(error)
            assert stream_started.is_set(), "startup deadline expired before the SSE stream opened"
            await asyncio.wait_for(stream_closed.wait(), timeout=5)
            await _wait_closed(remote)
            assert remote.active_requests == 0
            assert remote.methods == []

            remaining = asyncio.all_tasks() - baseline_tasks
            if remaining:
                await asyncio.wait_for(
                    asyncio.gather(*remaining, return_exceptions=True), timeout=5,
                )
            assert not (asyncio.all_tasks() - baseline_tasks)
        finally:
            keep_stream_open.set()
