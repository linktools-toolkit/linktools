#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import socket

import acp.schema as schema
import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

from linktools.ai.agent.mcp.client import MCPClient
from linktools.ai.agent.mcp.connection import MCPConnectionPool
from linktools.ai.agent.mcp.spec import MCPServerSpec
from linktools.ai.acp.mcp import mcp_spec


@pytest.mark.parametrize(
    ("descriptor", "transport"),
    (
        (schema.McpServerStdio(name="stdio", command="server", args=[], env=[]), "stdio"),
        (schema.McpServerHttp(name="http", url="https://example.test", headers=[]), "http"),
        (schema.McpServerSse(name="sse", url="https://example.test", headers=[]), "sse"),
    ),
)
def test_mcp_descriptor_converter_supports_advertised_transports(descriptor, transport) -> None:
    assert mcp_spec(descriptor).transport == transport


def test_mcp_acp_descriptor_is_rejected_explicitly() -> None:
    with pytest.raises(Exception) as raised:
        mcp_spec(schema.McpServerAcp(name="acp", serverId="server-1"))

    assert raised.value.data["reason"] == "unsupported_mcp_transport"
    assert raised.value.data["transport"] == "acp"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport", "path", "app_factory"),
    (
        ("http", "/mcp", lambda server: server.streamable_http_app()),
        ("sse", "/sse", lambda server: server.sse_app()),
    ),
)
async def test_advertised_mcp_transport_connects_lists_calls_and_closes(
    transport: str,
    path: str,
    app_factory,
) -> None:
    server = FastMCP(f"acp-{transport}")

    @server.tool()
    def hello(name: str) -> str:
        return f"hello {name}"

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    process = uvicorn.Server(
        uvicorn.Config(
            app_factory(server),
            host="127.0.0.1",
            port=port,
            log_level="critical",
        )
    )
    process_task = asyncio.create_task(process.serve())
    try:
        for _ in range(200):
            if process.started:
                break
            await asyncio.sleep(0.01)
        assert process.started
        pool = MCPConnectionPool()
        spec = MCPServerSpec(
            id=transport,
            name=transport,
            transport=transport,
            url=f"http://127.0.0.1:{port}{path}",
        )
        handle = await pool.get_toolset(spec)
        discovery = await MCPClient(handle.toolset).discover(
            server_id=transport,
            connection_ref=handle.connection_ref,
        )
        assert discovery.verified
        assert [tool.name for tool in discovery.tools] == ["hello"]
        assert await MCPClient(handle.toolset).call(
            server_id=transport,
            tool_name="hello",
            arguments={"name": "ACP"},
        ) == "hello ACP"
        await pool.close()
        assert not pool._toolsets
    finally:
        process.should_exit = True
        await asyncio.wait_for(process_task, timeout=10)
