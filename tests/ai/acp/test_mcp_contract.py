#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import acp.schema as schema
import pytest

from linktools.ai.acp.sessions import mcp_spec


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
