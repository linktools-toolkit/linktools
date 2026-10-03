#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable model-visible MCP naming projections."""

from ..core import canonical_sha256


def mcp_server_name_token(server_id: str) -> str:
    """Return the model-name token for an admitted logical MCP server ID."""
    return canonical_sha256(
        {
            "version": 1,
            "kind": "mcp-server-name",
            "server_id": server_id,
        }
    )[:24]


__all__ = ["mcp_server_name_token"]
