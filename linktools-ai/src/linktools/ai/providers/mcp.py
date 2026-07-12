#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCPServerSpecProvider: source-agnostic surface for MCPServerSpec objects."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..registry.mcp import MCPServerSpec


@runtime_checkable
class MCPServerSpecProvider(Protocol):
    """Provides MCPServerSpec objects from any configuration source."""

    async def list_ids(self) -> "tuple[str, ...]": ...

    async def get(self, server_id: str) -> "MCPServerSpec": ...
