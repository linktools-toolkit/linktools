#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Explicit MCP endpoint registry."""


class MCPRegistry:
    def __init__(self, endpoints: "tuple[str, ...]" = ()) -> None:
        self._endpoints = tuple(sorted(endpoints))

    def resolve(self, pattern: str) -> "tuple[str, ...]":
        if pattern == "*":
            return self._endpoints
        return tuple(endpoint for endpoint in self._endpoints if endpoint == pattern)


__all__ = ["MCPRegistry"]
