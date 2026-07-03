#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ToolSearchCapability: naive substring tool search, exposed as a `search_tools`
meta-tool. This is the spec's explicit v1 design -- real semantic search is out of
scope. Off by default (`enable_tool_search=False` in BaseAgent)."""

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class ToolSearchCapability(AbstractCapability[None]):
    tool_names: "tuple[str, ...]" = ()

    def get_toolset(self) -> FunctionToolset:
        names = self.tool_names

        toolset: FunctionToolset = FunctionToolset()

        async def search_tools(query: str) -> "list[str]":
            """Search available tool names for a substring match (case-insensitive)."""
            needle = query.lower()
            return [name for name in names if needle in name.lower()]

        toolset.add_function(search_tools)
        return toolset
