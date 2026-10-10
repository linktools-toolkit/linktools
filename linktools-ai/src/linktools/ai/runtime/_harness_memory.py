#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness memory adapters for Runtime-owned persistence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset, FunctionToolset
from pydantic_ai_harness.memory import (
    Memory,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryStore,
)

from ..capability import (
    TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY,
    TOOL_PLAN_SAFE_METADATA_KEY,
    ToolCallRetry,
    tool_metadata,
)

_MEMORY_TOOL_DECLARATIONS: dict[str, Mapping[str, object]] = {
    "delete_memory": tool_metadata(compaction_keep_result=True),
    "read_memory": tool_metadata(
        plan_safe=True,
        compaction_keep_result=True,
    ),
    "search_memory": tool_metadata(
        plan_safe=True,
        compaction_keep_result=True,
    ),
    "write_memory": tool_metadata(compaction_keep_result=True),
}


def select_harness_memory_tools(
    allow_tools: Sequence[str],
) -> tuple[str, ...]:
    """Select the Memory tools owned by this Runtime capability."""
    if "*" in allow_tools:
        return tuple(_MEMORY_TOOL_DECLARATIONS)
    return tuple(
        name for name in _MEMORY_TOOL_DECLARATIONS if name in allow_tools
    )


class HarnessMemoryStoreAdapter:
    """Expose the Runtime memory store through Harness' public contract."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        return await self._store.read(path, max_chars=max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        return await self._store.get_operation(operation)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await self._store.write(
            path,
            content,
            expected_version=expected_version,
            operation=operation,
        )

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await self._store.delete(
            path,
            expected_version=expected_version,
            operation=operation,
        )

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        return await self._store.list_paths(prefix, limit=limit)


def _validate_memory_content(
    ctx: RunContext[None],
    content: str,
    **_kwargs: object,
) -> None:
    """Reject unsupported model-authored memory text before accessing storage."""
    del ctx
    if "\x00" in content:
        raise ToolCallRetry(
            "Memory content cannot contain NUL characters. Remove them "
            "from content and retry."
        )
    try:
        content.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ToolCallRetry(
            "Memory content must be valid UTF-8 text. Replace invalid "
            "Unicode characters in content and retry."
        ) from None


@dataclass
class HarnessSelectedMemory(Memory[None]):
    """Harness Memory with Runtime-selected tool exposure."""

    selected_tool_names: tuple[str, ...] = ()

    def get_toolset(self) -> AbstractToolset[None] | None:
        toolset = cast(
            "FunctionToolset[None] | None",
            super().get_toolset(),
        )
        if toolset is None:
            return None
        for name in self.selected_tool_names:
            tool = toolset.tools[name]
            if name == "write_memory":
                tool.args_validator = _validate_memory_content
            toolset.tools[name].metadata = tool_metadata(
                base=toolset.tools[name].metadata,
                **_memory_metadata_kwargs(name),
            )
        selected = frozenset(self.selected_tool_names)
        return FilteredToolset(
            toolset,
            lambda _ctx, tool: tool.name in selected,
        )

def build_harness_memory(
    store: MemoryStore,
    *,
    allow_tools: Sequence[str],
    capability_id: str,
) -> HarnessSelectedMemory:
    """Build the Harness Memory capability over one Runtime memory store."""
    selected_tool_names = select_harness_memory_tools(allow_tools)
    guidance = (
        "Use the available persistent-memory tools when durable notes are useful: "
        + ", ".join(f"`{name}`" for name in selected_tool_names)
        + "."
    )
    return HarnessSelectedMemory(
        store=HarnessMemoryStoreAdapter(store),
        agent_name="memory",
        inject_memory=False,
        guidance=guidance,
        selected_tool_names=selected_tool_names,
        id=capability_id,
    )


def _memory_metadata_kwargs(name: str) -> dict[str, object]:
    metadata = _MEMORY_TOOL_DECLARATIONS[name]
    return {
        "plan_safe": metadata.get(TOOL_PLAN_SAFE_METADATA_KEY),
        "compaction_keep_result": metadata.get(
            TOOL_COMPACTION_KEEP_RESULT_METADATA_KEY
        ),
    }


__all__ = [
    "HarnessMemoryStoreAdapter",
    "HarnessSelectedMemory",
    "build_harness_memory",
    "select_harness_memory_tools",
]
