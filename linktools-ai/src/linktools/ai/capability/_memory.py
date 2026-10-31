#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped memory tools over a host-owned store."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, FilteredToolset, FunctionToolset
from pydantic_ai_harness.memory import Memory, MemoryFile, MemoryMutation, MemoryOperation, MemoryStore

from ..core import validate_memory_scope
from ..errors import AIError, ErrorCode
from ._context import AgentContext
from ._tool_metadata import tool_metadata
from ._tool_signal import ToolCallRetry


_MEMORY_TOOLS: dict[str, Mapping[str, object]] = {
    "delete_memory": tool_metadata(tool_class="memory", effect_policy="replay_safe", compaction_keep_result=True),
    "read_memory": tool_metadata(tool_class="memory", effect_policy="none", plan_safe=True, compaction_keep_result=True),
    "search_memory": tool_metadata(tool_class="memory", effect_policy="none", plan_safe=True, compaction_keep_result=True),
    "write_memory": tool_metadata(tool_class="memory", effect_policy="replay_safe", compaction_keep_result=True),
}


class MemoryCapability(AbstractCapability[AgentContext[object]]):
    """Expose memory tools using the scope captured by the owning Runtime.

    Runtime owns storage; admitted executions retain their original scope.
    Imported input contexts use isolated captured memory. Omitting this
    declaration disables memory for ordinary new executions.
    """

    id = "runtime.memory"

    def __init__(self, *, scope: str = "default") -> None:
        self._scope = validate_memory_scope(scope)

    @property
    def scope(self) -> str:
        return self._scope

    def tool_definitions(self) -> tuple[Tool[Any], ...]:
        """Declare the same native tools that are bound for each execution."""
        return _memory(None, tuple(_MEMORY_TOOLS)).tool_definitions()

    @classmethod
    def select_tools(cls, names: Sequence[str]) -> tuple[str, ...]:
        """Select this capability's tools from an Agent selection."""
        return tuple(_MEMORY_TOOLS) if "*" in names else tuple(name for name in _MEMORY_TOOLS if name in names)

    def bind(self, selected_tool_names: Sequence[str], *, store: MemoryStore) -> AbstractCapability[Any]:
        """Borrow the store selected from the admitted execution, not the host default."""
        selected = frozenset(selected_tool_names)
        if selected.difference(_MEMORY_TOOLS):
            raise ValueError("unknown memory tools")
        return _memory(store, tuple(name for name in _MEMORY_TOOLS if name in selected))

    def for_agent(self, agent: object) -> "MemoryCapability":
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                      safe_details={"reason": "memory_declaration_requires_host"})


class _MemoryStoreAdapter:
    def __init__(self, store: MemoryStore | None) -> None:
        self._store = store

    def _bound_store(self) -> MemoryStore:
        if self._store is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._store

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        return await self._bound_store().read(path, max_chars=max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        return await self._bound_store().get_operation(operation)

    async def write(self, path: str, content: str, *, expected_version: str | None,
                    operation: MemoryOperation | None = None) -> MemoryMutation:
        return await self._bound_store().write(path, content, expected_version=expected_version, operation=operation)

    async def delete(self, path: str, *, expected_version: str | None,
                     operation: MemoryOperation | None = None) -> MemoryMutation:
        return await self._bound_store().delete(path, expected_version=expected_version, operation=operation)

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        return await self._bound_store().list_paths(prefix, limit=limit)


def _validate_memory_content(ctx: RunContext[None], content: str, **_kwargs: object) -> None:
    """Reject unsupported model-authored text before accessing storage."""
    del ctx
    if "\x00" in content:
        raise ToolCallRetry("Memory content cannot contain NUL characters. Remove them from content and retry.")
    try:
        content.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ToolCallRetry("Memory content must be valid UTF-8 text. Replace invalid Unicode characters in content and retry.") from None


@dataclass
class _BoundMemory(Memory[None]):
    selected_tool_names: tuple[str, ...] = ()

    def tool_definitions(self) -> tuple[Tool[Any], ...]:
        toolset = cast("FunctionToolset[None]", super().get_toolset())
        for name, metadata in _MEMORY_TOOLS.items():
            tool = toolset.tools[name]
            if name == "write_memory":
                tool.args_validator = _validate_memory_content
            tool.metadata = {**(tool.metadata or {}), **metadata}
        return tuple(toolset.tools[name] for name in _MEMORY_TOOLS)

    def get_toolset(self) -> AbstractToolset[None] | None:
        self.tool_definitions()
        toolset = super().get_toolset()
        if toolset is None:
            return None
        selected = frozenset(self.selected_tool_names)
        return FilteredToolset(toolset, lambda _ctx, tool: tool.name in selected)


def _memory(store: MemoryStore | None, selected: tuple[str, ...]) -> _BoundMemory:
    guidance = "Use the available persistent-memory tools when durable notes are useful: "
    guidance += ", ".join(f"`{name}`" for name in selected) + "."
    return _BoundMemory(store=_MemoryStoreAdapter(store), agent_name="memory", inject_memory=False,
                        guidance=guidance, selected_tool_names=selected, id="linktools.ai.memory")


__all__ = ["MemoryCapability"]
