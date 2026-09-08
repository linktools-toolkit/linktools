#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness memory adapters for Runtime-owned persistence."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.toolsets import AbstractToolset, FilteredToolset
from pydantic_ai_harness.memory import (
    Memory,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryStore,
)

from ._harness import current_tool_operation_id


class HarnessMemoryStoreAdapter:
    """Map Harness mutation identity to the stable Runtime tool operation."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        return await self._store.read(path, max_chars=max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        return await self._store.get_operation(self._operation(operation))

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
            operation=None if operation is None else self._operation(operation),
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
            operation=None if operation is None else self._operation(operation),
        )

    async def list_paths(self, prefix: str = "", *, limit: int) -> list[str]:
        return await self._store.list_paths(prefix, limit=limit)

    @staticmethod
    def _operation(operation: MemoryOperation) -> MemoryOperation:
        stable_id = current_tool_operation_id()
        if stable_id is None:
            return operation
        return MemoryOperation(id=stable_id, fingerprint=operation.fingerprint)


@dataclass
class HarnessSelectedMemory(Memory[None]):
    """Harness Memory with Runtime-selected tool exposure."""

    selected_tool_names: tuple[str, ...] = ()

    def get_toolset(self) -> AbstractToolset[None] | None:
        toolset = super().get_toolset()
        if toolset is None:
            return None
        selected = frozenset(self.selected_tool_names)
        return FilteredToolset(
            toolset,
            lambda _ctx, tool: tool.name in selected,
        )


def build_harness_memory(
    store: MemoryStore,
    *,
    selected_tool_names: tuple[str, ...],
    capability_id: str,
) -> HarnessSelectedMemory:
    """Build the Harness Memory capability over one Runtime memory store."""
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


__all__ = [
    "HarnessMemoryStoreAdapter",
    "HarnessSelectedMemory",
    "build_harness_memory",
]
