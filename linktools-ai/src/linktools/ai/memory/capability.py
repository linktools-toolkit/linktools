#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MemoryCapability: cross-session long-term notes, exposed as tools.

Stores notes as a plain file under `root` -- no storage abstraction, callers
decide `root` (typically a subdirectory of the session's workspace)."""

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

_MEMORY_FILE = "notes.md"


@dataclass
class MemoryCapability(AbstractCapability[None]):
    root: Path

    def get_toolset(self) -> FunctionToolset:
        root = self.root

        toolset: FunctionToolset = FunctionToolset()

        async def read_memory() -> str:
            """Return the current long-term memory notes (empty string if none written yet)."""
            path = root / _MEMORY_FILE
            return path.read_text(encoding="utf-8") if path.exists() else ""

        async def write_memory(content: str) -> dict:
            """Overwrite the long-term memory notes with `content`."""
            path = root / _MEMORY_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"written": True}

        for fn in (read_memory, write_memory):
            toolset.add_function(fn)
        return toolset
