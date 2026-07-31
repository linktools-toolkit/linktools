#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime storage composition with lazy optional stores."""


from typing import TYPE_CHECKING
import asyncio
from dataclasses import dataclass
from pathlib import Path
from ..execution.persistence.local import LocalExecutionBackend

if TYPE_CHECKING:
    from ..execution.store import ExecutionStore
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ..artifact.store import ArtifactStore
    from ..agent.memory.store import MemoryStore
    from ..tasks.store import TaskStore
    from ..agent.tool.store import ToolStateStore


@dataclass(frozen=True, slots=True)
class RuntimeStorage:
    execution: "ExecutionStore"
    tools: "ToolStateStore | None" = None
    tasks: "TaskStore | None" = None
    memory: "MemoryStore | None" = None
    artifacts: "ArtifactStore | None" = None


class LocalDirectoryStorage(RuntimeStorage):
    """Single-process local storage; construction does not create directories."""

    def __init__(self, root: "str | Path" = ".linktools", *, tools=None, tasks=None, memory=None, artifacts=None) -> None:
        object.__setattr__(self, "root", Path(root))
        super().__init__(
            execution=LocalExecutionBackend(self.root / "execution"),
            tools=tools,
            tasks=tasks,
            memory=memory,
            artifacts=artifacts,
        )

    async def initialize_storage(self) -> None:
        await asyncio.to_thread((self.root / "execution").mkdir, parents=True, exist_ok=True)
        for store in (self.tools, self.tasks, self.memory, self.artifacts):
            if store is not None:
                await store.initialize_storage()


class SqlAlchemyRuntimeStorage(RuntimeStorage):
    """SQLAlchemy composition root; schema setup is explicit."""

    def __init__(self, execution: "ExecutionStore", *, tools=None, tasks=None, memory=None, artifacts=None) -> None:
        super().__init__(execution=execution, tools=tools, tasks=tasks, memory=memory, artifacts=artifacts)

    async def initialize_storage(self, engine: "AsyncEngine") -> None:
        await self.execution.initialize_storage(engine)
        for store in (self.tools, self.tasks, self.memory, self.artifacts):
            if store is not None:
                await store.initialize_storage(engine)


__all__ = ["LocalDirectoryStorage", "RuntimeStorage", "SqlAlchemyRuntimeStorage"]
