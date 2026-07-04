#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""PlanCapability: self-directed TODO/subtask tracking, exposed as tools.

Persists via AgentArtifactStore (resource/protocols.py) -- a general
content-addressable store, not session-turn history."""

import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import FunctionToolset

from ..resource.protocols import ArtifactRef

if TYPE_CHECKING:
    from ..resource.protocols import AgentArtifactStore


@dataclass
class PlanCapability(AbstractCapability[None]):
    session_id: str
    artifact_store: "AgentArtifactStore"

    def _ref(self) -> ArtifactRef:
        return ArtifactRef(domain="session", scope=self.session_id, kind="plan", path="todos.json")

    async def _load(self) -> "list[dict[str, Any]]":
        content = await self.artifact_store.get(self._ref())
        if content is None:
            return []
        return json.loads(content)

    async def _save(self, todos: "list[dict[str, Any]]") -> None:
        content = json.dumps(todos).encode("utf-8")
        await self.artifact_store.put(self._ref(), content, idempotency_key=str(uuid.uuid4()))

    def get_toolset(self) -> FunctionToolset:
        toolset: FunctionToolset = FunctionToolset()

        async def write_todos(todos: "list[dict[str, Any]]") -> "dict[str, Any]":
            """Replace the current TODO list with `todos` (each item: id, content, status)."""
            await self._save(todos)
            return {"todos": todos}

        async def update_todo(todo_id: str, status: str) -> "dict[str, Any]":
            """Update one TODO's status by id. Returns the updated TODO, or an error if not found."""
            todos = await self._load()
            for todo in todos:
                if todo.get("id") == todo_id:
                    todo["status"] = status
                    await self._save(todos)
                    return todo
            return {"error": f"no todo with id {todo_id!r}"}

        async def list_todos() -> "list[dict[str, Any]]":
            """Return the current TODO list."""
            return await self._load()

        for fn in (write_todos, update_todo, list_todos):
            toolset.add_function(fn)
        return toolset
