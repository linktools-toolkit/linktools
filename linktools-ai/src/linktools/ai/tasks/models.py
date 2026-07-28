"""Unified task plan and execution values."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskNode:
    id: str
    payload: Any = None
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskPlan:
    id: str
    nodes: tuple[TaskNode, ...]


@dataclass(frozen=True, slots=True)
class TaskExecution:
    id: str
    plan_id: str
    node_id: str
    status: str
    owner: str | None = None
    fence: int = 0
    attempt: int = 0
    result: Any = None
