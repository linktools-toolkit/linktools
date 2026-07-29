"""Unified task plan and execution values."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from ..storage.coordination.lease import Lease


class TaskStatus(StrEnum):
    READY = "ready"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


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
    status: TaskStatus
    lease: Lease = field(default_factory=Lease)
    attempt: int = 0
    result: Any = None
    error: Any = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            object.__setattr__(self, "status", TaskStatus(self.status))

    @property
    def owner(self) -> str | None:
        return self.lease.owner

    @property
    def fence(self) -> int:
        return self.lease.fence
