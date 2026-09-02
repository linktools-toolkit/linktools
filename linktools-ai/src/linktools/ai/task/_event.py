#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable TaskGraph event contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..core import TaskStatus, validate_lease_owner
from ..errors import AIError, ErrorCode


class TaskEventType(str, Enum):
    GRAPH_ADMITTED = "GRAPH_ADMITTED"
    GRAPH_CHANGED = "GRAPH_CHANGED"
    NODE_CHANGED = "NODE_CHANGED"


@dataclass(frozen=True, slots=True)
class TaskEvent:
    version: int
    graph_id: str
    sequence: int
    event_type: TaskEventType
    occurred_at: datetime
    status: TaskStatus
    previous_status: "TaskStatus | None" = None
    node_id: "str | None" = None
    owner: "str | None" = None
    fence: int = 0
    execution_id: "str | None" = None
    result_digest: "str | None" = None
    error_code: "str | None" = None
    error_digest: "str | None" = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, int)
            or isinstance(self.version, bool)
            or self.version < 1
        ):
            raise ValueError("task event version is invalid")
        if self.version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if not isinstance(self.graph_id, str) or not self.graph_id.strip():
            raise ValueError("task event graph id is required")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence < 1:
            raise ValueError("task event sequence must be positive")
        if not isinstance(self.event_type, TaskEventType):
            raise TypeError("task event type is invalid")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.tzinfo is None:
            raise ValueError("task event time must be timezone-aware")
        if not isinstance(self.status, TaskStatus):
            raise TypeError("task event status is invalid")
        if self.previous_status is not None and not isinstance(self.previous_status, TaskStatus):
            raise TypeError("task event previous status is invalid")
        if self.event_type is TaskEventType.NODE_CHANGED:
            if not isinstance(self.node_id, str) or not self.node_id.strip():
                raise ValueError("task node event requires a node id")
        elif self.node_id is not None:
            raise ValueError("task graph event cannot carry a node id")
        if isinstance(self.fence, bool) or not isinstance(self.fence, int) or self.fence < 0:
            raise ValueError("task event fence must be non-negative")
        if self.owner is not None:
            try:
                validate_lease_owner(self.owner)
            except AIError as error:
                raise ValueError("task event owner is invalid") from error
        if self.execution_id is not None and (
            not isinstance(self.execution_id, str) or not self.execution_id.strip()
        ):
            raise ValueError("task event execution id is invalid")


__all__ = ["TaskEvent", "TaskEventType"]
