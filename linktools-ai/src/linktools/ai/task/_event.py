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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("task event sequence must be positive")
        if not isinstance(self.event_type, TaskEventType):
            raise TypeError("task event type is invalid")
        if (
            not isinstance(self.occurred_at, datetime)
            or self.occurred_at.utcoffset() is None
        ):
            raise ValueError("task event time must be timezone-aware")
        if not isinstance(self.status, TaskStatus):
            raise TypeError("task event status is invalid")
        if self.previous_status is not None and not isinstance(
            self.previous_status, TaskStatus
        ):
            raise TypeError("task event previous status is invalid")
        if (
            isinstance(self.fence, bool)
            or not isinstance(self.fence, int)
            or self.fence < 0
        ):
            raise ValueError("task event fence must be non-negative")
        if self.result_digest is not None and not _is_sha256(self.result_digest):
            raise ValueError("task event result digest must be lowercase SHA-256")
        if self.error_digest is not None and not _is_sha256(self.error_digest):
            raise ValueError("task event error digest must be lowercase SHA-256")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code.strip()
        ):
            raise ValueError("task event error code is invalid")

        if self.event_type is TaskEventType.GRAPH_ADMITTED:
            if self.status not in {TaskStatus.PENDING, TaskStatus.SUCCEEDED}:
                raise ValueError("task graph admission event status is invalid")
            if self.previous_status is not None:
                raise ValueError(
                    "task graph admission event cannot have previous status"
                )
            self._validate_graph_only_fields()
            return

        if self.event_type is TaskEventType.GRAPH_CHANGED:
            if self.previous_status is None:
                raise ValueError("task graph change event requires previous status")
            if self.status is TaskStatus.READY or self.previous_status is TaskStatus.READY:
                raise ValueError(
                    "task graph change event cannot use node-only READY status"
                )
            if self.previous_status is self.status:
                raise ValueError("task graph change event requires a status transition")
            self._validate_graph_only_fields()
            return

        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("task node event requires a node id")
        if self.previous_status is None:
            raise ValueError("task node event requires previous status")
        if self.owner is not None:
            try:
                validate_lease_owner(self.owner)
            except AIError as error:
                raise ValueError("task event owner is invalid") from error
        if self.execution_id is not None and (
            not isinstance(self.execution_id, str) or not self.execution_id.strip()
        ):
            raise ValueError("task event execution id is invalid")
        self._validate_node_state_fields()

    def _validate_graph_only_fields(self) -> None:
        if self.node_id is not None:
            raise ValueError("task graph event cannot carry a node id")
        if (
            self.owner is not None
            or self.fence != 0
            or self.execution_id is not None
            or self.result_digest is not None
            or self.error_code is not None
            or self.error_digest is not None
        ):
            raise ValueError("task graph event cannot carry node state")

    def _validate_node_state_fields(self) -> None:
        if self.status in {TaskStatus.PENDING, TaskStatus.READY}:
            if (
                self.owner is not None
                or self.fence != 0
                or self.execution_id is not None
                or self.result_digest is not None
                or self.error_code is not None
                or self.error_digest is not None
            ):
                raise ValueError("pending task event carries active or terminal state")
            return
        if self.status is TaskStatus.RUNNING:
            if (
                self.owner is None
                or self.fence < 1
                or self.result_digest is not None
                or self.error_code is not None
                or self.error_digest is not None
            ):
                raise ValueError("running task event state is invalid")
            return
        if self.status is TaskStatus.SUCCEEDED:
            if (
                self.owner is not None
                or self.fence < 1
                or self.result_digest is None
                or self.error_code is not None
                or self.error_digest is not None
            ):
                raise ValueError("successful task event state is invalid")
            return
        if self.status is TaskStatus.FAILED:
            if (
                self.owner is not None
                or self.fence < 1
                or self.result_digest is not None
                or self.error_code is None
                or self.error_digest is None
            ):
                raise ValueError("failed task event state is invalid")
            return
        if self.status is TaskStatus.BLOCKED:
            if (
                self.owner is not None
                or self.execution_id is not None
                or self.result_digest is not None
                or self.error_code is None
            ):
                raise ValueError("blocked task event state is invalid")
            return
        if self.status is TaskStatus.CANCELLED:
            if (
                self.owner is not None
                or self.result_digest is not None
                or self.error_code is not None
                or self.error_digest is not None
            ):
                raise ValueError("cancelled task event state is invalid")
            return
        raise ValueError("task node event status is unsupported")


__all__ = ["TaskEvent", "TaskEventType"]
