#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derived terminal counts owned by the Task graph record."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ..core import TaskStatus
from ..errors import AIError, ErrorCode
from ..task import TaskNode, TaskNodeView
from .state import StoredRecord


@dataclass(frozen=True)
class _TaskSummary:
    counts: Mapping[TaskStatus, int]
    propagated_failed: int
    propagated_blocked: int

    @classmethod
    def build(cls, states: tuple[TaskNodeView, ...], definitions: tuple[TaskNode, ...]) -> "_TaskSummary":
        return cls.from_states(states, {node.node_id: node.failure_policy for node in definitions})

    @classmethod
    def from_states(cls, states: tuple[TaskNodeView, ...], policies: Mapping[str, str]) -> "_TaskSummary":
        counts = {status: 0 for status in TaskStatus}
        failed = blocked = 0
        for node in states:
            counts[node.status] += 1
            if node.status in {TaskStatus.FAILED, TaskStatus.BLOCKED}:
                if node.node_id not in policies:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if policies[node.node_id] == "propagate":
                    failed += node.status is TaskStatus.FAILED
                    blocked += node.status is TaskStatus.BLOCKED
        return cls(counts, failed, blocked)

    @classmethod
    def read(cls, record: StoredRecord) -> "_TaskSummary | None":
        if "task_summary" not in record.data:
            return None
        raw = record.data["task_summary"]
        if not isinstance(raw, Mapping) or not {"counts", "propagated_failed", "propagated_blocked"}.issubset(raw):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        counts = raw["counts"]
        if not isinstance(counts, Mapping) or not set(counts).issubset({status.value for status in TaskStatus}):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values = (*counts.values(), raw["propagated_failed"], raw["propagated_blocked"])
        if any(type(value) is not int or value < 0 for value in values):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = cls({status: counts.get(status.value, 0) for status in TaskStatus}, raw["propagated_failed"], raw["propagated_blocked"])
        if result.propagated_failed > result.counts[TaskStatus.FAILED] or result.propagated_blocked > result.counts[TaskStatus.BLOCKED]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result

    def attach(self, record: StoredRecord) -> StoredRecord:
        return replace(record, data={**record.data, "task_summary": {
            "counts": {status.value: count for status, count in self.counts.items() if count},
            "propagated_failed": self.propagated_failed,
            "propagated_blocked": self.propagated_blocked,
        }})

    def transition(self, before: TaskNodeView, after: TaskNodeView, definition: TaskNode) -> "_TaskSummary":
        counts = dict(self.counts)
        if counts[before.status] < 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        counts[before.status] -= 1
        counts[after.status] += 1
        propagate = definition.failure_policy == "propagate"
        return _TaskSummary(
            counts,
            self.propagated_failed + propagate * ((after.status is TaskStatus.FAILED) - (before.status is TaskStatus.FAILED)),
            self.propagated_blocked + propagate * ((after.status is TaskStatus.BLOCKED) - (before.status is TaskStatus.BLOCKED)),
        )

    def status(self) -> TaskStatus:
        for candidates, status in (
            ((TaskStatus.RECOVERY_REQUIRED,), TaskStatus.RECOVERY_REQUIRED),
            ((TaskStatus.RUNNING, TaskStatus.WAITING), TaskStatus.RUNNING),
            ((TaskStatus.PENDING, TaskStatus.READY), TaskStatus.PENDING),
        ):
            if any(self.counts[candidate] for candidate in candidates):
                return status
        if self.propagated_failed:
            return TaskStatus.FAILED
        if self.propagated_blocked:
            return TaskStatus.BLOCKED
        return TaskStatus.CANCELLED if self.counts[TaskStatus.CANCELLED] else TaskStatus.SUCCEEDED

    def validate(self, states: tuple[TaskNodeView, ...], definitions: tuple[TaskNode, ...]) -> None:
        if self != self.build(states, definitions):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
