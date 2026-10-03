#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen invocation inputs and reusable graph declarations."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..core import ImmutableJsonMapping, JsonValue, TaskStatus
from ._definitions import TaskRef
from ._graph import TaskGraph, TaskGraphLimits, TaskNode, TaskResultRef
from ._handler import TaskDependencyState


@dataclass(frozen=True, slots=True)
class TaskDependencyCapture:
    name: str
    state: TaskDependencyState
    source_ref: TaskResultRef | None = None
    execution_id: str | None = None
    body_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("capture dependency name is required")
        if self.state.status is TaskStatus.SUCCEEDED:
            if (self.source_ref is None or not self.execution_id
                or self.body_digest != self.state.result_digest
                or self.source_ref.result_digest != self.body_digest):
                raise ValueError("captured successful dependency is incomplete")
        elif any(value is not None for value in (self.source_ref, self.execution_id, self.body_digest)):
            raise ValueError("failed capture dependency cannot contain a result")


@dataclass(frozen=True, slots=True)
class TaskInvocationInputContract:
    source_execution_id: str
    task_ref: TaskRef
    input: Mapping[str, JsonValue]
    original_input: Mapping[str, JsonValue]
    binding: Mapping[str, JsonValue]
    dependencies: tuple[TaskDependencyCapture, ...] = ()
    input_mode: str = "fixed_input"
    excluded_dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.input_mode not in {"fixed_input", "reproject_input"}:
            raise ValueError("capture input mode is invalid")
        for field in ("input", "original_input", "binding"):
            object.__setattr__(self, field, ImmutableJsonMapping(getattr(self, field)))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        if any(not isinstance(name, str) or not name for name in self.excluded_dependencies):
            raise ValueError("excluded dependency names are invalid")
        object.__setattr__(self, "excluded_dependencies", tuple(sorted(set(self.excluded_dependencies))))
        if len({item.name for item in self.dependencies}) != len(self.dependencies):
            raise ValueError("capture dependency names must be unique")


@dataclass(frozen=True, slots=True)
class TaskGraphTemplate:
    nodes: tuple[TaskNode, ...]
    limits: TaskGraphLimits | None = None
    task_contracts: tuple[Mapping[str, JsonValue], ...] = ()
    expander_contracts: tuple[Mapping[str, JsonValue], ...] = ()
    context_policy: str = "clean"

    def __post_init__(self) -> None:
        if self.context_policy not in {"captured", "clean"}:
            raise ValueError("graph template context policy is invalid")
        object.__setattr__(self, "nodes", TaskGraph("template", self.nodes).nodes)
        object.__setattr__(self, "task_contracts", tuple(ImmutableJsonMapping(item) for item in self.task_contracts))
        object.__setattr__(self, "expander_contracts", tuple(ImmutableJsonMapping(item) for item in self.expander_contracts))
        if self.limits is not None and not isinstance(self.limits, TaskGraphLimits):
            raise TypeError("template limits must be TaskGraphLimits")


__all__ = ["TaskDependencyCapture", "TaskInvocationInputContract", "TaskGraphTemplate"]
