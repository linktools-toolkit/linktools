#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-node runner contracts shared by TaskGraph schedulers and Runtime adapters."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core import CorrelationData, JsonValue, Principal
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ._graph import TaskDependencyResult, TaskNode


@dataclass(frozen=True, slots=True)
class TaskNodeRunResult:
    result_digest: str
    execution_id: "str | None" = None
    result_payload: "StoredPayload | None" = None
    expanded_nodes: "tuple[TaskNode, ...]" = ()
    deferred: bool = False

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.result_digest) is None:
            raise ValueError("task node result digest is invalid")
        if self.execution_id is not None and (
            not isinstance(self.execution_id, str) or not self.execution_id.strip()
        ):
            raise ValueError("task node result execution id is invalid")
        if (
            self.result_payload is not None
            and self.result_payload.digest != self.result_digest
        ):
            raise ValueError("task node result payload digest does not match result")
        expanded_nodes = tuple(self.expanded_nodes)
        if any(not isinstance(node, TaskNode) for node in expanded_nodes):
            raise TypeError("expanded task nodes are invalid")
        object.__setattr__(self, "expanded_nodes", expanded_nodes)


class TaskNodeRunError(AIError):
    """A task-node failure tied to one concrete execution."""

    def __init__(
        self,
        code: ErrorCode,
        execution_id: str,
        *,
        safe_details: "Mapping[str, JsonValue] | None" = None,
    ) -> None:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise ValueError("task node failure execution id is required")
        super().__init__(code, safe_details=safe_details)
        self.execution_id = execution_id


@runtime_checkable
class TaskNodeRunControl(Protocol):
    async def handoff_execution(self, execution_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class TaskNodeInvocation:
    node: TaskNode
    graph_id: str
    principal: Principal
    correlation: CorrelationData
    dependency_results: "Mapping[str, TaskDependencyResult]"


class TaskNodeRunner(Protocol):
    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult: ...

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult: ...

    async def cancel(self, invocation: TaskNodeInvocation) -> None: ...


__all__ = [
    "TaskNodeInvocation",
    "TaskNodeRunControl",
    "TaskNodeRunError",
    "TaskNodeRunResult",
    "TaskNodeRunner",
]
