#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic TaskGraph service contracts."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from ..core import JsonValue, Page, Principal, validate_idempotency_key
from ._handler import TaskEffectResolution
from ._event import TaskEvent
from ._graph import (
    CancelGraphRequest,
    RecoverGraphRequest,
    TaskInputSupplyRequest,
    TaskGraphHandle,
    TaskGraphLaunch,
    TaskGraph,
    TaskGraphRequest,
    TaskGraphResult,
    TaskGraphState,
    TaskGraphView,
    TaskNodeInfo,
    TaskNodeView,
)


@dataclass(frozen=True, slots=True)
class TaskEffectResolutionRequest:
    principal: Principal
    expected_fence: int
    resolution: TaskEffectResolution
    idempotency_key: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.expected_fence, bool)
            or not isinstance(self.expected_fence, int)
            or self.expected_fence < 1
        ):
            raise ValueError("task effect fence is invalid")
        if not isinstance(self.resolution, TaskEffectResolution):
            raise TypeError("task effect resolution is invalid")
        validate_idempotency_key(self.idempotency_key)


class TaskGraphQueryService(Protocol):
    async def inspect(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphView: ...

    async def state(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> TaskGraphState: ...

    async def result_header(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> tuple[TaskGraph, int]: ...

    async def result_node_states(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        principal: Principal,
    ) -> tuple[TaskNodeView, ...]: ...

    async def list_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> Page[TaskEvent]: ...

    def stream_events(
        self,
        graph_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[TaskEvent]: ...

    async def wait(
        self,
        graph_id: str,
        *,
        principal: Principal,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult: ...


class TaskGraphService(TaskGraphQueryService, Protocol):
    async def start(self, request: TaskGraphRequest) -> TaskGraphResult: ...

    async def run(
        self,
        request: TaskGraphRequest,
        *,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult: ...

    async def recovery_nodes(
        self,
        graph_id: str,
        *,
        principal: Principal,
    ) -> tuple[TaskNodeInfo, ...]: ...

    async def recover(
        self,
        graph_id: str,
        request: RecoverGraphRequest,
    ) -> TaskGraphResult: ...

    async def resume(
        self,
        graph_id: str,
        node_id: str,
        request: TaskInputSupplyRequest,
    ) -> TaskGraphResult: ...

    async def resolve_effect(
        self,
        graph_id: str,
        node_id: str,
        request: TaskEffectResolutionRequest,
    ) -> TaskGraphResult: ...

    async def cancel(
        self,
        graph_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView: ...

    async def cancel_node(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        request: CancelGraphRequest,
    ) -> TaskGraphView: ...

    async def settle_execution_cancellation(
        self,
        graph_id: str,
        node_id: str,
        execution_id: str,
        request: CancelGraphRequest,
        *,
        cancel_confirmed: bool | None,
    ) -> TaskGraphView: ...

    async def preflight_close(self) -> None: ...


class TaskGraphLauncher(Protocol):
    async def start(self, launch: TaskGraphLaunch) -> TaskGraphHandle: ...

    async def cancel(self, launch: TaskGraphLaunch) -> TaskGraphView: ...

    async def cancel_node(
        self,
        launch: TaskGraphLaunch,
        node_id: str,
        execution_id: str,
    ) -> TaskGraphView: ...

    async def settle_cancel(
        self,
        launch: TaskGraphLaunch,
        *,
        invoke_effects: bool,
    ) -> TaskGraphView: ...

    async def supply_input(
        self,
        launch: TaskGraphLaunch,
        node_id: str,
        execution_id: str,
        value: JsonValue,
    ) -> TaskGraphView: ...

    async def resolve_effect(
        self,
        launch: TaskGraphLaunch,
        node_id: str,
        execution_id: str,
        expected_fence: int,
        resolution: TaskEffectResolution,
    ) -> TaskGraphView: ...


__all__ = [
    "TaskEffectResolutionRequest",
    "TaskGraphLauncher",
    "TaskGraphQueryService",
    "TaskGraphService",
]
