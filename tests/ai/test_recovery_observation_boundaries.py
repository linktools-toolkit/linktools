#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery observation boundaries for events and task results."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionEventType,
    ExecutionStatus,
    Page,
    Principal,
    TaskStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._event import DefaultEventService, LiveExecutionEventBroker
from linktools.ai.task import TaskGraphSnapshot, TaskNode, TaskNodeView


class _Authorization:
    async def authorize(self, principal, action, resource) -> None:
        del principal, action, resource


class _Executions:
    def __init__(self, *, event_sequence: int = 1) -> None:
        self.record = SimpleNamespace(
            execution_id="execution",
            tenant_id="tenant",
            status=ExecutionStatus.RECOVERY_REQUIRED,
            event_sequence=event_sequence,
        )

    async def get_header(self, execution_id: str, *, tenant_id: str) -> object:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return object()

    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return self.record


class _NoDurableEvents:
    pass


class _DurableEvents:
    async def list(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ) -> Page[object]:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        assert limit > 0
        if after_sequence == 0:
            return Page(
                (
                    SimpleNamespace(
                        execution_id="execution",
                        sequence=1,
                        event_type=ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
                        payload={
                            "error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value
                        },
                    ),
                ),
                None,
            )
        return Page((), None)


@pytest.mark.asyncio
async def test_live_recovery_event_ends_observation_cleanly() -> None:
    broker = LiveExecutionEventBroker()
    broker.prepare_local_producer("execution")
    broker.register_local_producer("execution", 0)
    broker.publish_event(
        "execution",
        ExecutionEventType.EXECUTION_RECOVERY_REQUIRED,
        {"error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value},
        durable_sequence=1,
    )
    broker.complete("execution")
    service = DefaultEventService(
        _Executions(),
        _NoDurableEvents(),
        _Authorization(),
        lambda execution_id, *, tenant_id: None,
        broker,
    )

    values = [
        event
        async for event in service.stream(
            "execution",
            principal=Principal("principal", "tenant"),
        )
    ]

    assert [value.event_type for value in values] == [
        ExecutionEventType.EXECUTION_RECOVERY_REQUIRED
    ]


@pytest.mark.asyncio
async def test_durable_recovery_event_ends_restart_observation_cleanly() -> None:
    service = DefaultEventService(
        _Executions(),
        _DurableEvents(),
        _Authorization(),
        lambda execution_id, *, tenant_id: None,
        LiveExecutionEventBroker(),
    )

    first = [
        event
        async for event in service.stream(
            "execution",
            principal=Principal("principal", "tenant"),
        )
    ]
    replay = [
        event
        async for event in service.stream(
            "execution",
            principal=Principal("principal", "tenant"),
            after_sequence=1,
        )
    ]

    assert [value.event_type for value in first] == [
        ExecutionEventType.EXECUTION_RECOVERY_REQUIRED
    ]
    assert replay == []


class _RecoveryTaskService:
    async def snapshot_graph(self, graph_id: str, *, principal: Principal):
        assert graph_id == "graph"
        assert principal.tenant_id == "tenant"
        state = TaskNodeView(
            graph_id="graph",
            node_id="node",
            dependencies=(),
            status=TaskStatus.RECOVERY_REQUIRED,
            owner=None,
            fence=1,
            lease_expires_at=None,
            result_digest=None,
            error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            error_digest="a" * 64,
            execution_id="execution",
        )
        return TaskGraphSnapshot(
            "graph",
            TaskStatus.RECOVERY_REQUIRED,
            (TaskNode("node"),),
            (state,),
        )


@pytest.mark.asyncio
async def test_recovery_required_task_result_is_not_ready_not_corrupt() -> None:
    runtime = object.__new__(Runtime)
    runtime._closed = False
    runtime._closing = False
    runtime.task = _RecoveryTaskService()
    runtime._task_node_runtime = None

    with pytest.raises(AIError) as raised:
        await runtime.read_task_result(
            "graph",
            "node",
            principal=Principal("principal", "tenant"),
        )

    assert raised.value.code is ErrorCode.TASK_NOT_READY
