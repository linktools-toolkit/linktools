#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from linktools.ai.runtime import ExecutionRecoveryEffect, Runtime
from linktools.ai.runtime._event import (
    DefaultEventService,
    LiveExecutionEventBroker,
)
from linktools.ai.runtime._recovery_impl import RecoveryLocalExecutionBackend
from linktools.ai.runtime.state._contracts import (
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
)
from linktools.ai.task import TaskGraphSnapshot, TaskNode, TaskNodeView


class _Authorization:
    async def authorize(self, principal, action, resource) -> None:
        del principal, action, resource


class _RecoveryExecutions:
    def __init__(self, *, event_sequence: int = 1) -> None:
        self.record = SimpleNamespace(
            execution_id="execution",
            tenant_id="tenant",
            status=ExecutionStatus.RECOVERY_REQUIRED,
            event_sequence=event_sequence,
        )

    async def get_header(self, execution_id: str, *, tenant_id: str):
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return object()

    async def get(self, execution_id: str, *, tenant_id: str):
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return self.record


class _NoDurableEvents:
    pass


class _DurableRecoveryEvents:
    async def list(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        after_sequence: int,
        limit: int,
    ):
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
                        payload={"error_code": ErrorCode.TOOL_EFFECT_UNKNOWN.value},
                    ),
                ),
                None,
            )
        assert after_sequence == 1
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
        _RecoveryExecutions(),
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
        _RecoveryExecutions(),
        _DurableRecoveryEvents(),
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


class _CrashWindowExecutions:
    def __init__(self) -> None:
        self.record = SimpleNamespace(
            execution_id="execution",
            tenant_id="tenant",
            status=ExecutionStatus.STARTED,
        )

    async def get(self, execution_id: str, *, tenant_id: str):
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return self.record


class _CrashWindowBackend(RecoveryLocalExecutionBackend):
    def __init__(self) -> None:
        self._execution = SimpleNamespace(executions=_CrashWindowExecutions())
        self.committed = None

    def _validate_recovery_identity(self, execution, recovery_input) -> None:
        del execution, recovery_input

    async def recovery_effects(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecoveryEffect, ...]:
        assert execution_id == "execution"
        assert tenant_id == "tenant"
        return (
            ExecutionRecoveryEffect(
                operation_id="operation",
                execution_id="execution",
                step_run_id="step",
                tool_call_id="call",
                tool_name="write_file",
                fence=3,
                idempotency_key_digest="b" * 64,
                replay_safe=False,
                error_code=ErrorCode.TOOL_EFFECT_UNKNOWN.value,
            ),
        )

    async def _commit_recovery_required(self, execution, error, effects):
        self.committed = (execution, error, effects)
        return execution


@pytest.mark.asyncio
async def test_startup_unknown_effect_converges_before_agent_relaunch() -> None:
    backend = _CrashWindowBackend()
    checkpoint = SimpleNamespace(
        execution_id="execution",
        tenant_id="tenant",
        input=object(),
        state=RecoveryCheckpointState.ACTIVE,
        handoff_phase=RecoveryHandoffPhase.NONE,
    )

    await backend._reconcile_checkpoint(checkpoint)

    assert backend.committed is not None
    execution, error, effects = backend.committed
    assert execution.status is ExecutionStatus.STARTED
    assert error.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert error.safe_details["operation_id"] == "operation"
    assert effects[0].fence == 3
