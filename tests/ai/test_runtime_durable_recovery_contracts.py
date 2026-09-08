#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.runtime import (
    ExecutionRecoveryEffect,
    ResolveToolEffectRequest,
    ToolEffectApplied,
    ToolEffectFailed,
    ToolEffectNotApplied,
)
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.task import TaskEvent, TaskEventType, TaskGraphSnapshot, TaskGraphView, TaskNode, TaskNodeView


def _node(
    node_id: str,
    status: TaskStatus,
    *,
    dependencies: tuple[str, ...] = (),
    fence: int = 0,
    execution_id: str | None = None,
    error_code: str | None = None,
    error_digest: str | None = None,
) -> TaskNodeView:
    return TaskNodeView(
        graph_id="graph",
        node_id=node_id,
        dependencies=dependencies,
        status=status,
        owner=None,
        fence=fence,
        lease_expires_at=None,
        result_digest=None,
        error_code=error_code,
        error_digest=error_digest,
        execution_id=execution_id,
    )


def test_default_output_retries_do_not_change_canonical_agent_payload() -> None:
    codec = AgentSpecCodec()
    implicit = codec.to_payload(AgentSpec("agent"))
    explicit = codec.to_payload(AgentSpec("agent", output_retries=1))

    assert implicit == explicit
    assert "output_retries" not in implicit
    assert codec.to_payload(AgentSpec("agent", output_retries=0))["output_retries"] == 0


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_output_retries_reject_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentSpec("agent", output_retries=value)  # type: ignore[arg-type]


def test_recovery_required_graph_status_precedes_failed_nodes() -> None:
    nodes = (TaskNode("recover"), TaskNode("failed"))
    states = (
        _node(
            "recover",
            TaskStatus.RECOVERY_REQUIRED,
            fence=2,
            execution_id="execution",
            error_code="TOOL_EFFECT_UNKNOWN",
            error_digest="a" * 64,
        ),
        _node(
            "failed",
            TaskStatus.FAILED,
            fence=1,
            error_code="TASK_NODE_FAILED",
            error_digest="b" * 64,
        ),
    )

    snapshot = TaskGraphSnapshot(
        "graph",
        TaskStatus.RECOVERY_REQUIRED,
        nodes,
        states,
    )

    assert snapshot.status is TaskStatus.RECOVERY_REQUIRED


def test_recovered_pending_event_preserves_execution_and_fence() -> None:
    event = TaskEvent(
        version=1,
        graph_id="graph",
        sequence=2,
        event_type=TaskEventType.NODE_CHANGED,
        occurred_at=datetime.now(timezone.utc),
        status=TaskStatus.PENDING,
        previous_status=TaskStatus.RECOVERY_REQUIRED,
        node_id="node",
        fence=3,
        execution_id="execution",
    )

    assert event.fence == 3
    assert event.execution_id == "execution"


def test_execution_recovery_contracts_validate_identity() -> None:
    principal = Principal("principal", "tenant")
    request = ResolveToolEffectRequest(
        principal=principal,
        operation_id="operation",
        expected_fence=2,
        resolution=ToolEffectNotApplied(),
        idempotency_key="recovery-key",
    )
    effect = ExecutionRecoveryEffect(
        operation_id="operation",
        execution_id="execution",
        step_run_id="step",
        tool_call_id="call",
        tool_name="write_file",
        fence=2,
        idempotency_key_digest="c" * 64,
        replay_safe=False,
        error_code="TOOL_EFFECT_UNKNOWN",
    )

    assert request.expected_fence == effect.fence
    assert isinstance(ToolEffectApplied({"ok": True}), ToolEffectApplied)
    assert isinstance(ToolEffectFailed(), ToolEffectFailed)
