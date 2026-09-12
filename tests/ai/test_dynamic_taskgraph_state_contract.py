#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State invariants for dynamic TaskGraph execution handoff."""

from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.core import TaskStatus
from linktools.ai.task import TaskEvent, TaskEventType, TaskNodeView


@pytest.mark.parametrize(
    ("status", "owner", "fence", "lease_expires_at"),
    (
        (TaskStatus.PENDING, None, 0, None),
        (TaskStatus.READY, None, 0, None),
        (
            TaskStatus.RUNNING,
            "worker",
            1,
            datetime.now(timezone.utc) + timedelta(seconds=30),
        ),
    ),
)
def test_unbound_task_states_reject_execution_id(
    status: TaskStatus,
    owner: str | None,
    fence: int,
    lease_expires_at: datetime | None,
) -> None:
    with pytest.raises(ValueError, match="cannot carry an execution id"):
        TaskNodeView(
            "graph",
            "node",
            (),
            status,
            owner,
            fence,
            lease_expires_at,
            None,
            None,
            None,
            "execution",
        )


def test_running_event_rejects_execution_id() -> None:
    with pytest.raises(ValueError, match="running task event state is invalid"):
        TaskEvent(
            1,
            "graph",
            1,
            TaskEventType.NODE_CHANGED,
            datetime.now(timezone.utc),
            TaskStatus.RUNNING,
            TaskStatus.READY,
            "node",
            "worker",
            1,
            "execution",
        )


def test_waiting_is_the_only_nonterminal_bound_execution_state() -> None:
    state = TaskNodeView(
        "graph",
        "node",
        (),
        TaskStatus.WAITING,
        None,
        1,
        None,
        None,
        None,
        None,
        "execution",
    )

    assert state.execution_id == "execution"
    assert state.owner is None
    assert state.lease_expires_at is None


def test_graph_event_rejects_node_only_waiting_status() -> None:
    with pytest.raises(ValueError, match="node-only status"):
        TaskEvent(
            1,
            "graph",
            1,
            TaskEventType.GRAPH_CHANGED,
            datetime.now(timezone.utc),
            TaskStatus.WAITING,
            TaskStatus.RUNNING,
        )


def test_recovery_required_requires_error_digest() -> None:
    with pytest.raises(ValueError, match="recovery-required task node state is invalid"):
        TaskNodeView(
            "graph",
            "node",
            (),
            TaskStatus.RECOVERY_REQUIRED,
            None,
            1,
            None,
            None,
            "STORAGE_RECOVERY_REQUIRED",
            None,
            "execution",
        )
