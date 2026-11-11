#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphLimits, TaskGraphRequest, TaskNode


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [3])
async def test_completion_does_not_materialize_unrelated_task_definitions(
    monkeypatch: pytest.MonkeyPatch, width: int,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="completion-reads", tenant_id="tenant")
    try:
        graph = TaskGraph("graph", tuple(TaskNode(f"node-{i}") for i in range(width)))
        request = TaskGraphRequest(
            graph, Principal("tester", "tenant"), "submit:graph",
            limits=TaskGraphLimits(max_nodes=200),
        )
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
        repository = state.task.tasks
        lease = await repository.claim(
            "graph", "node-0", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        decoded_definitions = []
        decode = repository._decode

        async def capture(record, target):
            value = await decode(record, target)
            if target is TaskNode:
                decoded_definitions.append(value)
            return value

        monkeypatch.setattr(repository, "_decode", capture)
        terminal = await repository.complete(
            lease, tenant_id="tenant", execution_id="execution-0", result_digest="a" * 64,
        )
        assert terminal.status is TaskStatus.SUCCEEDED
        assert {value.node_id for value in decoded_definitions} <= {"node-0"}
        monkeypatch.setattr(repository, "_decode", decode)
        current = await repository.graph_state("graph", tenant_id="tenant")
        assert current.status is TaskStatus.PENDING
        assert len(current.nodes) == width
        assert next(value for value in current.node_states if value.node_id == "node-0").status is TaskStatus.SUCCEEDED
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_policy", "blocked_policy", "expected"),
    [("propagate", "isolate", TaskStatus.FAILED),
     ("isolate", "propagate", TaskStatus.BLOCKED),
     ("isolate", "isolate", TaskStatus.SUCCEEDED)],
)
async def test_completion_aggregates_expanded_failure_and_blocked_nodes(
    failed_policy: str, blocked_policy: str, expected: TaskStatus,
) -> None:
    from linktools.ai.task import TaskExpanderRef

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="completion-expanded", tenant_id="tenant")
    try:
        graph = TaskGraph("graph", (TaskNode("root", expander=TaskExpanderRef("expand", 1)),))
        request = TaskGraphRequest(graph, Principal("tester", "tenant"), "submit:graph")
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
        repository = state.task.tasks
        root = await repository.claim(
            "graph", "root", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        await repository.complete(
            root, tenant_id="tenant", execution_id="execution-root", result_digest="a" * 64,
            expanded_nodes=(
                TaskNode("failed", failure_policy=failed_policy),
                TaskNode("blocked", dependencies=("failed",), failure_policy=blocked_policy),
                TaskNode("success"),
            ),
        )
        failed = await repository.claim(
            "graph", "failed", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        success = await repository.claim(
            "graph", "success", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        await repository.fail(
            failed, tenant_id="tenant", error_code="TEST_FAILURE", error_digest="b" * 64,
        )
        await repository.scheduler_state("graph", tenant_id="tenant")
        await repository.complete(
            success, tenant_id="tenant", execution_id="execution-success", result_digest="c" * 64,
        )
        current = await repository.graph_state("graph", tenant_id="tenant")
        assert current.status is expected
        assert {node.node_id for node in current.nodes} == {"root", "failed", "blocked", "success"}
        events = await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=100)
        assert events.items[-1].status is expected
    finally:
        await state.close()
