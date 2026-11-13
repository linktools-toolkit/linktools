#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Failure settlement reads the definitions needed for terminal policy only."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import (
    TaskGraph, TaskGraphAdmission, TaskGraphLimits, TaskGraphRequest, TaskNode,
)
from .test_task_io_measurements import _measure


async def _admit(state: Persistence, nodes: tuple[TaskNode, ...]) -> None:
    graph = TaskGraph("graph", nodes)
    request = TaskGraphRequest(
        graph, Principal("tester", "tenant"), "submit:graph",
        limits=TaskGraphLimits(max_nodes=len(nodes)),
    )
    await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize("width", [3])
async def test_failure_does_not_materialize_unrelated_definitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sqlite: bool, width: int,
) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite") if sqlite else PersistenceStorage.in_memory()
    async with Persistence.open("failure-reads", storage=storage, tenant_id="tenant") as state:
        await _admit(state, tuple(TaskNode(f"node-{index}") for index in range(width)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "node-0", tenant_id="tenant", owner="worker", lease_seconds=30)
        decoded = []
        original = repository._decode

        async def decode(record, target):
            value = await original(record, target)
            if target is TaskNode:
                decoded.append(value.node_id)
            return value

        with monkeypatch.context() as patch:
            patch.setattr(repository, "_decode", decode)
            result = await repository.fail(lease, tenant_id="tenant", error_code="FAILED", error_digest="a" * 64)
        assert result.status is TaskStatus.FAILED
        assert decoded == ["node-0"]
        graph = await repository.graph_state("graph", tenant_id="tenant")
        assert graph.status is TaskStatus.PENDING
        assert len(graph.nodes) == width
        node = next(node for node in graph.node_states if node.node_id == "node-0")
        assert node.status is TaskStatus.FAILED and node.fence == lease.fence
        events = await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=100)
        replay = await repository.fail(lease, tenant_id="tenant", error_code="FAILED", error_digest="a" * 64)
        assert replace(replay, completed_at=result.completed_at) == result
        assert (await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=100)).items == events.items
        with pytest.raises(AIError) as raised:
            await repository.fail(replace(lease, fence=lease.fence + 1), tenant_id="tenant", error_code="FAILED", error_digest="a" * 64)
        assert raised.value.code is ErrorCode.TASK_FENCE_STALE


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize("failed_policy,blocked_policy,target_policy,expected", [
    ("propagate", "isolate", "isolate", TaskStatus.FAILED),
    ("isolate", "propagate", "isolate", TaskStatus.BLOCKED),
    ("isolate", "isolate", "propagate", TaskStatus.FAILED),
    ("isolate", "isolate", "isolate", TaskStatus.SUCCEEDED),
])
async def test_failure_aggregates_existing_terminal_policies(
    tmp_path: Path, sqlite: bool, failed_policy: str, blocked_policy: str,
    target_policy: str, expected: TaskStatus,
) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite") if sqlite else PersistenceStorage.in_memory()
    async with Persistence.open("failure-policy", storage=storage, tenant_id="tenant") as state:
        await _admit(state, (
            TaskNode("failed", failure_policy=failed_policy),
            TaskNode("blocked", ("failed",), failure_policy=blocked_policy),
            TaskNode("target", failure_policy=target_policy),
        ))
        repository = state.task.tasks
        first = await repository.claim("graph", "failed", tenant_id="tenant", owner="worker", lease_seconds=30)
        target = await repository.claim("graph", "target", tenant_id="tenant", owner="worker", lease_seconds=30)
        await repository.fail(first, tenant_id="tenant", error_code="FIRST", error_digest="a" * 64)
        await repository.scheduler_state("graph", tenant_id="tenant")
        terminal = await repository.fail(target, tenant_id="tenant", error_code="TARGET", error_digest="b" * 64)
        assert terminal.status is TaskStatus.FAILED
        graph = await repository.graph_state("graph", tenant_id="tenant")
        assert graph.status is expected
        assert {node.node_id: node.status for node in graph.node_states} == {
            "failed": TaskStatus.FAILED, "blocked": TaskStatus.BLOCKED, "target": TaskStatus.FAILED,
        }
        events = await repository.list_events("graph", tenant_id="tenant", after_event_seq=0, limit=100)
        assert events.items[-1].status is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["graph", "state", "target_definition", "failed_definition", "blocked_definition"])
async def test_failure_preserves_required_record_errors(missing: str) -> None:
    async with Persistence.open("failure-integrity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await _admit(state, (TaskNode("failed"), TaskNode("blocked", ("failed",)), TaskNode("target")))
        repository = state.task.tasks
        first = await repository.claim("graph", "failed", tenant_id="tenant", owner="worker", lease_seconds=30)
        target = await repository.claim("graph", "target", tenant_id="tenant", owner="worker", lease_seconds=30)
        await repository.fail(first, tenant_id="tenant", error_code="FIRST", error_digest="a" * 64)
        await repository.scheduler_state("graph", tenant_id="tenant")
        key = repository._graph_key("graph") if missing == "graph" else (
            repository._state_key("graph", "target") if missing == "state"
            else repository._definition_key("graph", missing.removesuffix("_definition"))
        )

        async def remove(transaction):
            record = await transaction.get_record(key)
            assert await transaction.delete_record(key, expected_storage_revision=record.storage_revision)

        await repository.state_store.mutate(remove)
        with pytest.raises(AIError) as raised:
            await repository.fail(target, tenant_id="tenant", error_code="TARGET", error_digest="b" * 64)
        expected = ErrorCode.TASK_FENCE_STALE if missing in {"graph", "state"} else ErrorCode.STORAGE_INTEGRITY_ERROR
        assert raised.value.code is expected


@pytest.mark.manual
@pytest.mark.asyncio
@pytest.mark.parametrize("width", [2, 200])
async def test_task_failure_io_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int) -> None:
    async with Persistence.open("failure-probe", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"), tenant_id="tenant") as state:
        await _admit(state, tuple(TaskNode(f"node-{index}") for index in range(width)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "node-0", tenant_id="tenant", owner="worker", lease_seconds=30)
        with _measure(monkeypatch) as counts:
            terminal = await repository.fail(lease, tenant_id="tenant", error_code="FAILED", error_digest="a" * 64)
        assert terminal.status is TaskStatus.FAILED
        print("TASK_FAILURE_IO_METRICS " + json.dumps({"width": width, "metrics": dict(counts)}, sort_keys=True))


@pytest.mark.asyncio
async def test_failure_bounds_large_terminal_policy_reads(tmp_path: Path) -> None:
    async with Persistence.open("many-failures", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"), tenant_id="tenant") as state:
        prior_ids = tuple(f"failed-{index}" for index in range(501))
        await _admit(state, tuple(TaskNode(node_id, failure_policy="isolate") for node_id in (*prior_ids, "target")))
        repository = state.task.tasks
        lease = await repository.claim("graph", "target", tenant_id="tenant", owner="worker", lease_seconds=30)

        async def terminal_fixture(transaction):
            before = await repository._event_state_in_transaction(transaction, "graph")
            graph_record = await transaction.get_record(repository._graph_key("graph"))
            next_nodes = tuple(
                replace(node, status=TaskStatus.FAILED, fence=1, error_code="FAILED", error_digest="a" * 64)
                if node.node_id in prior_ids else node
                for node in before.node_states
            )
            # Large-fixture setup uses the owner's atomic transition so its
            # graph projection and event stream remain consistent with nodes.
            await repository._apply_graph_transition(
                transaction, before, graph_record, next_nodes, TaskStatus.RUNNING,
            )

        await repository.state_store.mutate(terminal_fixture)
        prepared = await repository.graph_state("graph", tenant_id="tenant")
        assert prepared.status is TaskStatus.RUNNING
        assert sum(node.status is TaskStatus.FAILED for node in prepared.node_states) == len(prior_ids)
        assert next(node for node in prepared.node_states if node.node_id == "target").status is TaskStatus.RUNNING
        reads = []

        def before_sql(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "FROM ai_state_records" in statement:
                assert not many
                assert len(parameters) <= 257
                reads.append(len(parameters))

        event.listen(Engine, "before_cursor_execute", before_sql)
        try:
            terminal = await repository.fail(lease, tenant_id="tenant", error_code="TARGET", error_digest="b" * 64)
        finally:
            event.remove(Engine, "before_cursor_execute", before_sql)
        assert terminal.status is TaskStatus.FAILED
        assert len(reads) == 4
        graph = await repository.graph_state("graph", tenant_id="tenant")
        assert graph.status is TaskStatus.SUCCEEDED
        assert len(graph.node_states) == 502
        assert {node.status for node in graph.node_states} == {TaskStatus.FAILED}
