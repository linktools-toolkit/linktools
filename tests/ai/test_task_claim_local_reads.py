#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._memory_transaction import _MemoryTransaction
from linktools.ai.task import TaskGraph, TaskGraphAdmission, TaskGraphLimits, TaskGraphRequest, TaskNode


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [2, 200])
async def test_task_claim_materialization_is_independent_of_unrelated_nodes(
    monkeypatch: pytest.MonkeyPatch, width: int,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="claim-local", tenant_id="tenant")
    try:
        request = TaskGraphRequest(
            TaskGraph("graph", tuple(TaskNode(f"node-{i}") for i in range(width))),
            Principal("tester", "tenant"),
            "submit:graph",
            limits=TaskGraphLimits(max_nodes=200),
        )
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), request.graph)
        reads = []
        original = _MemoryTransaction.list_records

        async def capture(transaction, query):
            result = await original(transaction, query)
            reads.append((query, len(result)))
            return result

        monkeypatch.setattr(_MemoryTransaction, "list_records", capture)
        lease = await state.task.tasks.claim(
            "graph", "node-0", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        assert lease.node_id == "node-0"
        assert sum(count for _, count in reads) < 2
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("occupies", [True, False])
async def test_task_claim_respects_waiting_execution_capacity(occupies: bool) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="claim-waiting", tenant_id="tenant")
    try:
        request = TaskGraphRequest(
            TaskGraph("graph", (TaskNode("a"), TaskNode("b"))),
            Principal("tester", "tenant"),
            "submit:graph",
            limits=TaskGraphLimits(max_concurrency=1),
        )
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), request.graph)
        lease = await state.task.tasks.claim(
            "graph", "a", tenant_id="tenant", owner="first", lease_seconds=30,
        )
        await state.task.tasks.handoff_execution(
            lease, tenant_id="tenant", execution_id="execution-a", occupies_concurrency=occupies,
        )
        if occupies:
            with pytest.raises(AIError) as error:
                await state.task.tasks.claim(
                    "graph", "b", tenant_id="tenant", owner="second", lease_seconds=30,
                )
            assert error.value.code is ErrorCode.TASK_NOT_READY
        else:
            second = await state.task.tasks.claim(
                "graph", "b", tenant_id="tenant", owner="second", lease_seconds=30,
            )
            assert second.node_id == "b"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_task_claim_rejects_definition_identity_mismatch() -> None:
    from dataclasses import replace

    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="claim-identity", tenant_id="tenant")
    try:
        graph = TaskGraph("graph", (TaskNode("node"),))
        request = TaskGraphRequest(graph, Principal("tester", "tenant"), "submit:graph")
        await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)
        repository = state.task.tasks

        async def corrupt(transaction):
            record = await transaction.get_record(repository._definition_key("graph", "node"))
            candidate = repository._stored(
                "task_node_definition", ["graph", "node"], TaskNode("another"),
                parent=repository._definition_parent("graph"),
            )
            assert await transaction.replace_record(
                replace(candidate, storage_revision=record.storage_revision + 1),
                expected_storage_revision=record.storage_revision,
            )

        await repository.state_store.mutate(corrupt)
        with pytest.raises(AIError) as error:
            await repository.claim(
                "graph", "node", tenant_id="tenant", owner="worker", lease_seconds=30,
            )
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()
