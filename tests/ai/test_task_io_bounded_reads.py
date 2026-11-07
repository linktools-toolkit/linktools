#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task reads stay local without weakening durable dependency checks."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.state import RecordReplacement
from linktools.ai.task import (
    LocalTaskGraphLauncher, TaskGraph, TaskGraphAdmission, TaskGraphLimits,
    TaskGraphRequest, TaskNode, TaskNodeView,
)


async def _admit(state: Persistence, nodes: tuple[TaskNode, ...]) -> None:
    graph = TaskGraph("graph", nodes)
    request = TaskGraphRequest(
        graph, Principal("tester", "tenant"), "submit:graph",
        limits=TaskGraphLimits(max_nodes=len(nodes), max_budget=max(1000, len(nodes))),
    )
    await state.task.admissions.admit(TaskGraphAdmission.from_request(request), graph)


def _launcher(repository) -> LocalTaskGraphLauncher:
    return LocalTaskGraphLauncher(repository, None, owner="worker")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation,expected_reads", [("renew", 1), ("complete", 2)])
async def test_task_prefetch_definition_read_budget(
    tmp_path: Path, operation: str, expected_reads: int,
) -> None:
    async with Persistence.open(
        "prefetch", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"), tenant_id="tenant",
    ) as state:
        await _admit(state, (TaskNode("first"), TaskNode("unrelated")))
        repository = state.task.tasks
        lease = await repository.claim(
            "graph", "first", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        reads = []

        def before_sql(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "FROM ai_state_records" in statement:
                reads.append(statement)

        event.listen(Engine, "before_cursor_execute", before_sql)
        try:
            if operation == "renew":
                renewed = await repository.renew(lease, tenant_id="tenant", lease_seconds=30)
                assert (renewed.owner, renewed.fence) == (lease.owner, lease.fence)
            else:
                terminal = await repository.complete(
                    lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
                )
                assert terminal.status is TaskStatus.SUCCEEDED
        finally:
            event.remove(Engine, "before_cursor_execute", before_sql)
        assert len(reads) == expected_reads


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [2, 200])
async def test_dependency_context_reads_only_direct_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int,
) -> None:
    async with Persistence.open(
        "local-dependencies", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"),
        tenant_id="tenant",
    ) as state:
        target = TaskNode("target", ("source",))
        await _admit(state, (TaskNode("source"), target) + tuple(
            TaskNode(f"unrelated-{i}") for i in range(width - 2)
        ))
        repository = state.task.tasks
        lease = await repository.claim(
            "graph", "source", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        await repository.complete(
            lease, tenant_id="tenant", execution_id="source-execution", result_digest="a" * 64,
        )
        original_decode = repository._decode
        decoded = []

        async def decode(record, target_type):
            result = await original_decode(record, target_type)
            if target_type in (TaskNode, TaskNodeView):
                decoded.append(result.node_id)
            return result

        async def reject_scan(*args, **kwargs):
            raise AssertionError("dependency input must not scan the graph")

        monkeypatch.setattr(repository, "_decode", decode)
        monkeypatch.setattr(repository, "list_nodes", reject_scan)
        launcher = _launcher(repository)
        results, states = await launcher._dependency_context("graph", target, tenant_id="tenant")
        assert set(results) == set(states) == {"source"}
        assert results["source"].execution_id == "source-execution"
        assert results["source"].result_digest == "a" * 64
        assert decoded == ["source", "source"]
        assert await launcher._dependency_states("graph", target, tenant_id="tenant") == states


@pytest.mark.asyncio
async def test_high_fan_in_dependency_reads_bound_parameters(tmp_path: Path) -> None:
    async with Persistence.open(
        "fan-in", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"), tenant_id="tenant",
    ) as state:
        dependencies = tuple(f"source-{i}" for i in range(501))
        target = TaskNode("target", dependencies, dependency_policy="all_terminal")
        await _admit(state, tuple(TaskNode(key) for key in dependencies) + (target,))
        repository = state.task.tasks

        async def terminal_fixture(transaction):
            records = await transaction.get_records(tuple(
                repository._state_key("graph", key) for key in dependencies
            ))
            replacements = []
            for record in records.values():
                node = await repository._decode(record, TaskNodeView)
                replacements.append(RecordReplacement(
                    repository._task_node_record(record, replace(node, status=TaskStatus.CANCELLED)),
                    record.storage_revision,
                ))
            await transaction.replace_records(tuple(replacements))

        await repository.state_store.mutate(terminal_fixture)
        reads = []

        def before_sql(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT") and "FROM ai_state_records" in statement:
                assert not many
                assert len(parameters) <= 999
                reads.append(len(parameters))

        event.listen(Engine, "before_cursor_execute", before_sql)
        try:
            states = await _launcher(repository)._dependency_states("graph", target, tenant_id="tenant")
            assert set(states) == set(dependencies)
            assert {item.status for item in states.values()} == {TaskStatus.CANCELLED}
            assert reads == [513, 491]
            reads.clear()
            result_ids = dependencies + tuple(f"missing-{i}" for i in range(500))
            assert await repository.get_results("graph", result_ids, tenant_id="tenant") == {}
            assert reads == [257, 257, 257, 234]
        finally:
            event.remove(Engine, "before_cursor_execute", before_sql)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["state", "definition", "result", "digest", "execution"])
async def test_dependency_context_rejects_missing_or_inconsistent_facts(
    monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    async with Persistence.open(
        "dependency-integrity", storage=PersistenceStorage.in_memory(), tenant_id="tenant",
    ) as state:
        target = TaskNode("target", ("source",))
        await _admit(state, (TaskNode("source"), target))
        repository = state.task.tasks
        lease = await repository.claim(
            "graph", "source", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        await repository.complete(
            lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
        )
        if fault in {"state", "definition"}:
            key = (repository._state_key if fault == "state" else repository._definition_key)("graph", "source")

            async def remove(transaction):
                record = await transaction.get_record(key)
                assert await transaction.delete_record(key, expected_storage_revision=record.storage_revision)

            await repository.state_store.mutate(remove)
        else:
            original = repository.get_results

            async def wrong_result(*args, **kwargs):
                results = await original(*args, **kwargs)
                if fault == "result":
                    return {}
                field = {"result_digest": "b" * 64} if fault == "digest" else {"execution_id": "other"}
                return {"source": replace(results["source"], **field)}

            monkeypatch.setattr(repository, "get_results", wrong_result)
        with pytest.raises(AIError) as raised:
            await _launcher(repository)._dependency_context("graph", target, tenant_id="tenant")
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["all_terminal", "any_succeeded"])
async def test_dependency_context_preserves_mixed_terminal_results(policy: str) -> None:
    async with Persistence.open(
        "dependency-terminal", storage=PersistenceStorage.in_memory(), tenant_id="tenant",
    ) as state:
        target = TaskNode("target", ("success", "failed"), dependency_policy=policy)
        await _admit(state, (TaskNode("success"), TaskNode("failed", failure_policy="isolate"), target))
        repository = state.task.tasks
        for node_id in ("success", "failed"):
            lease = await repository.claim(
                "graph", node_id, tenant_id="tenant", owner="worker", lease_seconds=30,
            )
            if node_id == "success":
                await repository.complete(
                    lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
                )
            else:
                await repository.fail(
                    lease, tenant_id="tenant", error_code="FAILED", error_digest="b" * 64,
                )
        launcher = _launcher(repository)
        results, states = await launcher._dependency_context("graph", target, tenant_id="tenant")
        assert set(results) == {"success"}
        assert {key: item.status for key, item in states.items()} == {
            "success": TaskStatus.SUCCEEDED, "failed": TaskStatus.FAILED,
        }
        assert await launcher._dependency_states("graph", target, tenant_id="tenant") == states


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["renew", "complete"])
@pytest.mark.parametrize("missing", ["graph", "state", "definition"])
async def test_prefetch_preserves_missing_record_error_codes(operation: str, missing: str) -> None:
    async with Persistence.open(
        "prefetch-missing", storage=PersistenceStorage.in_memory(), tenant_id="tenant",
    ) as state:
        await _admit(state, (TaskNode("source"),))
        repository = state.task.tasks
        lease = await repository.claim(
            "graph", "source", tenant_id="tenant", owner="worker", lease_seconds=30,
        )
        key = repository._graph_key("graph") if missing == "graph" else (
            repository._state_key if missing == "state" else repository._definition_key
        )("graph", "source")

        async def remove(transaction):
            record = await transaction.get_record(key)
            assert await transaction.delete_record(key, expected_storage_revision=record.storage_revision)

        await repository.state_store.mutate(remove)
        with pytest.raises(AIError) as raised:
            if operation == "renew":
                await repository.renew(lease, tenant_id="tenant", lease_seconds=30)
            else:
                await repository.complete(
                    lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
                )
        expected = ErrorCode.STORAGE_INTEGRITY_ERROR if missing == "definition" else ErrorCode.TASK_FENCE_STALE
        assert raised.value.code is expected
