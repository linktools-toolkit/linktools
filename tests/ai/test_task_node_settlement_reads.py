#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Node settlement stays fresh without reading unrelated graph nodes."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task import (
    LocalTaskGraphLauncher, TaskGraphLaunch, TaskGraphLimits, TaskNode,
    TaskNodeRunResult, TaskNodeView,
)
from linktools.ai.task._local import _GraphRun

from .test_task_io_bounded_reads import _admit


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_waiting_settlement_reads_current_node_and_preserves_its_hold(
    tmp_path: Path, backend: str,
) -> None:
    storage = {
        "memory": PersistenceStorage.in_memory,
        "local_sqlite": lambda: PersistenceStorage.from_root(tmp_path, exclusive_writer=True),
        "sqlite": lambda: PersistenceStorage.sqlite(tmp_path / "state.sqlite"),
    }[backend]()
    async with Persistence.open("node-settlement", storage=storage, tenant_id="tenant") as persistence:
        node = TaskNode("node")
        await _admit(persistence, (node, TaskNode("other")))
        repository = persistence.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="worker", lease_seconds=30)
        await repository.handoff_execution(lease, tenant_id="tenant", execution_id="execution")
        released = []

        async def release(execution_id: str, **kwargs: object) -> None:
            released.append(execution_id)

        launcher = LocalTaskGraphLauncher(repository, None, owner="worker", release_execution_hold=release)
        run = _GraphRun(TaskGraphLaunch("graph", Principal("tester", "tenant"), TaskGraphLimits()), "worker")
        result = TaskNodeRunResult("a" * 64, "execution")
        assert not await launcher._completion_committed(
            "graph", "node", result, tenant_id="tenant", expected_fence=lease.fence,
        )
        await launcher._finish_execution_hold(
            "execution", tenant_id="tenant", hold_id="hold", graph_id="graph", node_id="node",
            expected_fence=lease.fence,
        )
        assert not released
        await launcher._defer_waiting_recovery(
            run, node, "execution", expected_fence=lease.fence,
            cause=AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN),
        )
        state = await repository.graph_state("graph", tenant_id="tenant")
        assert state is not None
        selected = next(value for value in state.node_states if value.node_id == "node")
        assert selected.status is TaskStatus.RECOVERY_REQUIRED
        assert selected.execution_id == "execution" and selected.fence == lease.fence
        assert next(value for value in state.node_states if value.node_id == "other").status is TaskStatus.READY
        assert await launcher._completion_committed(
            "graph", "node", result, tenant_id="tenant", expected_fence=lease.fence,
        )
        await launcher._finish_execution_hold(
            "execution", tenant_id="tenant", hold_id="hold", graph_id="graph", node_id="node",
            expected_fence=lease.fence,
        )
        assert not released


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("hold", "readback"))
async def test_terminal_node_settlement_reads_only_its_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    async with Persistence.open(
        "node-read-cost", storage=PersistenceStorage.sqlite(tmp_path / "state.sqlite"), tenant_id="tenant",
    ) as persistence:
        await _admit(persistence, (TaskNode("node"), TaskNode("other-a"), TaskNode("other-b")))
        repository = persistence.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="worker", lease_seconds=30)
        await repository.complete(lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64)
        released, decoded, selects = [], [], []
        original_decode = repository._decode

        async def decode(record, value_type):
            value = await original_decode(record, value_type)
            if value_type in (TaskNode, TaskNodeView):
                decoded.append(value.node_id)
            return value

        async def release(execution_id: str, **kwargs: object) -> None:
            released.append(execution_id)

        def before_sql(connection, cursor, statement, parameters, context, many):
            if statement.lstrip().upper().startswith("SELECT"):
                selects.append(statement)

        launcher = LocalTaskGraphLauncher(repository, None, owner="worker", release_execution_hold=release)
        monkeypatch.setattr(repository, "_decode", decode)
        event.listen(Engine, "before_cursor_execute", before_sql)
        try:
            if operation == "hold":
                await launcher._finish_execution_hold(
                    "execution", tenant_id="tenant", hold_id="hold", graph_id="graph", node_id="node",
                    expected_fence=lease.fence,
                )
                assert released == ["execution"]
            else:
                assert await launcher._completion_committed(
                    "graph", "node", TaskNodeRunResult("a" * 64, "execution"),
                    tenant_id="tenant", expected_fence=lease.fence,
                )
                assert not released
        finally:
            event.remove(Engine, "before_cursor_execute", before_sql)
        assert decoded == ["node", "node"]
        assert len(selects) <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("missing_graph", "missing_node", "state", "definition", "state_payload", "definition_identity", "tenant", "fence"))
async def test_hold_settlement_rejects_missing_or_invalid_target(fault: str) -> None:
    async with Persistence.open("node-integrity", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as persistence:
        await _admit(persistence, (TaskNode("node"),))
        repository = persistence.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="worker", lease_seconds=30)
        await repository.complete(lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64)
        if fault in {"state", "definition", "state_payload", "definition_identity"}:
            key = (repository._state_key if fault.startswith("state") else repository._definition_key)("graph", "node")

            async def damage(transaction):
                record = await transaction.get_record(key)
                if fault in {"state", "definition"}:
                    assert await transaction.delete_record(key)
                elif fault == "state_payload":
                    assert await transaction.replace_record(
                        replace(record, data={}, storage_revision=record.storage_revision + 1),
                        expected_storage_revision=record.storage_revision,
                    )
                else:
                    assert await transaction.delete_record(key)
                    await transaction.insert_record(replace(record, parent_key=b"x" * 32))

            await repository.state_store.mutate(damage)
        released = []

        async def release(execution_id: str, **kwargs: object) -> None:
            released.append(execution_id)

        launcher = LocalTaskGraphLauncher(repository, None, owner="worker", release_execution_hold=release)
        with pytest.raises(AIError) as raised:
            await launcher._finish_execution_hold(
                "execution", tenant_id="other" if fault == "tenant" else "tenant", hold_id="hold",
                graph_id="missing" if fault == "missing_graph" else "graph",
                node_id="missing" if fault == "missing_node" else "node",
                expected_fence=lease.fence + int(fault == "fence"),
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert launcher._hold_cleanup_failure is raised.value
        assert not released


@pytest.mark.asyncio
async def test_node_settlement_observes_another_repository_completion(tmp_path: Path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite")
    async with Persistence.open("node-freshness", storage=storage, tenant_id="tenant") as reader:
        await _admit(reader, (TaskNode("node"), TaskNode("other")))
        repository = reader.task.tasks
        lease = await repository.claim("graph", "node", tenant_id="tenant", owner="worker", lease_seconds=30)
        await repository.handoff_execution(lease, tenant_id="tenant", execution_id="execution")
        released = []

        async def release(execution_id: str, **kwargs: object) -> None:
            released.append(execution_id)

        launcher = LocalTaskGraphLauncher(repository, None, owner="worker", release_execution_hold=release)
        result = TaskNodeRunResult("a" * 64, "execution")
        for completed in (False, True):
            assert await launcher._completion_committed(
                "graph", "node", result, tenant_id="tenant", expected_fence=lease.fence,
            ) is completed
            await launcher._finish_execution_hold(
                "execution", tenant_id="tenant", hold_id="hold", graph_id="graph", node_id="node",
                expected_fence=lease.fence,
            )
            assert released == (["execution"] if completed else [])
            if not completed:
                async with Persistence.open("node-freshness", storage=storage, tenant_id="tenant") as writer:
                    await writer.task.tasks.complete(
                        None, tenant_id="tenant", graph_id="graph", node_id="node", execution_id="execution",
                        result_digest="a" * 64, expected_fence=lease.fence,
                    )
        for mismatched in (TaskNodeRunResult("b" * 64, "execution"), TaskNodeRunResult("a" * 64, "other")):
            with pytest.raises(AIError) as raised:
                await launcher._completion_committed(
                    "graph", "node", mismatched, tenant_id="tenant", expected_fence=lease.fence,
                )
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
