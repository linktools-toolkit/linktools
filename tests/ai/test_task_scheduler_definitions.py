#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scheduler definition reuse preserves live state and transaction ownership."""

import asyncio
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from linktools.ai.core import Principal, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.task import (
    Task, TaskExpanderRef, TaskGraph, TaskNode, TaskNodeResultRef, TaskRef,
)

from linktools.ai.task.engine import TaskEngine

from ._task_test_helpers import admit_graph
from .test_task_io_bounded_reads import _admit
from .test_task_io_measurements import _measure


@pytest.fixture(params=("memory", "local_sqlite", "sqlite"))
def storage(tmp_path: Path, request: pytest.FixtureRequest) -> PersistenceStorage:
    if request.param == "memory":
        return PersistenceStorage.in_memory()
    if request.param == "local_sqlite":
        return PersistenceStorage.from_root(tmp_path, exclusive_writer=True)
    return PersistenceStorage.sqlite(tmp_path / "state.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("width", (2, 200))
@pytest.mark.parametrize("input_size", (0, 4096))
async def test_scheduler_definition_read_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int, input_size: int,
) -> None:
    async with Persistence.open(
        "scheduler-budget", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
        tenant_id="tenant",
    ) as state:
        graph = TaskGraph("graph", tuple(
            TaskNode(f"node-{index}", input={"value": "x" * input_size})
            for index in range(width)
        ))
        await _admit(state, graph.nodes)
        repository = state.task.tasks
        measurements = {}
        for phase in ("cold", "warm", "nested", "warm_after_nested", "public"):
            with _measure(monkeypatch) as counts:
                if phase == "nested":
                    actual = await repository.state_store.mutate(
                        lambda transaction: repository.scheduler_state("graph", tenant_id="tenant")
                    )
                elif phase == "public":
                    actual = await repository.graph_state("graph", tenant_id="tenant")
                else:
                    actual = await repository.scheduler_state("graph", tenant_id="tenant")
            assert actual is not None and actual.nodes == tuple(sorted(graph.nodes, key=lambda node: node.node_id))
            full = phase in {"cold", "nested", "public"}
            assert counts["record_rows"] == (2 * width + 1 if full else width + 1)
            assert counts["SELECT"] <= (5 if full else 4)
            assert counts["begins"] == counts["commits"] == 1
            assert counts["write_commits"] == 0
            measurements[phase] = dict(counts)
        assert measurements["warm"]["record_payload_json_bytes"] < measurements["cold"]["record_payload_json_bytes"]
        print("SCHEDULER_DEFINITION_IO " + json.dumps({
            "width": width, "input_size": input_size, "measurements": measurements,
        }, sort_keys=True))


@pytest.mark.asyncio
async def test_scheduler_definition_miss_does_not_repeat_state_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open(
        "scheduler-miss", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
        tenant_id="tenant",
    ) as state:
        await admit_graph(state, TaskGraph("graph", (
            TaskNode("source", expander=TaskExpanderRef("expand", 1)),
        )))
        await admit_graph(state, TaskGraph("other", (TaskNode("source", input={"value": "other"}),)))
        repository = state.task.tasks
        measurements = {}
        with _measure(monkeypatch) as counts:
            await repository.state_store.mutate(
                lambda transaction: repository.scheduler_state("graph", tenant_id="tenant")
            )
        measurements["nested_cold"] = dict(counts)
        with _measure(monkeypatch) as counts:
            await repository.scheduler_state("graph", tenant_id="tenant")
        assert counts["record_rows"] == 3
        measurements["cold_after_nested"] = dict(counts)
        lease = await repository.claim("graph", "source", tenant_id="tenant", owner="worker", lease_seconds=300)
        await repository.complete(
            lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
            expanded_nodes=(TaskNode("child", ("source",)),),
        )
        for phase, graph_id, rows in (
            ("expansion", "graph", 5), ("expanded_warm", "graph", 3),
            ("alternate", "other", 3), ("return", "graph", 5),
        ):
            with _measure(monkeypatch) as counts:
                actual = await repository.scheduler_state(graph_id, tenant_id="tenant")
            assert counts["record_rows"] == rows
            assert counts["SELECT"] <= (4 if phase == "expanded_warm" else 5)
            assert counts["write_commits"] == (1 if phase == "expansion" else 0)
            if phase == "expansion":
                assert actual.node_states[0].status is TaskStatus.READY
            assert actual == await repository.graph_state(graph_id, tenant_id="tenant")
            measurements[phase] = dict(counts)
        print("SCHEDULER_DEFINITION_MISS_IO " + json.dumps(measurements, sort_keys=True))


@pytest.mark.asyncio
async def test_scheduler_definitions_keep_lease_and_expansion_current(storage: PersistenceStorage) -> None:
    async with Persistence.open("scheduler-live", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (
            TaskNode("source", expander=TaskExpanderRef("expand", 1)),
        )))
        repository = state.task.tasks
        lease = await repository.claim("graph", "source", tenant_id="tenant", owner="worker", lease_seconds=30)
        captured = await repository.scheduler_state("graph", tenant_id="tenant")
        lease = await repository.renew(lease, tenant_id="tenant", lease_seconds=300)
        renewed = await repository.scheduler_state("graph", tenant_id="tenant")
        assert renewed == await repository.graph_state("graph", tenant_id="tenant")
        assert renewed.node_states[0].lease_expires_at > captured.node_states[0].lease_expires_at
        await repository.complete(
            lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
            expanded_nodes=(TaskNode("child", ("source",)),),
        )
        expanded = await repository.scheduler_state("graph", tenant_id="tenant")
        assert expanded == await repository.graph_state("graph", tenant_id="tenant")
        assert [node.node_id for node in expanded.nodes] == ["child", "source"]
        assert expanded.node_states[0].status is TaskStatus.READY
        await repository.cancel_graph("graph", tenant_id="tenant")
        cancelled = await repository.scheduler_state("graph", tenant_id="tenant")
        assert cancelled.status is TaskStatus.CANCELLED
        assert cancelled == await repository.graph_state("graph", tenant_id="tenant")


@pytest.mark.asyncio
async def test_scheduler_definitions_do_not_escape_nested_rollback(storage: PersistenceStorage) -> None:
    async with Persistence.open("scheduler-rollback", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (
            TaskNode("source", expander=TaskExpanderRef("expand", 1)),
        )))
        repository = state.task.tasks
        lease = await repository.claim("graph", "source", tenant_id="tenant", owner="worker", lease_seconds=300)
        await repository.scheduler_state("graph", tenant_id="tenant")

        async def aborted(transaction):
            await repository.complete(
                lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
                expanded_nodes=(TaskNode("child", ("source",), input={"value": "uncommitted"}),),
            )
            provisional = await repository.scheduler_state("graph", tenant_id="tenant")
            assert provisional.nodes[0].input == {"value": "uncommitted"}
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as aborted_error:
            await repository.state_store.mutate(aborted)
        assert aborted_error.value.code is ErrorCode.STORAGE_CONFLICT
        await repository.complete(
            lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
            expanded_nodes=(TaskNode("child", ("source",), input={"value": "committed"}),),
        )
        current = await repository.scheduler_state("graph", tenant_id="tenant")
        assert current.nodes[0].input == {"value": "committed"}
        assert current == await repository.graph_state("graph", tenant_id="tenant")


@pytest.mark.asyncio
async def test_scheduler_definitions_observe_expansion_after_older_read_returns(
    storage: PersistenceStorage, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open("scheduler-concurrent", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", tuple(
            TaskNode(node_id, expander=TaskExpanderRef("expand", 1)) for node_id in ("a", "b")
        )))
        repository = state.task.tasks
        leases = [await repository.claim("graph", node_id, tenant_id="tenant", owner="worker", lease_seconds=300)
                  for node_id in ("a", "b")]
        captured = asyncio.Event()
        release = asyncio.Event()
        original_mutate = repository.state_store.mutate
        older = None

        async def delayed_return(operation):
            result = await original_mutate(operation)
            if asyncio.current_task() is older:
                captured.set()
                await release.wait()
            return result

        with monkeypatch.context() as patch:
            patch.setattr(repository.state_store, "mutate", delayed_return)
            older = asyncio.create_task(repository.scheduler_state("graph", tenant_id="tenant"))
            try:
                await asyncio.wait_for(captured.wait(), 5)
                for lease in leases:
                    await repository.complete(
                        lease, tenant_id="tenant", execution_id=f"execution-{lease.node_id}", result_digest="a" * 64,
                        expanded_nodes=(TaskNode(f"child-{lease.node_id}", (lease.node_id,)),),
                    )
                current = await repository.scheduler_state("graph", tenant_id="tenant")
                assert {node.node_id for node in current.nodes} == {"a", "b", "child-a", "child-b"}
            finally:
                release.set()
                await older
        assert await repository.scheduler_state("graph", tenant_id="tenant") == current
        assert current == await repository.graph_state("graph", tenant_id="tenant")


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("definition", "state", "state_payload"))
async def test_scheduler_definition_reuse_preserves_full_read_integrity(
    storage: PersistenceStorage, fault: str,
) -> None:
    async with Persistence.open("scheduler-integrity", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (TaskNode("target"), TaskNode("other"))))
        await admit_graph(state, TaskGraph("alternate", (TaskNode("target"),)))
        repository = state.task.tasks
        captured = await repository.scheduler_state("graph", tenant_id="tenant")
        key = (repository._definition_key if fault == "definition" else repository._state_key)("graph", "other")
        async def corrupt(transaction):
            if fault == "state_payload":
                record = await transaction.get_record(key)
                assert await transaction.replace_record(
                    replace(record, data={}, storage_revision=record.storage_revision + 1),
                    expected_storage_revision=record.storage_revision,
                )
            else:
                await transaction.delete_record(key)

        await repository.state_store.mutate(corrupt)
        if fault == "definition":
            assert await repository.scheduler_state("graph", tenant_id="tenant") == captured
        else:
            with pytest.raises(AIError) as warm:
                await repository.scheduler_state("graph", tenant_id="tenant")
            assert warm.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as public:
            await repository.graph_state("graph", tenant_id="tenant")
        assert public.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        await repository.scheduler_state("alternate", tenant_id="tenant")
        with pytest.raises(AIError) as cold:
            await repository.scheduler_state("graph", tenant_id="tenant")
        assert cold.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_scheduler_definitions_isolate_returned_nested_values(storage: PersistenceStorage) -> None:
    nested = {"items": [{"value": "original"}]}
    node = TaskNode.from_resolved(
        "target", ("source",), task=TaskRef("work", 1), input=nested,
        original_input=nested, output_contract={"properties": {"nested": nested}},
        input_refs={"parent": TaskNodeResultRef("source")}, expander=TaskExpanderRef("expand", 1),
    )
    nested["items"][0]["value"] = "author-mutated"
    async with Persistence.open("scheduler-copy", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (TaskNode("source"), node)))
        repository = state.task.tasks
        captured = await repository.scheduler_state("graph", tenant_id="tenant")
        returned = next(value for value in captured.nodes if value.node_id == "target")
        returned.input["items"][0]["value"] = "input-mutated"
        returned.original_input["items"][0]["value"] = "original-mutated"
        returned.output_contract["properties"]["nested"]["items"][0]["value"] = "schema-mutated"
        returned.to_mapping()["input"]["items"][0]["value"] = "projection-mutated"
        with pytest.raises(TypeError):
            returned.input_refs["parent"] = TaskNodeResultRef("target")
        with pytest.raises(FrozenInstanceError):
            returned.task.id = "changed"
        actual = await repository.scheduler_state("graph", tenant_id="tenant")
        assert actual == captured == await repository.graph_state("graph", tenant_id="tenant")
        assert actual.nodes[1].input == {"items": [{"value": "original"}]}


@pytest.mark.asyncio
async def test_scheduler_definitions_do_not_replace_terminal_graph(storage: PersistenceStorage) -> None:
    async with Persistence.open("scheduler-incarnation", storage=storage, tenant_id="tenant") as state:
        graph = TaskGraph("graph", (TaskNode("target", input={"value": "original"}),))
        await admit_graph(state, graph)
        repository = state.task.tasks
        await repository.scheduler_state("graph", tenant_id="tenant")
        await repository.cancel_graph("graph", tenant_id="tenant")
        with pytest.raises(AIError) as changed:
            await admit_graph(state, TaskGraph("graph", (TaskNode("target", input={"value": "replacement"}),)))
        assert changed.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        await admit_graph(state, graph)
        current = await repository.scheduler_state("graph", tenant_id="tenant")
        assert current.status is TaskStatus.CANCELLED
        assert current.nodes[0].input == {"value": "original"}


@pytest.mark.asyncio
async def test_scheduler_definitions_restore_into_fresh_source(tmp_path: Path) -> None:
    archive = InMemoryObjectStore("archive")
    limits = SnapshotLimits(max_entries=1000, max_bytes=1024 * 1024)
    async def work(context):
        return "unused"

    task = Task("restore-task", work, effect_policy="none")
    async with Persistence.open(
        "scheduler-restore", storage=PersistenceStorage.from_root(tmp_path / "source"), tenant_id="tenant",
    ) as source:
        async with TaskEngine.open(task, persistence=source) as engine:
            submission = await engine.prepare_submission(
                TaskGraph("graph", (TaskNode("target", task=task, input={"value": "original"}),)),
                idempotency_key="restore", principal=Principal("tester", "tenant"),
            )
            await source.task.admissions.admit_prepared(submission)
            captured = await source.task.tasks.scheduler_state("graph", tenant_id="tenant")
    reader = Persistence(PersistenceStorage.from_root(tmp_path / "source"))
    await reader.initialize(namespace="scheduler-restore", tenant_id="tenant", read_only=True)
    try:
        reference = await reader.export_snapshot(object_store=archive, limits=limits)
    finally:
        await reader.close()
    await Persistence.restore_snapshot(reference, object_store=archive, root=tmp_path / "restored", limits=limits)
    async with Persistence.open(
        "scheduler-restore", storage=PersistenceStorage.from_root(tmp_path / "restored"), tenant_id="tenant",
    ) as restored:
        repository = restored.task.tasks
        assert (await repository.graph_state("graph", tenant_id="tenant")).nodes == captured.nodes
        await repository.state_store.mutate(
            lambda transaction: transaction.delete_record(repository._definition_key("graph", "target"))
        )
        with pytest.raises(AIError) as cold:
            await repository.scheduler_state("graph", tenant_id="tenant")
        assert cold.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("failure", "cancel", "unknown", "confirmed"))
async def test_scheduler_definitions_publish_only_confirmed_reads(
    storage: PersistenceStorage, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    async with Persistence.open("scheduler-outcome", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (TaskNode("source"),)))
        repository = state.task.tasks
        original_mutate = repository.state_store.mutate
        original_decode = repository._decode
        definition_reads = []

        async def decode(record, target):
            if target is TaskNode:
                definition_reads.append(record.key_digest)
            return await original_decode(record, target)

        async def interrupt(operation):
            if outcome == "confirmed":
                await original_mutate(operation)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

            async def aborted(transaction):
                await operation(transaction)
                if outcome == "cancel":
                    raise asyncio.CancelledError()
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

            try:
                return await original_mutate(aborted)
            except AIError as error:
                if outcome == "unknown":
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from error
                raise

        async def unavailable(operation):
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

        with monkeypatch.context() as patch:
            patch.setattr(repository.state_store, "mutate", interrupt)
            if outcome == "unknown":
                patch.setattr(repository.state_store, "read", unavailable)
            if outcome == "confirmed":
                confirmed = await repository.scheduler_state("graph", tenant_id="tenant")
                assert confirmed.nodes[0].node_id == "source"
            elif outcome == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await repository.scheduler_state("graph", tenant_id="tenant")
            else:
                with pytest.raises(AIError) as failed:
                    await repository.scheduler_state("graph", tenant_id="tenant")
                assert failed.value.code is ErrorCode.STORAGE_UNAVAILABLE
        with monkeypatch.context() as patch:
            patch.setattr(repository, "_decode", decode)
            current = await repository.scheduler_state("graph", tenant_id="tenant")
        assert len(definition_reads) == (0 if outcome == "confirmed" else 1)
        assert current == await repository.graph_state("graph", tenant_id="tenant")


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("terminal", "recovery"))
async def test_scheduler_definitions_release_at_inactive_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str,
) -> None:
    async with Persistence.open(
        "scheduler-inactive", storage=PersistenceStorage.sqlite(tmp_path / "state.db"), tenant_id="tenant",
    ) as state:
        await admit_graph(state, TaskGraph("graph", (TaskNode("source", input={"body": "x" * 4096}),)))
        repository = state.task.tasks
        lease = await repository.claim("graph", "source", tenant_id="tenant", owner="worker", lease_seconds=300)
        await repository.scheduler_state("graph", tenant_id="tenant")
        if boundary == "terminal":
            await repository.complete(
                lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
            )
        else:
            await repository.mark_recovery_required(
                lease, tenant_id="tenant", execution_id="execution",
                error_code=ErrorCode.STORAGE_RECOVERY_REQUIRED.value, error_digest="b" * 64,
            )
        inactive = await repository.scheduler_state("graph", tenant_id="tenant")
        assert inactive.status is (TaskStatus.SUCCEEDED if boundary == "terminal" else TaskStatus.RECOVERY_REQUIRED)
        with _measure(monkeypatch) as counts:
            assert await repository.scheduler_state("graph", tenant_id="tenant") == inactive
        assert counts["record_rows"] == 3
        assert counts["write_commits"] == 0


@pytest.mark.asyncio
async def test_scheduler_definitions_preserve_delegated_nested_scope(storage: PersistenceStorage) -> None:
    from linktools.ai.persistence._task_repository import TaskRepositoryImpl

    async with Persistence.open("scheduler-delegated", storage=storage, tenant_id="tenant") as state:
        await admit_graph(state, TaskGraph("graph", (
            TaskNode("source", expander=TaskExpanderRef("expand", 1)),
        )))
        store = state.task.tasks.state_store

        class DelegatingStore:
            @property
            def storage_group(self):
                return store.storage_group

            async def read(self, operation):
                return await store.read(operation)

            async def mutate(self, operation):
                return await store.mutate(operation)

        repository = TaskRepositoryImpl(DelegatingStore(), namespace="scheduler-delegated", tenant_id="tenant")
        lease = await repository.claim("graph", "source", tenant_id="tenant", owner="worker", lease_seconds=300)
        await repository.scheduler_state("graph", tenant_id="tenant")

        async def aborted(transaction):
            await repository.complete(
                lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
                expanded_nodes=(TaskNode("child", ("source",), input={"value": "uncommitted"}),),
            )
            assert (await repository.scheduler_state("graph", tenant_id="tenant")).nodes[0].input == {"value": "uncommitted"}
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as aborted_error:
            await store.mutate(aborted)
        assert aborted_error.value.code is ErrorCode.STORAGE_CONFLICT
        await repository.complete(
            lease, tenant_id="tenant", execution_id="execution", result_digest="a" * 64,
            expanded_nodes=(TaskNode("child", ("source",), input={"value": "committed"}),),
        )
        current = await repository.scheduler_state("graph", tenant_id="tenant")
        assert current.nodes[0].input == {"value": "committed"}
        assert current == await repository.graph_state("graph", tenant_id="tenant")
