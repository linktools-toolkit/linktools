#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pending admission reads do not hydrate an unrelated terminal outcome."""

from pathlib import Path

import pytest

from linktools.ai.core import ToolOperationStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence._pending_operation_reference import read_pending_operations
from linktools.ai.persistence.contracts import ToolOperationRecord
from linktools.ai.storage import StoredPayload
from .test_step_projection import _execution


@pytest.mark.asyncio
async def test_pending_projection_reads_only_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    execution = _execution()
    arguments = StoredPayload.inline_json({"argument": True})
    output = StoredPayload.inline_json({"unrelated": "terminal result"})
    operation = ToolOperationRecord("operation", "execution", "run", "call", "tool", arguments.digest,
        "binding", True, ToolOperationStatus.COMPLETED, None, 1, None,
        execution.created_at, execution.updated_at, arguments_payload=arguments, result_payload=output)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        await state.checkpoint.tools.reserve(operation)
    calls = []
    original = ContentArchive.read_contents

    async def read(archive: ContentArchive, identities: tuple, **kwargs: object) -> tuple:
        calls.append(tuple(identities))
        return await original(archive, identities, **kwargs)

    monkeypatch.setattr(ContentArchive, "read_contents", read)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        values = await read_pending_operations(state.checkpoint.tools.state_store, ("operation",),
            namespace="native-result", tenant_id="tenant")
        assert values[0].arguments_payload == arguments
        assert values[0].execution_id == "execution"
    assert len(calls) == 1
    assert len(calls[0]) == 1


@pytest.mark.asyncio
async def test_pending_owner_survives_split_partition_cold_snapshot(tmp_path: Path) -> None:
    from dataclasses import replace
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.execution import ExecutionRecoveryDetail, PendingDeferredCall, PendingToolContinuation
    from linktools.ai.persistence.codec import decode_envelope, pending_operation_reference
    from linktools.ai.persistence.snapshots import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore
    from .test_execution_recovery_owner_roundtrip import Guard

    arguments = StoredPayload.inline_json({"argument": True})
    current = _execution()
    pending = PendingToolContinuation("run", approvals=(
        PendingDeferredCall("call", "tool", arguments, operation_id="operation"),))
    current = replace(current, status=ExecutionStatus.WAITING_DEFERRED,
        recovery=ExecutionRecoveryDetail(agent_run_id="run", pending_tools=pending))
    operation = ToolOperationRecord("operation", "execution", "run", "call", "tool", arguments.digest,
        "binding", True, ToolOperationStatus.PENDING, None, 0, None,
        current.created_at, current.updated_at, arguments_payload=arguments)
    storage = PersistenceStorage.sqlite(tmp_path / "source.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        repository = state.execution.executions
        assert repository.state_store is not state.checkpoint.tools.state_store
        assert repository.state_store.storage_group is state.checkpoint.tools.state_store.storage_group
        await state.checkpoint.tools.reserve(operation)
        await repository.create_execution(current)
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert record is not None
        fields = decode_envelope(record.data).value["payload"]["fields"]
        argument = fields["recovery"]["fields"]["pending_tools"]["fields"]["arguments"]["0"]
        assert pending_operation_reference(argument) == "operation"
        assert await repository.get_recovery("execution", tenant_id="tenant") == current
        await state.purge_unreferenced_objects((), exclusive=Guard())
    objects = InMemoryObjectStore()
    limits = SnapshotLimits(10000, 20_000_000)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=tmp_path / "restored", limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant",
                                storage=PersistenceStorage.from_root(tmp_path / "restored")) as state:
        assert await state.execution.executions.get_recovery("execution", tenant_id="tenant") == current
        await state.purge_unreferenced_objects((), exclusive=Guard())
