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


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("missing_fields", "non_mapping_fields", "future_schema"))
async def test_pending_source_wrapper_errors_keep_storage_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    from dataclasses import replace
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence.codec import decode_envelope, encode_envelope
    from linktools.ai.persistence.state import record_key

    execution = _execution()
    arguments = StoredPayload.inline_json({"argument": True})
    operation = ToolOperationRecord("operation", "execution", "run", "call", "tool", arguments.digest,
        "binding", True, ToolOperationStatus.COMPLETED, None, 1, None,
        execution.created_at, execution.updated_at, arguments_payload=arguments,
        result_payload=StoredPayload.inline_json({"unrelated": "must not be read"}))
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        await state.checkpoint.tools.reserve(operation)
        store = state.checkpoint.tools.state_store
        key = record_key("native-result", "tenant", "checkpoint", "tool_operation", ("operation",))
        async def corrupt(transaction):
            record = await transaction.get_record(key)
            envelope = decode_envelope(record.data)
            payload = dict(envelope.value["payload"])
            if corruption == "missing_fields":
                payload.pop("fields")
            elif corruption == "non_mapping_fields":
                payload["fields"] = []
            else:
                payload["schema"] = 999
            changed = replace(record, storage_revision=record.storage_revision + 1,
                data=encode_envelope({**envelope.value, "payload": payload}))
            return changed
        changed = await store.read(corrupt)
        # A normal writer rejects malformed wrappers before publication.
        # Inject damaged persisted bytes directly to exercise the cold reader.
        from sqlalchemy import update
        from linktools.ai.persistence._sql import _record_values
        table = store.storage_group.metadata.tables["ai_step_tool_operations"]
        async with store.context.engine.begin() as connection:
            await connection.execute(update(table).where(table.c.store_key == store.store_key, table.c.key == key)
                                     .values(payload_json=_record_values(changed)["payload_json"]))
    async def forbidden(*args, **kwargs):
        raise AssertionError("malformed pending wrapper must fail before content hydration")
    monkeypatch.setattr(ContentArchive, "read_contents", forbidden)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        with pytest.raises(AIError) as raised:
            await read_pending_operations(state.checkpoint.tools.state_store, ("operation",),
                                          namespace="native-result", tenant_id="tenant")
        expected = (ErrorCode.STORAGE_VERSION_UNSUPPORTED if corruption == "future_schema"
                    else ErrorCode.STORAGE_INTEGRITY_ERROR)
        assert raised.value.code is expected
