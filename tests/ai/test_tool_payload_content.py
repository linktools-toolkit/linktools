#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Large tool receipts retain atomic database content and portable recovery."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive_closure import iter_archive_content_refs
from linktools.ai.persistence._snapshot import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, StoredPayload

from .test_tool_execution_boundary import _request


def _storage(backend: str, path: Path) -> PersistenceStorage:
    return PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(path / "tool.db")


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_tool_receipts_share_large_content_without_changing_payload_contract(tmp_path: Path, backend: str) -> None:
    payload = StoredPayload.inline_bytes(b"large independent tool result\n" * 20000)
    async with Persistence.open("tool-content", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        tools = state.checkpoint.tools
        _, archive = await tools.state_store.storage_group.open_archive(create=True)
        for index in range(2):
            operation = await tools.admit(replace(_request(), tool_operation_id=f"operation-{index}", tool_call_id=f"call-{index}"))
            await tools.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
                                         fence=operation.fence, result_payload=payload)
            loaded = await tools.get_operation(operation.tool_operation_id, tenant_id="tenant")
            assert loaded.result_payload == payload
        records = await tools.state_store.read(lambda tx: tx.scan_records())
        assert len(records) == 2
        references = [tuple(iter_archive_content_refs(record.data)) for record in records]
        assert references[0] == references[1] and len(references[0]) == 1
        assert all(len(canonical_json_bytes(record.data)) < 5000 for record in records)
        contents = await archive.read(lambda tx: tx.scan_records())
        assert len(contents) == 1 and contents[0].payload_bytes is not None
        assert sum(len(canonical_json_bytes(row.data)) + len(row.payload_bytes or b"") for row in (*records, *contents)) < payload.size


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_tool_payload_publication_rolls_back_with_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    async with Persistence.open("tool-content", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        tools = state.checkpoint.tools
        _, archive = await tools.state_store.storage_group.open_archive(create=True)
        operation = await tools.admit(_request())
        async def transaction_type(tx):
            return type(tx)
        transaction = await tools.state_store.read(transaction_type)
        original = transaction.replace_record
        async def fail_owner(tx, record, **kwargs):
            if record.kind == "tool_operation":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await original(tx, record, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(transaction, "replace_record", fail_owner)
            with pytest.raises(AIError):
                await tools.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
                    fence=operation.fence, result_payload=StoredPayload.inline_bytes(b"large" * 20000))
        assert await archive.read(lambda tx: tx.scan_records()) == ()
        assert await tools.get_operation(operation.tool_operation_id, tenant_id="tenant") == operation


@pytest.mark.asyncio
async def test_tool_content_cold_lookup_and_snapshot_preserve_exact_result(tmp_path: Path) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "tool.db")
    payload = StoredPayload.inline_bytes(b"portable tool result\n" * 20000)
    async with Persistence.open("tool-content", tenant_id="tenant", storage=config) as state:
        tools = state.checkpoint.tools
        await tools.state_store.storage_group.open_archive(create=True)
        operation = await tools.admit(_request())
        await tools.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
            fence=operation.fence, result_payload=payload)
    objects = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=10_000_000)
    async with Persistence.open("tool-content", tenant_id="tenant", storage=config, read_only=True) as state:
        loaded = await state.checkpoint.tools.get_by_call_ids(operation.agent_run_id, (operation.tool_call_id,), tenant_id="tenant")
        assert loaded[0].result_payload == payload
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=tmp_path / "restored", limits=limits)
    async with Persistence.open("tool-content", tenant_id="tenant", storage=PersistenceStorage.from_root(tmp_path / "restored"), read_only=True) as state:
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload


@pytest.mark.asyncio
async def test_small_tool_payload_and_opaque_reference_shaped_data_remain_inline() -> None:
    payload = StoredPayload.inline_json({"$tool_payload": "opaque", "archive_scope": "not-a-reference"})
    async with Persistence.open("tool-content", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        tools = state.checkpoint.tools
        _, archive = await tools.state_store.storage_group.open_archive(create=True)
        operation = await tools.admit(_request())
        await tools.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
            fence=operation.fence, result_payload=payload)
        records = await tools.state_store.read(lambda tx: tx.scan_records())
        assert tuple(iter_archive_content_refs(records[0].data)) == ()
        assert await archive.read(lambda tx: tx.scan_records()) == ()
        assert (await tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload


@pytest.mark.asyncio
async def test_missing_tool_content_is_not_treated_as_an_empty_result() -> None:
    async with Persistence.open("tool-content", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        tools = state.checkpoint.tools
        _, archive = await tools.state_store.storage_group.open_archive(create=True)
        operation = await tools.admit(_request())
        await tools.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
            fence=operation.fence, result_payload=StoredPayload.inline_bytes(b"large" * 20000))
        contents = await archive.read(lambda tx: tx.scan_records())
        assert len(contents) == 1
        await archive.mutate(lambda tx: tx.delete_record(contents[0].key))
        with pytest.raises(AIError) as error:
            await tools.get_operation(operation.tool_operation_id, tenant_id="tenant")
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_archived_tool_attachment_survives_cleanup_and_snapshot(tmp_path: Path) -> None:
    from pydantic_ai import messages as sdk
    from linktools.ai.model.attachments import read_attachment_payload, store_attachment_payload
    from linktools.ai.model.messages import encode_model_messages
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence.objects import ObjectKeyFactory, put_runtime_object
    from .test_evaluation_retention import Offline

    objects = InMemoryObjectStore()
    config = PersistenceStorage.sqlite(tmp_path / "attachments.db", object_store=objects)
    raw = encode_model_messages([sdk.ModelRequest(parts=[sdk.ToolReturnPart(
        "tool", ["large text" * 20000, sdk.BinaryContent(b"attached file", media_type="image/png")],
        tool_call_id="call",
    )])])
    async with Persistence.open("tool-content", tenant_id="tenant", storage=config) as state:
        tools = state.checkpoint.tools
        await tools.state_store.storage_group.open_archive(create=True)
        payload = await store_attachment_payload(raw, format="model-messages", write_object=lambda data: put_runtime_object(
            state.object_store(PersistenceDomain.CHECKPOINT), ObjectKeyFactory("tool-content"),
            PersistenceDomain.CHECKPOINT, "tenant", data,
        ))
        assert payload is not None
        operation = await tools.admit(_request())
        await tools.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
            fence=operation.fence, result_payload=payload)
        owner, = await tools.state_store.read(lambda tx: tx.scan_records())
        assert len(tuple(iter_archive_content_refs(owner.data))) == 1
        candidates = tuple((PersistenceDomain.CHECKPOINT, ref) for ref in payload.object_refs())
        assert candidates
        cleanup = await state.purge_unreferenced_objects(candidates, exclusive=Offline())
        assert cleanup.retained == candidates and not cleanup.deleted and not cleanup.blocked
    exported = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=1000, max_bytes=10_000_000)
    async with Persistence.open("tool-content", tenant_id="tenant", storage=config, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=exported, limits=limits)
    await Persistence.restore_snapshot(snapshot, object_store=exported, root=tmp_path / "restored", limits=limits)
    async with Persistence.open("tool-content", tenant_id="tenant", storage=PersistenceStorage.from_root(tmp_path / "restored"), read_only=True) as state:
        restored = await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")
        assert await read_attachment_payload(restored.result_payload, read_object=lambda ref: state.read_object(
            PersistenceDomain.CHECKPOINT, ref,
        )) == raw
