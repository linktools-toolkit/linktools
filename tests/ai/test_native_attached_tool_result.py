#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native receipts retain original attachment manifests and file ownership."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ToolReturnPart

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError
from linktools.ai.model.attachments import read_attachment_payload, store_attachment_payload
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._archive import archive_content_scope
from linktools.ai.persistence._native_tool_result import restore_native_tool_result_payload
from linktools.ai.persistence.codec import decode_envelope, iter_runtime_object_refs, native_tool_result_reference
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore, ObjectRef, StoredPayload

from .test_native_result_portable_gc import Guard
from .test_native_text_result_runtime import _prepare, _publish
from .test_tool_execution_boundary import _request


async def _publish_attached(state: Persistence, *, archive_file: bool) -> tuple[str, StoredPayload, bytes]:
    commands, commit, current, checkpoint = await _prepare(state, "done", "done")
    operation = await state.checkpoint.tools.admit(_request())
    content = BinaryContent(b"original file bytes", media_type="application/octet-stream")
    original = ModelRequest(parts=[ToolReturnPart("runtime", content, tool_call_id=operation.tool_operation_id)])
    raw = encode_model_messages((original,))
    scope, _store = await state.checkpoint.tools.state_store.storage_group.open_archive()
    objects = (state.archive_object_store(PersistenceDomain.CHECKPOINT) if archive_file
               else state.object_store(PersistenceDomain.CHECKPOINT))

    async def write(body: bytes) -> ObjectRef:
        digest = hashlib.sha256(body).hexdigest()
        key = (f"v1/archive/{scope}/{archive_content_scope('native-result', 'tenant')}/{digest}" if archive_file
               else f"v1/runtime/receipt-file/{digest}")
        async def chunks():
            yield body
        await objects.put(key, chunks(), expected_size=len(body), expected_digest=digest)
        return ObjectRef(objects.store_id, key, digest, len(body))

    payload = await store_attachment_payload(raw, format="model-messages", write_object=write)
    assert payload is not None
    await state.checkpoint.tools.complete_payload(operation.tool_operation_id, tenant_id="tenant",
        owner=operation.owner, fence=operation.fence, result_payload=payload)
    checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[
        replace(original.parts[0], tool_name="tool", tool_call_id="call")]), *checkpoint.messages])
    await _publish(commands, commit, current, checkpoint)
    return operation.tool_operation_id, payload, raw


@pytest.mark.asyncio
@pytest.mark.parametrize("archive_file", (False, True))
async def test_attached_native_receipt_preserves_format_and_portable_file_owner(tmp_path: Path, archive_file: bool) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "source.db")
    portable = InMemoryObjectStore("portable")
    limits = SnapshotLimits(10000, 10_000_000)
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        operation_id, payload, raw = await _publish_attached(state, archive_file=archive_file)
        records = await state.checkpoint.tools.state_store.read(lambda tx: tx.scan_records())
        row = next(record for record in records if record.kind == "tool_operation")
        wire = decode_envelope(row.data).value["payload"]["fields"]["result_payload"]
        reference = native_tool_result_reference(wire)
        assert reference is not None and reference.attachment_format == "model-messages"
        assert reference.object_refs() == payload.object_refs()
        assert tuple(ref for _, ref in iter_runtime_object_refs(row.data, default_domain=PersistenceDomain.CHECKPOINT)) == payload.object_refs()
        assert (await state.checkpoint.tools.get_operation(operation_id, tenant_id="tenant")).result_payload == payload
        await state.purge_unreferenced_objects((), exclusive=Guard())
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=portable, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=portable, root=target, limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.from_root(target)) as state:
        result = (await state.checkpoint.tools.get_operation(operation_id, tenant_id="tenant")).result_payload
        assert result.encoding == "json-attachments"
        assert result.value["format"] == payload.value["format"]
        assert [entry["path"] for entry in result.value["attachments"]] == [entry["path"] for entry in payload.value["attachments"]]
        if not archive_file:
            assert canonical_json_bytes(result.to_json()) == canonical_json_bytes(payload.to_json())
        async def read(ref: ObjectRef) -> bytes:
            store = (state.archive_object_store(PersistenceDomain.CHECKPOINT) if ref.key.startswith("v1/archive/")
                     else state.object_store(PersistenceDomain.CHECKPOINT))
            return b"".join([chunk async for chunk in store.open(ref.key)])
        assert await read_attachment_payload(result, read_object=read) == raw
        await state.purge_unreferenced_objects((), exclusive=Guard())
