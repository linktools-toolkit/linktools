#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native receipt roots survive portable restore and retire with their last owner."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from sqlalchemy import text

from linktools.ai.core import canonical_json_bytes
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive_closure import iter_native_message_roots
from linktools.ai.persistence.codec import decode_envelope, encode_domain, native_tool_result_reference
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.storage import InMemoryObjectStore, StoredPayload
from linktools.ai.steps import native_store_key

from .test_native_result_portable_gc import Guard
from .test_native_text_result_runtime import _prepare, _publish
from .test_native_tool_result_runtime import _settle


async def _old_receipts(state: Persistence, payload: StoredPayload) -> tuple[StoredRecord, ...]:
    _, archive = await state.checkpoint.tools.state_store.storage_group.open_archive()
    records = await archive.read(lambda tx: tx.scan_records())
    identity = "execution-payload:" + hashlib.sha256(canonical_json_bytes(encode_domain(payload))).hexdigest()
    return tuple(row for row in records if row.kind == "archive_content" and row.sort_key == identity.encode())


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ("large_text", "small_structured"))
async def test_native_tool_readonly_snapshot_and_last_owner_collection(tmp_path: Path, content: str) -> None:
    body = "terminal body " * 10000 if content == "large_text" else {"items": [None, True, 1, 1.0]}
    storage = PersistenceStorage.sqlite(tmp_path / "source.db")
    objects = InMemoryObjectStore("native-tool-portable")
    limits = SnapshotLimits(10000, 20_000_000)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        operation, payload = await _settle(state, body)
        checkpoint = replace(checkpoint, messages=[
            ModelRequest(parts=[ToolReturnPart("tool", body, tool_call_id="call")]), *checkpoint.messages])
        await _publish(commands, commit, current, checkpoint)
        tools = state.checkpoint.tools
        records = await tools.state_store.read(lambda tx: tx.scan_records())
        record = next(row for row in records if row.kind == "tool_operation")
        assert native_tool_result_reference(
            decode_envelope(record.data).value["payload"]["fields"]["result_payload"]) is not None
        # Native conversion releases the original shared Step content on the next GC.
        assert len(await _old_receipts(state, payload)) == 1
        await state.purge_unreferenced_objects((), exclusive=Guard())
        assert await _old_receipts(state, payload) == ()
        assert (await tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.from_root(target)) as state:
        tools = state.checkpoint.tools
        store = tools.state_store
        async with store.storage_group.context.engine.connect() as connection:
            assert await connection.scalar(text("SELECT count(*) FROM ai_step_tool_operations")) == 1
            assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='tool_operation'")) == 0
        assert (await tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload
        for partition in await store.storage_group.scan_stores():
            records = await partition.read(lambda tx: tx.scan_records_page(after=None, limit=1000))
            keys = tuple(row.key for row in records if row.kind != "tool_operation"
                         and tuple(iter_native_message_roots(row.data)))
            if keys:
                await partition.mutate(lambda tx: tx.delete_records(keys))
            facts = await partition.read(lambda tx: tx.scan_facts_page(after=None, limit=1000))
            owners = {fact.owner_key for fact in facts if tuple(iter_native_message_roots(fact.data))}
            for owner in owners:
                await partition.mutate(lambda tx: tx.delete_fact_streams(owner))
        await state.purge_unreferenced_objects((), exclusive=Guard())
        async with store.storage_group.context.engine.connect() as connection:
            assert await connection.scalar(text("select count(*) from ai_step_messages where store_key=:store"),
                {"store": native_store_key("native-result", "tenant")}) == 1
        assert (await tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload
        records = await store.read(lambda tx: tx.scan_records())
        keys = tuple(row.key for row in records if row.kind == "tool_operation")
        await store.mutate(lambda tx: tx.delete_records(keys))
        await state.purge_unreferenced_objects((), exclusive=Guard())
        async with store.storage_group.context.engine.connect() as connection:
            assert await connection.scalar(text("select count(*) from ai_step_messages where store_key=:store"),
                {"store": native_store_key("native-result", "tenant")}) == 0
            assert await connection.scalar(text("select count(*) from ai_step_sequences where store_key=:store"),
                {"store": native_store_key("native-result", "tenant")}) == 0
