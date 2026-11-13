#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native input snapshots retain their only body owner through offline cleanup."""

from pathlib import Path
import hashlib

from linktools.ai.core import canonical_json_bytes

import pytest
from sqlalchemy import text

from linktools.ai.model import StoredUserInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive_closure import iter_native_message_roots
from linktools.ai.steps import native_store_key
from linktools.ai.persistence.codec import decode_envelope, encode_domain, native_user_input_reference
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.storage import InMemoryObjectStore, StoredPayload

from .test_native_input_runtime import _prepare_input
from .test_native_result_portable_gc import Guard
from .test_native_text_result_runtime import _publish


async def _old_inputs(state: Persistence, original: StoredUserInput) -> tuple[StoredRecord, ...]:
    _, archive = await state.execution.executions.state_store.storage_group.open_archive()
    records = await archive.read(lambda tx: tx.scan_records())
    return tuple(row for row in records if row.kind == "archive_content"
                 and row.data.get("codec") == "execution-payload"
                 and row.data.get("raw_digest") == hashlib.sha256(canonical_json_bytes(encode_domain(original))).hexdigest())


@pytest.mark.asyncio
async def test_native_input_snapshot_and_last_owner_collection(tmp_path: Path) -> None:
    body = "terminal body " * 10000
    original = StoredUserInput("text", StoredPayload.inline_text(body))
    storage = PersistenceStorage.sqlite(tmp_path / "source.db")
    objects = InMemoryObjectStore("native-input-portable")
    limits = SnapshotLimits(10000, 20_000_000)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, original, body)
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert native_user_input_reference(
            decode_envelope(record.data).value["payload"]["fields"]["stored_user_input"]) is not None
        assert len(await _old_inputs(state, original)) == 1
    # A portable snapshot includes only reachable bodies even before local GC.
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        await state.purge_unreferenced_objects((), exclusive=Guard())
        assert await _old_inputs(state, original) == ()
        assert (await state.execution.executions.get_fresh("execution", tenant_id="tenant")).stored_user_input == original
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.from_root(target)) as state:
        repository = state.execution.executions
        store = repository.state_store
        assert await _old_inputs(state, original) == ()
        assert (await repository.get_fresh("execution", tenant_id="tenant")).stored_user_input == original
        for partition in await store.storage_group.scan_stores():
            records = await partition.read(lambda tx: tx.scan_records_page(after=None, limit=1000))
            keys = tuple(row.key for row in records if row.kind != "execution"
                         and tuple(iter_native_message_roots(row.data)))
            if keys:
                await partition.mutate(lambda tx: tx.delete_records(keys))
            facts = await partition.read(lambda tx: tx.scan_facts_page(after=None, limit=1000))
            owners = {fact.owner_key for fact in facts if tuple(iter_native_message_roots(fact.data))}
            for owner in owners:
                await partition.mutate(lambda tx: tx.delete_fact_streams(owner))
        await state.purge_unreferenced_objects((), exclusive=Guard())
        async with store.storage_group.context.engine.connect() as connection:
            assert await connection.scalar(text("select count(*) from ai_step_messages where store_key = :key"), {"key": native_store_key("native-result", "tenant")}) == 2  # Execution retains both input and result Native messages.
        assert (await repository.get_fresh("execution", tenant_id="tenant")).stored_user_input == original
        await store.mutate(lambda tx: tx.delete_records((repository.execution_record_key("execution"),)))
        await state.purge_unreferenced_objects((), exclusive=Guard())
        async with store.storage_group.context.engine.connect() as connection:
            assert await connection.scalar(text("select count(*) from ai_step_messages")) == 0
            assert await connection.scalar(text("select count(*) from ai_step_sequences")) == 0
