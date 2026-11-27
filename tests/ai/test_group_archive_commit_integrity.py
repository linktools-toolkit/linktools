#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completed metadata reads cannot bypass body checks or fresh publication."""

import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import insert, select
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart

from linktools.ai.model.messages import encode_model_messages
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import GroupRecordReader
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.state import StateGroupTransaction, StateStore, StoredRecord, active_state_scope
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.steps import native_store_locator

from .test_incremental_projection_boundaries import _storage
from .test_session_turn_sql import _TRANSCRIPT, accept
from .test_state_group_record_batch import _database, _record

pytestmark = pytest.mark.asyncio


class _OutsideScopeObjects(InMemoryObjectStore):
    async def _checked_open(self, key: str) -> AsyncIterator[bytes]:
        assert active_state_scope() is None, "Body reads must leave the metadata scope"
        async for chunk in super().open(key):
            yield chunk

    def open(self, key: str) -> AsyncIterator[bytes]:
        return self._checked_open(key)


async def _archive(owner: StateStore, objects: InMemoryObjectStore | None = None) -> ContentArchive:
    scope, store = await owner.storage_group.open_archive(create=True)
    return ContentArchive(store, scope, namespace="integrity", tenant_id="tenant", object_store=objects)


async def _observe(
    owner: StateStore, archive: ContentArchive, values: Sequence[tuple[str, bytes]],
    prepared: PreparedArchive,
) -> dict[bytes, StoredRecord]:
    keys = archive.content_record_keys(values)

    async def capture(group: StateGroupTransaction) -> dict[bytes, StoredRecord]:
        assert isinstance(group, GroupRecordReader)
        records = await group.get_records({owner: (_record("owner").key,), archive.store: keys})
        return dict(records[archive.store])

    observed = await owner.storage_group.read_many((owner, archive.store), capture)
    assert active_state_scope() is None
    archive.observe_content(prepared, keys, observed)
    return observed


async def _publish(archive: ContentArchive, prepared: PreparedArchive) -> None:
    await archive.store.mutate(lambda transaction: archive.publish(transaction, prepared))


async def test_native_turn_rejects_content_winner_after_preparation_and_rolls_back_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence.turns import SessionTurnKey

    key = SessionTurnKey("chat", "turn")
    async with Persistence.open("integrity", tenant_id="tenant", storage=_storage("sqlite", tmp_path)) as state:
        turns = state.session_turns
        await turns.create("chat", config={})
        assert await accept(turns, key.turn_id)
        native = turns._transcript_write_binding[2]
        original = native.prepare_archived_messages
        captured: list[tuple[object, ...]] = []
        rivals: list[dict[str, object]] = []
        metadata = build_step_sql_metadata()
        messages = metadata.tables["ai_messages"]

        async def snapshot() -> tuple[object, ...]:
            async with native.storage_context.engine.connect() as connection:
                rows = tuple([
                    tuple(dict(row) for row in (await connection.execute(
                        select(table).order_by(table.c.id),
                    )).mappings())
                    for table in metadata.sorted_tables
                ])
            return await turns.get_session("chat"), await turns.get_turn(key), rows

        async def inject_after_prepare(messages_input, prepared):
            result = await original(messages_input, prepared)
            assert active_state_scope() is None
            retained = next(iter(prepared.native_messages.values()))
            identity, body = retained.messages[0]
            rival = {"store_key": native.store_key, "store_locator": native_store_locator(native.store_key), "sha256": identity,
                     **body.to_native_values(), "raw_size": body.raw_size + 1}
            async with native.storage_context.engine.begin() as connection:
                await connection.execute(insert(messages).values(**rival))
            rivals.append(rival)
            captured.append(await snapshot())
            return result

        monkeypatch.setattr(native, "prepare_archived_messages", inject_after_prepare)
        with pytest.raises(AIError) as raised:
            await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert len(captured) == len(rivals) == 1
        assert await snapshot() == captured[0]
        async with native.storage_context.engine.connect() as connection:
            remaining = await connection.execute(select(
                messages.c.store_key, messages.c.store_locator, messages.c.sha256, messages.c.message,
                messages.c.compression, messages.c.raw_size, messages.c.payload_json,
            ))
            assert tuple(dict(row) for row in remaining.mappings()) == tuple(rivals)
        assert await state.execution.executions.get(key.turn_id, tenant_id="tenant") is None
        assert (await turns.get_session("chat"))["active_turn_id"] == b"turn"


@pytest.mark.parametrize("operation", ("content", "append"))
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
async def test_observed_existing_metadata_still_requires_the_object_body(
    tmp_path: Path, operation: str, damage: str,
) -> None:
    async with _database(tmp_path / "objects.db") as (_, owner, unused):
        objects = _OutsideScopeObjects()
        archive = await _archive(owner, objects)
        raw = b"".join(hashlib.sha256(str(index).encode()).digest() for index in range(1100))
        raw = encode_model_messages((ModelRequest(parts=[UserPromptPart([
            BinaryContent(raw, media_type="application/octet-stream"),
        ])]),))
        values = (("model-message", raw),)
        initial = PreparedArchive()
        await archive.prepare_content(values, initial)
        await _publish(archive, initial)
        prepared = PreparedArchive()
        observed = await _observe(owner, archive, values, prepared)
        record = next(iter(observed.values()))
        reference, = ContentRecord.from_record(record).object_refs()
        assert await objects.delete_object(reference.key, expected_digest=reference.digest)
        if damage == "corrupt":
            corrupt = b"!" * reference.size

            async def chunks() -> AsyncIterator[bytes]:
                yield corrupt

            await objects.put(reference.key, chunks(), expected_size=len(corrupt),
                              expected_digest=hashlib.sha256(corrupt).hexdigest())
        with pytest.raises(AIError) as raised:
            if operation == "content":
                await archive.prepare_content(values, prepared)
            else:
                await archive.prepare_append(archive.empty_root(), values, prepared)
        expected = ErrorCode.STORAGE_NOT_FOUND if damage == "missing" else ErrorCode.STORAGE_INTEGRITY_ERROR
        assert raised.value.code is expected
        assert not prepared.records and not prepared._observed_content and not prepared._verified_content
        assert await archive.store.read_records(tuple(observed)) == observed


async def test_consumed_missing_observation_cannot_hide_a_later_record_on_retry(tmp_path: Path) -> None:
    async with _database(tmp_path / "retry.db") as (_, owner, unused):
        archive = await _archive(owner)
        values = (("bytes", b"unknown-first"), ("bytes", b"observed-missing"))
        rival = PreparedArchive()
        await archive.prepare_content(values, rival)
        first_key, missing_key = archive.content_record_keys(values)
        first, later = rival.records[first_key], rival.records[missing_key]
        corrupt_first = replace(first, data={**first.data, "raw_size": first.data["raw_size"] + 1})
        await archive.store.mutate(lambda transaction: transaction.insert_record(corrupt_first))
        prepared = PreparedArchive()
        assert await _observe(owner, archive, values[1:], prepared) == {}
        with pytest.raises(AIError) as raised:
            await archive.prepare_content(values, prepared)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert not prepared.records and not prepared._observed_content
        corrupt_later = replace(later, data={**later.data, "raw_size": later.data["raw_size"] + 1})

        async def repair_and_insert(group: StateGroupTransaction) -> None:
            transaction = group.transaction(archive.store)
            assert await transaction.replace_record(replace(first, storage_revision=1), expected_storage_revision=0)
            await transaction.insert_record(corrupt_later)

        await archive.store.storage_group.mutate((archive.store,), repair_and_insert)
        with pytest.raises(AIError) as retry:
            await archive.prepare_content(values, prepared)
        assert retry.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert missing_key not in prepared.records
        assert await archive.store.read_records((missing_key,)) == {missing_key: corrupt_later}


async def test_combining_preparations_does_not_carry_body_validation_to_another_use(tmp_path: Path) -> None:
    async with _database(tmp_path / "combine.db") as (_, owner, unused):
        objects = _OutsideScopeObjects()
        archive = await _archive(owner, objects)
        values = (("model-message", encode_model_messages((ModelRequest(parts=[UserPromptPart([
            BinaryContent(b"large" * 4000, media_type="application/octet-stream"),
        ])]),))),)
        initial = PreparedArchive()
        await archive.prepare_content(values, initial)
        await _publish(archive, initial)
        prepared = PreparedArchive()
        observed = await _observe(owner, archive, values, prepared)
        await archive.prepare_content(values, prepared)
        assert prepared._verified_content
        combined = PreparedArchive.combine((prepared,))
        reference, = ContentRecord.from_record(next(iter(observed.values()))).object_refs()
        assert await objects.delete_object(reference.key, expected_digest=reference.digest)
        with pytest.raises(AIError) as raised:
            await archive.prepare_content(values, combined)
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
        assert not combined._verified_content and not prepared._verified_content
