#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completed metadata reads cannot bypass body checks or fresh publication."""

import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.state import GroupRecordReader
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.persistence._step_archive import PreparedAgentRunCheckpointBatch
from linktools.ai.persistence._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.persistence.state import StateGroupTransaction, StateStore, StoredRecord, active_state_scope
from linktools.ai.storage import InMemoryObjectStore

from .test_pending_turn_commit import _commit, _pending_turn, _snapshot, _turn_state
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
        records = await group.get_records({owner: (_record("owner").key_digest,), archive.store: keys})
        return dict(records[archive.store])

    observed = await owner.storage_group.read_many((owner, archive.store), capture)
    assert active_state_scope() is None
    archive.observe_content(prepared, keys, observed)
    return observed


async def _publish(archive: ContentArchive, prepared: PreparedArchive) -> None:
    await archive.store.mutate(lambda transaction: archive.publish(transaction, prepared))


async def test_pending_turn_rejects_content_winner_after_preparation_and_rolls_back_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _turn_state("sqlite", tmp_path) as state:
        turn = _pending_turn(state)
        step = state.storage.run_store.read_store(PersistenceDomain.CONVERSATION)
        original = step.prepare_conversation_checkpoint
        captured: list[tuple[object, ...]] = []
        rivals: list[StoredRecord] = []
        content = await step.transcript_repository.content_archive(create=True)

        async def inject_after_prepare(
            run: AgentRunRecord, checkpoint: AgentRunCheckpoint, *, archive: PreparedArchive | None = None,
            observed_records=None,
        ) -> tuple[AgentRunCheckpoint, PreparedAgentRunCheckpointBatch]:
            result = await original(run, checkpoint, archive=archive, observed_records=observed_records)
            assert active_state_scope() is None
            prepared = result[1][0].archive
            record = next(value for value in prepared.records.values() if value.kind == "archive_content")
            rival = replace(record, data={**record.data, "raw_size": record.data["raw_size"] + 1})
            await content.store.mutate(lambda transaction: transaction.insert_record(rival))
            rivals.append(rival)
            captured.append(await _snapshot(state))
            return result

        monkeypatch.setattr(step, "prepare_conversation_checkpoint", inject_after_prepare)
        with pytest.raises(AIError) as raised:
            await _commit(state, turn)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert len(captured) == len(rivals) == 1
        assert await _snapshot(state) == captured[0]
        assert await content.store.read(lambda transaction: transaction.scan_records()) == tuple(rivals)
        assert await state.storage.execution.executions.get("execution", tenant_id="tenant") is None
        assert await state.storage.conversation.sessions.get("session", tenant_id="tenant") == state.session


@pytest.mark.parametrize("operation", ("content", "append"))
@pytest.mark.parametrize("damage", ("missing", "corrupt"))
async def test_observed_existing_metadata_still_requires_the_object_body(
    tmp_path: Path, operation: str, damage: str,
) -> None:
    async with _database(tmp_path / "objects.db") as (_, owner, unused):
        objects = _OutsideScopeObjects()
        archive = await _archive(owner, objects)
        raw = b"".join(hashlib.sha256(str(index).encode()).digest() for index in range(1100))
        values = (("bytes", raw),)
        initial = PreparedArchive()
        await archive.prepare_content(values, initial)
        await _publish(archive, initial)
        prepared = PreparedArchive()
        observed = await _observe(owner, archive, values, prepared)
        record = next(iter(observed.values()))
        reference = ContentRecord.from_json(record.data).payload.ref
        assert reference is not None
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
        values = (("bytes", b"large" * 4000),)
        initial = PreparedArchive()
        await archive.prepare_content(values, initial)
        await _publish(archive, initial)
        prepared = PreparedArchive()
        observed = await _observe(owner, archive, values, prepared)
        await archive.prepare_content(values, prepared)
        assert prepared._verified_content
        combined = PreparedArchive.combine((prepared,))
        reference = ContentRecord.from_json(next(iter(observed.values())).data).payload.ref
        assert reference is not None
        assert await objects.delete_object(reference.key, expected_digest=reference.digest)
        with pytest.raises(AIError) as raised:
            await archive.prepare_content(values, combined)
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
        assert not combined._verified_content and not prepared._verified_content
