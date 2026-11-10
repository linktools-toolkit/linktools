#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native message reuse follows the actual enclosing owner commit outcome."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
from linktools.ai.persistence._message_root import NativeMessageRoot
from linktools.ai.persistence._transcript_staging import MessageArchive

from .test_state_group_record_batch import _database, _record


async def _bound(state):
    opened = await state.storage_group.open_archive(create=True)
    assert opened is not None
    group_id, records = opened
    archive = ContentArchive(records, group_id, namespace="native", tenant_id="one", object_store=None)
    native = SqlStepStore(state.context, namespace="native", tenant_id="one")
    native.bind_message_archive(archive)
    return native, archive


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("commit", "rollback", "cancel", "unknown"))
async def test_nested_owner_receipts_wait_for_outer_commit(tmp_path: Path, monkeypatch, outcome: str) -> None:
    async with _database(tmp_path / "outcomes.db") as (_engine, state, _other):
        native, archive = await _bound(state)
        messages = [ModelRequest(parts=[UserPromptPart("owner body")])]
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages(messages, prepared)
        owner = replace(_record("owner"), data={"root": root.to_json()})
        actual_mutation = type(state.context).run_mutation

        if outcome == "unknown":
            async def lose_ack(self, callback, **kwargs):
                await actual_mutation(self, callback, **kwargs)
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)
            monkeypatch.setattr(type(state.context), "run_mutation", lose_ack)

        async def write(transaction):
            await transaction.insert_record(owner)
            await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
            # A nested mutate result is not proof that its outer owner committed.
            assert not native._message_proofs
            if outcome == "rollback":
                raise ValueError("reject owner tail")
            if outcome == "cancel":
                raise asyncio.CancelledError

        if outcome == "commit":
            await state.mutate(write)
            assert native._message_proofs
        else:
            with pytest.raises(asyncio.CancelledError if outcome == "cancel" else AIError):
                await state.mutate(write)
            assert not native._message_proofs
            assert not native._message_inputs
        monkeypatch.setattr(type(state.context), "run_mutation", actual_mutation)
        assert (await state.read(lambda tx: tx.get_record(owner.key_digest)) is not None) == (outcome in {"commit", "unknown"})
        if outcome == "unknown":
            assert len((await native.read_message_roots((root,)))[0]) == 1
        if outcome in {"rollback", "cancel"}:
            retry = PreparedArchive()
            retry_root = await native.prepare_archived_messages(messages, retry)
            async def retry_owner(transaction):
                await transaction.insert_record(owner)
                await archive.store.mutate(lambda tx: archive.publish(tx, retry))
            await state.mutate(retry_owner)
            assert len((await native.read_message_roots((retry_root,)))[0]) == 1


@pytest.mark.asyncio
async def test_expired_receipt_cannot_turn_a_committed_owner_into_failure(tmp_path: Path) -> None:
    async with _database(tmp_path / "expired.db") as (_engine, state, _other):
        native, archive = await _bound(state)
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages([ModelRequest(parts=[UserPromptPart("body")])], prepared)
        owner = replace(_record("owner"), data={"root": root.to_json()})
        async def write(transaction):
            await transaction.insert_record(owner)
            await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
            # Another failed publication or offline epoch can expire a cache receipt.
            native.invalidate_message_content()
        await state.mutate(write)
        assert not native._message_proofs
        assert await state.read(lambda tx: tx.get_record(owner.key_digest)) == owner
        assert len((await native.read_message_roots((root,)))[0]) == 1


@pytest.mark.asyncio
async def test_foreign_baseline_rejected_even_when_its_content_matches_local_proof(tmp_path: Path) -> None:
    async with _database(tmp_path / "scope.db") as (_engine, state, _other):
        native, archive = await _bound(state)
        messages = MessageArchive().snapshot([ModelRequest(parts=[UserPromptPart("body")])])
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages(messages, prepared)
        async def write(transaction):
            await transaction.insert_record(replace(_record("owner"), data={"root": root.to_json()}))
            await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
        await state.mutate(write)
        digest, size = messages.encoded_digest()
        foreign = NativeMessageRoot("another-group", root.store_digest, root.slices)
        with pytest.raises(ValueError, match="scope"):
            await native.prepare_message_content(messages, baseline=(foreign, digest, size))


@pytest.mark.asyncio
async def test_failed_owner_cas_cannot_confirm_candidate_and_retry_can_publish(tmp_path: Path) -> None:
    async with _database(tmp_path / "cas.db") as (_engine, state, _other):
        native, _archive = await _bound(state)
        original = _record("owner")
        await state.mutate(lambda tx: tx.insert_record(original))
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages([ModelRequest(parts=[UserPromptPart("candidate")])], prepared)

        async def replace_owner(transaction, expected):
            updated = replace(original, storage_revision=expected + 1, data={"root": root.to_json()})
            if not await transaction.replace_record(updated, expected_storage_revision=expected):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            await native.publish_archived_messages(transaction, prepared)

        with pytest.raises(AIError) as caught:
            await state.mutate(lambda tx: replace_owner(tx, 1))
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert not native._message_proofs
        await state.mutate(lambda tx: replace_owner(tx, 0))
        assert native._message_proofs
        assert len((await native.read_message_roots((root,)))[0]) == 1


@pytest.mark.asyncio
async def test_concurrent_rollback_does_not_change_other_owners_confirmed_commit(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.persistence._sql import SqlStateStore

    async with _database(tmp_path / "concurrent.db") as (engine, state, _other):
        native, _archive = await _bound(state)
        concurrent = SqlStateStore(engine, context=state.context, runtime_domain="concurrent")
        await concurrent.initialize()
        success = PreparedArchive()
        failure = PreparedArchive()
        root = await native.prepare_archived_messages([ModelRequest(parts=[UserPromptPart("committed")])], success)
        await native.prepare_archived_messages([ModelRequest(parts=[UserPromptPart("rolled back")])], failure)
        actual_mutation = type(state.context).run_mutation
        committed, resume = asyncio.Event(), asyncio.Event()
        successful_task = None

        async def delay_success(self, callback, **kwargs):
            result = await actual_mutation(self, callback, **kwargs)
            if asyncio.current_task() is successful_task:
                committed.set()
                await resume.wait()
            return result

        monkeypatch.setattr(type(state.context), "run_mutation", delay_success)

        async def success_owner(transaction):
            await transaction.insert_record(_record("success"))
            await native.publish_archived_messages(transaction, success)

        async def failed_owner(transaction):
            await transaction.insert_record(_record("failure"))
            await native.publish_archived_messages(transaction, failure)
            raise ValueError("rollback after publication")

        successful_task = asyncio.create_task(state.mutate(success_owner))
        try:
            await asyncio.wait_for(committed.wait(), 5)
            with pytest.raises(AIError):
                await concurrent.mutate(failed_owner)
            resume.set()
            await asyncio.wait_for(successful_task, 5)
            assert await state.read(lambda tx: tx.get_record(_record("success").key_digest)) is not None
            assert await concurrent.read(lambda tx: tx.get_record(_record("failure").key_digest)) is None
            assert len((await native.read_message_roots((root,)))[0]) == 1
        finally:
            resume.set()
            await asyncio.gather(successful_task, return_exceptions=True)
            monkeypatch.setattr(type(state.context), "run_mutation", actual_mutation)
            await concurrent.close()


@pytest.mark.asyncio
async def test_retryable_abort_reuses_frozen_preparation_inside_original_mutation(tmp_path: Path, monkeypatch) -> None:
    from linktools.ai.storage import SqlTransactionDisposition

    class RetryBody(Exception):
        pass

    async with _database(tmp_path / "retry.db") as (_engine, state, _other):
        native, _archive = await _bound(state)
        prepared = PreparedArchive()
        root = await native.prepare_archived_messages([ModelRequest(parts=[UserPromptPart("retry body")])], prepared)
        dialect = type(state.context.dialect)
        original_classifier = dialect.classify_transaction_error
        def classify(self, error, *, phase, connection_invalidated):
            if isinstance(error, RetryBody):
                return SqlTransactionDisposition.RETRYABLE_ABORTED
            return original_classifier(self, error, phase=phase, connection_invalidated=connection_invalidated)
        monkeypatch.setattr(dialect, "classify_transaction_error", classify)
        attempts = 0
        async def write(transaction):
            nonlocal attempts
            attempts += 1
            await transaction.insert_record(replace(_record("owner"), data={"root": root.to_json()}))
            await native.publish_archived_messages(transaction, prepared)
            assert not native._message_proofs
            if attempts == 1:
                raise RetryBody
        await state.mutate(write)
        assert attempts == 2
        assert native._message_proofs
        assert len((await native.read_message_roots((root,)))[0]) == 1


@pytest.mark.asyncio
async def test_archive_payload_rejects_foreign_group_before_object_io(tmp_path: Path) -> None:
    from linktools.ai.storage import ObjectRef

    async with _database(tmp_path / "object-scope.db") as (_engine, state, _other):
        _native, archive = await _bound(state)
        with pytest.raises(AIError) as failure:
            await archive.read_payload_object(ObjectRef("runtime", "v1/archive/foreign/" + "0" * 64, "0" * 64, 0))
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_bound_snapshot_media_and_text_metadata_roundtrip_through_payload_owner(tmp_path: Path) -> None:
    from linktools.ai.persistence._archive import ContentRecord
    from pydantic_ai.messages import BinaryContent, TextContent
    from pydantic_ai_harness.step_persistence import ContinuableSnapshot
    from sqlalchemy import text
    from linktools.ai.model.messages import encode_model_messages
    from linktools.ai.persistence import Persistence, PersistenceStorage

    messages = [ModelRequest(parts=[UserPromptPart([
        BinaryContent(data=b"binary-image" * 10000, media_type="image/png"),
        TextContent("caption", metadata={"application": {"kind": "binary", "literal": True}}),
    ])], metadata={"owner": "application"})]
    async with Persistence.open("media", storage=PersistenceStorage.from_root(tmp_path / "media"), tenant_id="one") as persistence:
        writer = await persistence.create_step_store()
        reader = await persistence.create_step_store()
        try:
            snapshot = ContinuableSnapshot(run_id="media", step_index=1, messages=messages, idempotency_key="one")
            await writer.save_snapshot(snapshot)
            restored = await reader.latest_snapshot(run_id="media")
            assert encode_model_messages(restored.messages) == encode_model_messages(messages)
            async with writer.storage_context.sessions() as session:
                row = (await session.execute(text("SELECT * FROM ai_step_messages"))).mappings().one()
            content = ContentRecord.from_native_row(row)
            assert content.body is None and content.payload.encoding == "json-attachments"
            reference, = content.object_refs()
            assert reference.size == len(messages[0].parts[0].content[0].data)
        finally:
            await reader.close()
            await writer.close()
