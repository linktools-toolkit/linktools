#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CI contracts for canonical native message ownership and commit-scoped reuse."""

import hashlib
import json
import re
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai import RequestUsage, messages as sdk
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.model.attachments import store_attachment_payload
from linktools.ai.model.messages import SummaryPromptPart, decode_model_messages, encode_model_messages
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence.contracts import ArchivedContextBlock
from linktools.ai.persistence._payload import decode_payload_bytes, encode_payload_bytes
from linktools.ai.persistence._step_message_content import (
    NativeMessageContent, NativeMessageReadback, NativeMessageSequence, PreparedMessagePayload,
)
from linktools.ai.persistence._transcript_staging import MessageArchive
from linktools.ai.storage import ObjectRef, SqlStorageContext, StoredPayload


_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _Transaction:
    def __init__(self, context: SqlStorageContext, session: AsyncSession) -> None:
        self.context, self.session = context, session

    async def run_sql(self, context, callback):
        if context is not self.context:
            raise ValueError("foreign SQL context")
        return await callback(self.session)


class _Traffic:
    def __init__(self) -> None:
        self.verbs: Counter[str] = Counter()
        self.submitted_rows: Counter[str] = Counter()
        self.bound_bytes = 0
        self.body_reads = 0

    def capture(self, connection, cursor, statement, parameters, context, many) -> None:
        verb = statement.split()[0].upper()
        self.verbs["SELECT" if verb == "WITH" else verb] += 1
        if verb in {"SELECT", "WITH"} and re.search(r"\bai_step_messages\.message\b", statement):
            self.body_reads += 1
        batches = parameters if many else (parameters,)
        self.bound_bytes += sum(len(value.encode("utf-8")) if isinstance(value, str) else len(value)
                                for batch in batches
                                for value in (batch.values() if isinstance(batch, dict) else batch)
                                if isinstance(value, (str, bytes)))
        for table in ("ai_step_messages", "ai_step_sequences"):
            if statement.startswith("INSERT INTO " + table):
                width = len(statement.partition("(")[2].partition(")")[0].split(","))
                self.submitted_rows[table] += sum(len(batch) // width for batch in batches)


async def _inline(raw: bytes) -> PreparedMessagePayload:
    compression, encoded = encode_payload_bytes(raw)
    return PreparedMessagePayload(raw, encoded=encoded, compression=compression,
                                  payload=None)


def _messages(*contents: str):
    return [sdk.ModelRequest(parts=[sdk.UserPromptPart(value, timestamp=_STAMP)], timestamp=_STAMP)
            for value in contents]


async def _publish(native, content, prepared, *, owner_inserted=True, confirmed_content=None):
    return await native.storage_context.run_mutation(lambda session: content.publish(
        _Transaction(native.storage_context, session), prepared,
        owner_inserted=owner_inserted, confirmed_content=confirmed_content,
    ))


@pytest.mark.asyncio
async def test_frozen_append_prepares_only_delta_and_keeps_bounded_knowledge(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'append.db'}")
    native = SqlStepStore(engine, namespace="messages", tenant_id="one")
    await native.initialize(local=True)
    content = NativeMessageContent(native.storage_context, group_id="execution", store_digest=native.store_digest,
                                   max_cached_messages=2, max_cached_bytes=4096, max_cached_nodes=4)
    archive, traffic, proof = MessageArchive(), _Traffic(), None
    calls = 0

    async def prepare(raw):
        nonlocal calls
        calls += 1
        return await _inline(raw)

    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        for index in range(40):
            prepared = await content.prepare_append(archive.snapshot(_messages(str(index))),
                                                    prepare_payload=prepare, base=proof)
            assert len(prepared.nodes) == 1
            assert len(prepared.messages) == 1
            assert len(prepared.frontier) <= (index + 2).bit_length()
            receipt = await _publish(native, content, prepared)
            with pytest.raises(ValueError, match="committed proof"):
                await content.prepare_append(archive.snapshot([]), prepare_payload=prepare, base=receipt)
            proof = content.confirm_committed(receipt)
        assert calls == 40
        assert traffic.body_reads == 0
        assert traffic.submitted_rows == {"ai_step_messages": 40, "ai_step_sequences": 40}
        assert traffic.bound_bytes > 0
        assert len(content._known_messages) <= 2 and content._cache_bytes <= 4096
        assert len(content._known_nodes) <= 4
        assert proof.root.count == 40
        content.invalidate()
        with pytest.raises(ValueError, match="committed proof"):
            await content.prepare_append(archive.snapshot([]), prepare_payload=prepare, base=proof)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_mutable_capture_checks_all_messages_but_frozen_delta_never_decodes(monkeypatch) -> None:
    import linktools.ai.persistence._transcript_staging as staging
    from sqlalchemy.dialects import sqlite
    from types import SimpleNamespace

    archive, calls = MessageArchive(), []
    original = staging.encode_model_messages

    def encode(messages):
        calls.append(len(messages))
        return original(messages)

    monkeypatch.setattr(staging, "encode_model_messages", encode)
    mutable = _messages("a", "b", "c")
    first = archive.snapshot(mutable)
    mutable[0].parts[0].content = "changed"
    changed = archive.snapshot(mutable)
    assert len(calls) == 6
    assert tuple(first.encoded_payloads()) != tuple(changed.encoded_payloads())
    assert archive.snapshot(first) is first
    assert len(calls) == 6
    owner = NativeMessageContent(SimpleNamespace(engine=SimpleNamespace(dialect=sqlite.dialect())),
                                 group_id="execution", store_digest="a" * 64)

    def forbid_decode(self):
        raise AssertionError("frozen prepare must not reconstruct SDK objects")

    monkeypatch.setattr(staging.MessageSnapshot, "__iter__", forbid_decode)
    prepared = await owner.prepare_append(first[2:], prepare_payload=_inline)
    assert len(prepared.nodes) == 1 and len(calls) == 6
    with pytest.raises(TypeError, match="frozen"):
        await owner.prepare_append(mutable, prepare_payload=_inline)


def test_encoded_snapshot_keeps_omitted_wire_fields_without_reencoding(monkeypatch) -> None:
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.persistence import _transcript_staging as staging

    wire = json.loads(encode_model_messages([sdk.ModelResponse(parts=[sdk.TextPart("answer")], timestamp=_STAMP)]))
    wire[0]["parts"][0].pop("id")
    payload = canonical_json_bytes(wire)

    def reject_encoding(*args, **kwargs):
        raise AssertionError("frozen wire bytes must not be reencoded through authoring values")

    monkeypatch.setattr(staging, "encode_model_messages", reject_encoding)
    snapshot = MessageArchive().snapshot_payloads((payload,))
    assert tuple(snapshot.encoded_payloads()) == (payload,)
    assert snapshot.encoded_digest() == (hashlib.sha256(payload).hexdigest(), len(payload))
    assert snapshot[0].parts[0].content == "answer"
    assert snapshot[0].parts[0].id is None
    for invalid in (b"[]", canonical_json_bytes(wire * 2)):
        with pytest.raises(ValueError, match="exactly one"):
            MessageArchive().snapshot_payloads((invalid,))


@pytest.mark.asyncio
async def test_cold_duplicates_are_idempotent_without_body_reads(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'duplicates.db'}")
    native = SqlStepStore(engine, namespace="messages", tenant_id="one")
    await native.initialize(local=True)
    traffic = _Traffic()
    event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
    try:
        frozen = MessageArchive().snapshot(_messages("好", "好", "好"))
        for _ in range(2):
            content = NativeMessageContent(native.storage_context, group_id="execution",
                                           store_digest=native.store_digest, max_cached_messages=0, max_cached_nodes=0)
            prepared = await content.prepare_append(frozen, prepare_payload=_inline)
            content.confirm_committed(await _publish(native, content, prepared))
        assert traffic.body_reads == 0
        assert traffic.submitted_rows["ai_step_messages"] == 2
        assert traffic.submitted_rows["ai_step_sequences"] == 6
        async with engine.connect() as connection:
            assert len((await connection.execute(select(content._messages))).all()) == 1
            assert len((await connection.execute(select(content._nodes))).all()) == 3
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_owner_conflicts_rollback_and_foreign_receipts_never_grant_candidate_roots(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'receipts.db'}")
    native = SqlStepStore(engine, namespace="messages", tenant_id="one")
    await native.initialize(local=True)
    content = NativeMessageContent(native.storage_context, group_id="execution", store_digest=native.store_digest)
    archive = MessageArchive()
    try:
        prepared = await content.prepare_append(archive.snapshot(_messages("candidate")), prepare_payload=_inline)
        conflict = await _publish(native, content, prepared, owner_inserted=False)
        assert conflict.root is None
        with pytest.raises(ValueError, match="committed root"):
            content.confirm_committed(conflict)
        traffic = _Traffic()
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)

        async def abort(session):
            await content.publish(_Transaction(native.storage_context, session), prepared, owner_inserted=True)
            raise RuntimeError("rollback")

        with pytest.raises(RuntimeError, match="rollback"):
            await native.storage_context.run_mutation(abort)
        assert not content._known_messages and not content._known_nodes
        receipt = await _publish(native, content, prepared)
        proof = content.confirm_committed(receipt)
        assert traffic.submitted_rows["ai_step_messages"] == 2
        replay = await _publish(native, content, prepared, owner_inserted=False, confirmed_content=proof)
        assert replay.root is None
        with pytest.raises(ValueError, match="committed root"):
            content.confirm_committed(replay)
        other = await content.prepare_append(archive.snapshot(_messages("different")), prepare_payload=_inline)
        with pytest.raises(ValueError, match="does not match"):
            await _publish(native, content, other, owner_inserted=False, confirmed_content=proof)
        foreign = NativeMessageContent(native.storage_context, group_id="other", store_digest=native.store_digest)
        with pytest.raises(ValueError, match="committed root"):
            foreign.confirm_committed(receipt)
        with pytest.raises(ValueError, match="prepared content"):
            await _publish(native, content, replace(prepared))
        event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)
    finally:
        await native.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_canonical_business_fields_roundtrip_with_binary_body_and_typed_attachment(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'codec.db'}")
    native = SqlStepStore(engine, namespace="messages", tenant_id="one")
    await native.initialize(local=True)
    content = NativeMessageContent(native.storage_context, group_id="execution", store_digest=native.store_digest)
    messages = [
        sdk.ModelRequest(parts=[
            SummaryPromptPart("trusted summary", timestamp=_STAMP),
            sdk.UserPromptPart([sdk.BinaryContent(b"\x00\xff", media_type="image/png")], timestamp=_STAMP),
            sdk.ToolReturnPart("read", {"ordinary": "value"}, tool_call_id="call", timestamp=_STAMP),
            sdk.RetryFeedbackPart("retry", retry=sdk.RetryInfo(cause="model_retry"), timestamp=_STAMP),
        ], timestamp=_STAMP),
        sdk.ModelResponse(parts=[sdk.TextPart("x" * 30_000)], timestamp=_STAMP,
                          usage=RequestUsage(input_tokens=3, provider_extra=7)),
    ]
    frozen = MessageArchive().snapshot(messages)
    objects, reads = {}, []

    async def write_object(data):
        digest = hashlib.sha256(data).hexdigest()
        reference = ObjectRef("runtime", "native-fixture/" + digest, digest, len(data))
        objects[reference.key] = data
        return reference

    async def object_payload(raw):
        payload = await store_attachment_payload(raw, format="model-messages", write_object=write_object)
        if payload is not None:
            return PreparedMessagePayload(raw, encoded=raw, compression="raw", payload=payload)
        return await _inline(raw)

    async def read(reference):
        reads.append(reference.key)
        return objects[reference.key]

    try:
        prepared = await content.prepare_append(frozen, prepare_payload=object_payload)
        proof = content.confirm_committed(await _publish(native, content, prepared))
        plan = content.range_query((proof.root,))
        async with engine.connect() as connection:
            rows = (await connection.execute(plan.statement)).mappings().all()
        encoded, = await content.hydrate_ranges((proof.root,), rows, read_object=read)
        assert encoded == tuple(frozen.encoded_payloads())
        restored = [decode_model_messages(raw)[0] for raw in encoded]
        assert encode_model_messages(restored) == encode_model_messages(messages)
        assert isinstance(restored[0].parts[0], SummaryPromptPart)
        media = restored[0].parts[1].content[0]
        original_media = messages[0].parts[1].content[0]
        assert (media.data, media.media_type, media.identifier) == (
            original_media.data, original_media.media_type, original_media.identifier,
        )
        assert restored[0].parts[2:] == messages[0].parts[2:]
        assert restored[1] == messages[1]
        assert len(reads) == 1 and list(objects.values()) == [b"\x00\xff"]
        assert any(isinstance(row["message"], bytes) and row["compression"] == "zlib" for row in rows)
        with pytest.raises(ValueError, match="resolver"):
            await content.hydrate_ranges((proof.root,), rows)
    finally:
        await native.close()
        await engine.dispose()


def test_internal_sequences_do_not_invent_group_scope_or_portable_authority() -> None:
    from sqlalchemy.dialects import sqlite
    from types import SimpleNamespace
    from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice

    context = SimpleNamespace(engine=SimpleNamespace(dialect=sqlite.dialect()))
    content = NativeMessageContent(context, store_digest="a" * 64)
    sequence = NativeMessageSequence((NativeMessageSlice("b" * 64, 3, 0, 3),))
    assert NativeMessageSequence.from_json(sequence.to_json()) == sequence
    assert sequence.slice(1, 3).count == 2
    assert content.empty_root() == NativeMessageSequence()
    assert content.range_query((sequence,)) is not None
    with pytest.raises(ValueError, match="must be bound"):
        content.scoped_root(sequence)
    portable = NativeMessageRoot("actual-group", "a" * 64, sequence.slices)
    with pytest.raises(ValueError, match="scope"):
        content.range_query((portable,))
    content.bind_group("actual-group")
    assert content.scoped_root(sequence) == portable and portable.sequence == sequence
    content.bind_group("actual-group")
    with pytest.raises(ValueError, match="another group"):
        content.bind_group("invented-other-group")
    with pytest.raises(ValueError, match="validated readback"):
        content.adopt_read(NativeMessageReadback(sequence, (), (), 0))
    with pytest.raises(ValueError, match="wire"):
        NativeMessageSequence.from_json(portable.to_json())


@pytest.mark.asyncio
async def test_direct_session_publication_and_retained_cas_keep_outer_confirmation(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'retained.db'}")
    native = SqlStepStore(engine, namespace="messages", tenant_id="one")
    await native.initialize(local=True)
    content = NativeMessageContent(native.storage_context, store_digest=native.store_digest)
    prepared = await content.prepare_append(MessageArchive().snapshot(_messages("body")), prepare_payload=_inline)
    try:
        async with native.storage_context.sessions() as session:
            with pytest.raises(ValueError, match="active same-engine"):
                await content.publish_in_session(session, prepared, owner_inserted=True)
        rejected = await native.storage_context.run_mutation(lambda session: content.publish_retained(
            _Transaction(native.storage_context, session), prepared, owner_written=False,
        ))
        assert rejected.root is None
        with pytest.raises(ValueError, match="committed root"):
            content.confirm_committed(rejected)
        receipt = await native.storage_context.run_mutation(lambda session: content.publish_in_session(
            session, prepared, owner_inserted=True,
        ))
        proof = content.confirm_committed(receipt)
        assert proof.root == prepared.root
        retained = await native.storage_context.run_mutation(lambda session: content.publish_retained(
            _Transaction(native.storage_context, session), prepared, owner_written=True, confirmed_content=proof,
        ))
        assert content.confirm_committed(retained).root == prepared.root
        foreign_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with AsyncSession(foreign_engine) as session:
                async with session.begin():
                    with pytest.raises(ValueError, match="same-engine"):
                        await content.publish_in_session(session, prepared, owner_inserted=True)
        finally:
            await foreign_engine.dispose()
    finally:
        await native.close()
        await engine.dispose()


async def _payload_owner(state, objects):
    from linktools.ai.persistence._archive import ContentArchive

    opened = await state.storage_group.open_archive(create=True)
    assert opened is not None
    group_id, records = opened
    archive = ContentArchive(records, group_id, namespace="payloads", tenant_id="one", object_store=objects)
    native = SqlStepStore(state.context, namespace="payloads", tenant_id="one")
    native.bind_message_archive(archive)
    return native, archive


@pytest.mark.asyncio
@pytest.mark.parametrize("damaged", ("messages", "nodes"))
async def test_native_winners_are_exact_and_mismatch_poisons_owner_mutation(
    tmp_path: Path, damaged: str,
) -> None:
    from sqlalchemy import update
    from linktools.ai.persistence._archive import PreparedArchive
    from .test_state_group_record_batch import _database, _record

    async with _database(tmp_path / "native-winners.db") as (engine, state, _other):
        native, archive = await _payload_owner(state, None)
        await native.initialize(local=True)
        messages = _messages("one", "two")
        bundle = PreparedArchive()
        root = await native.prepare_archived_messages(messages, bundle)

        async def publish_owner(transaction):
            from linktools.ai.persistence.codec import encode_domain
            await archive.publish(transaction, bundle)
            await transaction.insert_record(replace(_record("retained-root"), data={"context": encode_domain(ArchivedContextBlock(
                root, hashlib.sha256(encode_model_messages(messages)).hexdigest(), len(encode_model_messages(messages)),
            ))}))

        await archive.store.mutate(publish_owner)
        tables = (native._message_content._messages, native._message_content._nodes)

        async def rows():
            async with engine.connect() as connection:
                return tuple([tuple((await connection.execute(select(table).order_by(table.c.id))).mappings().all())
                              for table in tables])

        original = await rows()
        assert all(original)
        cold, _ = await _payload_owner(state, None)
        replay = PreparedArchive()
        await cold.prepare_archived_messages(messages, replay)
        await archive.store.mutate(lambda transaction: archive.publish(transaction, replay))
        replayed = await rows()
        # Ownership refreshes may advance GC fencing without changing immutable content.
        assert tuple(tuple({key: value for key, value in row.items() if key != "gc_revision"}
                           for row in group) for group in replayed) == tuple(
            tuple({key: value for key, value in row.items() if key != "gc_revision"}
                  for row in group) for group in original
        )
        table = tables[0 if damaged == "messages" else 1]
        async with engine.begin() as connection:
            for row in original[0 if damaged == "messages" else 1]:
                if damaged == "messages":
                    values = {"raw_size": row["raw_size"] + 1}
                else:
                    values = {"tree_count": row["tree_count"] + 1}
                await connection.execute(update(table).where(table.c.id == row["id"]).values(**values))
        corrupted = await rows()
        candidate, _ = await _payload_owner(state, None)
        retry = PreparedArchive()
        await candidate.prepare_archived_messages([*messages, *_messages("new")], retry)

        async def swallow(transaction):
            try:
                await archive.publish(transaction, retry)
            except AIError as error:
                assert error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            else:
                pytest.fail("corrupt native winner was accepted")
            await transaction.insert_record(_record("must-roll-back"))

        with pytest.raises(AIError) as caught:
            await archive.store.mutate(swallow)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await rows() == corrupted
        assert await archive.store.read_record(_record("must-roll-back").key_digest) is None
        assert not candidate._message_proofs and not candidate._message_content._known_messages


@pytest.mark.asyncio
async def test_archived_message_rollback_revokes_receipts_but_preserves_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence._archive import PreparedArchive
    from .test_state_group_record_batch import _database, _record
    from linktools.ai.persistence.codec import encode_domain

    async with _database(tmp_path / "archive-retry.db") as (_engine, state, _other):
        native, archive = await _payload_owner(state, None)
        await native.initialize(local=True)
        bundle, receipts = PreparedArchive(), []
        messages = _messages("retry")
        root = await native.prepare_archived_messages(messages, bundle)
        content = native._message_content
        publish = content.publish_retained

        async def capture(*args, **kwargs):
            receipt = await publish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setattr(content, "publish_retained", capture)
        marker = AIError(ErrorCode.STORAGE_UNAVAILABLE)

        async def publish_owner(transaction):
            await archive.publish(transaction, bundle)
            await transaction.insert_record(replace(_record("retained-root"), data={"context": encode_domain(ArchivedContextBlock(
                root, hashlib.sha256(encode_model_messages(messages)).hexdigest(), len(encode_model_messages(messages)),
            ))}))

        async def rollback(transaction):
            await publish_owner(transaction)
            raise marker

        with pytest.raises(AIError) as caught:
            await archive.store.mutate(rollback)
        assert caught.value is marker
        assert not content._known_messages and not native._message_proofs
        assert len(receipts) == 1
        with pytest.raises(ValueError, match="committed root"):
            content.confirm_committed(receipts[0])
        await archive.store.mutate(publish_owner)
        assert root.sequence in native._message_proofs
        assert len(receipts) == 2
        assert await native.read_message_roots((root,)) == (
            tuple(MessageArchive().snapshot(_messages("retry")).encoded_payloads()),
        )
        native.invalidate_message_content()
        with pytest.raises(AIError) as stale:
            await archive.store.mutate(lambda transaction: archive.publish(transaction, bundle))
        assert isinstance(stale.value.__cause__, ValueError)
        assert "prepared content" in str(stale.value.__cause__)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", (32, 65536))
async def test_prepared_payload_reuse_is_bounded_by_batch_owner_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int,
) -> None:
    from linktools.ai.persistence import _archive
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.storage import InMemoryObjectStore
    from .test_state_group_record_batch import _database

    objects, puts = InMemoryObjectStore(), []
    original_put = objects.put

    async def put(key, chunks, **kwargs):
        puts.append(key)
        return await original_put(key, chunks, **kwargs)

    monkeypatch.setattr(objects, "put", put)
    async with _database(tmp_path / "batch.db") as (engine, state, _other):
        native, archive = await _payload_owner(state, objects)
        bundle, traffic = PreparedArchive(), _Traffic()
        def media(value):
            return (sdk.ModelRequest(parts=[sdk.UserPromptPart([
                sdk.BinaryContent(value.encode(), media_type="image/png"),
            ], timestamp=_STAMP)]),)
        messages = media("x" * size)
        event.listen(engine.sync_engine, "before_cursor_execute", traffic.capture)
        try:
            first = await native.prepare_archived_messages(messages, bundle)
            second = await native.prepare_archived_messages(messages, bundle)
            assert first == second and len(puts) == 1
            repeated = await native.prepare_archived_messages(messages * 2, bundle)
            assert repeated.count == 2 and len(puts) == 1
            prepared = bundle.native_messages[repeated.sequence]
            assert len(prepared.messages) == 1
            identity, content = prepared.messages[0]
            raw = next(MessageArchive().snapshot(messages).encoded_payloads())
            assert identity == content.raw_digest == hashlib.sha256(raw).hexdigest()
            assert content.raw_size == len(raw)
            assert content.payload.encoding == "json-attachments"
            from linktools.ai.model.attachments import read_attachment_payload
            assert await read_attachment_payload(content.payload, read_object=archive.read_payload_object) == raw
            await native.prepare_archived_messages(media("y" * size), bundle)
            assert len(puts) == 2 and puts[0] != puts[1]
            await native.prepare_archived_messages(messages, PreparedArchive.combine((bundle,)))
            assert len(puts) == 3
            foreign = SqlStepStore(state.context, namespace="payloads", tenant_id="one")
            foreign.bind_message_archive(archive)
            await foreign.prepare_archived_messages(messages, bundle)
            assert len(puts) == 4
            native.invalidate_message_content()
            await native.prepare_archived_messages(messages, bundle)
            assert len(puts) == 5
            assert not native._message_proofs and not native._message_content._known_messages
            assert traffic.verbs == {}
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", traffic.capture)


@pytest.mark.asyncio
async def test_prepared_payload_reuse_preserves_current_database_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence import _archive
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.storage import InMemoryObjectStore
    from .test_state_group_record_batch import _database

    objects = InMemoryObjectStore()
    async with _database(tmp_path / "encoding.db") as (_engine, state, _other):
        native, _content = await _payload_owner(state, objects)
        bundle = PreparedArchive()
        messages = _messages("compressible " * 6000)
        root = await native.prepare_archived_messages(messages, bundle)
        first = bundle.native_messages[root.sequence].messages[0][1]
        assert first.compression == "zlib" and first.payload is None

        def uncompressed(raw):
            return "raw", raw

        monkeypatch.setattr(_archive, "encode_payload_bytes", uncompressed)
        bundle = PreparedArchive()
        root = await native.prepare_archived_messages(messages, bundle)
        second = bundle.native_messages[root.sequence].messages[0][1]
        assert second.compression == "raw" and second.payload is None
        assert first.raw_digest == second.raw_digest and first.raw_size == second.raw_size
        assert first.body != second.body
        assert [value async for value in objects.list_objects()] == []


@pytest.mark.asyncio
async def test_failed_preparation_does_not_retain_partial_payload_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence import _archive
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.storage import InMemoryObjectStore
    from .test_state_group_record_batch import _database

    objects, puts = InMemoryObjectStore(), []
    original_put = objects.put

    async def put(key, chunks, **kwargs):
        puts.append(key)
        if len(puts) == 2:
            raise RuntimeError("interrupted payload preparation")
        return await original_put(key, chunks, **kwargs)

    monkeypatch.setattr(objects, "put", put)
    async with _database(tmp_path / "failure.db") as (_engine, state, _other):
        native, _content = await _payload_owner(state, objects)
        bundle = PreparedArchive()
        messages = tuple(sdk.ModelRequest(parts=[sdk.UserPromptPart([
            sdk.BinaryContent(value, media_type="image/png"),
        ], timestamp=_STAMP)]) for value in (b"first", b"second"))
        with pytest.raises(RuntimeError, match="interrupted payload preparation"):
            await native.prepare_archived_messages(messages, bundle)
        assert not bundle.native_messages and not bundle.native_payloads
        assert not native._message_proofs and not native._message_content._known_messages
        await native.prepare_archived_messages(messages, bundle)
        assert len(puts) == 4 and puts[:2] == puts[2:]
