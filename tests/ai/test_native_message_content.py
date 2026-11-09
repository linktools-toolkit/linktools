#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CI contracts for canonical native message ownership and commit-scoped reuse."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai import RequestUsage, messages as sdk
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from linktools.ai.model.messages import SummaryPromptPart, decode_model_messages
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._payload import encode_payload_bytes
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

    def capture(self, connection, cursor, statement, parameters, context, many) -> None:
        verb = statement.split()[0].upper()
        self.verbs["SELECT" if verb == "WITH" else verb] += 1
        batches = parameters if many else (parameters,)
        self.bound_bytes += sum(len(value.encode("utf-8")) for batch in batches
                                for value in (batch.values() if isinstance(batch, dict) else batch)
                                if isinstance(value, str))
        for table in ("ai_step_messages", "ai_step_sequences"):
            if statement.startswith("INSERT INTO " + table):
                self.submitted_rows[table] += len(batches)


async def _inline(raw: bytes) -> PreparedMessagePayload:
    compression, _, _, encoded = encode_payload_bytes(raw)
    return PreparedMessagePayload(raw, encoded=encoded, compression=compression,
                                  payload=StoredPayload.inline_bytes(encoded))


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
        assert traffic.verbs["SELECT"] == traffic.verbs["UPDATE"] == 0
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


@pytest.mark.asyncio
async def test_cold_duplicates_are_idempotent_without_reads_or_updates(tmp_path: Path) -> None:
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
        assert traffic.verbs["SELECT"] == traffic.verbs["UPDATE"] == 0
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
@pytest.mark.parametrize("representation", ["object", "json", "text", "bytes"])
async def test_canonical_business_fields_and_existing_object_compression_roundtrip(tmp_path: Path, representation: str) -> None:
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

    async def object_payload(raw):
        if representation == "json":
            return PreparedMessagePayload(raw, encoded=raw, compression="raw",
                                          payload=StoredPayload.inline_json(json.loads(raw)))
        if representation == "text":
            return PreparedMessagePayload(raw, encoded=raw, compression="raw",
                                          payload=StoredPayload.inline_text(raw.decode("utf-8")))
        if representation == "bytes":
            return await _inline(raw)
        compression, _, _, encoded = encode_payload_bytes(raw)
        digest = hashlib.sha256(encoded).hexdigest()
        reference = ObjectRef("runtime", "native-fixture/" + digest, digest, len(encoded))
        objects[reference.key] = encoded
        return PreparedMessagePayload(raw, encoded=encoded, compression=compression, payload=StoredPayload.object(reference))

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
        assert restored == messages
        assert len(reads) == (2 if representation == "object" else 0)
        if representation == "object":
            assert any(len(value) < 1000 for value in objects.values())
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
