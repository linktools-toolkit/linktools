#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Database content keeps binary bodies separate from queryable state metadata."""

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import select
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._archive import ContentArchive, ContentRecord, PreparedArchive
from linktools.ai.persistence._payload import decode_payload_bytes, encode_payload_bytes
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._snapshot import decode_native_message_row, encode_native_message_row
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence._step_message_content import PreparedMessagePayload
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence._transcript_staging import MessageArchive
from linktools.ai.persistence.codec import decode_record, encode_record


@pytest.mark.parametrize("dialect, expected", ((mysql.dialect(), "LONGBLOB"),
                                              (postgresql.dialect(), "BYTEA"),
                                              (sqlite.dialect(), "BLOB")))
def test_database_content_binary_columns_have_full_dialect_capacity(dialect, expected: str) -> None:
    records = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION})).tables["ai_state_records"]
    messages = build_step_sql_metadata().tables["ai_step_messages"]
    assert records.c.payload_bytes.type.compile(dialect=dialect) == expected
    assert messages.c.message.type.compile(dialect=dialect) == expected
    assert records.c.payload_json.type.compile(dialect=dialect) == "JSON"


@pytest.mark.asyncio
@pytest.mark.parametrize("body", (b"short text", b"compressible text\x00" * 8192,
                                  random.Random(31).randbytes(128 * 1024)))
async def test_archive_body_is_binary_in_database_and_round_trips_portable_codec(tmp_path: Path, body: bytes) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'content.db'}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    archive = ContentArchive(store, "binary", namespace="test", tenant_id="tenant", object_store=None)
    prepared = PreparedArchive()
    try:
        identities = await archive.prepare_content((("text", body),), prepared)
        record, = prepared.records.values()
        codec, encoded = encode_payload_bytes(body)
        digest, size = hashlib.sha256(body).hexdigest(), len(body)
        assert record.payload_bytes == encoded
        assert record.data == {"codec": "text", "compression": codec, "raw_digest": digest,
                               "raw_size": size, "payload": None}
        assert decode_record(encode_record(record)) == record
        await store.mutate(lambda tx: archive.publish(tx, prepared))
        table = build_runtime_sql_metadata(frozenset({PersistenceDomain.CONVERSATION})).tables["ai_state_records"]
        async with engine.connect() as connection:
            physical = (await connection.execute(select(table).where(
                table.c.key_digest == record.key_digest.hex(),
            ))).mappings().one()
        assert type(physical["payload_bytes"]) is bytes
        assert physical["payload_bytes"] == encoded
        assert json.loads(physical["payload_json"]) == dict(record.data)
        cold = ContentArchive(store, "binary", namespace="test", tenant_id="tenant", object_store=None)
        assert await cold.read_contents(identities, codec="text") == (body,)
        altered = replace(record, payload_bytes=encoded + b"changed")
        with pytest.raises(AIError) as rejected:
            await store.mutate(lambda tx: tx.insert_immutable_records((altered,)))
        assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await cold.read_contents(identities, codec="text") == (body,)
    finally:
        await store.close()
        await engine.dispose()


@pytest.mark.parametrize("damage", ("truncated", "trailing", "size", "digest", "codec"))
def test_binary_content_uses_existing_source_integrity_and_bounded_decode(damage: str) -> None:
    raw = b"compressible content" * 8192
    codec, body = encode_payload_bytes(raw)
    digest, size = hashlib.sha256(raw).hexdigest(), len(raw)
    if damage == "truncated":
        body = body[:-1]
    elif damage == "trailing":
        body += b"unconsumed bytes"
    elif damage == "size":
        size -= 1
    elif damage == "digest":
        digest = "0" * 64
    else:
        codec = "unknown"
    with pytest.raises(AIError) as rejected:
        decode_payload_bytes(body, codec=codec, raw_digest=digest, raw_size=size)
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_native_body_stays_binary_through_publication_and_portable_boundary(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'native.db'}")
    store = SqlStepStore(engine, namespace="test", tenant_id="tenant")
    await store.initialize(local=True)
    native = store._message_content
    frozen = MessageArchive().snapshot((ModelResponse(parts=[TextPart("message" * 20000)]),))

    async def prepare(raw: bytes) -> PreparedMessagePayload:
        compression, encoded = encode_payload_bytes(raw)
        return PreparedMessagePayload(raw, encoded=encoded, compression=compression, payload=None)

    try:
        prepared = await native.prepare_append(frozen, prepare_payload=prepare)
        receipt = await store.storage_context.run_mutation(
            lambda session: native.publish_in_session(session, prepared, owner_inserted=True),
        )
        proof = native.confirm_committed(receipt)
        table = build_step_sql_metadata().tables["ai_step_messages"]
        async with engine.connect() as connection:
            physical = dict((await connection.execute(select(table))).mappings().one())
            rows = (await connection.execute(native.range_query((proof.root,)).statement)).mappings().all()
        assert type(physical["message"]) is bytes and physical["compression"] == "zlib"
        assert physical["payload_json"] is None
        raw, = frozen.encoded_payloads()
        assert physical["sha256"] == hashlib.sha256(raw).hexdigest()
        assert physical["raw_size"] == len(raw)
        assert await native.hydrate_ranges((proof.root,), rows) == ((raw,),)
        portable = encode_native_message_row(physical)
        assert isinstance(portable["message"], str)
        restored = ContentRecord.from_native_row(decode_native_message_row(portable))
        assert restored.body == physical["message"]
        assert decode_payload_bytes(restored.body, codec=restored.compression,
                                    raw_digest=restored.raw_digest, raw_size=restored.raw_size) == raw
    finally:
        await store.close()
        await engine.dispose()
