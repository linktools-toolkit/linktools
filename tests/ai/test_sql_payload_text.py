#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL payload reads preserve JSON numbers independently of driver decoding."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.schema import CreateTable
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._sql import (
    SqlStateStore, _fact_from_row, _fact_values, _operation_from_row,
    _operation_values, _record_from_row, _record_values,
)
from linktools.ai.persistence.state import (
    FactQuery, OperationQuery, RecordQuery, StoredFact, StoredOperation, StoredRecord,
)

from .test_evaluation_numeric_transport import _transport_json


DATA = {"numbers": [1.0, 1, True, -0.0, 0.0, 1.5, 1e20, 2**53 + 1,
                    float.fromhex("0x1.fffffffffffffp+1023")],
        "nested": {"$float": "ordinary user data", "values": [0, 0.0, -0.0]},
        "large_text": "数" * 25000}
RECORD = StoredRecord(b"r" * 32, None, None, "numeric", "one", None, 0, None, 0, None, DATA)
FACT = StoredFact(b"s" * 32, 1, RECORD.key_digest, "numeric", "subject", None, DATA)
OPERATION = StoredOperation(b"o" * 32, b"s" * 32, 1, "pending", False, DATA)


@pytest.mark.parametrize("value,encode,decode", (
    (RECORD, _record_values, _record_from_row),
    (FACT, _fact_values, _fact_from_row),
    (OPERATION, _operation_values, _operation_from_row),
))
def test_sql_payload_text_preserves_original_json_and_rejects_invalid_roots(value, encode, decode) -> None:
    row = encode(value)
    assert isinstance(row["payload_json"], str)
    assert row["payload_json"].encode("utf-8") == canonical_json_bytes(DATA)
    restored = decode(row)
    assert canonical_json_bytes(dict(restored.data)) == canonical_json_bytes(DATA)
    for invalid in ("{broken", "[]", "null", "1", '"text"', dict(DATA),
                    row["payload_json"].encode("utf-8")):
        with pytest.raises(AIError) as raised:
            decode({**row, "payload_json": invalid})
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("dialect,storage_type", ((mysql.dialect(), "JSON"),
                                                (postgresql.dialect(), "JSON"),
                                                (sqlite.dialect(), "JSON")))
@pytest.mark.parametrize("name", ("ai_state_records", "ai_state_facts", "ai_state_operations"))
def test_sql_payload_schema_uses_json_string_without_numeric_normalization(dialect, storage_type: str, name: str) -> None:
    table = build_runtime_sql_metadata(frozenset({PersistenceDomain.EVALUATION})).tables[name]
    payload = table.c.payload_json
    assert storage_type in str(payload.type.compile(dialect=dialect))
    assert f"payload_json {storage_type}" in str(CreateTable(table).compile(dialect=dialect))
    assert not payload.nullable


@pytest.mark.asyncio
async def test_sql_payloads_keep_numeric_identity_across_read_queries_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "payload.db"
    for reopened in (False, True):
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}", json_deserializer=_transport_json)
        if not reopened:
            await provision_database(engine)
        store = SqlStateStore(engine)
        await store.initialize()
        try:
            if not reopened:
                async def write(tx):
                    await tx.insert_record(RECORD)
                    await tx.insert_fact(FACT)
                    await tx.insert_operation(OPERATION)
                await store.mutate(write)
            async with engine.connect() as connection:
                for table_name in ("ai_state_records", "ai_state_facts", "ai_state_operations"):
                    raw = (await connection.execute(text(f"SELECT payload_json FROM {table_name}"))).scalar_one()
                    assert isinstance(raw, str)
                    assert len(raw.encode("utf-8")) > 65535
                    assert json.loads(raw).encode("utf-8") == canonical_json_bytes(DATA)
            reads = (
                lambda tx: tx.get_record(RECORD.key_digest),
                lambda tx: tx.get_records((RECORD.key_digest,)),
                lambda tx: tx.list_records(RecordQuery(kind="numeric")),
                lambda tx: tx.scan_records(),
                lambda tx: tx.scan_records_page(after=None, limit=2),
                lambda tx: tx.list_facts(FactQuery(FACT.stream_digest)),
                lambda tx: tx.list_facts(FactQuery(FACT.stream_digest, latest=True)),
                lambda tx: tx.list_facts(FactQuery(FACT.stream_digest, latest_per_subject=True)),
                lambda tx: tx.scan_facts(),
                lambda tx: tx.scan_facts_page(after=None, limit=2),
                lambda tx: tx.get_operation(OPERATION.key),
                lambda tx: tx.list_operations(OperationQuery()),
                lambda tx: tx.scan_operations(),
                lambda tx: tx.scan_operations_page(after=None, limit=2),
            )
            for read in reads:
                result = await store.read(read)
                values = tuple(result.values()) if isinstance(result, dict) else result if isinstance(result, tuple) else (result,)
                assert len(values) == 1
                assert canonical_json_bytes(dict(values[0].data)) == canonical_json_bytes(DATA)
            async def grouped(tx):
                result = await tx.get_records({store: (RECORD.key_digest,)})
                assert canonical_json_bytes(dict(result[store][RECORD.key_digest].data)) == canonical_json_bytes(DATA)
            await store.storage_group.read_many((store,), grouped)
            if not reopened:
                assert await store.mutate(lambda tx: tx.replace_record(
                    replace(RECORD, storage_revision=1), expected_storage_revision=0))
                assert not await store.mutate(lambda tx: tx.replace_record(
                    replace(RECORD, storage_revision=1), expected_storage_revision=0))
                assert await store.mutate(lambda tx: tx.replace_operation(
                    replace(OPERATION, state="complete"), expected_state="pending"))
            else:
                deleted = await store.mutate(lambda tx: tx.delete_operations(OperationQuery()))
                assert canonical_json_bytes(dict(deleted[0].data)) == canonical_json_bytes(DATA)
        finally:
            await store.close()
            await engine.dispose()
