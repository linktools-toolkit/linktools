#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral bounded binary ordering and cursor contracts."""

import base64
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._repository_common import decode_record_cursor, record_cursor
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._sql import SqlStateStore, _record_from_row, _record_values
from linktools.ai.persistence.codec import decode_record, encode_record
from linktools.ai.persistence.state import SORT_KEY_MAX_BYTES, RecordQuery, StateStore, StoredRecord

_SCOPE = b"s" * 32


def _record(sort_key: bytes) -> StoredRecord:
    return StoredRecord(hashlib.sha256(sort_key).digest(), _SCOPE, None, "probe", sort_key,
                        None, 0, None, 0, None, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_binary_sort_prefix_and_pagination_match_byte_order(tmp_path: Path, backend: str) -> None:
    engine = None
    store: StateStore
    if backend == "memory":
        store = InMemoryStateStore()
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'range.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine)
    await store.initialize()
    try:
        sorts = (b"A-one", b"A-two", b"a-one", b"\x00", b"\x00\xff", b"%_",
                 b"\x7f", b"\x80", b"\xff", b"\xff\x00", b"\xff\xff",
                 b"x" * SORT_KEY_MAX_BYTES)
        records = tuple(_record(value) for value in sorts)
        tied = replace(_record(b"A-one"), key_digest=b"\xff" * 32)
        await store.mutate(lambda tx: tx.insert_records((*records, tied)))
        ordered = tuple(sorted((*records, tied), key=lambda record: (record.sort_key, record.key_digest)))
        for prefix in (None, b"A", b"a", b"\x00", b"%_", b"\xff", b"\xff\xff", b"x" * SORT_KEY_MAX_BYTES):
            found = []
            cursor = None
            while True:
                sort, key = decode_record_cursor(cursor)
                page = await store.read(lambda tx: tx.list_records(RecordQuery(
                    kind="probe", scope_digest=_SCOPE, sort_key_prefix=prefix,
                    after_sort_key=sort, after_key_digest=key, limit=2,
                )))
                if not page:
                    break
                found.extend(page)
                assert len(found) <= len(ordered)
                cursor = record_cursor(page[-1])
            assert tuple(found) == tuple(record for record in ordered
                                        if prefix is None or record.sort_key.startswith(prefix))
        exclusive = await store.read(lambda tx: tx.list_records(RecordQuery(
            kind="probe", after_sort_key=b"A-one",
        )))
        assert exclusive == tuple(record for record in ordered if record.sort_key > b"A-one")
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()


def test_sort_columns_use_binary_types_for_supported_backends() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.EXECUTION}))
    column_type = metadata.tables["ai_state_records"].c.sort_key.type
    assert column_type.compile(dialect=postgresql.dialect()) == "BYTEA"
    assert column_type.compile(dialect=sqlite.dialect()) == "BLOB"
    assert column_type.compile(dialect=mysql.dialect()) == f"VARBINARY({SORT_KEY_MAX_BYTES})"


@pytest.mark.parametrize("sort", (b"\x00", bytes(range(256)), b"\xff" * SORT_KEY_MAX_BYTES))
def test_binary_sorts_roundtrip_through_current_wire_rows_and_cursors(sort: bytes) -> None:
    record = _record(sort)
    assert decode_record(encode_record(record)) == record
    assert _record_from_row(_record_values(record)) == record
    assert decode_record_cursor(record_cursor(record)) == (sort, record.key_digest)


@pytest.mark.parametrize("sort", (b"", b"x" * (SORT_KEY_MAX_BYTES + 1), "text", bytearray(b"x")))
def test_sort_producers_queries_and_readers_reject_unbounded_or_nonbinary_values(sort: object) -> None:
    record = _record(b"valid")
    with pytest.raises(ValueError):
        replace(record, sort_key=sort)
    for field in ("sort_key_prefix", "after_sort_key"):
        with pytest.raises(ValueError):
            RecordQuery(kind="probe", **{field: sort})
    with pytest.raises(AIError) as error:
        _record_from_row({**_record_values(record), "sort_key": sort})
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    encoded = (base64.b64encode(sort).decode("ascii") if isinstance(sort, bytes)
               else "invalid Base64 sort" if isinstance(sort, str) else sort)
    with pytest.raises(AIError) as error:
        decode_record({**encode_record(record), "sort": encoded})
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("corruption", ("truncated", "trailing", "sort-length", "key-length", "oversized", "alphabet"))
def test_record_cursors_reject_malformed_binary_framing(corruption: str) -> None:
    record = _record(b"\xff" * SORT_KEY_MAX_BYTES)
    cursor = record_cursor(record)
    raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
    if corruption == "truncated":
        raw = raw[:-1]
    elif corruption == "trailing":
        raw += b"\x00"
    elif corruption == "sort-length":
        raw = b"\x00\x00" + raw[2:]
    elif corruption == "key-length":
        raw = raw[:2 + len(record.sort_key)] + b"\x00\x1f" + raw[4 + len(record.sort_key):]
    elif corruption == "oversized":
        raw = b"x" * (SORT_KEY_MAX_BYTES + 100)
    cursor = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if corruption == "alphabet":
        cursor = cursor.replace("_", "/")
    with pytest.raises(AIError) as error:
        decode_record_cursor(cursor)
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_raw_sort_and_cursor_fields_do_not_reveal_key_material_in_repr() -> None:
    secret = b"secret-identity-MUST-NOT-APPEAR"
    record = _record(secret)
    query = RecordQuery(kind="probe", sort_key_prefix=secret, after_sort_key=secret,
                        after_key_digest=record.key_digest)
    assert secret.decode("ascii") not in repr(record)
    assert secret.decode("ascii") not in repr(query)


@pytest.mark.parametrize("encoded", ("eB==", "eA", "eA====", "x" * 1600))
def test_record_wire_rejects_noncanonical_or_oversized_sort_encoding(encoded: str) -> None:
    with pytest.raises(AIError) as error:
        decode_record({**encode_record(_record(b"x")), "sort": encoded})
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
