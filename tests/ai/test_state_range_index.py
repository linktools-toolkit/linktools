#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral bounded binary ordering and cursor contracts."""

import base64
import hashlib
from collections.abc import Callable, Sequence
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
from linktools.ai.persistence.state import (
    RELATION_KEY_MAX_BYTES, PARENT_KEY_MAX_BYTES, SORT_KEY_MAX_BYTES, RecordQuery, StateStore, StoredRecord,
    parent_key, scope_key, validate_record_identity,
)

_SCOPE = b"s" * 32
_RelationAddress = Callable[[str, str, str, str, str, Sequence[str]], bytes]


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
        tied = replace(_record(b"A-one"), key=b"\xff" * 32)
        await store.mutate(lambda tx: tx.insert_records((*records, tied)))
        ordered = tuple(sorted((*records, tied), key=lambda record: (record.sort_key, record.key)))
        for prefix in (None, b"A", b"a", b"\x00", b"%_", b"\xff", b"\xff\xff", b"x" * SORT_KEY_MAX_BYTES):
            found = []
            cursor = None
            while True:
                sort, key = decode_record_cursor(cursor)
                page = await store.read(lambda tx: tx.list_records(RecordQuery(
                    kind="probe", scope_key=_SCOPE, sort_key_prefix=prefix,
                    after_sort_key=sort, after_key=key, limit=2,
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
    assert _record_from_row({**_record_values(record), "store_key": b"store"}) == record
    assert decode_record_cursor(record_cursor(record)) == (sort, record.key)


@pytest.mark.parametrize("sort", (b"", b"x" * (SORT_KEY_MAX_BYTES + 1), "text", bytearray(b"x")))
def test_sort_producers_queries_and_readers_reject_unbounded_or_nonbinary_values(sort: object) -> None:
    record = _record(b"valid")
    with pytest.raises(ValueError):
        replace(record, sort_key=sort)
    for field in ("sort_key_prefix", "after_sort_key"):
        with pytest.raises(ValueError):
            RecordQuery(kind="probe", **{field: sort})
    with pytest.raises(AIError) as error:
        _record_from_row({**_record_values(record), "store_key": b"store", "sort_key": sort})
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
                        after_key=record.key)
    assert secret.decode("ascii") not in repr(record)
    assert secret.decode("ascii") not in repr(query)


@pytest.mark.parametrize("encoded", ("eB==", "eA", "eA====", "x" * 1600))
def test_record_wire_rejects_noncanonical_or_oversized_sort_encoding(encoded: str) -> None:
    with pytest.raises(AIError) as error:
        decode_record({**encode_record(_record(b"x")), "sort": encoded})
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("address,role", ((scope_key, "scope"), (parent_key, "parent")))
def test_relation_addresses_preserve_exact_flat_global_identity(address: _RelationAddress, role: str) -> None:
    parts = ("namespace", "tenant", "execution", "record", "owner", "é", "e\u0301")
    framed = (role, *parts)
    expected = b"".join(len(part.encode("utf-8")).to_bytes(2, "big") + part.encode("utf-8") for part in framed)
    assert address(*parts[:5], parts[5:]) == expected
    identities = (("a:b", "c"), ("a", "b:c"), ("ab", "c"), ("a", "bc"), ("é",), ("e\u0301",))
    assert len({address(*parts[:5], identity) for identity in identities}) == len(identities)
    for index in range(5):
        other = list(parts[:5])
        other[index] += "-other"
        assert address(*other, parts[5:]) != expected
    for invalid in ("scalar", b"bytes", [["nested"]], {"owner": "mapping"}):
        with pytest.raises(TypeError):
            address(*parts[:5], invalid)
    with pytest.raises(ValueError):
        address(*parts[:5], ())


@pytest.mark.parametrize("address", (scope_key, parent_key))
def test_relation_address_enforces_the_complete_encoded_byte_budget(address: _RelationAddress) -> None:
    prefix = ("n", "t", "d", "k", "r")
    overhead = len(address(*prefix, ("",)))
    assert len(address(*prefix, ("x" * (RELATION_KEY_MAX_BYTES - overhead),))) == RELATION_KEY_MAX_BYTES
    for identity in ("x" * (RELATION_KEY_MAX_BYTES - overhead + 1), "x" * 65536):
        with pytest.raises(ValueError):
            address(*prefix, (identity,))
    assert len(address("n" * 256, "t" * 128, "task", "task_graph_binding_capture", "owner", ("p" * 256,))) == (694 if address is scope_key else 695)
    assert len(address("n" * 256, "t" * 128, "execution", "turn", "scope", ("s" * 256,))) == (677 if address is scope_key else 678)


@pytest.mark.parametrize("relation", (b"\x00", bytes(range(256)), b"\xff" * RELATION_KEY_MAX_BYTES))
def test_binary_relations_roundtrip_through_current_wire_and_rows(relation: bytes) -> None:
    record = replace(_record(b"sort"), scope_key=relation, parent_key=relation)
    assert decode_record(encode_record(record)) == record
    assert _record_from_row({**_record_values(record), "store_key": b"store"}) == record
    validate_record_identity(record)


@pytest.mark.parametrize("field", ("scope_key", "parent_key"))
@pytest.mark.parametrize("relation", (b"", "oversized", "text", bytearray(b"x")))
def test_relation_producers_queries_and_readers_reject_invalid_binary_bounds(field: str, relation: object) -> None:
    if relation == "oversized":
        relation = b"x" * ((PARENT_KEY_MAX_BYTES if field == "parent_key" else RELATION_KEY_MAX_BYTES) + 1)
    record = _record(b"sort")
    with pytest.raises(ValueError):
        replace(record, **{field: relation})
    with pytest.raises(ValueError):
        RecordQuery(kind="probe", **{field: relation})
    with pytest.raises(AIError) as error:
        _record_from_row({**_record_values(record), "store_key": b"store", field: relation})
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    encoded = relation.hex() if isinstance(relation, bytes) else relation
    with pytest.raises(AIError) as error:
        decode_record({**encode_record(record), field.removesuffix("_key"): encoded})
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_relation_query_indexes_fit_full_binary_values_on_each_dialect() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.EXECUTION}))
    records = metadata.tables["ai_state_records"]
    for name in ("scope_key", "parent_key"):
        column_type = records.c[name].type
        assert column_type.compile(dialect=postgresql.dialect()) == "BYTEA"
        assert column_type.compile(dialect=sqlite.dialect()) == "BLOB"
        assert column_type.compile(dialect=mysql.dialect()) == f"VARBINARY({PARENT_KEY_MAX_BYTES if name == 'parent_key' else RELATION_KEY_MAX_BYTES})"
        index, = (index for index in records.indexes if index.info.get("ddl_dialect") == "portable"
                      and tuple(column.name for column in index.columns) == (name, "sort_key"))
        assert sum(column.type.length for column in index.columns) == (2304 if name == "parent_key" else 2176)
    subject_index, = (index for index in records.indexes
                      if index.info.get("ddl_dialect") == "portable"
                      and tuple(column.name for column in index.columns) == ("scope_key", "subject_id"))
    assert sum(column.type.length for column in subject_index.columns) == 1152


def test_relation_fields_do_not_reveal_raw_owners_in_repr() -> None:
    secret = b"private-owner-MUST-NOT-APPEAR"
    record = replace(_record(b"sort"), scope_key=secret, parent_key=secret)
    query = RecordQuery(scope_key=secret, parent_key=secret)
    assert secret.decode("ascii") not in repr(record)
    assert secret.decode("ascii") not in repr(query)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_relation_queries_isolate_all_global_components_with_exact_paging(tmp_path: Path, backend: str) -> None:
    engine = None
    if backend == "memory":
        store = InMemoryStateStore()
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'relations.db'}")
        await provision_database(engine)
        store = SqlStateStore(engine)
    await store.initialize()
    try:
        parts = ("namespace", "tenant", "execution", "probe", "owner", ("é",))
        scopes = [parts]
        for index in range(5):
            other = list(parts)
            other[index] = str(other[index]) + "-other"
            scopes.append(tuple(other))
        scopes.append((*parts[:5], ("e\u0301",)))
        records = tuple(replace(_record(str(index).encode()),
                                scope_key=scope_key(*scope), parent_key=parent_key(*scope))
                        for index, scope in enumerate(scopes))
        same_scope = replace(records[0], key=b"z" * 32, sort_key=b"z" * SORT_KEY_MAX_BYTES)
        await store.mutate(lambda tx: tx.insert_records((*records, same_scope)))
        for field, address in (("scope_key", scope_key), ("parent_key", parent_key)):
            first = await store.read(lambda tx: tx.list_records(RecordQuery(**{field: address(*parts)}, limit=1)))
            assert first == (records[0],)
            last = await store.read(lambda tx: tx.list_records(RecordQuery(
                **{field: address(*parts)}, after_sort_key=first[0].sort_key,
                after_key=first[0].key, limit=1,
            )))
            assert last == (same_scope,)
        longest = replace(_record(b"longest"), scope_key=b"s" * RELATION_KEY_MAX_BYTES,
                          parent_key=b"p" * RELATION_KEY_MAX_BYTES, sort_key=b"x" * SORT_KEY_MAX_BYTES)
        await store.mutate(lambda tx: tx.insert_record(longest))
        assert await store.read(lambda tx: tx.list_records(RecordQuery(scope_key=longest.scope_key))) == (longest,)
    finally:
        await store.close()
        if engine is not None:
            await engine.dispose()
