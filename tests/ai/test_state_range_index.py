#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral bounded range-index contracts."""

import hashlib
from pathlib import Path

import pytest
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence._plan import PersistenceDomain
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.state import RecordQuery, StoredRecord

pytestmark = pytest.mark.asyncio
_SCOPE = b"s" * 32


def _record(sort_key: str) -> StoredRecord:
    return StoredRecord(
        hashlib.sha256(sort_key.encode("ascii")).digest(),
        _SCOPE,
        None,
        "probe",
        sort_key,
        None,
        0,
        None,
        0,
        None,
        {},
    )


async def _seed(store, values: tuple[StoredRecord, ...]) -> None:
    await store.mutate(lambda transaction: transaction.insert_records(values))


async def _list_prefix(store, prefix: str | None, *, limit: int = 10):
    return await store.read(
        lambda transaction: transaction.list_records(
            RecordQuery(
                scope_digest=_SCOPE,
                kind="probe",
                sort_key_prefix=prefix,
                limit=limit,
            )
        )
    )


async def test_sql_prefix_is_case_sensitive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "range.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    store = SqlStateStore(engine)
    await store.initialize()
    try:
        await _seed(
            store,
            tuple(_record(value) for value in ("A-one", "A-two", "a-one")),
        )
        values = await _list_prefix(store, "A")
        assert [value.sort_key for value in values] == ["A-one", "A-two"]
    finally:
        await store.close()
        await engine.dispose()


async def test_sql_sort_key_collation_is_explicit_for_supported_backends() -> None:
    metadata = build_runtime_sql_metadata(frozenset({PersistenceDomain.EXECUTION}))
    column_type = metadata.tables["ai_state_records"].c.sort_key.type
    assert column_type.dialect_impl(postgresql.dialect()).collation == "C"
    assert column_type.dialect_impl(sqlite.dialect()).collation == "BINARY"
    assert column_type.dialect_impl(mysql.dialect()).collation == "utf8mb4_bin"
