#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL inspection stays bounded and releases connections before body reads."""

import asyncio
import hashlib
from pathlib import Path

import pytest
from sqlalchemy import event, insert, update
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.storage import SqlObjectStore, build_object_sql_metadata, read_object


@pytest.mark.asyncio
async def test_sql_inspection_pages_seek_within_store_and_release_before_yield(tmp_path: Path) -> None:
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'inspection.db'}", pool_size=1, max_overflow=0,
    )
    digest = hashlib.sha256(b"").hexdigest()
    keys = {f"opaque/{index}/路径" for index in range(385)}
    table = build_object_sql_metadata().tables["ai_objects"]
    queries = []

    def capture(_connection, _cursor, statement, parameters, _context, _many):
        if statement.lstrip().upper().startswith("SELECT") and "ORDER BY ai_objects.object_locator" in statement:
            queries.append((statement, parameters))

    try:
        await provision_database(engine)
        async with engine.begin() as connection:
            await connection.execute(insert(table), [
                {"store_id": store_id, "object_key": key.encode(),
                 "object_locator": hashlib.sha256(key.encode()).hexdigest(),
                 "content_digest": digest, "size": 0}
                for store_id in ("selected", "other") for key in keys
            ])
        store = SqlObjectStore(engine, store_id="selected")
        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        found = []
        stream = store.list_objects()
        try:
            async for item in stream:
                assert engine.pool.checkedout() == 0
                assert item.digest == digest and item.size == 0
                if not found:
                    assert await asyncio.wait_for(read_object(
                        store, item.key, expected_digest=digest, expected_size=0,
                    ), timeout=5) == b""
                    # A removed, already returned key cannot shift later pages.
                    assert await store.delete_object(item.key, expected_digest=digest)
                found.append(item.key)
        finally:
            await stream.aclose()
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len(found) == len(keys) and set(found) == keys
        assert len(queries) > 1
        async with engine.connect() as connection:
            for statement, parameters in queries:
                assert "LIMIT" in statement
                assert "ai_objects.id" not in statement
                assert "ai_objects.updated_at" not in statement
                assert "ai_objects.created_at" not in statement
                plan = (await connection.exec_driver_sql(
                    "EXPLAIN QUERY PLAN " + statement, parameters,
                )).all()
                details = " ".join(row[-1] for row in plan)
                assert "SEARCH ai_objects" in details
                assert "store_id=?" in details
                assert "TEMP B-TREE" not in details
                if "object_locator >" in statement:
                    assert "object_locator>?" in details
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_inspection_rejects_locator_mismatch_and_closes_connection(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'corrupt.db'}")
    table = build_object_sql_metadata().tables["ai_objects"]
    digest = hashlib.sha256(b"").hexdigest()
    try:
        await provision_database(engine)
        async with engine.begin() as connection:
            await connection.execute(insert(table).values(
                store_id="selected", object_key=b"original",
                object_locator=hashlib.sha256(b"original").hexdigest(),
                content_digest=digest, size=0,
            ))
            await connection.execute(update(table).where(table.c.store_id == "selected").values(
                object_key=b"changed",
            ))
        stream = SqlObjectStore(engine, store_id="selected").list_objects()
        try:
            with pytest.raises(AIError) as error:
                await anext(stream)
            assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert engine.pool.checkedout() == 0
        finally:
            await stream.aclose()
    finally:
        await engine.dispose()
