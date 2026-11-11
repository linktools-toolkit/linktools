#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL object reads bound roundtrips without weakening stream integrity."""

import asyncio
import getpass
import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import delete, event, insert, select
from sqlalchemy.engine import Connection, URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.storage import SqlObjectStore
from linktools.ai.storage._object_sql import build_object_sql_metadata


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    if value:
        yield value


async def _assert_reads(engine: AsyncEngine) -> None:
    await provision_database(engine)
    store = SqlObjectStore(engine)
    statements: list[str] = []

    def capture(
        _connection: object, _cursor: object, sql: str,
        _parameters: object, _context: object, _many: bool,
    ) -> None:
        if sql.lstrip().upper().startswith("SELECT"):
            statements.append(sql)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        for size in (0, 16384, 3 * 1024 * 1024):
            key = f"{size}:" + "k" * (1024 - len(f"{size}:"))
            payload = b"x" * size
            await store.put(key, _chunks(payload), expected_size=size,
                            expected_digest=hashlib.sha256(payload).hexdigest())
            statements.clear()
            assert b"".join([part async for part in store.open(key)]) == payload
            assert len(statements) <= 1
            statements.clear()
            info = await store.stat(key)
            assert info is not None and info.size == size
            assert len(statements) <= 1
            assert all("ai_object_chunks" not in sql for sql in statements)

        for opaque_key, value in (("name", b"one"), ("name ", b"two"), ("../路径/", b"three"),
                                  ("🪷" * 256, b"four")):
            digest = hashlib.sha256(value).hexdigest()
            await store.put(opaque_key, _chunks(value), expected_size=len(value), expected_digest=digest)
            assert b"".join([part async for part in store.open(opaque_key)]) == value
        assert b"".join([part async for part in store.open("name")]) == b"one"

        with pytest.raises(AIError) as missing:
            _ = [part async for part in store.open("missing")]
        assert missing.value.code is ErrorCode.STORAGE_NOT_FOUND

        started = asyncio.Event()
        paused = asyncio.Event()

        async def consume() -> None:
            stream = store.open(key)
            try:
                async for _part in stream:
                    started.set()
                    await paused.wait()
            finally:
                await stream.aclose()

        reader = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started.wait(), 5)
            reader.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reader
            async with engine.connect() as connection:
                assert await connection.scalar(select(1)) == 1
        finally:
            paused.set()
            await asyncio.gather(reader, return_exceptions=True)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)


async def _assert_corruption(engine: AsyncEngine) -> None:
    metadata = build_object_sql_metadata()
    objects = metadata.tables["ai_objects"]
    chunks = metadata.tables["ai_object_chunks"]
    store = SqlObjectStore(engine)
    for fault, indices in (
        ("missing_first", (1, 2)), ("nonzero_first", (7, 8, 9)),
        ("gap", (0, 2)), ("truncated", (0, 1)), ("no_chunks", ()),
        ("size", (0, 1, 2)),
        ("digest", (0, 1, 2)), ("empty_digest", ()),
    ):
        key = f"corrupt:{fault}"
        async with engine.begin() as connection:
            result = await connection.execute(insert(objects).values(
                store_id=store.store_id, object_key=key.encode("utf-8"),
                object_locator=hashlib.sha256(key.encode("utf-8")).hexdigest(),
                content_digest=hashlib.sha256(b"wrong" if fault == "digest" else b"abc").hexdigest(),
                size=4 if fault == "size" else 0 if fault == "empty_digest" else 3,
            ))
            if indices:
                await connection.execute(insert(chunks), [
                    {"object_id": result.inserted_primary_key[0], "chunk_index": index, "content": bytes((97 + index % 3,))}
                    for index in indices
                ])
        with pytest.raises(AIError) as corrupt:
            _ = [part async for part in store.open(key)]
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR, fault

    key = "duplicate-index"
    digest = hashlib.sha256(b"x").hexdigest()
    await store.put(key, _chunks(b"x"), expected_size=1, expected_digest=digest)
    with pytest.raises(IntegrityError):
        async with engine.begin() as connection:
            object_id = await connection.scalar(select(objects.c.id).where(
                objects.c.store_id == store.store_id, objects.c.object_key == key.encode("utf-8"),
            ))
            await connection.execute(insert(chunks).values(
                object_id=object_id, chunk_index=0, content=b"duplicate",
            ))
    assert b"".join([part async for part in store.open(key)]) == b"x"


@pytest.mark.asyncio
async def test_sqlite_object_read_roundtrips_and_integrity(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'objects.db'}",
                                 pool_size=1, max_overflow=0)
    try:
        await _assert_reads(engine)
        await _assert_corruption(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_server_object_read_roundtrips_and_integrity(
    _server: tuple[str, list[str]],
) -> None:
    name, command = _server
    if name == "postgresql":
        url = URL.create("postgresql+asyncpg", username=getpass.getuser(), database=command[-1],
                         query={"host": command[command.index("-h") + 1]})
    else:
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url, pool_size=1, max_overflow=0)
    try:
        await _assert_reads(engine)
        await _assert_corruption(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_object_chunk_ids_isolate_same_keys_and_delete(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ids.db'}")
    await provision_database(engine)
    first = SqlObjectStore(engine, store_id="first")
    second = SqlObjectStore(engine, store_id="second")
    metadata = build_object_sql_metadata()
    objects = metadata.tables["ai_objects"]
    chunks = metadata.tables["ai_object_chunks"]
    statements: list[str] = []

    def capture(
        _connection: object, _cursor: object, sql: str,
        _parameters: object, _context: object, _many: bool,
    ) -> None:
        if sql.lstrip().upper().startswith("SELECT"):
            statements.append(sql)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        assert "key_digest" not in chunks.c
        for store, value in ((first, b"one"), (second, b"two")):
            statements.clear()
            await store.put("same", _chunks(value), expected_size=len(value),
                            expected_digest=hashlib.sha256(value).hexdigest())
            assert statements == []
        async with engine.connect() as connection:
            rows = (await connection.execute(
                select(objects.c.store_id, chunks.c.content)
                .join(chunks, chunks.c.object_id == objects.c.id)
                .order_by(objects.c.store_id)
            )).all()
            assert rows == [("first", b"one"), ("second", b"two")]
        statements.clear()
        assert await first.delete_object("same", expected_digest=hashlib.sha256(b"one").hexdigest())
        assert len(statements) == 1
        assert await first.stat("same") is None
        assert b"".join([part async for part in second.open("same")]) == b"two"
        await second.validate_integrity()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await engine.dispose()


@pytest.mark.asyncio
async def test_object_delete_fences_the_header_identity_read(tmp_path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'replacement.db'}")
    await provision_database(engine)
    store = SqlObjectStore(engine)
    metadata = build_object_sql_metadata()
    objects = metadata.tables["ai_objects"]
    chunks = metadata.tables["ai_object_chunks"]
    key = "replaceable"
    old_digest = hashlib.sha256(b"old").hexdigest()
    new_digest = hashlib.sha256(b"new").hexdigest()
    await store.put(key, _chunks(b"old"), expected_size=3, expected_digest=old_digest)
    replaced = False

    def replace_header(
        connection: Connection, _cursor: object, sql: str,
        _parameters: object, _context: object, _many: bool,
    ) -> None:
        nonlocal replaced
        if replaced or not sql.startswith("DELETE FROM ai_object_chunks"):
            return
        replaced = True
        # Model a same-key replacement after the deletion's identity read.
        connection.execute(delete(objects).where(objects.c.store_id == store.store_id, objects.c.object_key == key.encode("utf-8")))
        connection.execute(insert(objects).values(
            id=999, store_id=store.store_id, object_key=key.encode("utf-8"),
            object_locator=hashlib.sha256(key.encode("utf-8")).hexdigest(),
            content_digest=new_digest, size=3,
        ))
        connection.execute(insert(chunks).values(object_id=999, chunk_index=0, content=b"new"))

    event.listen(engine.sync_engine, "before_cursor_execute", replace_header)
    try:
        assert not await store.delete_object(key, expected_digest=old_digest)
        assert replaced
        assert b"".join([part async for part in store.open(key)]) == b"new"
        await store.validate_integrity()
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", replace_header)
        await engine.dispose()
