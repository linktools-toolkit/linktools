#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for storage stability boundaries."""

import errno
import hashlib
import json
import os
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import linktools.ai.runtime.state._materializer as materializer
import linktools.ai.storage._files as files_module
import linktools.ai.storage._object_filesystem as object_module
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage
from linktools.ai.storage import (
    FilesystemObjectStore,
    MySQLDialect,
    SqlObjectStore,
    build_object_sql_metadata,
    read_object,
    validate_sql,
)
from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.engine import URL


async def _chunks(value: bytes) -> AsyncIterator[bytes]:
    yield value


def test_object_store_id_limit_matches_sql_schema() -> None:
    metadata = build_object_sql_metadata()
    store_id_type = metadata.tables["ai_objects"].c.store_id.type
    assert store_id_type.length == 128


@pytest.mark.asyncio
async def test_sql_object_store_accepts_maximum_store_id(tmp_path: Path) -> None:
    engine = create_async_engine(
        URL.create("sqlite+aiosqlite", database=str(tmp_path / "objects.db"))
    )
    await provision_database(engine)
    store_id = "s" * 128
    store = SqlObjectStore(engine, store_id=store_id)
    payload = b"object-store-id"
    digest = hashlib.sha256(payload).hexdigest()
    try:
        result = await store.put(
            "payload",
            _chunks(payload),
            expected_size=len(payload),
            expected_digest=digest,
        )
        assert result.digest == digest
        assert await store.stat("payload") == result
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_filesystem_object_publish_sync_failure_keeps_complete_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"atomic-object"
    digest = hashlib.sha256(payload).hexdigest()
    destination = store._path("payload")
    original_sync = object_module.sync_directory
    failed = False

    def sync(path: Path) -> None:
        nonlocal failed
        if path == destination.parent and destination.exists() and not failed:
            failed = True
            raise OSError(errno.EIO, "injected directory sync failure")
        original_sync(path)

    monkeypatch.setattr(object_module, "sync_directory", sync)
    with pytest.raises(AIError) as raised:
        await store.put(
            "payload",
            _chunks(payload),
            expected_size=len(payload),
            expected_digest=digest,
        )
    assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

    monkeypatch.setattr(object_module, "sync_directory", original_sync)
    stat = await store.stat("payload")
    assert stat is not None
    assert stat.digest == digest
    assert b"".join([chunk async for chunk in store.open("payload")]) == payload

    duplicate = await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )
    assert duplicate.digest == digest


@pytest.mark.asyncio
async def test_filesystem_object_delete_sync_failure_is_logically_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"delete-object"
    digest = hashlib.sha256(payload).hexdigest()
    await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )
    destination = store._path("payload")
    original_sync = object_module.sync_directory
    failed = False

    def sync(path: Path) -> None:
        nonlocal failed
        if path == destination.parent and not destination.exists() and not failed:
            failed = True
            raise OSError(errno.EIO, "injected directory sync failure")
        original_sync(path)

    monkeypatch.setattr(object_module, "sync_directory", sync)
    with pytest.raises(AIError) as raised:
        await store.delete_object("payload", expected_digest=digest)
    assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED

    monkeypatch.setattr(object_module, "sync_directory", original_sync)
    assert await store.stat("payload") is None
    assert [value async for value in store.list_objects()] == []
    await store.validate_integrity()

    restored = await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )
    assert restored.digest == digest


def test_directory_sync_propagates_real_io_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("Windows directory fsync is intentionally unsupported")

    opened: list[int] = []
    closed: list[int] = []
    original_open = files_module.os.open
    original_close = files_module.os.close

    def tracked_open(path: Path, flags: int) -> int:
        descriptor = original_open(path, flags)
        opened.append(descriptor)
        return descriptor

    def tracked_close(descriptor: int) -> None:
        closed.append(descriptor)
        original_close(descriptor)

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(errno.EIO, "injected fsync failure")

    monkeypatch.setattr(files_module.os, "open", tracked_open)
    monkeypatch.setattr(files_module.os, "close", tracked_close)
    monkeypatch.setattr(files_module.os, "fsync", fail_fsync)

    with pytest.raises(OSError) as raised:
        files_module.sync_directory(tmp_path)
    assert raised.value.errno == errno.EIO
    assert opened
    assert closed == opened


class _ClosableStream:
    def __init__(self) -> None:
        self._chunks = (b"ab", b"cd", b"should-not-be-read")
        self.index = 0
        self.closed = False

    def __aiter__(self) -> "_ClosableStream":
        return self

    async def __anext__(self) -> bytes:
        if self.index >= len(self._chunks):
            raise StopAsyncIteration
        value = self._chunks[self.index]
        self.index += 1
        return value

    async def aclose(self) -> None:
        self.closed = True


class _ClosableStore:
    def __init__(self, stream: _ClosableStream) -> None:
        self.stream = stream

    def open(self, key: str) -> AsyncIterator[bytes]:
        del key
        return self.stream


@pytest.mark.asyncio
async def test_read_object_rejects_overflow_before_buffering_or_reading_more() -> None:
    stream = _ClosableStream()
    store = _ClosableStore(stream)
    with pytest.raises(AIError) as raised:
        await read_object(
            store,  # type: ignore[arg-type]
            "payload",
            expected_digest="0" * 64,
            expected_size=3,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert stream.index == 2
    assert stream.closed


@pytest.mark.asyncio
async def test_filesystem_object_v1_layout_remains_readable(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    store_id = "builtin"
    key = "v1/runtime/object"
    payload = b"filesystem-object-v1"
    content_digest = hashlib.sha256(payload).hexdigest()
    key_digest = hashlib.sha256(
        store_id.encode("utf-8") + b"\0" + key.encode("utf-8")
    ).hexdigest()
    object_path = root / key_digest[:2] / key_digest
    object_path.mkdir(parents=True)
    (object_path / "data").write_bytes(payload)
    (object_path / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "key": key,
                "digest": content_digest,
                "size": len(payload),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    store = FilesystemObjectStore(root, store_id=store_id)
    stat = await store.stat(key)
    assert stat is not None
    assert stat.digest == content_digest
    assert stat.size == len(payload)
    assert b"".join([chunk async for chunk in store.open(key)]) == payload


@pytest.mark.asyncio
async def test_filesystem_object_future_layout_version_is_rejected(
    tmp_path: Path,
) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"future-version"
    digest = hashlib.sha256(payload).hexdigest()
    await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )
    metadata = store._path("payload") / "metadata.json"
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["version"] = 2
    metadata.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AIError) as raised:
        await store.stat("payload")
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("size", True),
        ("size", "1"),
        ("size", 1.5),
        ("digest", "A" * 64),
        ("digest", 123),
        ("key", 123),
    ),
)
async def test_filesystem_object_metadata_rejects_implicit_conversion(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"metadata"
    digest = hashlib.sha256(payload).hexdigest()
    await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )
    metadata = store._path("payload") / "metadata.json"
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document[field] = value
    metadata.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AIError) as raised:
        await store.stat("payload")
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_filesystem_object_corrupt_metadata_fails_all_read_surfaces(
    tmp_path: Path,
) -> None:
    store = FilesystemObjectStore(tmp_path / "objects")
    payload = b"metadata"
    digest = hashlib.sha256(payload).hexdigest()
    await store.put(
        "payload",
        _chunks(payload),
        expected_size=len(payload),
        expected_digest=digest,
    )
    metadata = store._path("payload") / "metadata.json"
    document = json.loads(metadata.read_text(encoding="utf-8"))
    document["size"] = True
    metadata.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(AIError) as stat_error:
        await store.stat("payload")
    assert stat_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    stream = store.open("payload")
    with pytest.raises(AIError) as open_error:
        await anext(stream)
    assert open_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    with pytest.raises(AIError) as list_error:
        _ = [value async for value in store.list_objects()]
    assert list_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    with pytest.raises(AIError) as validate_error:
        await store.validate_integrity()
    assert validate_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def _expected_constraint_metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "runtime_test",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("token", String(32), nullable=False),
        UniqueConstraint("token"),
    )
    return metadata


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ddl",
    (
        "CREATE TABLE runtime_test (id INTEGER, token VARCHAR(32) NOT NULL, UNIQUE (token))",
        "CREATE TABLE runtime_test (id INTEGER PRIMARY KEY, token VARCHAR(32) NOT NULL)",
    ),
)
async def test_sql_validation_rejects_missing_identity_constraints(
    tmp_path: Path,
    ddl: str,
) -> None:
    path = tmp_path / (hashlib.sha256(ddl.encode()).hexdigest() + ".db")
    engine = create_async_engine(
        URL.create("sqlite+aiosqlite", database=str(path))
    )
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(ddl)
        with pytest.raises(AIError) as raised:
            await validate_sql(engine, _expected_constraint_metadata())
        assert raised.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_validation_accepts_equivalent_unique_name(
    tmp_path: Path,
) -> None:
    path = tmp_path / "equivalent.db"
    engine = create_async_engine(
        URL.create("sqlite+aiosqlite", database=str(path))
    )
    try:
        async with engine.begin() as connection:
            await connection.exec_driver_sql(
                "CREATE TABLE runtime_test ("
                "id INTEGER PRIMARY KEY, "
                "token VARCHAR(32) NOT NULL, "
                "CONSTRAINT differently_named UNIQUE (token)"
                ")"
            )
        await validate_sql(engine, _expected_constraint_metadata())
    finally:
        await engine.dispose()


class _MySQLSession:
    def __init__(self, *, duplicate: bool, target_exists: bool = True) -> None:
        self.duplicate = duplicate
        self.target_exists = target_exists

    async def execute(self, _statement: object) -> object:
        if self.duplicate:
            raise IntegrityError(
                "INSERT",
                {},
                RuntimeError("1062 Duplicate entry"),
            )
        return SimpleNamespace(lastrowid=7)

    async def scalar(self, _statement: object) -> int | None:
        return 3 if self.target_exists else None


@pytest.mark.asyncio
async def test_mysql_insert_result_does_not_depend_on_found_rows_rowcount() -> None:
    metadata = MetaData()
    table = Table(
        "items",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("token", String(32), nullable=False),
        UniqueConstraint("token"),
    )
    dialect = MySQLDialect()

    inserted = await dialect.insert_ignore_conflict(
        _MySQLSession(duplicate=False),  # type: ignore[arg-type]
        table=table,
        values={"token": "value"},
        index_elements=("token",),
    )
    assert inserted.inserted is True
    assert inserted.row_id is None

    duplicate = await dialect.insert_ignore_conflict(
        _MySQLSession(duplicate=True),  # type: ignore[arg-type]
        table=table,
        values={"token": "value"},
        index_elements=("token",),
    )
    assert duplicate.inserted is False


@pytest.mark.asyncio
async def test_mysql_insert_does_not_hide_other_unique_conflict() -> None:
    metadata = MetaData()
    table = Table(
        "items",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("token", String(32), nullable=False),
        UniqueConstraint("token"),
    )
    dialect = MySQLDialect()
    with pytest.raises(IntegrityError):
        await dialect.insert_ignore_conflict(
            _MySQLSession(duplicate=True, target_exists=False),  # type: ignore[arg-type]
            table=table,
            values={"token": "value"},
            index_elements=("token",),
        )


@pytest.mark.asyncio
async def test_sqlite_route_preserves_question_mark_in_filename(
    tmp_path: Path,
) -> None:
    if os.name == "nt":
        pytest.skip("question mark is not a legal Windows filename")

    first = tmp_path / "runtime?one.sqlite"
    second = tmp_path / "runtime?two.sqlite"
    for path, namespace in ((first, "first"), (second, "second")):
        state = RuntimeStorage.sqlite(path)
        await state.initialize(namespace=namespace, tenant_id="tenant")
        await state.close()

    assert first.is_file()
    assert second.is_file()
    assert first != second
    for path in (first, second):
        with sqlite3.connect(path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        assert "ai_state_records" in tables


@pytest.mark.asyncio
async def test_sqlite_initial_publish_failure_leaves_no_final_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "runtime.sqlite"
    original_sync = materializer._sync_file

    def fail_sync(_path: Path) -> None:
        raise OSError(errno.EIO, "injected database sync failure")

    monkeypatch.setattr(materializer, "_sync_file", fail_sync)
    failed = RuntimeStorage.sqlite(database)
    with pytest.raises(OSError):
        await failed.initialize(namespace="sqlite-failure", tenant_id="tenant")
    assert not database.exists()

    monkeypatch.setattr(materializer, "_sync_file", original_sync)
    retry = RuntimeStorage.sqlite(database)
    await retry.initialize(namespace="sqlite-failure", tenant_id="tenant")
    await retry.close()
    assert database.is_file()
    assert not list(tmp_path.glob(".runtime.sqlite.init-*.db*"))


@pytest.mark.asyncio
async def test_sqlite_post_publish_sync_failure_keeps_complete_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "runtime.sqlite"
    original_sync = materializer.sync_directory
    failed = False

    def sync(path: Path) -> None:
        nonlocal failed
        if path == tmp_path and database.exists() and not failed:
            failed = True
            raise OSError(errno.EIO, "injected parent sync failure")
        original_sync(path)

    monkeypatch.setattr(materializer, "sync_directory", sync)
    state = RuntimeStorage.sqlite(database)
    with pytest.raises(AIError) as raised:
        await state.initialize(namespace="sqlite-published", tenant_id="tenant")
    assert raised.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert database.is_file()

    monkeypatch.setattr(materializer, "sync_directory", original_sync)
    retry = RuntimeStorage.sqlite(database)
    await retry.initialize(namespace="sqlite-published", tenant_id="tenant")
    await retry.close()
