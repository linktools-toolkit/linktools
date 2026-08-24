#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage dialect and raw Asset backend contract checks."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.asset import (
    AssetKey,
    AssetRoot,
    AssetStore,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    SqlAssetBackend,
    StrictConfigReader,
    build_asset_sql_metadata,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_asset_database
from linktools.ai.storage import (
    MySQLDialect,
    PayloadPolicy,
    PostgreSQLDialect,
    SqlErrorKind,
    SQLiteDialect,
    StorageChange,
    StorageLayer,
    StorageOperation,
    StorageOverlay,
    build_object_sql_metadata,
    classify_sql_error,
    is_retryable_sql_transaction,
    resolve_dialect,
)


class _DialectSession:
    def __init__(self, name: str) -> None:
        self.bind = SimpleNamespace(dialect=SimpleNamespace(name=name))
        self.statements = []

    def get_bind(self) -> object:
        return self.bind

    async def execute(self, statement: object) -> "_DialectResult":
        self.statements.append(statement)
        return _DialectResult()


class _DialectResult:
    rowcount = 1
    lastrowid = None

    def scalar_one(self) -> int:
        return 1


class _MappedPathAdapter:
    def validate(self, kinds: "Sequence[str]") -> None:
        del kinds

    def root_path(self, kind: str) -> str:
        return f"mapped/{kind}"

    def to_path(self, key: AssetKey) -> str:
        return f"mapped/{key.kind}/{key.id}.json"

    def from_path(self, path: str) -> "AssetKey | None":
        parts = path.split("/")
        if len(parts) != 3 or parts[0] != "mapped" or not parts[2].endswith(".json"):
            return None
        return AssetKey(parts[1], parts[2][:-5])


def test_sql_error_classification_is_owned_by_storage() -> None:
    from sqlalchemy.exc import IntegrityError, OperationalError

    integrity = IntegrityError("insert", {}, RuntimeError("unique constraint failed"))
    locked = OperationalError("update", {}, RuntimeError("database is locked"))
    unavailable = OperationalError("select", {}, RuntimeError("connection refused"))
    assert classify_sql_error(integrity) is SqlErrorKind.INTEGRITY
    assert not is_retryable_sql_transaction(integrity)
    assert classify_sql_error(locked) is SqlErrorKind.RETRYABLE_TRANSACTION
    assert is_retryable_sql_transaction(locked)
    assert classify_sql_error(unavailable) is SqlErrorKind.DATABASE
    assert not is_retryable_sql_transaction(unavailable)
    assert classify_sql_error(RuntimeError("application failure")) is SqlErrorKind.UNKNOWN
    assert not is_retryable_sql_transaction(RuntimeError("application failure"))


def test_sql_error_classification_reads_structured_driver_codes() -> None:
    from sqlalchemy.exc import OperationalError

    class SQLiteBusySnapshot(Exception):
        sqlite_errorcode = 517

    class PostgreSQLSerializationFailure(Exception):
        sqlstate = "40001"

    class MySQLDeadlock(Exception):
        errno = 1213

    for original in (
        SQLiteBusySnapshot("busy snapshot"),
        PostgreSQLSerializationFailure("serialization failure"),
        MySQLDeadlock("deadlock"),
    ):
        error = OperationalError("update", {}, original)
        assert classify_sql_error(error) is SqlErrorKind.RETRYABLE_TRANSACTION
        assert is_retryable_sql_transaction(error)

    class SQLiteLocked(Exception):
        sqlite_errorcode = 6

    assert not is_retryable_sql_transaction(
        OperationalError("update", {}, SQLiteLocked("locked"))
    )


def test_asset_path_adapter_and_config_are_available() -> None:
    adapter = _MappedPathAdapter()
    key = AssetKey("mcp", "one")
    assert adapter.to_path(key) == "mapped/mcp/one.json"
    assert adapter.from_path("mapped/mcp/one.json") == key
    assert StrictConfigReader({}, context="asset").str_or_bool("missing") is None


@pytest.mark.asyncio
async def test_local_directory_asset_backend_maps_single_files(tmp_path: Path) -> None:
    backend = DirectoryAssetBackend(
        AssetRoot("file:directory", "file", str(tmp_path), "directory"),
        path_adapter=_MappedPathAdapter(),
    )
    await backend.initialize()
    key = AssetKey("mcp", "one")
    (tmp_path / "mapped/mcp").mkdir(parents=True)
    (tmp_path / "mapped/mcp/one.json").write_bytes(b"one")
    assert backend.writable is False
    assert await backend.get(key) == b"one"
    info = await backend.stat(key)
    assert info is not None
    assert isinstance(info.revision.value, int)
    repeated = await backend.stat(key)
    assert repeated is not None
    assert info.revision == repeated.revision
    assert not hasattr(backend, "put")


@pytest.mark.asyncio
async def test_local_directory_asset_layer_stat_has_integer_revision(tmp_path: Path) -> None:
    path = tmp_path / "mapped" / "mcp" / "one.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"one")
    primary = InMemoryAssetBackend(AssetRoot("memory:primary", "memory", "primary", "primary"))
    builtin = DirectoryAssetBackend(
        AssetRoot("file:directory", "file", str(tmp_path), "directory"),
        path_adapter=_MappedPathAdapter(),
    )
    store = AssetStore(
        StorageOverlay(
            primary,
            writer=primary,
            layers=(StorageLayer("builtin", builtin),),
        )
    )
    await store.initialize()
    info = await store.stat(AssetKey("mcp", "one"))
    assert info is not None
    assert isinstance(info.revision.value, int)


@pytest.mark.asyncio
async def test_sql_dialect_upsert_uses_vendor_statement() -> None:
    from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint
    from sqlalchemy.dialects import mysql, postgresql, sqlite

    table = Table(
        "asset",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("path", String(128), nullable=False),
        Column("value", String(128), nullable=False),
        Column("revision", Integer, nullable=False),
        UniqueConstraint("path"),
    )
    cases = (
        (SQLiteDialect(), sqlite.dialect(), "ON CONFLICT"),
        (PostgreSQLDialect(), postgresql.dialect(), "ON CONFLICT"),
        (MySQLDialect(), mysql.dialect(), "ON DUPLICATE KEY UPDATE"),
    )
    for dialect, compiler_dialect, marker in cases:
        session = _DialectSession(dialect.name)
        assert resolve_dialect(session).name == dialect.name
        await dialect.upsert(
            session,
            table=table,
            values={"path": "sample/one", "value": "one"},
            set_values={"value": "one"},
            index_elements=("path",),
        )
        assert marker in str(session.statements[0].compile(dialect=compiler_dialect))
        session.statements.clear()
        assert await dialect.upsert_increment(
            session,
            table=table,
            values={"path": "sample/one", "value": "one"},
            column="revision",
            index_elements=("path",),
        ) == 1
        assert marker in str(session.statements[0].compile(dialect=compiler_dialect))


@pytest.mark.asyncio
async def test_sql_asset_backend_uses_normalized_history_tables(tmp_path: Path) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'asset.db'}")
    try:
        from sqlalchemy import MetaData

        metadata = MetaData()
        build_asset_sql_metadata(metadata=metadata)
        build_object_sql_metadata(metadata=metadata)
        tables = metadata.tables
        backend = SqlAssetBackend(engine, namespace="test")
        assert backend.root.scheme == "sql"
        assert tuple(table.name for table in (tables["ai_asset_entries"], tables["ai_asset_changes"], tables["ai_asset_heads"])) == (
            "ai_asset_entries",
            "ai_asset_changes",
            "ai_asset_heads",
        )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_asset_namespace_matches_the_persisted_column_limit() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    with pytest.raises(AIError) as raised:
        SqlAssetBackend(engine, namespace="a" * 129)
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_asset_namespace_isolates_the_same_logical_key(tmp_path: Path) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'asset-isolation.db'}")
    first = SqlAssetBackend(engine, namespace="assets-a")
    second = SqlAssetBackend(engine, namespace="assets-b")
    key = AssetKey("prompt", "shared")
    try:
        await provision_asset_database(engine)
        await first.initialize()
        await second.initialize()
        await first.put(key, b"first")
        await second.put(key, b"second")

        assert await first.get(key) == b"first"
        assert await second.get(key) == b"second"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_asset_backend_persists_history_outside_revision_row(tmp_path: Path) -> None:
    from sqlalchemy import event, func, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'asset.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    statements: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def capture_statement(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    try:
        from sqlalchemy import MetaData

        metadata = MetaData()
        build_asset_sql_metadata(metadata=metadata)
        build_object_sql_metadata(metadata=metadata)
        tables = metadata.tables
        backend = SqlAssetBackend(
            engine,
            namespace="history",
            payload_policy=PayloadPolicy(inline_limit_bytes=1),
        )
        await provision_asset_database(engine)
        await backend.initialize()
        key = AssetKey("mcp", "history.yaml")
        first = await backend.put(key, b"one")
        second = await backend.put(key, b"two", expected_revision=first.entry_revision)
        deleted = await backend.delete(key, expected_revision=second.entry_revision)
        assert await backend.get_at_version(key, 1) == b"one"
        assert await backend.get_at_version(key, 2) == b"two"
        assert await backend.get(key) is None
        assert len(await backend.list_versions(key)) == 3
        reset = await backend.reset(key, expected_revision=deleted.entry_revision)
        assert reset.reset is True
        assert await backend.get(key) is None
        assert len(await backend.list_versions(key)) == 4
        await backend.initialize()
        assert await backend.get(key) is None
        statements.clear()
        assert await backend.stat(key) is not None
        assert await backend.stat(key) is not None
        selects = tuple(
            statement
            for statement in statements
            if statement.lstrip().upper().startswith("SELECT")
        )
        assert len(selects) == 2
        assert all("ai_asset_" in statement for statement in selects)
        async with session_factory() as session:
            counts = []
            for table in (tables["ai_asset_entries"], tables["ai_asset_changes"], tables["ai_objects"], tables["ai_asset_heads"]):
                counts.append(await session.scalar(select(func.count()).select_from(table)))
        assert counts == [1, 4, 2, 1]
        assert not hasattr(tables["ai_asset_heads"].c, "payload")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_asset_backend_provisions_its_owner_schema(tmp_path: Path) -> None:
    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'asset.db'}")
    try:
        backend = SqlAssetBackend(engine, namespace="missing")
        await provision_asset_database(engine)
        await backend.initialize()
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda sync_connection: inspect(sync_connection).get_table_names())
        assert {"ai_asset_entries", "ai_asset_changes", "ai_asset_heads"} <= set(tables)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_dialects_restore_batch_upsert_surface() -> None:
    from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint
    from sqlalchemy.dialects import mysql, postgresql, sqlite

    table = Table(
        "asset_batch",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("path", String(128), nullable=False),
        Column("value", String(128), nullable=False),
        UniqueConstraint("path"),
    )
    cases = (
        (SQLiteDialect(), sqlite.dialect(), "ON CONFLICT"),
        (PostgreSQLDialect(), postgresql.dialect(), "ON CONFLICT"),
        (MySQLDialect(), mysql.dialect(), "ON DUPLICATE KEY UPDATE"),
    )
    for dialect, compiler_dialect, marker in cases:
        session = _DialectSession(dialect.name)
        await dialect.upsert_many(
            session,
            table=table,
            rows=(
                {"path": "sample/one", "value": "one"},
                {"path": "sample/two", "value": "two"},
            ),
            set_columns=("value",),
            index_elements=("path",),
        )
        assert marker in str(session.statements[0].compile(dialect=compiler_dialect))


@pytest.mark.asyncio
async def test_sqlite_dialect_uses_native_atomic_returning_operations() -> None:
    from sqlalchemy import Column, Integer, MetaData, String, Table, UniqueConstraint
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    metadata = MetaData()
    table = Table(
        "dialect_counter",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("namespace", String(64), nullable=False),
        Column("revision", Integer, nullable=False),
        UniqueConstraint("namespace"),
    )
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as connection:
            await connection.run_sync(metadata.create_all)
        async with session_factory() as session, session.begin():
            dialect = resolve_dialect(session)
            assert await dialect.upsert_increment(
                session,
                table=table,
                values={"namespace": "test"},
                column="revision",
                index_elements=("namespace",),
            ) == 1
            assert await dialect.upsert_increment(
                session,
                table=table,
                values={"namespace": "test"},
                column="revision",
                index_elements=("namespace",),
            ) == 2
            deleted = await dialect.delete_returning(
                session,
                table=table,
                where=table.c.namespace == "test",
                returning=("revision",),
            )
            assert tuple(row["revision"] for row in deleted) == (2,)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_sql_asset_backend_batches_large_file_sets(tmp_path: Path) -> None:
    from sqlalchemy import func, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'asset.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        from sqlalchemy import MetaData

        metadata = MetaData()
        build_asset_sql_metadata(metadata=metadata)
        build_object_sql_metadata(metadata=metadata)
        tables = metadata.tables
        backend = SqlAssetBackend(
            engine,
            namespace="batch",
            payload_policy=PayloadPolicy(inline_limit_bytes=1),
        )
        await provision_asset_database(engine)
        await backend.initialize()
        changes = tuple(
            StorageChange(
                StorageOperation.PUT,
                AssetKey("skill", f"skill-{index}/SKILL.md"),
                f"skill-{index}".encode(),
                None,
            )
            for index in range(130)
        )
        result = await backend.apply_batch(changes)
        assert len(result.results) == 130
        assert await backend.get(AssetKey("skill", "skill-129/SKILL.md")) == b"skill-129"
        async with session_factory() as session:
            counts = []
            for table in (tables["ai_asset_entries"], tables["ai_asset_changes"], tables["ai_objects"]):
                counts.append(await session.scalar(select(func.count()).select_from(table)))
        assert tuple(counts) == (130, 130, 130)
    finally:
        await engine.dispose()
