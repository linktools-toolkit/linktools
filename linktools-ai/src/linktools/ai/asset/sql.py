#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transactional SQL backend using the frozen historical asset schema."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypeAlias

from linktools.core import environ

from ..core.errors import ErrorCode, AIError
from ..storage.database import SqlSchemaRegistry
from ..storage.dialects import SqlAlchemyDialect, resolve_dialect
from ..storage.names import storage_name
from ..storage import (
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageOperation,
    StoragePutResult,
    StorageResetResult,
    VersionSummary,
)
from .domain import AssetInfo, AssetKey, AssetRevision, AssetRoot, AssetStoreRevision

try:
    from sqlalchemy import (
        BigInteger,
        Boolean,
        Column,
        DateTime,
        Index,
        Integer,
        LargeBinary,
        MetaData,
        String,
        Table,
        UniqueConstraint,
        insert,
        select,
        update,
    )
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
except ModuleNotFoundError as error:
    if error.name == "sqlalchemy":
        raise AIError(
            ErrorCode.OPTIONAL_DEPENDENCY_MISSING,
            "SQLAlchemy is required for linktools.ai.asset.sql",
        ) from error
    raise

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


_logger = environ.get_logger("ai.asset.sql")
_TOMBSTONE_ETAG = hashlib.sha256(b"").hexdigest()
SqlValue: TypeAlias = str | int | bool | bytes | datetime | None


@dataclass(frozen=True, slots=True)
class SqlAssetTables:
    """Handles for the four tables used by the asset backend."""

    root: Table
    entry: Table
    version: Table
    blob: Table


class SqlAlchemyAssetBackend:
    """SQL asset backend with the historical lightweight constructor."""

    @classmethod
    def register_schema(cls, registry: SqlSchemaRegistry) -> SqlAssetTables:
        tables = _build_tables(registry.metadata)
        root, entry, version, blob = tables.root, tables.entry, tables.version, tables.blob
        registry.add_table(root, owner="asset.sql")
        registry.add_table(entry, owner="asset.sql")
        registry.add_table(version, owner="asset.sql")
        registry.add_table(blob, owner="asset.sql")
        return tables

    def __init__(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
        *,
        dialect: "SqlAlchemyDialect | None" = None,
    ) -> None:
        self._sessions = session_factory
        self._tables = _build_tables(MetaData())
        self._root = _DEFAULT_ROOT
        self._dialect = dialect

    async def initialize_storage(self, engine: "AsyncEngine") -> None:
        """Create the asset tables using the backend's historical API."""
        async with engine.begin() as connection:
            await connection.run_sync(self._tables.root.metadata.create_all)
        _logger.info("SQL asset schema initialized")

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return True

    async def initialize(self) -> None:
        async with self._sessions() as session:
            async with session.begin():
                if await session.scalar(select(self._tables.root.c.id).where(self._tables.root.c.id == 1)) is None:
                    now = datetime.now(timezone.utc)
                    try:
                        async with session.begin_nested():
                            await session.execute(
                                insert(self._tables.root).values(
                                    id=1,
                                    revision=0,
                                    created_at=now,
                                    updated_at=now,
                                )
                            )
                    except IntegrityError:
                        pass
        _logger.info("sql asset backend initialized: root=%s", self._root.root_id)

    async def head_revision(self) -> AssetStoreRevision:
        async with self._sessions() as session:
            return AssetStoreRevision(str(await self._current_revision(session)))

    async def load_metadata(
        self,
        after_revision: "AssetStoreRevision | None",
    ) -> "MetadataLoad[AssetKey, AssetInfo, AssetStoreRevision]":
        async with self._sessions() as session:
            current = await self._current_revision(session)
            if after_revision is not None:
                previous = _revision_number(after_revision)
                if previous == current:
                    return MetadataLoad(MetadataLoadMode.PATCH, AssetStoreRevision(str(current)), ())
                if 0 <= previous < current:
                    rows = (
                        await session.execute(
                            select(self._tables.version)
                            .where(self._tables.version.c.revision > previous)
                            .order_by(self._tables.version.c.revision, self._tables.version.c.id)
                        )
                    ).mappings().all()
                    if rows:
                        changes = tuple(
                            MetadataChange(
                                _key_from_path(str(row["path"])),
                                await self._info_for_change(session, row, current),
                            )
                            for row in rows
                        )
                        return MetadataLoad(
                            MetadataLoadMode.PATCH,
                            AssetStoreRevision(str(current)),
                            changes,
                        )
            rows = (
                await session.execute(
                    select(self._tables.version)
                    .order_by(self._tables.version.c.path, self._tables.version.c.revision.desc(), self._tables.version.c.id.desc())
                )
            ).mappings().all()
            latest: dict[str, Mapping[str, SqlValue]] = {}
            for row in rows:
                path = str(row["path"])
                if path not in latest:
                    latest[path] = row
            changes_list: list[MetadataChange[AssetKey, AssetInfo]] = []
            for path, row in latest.items():
                changes_list.append(
                    MetadataChange(
                        _key_from_path(path),
                        await self._info_for_change(session, row, current),
                    )
                )
            return MetadataLoad(
                MetadataLoadMode.REPLACE,
                AssetStoreRevision(str(current)),
                tuple(changes_list),
            )

    async def get(self, key: AssetKey) -> "bytes | None":
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(self._tables.entry).where(
                        self._tables.entry.c.path == _path_for_key(key),
                        self._tables.entry.c.active.is_(True),
                    )
                )
            ).mappings().first()
            return None if row is None else _entry_content(row)

    async def get_many(self, keys: "Sequence[AssetKey]") -> "Mapping[AssetKey, bytes]":
        if not keys:
            return {}
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(self._tables.entry).where(
                        self._tables.entry.c.path.in_([_path_for_key(key) for key in keys]),
                        self._tables.entry.c.active.is_(True),
                    )
                )
            ).mappings().all()
            result: dict[AssetKey, bytes] = {}
            for row in rows:
                result[_key_from_path(str(row["path"]))] = _entry_content(row)
            return result

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_entry_revision: "AssetRevision | None" = None,
    ) -> "StoragePutResult[AssetInfo, AssetRevision, AssetStoreRevision]":
        async with self._sessions() as session:
            async with session.begin():
                current = await self._current_revision(session)
                previous = await self._entry(session, key)
                result = await self._put_transaction(
                    session,
                    key,
                    value,
                    expected_entry_revision,
                    current + 1,
                    previous,
                )
                if result.changed:
                    await self._reserve_revision(session, current)
        _logger.info("sql asset put: kind=%s id=%s changed=%s revision=%s", key.kind, key.id, result.changed, result.store_revision.value)
        return result

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_entry_revision: "AssetRevision | None" = None,
    ) -> "StorageDeleteResult[AssetKey, AssetRevision, AssetStoreRevision]":
        async with self._sessions() as session:
            async with session.begin():
                current = await self._current_revision(session)
                previous = await self._entry(session, key)
                result = await self._delete_transaction(
                    session,
                    key,
                    expected_entry_revision,
                    current + 1,
                    previous,
                )
                if result.deleted:
                    await self._reserve_revision(session, current)
        _logger.info("sql asset delete: kind=%s id=%s deleted=%s", key.kind, key.id, result.deleted)
        return result

    async def reset(self) -> "StorageResetResult[AssetStoreRevision]":
        async with self._sessions() as session:
            async with session.begin():
                current = await self._current_revision(session)
                rows = (
                    await session.execute(
                        select(self._tables.entry).where(self._tables.entry.c.active.is_(True))
                    )
                ).mappings().all()
                if not rows:
                    return StorageResetResult(AssetStoreRevision(str(current)), 0)
                revision = current + 1
                for row in rows:
                    await self._delete_transaction(
                        session,
                        _key_from_path(str(row["path"])),
                        None,
                        revision,
                        row,
                    )
                await self._reserve_revision(session, current)
        return StorageResetResult(AssetStoreRevision(str(revision)), len(rows))

    async def apply_batch(
        self,
        changes: "Sequence[StorageChange[AssetKey, bytes, AssetRevision]]",
        *,
        expected_store_revision: "AssetStoreRevision | None" = None,
    ) -> "StorageBatchResult[AssetInfo, AssetKey, AssetRevision, AssetStoreRevision]":
        self._validate_batch(changes)
        async with self._sessions() as session:
            async with session.begin():
                current = await self._current_revision(session)
                if expected_store_revision is not None and _revision_number(expected_store_revision) != current:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                previous = {change.key: await self._entry(session, change.key) for change in changes}
                mutates = any(_change_mutates(change, previous[change.key]) for change in changes)
                revision = current + 1 if mutates else current
                results: list[StoragePutResult[AssetInfo, AssetRevision, AssetStoreRevision] | StorageDeleteResult[AssetKey, AssetRevision, AssetStoreRevision]] = []
                for change in changes:
                    if change.operation is StorageOperation.PUT:
                        results.append(
                            await self._put_transaction(
                                session,
                                change.key,
                                change.value or b"",
                                change.expected_entry_revision,
                                revision,
                                previous[change.key],
                                current,
                            )
                        )
                    else:
                        results.append(
                            await self._delete_transaction(
                                session,
                                change.key,
                                change.expected_entry_revision,
                                revision,
                                previous[change.key],
                                current,
                            )
                        )
                if mutates:
                    await self._reserve_revision(session, current)
        return StorageBatchResult(AssetStoreRevision(str(revision)), True, tuple(results))

    async def list_versions(self, key: AssetKey) -> "tuple[VersionSummary[AssetRevision], ...]":
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(self._tables.version)
                    .where(self._tables.version.c.path == _path_for_key(key))
                    .order_by(self._tables.version.c.version.desc(), self._tables.version.c.id.desc())
                )
            ).mappings().all()
            summaries: list[VersionSummary[AssetRevision]] = []
            for row in rows:
                summaries.append(await self._version_summary(session, row))
            return tuple(summaries)

    async def get_at_revision(self, key: AssetKey, entry_revision: AssetRevision) -> "bytes | None":
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(self._tables.version)
                    .where(
                        self._tables.version.c.path == _path_for_key(key),
                        self._tables.version.c.version == entry_revision.value,
                    )
                    .order_by(self._tables.version.c.id.desc())
                )
            ).mappings().first()
            if row is None or bool(row["deleted"]):
                return None
            return await self._read_blob(session, str(row["object_id"]))

    async def get_at_version(self, key: AssetKey, version: int) -> "bytes | None":
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(self._tables.version)
                    .where(
                        self._tables.version.c.path == _path_for_key(key),
                        self._tables.version.c.version == version,
                    )
                    .order_by(self._tables.version.c.revision.desc(), self._tables.version.c.id.desc())
                )
            ).mappings().first()
            if row is None or bool(row["deleted"]):
                return None
            return await self._read_blob(session, str(row["object_id"]))

    async def _put_transaction(
        self,
        session: AsyncSession,
        key: AssetKey,
        value: bytes,
        expected: "AssetRevision | None",
        store_revision: int,
        previous: "Mapping[str, SqlValue] | None",
        previous_store_revision: int | None = None,
    ) -> "StoragePutResult[AssetInfo, AssetRevision, AssetStoreRevision]":
        previous_info = None if previous is None else _info_from_entry(
            previous,
            self._root,
            AssetStoreRevision(str(store_revision - 1 if previous_store_revision is None else previous_store_revision)),
        )
        _check_entry_revision(previous_info, expected)
        etag = _etag(value)
        if previous_info is not None and not previous_info.deleted and previous_info.etag == etag:
            return StoragePutResult(previous_info, previous_info.entry_revision, previous_info.store_revision, False)
        entry_revision = AssetRevision(0 if previous_info is None else previous_info.entry_revision.value + 1)
        now = datetime.now(timezone.utc)
        info = AssetInfo(key, entry_revision, AssetStoreRevision(str(store_revision)), etag, len(value), False, self._root.root_id, self._root.digest, now)
        await self._write_blob(session, etag, value, now)
        await self._write_entry(session, info, value, now)
        await self._write_change(session, info, etag, False, now)
        return StoragePutResult(info, info.entry_revision, info.store_revision, True)

    async def _delete_transaction(
        self,
        session: AsyncSession,
        key: AssetKey,
        expected: "AssetRevision | None",
        store_revision: int,
        previous: "Mapping[str, SqlValue] | None",
        previous_store_revision: int | None = None,
    ) -> "StorageDeleteResult[AssetKey, AssetRevision, AssetStoreRevision]":
        previous_info = None if previous is None else _info_from_entry(
            previous,
            self._root,
            AssetStoreRevision(str(store_revision - 1 if previous_store_revision is None else previous_store_revision)),
        )
        _check_entry_revision(previous_info, expected)
        if previous_info is None or previous_info.deleted:
            return StorageDeleteResult(
                key,
                False,
                None,
                AssetStoreRevision(str(store_revision - 1 if previous_store_revision is None else previous_store_revision)),
            )
        entry_revision = AssetRevision(previous_info.entry_revision.value + 1)
        now = datetime.now(timezone.utc)
        info = AssetInfo(key, entry_revision, AssetStoreRevision(str(store_revision)), _TOMBSTONE_ETAG, 0, True, self._root.root_id, self._root.digest, now)
        await self._write_entry(session, info, b"", now)
        await self._write_change(session, info, None, True, now)
        return StorageDeleteResult(key, True, entry_revision, info.store_revision)

    async def _entry(self, session: AsyncSession, key: AssetKey) -> "Mapping[str, SqlValue] | None":
        return (
            await session.execute(
                select(self._tables.entry).where(self._tables.entry.c.path == _path_for_key(key))
            )
        ).mappings().first()

    async def _current_revision(self, session: AsyncSession) -> int:
        value = await session.scalar(
            select(self._tables.root.c.revision).where(self._tables.root.c.id == 1)
        )
        return int(value or 0)

    async def _reserve_revision(self, session: AsyncSession, expected: int) -> int:
        now = datetime.now(timezone.utc)
        result = await session.execute(
            update(self._tables.root)
            .where(
                self._tables.root.c.id == 1,
                self._tables.root.c.revision == expected,
            )
            .values(revision=expected + 1, updated_at=now)
        )
        if result.rowcount != 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return expected + 1

    async def _write_entry(self, session: AsyncSession, info: AssetInfo, value: bytes, now: datetime) -> None:
        values = {
            "path": _path_for_key(info.key),
            "kind": info.key.kind,
            "version": info.entry_revision.value,
            "etag": info.etag,
            "active": not info.deleted,
            "content": value,
            "updated_at": now,
            "created_at": now,
        }
        await self._dialect_for(session).upsert(
            session,
            table=self._tables.entry,
            values=values,
            set_values={key: item for key, item in values.items() if key != "created_at" and key != "path"},
            index_elements=("path",),
        )

    async def _write_blob(self, session: AsyncSession, digest: str, value: bytes, now: datetime) -> None:
        await self._dialect_for(session).insert_ignore_conflict(
            session,
            table=self._tables.blob,
            values={"sha256": digest, "content": value, "created_at": now, "updated_at": now},
            index_elements=("sha256",),
        )

    async def _write_change(self, session: AsyncSession, info: AssetInfo, object_id: "str | None", deleted: bool, now: datetime) -> None:
        await session.execute(
            insert(self._tables.version).values(
                revision=_revision_number(info.store_revision),
                path=_path_for_key(info.key),
                kind=info.key.kind,
                version=info.entry_revision.value,
                etag=info.etag,
                object_id=object_id,
                active=not deleted,
                deleted=deleted,
                created_at=now,
                updated_at=now,
            )
        )

    async def _read_blob(self, session: AsyncSession, digest: str) -> "bytes | None":
        row = (await session.execute(select(self._tables.blob.c.content).where(self._tables.blob.c.sha256 == digest))).first()
        if row is None:
            return None
        content = row[0]
        if not isinstance(content, bytes) or _etag(content) != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return content

    async def _info_for_change(self, session: AsyncSession, row: Mapping[str, SqlValue], store_revision: int) -> AssetInfo:
        key = _key_from_path(str(row["path"]))
        if bool(row["deleted"]):
            return _info_from_change(row, self._root, AssetStoreRevision(str(row["revision"])))
        entry = await self._entry(session, key)
        if entry is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _info_from_entry(entry, self._root, AssetStoreRevision(str(row["revision"])))

    async def _version_summary(self, session: AsyncSession, row: Mapping[str, SqlValue]) -> VersionSummary[AssetRevision]:
        digest = str(row["etag"] or _TOMBSTONE_ETAG)
        size = 0
        if not bool(row["deleted"]):
            content = await self._read_blob(session, str(row["object_id"]))
            if content is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            size = len(content)
        return VersionSummary(AssetRevision(int(row["version"] or 0)), digest, size, _as_utc(row["created_at"]), bool(row["deleted"]))

    def _dialect_for(self, session: AsyncSession) -> SqlAlchemyDialect:
        if self._dialect is None:
            self._dialect = resolve_dialect(session)
        return self._dialect

    def _validate_batch(self, changes: "Sequence[StorageChange[AssetKey, bytes, AssetRevision]]") -> None:
        seen: set[AssetKey] = set()
        for change in changes:
            if change.key in seen:
                raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
            seen.add(change.key)


def _check_entry_revision(previous: "AssetInfo | None", expected: "AssetRevision | None") -> None:
    if expected is not None and (previous is None or previous.entry_revision != expected):
        raise AIError(ErrorCode.ASSET_REVISION_CONFLICT)


def _change_mutates(change: StorageChange[AssetKey, bytes, AssetRevision], previous: "Mapping[str, SqlValue] | None") -> bool:
    if change.operation is StorageOperation.DELETE:
        return previous is not None and bool(previous["active"])
    return previous is None or not bool(previous["active"]) or str(previous["etag"]) != _etag(change.value or b"")


def _path_for_key(key: AssetKey) -> str:
    return f"{key.kind}/{key.id}"


def _key_from_path(path: str) -> AssetKey:
    kind, separator, asset_id = path.partition("/")
    if not separator or not kind or not asset_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AssetKey(kind, asset_id)


def _info_from_entry(row: Mapping[str, SqlValue], root: AssetRoot, store_revision: AssetStoreRevision) -> AssetInfo:
    deleted = not bool(row["active"])
    etag = _TOMBSTONE_ETAG if deleted else str(row["etag"])
    content = row["content"]
    if not isinstance(content, bytes):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if deleted:
        if content:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        size = 0
    else:
        if _etag(content) != etag:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        size = len(content)
    return AssetInfo(
        _key_from_path(str(row["path"])),
        AssetRevision(int(row["version"])),
        store_revision,
        etag,
        size,
        deleted,
        root.root_id,
        root.digest,
        _as_utc(row["updated_at"]),
    )


def _entry_content(row: Mapping[str, SqlValue]) -> bytes:
    content = row["content"]
    if not isinstance(content, bytes) or _etag(content) != str(row["etag"]):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return content


def _info_from_change(row: Mapping[str, SqlValue], root: AssetRoot, store_revision: AssetStoreRevision) -> AssetInfo:
    deleted = bool(row["deleted"])
    return AssetInfo(
        _key_from_path(str(row["path"])),
        AssetRevision(int(row["version"] or 0)),
        store_revision,
        _TOMBSTONE_ETAG if deleted else str(row["etag"]),
        0,
        deleted,
        root.root_id,
        root.digest,
        _as_utc(row["updated_at"]),
    )


def _as_utc(value: SqlValue) -> datetime:
    if not isinstance(value, datetime):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _etag(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _revision_number(revision: AssetStoreRevision) -> int:
    try:
        value = int(revision.value)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if value < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _build_tables(metadata: "MetaData") -> SqlAssetTables:
    integer_id = BigInteger().with_variant(Integer, "sqlite")
    root_name = storage_name("asset_revision")
    entry_name = storage_name("asset_contents")
    version_name = storage_name("asset_changes")
    blob_name = storage_name("asset_blobs")
    root = Table(
        root_name,
        metadata,
        Column("id", integer_id, primary_key=True, autoincrement=True),
        Column("revision", Integer, nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Index(storage_name("asset_revision_ix_revision"), "revision"),
        Index(storage_name("asset_revision_ix_updated_at"), "updated_at"),
        Index(storage_name("asset_revision_ix_created_at"), "created_at"),
    )
    entry = Table(
        entry_name,
        metadata,
        Column("id", integer_id, primary_key=True, autoincrement=True),
        Column("path", String(512), nullable=False),
        Column("kind", String(128), nullable=False),
        Column("version", Integer, nullable=False),
        Column("etag", String(255), nullable=False),
        Column("active", Boolean, nullable=False),
        Column("content", LargeBinary, nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        UniqueConstraint("path", name=storage_name("asset_contents_uk_path")),
        Index(storage_name("asset_contents_ix_kind"), "kind"),
        Index(storage_name("asset_contents_ix_updated_at"), "updated_at"),
        Index(storage_name("asset_contents_ix_created_at"), "created_at"),
    )
    version = Table(
        version_name,
        metadata,
        Column("id", integer_id, primary_key=True, autoincrement=True),
        Column("revision", Integer, nullable=False),
        Column("path", String(512), nullable=False),
        Column("kind", String(128), nullable=True),
        Column("version", Integer, nullable=True),
        Column("etag", String(255), nullable=True),
        Column("object_id", String(128), nullable=True),
        Column("active", Boolean, nullable=True),
        Column("deleted", Boolean, nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        UniqueConstraint("revision", "path", name=storage_name("asset_changes_uk_revision_path")),
        Index(storage_name("asset_changes_ix_object_id"), "object_id"),
        Index(storage_name("asset_changes_ix_updated_at"), "updated_at"),
        Index(storage_name("asset_changes_ix_created_at"), "created_at"),
    )
    blob = Table(
        blob_name,
        metadata,
        Column("id", integer_id, primary_key=True, autoincrement=True),
        Column("sha256", String(64), nullable=False),
        Column("content", LargeBinary, nullable=False),
        Column("updated_at", DateTime(timezone=True), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        UniqueConstraint("sha256", name=storage_name("asset_blobs_uk_sha256")),
        Index(storage_name("asset_blobs_ix_updated_at"), "updated_at"),
        Index(storage_name("asset_blobs_ix_created_at"), "created_at"),
    )
    return SqlAssetTables(root, entry, version, blob)


_DEFAULT_ROOT = AssetRoot(
    "sql:asset",
    "sql",
    "asset",
    hashlib.sha256(b"sql:asset").hexdigest(),
)


__all__ = ["SqlAlchemyAssetBackend", "SqlAssetTables"]
