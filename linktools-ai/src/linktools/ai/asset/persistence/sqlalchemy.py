#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLAlchemy asset backend with content-addressed history."""

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from sqlalchemy import Boolean, Integer, LargeBinary, String, delete, select, true
from sqlalchemy.orm import Mapped, mapped_column
from linktools.core import environ

from ...foundation.errors import AssetConflictError, StorageCorruptionError
from ...storage.multi import BatchStorageWriter, StorageWriter
from ...storage.revision import MetadataLoad, MetadataLoadMode, StorageChange, StorageMetadataBackend
from ...storage.sql.base import Base
from ...storage.sql.blob import put_blob, put_blobs, read_blob
from ...storage.sql.conventions import TABLE_PREFIX, as_utc, timestamp_indexes
from ...storage.sql.dialects import SqlAlchemyDialect, resolve_dialect
from ...storage.versioning import VersionSummary, VersionedStorage
from ..content import AssetContent, AssetContentInfo

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

logger = environ.get_logger("ai.asset.persistence.sqlalchemy")


class AssetEntryRow(Base):
    __tablename__ = f"{TABLE_PREFIX}asset_contents"

    path: "Mapped[str]" = mapped_column(String(512), unique=True)
    kind: "Mapped[str]" = mapped_column(String(128), index=True)
    version: "Mapped[int]" = mapped_column(Integer)
    etag: "Mapped[str]" = mapped_column(String(255))
    active: "Mapped[bool]" = mapped_column(Boolean, default=True)
    content: "Mapped[bytes]" = mapped_column(LargeBinary)


class AssetBlobRow(Base):
    __tablename__ = f"{TABLE_PREFIX}asset_blobs"
    __table_args__ = timestamp_indexes()

    sha256: "Mapped[str]" = mapped_column(String(64), unique=True)
    content: "Mapped[bytes]" = mapped_column(LargeBinary)


class AssetRevisionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}asset_revision"

    revision: "Mapped[int]" = mapped_column(Integer, default=0)
    minimum_delta_revision: "Mapped[int]" = mapped_column(Integer, default=0)


class AssetChangeRow(Base):
    __tablename__ = f"{TABLE_PREFIX}asset_changes"

    revision: "Mapped[int]" = mapped_column(Integer, index=True)
    path: "Mapped[str]" = mapped_column(String(512))
    kind: "Mapped[str | None]" = mapped_column(String(128), nullable=True)
    version: "Mapped[int | None]" = mapped_column(Integer, nullable=True)
    etag: "Mapped[str | None]" = mapped_column(String(255), nullable=True)
    object_id: "Mapped[str | None]" = mapped_column(String(128), nullable=True, index=True)
    active: "Mapped[bool | None]" = mapped_column(Boolean, nullable=True)
    deleted: "Mapped[bool]" = mapped_column(Boolean, default=False)


class SqlAlchemyAssetBackend(
    StorageMetadataBackend[int, str, AssetContentInfo],
    StorageWriter[str, AssetContent, int],
    BatchStorageWriter[str, AssetContent, int],
    VersionedStorage[int, str, AssetContent],
):
    def __init__(self, session_factory: Any, *, dialect: "SqlAlchemyDialect | None" = None) -> None:
        self.session_factory = session_factory
        self._dialect = dialect

    async def initialize_storage(self, engine: "AsyncEngine") -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def get(self, path: str) -> "AssetContent | None":
        async with self.session_factory() as session:
            row = await session.scalar(select(AssetEntryRow).where(AssetEntryRow.path == path))
            return None if row is None else _content(row)

    async def get_many(self, paths: "tuple[str, ...]") -> "dict[str, AssetContent]":
        if not paths:
            return {}
        async with self.session_factory() as session:
            rows = (await session.scalars(select(AssetEntryRow).where(AssetEntryRow.path.in_(paths)))).all()
            return {row.path: _content(row) for row in rows}

    async def stat(self, path: str) -> "AssetContentInfo | None":
        async with self.session_factory() as session:
            row = (await session.execute(_metadata_query().where(AssetEntryRow.path == path))).first()
            return None if row is None else _metadata_info(row)

    async def list_info(self, *, kind: "str | None" = None) -> "tuple[AssetContentInfo, ...]":
        async with self.session_factory() as session:
            rows = await session.execute(_metadata_query(kind=kind))
            return tuple(_metadata_info(row) for row in rows)

    async def head_revision(self) -> int:
        async with self.session_factory() as session:
            revision = await session.scalar(select(AssetRevisionRow.revision).where(AssetRevisionRow.id == 1))
            return 0 if revision is None else revision

    async def load_metadata(self, after_revision: "int | None") -> "MetadataLoad[int, str, AssetContentInfo]":
        logger.debug("loading SQL asset metadata: after_revision=%s", after_revision)
        async with self.session_factory() as session:
            if after_revision is None:
                return await _load_snapshot(session)
            query = (
                select(AssetRevisionRow.revision, AssetRevisionRow.minimum_delta_revision, AssetChangeRow)
                .select_from(AssetRevisionRow)
                .outerjoin(AssetChangeRow, (AssetChangeRow.revision > after_revision) & (AssetChangeRow.revision <= AssetRevisionRow.revision))
                .order_by(AssetChangeRow.revision, AssetChangeRow.path)
            )
            rows = (await session.execute(query)).all()
            if not rows:
                return MetadataLoad(after_revision, MetadataLoadMode.PATCH, ())
            head, minimum = rows[0][0], rows[0][1]
            if after_revision > head or after_revision < minimum:
                return await _load_snapshot(session, head=head)
            changes = tuple(
                StorageChange(
                    change.path,
                    None if change.deleted else AssetContentInfo(
                        change.path, change.kind, change.version, change.etag, change.active
                    ),
                )
                for _, _, change in rows
                if change is not None
            )
            return MetadataLoad(head, MetadataLoadMode.PATCH, changes)

    async def put(self, entry: AssetContent) -> "tuple[AssetContent, int]":
        entry.validate_etag()
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                revision = await self._next_revision(session)
                object_id = await put_blob(session, dialect, AssetBlobRow, entry.content)
                values = _entry_values(entry)
                await dialect.upsert(
                    session,
                    model=AssetEntryRow,
                    values={"path": entry.info.path, **values},
                    set_values=values,
                    index_elements=("path",),
                )
                session.add(_change_row(revision, entry.info, deleted=False, object_id=object_id))
        logger.info("stored SQL asset: path=%s revision=%s", entry.info.path, revision)
        return entry, revision

    async def delete(self, path: str) -> "int | None":
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                rows = await dialect.delete_returning(
                    session,
                    model=AssetEntryRow,
                    where=AssetEntryRow.path == path,
                    returning=("path", "kind", "version", "etag", "active"),
                )
                if not rows:
                    return None
                revision = await self._next_revision(session)
                session.add(_change_row(revision, _metadata_info(rows[0]), deleted=True, object_id=None))
        logger.info("deleted SQL asset: path=%s revision=%s", path, revision)
        return revision

    async def reset(self, entries: "tuple[AssetContent, ...]") -> "int | None":
        for entry in entries:
            entry.validate_etag()
        if len({entry.info.path for entry in entries}) != len(entries):
            raise AssetConflictError("reset received duplicate asset paths")
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                existing = {row.path: _metadata_info(row) for row in (await session.execute(_metadata_query())).all()}
                desired = {entry.info.path: entry for entry in entries}
                puts = [desired[path] for path in sorted(desired) if path not in existing or desired[path].info != existing[path]]
                deleted = [path for path in sorted(existing) if path not in desired]
                if not puts and not deleted:
                    return None
                revision = await self._next_revision(session)
                object_ids = await put_blobs(session, dialect, AssetBlobRow, [entry.content for entry in puts])
                if puts:
                    await dialect.upsert_many(
                        session,
                        model=AssetEntryRow,
                        rows=[{"path": entry.info.path, **_entry_values(entry)} for entry in puts],
                        set_columns=("kind", "version", "etag", "active", "content"),
                        index_elements=("path",),
                    )
                if deleted:
                    await session.execute(delete(AssetEntryRow).where(AssetEntryRow.path.in_(deleted)))
                session.add_all([
                    *(_change_row(revision, entry.info, deleted=False, object_id=object_id) for entry, object_id in zip(puts, object_ids)),
                    *(_change_row(revision, existing[path], deleted=True, object_id=None) for path in deleted),
                ])
                await session.execute(AssetRevisionRow.__table__.update().where(AssetRevisionRow.id == 1).values(minimum_delta_revision=revision))
        logger.info("reset SQL assets: count=%s revision=%s", len(entries), revision)
        return revision

    async def apply_batch(self, puts: "tuple[AssetContent, ...]", deletes: "tuple[str, ...]") -> "int | None":
        for entry in puts:
            entry.validate_etag()
        if len({entry.info.path for entry in puts}) != len(puts):
            raise AssetConflictError("apply_batch received duplicate put paths")
        put_paths = {entry.info.path for entry in puts}
        delete_paths = [path for path in deletes if path not in put_paths]
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                tombstone_rows = []
                if delete_paths:
                    tombstone_rows = (await session.execute(_metadata_query().where(AssetEntryRow.path.in_(delete_paths)))).all()
                tombstones = {row.path: _metadata_info(row) for row in tombstone_rows}
                if not puts and not tombstones:
                    return None
                revision = await self._next_revision(session)
                object_ids = await put_blobs(session, dialect, AssetBlobRow, [entry.content for entry in puts])
                if puts:
                    await dialect.upsert_many(
                        session,
                        model=AssetEntryRow,
                        rows=[{"path": entry.info.path, **_entry_values(entry)} for entry in puts],
                        set_columns=("kind", "version", "etag", "active", "content"),
                        index_elements=("path",),
                    )
                if tombstones:
                    await session.execute(delete(AssetEntryRow).where(AssetEntryRow.path.in_(tuple(tombstones))))
                session.add_all([
                    *(_change_row(revision, entry.info, deleted=False, object_id=object_id) for entry, object_id in zip(puts, object_ids)),
                    *(_change_row(revision, tombstones[path], deleted=True, object_id=None) for path in delete_paths if path in tombstones),
                ])
        logger.info("applied SQL asset batch: puts=%s deletes=%s revision=%s", len(puts), len(tombstones), revision)
        return revision

    async def list_versions(self, path: str) -> "tuple[VersionSummary, ...]":
        async with self.session_factory() as session:
            rows = (await session.scalars(select(AssetChangeRow).where(AssetChangeRow.path == path).order_by(AssetChangeRow.revision.desc(), AssetChangeRow.id.desc()))).all()
            return tuple(_version_summary(row) for row in rows)

    async def get_at_revision(self, path: str, revision: int) -> "AssetContent | None":
        async with self.session_factory() as session:
            row = await session.scalar(select(AssetChangeRow).where(AssetChangeRow.path == path, AssetChangeRow.revision <= revision).order_by(AssetChangeRow.revision.desc(), AssetChangeRow.id.desc()).limit(1))
            return await _content_at(session, row, f"{path!r}@{revision}")

    async def get_at_version(self, path: str, version: int) -> "AssetContent | None":
        async with self.session_factory() as session:
            row = await session.scalar(select(AssetChangeRow).where(AssetChangeRow.path == path, AssetChangeRow.version == version).order_by(AssetChangeRow.revision.desc(), AssetChangeRow.id.desc()).limit(1))
            return await _content_at(session, row, f"{path!r}@version={version}")

    async def _next_revision(self, session: Any) -> int:
        return await (await self._dialect_for(session)).upsert_increment(session, model=AssetRevisionRow, pk=1, column="revision")

    async def _dialect_for(self, session: Any) -> SqlAlchemyDialect:
        if self._dialect is None:
            self._dialect = resolve_dialect(session)
        return self._dialect


async def _load_snapshot(session: Any, *, head: "int | None" = None) -> "MetadataLoad[int, str, AssetContentInfo]":
    if head is None:
        query = select(AssetRevisionRow.revision, AssetEntryRow.path, AssetEntryRow.kind, AssetEntryRow.version, AssetEntryRow.etag, AssetEntryRow.active).select_from(AssetRevisionRow).outerjoin(AssetEntryRow, true()).order_by(AssetEntryRow.path)
        rows = (await session.execute(query)).all()
        if not rows:
            return MetadataLoad(0, MetadataLoadMode.REPLACE, ())
        head = rows[0][0]
        changes = tuple(StorageChange(row.path, _metadata_info(row)) for row in rows if row.path is not None)
    else:
        rows = (await session.execute(_metadata_query())).all()
        changes = tuple(StorageChange(row.path, _metadata_info(row)) for row in rows)
    return MetadataLoad(head, MetadataLoadMode.REPLACE, changes)


def _metadata_query(*, kind: "str | None" = None) -> Any:
    query = select(AssetEntryRow.path, AssetEntryRow.kind, AssetEntryRow.version, AssetEntryRow.etag, AssetEntryRow.active)
    if kind is not None:
        query = query.where(AssetEntryRow.kind == kind)
    return query.order_by(AssetEntryRow.path)


def _metadata_info(row: Any) -> AssetContentInfo:
    return AssetContentInfo(row.path, row.kind, row.version, row.etag, row.active)


def _content(row: AssetEntryRow) -> AssetContent:
    return AssetContent(_metadata_info(row), row.content)


def _entry_values(entry: AssetContent) -> dict[str, object]:
    return {"kind": entry.info.kind, "version": entry.info.version, "etag": entry.info.etag, "active": entry.info.active, "content": entry.content}


def _change_row(revision: int, info: AssetContentInfo, *, deleted: bool, object_id: "str | None") -> AssetChangeRow:
    return AssetChangeRow(revision=revision, path=info.path, kind=info.kind, version=info.version, etag=info.etag, object_id=object_id, active=info.active, deleted=deleted)


def _version_summary(row: AssetChangeRow) -> VersionSummary:
    created_at = as_utc(row.created_at) or datetime.fromtimestamp(0, tz=timezone.utc)
    return VersionSummary(row.revision, row.version, row.etag, row.object_id, created_at, row.deleted)


async def _content_at(session: Any, row: "AssetChangeRow | None", label: str) -> "AssetContent | None":
    if row is None or row.deleted or row.object_id is None:
        return None
    content = await read_blob(session, AssetBlobRow, row.object_id)
    if content is None:
        raise StorageCorruptionError(f"asset blob {row.object_id} for {label} is missing")
    return AssetContent(AssetContentInfo(row.path, row.kind, row.version, row.etag, row.active), content)


__all__ = ["AssetChangeRow", "AssetEntryRow", "AssetRevisionRow", "AssetBlobRow", "SqlAlchemyAssetBackend"]
