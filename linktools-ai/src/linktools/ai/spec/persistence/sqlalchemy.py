#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLAlchemy specification persistence.

A single ``load_metadata`` call returns the head revision plus either the full
entry set (REPLACE) or the change log since the caller's revision (PATCH),
read from a consistent snapshot in one SQL statement. Metadata queries never
project the ``content`` column; only ``get``/``get_many`` read content."""


from typing import TYPE_CHECKING
from sqlalchemy import Boolean, Integer, LargeBinary, String, delete, select, true, update
from sqlalchemy.orm import Mapped, mapped_column
from ...errors import SpecConflictError, StorageConflictError, StorageCorruptionError
from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.blob import put_blob, read_blob
from ...storage.sqlalchemy.conventions import TABLE_PREFIX, as_utc, timestamp_indexes
from ...storage.sqlalchemy.dialects import resolve_dialect
from ...storage.versioning import VersionedStorage, VersionSummary
from ...storage.revision import MetadataLoad, MetadataLoadMode, StorageChange, StorageMetadataBackend
from ..document import SpecDocument, SpecDocumentInfo

if TYPE_CHECKING:
    from ...storage.sqlalchemy.dialects import SqlAlchemyDialect
    from sqlalchemy.ext.asyncio import AsyncEngine


class EntryRow(Base):
    __tablename__ = f"{TABLE_PREFIX}spec_documents"
    path: "Mapped[str]" = mapped_column(String(512), unique=True)
    kind: "Mapped[str]" = mapped_column(String(128), index=True)
    version: "Mapped[int]" = mapped_column(Integer)
    etag: "Mapped[str]" = mapped_column(String(255))
    active: "Mapped[bool]" = mapped_column(Boolean, default=True)
    content: "Mapped[bytes]" = mapped_column(LargeBinary)


class SpecBlobRow(Base):
    """Content-addressed spec document blob. Identical content (same sha256)
    shares one row -- dedup. Version history rows (``ChangeRow``) point here via
    ``object_id``; a blob is never deleted, so every historical version is
    retrievable."""

    __tablename__ = f"{TABLE_PREFIX}spec_blobs"
    __table_args__ = (*timestamp_indexes(),)
    sha256: "Mapped[str]" = mapped_column(String(64), unique=True)
    content: "Mapped[bytes]" = mapped_column(LargeBinary)


class RevisionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}spec_revision"
    revision: "Mapped[int]" = mapped_column(Integer, default=0)
    minimum_delta_revision: "Mapped[int]" = mapped_column(Integer, default=0)


class ChangeRow(Base):
    __tablename__ = f"{TABLE_PREFIX}spec_changes"
    revision: "Mapped[int]" = mapped_column(Integer, index=True)
    path: "Mapped[str]" = mapped_column(String(512))
    kind: "Mapped[str | None]" = mapped_column(String(128), nullable=True)
    version: "Mapped[int | None]" = mapped_column(Integer, nullable=True)
    etag: "Mapped[str | None]" = mapped_column(String(255), nullable=True)
    object_id: "Mapped[str | None]" = mapped_column(String(128), nullable=True, index=True)
    active: "Mapped[bool | None]" = mapped_column(Boolean, nullable=True)
    deleted: "Mapped[bool]" = mapped_column(Boolean, default=False)


def _info(row: "EntryRow | ChangeRow") -> SpecDocumentInfo:
    return SpecDocumentInfo(row.path, row.kind, row.version, row.etag, row.active)


class SqlAlchemySpecBackend(
    StorageMetadataBackend[int, str, SpecDocumentInfo],
    VersionedStorage[int, str, SpecDocument],
):
    def __init__(self, session_factory, *, dialect: "SqlAlchemyDialect | None" = None) -> None:
        self.session_factory = session_factory
        self._dialect = dialect

    async def initialize_storage(self, engine: "AsyncEngine") -> None:
        # No singleton is seeded here: the revision counter row is self-seeded
        # on the first write via ``upsert_increment``.
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    # ---- reader --------------------------------------------------------

    async def get(self, path: str) -> "SpecDocument | None":
        async with self.session_factory() as session:
            row = await session.scalar(select(EntryRow).where(EntryRow.path == path))
            return None if row is None else SpecDocument(_info(row), row.content)

    async def get_many(self, paths: "tuple[str, ...]") -> "dict[str, SpecDocument]":
        if not paths:
            return {}
        async with self.session_factory() as session:
            rows = (
                await session.scalars(select(EntryRow).where(EntryRow.path.in_(paths)))
            ).all()
            return {row.path: SpecDocument(_info(row), row.content) for row in rows}

    async def stat(self, path: str) -> "SpecDocumentInfo | None":
        async with self.session_factory() as session:
            row = (
                await session.execute(
                    _metadata_query().where(EntryRow.path == path)
                )
            ).first()
            return None if row is None else _metadata_info(row)

    async def list_info(self, *, kind: "str | None" = None) -> "tuple[SpecDocumentInfo, ...]":
        async with self.session_factory() as session:
            rows = await session.execute(_metadata_query(kind=kind))
            return tuple(_metadata_info(row) for row in rows)

    async def list_versions(self, path: str) -> "tuple[VersionSummary, ...]":
        # The ChangeRow history for one path, newest first. Each row is one
        # version (a put recorded deleted=False + object_id; a delete recorded
        # deleted=True + object_id=None). Projects metadata + object_id only;
        # no content read.
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(ChangeRow)
                    .where(ChangeRow.path == path)
                    .order_by(ChangeRow.revision.desc(), ChangeRow.id.desc())
                )
            ).all()
            return tuple(
                VersionSummary(
                    revision=row.revision,
                    etag=row.etag,
                    object_id=row.object_id,
                    created_at=as_utc(row.created_at),
                    deleted=row.deleted,
                )
                for row in rows
            )

    async def get_at_revision(self, path: str, revision: int) -> "SpecDocument | None":
        # The version of ``path`` in effect at ``revision``: the ChangeRow with
        # the largest revision <= the given one. A tombstone (deleted=True), no
        # row, or a null object_id means the path had no content at that
        # revision -> None.
        async with self.session_factory() as session:
            row = await session.scalar(
                select(ChangeRow)
                .where(ChangeRow.path == path, ChangeRow.revision <= revision)
                .order_by(ChangeRow.revision.desc(), ChangeRow.id.desc())
                .limit(1)
            )
            if row is None or row.deleted or row.object_id is None:
                return None
            content = await read_blob(session, SpecBlobRow, row.object_id)
            if content is None:
                # object_id references a missing blob -- corruption (every
                # ChangeRow is written in the same transaction as its blob, so
                # this is unreachable under normal operation).
                raise StorageCorruptionError(
                    f"spec blob {row.object_id} for {path!r}@{revision} is missing"
                )
            return SpecDocument(
                SpecDocumentInfo(row.path, row.kind, row.version, row.etag, row.active),
                content,
            )

    # ---- metadata backend ---------------------------------------------

    async def load_metadata(
        self,
        after_revision: "int | None",
    ) -> "MetadataLoad[int, str, SpecDocumentInfo]":
        # Each call issues exactly one SQL: the singleton RevisionRow (head +
        # minimum) LEFT JOINed with the data rows, so the returned revision and
        # entries/changes come from a single consistent read.
        async with self.session_factory() as session:
            if after_revision is None:
                return await self._load_snapshot(session)
            return await self._load_after(session, after_revision)

    async def _load_snapshot(
        self, session
    ) -> "MetadataLoad[int, str, SpecDocumentInfo]":
        # One SQL: RevisionRow LEFT JOIN Entry metadata. Every entry row joins
        # onto the single revision row, so each result row carries head + one
        # entry's metadata. An empty Entry table still yields one row (head +
        # NULL entry) via the LEFT JOIN, so head is returned with no documents.
        query = (
            select(RevisionRow.revision, *_metadata_expr(EntryRow))
            .select_from(RevisionRow)
            .outerjoin(EntryRow, true())
            .order_by(EntryRow.path)
        )
        rows = (await session.execute(query)).all()
        if not rows:
            return MetadataLoad(0, MetadataLoadMode.REPLACE, ())
        head = rows[0][0]
        changes = tuple(
            StorageChange(row.path, _metadata_info(row))
            for row in rows
            if row.path is not None
        )
        return MetadataLoad(head, MetadataLoadMode.REPLACE, changes)

    async def _load_after(
        self, session, after: int
    ) -> "MetadataLoad[int, str, SpecDocumentInfo]":
        # One SQL: RevisionRow (head + minimum) LEFT JOIN ChangeRow restricted to
        # revisions in (after, head]. head, minimum, and the change set come from
        # a single read; minimum decides whether the caller's after is too old to
        # patch (then REPLACE) or served as PATCH (or empty PATCH at head).
        query = (
            select(RevisionRow.revision, RevisionRow.minimum_delta_revision, ChangeRow)
            .select_from(RevisionRow)
            .outerjoin(
                ChangeRow,
                (ChangeRow.revision > after) & (ChangeRow.revision <= RevisionRow.revision),
            )
            .order_by(ChangeRow.revision, ChangeRow.path)
        )
        rows = (await session.execute(query)).all()
        if not rows:
            return MetadataLoad(after, MetadataLoadMode.PATCH, ())
        head = rows[0][0]
        minimum = rows[0][1]
        # after > head or after < minimum -> the caller is too new or too old to
        # patch; fall back to a full snapshot. (after < minimum means the change
        # history before `minimum` was compacted by a reset.)
        if after > head or after < minimum:
            return await self._load_snapshot(session)
        changes = tuple(
            StorageChange(
                change.path,
                None
                if change.deleted
                else SpecDocumentInfo(change.path, change.kind, change.version, change.etag, change.active),
            )
            for _, _, change in rows
            if change is not None and change.path is not None
        )
        return MetadataLoad(head, MetadataLoadMode.PATCH, changes)

    async def head_revision(self) -> int:
        # One SQL: the singleton RevisionRow's head only (no JOIN, no change
        # set). 0 when the counter row is not yet seeded (matches the empty
        # table branch of ``_load_snapshot``); never touches EntryRow/ChangeRow.
        async with self.session_factory() as session:
            row = await session.scalar(
                select(RevisionRow.revision).where(RevisionRow.id == 1)
            )
            return 0 if row is None else row

    # ---- writer --------------------------------------------------------

    async def put(self, entry: SpecDocument) -> SpecDocument:
        entry.validate_etag()
        values = _entry_values(entry)
        # Lock-free insert-or-update: try an atomic conflict-aware INSERT first
        # (the same insert_ignore_conflict primitive put_blob/_next_revision use);
        # only fall back to an UPDATE when a row already exists. The retry loop
        # covers the narrow window where a concurrent delete removes the row
        # between our failed insert attempt and the fallback UPDATE (that UPDATE
        # then matches zero rows -- retry the whole insert-or-update from scratch).
        for _ in range(3):
            async with self.session_factory() as session:
                async with session.begin():
                    dialect = await self._dialect_for(session)
                    revision = await self._next_revision(session)
                    object_id = await put_blob(session, dialect, SpecBlobRow, entry.content)
                    result = await dialect.insert_ignore_conflict(
                        session,
                        model=EntryRow,
                        values={"path": entry.info.path, **values},
                        index_elements=("path",),
                    )
                    if not result.inserted:
                        updated = await session.execute(
                            update(EntryRow)
                            .where(EntryRow.path == entry.info.path)
                            .values(**values)
                        )
                        if updated.rowcount == 0:
                            continue
                    session.add(
                        _change_row(revision, entry.info, deleted=False, object_id=object_id)
                    )
            return entry
        raise StorageConflictError(
            f"spec put for {entry.info.path!r} did not converge after retries"
        )

    async def delete(self, path: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                row = (
                    await session.execute(
                        _metadata_query().where(EntryRow.path == path)
                    )
                ).first()
                if row is None:
                    return
                tombstone = _metadata_info(row)
                revision = await self._next_revision(session)
                await session.execute(delete(EntryRow).where(EntryRow.path == path))
                session.add(_change_row(revision, tombstone, deleted=True, object_id=None))

    async def reset(self, entries: "tuple[SpecDocument, ...]") -> None:
        for entry in entries:
            entry.validate_etag()
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                paths = [entry.info.path for entry in entries]
                if len(set(paths)) != len(paths):
                    raise SpecConflictError("reset received duplicate spec paths")
                old_rows = (
                    await session.execute(_metadata_query())
                ).all()
                old = {row.path: _metadata_info(row) for row in old_rows}
                new = {entry.info.path: entry for entry in entries}
                revision = await self._next_revision(session)
                for path in sorted(set(old) | set(new)):
                    before = old.get(path)
                    doc = new.get(path)
                    if before is not None and doc is None:
                        await session.execute(
                            delete(EntryRow).where(EntryRow.path == path)
                        )
                        session.add(_change_row(revision, before, deleted=True, object_id=None))
                    elif doc is not None and before != doc.info:
                        object_id = await put_blob(session, dialect, SpecBlobRow, doc.content)
                        values = _entry_values(doc)
                        existing = await session.scalar(
                            select(EntryRow).where(EntryRow.path == path)
                        )
                        if existing is None:
                            session.add(EntryRow(path=path, **values))
                        else:
                            for key, value in values.items():
                                setattr(existing, key, value)
                        session.add(
                            _change_row(revision, doc.info, deleted=False, object_id=object_id)
                        )
                    # unchanged: leave the row (and its content) untouched.
                # Reset raises the incremental window's lower bound
                # (minimum_delta_revision) so readers below it take a full
                # REPLACE snapshot rather than replay history. History itself is
                # retained permanently: ChangeRow is the complete change +
                # version log (audit/rollback), not a truncatable patch window;
                # the minimum watermark distinguishes the two concerns.
                await session.execute(
                    RevisionRow.__table__.update()
                    .where(RevisionRow.id == 1)
                    .values(minimum_delta_revision=revision)
                )

    async def _next_revision(self, session) -> int:
        dialect = await self._dialect_for(session)
        return await dialect.upsert_increment(
            session,
            model=RevisionRow,
            pk=1,
            column="revision",
        )

    async def _dialect_for(self, session) -> "SqlAlchemyDialect":
        if self._dialect is None:
            self._dialect = resolve_dialect(session)
        return self._dialect


def _entry_values(entry: SpecDocument) -> "dict[str, object]":
    return _entry_values_by_info(entry.info) | {"content": entry.content}


def _entry_values_by_info(info: SpecDocumentInfo) -> "dict[str, object]":
    return {
        "kind": info.kind,
        "version": info.version,
        "etag": info.etag,
        "active": info.active,
    }


def _metadata_expr(model):
    # Project only metadata columns; never the LargeBinary content.
    return (getattr(model, "path"), getattr(model, "kind"), getattr(model, "version"), getattr(model, "etag"), getattr(model, "active"))


def _metadata_query(*, kind: "str | None" = None):
    query = select(*_metadata_expr(EntryRow)).order_by(EntryRow.path)
    if kind is not None:
        query = query.where(EntryRow.kind == kind)
    return query


def _metadata_info(row) -> SpecDocumentInfo:
    return SpecDocumentInfo(row.path, row.kind, row.version, row.etag, row.active)


def _change_row(
    revision: int,
    info: SpecDocumentInfo,
    *,
    deleted: bool,
    object_id: "str | None",
) -> ChangeRow:
    return ChangeRow(
        revision=revision,
        path=info.path,
        kind=info.kind,
        version=info.version,
        etag=info.etag,
        object_id=object_id,
        active=info.active,
        deleted=deleted,
    )


__all__ = ["EntryRow", "SpecBlobRow", "SqlAlchemySpecBackend", "ChangeRow", "RevisionRow"]
