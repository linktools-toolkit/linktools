#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SQLAlchemy specification persistence.

A single ``load_metadata`` call returns the head revision plus either the full
entry set (REPLACE) or the change log since the caller's revision (PATCH),
read from a consistent snapshot in one SQL statement. Metadata queries never
project the ``content`` column; only ``get``/``get_many`` read content."""

from typing import TYPE_CHECKING
from sqlalchemy import Boolean, Integer, LargeBinary, String, delete, select, true
from sqlalchemy.orm import Mapped, mapped_column
from ...errors import SpecConflictError, StorageCorruptionError
from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.blob import put_blob, put_blobs, read_blob
from ...storage.sqlalchemy.conventions import TABLE_PREFIX, as_utc, timestamp_indexes
from ...storage.sqlalchemy.dialects import resolve_dialect
from ...storage.versioning import VersionedStorage, VersionSummary
from ...storage.revision import (
    MetadataLoad,
    MetadataLoadMode,
    StorageChange,
    StorageMetadataBackend,
)
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
    object_id: "Mapped[str | None]" = mapped_column(
        String(128), nullable=True, index=True
    )
    active: "Mapped[bool | None]" = mapped_column(Boolean, nullable=True)
    deleted: "Mapped[bool]" = mapped_column(Boolean, default=False)


def _info(row: "EntryRow | ChangeRow") -> SpecDocumentInfo:
    return SpecDocumentInfo(row.path, row.kind, row.version, row.etag, row.active)


class SqlAlchemySpecBackend(
    StorageMetadataBackend[int, str, SpecDocumentInfo],
    VersionedStorage[int, str, SpecDocument],
):
    def __init__(
        self, session_factory, *, dialect: "SqlAlchemyDialect | None" = None
    ) -> None:
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
                await session.execute(_metadata_query().where(EntryRow.path == path))
            ).first()
            return None if row is None else _metadata_info(row)

    async def list_info(
        self, *, kind: "str | None" = None
    ) -> "tuple[SpecDocumentInfo, ...]":
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
                    version=row.version,
                    etag=row.etag,
                    object_id=row.object_id,
                    created_at=as_utc(row.created_at),
                    deleted=row.deleted,
                )
                for row in rows
            )

    async def get_at_version(self, path: str, version: int) -> "SpecDocument | None":
        # The declared version number is not unique to one history row (e.g.
        # a tombstone reuses the deleted entry's version), so pick the most
        # recent match. No content at that point (deleted, or no such
        # version at all) -> None, same as get_at_revision.
        async with self.session_factory() as session:
            row = await session.scalar(
                select(ChangeRow)
                .where(ChangeRow.path == path, ChangeRow.version == version)
                .order_by(ChangeRow.revision.desc(), ChangeRow.id.desc())
                .limit(1)
            )
            if row is None or row.deleted or row.object_id is None:
                return None
            content = await read_blob(session, SpecBlobRow, row.object_id)
            if content is None:
                raise StorageCorruptionError(
                    f"spec blob {row.object_id} for {path!r}@version={version} is missing"
                )
            return SpecDocument(
                SpecDocumentInfo(row.path, row.kind, row.version, row.etag, row.active),
                content,
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
        self, session, *, head: "int | None" = None
    ) -> "MetadataLoad[int, str, SpecDocumentInfo]":
        # When ``head`` is already known (the _load_after fallback already read it
        # in its own JOIN), skip the RevisionRow join and read EntryRow metadata
        # directly -- avoids re-reading the singleton the caller already has.
        # Otherwise (the top-level after_revision=None path) join RevisionRow to
        # fetch head alongside the entry set in one statement.
        if head is not None:
            rows = (
                await session.execute(_metadata_query().order_by(EntryRow.path))
            ).all()
            changes = tuple(
                StorageChange(row.path, _metadata_info(row))
                for row in rows
                if row.path is not None
            )
            return MetadataLoad(head, MetadataLoadMode.REPLACE, changes)
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
                (ChangeRow.revision > after)
                & (ChangeRow.revision <= RevisionRow.revision),
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
            return await self._load_snapshot(session, head=head)
        changes = tuple(
            StorageChange(
                change.path,
                None
                if change.deleted
                else SpecDocumentInfo(
                    change.path, change.kind, change.version, change.etag, change.active
                ),
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
        # One transaction, one EntryRow statement: a dialect upsert
        # (INSERT ... ON CONFLICT(path) DO UPDATE / ON DUPLICATE KEY UPDATE)
        # inserts a new row or overwrites an existing one keyed by path, with no
        # separate read and no retry loop. revision bump, content-addressed
        # blob, and the version-history row land in the same transaction, so a
        # put is atomic -- all four writes commit together or none do.
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                revision = await self._next_revision(session)
                object_id = await put_blob(session, dialect, SpecBlobRow, entry.content)
                await dialect.upsert(
                    session,
                    model=EntryRow,
                    values={"path": entry.info.path, **values},
                    set_values=values,
                    index_elements=("path",),
                )
                session.add(
                    _change_row(
                        revision, entry.info, deleted=False, object_id=object_id
                    )
                )
        return entry

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
                session.add(
                    _change_row(revision, tombstone, deleted=True, object_id=None)
                )

    async def reset(self, entries: "tuple[SpecDocument, ...]") -> None:
        for entry in entries:
            entry.validate_etag()
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                paths = [entry.info.path for entry in entries]
                if len(set(paths)) != len(paths):
                    raise SpecConflictError("reset received duplicate spec paths")
                old_rows = (await session.execute(_metadata_query())).all()
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
                        session.add(
                            _change_row(revision, before, deleted=True, object_id=None)
                        )
                    elif doc is not None and before != doc.info:
                        object_id = await put_blob(
                            session, dialect, SpecBlobRow, doc.content
                        )
                        values = _entry_values(doc)
                        # Lock-free insert-or-update, same shape as put(): one
                        # dialect upsert keyed on path, no SELECT existence check.
                        await dialect.upsert(
                            session,
                            model=EntryRow,
                            values={"path": path, **values},
                            set_values=values,
                            index_elements=("path",),
                        )
                        session.add(
                            _change_row(
                                revision, doc.info, deleted=False, object_id=object_id
                            )
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

    async def apply_batch(
        self,
        puts: "tuple[SpecDocument, ...]",
        deletes: "tuple[str, ...]",
    ) -> None:
        """Apply a mixed set of puts and deletes atomically: one transaction,
        one shared revision, other documents untouched. Distinct from
        :meth:`reset` (full replacement, which deletes every unlisted path).

        For N puts + M deletes: one multi-row blob insert-ignore (puts), one
        multi-row upsert (puts), one bulk metadata read + one bulk DELETE
        (deletes), and the ChangeRows flushed once -- ~4 statements regardless
        of N+M, vs N×4 + M×4 if each op went through ``put``/``delete``
        separately. History appends (``minimum_delta_revision`` is not touched,
        matching single put/delete).

        The revision is bumped ONLY when something actually changed: puts
        always change; a delete changes only when the path existed. A batch
        whose every op is a no-op (empty inputs, or deletes of nonexistent
        paths) advances no revision and writes no ChangeRows -- matching single
        ``delete``'s no-op behavior."""
        for entry in puts:
            entry.validate_etag()
        put_paths = [entry.info.path for entry in puts]
        if len(set(put_paths)) != len(put_paths):
            raise SpecConflictError("apply_batch received duplicate put paths")
        delete_paths = [p for p in deletes if p not in set(put_paths)]
        async with self.session_factory() as session:
            async with session.begin():
                dialect = await self._dialect_for(session)
                # Puts: write blobs + upsert entries (no revision needed yet).
                object_ids: "list[str]" = []
                if puts:
                    object_ids = await put_blobs(
                        session,
                        dialect,
                        SpecBlobRow,
                        [entry.content for entry in puts],
                    )
                    await dialect.upsert_many(
                        session,
                        model=EntryRow,
                        rows=[
                            {"path": entry.info.path, **_entry_values(entry)}
                            for entry in puts
                        ],
                        set_columns=["kind", "version", "etag", "active", "content"],
                        index_elements=("path",),
                    )
                # Deletes: read tombstones for existing rows, then bulk delete.
                tombstones: "dict[str, SpecDocumentInfo]" = {}
                if delete_paths:
                    tombstone_rows = (
                        await session.execute(
                            _metadata_query().where(EntryRow.path.in_(delete_paths))
                        )
                    ).all()
                    tombstones = {
                        row.path: _metadata_info(row) for row in tombstone_rows
                    }
                    if tombstones:
                        await session.execute(
                            delete(EntryRow).where(EntryRow.path.in_(list(tombstones)))
                        )
                # Bump the revision ONLY when something actually changed: puts
                # always do; a delete does only when the path existed.
                if not puts and not tombstones:
                    return
                revision = await self._next_revision(session)
                change_rows: "list[ChangeRow]" = [
                    _change_row(revision, entry.info, deleted=False, object_id=oid)
                    for entry, oid in zip(puts, object_ids)
                ]
                change_rows.extend(
                    _change_row(
                        revision, tombstones[path], deleted=True, object_id=None
                    )
                    for path in delete_paths
                    if path in tombstones
                )
                session.add_all(change_rows)

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


def _metadata_expr(model: "type[EntryRow]"):
    # Project only metadata columns; never the LargeBinary content.
    return (model.path, model.kind, model.version, model.etag, model.active)


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


__all__ = [
    "EntryRow",
    "SpecBlobRow",
    "SqlAlchemySpecBackend",
    "ChangeRow",
    "RevisionRow",
]
