"""SQLAlchemy specification persistence."""

from sqlalchemy import Boolean, Integer, LargeBinary, String, UniqueConstraint, delete, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import TABLE_PREFIX
from ..document import SpecDocument, SpecDocumentChange, SpecDocumentInfo


class EntryRow(Base):
    __tablename__ = f"{TABLE_PREFIX}spec_documents"
    path: Mapped[str] = mapped_column(String(512), unique=True)
    kind: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    etag: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    content: Mapped[bytes] = mapped_column(LargeBinary)


class RevisionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}spec_revision"
    revision: Mapped[int] = mapped_column(Integer, default=0)


class ChangeRow(Base):
    __tablename__ = f"{TABLE_PREFIX}spec_changes"
    __table_args__ = (
        UniqueConstraint("revision", "path", name="uq_spec_change_revision_path"),
    )
    revision: Mapped[int] = mapped_column(Integer)
    path: Mapped[str] = mapped_column(String(512))
    kind: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    etag: Mapped[str | None] = mapped_column(String(255), nullable=True)
    active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


def _info(row: EntryRow | ChangeRow) -> SpecDocumentInfo:
    return SpecDocumentInfo(row.path, row.kind, row.version, row.etag, row.active)


class SqlAlchemySpecBackend:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
            exists = await connection.scalar(
                select(RevisionRow.id).where(RevisionRow.id == 1)
            )
            if exists is None:
                await connection.execute(
                    insert(RevisionRow).values(id=1, revision=0)
                )

    async def get(self, path: str) -> SpecDocument | None:
        async with self.session_factory() as session:
            row = await session.scalar(select(EntryRow).where(EntryRow.path == path))
            return None if row is None else SpecDocument(_info(row), row.content)

    async def get_many(
        self,
        paths: tuple[str, ...],
    ) -> dict[str, SpecDocument]:
        if not paths:
            return {}
        async with self.session_factory() as session:
            rows = (
                await session.scalars(
                    select(EntryRow).where(EntryRow.path.in_(paths))
                )
            ).all()
            return {
                row.path: SpecDocument(_info(row), row.content)
                for row in rows
            }

    async def stat(self, path: str) -> SpecDocumentInfo | None:
        async with self.session_factory() as session:
            row = await session.scalar(select(EntryRow).where(EntryRow.path == path))
            return None if row is None else _info(row)

    async def list_info(self, *, kind: str | None = None) -> tuple[SpecDocumentInfo, ...]:
        async with self.session_factory() as session:
            query = select(EntryRow).order_by(EntryRow.path)
            if kind is not None:
                query = query.where(EntryRow.kind == kind)
            rows = (await session.scalars(query)).all()
            return tuple(_info(row) for row in rows)

    async def current_revision(self) -> int:
        async with self.session_factory() as session:
            row = await session.get(RevisionRow, 1)
            return 0 if row is None else row.revision

    async def _next_revision(self, session: AsyncSession) -> int:
        result = await session.execute(
            update(RevisionRow)
            .where(RevisionRow.id == 1)
            .values(revision=RevisionRow.revision + 1)
        )
        if result.rowcount != 1:
            raise RuntimeError("spec storage is not initialized")
        revision = await session.scalar(
            select(RevisionRow.revision).where(RevisionRow.id == 1)
        )
        return revision

    @staticmethod
    def _change(revision: int, entry: SpecDocument, *, deleted: bool = False) -> ChangeRow:
        info = entry.info
        return ChangeRow(
            revision=revision,
            path=info.path,
            kind=info.kind,
            version=info.version,
            etag=info.etag,
            active=info.active,
            deleted=deleted,
        )

    async def put(self, entry: SpecDocument) -> SpecDocument:
        for attempt in range(2):
            try:
                await self._put(entry)
                return entry
            except IntegrityError:
                if attempt:
                    raise
        raise AssertionError("unreachable")

    async def _put(self, entry: SpecDocument) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.scalar(
                    select(EntryRow)
                    .where(EntryRow.path == entry.info.path)
                    .with_for_update()
                )
                values = {"kind": entry.info.kind, "version": entry.info.version, "etag": entry.info.etag, "active": entry.info.active, "content": entry.content}
                if row is None:
                    session.add(EntryRow(path=entry.info.path, **values))
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
                revision = await self._next_revision(session)
                session.add(self._change(revision, entry))
                await session.flush()

    async def delete(self, path: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.scalar(
                    select(EntryRow).where(EntryRow.path == path).with_for_update()
                )
                if row is None:
                    return
                entry = SpecDocument(_info(row), row.content)
                await session.delete(row)
                revision = await self._next_revision(session)
                session.add(self._change(revision, entry, deleted=True))

    async def reset(self, entries: tuple[SpecDocument, ...]) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                old = (await session.scalars(select(EntryRow))).all()
                old_entries = {row.path: SpecDocument(_info(row), row.content) for row in old}
                new_entries = {entry.info.path: entry for entry in entries}
                await session.execute(delete(EntryRow))
                revision = await self._next_revision(session)
                for path in sorted(set(old_entries) | set(new_entries)):
                    before, after = old_entries.get(path), new_entries.get(path)
                    if before is not None and after is None:
                        session.add(self._change(revision, before, deleted=True))
                    elif after is not None and (before is None or before.info != after.info):
                        session.add(self._change(revision, after))
                for entry in new_entries.values():
                    session.add(EntryRow(path=entry.info.path, kind=entry.info.kind, version=entry.info.version, etag=entry.info.etag, active=entry.info.active, content=entry.content))

    async def list_changes(self, *, after_revision: int, through_revision: int) -> tuple[SpecDocumentChange, ...]:
        async with self.session_factory() as session:
            query = select(ChangeRow).where(ChangeRow.revision > after_revision, ChangeRow.revision <= through_revision).order_by(ChangeRow.revision, ChangeRow.path)
            rows = (await session.scalars(query)).all()
            return tuple(SpecDocumentChange(row.revision, row.path, None if row.deleted else SpecDocumentInfo(row.path, row.kind, row.version, row.etag, row.active)) for row in rows)


__all__ = ["Base", "SqlAlchemySpecBackend"]
