"""SQLAlchemy capability persistence."""

from sqlalchemy import Boolean, Integer, LargeBinary, String, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import BIGSERIAL, TABLE_PREFIX
from ..entries import CapabilityEntry, CapabilityEntryChange, CapabilityEntryInfo


class EntryRow(Base):
    __tablename__ = f"{TABLE_PREFIX}capability_entries"
    id: Mapped[int] = mapped_column(BIGSERIAL, nullable=True)
    path: Mapped[str] = mapped_column(String(512), primary_key=True)
    kind: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer)
    etag: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    content: Mapped[bytes] = mapped_column(LargeBinary)


class RevisionRow(Base):
    __tablename__ = f"{TABLE_PREFIX}capability_revision"
    revision: Mapped[int] = mapped_column(Integer, default=0)


class ChangeRow(Base):
    __tablename__ = f"{TABLE_PREFIX}capability_changes"
    id: Mapped[int] = mapped_column(BIGSERIAL, nullable=True)
    revision: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(String(512), primary_key=True)
    kind: Mapped[str | None] = mapped_column(String(128), nullable=True)
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    etag: Mapped[str | None] = mapped_column(String(255), nullable=True)
    active: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    deleted: Mapped[bool] = mapped_column(Boolean, default=False)


def _info(row: EntryRow | ChangeRow) -> CapabilityEntryInfo:
    return CapabilityEntryInfo(row.path, row.kind, row.version, row.etag, row.active)


class SqlAlchemyCapabilityStore:
    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def get(self, path: str) -> CapabilityEntry | None:
        async with self.session_factory() as session:
            row = await session.get(EntryRow, path)
            return None if row is None else CapabilityEntry(_info(row), row.content)

    async def stat(self, path: str) -> CapabilityEntryInfo | None:
        async with self.session_factory() as session:
            row = await session.get(EntryRow, path)
            return None if row is None else _info(row)

    async def list_info(self, *, kind: str | None = None) -> tuple[CapabilityEntryInfo, ...]:
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
        row = await session.get(RevisionRow, 1, with_for_update=True)
        if row is None:
            row = RevisionRow(id=1, revision=1)
            session.add(row)
            return 1
        row.revision += 1
        return row.revision

    @staticmethod
    def _change(revision: int, entry: CapabilityEntry, *, deleted: bool = False) -> ChangeRow:
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

    async def put(self, entry: CapabilityEntry) -> CapabilityEntry:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(EntryRow, entry.info.path, with_for_update=True)
                values = {"kind": entry.info.kind, "version": entry.info.version, "etag": entry.info.etag, "active": entry.info.active, "content": entry.content}
                if row is None:
                    session.add(EntryRow(path=entry.info.path, **values))
                else:
                    for key, value in values.items():
                        setattr(row, key, value)
                revision = await self._next_revision(session)
                session.add(self._change(revision, entry))
        return entry

    async def delete(self, path: str) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                row = await session.get(EntryRow, path, with_for_update=True)
                if row is None:
                    return
                entry = CapabilityEntry(_info(row), row.content)
                await session.delete(row)
                revision = await self._next_revision(session)
                session.add(self._change(revision, entry, deleted=True))

    async def reset(self, entries: tuple[CapabilityEntry, ...]) -> None:
        async with self.session_factory() as session:
            async with session.begin():
                old = (await session.scalars(select(EntryRow))).all()
                await session.execute(delete(EntryRow))
                revision = await self._next_revision(session)
                for row in old:
                    session.add(self._change(revision, CapabilityEntry(_info(row), row.content), deleted=True))
                for entry in entries:
                    session.add(EntryRow(path=entry.info.path, kind=entry.info.kind, version=entry.info.version, etag=entry.info.etag, active=entry.info.active, content=entry.content))
                    session.add(self._change(revision, entry))

    async def list_changes(self, *, after_revision: int, through_revision: int) -> tuple[CapabilityEntryChange, ...]:
        async with self.session_factory() as session:
            query = select(ChangeRow).where(ChangeRow.revision > after_revision, ChangeRow.revision <= through_revision).order_by(ChangeRow.revision, ChangeRow.path)
            rows = (await session.scalars(query)).all()
            return tuple(CapabilityEntryChange(row.revision, row.path, None if row.deleted else CapabilityEntryInfo(row.path, row.kind, row.version, row.etag, row.active)) for row in rows)


__all__ = ["Base", "SqlAlchemyCapabilityStore"]
