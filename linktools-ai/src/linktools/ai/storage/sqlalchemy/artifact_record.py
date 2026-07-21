#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyArtifactRecordStore: the SQL-backed ArtifactRecordStore.

Stores ArtifactRecord METADATA only -- the content blob is intentionally out of
scope (it lives on the filesystem via FilesystemArtifactBlobStore; a row here
never holds bytes). The store uses the caller-provided AsyncSession: a
``session_factory`` for standalone use, or a shared ``session`` so it can
participate in the same UnitOfWork as the other SQL stores. It holds no engine
and branches on no dialect. Record serialization goes through the public codec
(:func:`record_to_jsonable` / :func:`record_from_jsonable`) so the JSON shape is
owned in one place."""

import json
from typing import AsyncIterator, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...artifact.models import ArtifactRecord
from ...artifact.store import record_from_jsonable, record_to_jsonable
from .models import ArtifactRecordRow


def _row_to_record(row: ArtifactRecordRow) -> ArtifactRecord:
    return record_from_jsonable(json.loads(row.data_json))


class SqlAlchemyArtifactRecordStore:
    """ArtifactRecordStore backed by SQLAlchemy. The record's content blob is
    out of scope (metadata only); compose with a FilesystemArtifactBlobStore for
    the content-addressed bytes. ``session_factory`` for standalone use;
    ``session`` for UoW participation (shared with the other SQL stores)."""

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        self._session = session

    async def _run(self, action):
        if self._session is not None:
            result = await action(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            result = await action(session)
            await session.commit()
            return result

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        payload = json.dumps(record_to_jsonable(record))

        async def _action(session: AsyncSession) -> None:
            existing = await session.get(ArtifactRecordRow, record.ref.id)
            if existing is None:
                session.add(
                    ArtifactRecordRow(
                        artifact_id=record.ref.id,
                        tenant_id=record.tenant_id,
                        sha256=record.ref.sha256,
                        producer_kind=record.provenance.producer_kind,
                        producer_id=record.provenance.producer_id or None,
                        run_id=record.provenance.run_id,
                        data_json=payload,
                    )
                )
            elif existing.tenant_id != record.tenant_id:
                # Cross-tenant collision on the same id: refuse to re-home the
                # row to a different tenant (defense-in-depth -- the facade mints
                # fresh UUID ids so this is unreachable in normal use, but a
                # foreign tenant that knows an id must not overwrite another
                # tenant's record).
                raise ValueError(
                    f"artifact {record.ref.id} already belongs to tenant "
                    f"{existing.tenant_id!r}; cannot re-home to "
                    f"{record.tenant_id!r}"
                )
            else:
                existing.sha256 = record.ref.sha256
                existing.producer_kind = record.provenance.producer_kind
                existing.producer_id = record.provenance.producer_id or None
                existing.run_id = record.provenance.run_id
                existing.data_json = payload

        await self._run(_action)
        return record

    async def get(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None":
        async def _action(session: AsyncSession) -> "ArtifactRecord | None":
            row = await session.get(ArtifactRecordRow, artifact_id)
            if row is None or row.tenant_id != tenant_id:
                return None
            return _row_to_record(row)

        return await self._run(_action)

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        async def _action(session: AsyncSession) -> bool:
            row = await session.get(ArtifactRecordRow, artifact_id)
            if row is None or row.tenant_id != tenant_id:
                return False
            await session.delete(row)
            return True

        return await self._run(_action)

    async def iter_referenced_digests(self) -> AsyncIterator[str]:
        """Yield every sha256 referenced by some record (across tenants), for
        orphan sweeping -- the set of blobs that are NOT orphans."""
        async def _action(session: AsyncSession) -> "list[str]":
            rows = await session.execute(select(ArtifactRecordRow.sha256))
            return list(rows.scalars().all())

        digests = await self._run(_action)
        for digest in digests:
            yield digest

    async def iter_by_run_id(
        self, run_id: str, *, tenant_id: "str | None" = None
    ) -> AsyncIterator[ArtifactRecord]:
        """Parent/provenance index (plan §4.3): yield every record produced
        under ``run_id``, optionally tenant-scoped. Uses the indexed run_id
        column (no JSON scan)."""
        async def _action(session: AsyncSession) -> "list[ArtifactRecord]":
            stmt = select(ArtifactRecordRow).where(
                ArtifactRecordRow.run_id == run_id
            )
            if tenant_id is not None:
                stmt = stmt.where(ArtifactRecordRow.tenant_id == tenant_id)
            rows = await session.execute(stmt)
            return [_row_to_record(r) for r in rows.scalars().all()]

        for record in await self._run(_action):
            yield record

    async def iter_by_producer(
        self,
        producer_kind: str,
        producer_id: "str | None" = None,
        *,
        tenant_id: "str | None" = None,
    ) -> AsyncIterator[ArtifactRecord]:
        """Parent/provenance index (plan §4.3): yield every record from a given
        producer (kind [+ id]), optionally tenant-scoped. Uses the indexed
        (producer_kind, producer_id) column."""
        async def _action(session: AsyncSession) -> "list[ArtifactRecord]":
            stmt = select(ArtifactRecordRow).where(
                ArtifactRecordRow.producer_kind == producer_kind
            )
            if producer_id is not None:
                stmt = stmt.where(ArtifactRecordRow.producer_id == producer_id)
            if tenant_id is not None:
                stmt = stmt.where(ArtifactRecordRow.tenant_id == tenant_id)
            rows = await session.execute(stmt)
            return [_row_to_record(r) for r in rows.scalars().all()]

        for record in await self._run(_action):
            yield record


__all__: "list[str]" = ["SqlAlchemyArtifactRecordStore"]
