#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyArtifactRecordStore: the SQL-backed ArtifactRecordStore.

Stores ArtifactRecord METADATA only -- the content blob is intentionally out of
scope (it lives on the filesystem via FilesystemArtifactBlobStore; a row here
never holds bytes). The store uses the caller-provided AsyncSession: a
``session_factory`` for standalone use, or a shared ``session`` so it can
participate in the same UnitOfWork as the other SQL stores. The create-only
INSERT detects a primary-key conflict through the dialect strategy (SQLite /
PostgreSQL ``ON CONFLICT DO NOTHING``; MySQL SAVEPOINT) so it is portable across
the supported dialects. Record serialization goes through the public codec
(:func:`record_to_jsonable` / :func:`record_from_jsonable`) so the JSON shape is
owned in one place."""

import json
from typing import AsyncIterator, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...artifact.models import ArtifactRecord
from ...artifact.store import record_from_jsonable, record_to_jsonable
from ...errors import ArtifactRecordConflictError
from .dialects import SqlAlchemyDialectStrategy, resolve_dialect_strategy
from .models import ArtifactRecordRow


def _row_to_record(row: ArtifactRecordRow) -> ArtifactRecord:
    return record_from_jsonable(json.loads(row.data_json))


class SqlAlchemyArtifactRecordStore:
    """ArtifactRecordStore backed by SQLAlchemy. The record's content blob is
    out of scope (metadata only); compose with a FilesystemArtifactBlobStore for
    the content-addressed bytes. ``session_factory`` for standalone use;
    ``session`` for UoW participation (shared with the other SQL stores).

    Records are create-only: an INSERT that hits an existing primary key is
    reconciled -- byte-identical content is idempotent, any field change raises
    :class:`ArtifactRecordConflictError`. There is no UPDATE path; the lineage
    of a prior write can never be overwritten."""

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
        strategy: "SqlAlchemyDialectStrategy | None" = None,
    ) -> None:
        self._session_factory = session_factory
        self._session = session
        self._strategy = strategy or resolve_dialect_strategy(session_factory)

    async def _run(self, action):
        if self._session is not None:
            return await action(self._session)
        async with self._session_factory() as session:
            result = await action(session)
            await session.commit()
            return result

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        payload = json.dumps(record_to_jsonable(record))

        async def _action(session: AsyncSession) -> ArtifactRecord:
            # INSERT first. The dialect strategy runs a conflict-detecting
            # insert (SQLite/PostgreSQL ON CONFLICT DO NOTHING; MySQL SAVEPOINT)
            # so a concurrent same-id insert is absorbed without poisoning the
            # session: a reported conflict -> the row exists, read it and
            # reconcile (idempotent on identical content, conflict on a
            # different value).
            conflict = await self._strategy.execute_conflict_insert(
                session,
                ArtifactRecordRow,
                {
                    "artifact_id": record.ref.id,
                    "tenant_id": record.tenant_id,
                    "sha256": record.ref.sha256,
                    "producer_kind": record.provenance.producer_kind,
                    "producer_id": record.provenance.producer_id or None,
                    "run_id": record.provenance.run_id,
                    "data_json": payload,
                },
                index_elements=["artifact_id"],
            )
            if conflict:
                existing = await session.get(ArtifactRecordRow, record.ref.id)
                if existing is None:
                    # Conflict vanished after the no-op insert -- only possible
                    # if another writer deleted the row mid-flight. Fail closed.
                    raise ArtifactRecordConflictError(
                        f"artifact {record.ref.id} insert conflicted but the row is absent"
                    )
                return self._reconcile_row(existing, payload, record.ref.id)
            return record

        return await self._run(_action)

    def _reconcile_row(
        self, existing: ArtifactRecordRow, payload: str, artifact_id: str
    ) -> ArtifactRecord:
        if existing.data_json != payload:
            raise ArtifactRecordConflictError(
                f"artifact {artifact_id} already exists with different content"
            )
        return record_from_jsonable(json.loads(existing.data_json))

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

    async def is_digest_referenced(self, digest: str) -> bool:
        """Whether any record pins ``digest`` (across tenants). A single-row
        existence probe -- the orphan sweeper calls this under the per-digest
        lock so its delete decision reflects the current reference set, not a
        snapshot taken before the lock."""
        async def _action(session: AsyncSession) -> bool:
            result = await session.execute(
                select(ArtifactRecordRow.sha256)
                .where(ArtifactRecordRow.sha256 == digest)
                .limit(1)
            )
            return result.first() is not None

        return await self._run(_action)

    async def iter_by_run_id(
        self, run_id: str, *, tenant_id: "str | None" = None
    ) -> AsyncIterator[ArtifactRecord]:
        """Parent/provenance index: yield every record produced
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
        """Parent/provenance index: yield every record from a given
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
