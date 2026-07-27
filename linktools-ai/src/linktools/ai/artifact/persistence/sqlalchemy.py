#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyArtifactRecordStore: the SQL-backed ArtifactRecordStore.

Stores ArtifactRecord METADATA only -- the content blob is intentionally out of
scope (it lives on the filesystem via FilesystemArtifactBlobStore; a row here
never holds bytes). The store uses the caller-provided AsyncSession: a
``session_factory`` for standalone use, or a shared ``session`` so it can
participate in the same UnitOfWork as the other SQL stores. The create-only
INSERT detects a conflict on ``artifact_id`` through the injected
:class:`~linktools.ai.storage.sqlalchemy.dialects.SqlAlchemyDialect`'s
``insert_ignore_conflict`` -- the same seam the storage kernel's object
backend uses, so this store needs no vendor-specific SQL of its own. A
SAVEPOINT-based recovery would poison UoW rollback under aiosqlite (which
commits savepoints immediately), so the conflict-detecting insert (rather
than INSERT + catch IntegrityError) is mandatory for every dialect.
Record serialization goes through the public codec
(:func:`record_to_jsonable` / :func:`record_from_jsonable`) so the JSON shape is
owned in one place."""

import json
from typing import TYPE_CHECKING, AsyncIterator, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..digest import ArtifactDigest
from ..models import ArtifactRecord
from ..store import record_from_jsonable, record_to_jsonable
from ...errors import ArtifactRecordConflictError
from linktools.ai.storage.sqlalchemy.models import ArtifactRecordRow

if TYPE_CHECKING:
    from ...storage.features import ComponentCapabilities
    from ...storage.sqlalchemy.dialects import SqlAlchemyDialect


def _row_to_record(row: ArtifactRecordRow) -> ArtifactRecord:
    return record_from_jsonable(json.loads(row.data_json))


class SqlAlchemyArtifactRecordStore:
    """ArtifactRecordStore backed by SQLAlchemy. The record's content blob is
    out of scope (metadata only); compose with a FilesystemArtifactBlobStore for
    the content-addressed bytes. ``session_factory`` for standalone use;
    ``session`` for UoW participation (shared with the other SQL stores).

    Records are create-only: an INSERT that hits an existing artifact_id is
    reconciled -- byte-identical content is idempotent, any field change raises
    :class:`ArtifactRecordConflictError`. There is no UPDATE path; the lineage
    of a prior write can never be overwritten."""

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
        dialect: "SqlAlchemyDialect | None" = None,
    ) -> None:
        self._session_factory = session_factory
        self._session = session
        # dialect may be left None: _dialect_for() then auto-detects it from
        # an open session's bound engine on first use. A caller wanting a
        # vendor with no built-in (or a test double) passes its own dialect
        # in instead.
        self._dialect = dialect

    def _dialect_for(self, session: AsyncSession) -> "SqlAlchemyDialect":
        if self._dialect is not None:
            return self._dialect
        from ...storage.sqlalchemy.dialects import resolve_dialect

        return resolve_dialect(session)

    @property
    def capabilities(self) -> "ComponentCapabilities":
        # transaction_participation: shares the UoW AsyncSession (session=...),
        #   so a Run + ArtifactRecord written in one tx commit/roll back together.
        # optimistic_concurrency: the create-only INSERT + conflict reconciliation
        #   is the CAS-equivalent contract (a field change is rejected, not
        #   silently overwritten).
        # idempotency: byte-identical re-put is a no-op (reconciled).
        from ...storage.features import ComponentCapabilities

        return ComponentCapabilities(
            transaction_participation=True,
            optimistic_concurrency=True,
            idempotency=True,
            append_only=False,
        )

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
            # INSERT first. The dialect's ignore-conflict insert absorbs a
            # concurrent same-artifact_id insert without poisoning the session
            # (a SAVEPOINT-based recovery would break UoW rollback under
            # aiosqlite). A reported conflict -> the row exists, read it and
            # reconcile (idempotent on identical content, conflict on a
            # different value).
            insert_outcome = await self._dialect_for(session).insert_ignore_conflict(
                session,
                model=ArtifactRecordRow,
                values={
                    "artifact_id": record.ref.id,
                    "tenant_id": record.tenant_id,
                    "content_hash": record.ref.sha256,
                    "producer_kind": record.provenance.producer_kind,
                    "producer_id": record.provenance.producer_id or None,
                    "run_id": record.provenance.run_id,
                    "data_json": payload,
                },
                index_elements=["artifact_id"],
            )
            if not insert_outcome.inserted:
                existing = (
                    await session.execute(
                        select(ArtifactRecordRow).where(
                            ArtifactRecordRow.artifact_id == record.ref.id
                        )
                    )
                ).scalar_one_or_none()
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
            row = (
                await session.execute(
                    select(ArtifactRecordRow).where(
                        ArtifactRecordRow.artifact_id == artifact_id
                    )
                )
            ).scalar_one_or_none()
            if row is None or row.tenant_id != tenant_id:
                return None
            return _row_to_record(row)

        return await self._run(_action)

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        async def _action(session: AsyncSession) -> bool:
            row = (
                await session.execute(
                    select(ArtifactRecordRow).where(
                        ArtifactRecordRow.artifact_id == artifact_id
                    )
                )
            ).scalar_one_or_none()
            if row is None or row.tenant_id != tenant_id:
                return False
            await session.delete(row)
            return True

        return await self._run(_action)

    async def iter_referenced_digests(self) -> AsyncIterator[str]:
        """Yield every sha256 referenced by some record (across tenants), for
        orphan sweeping -- the set of blobs that are NOT orphans."""
        async def _action(session: AsyncSession) -> "list[str]":
            rows = await session.execute(select(ArtifactRecordRow.content_hash))
            return list(rows.scalars().all())

        digests = await self._run(_action)
        for digest in digests:
            yield digest

    async def is_digest_referenced(self, digest: ArtifactDigest) -> bool:
        """Whether any record pins ``digest`` (across tenants). A single-row
        existence probe -- the orphan sweeper calls this under the per-digest
        lock so its delete decision reflects the current reference set, not a
        snapshot taken before the lock."""
        async def _action(session: AsyncSession) -> bool:
            result = await session.execute(
                select(ArtifactRecordRow.content_hash)
                .where(ArtifactRecordRow.content_hash == digest.value)
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
