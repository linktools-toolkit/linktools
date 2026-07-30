#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL-first ``ArtifactBackend``: metadata in ``ai_artifacts``, content in a
:class:`FilesystemArtifactBlobStore`. Records are create-only -- inserting the
same id with byte-identical content is idempotent (a retried ``put``), but a
different sha256/tenant/provenance under an existing id is refused rather than
overwriting the prior write's lineage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Integer, String, UniqueConstraint, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from ...errors import ArtifactRecordConflictError
from ...storage.sqlalchemy.base import Base
from ...storage.sqlalchemy.conventions import TABLE_PREFIX, as_utc
from ..models import (
    ArtifactBlobNotFoundError,
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactRef,
)
from .blob import FilesystemArtifactBlobStore


class ArtifactRow(Base):
    __tablename__ = f"{TABLE_PREFIX}artifacts"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "artifact_id", name="uq_artifact_tenant_id"
        ),
    )
    artifact_id: Mapped[str] = mapped_column(String(128), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    media_type: Mapped[str] = mapped_column(String(128))
    size: Mapped[int] = mapped_column(Integer)
    tenant_id: Mapped[str] = mapped_column(String(128), index=True)
    producer_kind: Mapped[str] = mapped_column(String(64))
    producer_id: Mapped[str] = mapped_column(String(128))
    run_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    session_id: Mapped["str | None"] = mapped_column(String(128), nullable=True)
    parent_artifact_ids: Mapped[Any] = mapped_column(JSON)
    provenance_metadata: Mapped[Any] = mapped_column(JSON)


def _record(row: ArtifactRow) -> ArtifactRecord:
    return ArtifactRecord(
        ref=ArtifactRef(row.artifact_id, row.sha256, row.media_type, row.size),
        tenant_id=row.tenant_id,
        provenance=ArtifactProvenance(
            producer_kind=row.producer_kind,
            producer_id=row.producer_id,
            run_id=row.run_id,
            session_id=row.session_id,
            parent_artifact_ids=tuple(row.parent_artifact_ids),
            metadata=row.provenance_metadata,
        ),
        created_at=as_utc(row.created_at),
    )


class SqlArtifactBackend:
    def __init__(self, session_factory, blobs: "FilesystemArtifactBlobStore | str | Path") -> None:
        self._session_factory = session_factory
        self._blobs = blobs if isinstance(blobs, FilesystemArtifactBlobStore) else FilesystemArtifactBlobStore(blobs)

    async def initialize_storage(self, engine) -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        await self._blobs.initialize_storage()

    async def put(self, *, record: ArtifactRecord, content: AsyncIterator[bytes]) -> ArtifactRecord:
        await self._blobs.put(ref=record.ref, content=content)
        row = ArtifactRow(
            artifact_id=record.ref.id,
            sha256=record.ref.sha256,
            media_type=record.ref.media_type,
            size=record.ref.size,
            tenant_id=record.tenant_id,
            producer_kind=record.provenance.producer_kind,
            producer_id=record.provenance.producer_id,
            run_id=record.provenance.run_id,
            session_id=record.provenance.session_id,
            parent_artifact_ids=list(record.provenance.parent_artifact_ids),
            provenance_metadata=dict(record.provenance.metadata),
            created_at=record.created_at,
        )
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    session.add(row)
            return record
        except IntegrityError:
            existing = await self.get_record(
                record.ref.id,
                tenant_id=record.tenant_id,
            )
            if existing is None or not _same_artifact_identity(existing, record):
                raise ArtifactRecordConflictError(record.ref.id)
            return existing

    async def get_record(self, artifact_id: str, *, tenant_id: str) -> "ArtifactRecord | None":
        async with self._session_factory() as session:
            row = await session.scalar(
                select(ArtifactRow).where(
                    ArtifactRow.artifact_id == artifact_id,
                    ArtifactRow.tenant_id == tenant_id,
                )
            )
        return None if row is None else _record(row)

    async def open(self, artifact_id: str, *, tenant_id: str) -> AsyncIterator[bytes]:
        record = await self.get_record(artifact_id, tenant_id=tenant_id)
        if record is None:
            raise ArtifactBlobNotFoundError(artifact_id)
        async for chunk in self._blobs.open(record.ref.sha256):
            yield chunk

    async def delete(self, artifact_id: str, *, tenant_id: str) -> None:
        async with self._session_factory() as session:
            async with session.begin():
                row = await session.scalar(
                    select(ArtifactRow).where(
                        ArtifactRow.artifact_id == artifact_id,
                        ArtifactRow.tenant_id == tenant_id,
                    )
                )
                if row is not None:
                    await session.delete(row)


def _same_artifact_identity(
    left: ArtifactRecord,
    right: ArtifactRecord,
) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.ref.sha256 == right.ref.sha256
        and left.ref.media_type == right.ref.media_type
        and left.ref.size == right.ref.size
        and left.provenance == right.provenance
    )


__all__: "list[str]" = ["ArtifactRow", "SqlArtifactBackend"]
