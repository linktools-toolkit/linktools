#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactRecord UoW transaction participation.

A SqlAlchemyStorageAdapter's ArtifactRecordStore shares the UoW's AsyncSession,
so a Run + ArtifactRecord written in one transaction commit / roll back
together. The Filesystem record store honestly declares it does NOT
participate in a cross-store transaction."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from linktools.ai.runtime.persistence.features import StorageFeatures
from linktools.ai.storage.features import StorageComponent, TransactionScope


def test_sqlite_artifact_record_participates_in_transaction(tmp_path):
    """On a DATABASE-scoped Storage, ARTIFACT_RECORDS is in the
    transactional_components set (the SQL record store shares the UoW session)
    and a Run + ArtifactRecord written together roll back together on a
    second-step failure."""
    from linktools.ai.runtime.persistence.sqlite import SqliteStorage

    async def _run():
        storage = await SqliteStorage.create(database=str(tmp_path / "s.db"))
        f = storage.features
        # SQL storage groups every wired component into one UoW.
        assert f.transaction_scope is TransactionScope.DATABASE
        assert StorageComponent.ARTIFACT_RECORDS in f.transactional_components
        # tx.artifact_records shares the UoW session: write a record, then
        # force a rollback by raising inside the UoW -- the record must NOT
        # persist.
        from linktools.ai.artifact.digest import ArtifactDigest
        from linktools.ai.artifact.models import (
            ArtifactProvenance,
            ArtifactRecord,
            ArtifactRef,
        )

        def _record(art_id: str, producer_id: str) -> ArtifactRecord:
            return ArtifactRecord(
                ref=ArtifactRef(
                    id=art_id,
                    sha256="00" * 32,
                    media_type="text/plain",
                    size=1,
                ),
                tenant_id="tenant-1",
                provenance=ArtifactProvenance(
                    producer_kind="run", producer_id=producer_id
                ),
                created_at=datetime.now(timezone.utc),
            )

        # Commit one record successfully.
        async with storage.transaction() as tx:
            await tx.artifact_records.put(_record("art-1", "run-1"))
        async with storage.transaction() as tx:
            got = await tx.artifact_records.get("art-1", tenant_id="tenant-1")
            assert got is not None
            assert got.ref.id == "art-1"
        # Now a UoW that writes a record then raises: the whole tx rolls back.
        with pytest.raises(RuntimeError):
            async with storage.transaction() as tx:
                await tx.artifact_records.put(_record("art-2", "run-2"))
                raise RuntimeError("force rollback")
        # art-2 did NOT persist (rolled back with the UoW).
        async with storage.transaction() as tx:
            assert await tx.artifact_records.get("art-2", tenant_id="tenant-1") is None
            # art-1 is still there (its own UoW committed independently).
            assert await tx.artifact_records.get("art-1", tenant_id="tenant-1") is not None
        await storage._engine.dispose()

    asyncio.run(_run())


def test_filesystem_artifact_record_does_not_claim_transaction(tmp_path):
    """Negative case: the Filesystem record store honestly declares
    transaction_participation=False, so ARTIFACT_RECORDS is NOT in
    transactional_components on a FilesystemStorage (which has no cross-store
    transaction anyway)."""
    from linktools.ai.runtime.persistence.facade import FilesystemStorage

    storage = FilesystemStorage(root=tmp_path)
    assert storage.artifacts.record_store.capabilities.transaction_participation is False
    # Filesystem has no cross-store transaction, so no component is transactional.
    assert StorageComponent.ARTIFACT_RECORDS not in storage.features.transactional_components
