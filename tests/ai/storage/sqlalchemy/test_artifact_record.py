#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyArtifactRecordStore. Metadata-only
SQL store for ArtifactRecord: it uses the caller-provided AsyncSession, holds no
engine, branches on no dialect, and never stores blob bytes. Composes with
FilesystemArtifactBlobStore at the SqlAlchemyStorage layer."""

import asyncio
import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from linktools.ai.artifact.models import (  # noqa: E402
    ArtifactProvenance,
    ArtifactRecord,
    ArtifactRef,
)
from linktools.ai.storage.sqlalchemy.artifact_record import (  # noqa: E402
    SqlAlchemyArtifactRecordStore,
)
from linktools.ai.storage.sqlalchemy.models import Base  # noqa: E402


def _record(artifact_id="art-1", tenant_id="t1", sha="a" * 64):
    return ArtifactRecord(
        ref=ArtifactRef(id=artifact_id, sha256=sha, media_type="text/plain", size=4),
        tenant_id=tenant_id,
        provenance=ArtifactProvenance(
            producer_kind="job",
            producer_id="job-1",
            parent_artifact_ids=("p1",),
            metadata={"k": "v"},
        ),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_factory(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/art.db")

    async def _init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    asyncio.run(_init())
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def test_put_get_delete_roundtrip(tmp_path) -> None:
    engine, factory = _make_factory(tmp_path)
    store = SqlAlchemyArtifactRecordStore(session_factory=factory)

    async def run():
        rec = _record()
        await store.put(rec)
        fetched = await store.get(artifact_id="art-1", tenant_id="t1")
        assert fetched == rec
        assert fetched.provenance.producer_id == "job-1"
        assert await store.delete("art-1", tenant_id="t1") is True
        assert await store.get(artifact_id="art-1", tenant_id="t1") is None
        assert await store.delete("art-1", tenant_id="t1") is False

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())


def test_tenant_gate(tmp_path) -> None:
    engine, factory = _make_factory(tmp_path)
    store = SqlAlchemyArtifactRecordStore(session_factory=factory)

    async def run():
        await store.put(_record(artifact_id="art-1", tenant_id="tenant-A"))
        # A foreign tenant learns nothing on get OR delete.
        assert await store.get(artifact_id="art-1", tenant_id="tenant-B") is None
        assert await store.delete("art-1", tenant_id="tenant-B") is False
        # The owning tenant still sees it.
        assert await store.get(artifact_id="art-1", tenant_id="tenant-A") is not None

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())


def test_put_same_id_identical_content_is_idempotent(tmp_path) -> None:
    engine, factory = _make_factory(tmp_path)
    store = SqlAlchemyArtifactRecordStore(session_factory=factory)

    async def run():
        first = await store.put(_record(artifact_id="art-1", tenant_id="t1", sha="a" * 64))
        # Same id, byte-identical record -> idempotent return, no conflict.
        second = await store.put(_record(artifact_id="art-1", tenant_id="t1", sha="a" * 64))
        assert first == second
        fetched = await store.get(artifact_id="art-1", tenant_id="t1")
        assert fetched.ref.sha256 == "a" * 64

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())


def test_put_same_id_different_content_conflicts(tmp_path) -> None:
    from linktools.ai.errors import ArtifactRecordConflictError

    engine, factory = _make_factory(tmp_path)
    store = SqlAlchemyArtifactRecordStore(session_factory=factory)

    async def run():
        await store.put(_record(artifact_id="art-1", tenant_id="t1", sha="a" * 64))
        # Same id, different sha256 -> create-only conflict; lineage is preserved.
        with pytest.raises(ArtifactRecordConflictError):
            await store.put(_record(artifact_id="art-1", tenant_id="t1", sha="b" * 64))
        fetched = await store.get(artifact_id="art-1", tenant_id="t1")
        assert fetched.ref.sha256 == "a" * 64

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())


def test_put_refuses_cross_tenant_re_homing(tmp_path) -> None:
    # A foreign tenant cannot overwrite another tenant's record by reusing an
    # id; the cross-tenant content difference is a conflict, and the original
    # row stays with its tenant, unchanged.
    from linktools.ai.errors import ArtifactRecordConflictError

    engine, factory = _make_factory(tmp_path)
    store = SqlAlchemyArtifactRecordStore(session_factory=factory)

    async def run():
        await store.put(_record(artifact_id="art-1", tenant_id="tenant-A"))
        with pytest.raises(ArtifactRecordConflictError):
            await store.put(_record(artifact_id="art-1", tenant_id="tenant-EVIL"))
        # Original tenant still owns it, unchanged.
        fetched = await store.get(artifact_id="art-1", tenant_id="tenant-A")
        assert fetched is not None and fetched.tenant_id == "tenant-A"
        assert await store.get(artifact_id="art-1", tenant_id="tenant-EVIL") is None

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())


def test_iter_referenced_digests(tmp_path) -> None:
    engine, factory = _make_factory(tmp_path)
    store = SqlAlchemyArtifactRecordStore(session_factory=factory)

    async def run():
        await store.put(_record(artifact_id="art-1", sha="a" * 64))
        await store.put(_record(artifact_id="art-2", tenant_id="t2", sha="b" * 64))
        digests = [d async for d in store.iter_referenced_digests()]
        assert sorted(digests) == sorted(["a" * 64, "b" * 64])

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())


def test_uow_session_participation_uses_shared_session(tmp_path) -> None:
    # Constructed with a shared session, the store FLUSHES (does not commit):
    # a put is visible to a get on the SAME session before the surrounding
    # transaction commits, and a fresh session sees nothing until that commit.
    engine, factory = _make_factory(tmp_path)

    async def run():
        async with factory() as session:
            async with session.begin():
                uow_store = SqlAlchemyArtifactRecordStore(
                    session_factory=factory, session=session
                )
                await uow_store.put(_record(artifact_id="art-1", tenant_id="t1"))
                # Visible within the UoW session (flushed, not committed yet).
                assert await uow_store.get(artifact_id="art-1", tenant_id="t1") is not None
                # A SEPARATE connection opened BEFORE the UoW commits sees
                # nothing -- the store flushed (did not secret-commit), so the
                # write is still isolated to this transaction.
                async with factory() as peek:
                    peek_store = SqlAlchemyArtifactRecordStore(
                        session_factory=factory, session=peek
                    )
                    assert await peek_store.get(artifact_id="art-1", tenant_id="t1") is None
        # After the UoW block commits, a fresh session observes it.
        async with factory() as fresh:
            fresh_store = SqlAlchemyArtifactRecordStore(
                session_factory=factory, session=fresh
            )
            assert await fresh_store.get(artifact_id="art-1", tenant_id="t1") is not None

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())


def test_row_stores_metadata_not_blob_bytes(tmp_path) -> None:
    # The data_json column carries the full record envelope; no blob content is
    # persisted by this store (it is metadata only).
    engine, factory = _make_factory(tmp_path)
    store = SqlAlchemyArtifactRecordStore(session_factory=factory)

    async def run():
        rec = _record()
        rec = ArtifactRecord(
            ref=ArtifactRef(id=rec.ref.id, sha256=rec.ref.sha256, media_type=rec.ref.media_type, size=rec.ref.size),
            tenant_id=rec.tenant_id,
            provenance=ArtifactProvenance(
                producer_kind=rec.provenance.producer_kind,
                producer_id=rec.provenance.producer_id,
                parent_artifact_ids=rec.provenance.parent_artifact_ids,
                metadata={"nested": {"x": 1}},
            ),
            created_at=rec.created_at,
        )
        await store.put(rec)
        from linktools.ai.storage.sqlalchemy.models import ArtifactRecordRow

        async with factory() as session:
            row = await session.get(ArtifactRecordRow, "art-1")
            assert row is not None
            envelope = json.loads(row.data_json)
            assert envelope["tenant_id"] == "t1"
            assert envelope["ref"]["sha256"] == "a" * 64
            assert envelope["provenance"]["metadata"] == {"nested": {"x": 1}}

    try:
        asyncio.run(run())
    finally:
        asyncio.run(engine.dispose())
