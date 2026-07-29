import hashlib
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.artifact.models import ArtifactProvenance, ArtifactRecord, ArtifactRef
from linktools.ai.artifact.persistence.metadata import SqlArtifactBackend
from linktools.ai.artifact.store import ArtifactStore
from linktools.ai.errors import ArtifactRecordConflictError


def _record(artifact_id: str, data: bytes) -> ArtifactRecord:
    digest = hashlib.sha256(data).hexdigest()
    return ArtifactRecord(
        ArtifactRef(artifact_id, digest, "text/plain", len(data)),
        "tenant",
        ArtifactProvenance("test", "1"),
        datetime.now(timezone.utc),
    )


async def _stream(data: bytes):
    yield data


@pytest.mark.asyncio
async def test_sql_artifact_store_persists_metadata_and_blob(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'artifacts.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ArtifactStore(SqlArtifactBackend(factory, tmp_path / "artifacts"))
    await store.initialize_storage(engine)

    data = b"artifact content"
    record = _record("a", data)
    await store.put(record=record, content=_stream(data))

    assert await store.get_record("a") == record
    assert (tmp_path / "artifacts" / "blobs" / record.ref.sha256).exists()
    chunks = [chunk async for chunk in store.open("a")]
    assert b"".join(chunks) == data
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_artifact_store_put_is_idempotent_for_identical_content(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'artifacts-retry.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ArtifactStore(SqlArtifactBackend(factory, tmp_path / "artifacts"))
    await store.initialize_storage(engine)

    data = b"same bytes"
    record = _record("a", data)
    await store.put(record=record, content=_stream(data))
    replayed = await store.put(record=record, content=_stream(data))

    assert replayed == record
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_artifact_store_rejects_conflicting_content_for_same_id(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'artifacts-conflict.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ArtifactStore(SqlArtifactBackend(factory, tmp_path / "artifacts"))
    await store.initialize_storage(engine)

    await store.put(record=_record("a", b"first"), content=_stream(b"first"))
    with pytest.raises(ArtifactRecordConflictError):
        await store.put(record=_record("a", b"second"), content=_stream(b"second"))
    await engine.dispose()


@pytest.mark.asyncio
async def test_sql_artifact_store_deletes_metadata_without_touching_shared_blob(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'artifacts-delete.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = ArtifactStore(SqlArtifactBackend(factory, tmp_path / "artifacts"))
    await store.initialize_storage(engine)

    data = b"shared bytes"
    record = _record("a", data)
    await store.put(record=record, content=_stream(data))
    await store.delete("a")

    assert await store.get_record("a") is None
    assert (tmp_path / "artifacts" / "blobs" / record.ref.sha256).exists()
    await engine.dispose()
