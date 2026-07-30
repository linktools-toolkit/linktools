import hashlib
from datetime import datetime, timezone

import pytest

from linktools.ai.artifact.models import ArtifactProvenance, ArtifactRecord, ArtifactRef
from linktools.ai.artifact.persistence.local import LocalArtifactBackend
from linktools.ai.artifact.store import ArtifactStore


@pytest.mark.asyncio
async def test_local_artifact_store_uses_content_addressed_layout(tmp_path):
    data = b"artifact"
    digest = hashlib.sha256(data).hexdigest()
    record = ArtifactRecord(ArtifactRef("a", digest, "text/plain", len(data)), "tenant", ArtifactProvenance("test", "1"), datetime.now(timezone.utc))
    store = LocalArtifactBackend(tmp_path / "artifacts")
    await store.initialize_storage()

    async def content():
        yield data

    await store.put(record=record, content=content())
    assert await store.get_record("a", tenant_id="tenant") == record
    assert (tmp_path / "artifacts/blobs" / digest).exists()


@pytest.mark.asyncio
async def test_local_artifact_ids_are_tenant_scoped(tmp_path):
    store = LocalArtifactBackend(tmp_path / "artifacts")
    await store.initialize_storage()

    async def put(tenant, data):
        digest = hashlib.sha256(data).hexdigest()
        record = ArtifactRecord(
            ArtifactRef("shared", digest, "text/plain", len(data)),
            tenant,
            ArtifactProvenance("test", tenant),
            datetime.now(timezone.utc),
        )

        async def content():
            yield data

        await store.put(record=record, content=content())
        return record

    first, second = await __import__("asyncio").gather(
        put("tenant-a", b"a"), put("tenant-b", b"b")
    )
    assert await store.get_record("shared", tenant_id="tenant-a") == first
    assert await store.get_record("shared", tenant_id="tenant-b") == second
    assert b"".join(
        [chunk async for chunk in store.open("shared", tenant_id="tenant-a")]
    ) == b"a"
    assert b"".join(
        [chunk async for chunk in store.open("shared", tenant_id="tenant-b")]
    ) == b"b"
