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
    store = ArtifactStore(LocalArtifactBackend(tmp_path))

    async def content():
        yield data

    await store.put(record=record, content=content())
    assert await store.get_record("a") == record
    assert (tmp_path / "artifacts/blobs" / digest).exists()
