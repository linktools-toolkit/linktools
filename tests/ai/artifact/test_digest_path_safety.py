#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path-safety of the artifact digest boundary: a malformed ``expected_digest``
must raise BEFORE the coordinator lock is acquired, BEFORE any lock file is
created, and BEFORE the blob store is touched. Nothing in the put/sweep pipeline
ever sees an unvalidated string as a coordination key or a path component."""

from typing import AsyncIterator

import pytest

from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
from linktools.ai.artifact.digest import ArtifactDigest
from linktools.ai.artifact.models import ArtifactProvenance
from linktools.ai.artifact.store import ArtifactStore
from linktools.ai.errors import InvalidArtifactDigestError

_BAD = "../etc/passwd"


class _RecordingCoordinator:
    """Wraps a real coordinator, counting ``hold`` entries so the test can prove
    a bad digest never reached the lock."""

    def __init__(self, inner):
        self._inner = inner
        self.hold_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    class _Hold:
        def __init__(self, outer):
            self._outer = outer

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    def hold(self, digest: ArtifactDigest):
        self.hold_count += 1
        return self._inner.hold(digest)


class _RecordingBlob:
    """A minimal ArtifactBlobStore that records whether it was touched."""

    def __init__(self):
        self.put_count = 0

    async def put_if_absent(
        self, *, digest: ArtifactDigest, source: AsyncIterator[bytes], size
    ):
        self.put_count += 1
        raise AssertionError("blob store must not be called for a bad digest")

    async def open(self, *, digest: ArtifactDigest):
        raise AssertionError

    async def stat(self, *, digest: ArtifactDigest):
        raise AssertionError

    async def delete(self, *, digest: ArtifactDigest):
        raise AssertionError


class _NullRecords:
    async def put(self, record):
        raise AssertionError("record store must not be called for a bad digest")

    async def get(self, artifact_id, *, tenant_id):
        return None

    async def delete(self, artifact_id, *, tenant_id):
        return False


def _provenance() -> ArtifactProvenance:
    return ArtifactProvenance(producer_kind="anonymous", producer_id="")


async def _source():
    yield b"content"
    return


@pytest.mark.asyncio
async def test_bad_expected_digest_raises_before_any_side_effect():
    inner = InProcessArtifactDigestCoordinator()
    coord = _RecordingCoordinator(inner)
    blob = _RecordingBlob()
    store = ArtifactStore(blob, _NullRecords(), coord)

    with pytest.raises(InvalidArtifactDigestError):
        await store.put_stream(
            tenant_id="t1",
            source=_source(),
            media_type="",
            provenance=_provenance(),
            expected_digest=_BAD,
        )

    assert coord.hold_count == 0, "coordinator acquired for an invalid digest"
    assert blob.put_count == 0, "blob store touched for an invalid digest"


@pytest.mark.asyncio
async def test_bad_expected_digest_creates_no_lock_file(tmp_path):
    # Use the real in-process coordinator and inspect its registry: a bad digest
    # must not register a lock entry (the value object rejects before hold).
    inner = InProcessArtifactDigestCoordinator()
    coord = _RecordingCoordinator(inner)
    blob = _RecordingBlob()
    store = ArtifactStore(blob, _NullRecords(), coord)

    with pytest.raises(InvalidArtifactDigestError):
        await store.put_stream(
            tenant_id="t1",
            source=_source(),
            media_type="",
            provenance=_provenance(),
            expected_digest=_BAD,
        )

    assert inner.active_entry_count == 0, "a lock entry was registered for a bad digest"


@pytest.mark.asyncio
async def test_filesystem_coordinator_lock_path_stays_in_locks_dir(tmp_path):
    # A valid digest produces a lock path whose parent is exactly the locks dir;
    # the parent-check defense rejects anything that would escape it.
    from linktools.ai.storage.filesystem.artifact_coordination import (
        FilesystemArtifactDigestCoordinator,
    )

    coord = FilesystemArtifactDigestCoordinator(root=tmp_path / "artifacts")
    digest = ArtifactDigest.from_bytes(b"x")
    async with coord.hold(digest):
        # The lock file lives directly under .locks, named by the digest value.
        locks_dir = tmp_path / "artifacts" / ".locks"
        assert (locks_dir / digest.value).exists()
        assert (locks_dir / digest.value).parent == locks_dir
    # Permissions on the locks dir are owner-only.
    import os

    mode = os.stat(locks_dir).st_mode & 0o777
    assert mode == 0o700
