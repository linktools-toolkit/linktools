#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""streaming API: ArtifactStore.put_stream / open_stream + the 16 MiB
buffered-size cap that forces large content off the whole-bytes put/get path
.

A self-contained in-memory streaming backend exercises the facade WITHOUT
buffering the whole artifact (the blob ``open`` yields fixed-size chunks, so a
multi-chunk artifact must arrive as multiple yields -- proving the facade
streams rather than joining)."""

import asyncio
import hashlib
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pytest

from linktools.ai.artifact import (
    ANONYMOUS_PROVENANCE,
    ArtifactBlobNotFoundError,
    ArtifactBufferedSizeLimitError,
    ArtifactIntegrityError,
    ArtifactStore,
)
from linktools.ai.artifact import store as store_mod
from linktools.ai.artifact.coordination import InProcessArtifactDigestCoordinator
from linktools.ai.artifact.digest import ArtifactDigest
from linktools.ai.artifact.models import ArtifactProvenance
from linktools.ai.storage.protocols import BlobInfo

_CHUNK = 64


class _StreamBlob:
    """In-memory ArtifactBlobStore whose ``open`` yields fixed-size chunks, so a
    caller can observe that content arrives streamed (more than one chunk for
    multi-chunk data), not joined into one."""

    def __init__(self) -> None:
        self._blobs: "dict[str, bytes]" = {}

    async def put_if_absent(
        self, *, digest: ArtifactDigest, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo:
        acc: "list[bytes]" = []
        async for c in source:
            acc.append(c)
        data = b"".join(acc)
        if hashlib.sha256(data).hexdigest() != digest.value:
            raise ArtifactIntegrityError("digest mismatch")
        self._blobs.setdefault(digest.value, data)
        return BlobInfo(digest=digest.value, size=len(data), content_type=None)

    @asynccontextmanager
    async def open(self, *, digest: ArtifactDigest):
        data = self._blobs.get(digest.value)
        if data is None:
            raise ArtifactBlobNotFoundError("missing")

        async def _chunks() -> AsyncIterator[bytes]:
            for i in range(0, len(data), _CHUNK):
                yield data[i : i + _CHUNK]

        yield _chunks()

    async def stat(self, *, digest: ArtifactDigest) -> "BlobInfo | None":
        d = self._blobs.get(digest.value)
        return None if d is None else BlobInfo(digest=digest.value, size=len(d), content_type=None)

    async def delete(self, *, digest: ArtifactDigest) -> None:
        self._blobs.pop(digest.value, None)


class _StreamRecord:
    def __init__(self) -> None:
        self._records: "dict[tuple[str, str], object]" = {}

    async def put(self, record):
        self._records[(record.ref.id, record.tenant_id)] = record
        return record

    async def get(self, artifact_id: str, *, tenant_id: str):
        return self._records.get((artifact_id, tenant_id))

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        return self._records.pop((artifact_id, tenant_id), None) is not None


def _store() -> "tuple[ArtifactStore, _StreamBlob]":
    blob = _StreamBlob()
    return ArtifactStore(blob, _StreamRecord(), InProcessArtifactDigestCoordinator()), blob


def _aiter(chunks: "list[bytes]") -> AsyncIterator[bytes]:
    async def _g() -> AsyncIterator[bytes]:
        for c in chunks:
            yield c

    return _g()


def test_put_stream_roundtrips_through_open_stream() -> None:
    store, _ = _store()
    payload = b"AB" * 200  # 400 bytes -> multiple _CHUNK-sized yields

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([payload[:100], payload[100:200], payload[200:]]),
            media_type="text/plain",
            tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        assert record.ref.size == 400
        collected: "list[bytes]" = []
        async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
            async for chunk in chunks:
                collected.append(chunk)
        assert b"".join(collected) == payload
        # Streaming, not joined: 400 bytes / 64-byte blob chunks = 7 yields.
        assert len(collected) > 1

    asyncio.run(run())


def test_put_stream_dedupes_content_but_mints_distinct_records() -> None:
    store, _ = _store()
    payload = b"shared-content-streamed"

    async def run() -> None:
        a = await store.put_stream(
            source=_aiter([payload]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        b = await store.put_stream(
            source=_aiter([payload[:9], payload[9:]]),
            media_type="text/plain",
            tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        assert a.ref.sha256 == b.ref.sha256  # identical content -> one blob
        assert a.ref.id != b.ref.id  # distinct records

    asyncio.run(run())


def test_put_stream_keeps_per_call_provenance() -> None:
    store, _ = _store()

    async def run() -> None:
        a = await store.put_stream(
            source=_aiter([b"x"]), media_type="text/plain", tenant_id="t1",
            provenance=ArtifactProvenance(producer_kind="task", producer_id="A"),
        )
        b = await store.put_stream(
            source=_aiter([b"x"]), media_type="text/plain", tenant_id="t1",
            provenance=ArtifactProvenance(producer_kind="task", producer_id="B"),
        )
        assert a.provenance.producer_id == "A"
        assert b.provenance.producer_id == "B"

    asyncio.run(run())


def test_open_stream_tenant_gate_yields_nothing_to_foreign_tenant() -> None:
    store, _ = _store()

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([b"secret"]), media_type="text/plain", tenant_id="tenant-A", provenance=ANONYMOUS_PROVENANCE,
    )
        foreign: "list[bytes]" = []
        async with store.open_stream(artifact_id=record.ref.id, tenant_id="tenant-B") as chunks:
            async for chunk in chunks:
                foreign.append(chunk)
        assert foreign == []  # foreign tenant learns nothing

    asyncio.run(run())


def test_open_stream_missing_artifact_yields_nothing() -> None:
    store, _ = _store()

    async def run() -> None:
        collected: "list[bytes]" = []
        async with store.open_stream(artifact_id="art-nope", tenant_id="t1") as chunks:
            async for c in chunks:
                collected.append(c)
        assert collected == []

    asyncio.run(run())


def test_open_stream_detects_tampering_at_exhaustion() -> None:
    store, blob = _store()

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([b"original-payload"]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        # Tamper the stored blob; the pinned sha256 in the record no longer matches.
        blob._blobs[record.ref.sha256] = b"TAMPERED"
        with pytest.raises(ArtifactIntegrityError):
            async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
                async for _ in chunks:
                    pass  # exhaust so the at-exhaustion check runs

    asyncio.run(run())


def test_put_rejects_content_at_buffered_limit(monkeypatch) -> None:
    monkeypatch.setattr(store_mod, "BUFFERED_ARTIFACT_LIMIT", 8)
    store, _ = _store()

    async def run() -> None:
        with pytest.raises(ArtifactBufferedSizeLimitError):
            await store.put(content=b"0123456789", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)

    asyncio.run(run())


def test_get_rejects_oversized_record_even_when_written_via_stream(monkeypatch) -> None:
    # Content above the buffered limit written via put_stream must STILL be read
    # via open_stream: get() refuses to buffer it.
    monkeypatch.setattr(store_mod, "BUFFERED_ARTIFACT_LIMIT", 8)
    store, _ = _store()

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([b"0123456789"]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        assert record.ref.size >= 8
        with pytest.raises(ArtifactBufferedSizeLimitError):
            await store.get(artifact_id=record.ref.id, tenant_id="t1")
        # ...and open_stream reads it back fine.
        collected: "list[bytes]" = []
        async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
            async for c in chunks:
                collected.append(c)
        assert b"".join(collected) == b"0123456789"

    asyncio.run(run())


def test_small_put_still_works_under_default_limit() -> None:
    # The cap does not regress the normal small-bytes path at the real 16 MiB.
    store, _ = _store()

    async def run() -> None:
        record = await store.put(content=b"small", media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,)
        assert await store.get(artifact_id=record.ref.id, tenant_id="t1") == b"small"

    asyncio.run(run())


def test_put_stream_handles_payload_above_spool_threshold() -> None:
    # >64 KiB forces SpooledTemporaryFile.rollover() to disk during staging; the
    # round-trip must still be correct (regression guard for the roll).
    store, _ = _store()
    payload = b"AB" * (70 * 1024 // 2)  # 70 KiB, above the 64 KiB spool threshold

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([payload[:32768], payload[32768:]]),
            media_type="text/plain",
            tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        assert record.ref.size == len(payload)
        collected: "list[bytes]" = []
        async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
            async for c in chunks:
                collected.append(c)
        assert b"".join(collected) == payload

    asyncio.run(run())


def test_put_stream_handles_empty_stream() -> None:
    store, _ = _store()

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        assert record.ref.size == 0
        assert record.ref.sha256 == hashlib.sha256(b"").hexdigest()
        collected: "list[bytes]" = []
        async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
            async for c in chunks:
                collected.append(c)
        assert b"".join(collected) == b""

    asyncio.run(run())


def test_put_stream_propagates_source_error_and_stays_usable() -> None:
    # A source that raises mid-stream must propagate the error AND release its
    # staging temp file, leaving the store usable.
    store, _ = _store()

    class _Boom(Exception):
        pass

    async def _bad_source() -> AsyncIterator[bytes]:
        yield b"partial"
        raise _Boom("source failed mid-stream")

    async def run() -> None:
        with pytest.raises(_Boom):
            await store.put_stream(
                source=_bad_source(), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        # Store is still usable after the failed put (staging cleaned up).
        record = await store.put_stream(
            source=_aiter([b"ok"]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        assert await store.stat(artifact_id=record.ref.id, tenant_id="t1") is not None

    asyncio.run(run())


def test_put_stream_with_expected_digest_skips_staging_and_verifies() -> None:
    # 'caller provides expected_digest' branch: the source streams
    # straight into put_if_absent under the claimed digest (no staging file --
    # the caller vouches for the digest), and a WRONG expected_digest is caught
    # by the blob store's re-verification with no record written.
    store, _ = _store()
    payload = b"known-content-with-pregiven-digest"
    good_digest = hashlib.sha256(payload).hexdigest()

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([payload[:10], payload[10:]]),
            media_type="text/plain",
            tenant_id="t1",
            expected_digest=good_digest,
            size=len(payload), provenance=ANONYMOUS_PROVENANCE,
    )
        assert record.ref.sha256 == good_digest
        assert record.ref.size == len(payload)

        with pytest.raises(ArtifactIntegrityError):
            await store.put_stream(
                source=_aiter([payload]),
                media_type="text/plain",
                tenant_id="t1",
                expected_digest="0" * 64,  # wrong digest -> rejected
                size=len(payload), provenance=ANONYMOUS_PROVENANCE,
    )

    asyncio.run(run())


def test_open_stream_detects_size_mismatch_at_exhaustion() -> None:
    # step 5: open_stream verifies BOTH size and sha256. Replacing the
    # stored blob with fewer bytes makes the recorded size (100) wrong even
    # before the digest check; ArtifactIntegrityError surfaces at exhaustion.
    store, blob = _store()
    payload = b"x" * 100

    async def run() -> None:
        record = await store.put_stream(
            source=_aiter([payload]), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        blob._blobs[record.ref.sha256] = b"short"  # 5 bytes, not 100
        with pytest.raises(ArtifactIntegrityError):
            async with store.open_stream(artifact_id=record.ref.id, tenant_id="t1") as chunks:
                async for _ in chunks:
                    pass

    asyncio.run(run())


def test_put_stream_cancellation_publishes_no_partial_blob() -> None:
    # Cancelling put_stream mid-stream must release the staging file AND publish
    # no blob (put_if_absent is never reached). Deterministic: the source signals
    # after its first yield, so the test cancels exactly when the source is
    # blocked (not via a timing sleep).
    import asyncio

    store, blob = _store()
    yielded = asyncio.Event()

    async def _blocking_source() -> AsyncIterator[bytes]:
        yield b"partial"
        yielded.set()
        await asyncio.Event().wait()  # block forever
        yield b"never"

    async def run() -> None:
        task = asyncio.create_task(
            store.put_stream(
                source=_blocking_source(), media_type="text/plain", tenant_id="t1", provenance=ANONYMOUS_PROVENANCE,
    )
        )
        await yielded.wait()  # source yielded once + is now blocked
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # No blob was published: put_if_absent never completed.
        assert blob._blobs == {}

    asyncio.run(run())

