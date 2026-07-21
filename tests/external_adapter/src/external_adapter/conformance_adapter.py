#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A deterministic, in-memory EXTERNAL storage adapter built ONLY from the
public storage Protocols.

It imports nothing private: no ``_runtime``, no ``storage.filesystem`` /
``storage.sqlalchemy`` reference backends, no ``storage.coordination``. Its
sole ``linktools`` imports are ``linktools.ai.storage.protocols`` (LeaseToken,
BlobInfo) and ``linktools.ai.artifact.models`` (ArtifactRecord,
ArtifactIntegrityError) -- both public surfaces.

It exists to PROVE the public Protocol surface is sufficient: a downstream
adapter can implement conformant blob / record / lease storage against the
Protocols alone, without reaching into core private modules. The conformance
testkit (``testing``, a test-support package outside the shipped wheel) is
run against this adapter in ``test_conformance.py``. If that test ever fails
because this adapter NEEDS a private import to conform, the Protocol design
is inadequate."""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

from linktools.ai.artifact.models import (
    ArtifactBlobNotFoundError,
    ArtifactIntegrityError,
    ArtifactRecord,
)
from linktools.ai.storage.protocols import BlobInfo, LeaseToken


class InMemoryArtifactBlobStore:
    """Content-addressed in-memory ArtifactBlobStore. ``put_if_absent`` always
    verifies the streamed bytes against the claimed digest (mismatch ->
    ArtifactIntegrityError) and is idempotent on a matching digest."""

    def __init__(self) -> None:
        self._blobs: "dict[str, bytes]" = {}

    async def put_if_absent(
        self, *, digest: str, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo:
        chunks: "list[bytes]" = []
        async for chunk in source:
            chunks.append(chunk)
        actual = b"".join(chunks)
        if hashlib.sha256(actual).hexdigest() != digest:
            raise ArtifactIntegrityError(
                f"digest mismatch: claimed sha256 {digest[:12]}..."
            )
        if size is not None and len(actual) != size:
            raise ArtifactIntegrityError(
                f"size mismatch: claimed {size} bytes, got {len(actual)}"
            )
        # Idempotent on digest: a second put of the same content is a reuse,
        # not an error.
        self._blobs.setdefault(digest, actual)
        return BlobInfo(digest=digest, size=len(actual), content_type=None)

    @asynccontextmanager
    async def open(self, *, digest: str):
        existing = self._blobs.get(digest)
        if existing is None:
            raise ArtifactBlobNotFoundError(
                f"blob for sha256 {digest[:12]} missing"
            )

        async def _chunks() -> AsyncIterator[bytes]:
            yield existing

        yield _chunks()

    async def stat(self, *, digest: str) -> "BlobInfo | None":
        data = self._blobs.get(digest)
        if data is None:
            return None
        return BlobInfo(digest=digest, size=len(data), content_type=None)

    async def delete(self, *, digest: str) -> None:
        self._blobs.pop(digest, None)


class InMemoryArtifactRecordStore:
    """Tenant-scoped in-memory ArtifactRecordStore. The record is opaque to
    the store (keyed by ``(ref.id, tenant_id)``); a foreign tenant learns
    nothing -- not even that a record exists."""

    def __init__(self) -> None:
        self._records: "dict[tuple[str, str], ArtifactRecord]" = {}

    async def put(self, record: ArtifactRecord) -> ArtifactRecord:
        self._records[(record.ref.id, record.tenant_id)] = record
        return record

    async def get(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None":
        return self._records.get((artifact_id, tenant_id))

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool:
        return self._records.pop((artifact_id, tenant_id), None) is not None


class InMemoryLeaseCoordinator:
    """In-memory LeaseCoordinator with monotonic fencing. Correct within one
    process: acquires are mutually exclusive per key, the fencing token
    strictly increases on every (re)acquisition so a state commit recording
    the token can reject a stale write, and renew preserves the token. Not
    correct across processes -- a deployment needing that injects a real
    distributed coordinator implementing the same Protocol."""

    def __init__(self) -> None:
        # key -> (owner_id, fencing_token, expires_at)
        self._holders: "dict[str, tuple[str, int, datetime]]" = {}
        # key -> last fencing token granted (monotonic, ever-increasing)
        self._counter: "dict[str, int]" = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    async def acquire(
        self, *, key: str, owner_id: str, ttl: timedelta
    ) -> "LeaseToken | None":
        now = self._now()
        held = self._holders.get(key)
        if held is not None and held[2] > now:
            # Still held by an unexpired owner -> second acquirer loses.
            return None
        token = self._counter.get(key, 0) + 1
        self._counter[key] = token
        expires = now + ttl
        self._holders[key] = (owner_id, token, expires)
        return LeaseToken(
            lease_id=f"{key}:{token}",
            owner_id=owner_id,
            fencing_token=token,
            expires_at=expires,
            key=key,
        )

    async def renew(self, *, token: LeaseToken, ttl: timedelta) -> LeaseToken:
        now = self._now()
        held = self._holders.get(token.key)
        if held is None:
            raise LookupError(
                f"cannot renew a released lease for key {token.key!r}"
            )
        if held[0] != token.owner_id or held[1] != token.fencing_token:
            raise LookupError(
                f"cannot renew a lease not currently held by {token.owner_id!r}"
            )
        if held[2] <= now:
            raise LookupError(
                f"cannot renew an expired lease for key {token.key!r} "
                f"(expired at {held[2].isoformat()})"
            )
        expires = now + ttl
        # Renew preserves the fencing token (renew is not a re-acquire).
        self._holders[token.key] = (token.owner_id, token.fencing_token, expires)
        return LeaseToken(
            lease_id=token.lease_id,
            owner_id=token.owner_id,
            fencing_token=token.fencing_token,
            expires_at=expires,
            key=token.key,
        )

    async def release(self, *, token: LeaseToken) -> None:
        held = self._holders.get(token.key)
        if held is not None and held[0] == token.owner_id:
            del self._holders[token.key]
