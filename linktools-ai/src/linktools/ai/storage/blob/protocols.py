#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BlobStore: content-addressed immutable byte storage. A generic storage
capability -- the artifact domain is its primary consumer today, but the
Protocol carries no artifact-domain type. ``digest`` is the plain SHA-256 hex
digest string (64 lowercase hex chars); the caller validates it (the artifact
domain's ``ArtifactDigest`` value object) before it ever reaches this layer."""

from dataclasses import dataclass
from typing import AsyncContextManager, AsyncIterator, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class BlobInfo:
    """Metadata for a stored content-addressed blob (SHA-256 digest)."""

    digest: str
    size: int
    content_type: "str | None"


@runtime_checkable
class BlobStore(Protocol):
    """Content-addressed immutable byte storage.

    ``put_if_absent`` is idempotent on digest. Reads stream; callers re-verify
    the digest after reading.
    """

    async def put_if_absent(
        self,
        *,
        digest: str,
        source: AsyncIterator[bytes],
        size: "int | None",
    ) -> BlobInfo: ...

    def open(
        self, *, digest: str
    ) -> AsyncContextManager[AsyncIterator[bytes]]: ...

    async def stat(self, *, digest: str) -> "BlobInfo | None": ...

    async def delete(self, *, digest: str) -> None: ...


__all__: "list[str]" = ["BlobInfo", "BlobStore"]
