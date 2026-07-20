#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable storage extension Protocols -- the public surface a downstream or
external adapter implements to plug into the Runtime.

These Protocols depend only on the standard library and ``linktools-ai``
domain models. A backend (Filesystem, SQLAlchemy, or an external one) implements
them; the RuntimeBuilder capability-gates on StorageFeatures + Protocol
availability, never on ``isinstance`` against a concrete class. External
adapters hide multipart uploads, connection pools, retries and vendor errors
behind these Protocols; they convert all exceptions to core error types via
``raise ... from exc``.

The AssetStore / ArtifactRecord-store Protocols that reference the asset
domain types are hosted here once ``asset/`` lands; this module currently
defines the Protocols whose signatures are independent of that rename:
leasing, artifact blobs, and the transaction unit-of-work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, AsyncContextManager, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    # ArtifactRecord appears only in ArtifactRecordStore method signatures
    # (string annotations, resolved lazily). Keeping this import TYPE_CHECKING-
    # only breaks the runtime cycle storage.protocols -> artifact.models ->
    # artifact (.__init__ imports .store) -> storage.protocols, so an external
    # adapter can ``from linktools.ai.storage.protocols import ...`` in any
    # import order without hitting a partially-initialized module.
    from ..artifact.models import ArtifactRecord


@dataclass(frozen=True, slots=True)
class LeaseToken:
    """The handle a LeaseCoordinator returns for a held lease.

    ``fencing_token`` is monotonically increasing across (re)acquisitions of
    the same key; a re-acquire after expiry must yield a LARGER token. Renewing
    a held lease must NOT change the token. JobStore state commits check the
    fencing token rather than trusting the coordinator's claim that the lock
    is still held.
    """

    lease_id: str
    owner_id: str
    fencing_token: int
    expires_at: datetime
    key: str


@runtime_checkable
class LeaseCoordinator(Protocol):
    """Distributed-lease coordination with monotonic fencing tokens.

    Protocol-level timing contract: acquire/renew/release call timeouts must
    not exceed ``min(1 second, lease_ttl / 3)``; adapters must support
    cancellation and return a concrete error on timeout (never a fake success).
    """

    async def acquire(
        self, *, key: str, owner_id: str, ttl: timedelta
    ) -> "LeaseToken | None": ...

    async def renew(self, *, token: LeaseToken, ttl: timedelta) -> LeaseToken: ...

    async def release(self, *, token: LeaseToken) -> None: ...


@dataclass(frozen=True, slots=True)
class BlobInfo:
    """Metadata for a stored content-addressed blob (SHA-256 digest)."""

    digest: str
    size: int
    content_type: "str | None"


@runtime_checkable
class ArtifactBlobStore(Protocol):
    """Content-addressed immutable byte storage for artifacts.

    ``put_if_absent`` is idempotent on digest. Reads stream; callers re-verify
    the digest after reading.
    """

    async def put_if_absent(
        self, *, digest: str, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo: ...

    def open(self, *, digest: str) -> AsyncContextManager[AsyncIterator[bytes]]: ...

    async def stat(self, *, digest: str) -> "BlobInfo | None": ...

    async def delete(self, *, digest: str) -> None: ...


@runtime_checkable
class ArtifactRecordStore(Protocol):
    """The access-control + provenance fact source for artifacts.

    Every read loads the record by artifact id and checks tenant ownership
    first; a digest alone is never enough to fetch bytes.
    """

    async def put(self, record: ArtifactRecord) -> ArtifactRecord: ...

    async def get(
        self, artifact_id: str, *, tenant_id: str
    ) -> "ArtifactRecord | None": ...

    async def delete(self, artifact_id: str, *, tenant_id: str) -> bool: ...


@runtime_checkable
class StorageTransactionManager(Protocol):
    """Cross-store atomic scope. ``transaction()`` commits once on clean exit
    and rolls back once on exception; callers never call backend commit/rollback
    directly. An unsupported scope raises StorageTransactionNotSupportedError at
    the call."""

    def transaction(self) -> AsyncContextManager["StorageUnitOfWork"]: ...


@runtime_checkable
class StorageUnitOfWork(Protocol):
    """The stores sharing one transaction. Optional stores are ``None`` where
    the backend does not provide them in this scope."""

    # Concrete stores are attached by the implementing backend (Filesystem
    # journal / SQLAlchemy AsyncSession). Typed as Any so the Protocol stays
    # free of concrete store imports.
    runs: Any
    sessions: Any
    events: Any
    checkpoints: Any
    approvals: Any
    idempotency: Any


__all__: "list[str]" = [
    "ArtifactBlobStore",
    "ArtifactRecordStore",
    "BlobInfo",
    "LeaseCoordinator",
    "LeaseToken",
    "StorageTransactionManager",
    "StorageUnitOfWork",
]
