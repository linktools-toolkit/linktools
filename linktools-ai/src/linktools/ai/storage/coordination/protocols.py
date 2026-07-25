#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coordination Protocols: generic mutual-exusion + lease machinery with no
dependency on any domain package.

- :class:`KeyedCoordinator` -- per-key mutual exclusion. A caller-owned string
  key (the artifact domain uses a digest; a downstream Capability store uses a
  capability path), so the Protocol carries no domain-specific type.
- :class:`LeaseCoordinator` -- distributed-lease coordination with monotonic
  fencing tokens. The Job domain uses it for claim/renew/release of task
  ownership across workers; the in-repo reference is
  :class:`~linktools.ai.storage.coordination.process_local.ProcessLocalLeaseCoordinator`.

``scope`` on each coordinator declares the coordination range it actually
provides -- PROCESS_LOCAL (this process only) or DISTRIBUTED (spans workers/
processes). The Runtime multi-worker gate reads it to refuse a process-local
coordinator under a topology that shares state across workers."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, AsyncIterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..features import CoordinationScope


@dataclass(frozen=True, slots=True)
class LeaseToken:
    """The handle a LeaseCoordinator returns for a held lease.

    ``fencing_token`` is monotonically increasing across (re)acquisitions of
    the same key; a re-acquire after expiry must yield a LARGER token.
    Renewing a held lease must NOT change the token. JobStore state commits
    check the fencing token rather than trusting the coordinator's claim that
    the lock is still held.
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
    cancellation and return a concrete error on timeout (never a fake
    success).
    """

    async def acquire(
        self, *, key: str, owner_id: str, ttl: timedelta
    ) -> "LeaseToken | None": ...

    async def renew(self, *, token: LeaseToken, ttl: timedelta) -> LeaseToken: ...

    async def release(self, *, token: LeaseToken) -> None: ...


@runtime_checkable
class KeyedCoordinator(Protocol):
    """Per-key mutual exclusion. The same key serializes; different keys run
    in parallel (no global bottleneck)."""

    scope: "CoordinationScope"

    @asynccontextmanager
    async def hold(self, key: str) -> AsyncIterator[None]:
        ...
        yield


__all__: "list[str]" = ["KeyedCoordinator", "LeaseCoordinator", "LeaseToken"]
