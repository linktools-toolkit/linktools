#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional atomic ownership of live turns on a borrowed cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._lock import Lease


@dataclass(frozen=True, slots=True)
class TurnCoordination:
    """Live execution identity, cancellation state, and optional ownership proof.

    Only a newly acquired claim or successful owner renewal contains a lease.
    Duplicate claims and reads never reveal an ownership token.
    """

    execution_id: str
    owner_id: str
    cancel_requested: bool
    lease: Lease | None = None


@runtime_checkable
class TurnCoordinationBackend(Protocol):
    """One optional cache capability; all five methods must be implemented.

    ``scope`` is the caller's opaque tenant-and-Runtime namespace. Session,
    request, and execution identities are isolated by that scope. A request key
    binds both its session key and request digest while the claim is live.
    Execution and owner IDs must be fresh for every new ownership generation;
    execution IDs must never be reused after release or expiry. Lease keys are
    backend-owned opaque values and callers must not interpret them.

    Implementations atomically maintain session exclusion, request replay, and
    execution lookup. Reads, duplicates, and cancellation do not renew expiry.
    Expired or released claims no longer provide replay or ownership authority;
    durable history remains the source of completed request results.

    Backend failures propagate, including ambiguous mutation outcomes. They
    must never be converted into absence, contention, or permission to execute.
    This capability does not define a byte codec or require a Redis dependency.
    """

    async def claim_turn(
        self,
        scope: str,
        *,
        session_key: str,
        request_key: str,
        request_digest: str,
        execution_id: str,
        owner_id: str,
        lease_seconds: float,
    ) -> TurnCoordination | None:
        """Claim a free session or replay its live request without a lease.

        Return None for a busy session. A retained request with a different
        session or digest raises AIError(IDEMPOTENCY_CONFLICT). Reusing a live
        execution ID for a distinct request raises AIError(STORAGE_CONFLICT).
        All strings must be nonempty, and lease_seconds must be finite and
        positive. A replay ignores the proposed execution and owner identities.
        """
        ...

    async def read_turn(self, scope: str, execution_id: str) -> TurnCoordination | None:
        """Read a live execution without its lease; return None if absent."""
        ...

    async def renew_turn(
        self, lease: Lease, *, lease_seconds: float,
    ) -> TurnCoordination | None:
        """Renew the exact live lease and read cancellation; None means lost."""
        ...

    async def release_turn(self, lease: Lease) -> bool:
        """Release only this live lease; stale, expired, or absent returns False."""
        ...

    async def cancel_turn(self, scope: str, execution_id: str) -> bool:
        """Mark this live execution cancelled; return False if it is absent.

        Repeating cancellation of a live target returns True. Cancellation
        neither releases ownership nor renews its expiry.
        """
        ...


__all__ = ["TurnCoordination", "TurnCoordinationBackend"]
