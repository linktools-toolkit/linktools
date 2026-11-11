#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic ownership and backend-owned lifetimes of live turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Literal, Protocol, runtime_checkable

from ..core import validate_persistence_namespace, validate_tenant_id
from ._lock import Lease


@dataclass(frozen=True, slots=True)
class TurnScopeIdentity:
    """Exact coordination namespace and bounded purpose within that namespace."""

    namespace: str = field(repr=False)
    tenant_id: str = field(repr=False)
    scope: str = field(repr=False)

    def __post_init__(self) -> None:
        validate_persistence_namespace(self.namespace)
        validate_tenant_id(self.tenant_id)
        self.validate_scope(self.scope)

    @classmethod
    def validate_scope(cls, value: str) -> str:
        """Validate a local scope without changing its exact spelling."""
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 256:
            raise ValueError("turn scope must contain 1..256 UTF-8 bytes")
        return value

    def as_tuple(self) -> tuple[str, str, str]:
        return self.namespace, self.tenant_id, self.scope


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
class TurnScope(Protocol):
    """Single-use ownership lifetime, entered and exited by the same task.

    Entry preserves claim's busy/replay/owned outcomes. Only an owned outcome
    permits validity checks and signal waits. Validity is local and synchronous;
    owner loss is permanent. Cancellation is delivered once and does not release
    ownership or stop renewal. A subsequent wait can observe owner loss.
    Exit settles renewal and releases the exact lease, without suppressing the
    body's exception. Callers must settle their own work before exiting.
    """

    async def __aenter__(self) -> TurnCoordination | None: ...

    def require_active(self) -> None:
        """Raise AIError(STORAGE_CONFLICT) when this ownership is no longer valid."""
        ...

    async def wait_signal(self) -> Literal["owner_lost", "cancel_requested"]: ...

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class TurnCoordinationBackend(Protocol):
    """One optional coordination capability; all six methods must be implemented.

    ``scope`` explicitly identifies the namespace, tenant, and local purpose. Session,
    request, and execution identities are isolated by that scope. A request key
    binds both its session key and request digest while the claim is live.
    Execution IDs identify logical turns; owner IDs identify fresh ownership
    generations. After release or expiry, the same logical turn may be claimed
    by a new owner with its original session/request identity. Live claims are
    never preempted: duplicate claims return observation, not a new lease.
    Lease keys are backend-owned opaque values and callers must not interpret them.

    Implementations atomically maintain session exclusion, request replay, and
    execution lookup. Reads, duplicates, and cancellation do not renew expiry.
    Expired or released claims no longer provide replay or ownership authority;
    durable history remains the source of completed request results.

    Backend failures propagate, including ambiguous mutation outcomes. They
    must never be converted into absence, contention, or permission to execute.
    This capability does not define a byte codec or require a Redis dependency.
    """

    def enter(
        self, scope: TurnScopeIdentity, *, session_key: str, request_key: str,
        request_digest: str, execution_id: str, owner_id: str,
    ) -> TurnScope:
        """Create a lifetime using backend-owned lease and renewal policy."""
        ...

    async def claim_turn(
        self,
        scope: TurnScopeIdentity,
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
        Identity strings must be nonempty; lease_seconds must be finite and
        positive. A replay ignores the proposed execution and owner identities.
        """
        ...

    async def read_turn(self, scope: TurnScopeIdentity, execution_id: str) -> TurnCoordination | None:
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

    async def cancel_turn(self, scope: TurnScopeIdentity, execution_id: str) -> bool:
        """Cancel the logical execution's current owner; False means absent.

        A delayed cancellation also targets a recovered owner of that same
        execution. Repeating cancellation of a live target returns True.
        Cancellation neither releases ownership nor renews its expiry.
        """
        ...


@runtime_checkable
class TurnAdmissionReader(Protocol):
    """Optional indexed reads of live admissions without claiming ownership."""

    async def read_turn_by_request(
        self, scope: TurnScopeIdentity, request_key: str,
    ) -> TurnCoordination | None:
        """Return a live request's token-free owner, or None if absent.

        Reads must not claim, renew, or release ownership. Backend failures
        propagate rather than being interpreted as absence.
        """
        ...


@runtime_checkable
class TurnSessionReader(Protocol):
    """Optional authoritative reads of the existing live session admission."""

    async def read_turn_by_session(
        self, scope: TurnScopeIdentity, session_key: str,
    ) -> TurnCoordination | None:
        """Atomically observe a session's token-free live owner, or its absence.

        Read the ownership index, not a disposable display projection. None
        means no live claim at the observation point; it does not prevent a
        later claim. Inconsistent indexes and backend failures remain errors.
        Reads must not claim, renew or release ownership. All identity strings
        must be nonempty. This capability is independent of request lookup.
        """
        ...


__all__ = ["TurnScopeIdentity", "TurnAdmissionReader", "TurnCoordination", "TurnCoordinationBackend", "TurnSessionReader", "TurnScope"]
