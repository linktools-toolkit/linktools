#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live-turn coordination with optional non-authoritative cache views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from ..core import (
    ExecutionStatus,
    canonical_json_bytes,
    canonical_sha256,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import CacheBackend, RenewingTurnScope, StreamBackend, TurnAdmissionReader, TurnCoordination, TurnCoordinationBackend, TurnSessionReader, TurnScopeIdentity
from .codec import decode_domain, encode_domain
from .turns import StateTurnCoordination
from ._turn_sql_schema import RUNTIME_TURN_SCOPE
from ..execution import ExecutionRecord


if TYPE_CHECKING:
    from ..storage import TurnScope
    from ._runtime_commands import StateCommands
    from .contracts import ExecutionStartReservation, ExecutionStartReservationResult, IdempotencyRecord, IdempotencyRepository, SessionRecord


_PROMOTION_INTENT = b"promote"


class PendingTurnCoordination:
    """Coordinate pending turns without owning lifecycle or durable state.

    Ownership is independent of optional cache views on the borrowed backend.
    There is no implicit process-local substitute for missing ownership.
    Observation failures propagate to the caller; they never authorize a claim.
    """

    def __init__(self, cache: CacheBackend | None, backend: TurnCoordinationBackend, *, namespace: str, tenant_id: str) -> None:
        self._state_authority: StateTurnCoordination | None = None
        self._cache = cache
        self._backend = backend
        self._namespace = validate_persistence_namespace(namespace)
        self._tenant_id = validate_tenant_id(tenant_id)
        self._scope = TurnScopeIdentity(self._namespace, self._tenant_id, RUNTIME_TURN_SCOPE)
        self._scope_payload = list(self._scope.as_tuple())

    @classmethod
    def select(
        cls, cache: CacheBackend | TurnCoordinationBackend | StreamBackend | None, *, namespace: str, tenant_id: str,
    ) -> PendingTurnCoordination | None:
        if not isinstance(cache, TurnCoordinationBackend):
            return None
        return cls(
            cache if isinstance(cache, CacheBackend) else None, cache,
            namespace=namespace, tenant_id=tenant_id,
        )

    @classmethod
    def for_state(
        cls, cache: CacheBackend | None, backend: StateTurnCoordination, *, namespace: str, tenant_id: str,
    ) -> PendingTurnCoordination:
        bound = cls(cache, backend, namespace=namespace, tenant_id=tenant_id)
        bound._state_authority = backend
        return bound

    @property
    def supports_pending_views(self) -> bool:
        """Whether the backend can carry disposable pending views and hints."""
        return self._cache is not None

    @property
    def supports_request_lookup(self) -> bool:
        return isinstance(self._backend, TurnAdmissionReader)

    def enter(
        self, *, session_key: str, request_key: str, request_digest: str,
        execution_id: str, owner_id: str,
    ) -> TurnScope:
        """Enter backend-owned admission using the same scope as all readers."""
        return self._backend.enter(
            self._scope, session_key=session_key, request_key=request_key,
            request_digest=request_digest, execution_id=execution_id,
            owner_id=owner_id,
        )

    def can_group_start(
        self, commands: StateCommands, reservation: ExecutionStartReservation, session: SessionRecord,
    ) -> bool:
        authority = self._state_authority
        return authority is not None and commands.can_claim_and_reserve_start(authority, reservation, session)

    def enter_grouped_start(
        self, commands: StateCommands, reservation: ExecutionStartReservation, session: SessionRecord, *,
        session_key: str, request_key: str, owner_id: str,
        capture: Callable[[ExecutionStartReservationResult], None],
    ) -> TurnScope:
        authority = self._state_authority
        assert authority is not None
        execution_id = reservation.execution.execution_id
        digest = reservation.idempotency.request_digest
        claim_identity = dict(session_key=session_key, request_key=request_key,
                              request_digest=digest, execution_id=execution_id, owner_id=owner_id)

        async def claim(duration: float) -> TurnCoordination | None:
            result = await commands.claim_and_reserve_start(
                authority, self._scope, reservation, expected_session=session, session_key=session_key,
                request_key=request_key, owner_id=owner_id, lease_seconds=duration,
            )
            if result is None:
                return await authority.claim_turn(self._scope, **claim_identity, lease_seconds=duration)
            coordination, receipt = result
            if receipt is not None:
                capture(receipt)
            return coordination

        return RenewingTurnScope(authority, self._scope, **claim_identity, _claim=claim)

    async def publish(
        self, record: ExecutionRecord, coordination: TurnCoordination,
        *, request_identity_version: int = 2,
    ) -> None:
        """Publish an owner's pending observation, never a terminal result.

        Publication neither confirms ownership nor promises cache retention.
        Readers confirm authority after reading these generation-scoped bytes.
        """
        if type(request_identity_version) is not int or request_identity_version not in (1, 2):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if coordination.lease is None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if record.execution_id != coordination.execution_id or not self._is_pending(record):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._cache is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"reason": "pending_views_not_supported"},
            )
        envelope = {
            "scope": self._scope_payload,
            "execution_id": coordination.execution_id,
            "owner_id": coordination.owner_id,
            "principal_id": record.principal_id,
            "principal_kind": record.principal_kind,
            "request_identity_version": request_identity_version,
            "record": encode_domain(replace(record, input_context=None)),
        }
        state_fence = ({"fence": coordination.lease.fence}
                       if type(self._state_authority) is StateTurnCoordination else {})
        envelope.update(state_fence)
        await self._cache.put(self._record_key(coordination), canonical_json_bytes(envelope))
        if record.session_id is not None:
            # This index is deliberately not deleted on release: a delayed old
            # owner must never delete a new owner's observation. Readers still
            # validate live authority, so stale or reordered pointers grant none.
            await self._cache.put(
                self._session_key(record.session_id),
                canonical_json_bytes({
                    "scope": self._scope_payload,
                    "session_id": record.session_id,
                    "execution_id": coordination.execution_id,
                    "owner_id": coordination.owner_id,
                    **state_fence,
                }),
            )

    async def read_owner(
        self, execution_id: str, *, tenant_id: str,
    ) -> TurnCoordination | None:
        """Read live identity without its lease, even before a view is published."""
        if tenant_id != self._tenant_id:
            return None
        return await self._backend.read_turn(self._scope, execution_id)

    async def read_request_execution(
        self, request_key: str, *, tenant_id: str,
    ) -> ExecutionRecord | None:
        """Read an admitted request without turning an unavailable view into absence."""
        current = await self._read_request(request_key, tenant_id=tenant_id)
        return None if current is None else current[0]

    async def read_request_identity_version(self, request_key: str, *, tenant_id: str) -> int | None:
        """Read the live owner's projection version without inferring absence from a lost view."""
        current = await self._read_request(request_key, tenant_id=tenant_id)
        return None if current is None else current[2]

    async def _read_request(
        self, request_key: str, *, tenant_id: str,
    ) -> tuple[ExecutionRecord, TurnCoordination, int] | None:
        if tenant_id != self._tenant_id:
            return None
        if not isinstance(self._backend, TurnAdmissionReader):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"reason": "admission_lookup_not_supported"})
        owner = await self._backend.read_turn_by_request(self._scope, request_key)
        if owner is None:
            return None
        current = await self._read_execution(owner.execution_id, owner_id=owner.owner_id)
        if current is None:
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        return current

    async def _read_state_admission(
        self, authority: StateTurnCoordination, requests: IdempotencyRepository, *,
        idempotency_scope: str, idempotency_key: str, request_key: str,
    ) -> tuple[IdempotencyRecord | None, ExecutionRecord | None]:
        async def capture() -> tuple[IdempotencyRecord | None, TurnCoordination | None]:
            return await authority.read_admission_request(
                requests, self._scope, idempotency_scope=idempotency_scope,
                idempotency_key=idempotency_key, request_key=request_key,
            )

        receipt, owner = await capture()
        if receipt is not None or owner is None:
            return receipt, None
        # Do not hold the storage snapshot across a disposable cache read;
        # a fresh source capture confirms the generation or finds its durable receipt.
        pending = await self._read_pending_bytes(owner)
        receipt, confirmed = await capture()
        if receipt is not None:
            return receipt, None
        if pending is None or not self._same_generation(confirmed, owner):
            raise AIError(ErrorCode.EXECUTION_START_UNKNOWN)
        return None, pending[0]

    async def read_session_owner(
        self, session_id: str, *, tenant_id: str,
    ) -> TurnCoordination | None:
        """Read authoritative session admission, never the lossy display pointer.

        A configured backend without this optional capability cannot prove
        absence. Observation never claims, renews or releases a turn.
        """
        if tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if not isinstance(self._backend, TurnSessionReader):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"reason": "session_admission_lookup_not_supported"},
            )
        return await self._backend.read_turn_by_session(self._scope, f"session:{session_id}")

    async def read_execution(
        self, execution_id: str, *, tenant_id: str,
    ) -> ExecutionRecord | None:
        """Read a live observation; service authorization still uses its principal."""
        if tenant_id != self._tenant_id:
            return None
        current = await self._read_execution(execution_id)
        return None if current is None else current[0]

    async def read_stream(
        self, execution_id: str, *, tenant_id: str,
    ) -> tuple[ExecutionRecord, int, str] | None:
        """Bind an activated pending stream to its confirmed live owner."""
        if tenant_id != self._tenant_id:
            return None
        current = await self._read_execution(execution_id)
        if current is None:
            return None
        record, coordination, _ = current
        if (record.agent_run_seq != 1 or record.producer_generation < 1
                or record.producer_claim_id is None
                or record.producer_claim_id != coordination.owner_id):
            return None
        return record, record.producer_generation, record.producer_claim_id

    async def read_session_execution(
        self, session_id: str, *, tenant_id: str,
    ) -> ExecutionRecord | None:
        """Resolve a session observation without treating absence as admission."""
        if tenant_id != self._tenant_id:
            return None
        if self._cache is None:
            return None
        content = await self._cache.get(self._session_key(session_id))
        if content is None:
            return None
        try:
            pointer = json.loads(content)
            if (
                not isinstance(pointer, dict)
                or pointer.get("scope") != self._scope_payload
                or pointer.get("session_id") != session_id
                or not isinstance(pointer.get("execution_id"), str)
                or not pointer["execution_id"]
                or not isinstance(pointer.get("owner_id"), str)
                or not pointer["owner_id"]
            ):
                return None
        except (TypeError, ValueError):
            return None
        authority = self._state_authority
        fence = pointer.get("fence")
        if type(authority) is StateTurnCoordination and type(fence) is int and fence > 0:
            # Cache supplies only an address. Confirm the scoped, live session
            # generation after the read; the fence also rejects owner-ID reuse.
            hint = TurnCoordination(pointer["execution_id"], pointer["owner_id"], False, None)
            cache_error = None
            try:
                pending = await self._read_pending_bytes(hint, fence=fence)
            except Exception as error:
                cache_error = error
                pending = None
            if not await authority.matches_session_generation(
                self._scope, f"session:{session_id}", hint.execution_id, hint.owner_id, fence,
            ):
                return None
            if cache_error is not None:
                raise cache_error
            record = None if pending is None else pending[0]
        else:
            current = await self._read_execution(pointer["execution_id"], owner_id=pointer["owner_id"])
            record = None if current is None else current[0]
        return record if record is not None and record.session_id == session_id else None

    async def _read_execution(
        self, execution_id: str, *, owner_id: str | None = None,
    ) -> tuple[ExecutionRecord, TurnCoordination, int] | None:
        current = await self._backend.read_turn(self._scope, execution_id)
        if (
            current is None or current.execution_id != execution_id
            or (owner_id is not None and current.owner_id != owner_id)
        ):
            return None
        pending = await self._read_pending_bytes(current)
        if pending is None:
            return None
        confirmed = await self._backend.read_turn(self._scope, execution_id)
        if confirmed is None or not self._same_generation(confirmed, current):
            return None
        record, identity_version = pending
        return record, confirmed, identity_version

    async def _read_pending_bytes(
        self, current: TurnCoordination, *, fence: int | None = None,
    ) -> tuple[ExecutionRecord, int] | None:
        if self._cache is None:
            return None
        content = await self._cache.get(self._record_key(current))
        if content is None:
            return None
        try:
            envelope = json.loads(content)
            if (
                not isinstance(envelope, dict)
                or envelope.get("scope") != self._scope_payload
                or envelope.get("execution_id") != current.execution_id
                or envelope.get("owner_id") != current.owner_id
                or (fence is not None and (type(envelope.get("fence")) is not int or envelope["fence"] != fence))
            ):
                return None
            identity_version = envelope.get("request_identity_version", 1)
            if type(identity_version) is not int or identity_version not in (1, 2):
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
            record = decode_domain(envelope.get("record"), ExecutionRecord)
            if (
                record.execution_id != current.execution_id
                or record.principal_id != envelope.get("principal_id")
                or record.principal_kind != envelope.get("principal_kind")
                or not self._is_pending(record)
            ):
                return None
        except (AIError, TypeError, ValueError):
            return None
        return record, identity_version

    async def cancel(self, execution_id: str, *, tenant_id: str) -> bool:
        if tenant_id != self._tenant_id:
            return False
        return await self._backend.cancel_turn(self._scope, execution_id)

    async def request_promotion(self, execution_id: str, *, tenant_id: str) -> str | None:
        """Offer a lossy hint to this owner, without authorizing any effect."""
        if tenant_id != self._tenant_id:
            return None
        current = await self.read_owner(execution_id, tenant_id=tenant_id)
        if current is None or current.execution_id != execution_id:
            return None
        if self._cache is None:
            return None
        await self._cache.put(self._promotion_key(current), _PROMOTION_INTENT)
        if not await self.promotion_requested(current):
            return None
        confirmed = await self.read_owner(execution_id, tenant_id=tenant_id)
        return current.owner_id if self._same_generation(confirmed, current) else None

    async def promotion_requested(self, coordination: TurnCoordination) -> bool:
        """Read a lossy hint without granting or extending ownership."""
        return (self._cache is not None
                and await self._cache.get(self._promotion_key(coordination)) == _PROMOTION_INTENT)

    async def remote_promotion_pending(
        self, execution_id: str, owner_id: str, *, tenant_id: str,
    ) -> bool:
        """Observe this live owner's hint, never a durable promotion receipt."""
        if tenant_id != self._tenant_id:
            return False
        current = await self.read_owner(execution_id, tenant_id=tenant_id)
        if (
            current is None or current.execution_id != execution_id
            or current.owner_id != owner_id
        ):
            return False
        if not await self.promotion_requested(current):
            return False
        confirmed = await self.read_owner(execution_id, tenant_id=tenant_id)
        return self._same_generation(confirmed, current)

    def _promotion_key(self, coordination: TurnCoordination) -> str:
        digest = hashlib.sha256(canonical_json_bytes([
            self._scope_payload, coordination.execution_id, coordination.owner_id,
        ])).hexdigest()
        return f"runtime-pending-promotion:{digest}"

    def _record_key(self, coordination: TurnCoordination) -> str:
        digest = hashlib.sha256(canonical_json_bytes([
            self._scope_payload, coordination.execution_id, coordination.owner_id,
        ])).hexdigest()
        return f"runtime-pending-execution:{digest}"

    def _session_key(self, session_id: str) -> str:
        digest = hashlib.sha256(canonical_json_bytes([self._scope_payload, session_id])).hexdigest()
        return f"runtime-pending-session:{digest}"

    @classmethod
    def _same_generation(
        cls, current: TurnCoordination | None, expected: TurnCoordination,
    ) -> bool:
        return current is not None and (
            current.execution_id, current.owner_id
        ) == (expected.execution_id, expected.owner_id)

    @classmethod
    def _is_pending(cls, record: ExecutionRecord) -> bool:
        return record.status in (
            ExecutionStatus.PENDING_START, ExecutionStatus.STARTED, ExecutionStatus.CANCELLING,
        ) and record.result is None




async def read_admission_request(
    requests: IdempotencyRepository, coordination: PendingTurnCoordination | None, *,
    scope: str, key: str, tenant_id: str, local_pending: bool = False,
) -> tuple[IdempotencyRecord | None, ExecutionRecord | None]:
    """Resolve one request without reserving, settling, or recovering a turn."""
    if coordination is not None and coordination._tenant_id == tenant_id and not local_pending:
        request_key = canonical_sha256([scope, key])
        if coordination._state_authority is not None:
            return await coordination._read_state_admission(
                coordination._state_authority, requests, idempotency_scope=scope,
                idempotency_key=key, request_key=request_key,
            )
    else:
        coordination = None
    receipt = await requests.get(scope, key, tenant_id=tenant_id)
    if receipt is not None or coordination is None:
        return receipt, None
    try:
        observed = await coordination._read_request(request_key, tenant_id=tenant_id)
    except AIError as error:
        if error.code is not ErrorCode.EXECUTION_START_UNKNOWN:
            raise
        receipt = await requests.get(scope, key, tenant_id=tenant_id)
        if receipt is None:
            raise
        return receipt, None
    if observed is not None:
        return None, observed[0]
    # Independent authorities cannot prove absence from one consistent read.
    return await requests.get(scope, key, tenant_id=tenant_id), None


__all__ = ["PendingTurnCoordination", "read_admission_request"]
