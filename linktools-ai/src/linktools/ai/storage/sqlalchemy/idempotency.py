#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyIdempotencyStore: DB-backed IdempotencyStore (Protocol in
tool/idempotency.py). Mirrors SqlAlchemyApprovalStore's structure:
``session_factory: Callable[[], AsyncSession]`` constructor, ``_as_utc``
helper for aiosqlite's naive-datetime round-trip, and the
``_execute_in_session`` UoW hook so the store can participate in cross-store
transactions through SqlAlchemyStorage.transaction().

``reserve`` handles the race via the unique (scope, key) constraint:
INSERT a RESERVED row; on IntegrityError (concurrent insert from another
process) SELECT the winner and hash-check it. This is the multi-process
equivalent of FileIdempotencyStore's asyncio.Lock -- both backends enforce
"at most one RESERVED per (scope, key)" but the SQL backend does it via the
schema rather than an in-process lock."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ToolIdempotencyRow
from ...json import canonical_json
from ...tool.idempotency import (
    ClaimDisposition,
    ClaimResult,
    IdempotencyClaim,
    IdempotencyRecord,
    IdempotencyStatus,
)


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of
    # tzinfo, so reattach UTC on read to match the timezone-aware datetimes
    # IdempotencyRecord is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_record(row: ToolIdempotencyRow) -> IdempotencyRecord:
    return IdempotencyRecord(
        id=row.id,
        scope=row.scope,
        key=row.key,
        request_hash=row.request_hash,
        status=IdempotencyStatus(row.status),
        result=None if row.result_json is None else json.loads(row.result_json),
        error=row.error_text,
        created_at=_as_utc(row.created_at),
        completed_at=_as_utc(row.completed_at),
        owner_id=row.owner_id,
        generation=row.generation or 0,
        claimed_at=_as_utc(row.claimed_at),
        lease_expires_at=_as_utc(row.lease_expires_at),
    )


class SqlAlchemyIdempotencyStore:
    """Multi-process IdempotencyStore backed by SQLAlchemy/AsyncSession.

    Mirrors SqlAlchemyApprovalStore: ``session_factory`` constructor,
    optional shared ``session`` for UoW mode (every method reuses it instead
    of opening its own transaction). The unique (scope, key) constraint
    backs the reserve() race: a duplicate insert raises IntegrityError,
    which we translate into a SELECT of the existing row + hash check."""

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        # UoW mode: when set, every method uses this shared session directly
        # and does NOT open its own session or call session.begin() -- the
        # UoW owns the transaction. None means normal mode.
        self._session = session

    async def _execute_in_session(self, fn):
        """Run ``fn(session)`` in own transaction (normal mode) or against
        the shared session (UoW mode). See SqlAlchemyRunStore."""
        if self._session is not None:
            result = await fn(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            async with session.begin():
                return await fn(session)

    # -- read ----------------------------------------------------------

    async def get(self, scope: str, key: str) -> "IdempotencyRecord | None":
        async def _do(session):
            result = await session.execute(
                select(ToolIdempotencyRow).where(
                    ToolIdempotencyRow.scope == scope,
                    ToolIdempotencyRow.key == key,
                )
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_record(row)

        return await self._execute_in_session(_do)

    # -- claim / complete / fail (fenced) ---------------------------------

    async def claim(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        owner_id: str,
        lease_seconds: float = 300.0,
    ) -> ClaimResult:
        """Fenced claim: read-decide-write in one transaction. Re-claims (lease
        expired or FAILED) use a CAS UPDATE on the prior generation so a
        concurrent claimant cannot clobber a winner."""

        async def _do(session):
            now = datetime.now(timezone.utc)
            lease_at = datetime.fromtimestamp(
                now.timestamp() + lease_seconds, tz=timezone.utc
            )
            q = await session.execute(
                select(ToolIdempotencyRow).where(
                    ToolIdempotencyRow.scope == scope,
                    ToolIdempotencyRow.key == key,
                )
            )
            row = q.scalar_one_or_none()
            if row is None:
                fresh = ToolIdempotencyRow(
                    id=str(uuid.uuid4()),
                    scope=scope,
                    key=key,
                    request_hash=request_hash,
                    status=IdempotencyStatus.RESERVED.value,
                    result_json=None,
                    error_text=None,
                    created_at=now,
                    completed_at=None,
                    expires_at=None,
                    owner_id=owner_id,
                    generation=1,
                    claimed_at=now,
                    lease_expires_at=lease_at,
                )
                session.add(fresh)
                await session.flush()
                return ClaimResult(
                    disposition=ClaimDisposition.ACQUIRED,
                    claim=_claim_from_record(_row_to_record(fresh)),
                )
            if row.request_hash != request_hash:
                return ClaimResult(disposition=ClaimDisposition.CONFLICT)
            if row.status == IdempotencyStatus.COMPLETED.value:
                return ClaimResult(
                    disposition=ClaimDisposition.REPLAY,
                    record=_row_to_record(row),
                )
            if row.status == IdempotencyStatus.RESERVED.value:
                lease_valid = (
                    row.lease_expires_at is not None
                    and _as_utc(row.lease_expires_at) > now
                )
                if lease_valid and row.owner_id == owner_id:
                    return ClaimResult(
                        disposition=ClaimDisposition.ACQUIRED,
                        claim=_claim_from_record(_row_to_record(row)),
                    )
                if lease_valid:
                    return ClaimResult(
                        disposition=ClaimDisposition.IN_PROGRESS,
                        record=_row_to_record(row),
                    )
            # RESERVED+lease-expired OR FAILED: re-claim with generation+1 via a
            # CAS that pins the SOURCE state we read. The WHERE must include
            # status + request_hash + generation (and, for the RESERVED path,
            # that the lease is still expired) -- otherwise a record a concurrent
            # worker moved to COMPLETED (same generation) could be flipped back
            # to RESERVED and its side effect re-run.
            new_gen = (row.generation or 0) + 1
            where_clauses = [
                ToolIdempotencyRow.scope == scope,
                ToolIdempotencyRow.key == key,
                ToolIdempotencyRow.request_hash == request_hash,
                ToolIdempotencyRow.status == row.status,
                ToolIdempotencyRow.generation == (row.generation or 0),
            ]
            if row.status == IdempotencyStatus.RESERVED.value:
                # Only steal if the lease is still expired (a concurrent worker
                # may have re-leased it between our read and the UPDATE). The
                # column round-trips naive on sqlite, so compare against a naive
                # now to avoid the in-memory evaluator's tz mismatch.
                now_naive = now.replace(tzinfo=None)
                where_clauses.append(ToolIdempotencyRow.lease_expires_at <= now_naive)
            upd = (
                update(ToolIdempotencyRow)
                .where(*where_clauses)
                .values(
                    status=IdempotencyStatus.RESERVED.value,
                    result_json=None,
                    error_text=None,
                    completed_at=None,
                    owner_id=owner_id,
                    generation=new_gen,
                    claimed_at=now,
                    lease_expires_at=lease_at,
                )
            )
            proxy = await session.execute(upd)
            refreshed = await session.execute(
                select(ToolIdempotencyRow).where(
                    ToolIdempotencyRow.scope == scope,
                    ToolIdempotencyRow.key == key,
                )
            )
            winner = refreshed.scalar_one()
            if proxy.rowcount == 0:
                # The source state moved under us: re-classify the fresh record
                # so we never return ACQUIRED for a record we no longer own.
                if winner.request_hash != request_hash:
                    return ClaimResult(disposition=ClaimDisposition.CONFLICT)
                if winner.status == IdempotencyStatus.COMPLETED.value:
                    return ClaimResult(
                        disposition=ClaimDisposition.REPLAY,
                        record=_row_to_record(winner),
                    )
                return ClaimResult(
                    disposition=ClaimDisposition.IN_PROGRESS,
                    record=_row_to_record(winner),
                )
            return ClaimResult(
                disposition=ClaimDisposition.ACQUIRED,
                claim=_claim_from_record(_row_to_record(winner)),
            )

        try:
            return await self._execute_in_session(_do)
        except IntegrityError:
            # Concurrent fresh-INSERT collision: another worker inserted the
            # (scope, key) row between our SELECT and our INSERT, so the
            # unique constraint turned our INSERT into IntegrityError. The
            # only INSERT in ``_do`` is the fresh-row path above (the re-claim
            # branch uses a CAS UPDATE, which cannot raise a unique-constraint
            # IntegrityError), so this exception unambiguously means "lost
            # the insert race". The failed transaction (and its session) is
            # poisoned and must not be reused.
            if self._session is not None:
                # UoW mode: the shared transaction is poisoned too -- there is
                # no fresh session to recover in, so propagate (mirrors
                # ApprovalStore.create_or_get_pending).
                raise
            # Normal mode: re-run the fenced read-decide-write in a FRESH
            # session. The row now exists, so this takes the existing-record
            # branch (CONFLICT / REPLAY / IN_PROGRESS / CAS re-claim) instead
            # of inserting again -- there is no second fresh INSERT, hence no
            # second IntegrityError, and the loser gets a stable disposition.
            return await self._execute_in_session(_do)

    async def complete(self, claim: IdempotencyClaim, result: Any) -> None:
        """CAS to COMPLETED only if owner_id + generation still match the claim.
        rowcount != 1 raises LostIdempotencyClaimError (the claim was stolen) --
        never silently succeed."""
        now = datetime.now(timezone.utc)

        async def _do(session):
            proxy = await session.execute(
                update(ToolIdempotencyRow)
                .where(
                    ToolIdempotencyRow.scope == claim.scope,
                    ToolIdempotencyRow.key == claim.key,
                    ToolIdempotencyRow.request_hash == claim.request_hash,
                    ToolIdempotencyRow.owner_id == claim.owner_id,
                    ToolIdempotencyRow.generation == claim.generation,
                    ToolIdempotencyRow.status == IdempotencyStatus.RESERVED.value,
                )
                .values(
                    status=IdempotencyStatus.COMPLETED.value,
                    result_json=canonical_json(result),
                    error_text=None,
                    completed_at=now,
                )
            )
            if proxy.rowcount != 1:
                from ...errors import LostIdempotencyClaimError

                raise LostIdempotencyClaimError(
                    f"complete lost the claim for ({claim.scope}, {claim.key}): "
                    f"owner/generation no longer match"
                )

        await self._execute_in_session(_do)

    async def fail(self, claim: IdempotencyClaim, error: str) -> None:
        now = datetime.now(timezone.utc)

        async def _do(session):
            proxy = await session.execute(
                update(ToolIdempotencyRow)
                .where(
                    ToolIdempotencyRow.scope == claim.scope,
                    ToolIdempotencyRow.key == claim.key,
                    ToolIdempotencyRow.request_hash == claim.request_hash,
                    ToolIdempotencyRow.owner_id == claim.owner_id,
                    ToolIdempotencyRow.generation == claim.generation,
                    ToolIdempotencyRow.status == IdempotencyStatus.RESERVED.value,
                )
                .values(
                    status=IdempotencyStatus.FAILED.value,
                    result_json=None,
                    error_text=error,
                    completed_at=now,
                )
            )
            if proxy.rowcount != 1:
                from ...errors import LostIdempotencyClaimError

                raise LostIdempotencyClaimError(
                    f"fail lost the claim for ({claim.scope}, {claim.key}): "
                    f"owner/generation no longer match"
                )

        await self._execute_in_session(_do)


def _claim_from_record(record: IdempotencyRecord) -> IdempotencyClaim:
    return IdempotencyClaim(
        scope=record.scope,
        key=record.key,
        request_hash=record.request_hash,
        owner_id=record.owner_id or "",
        generation=record.generation,
        claimed_at=record.claimed_at or record.created_at,
        lease_expires_at=record.lease_expires_at or record.created_at,
    )
