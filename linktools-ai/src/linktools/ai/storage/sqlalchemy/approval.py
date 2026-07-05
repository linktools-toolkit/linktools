#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyApprovalStore: DB-backed ApprovalStore (the Protocol in
agent_runtime/approval.py). Mirrors SqlAlchemyMemoryStore's structure:
`session_factory: Callable[[], AsyncSession]` constructor, `_as_utc` helper for
aiosqlite's naive-datetime round-trip, and read-check-mutate-commit transactions.

Rejection reason: ``ApprovalRequest`` has no dedicated field for the rejection
reason, so ``reject(..., reason=...)`` stores it under
``metadata["rejection_reason"]`` (a None reason is still recorded as that key
mapped to None, so callers can distinguish "rejected, no reason given" from
"approved"). Any pre-existing ``metadata`` is preserved; ``approve`` never
touches the metadata so it cannot shadow a prior rejection reason on a
different request."""

import json
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ApprovalRow
from ...agent_runtime.approval import (
    ALLOWED_APPROVAL_TRANSITIONS,
    ApprovalRequest,
    ApprovalStatus,
)
from ...errors import (
    ApprovalConflictError,
    ApprovalNotFoundError,
    InvalidApprovalTransitionError,
)

#: Key under which ``reject(reason=...)`` is recorded in the request's metadata.
REJECTION_REASON_METADATA_KEY = "rejection_reason"


class _Unset:
    """Sentinel distinguishing "approve" (don't touch metadata) from
    "reject" (always record the key, even when reason is None)."""

    __slots__ = ()


_UNSET = _Unset()


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes ApprovalRequest is constructed with everywhere
    # else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _row_to_request(row: ApprovalRow) -> ApprovalRequest:
    return ApprovalRequest(
        id=row.id,
        run_id=row.run_id,
        tool_call_id=row.tool_call_id,
        tool_name=row.tool_name,
        reason=row.reason,
        arguments=json.loads(row.arguments_json),
        status=ApprovalStatus(row.status),
        version=row.version,
        created_at=_as_utc(row.created_at),
        resolved_at=_as_utc(row.resolved_at),
        resolved_by=row.resolved_by,
        metadata=json.loads(row.metadata_json),
    )


class SqlAlchemyApprovalStore:
    """Multi-process ApprovalStore backed by SQLAlchemy/AsyncSession.

    Optimistic concurrency on ``approve`` / ``reject`` mirrors
    ``SqlAlchemyMemoryStore.update`` (read-check-mutate-commit in one
    transaction). ``create`` relies on the primary-key constraint: a duplicate
    id raises ``IntegrityError``, which is translated to
    ``ApprovalConflictError``.
    """

    def __init__(self, *, session_factory: "Callable[[], AsyncSession]") -> None:
        self._session_factory = session_factory

    # -- read ----------------------------------------------------------

    async def get(self, approval_id: str) -> "ApprovalRequest | None":
        async with self._session_factory() as session:
            result = await session.execute(
                select(ApprovalRow).where(ApprovalRow.id == approval_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_request(row)

    async def list_pending(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        async with self._session_factory() as session:
            stmt = select(ApprovalRow).where(
                ApprovalRow.run_id == run_id,
                ApprovalRow.status == ApprovalStatus.PENDING.value,
            ).order_by(ApprovalRow.created_at)
            result = await session.execute(stmt)
            return tuple(_row_to_request(row) for row in result.scalars())

    # -- write ---------------------------------------------------------

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        async with self._session_factory() as session:
            try:
                async with session.begin():
                    session.add(ApprovalRow(
                        id=request.id,
                        run_id=request.run_id,
                        tool_call_id=request.tool_call_id,
                        tool_name=request.tool_name,
                        reason=request.reason,
                        arguments_json=json.dumps(dict(request.arguments)),
                        status=request.status.value,
                        version=request.version,
                        created_at=request.created_at,
                        resolved_at=request.resolved_at,
                        resolved_by=request.resolved_by,
                        metadata_json=json.dumps(dict(request.metadata)),
                    ))
            except IntegrityError as exc:
                # Duplicate primary key -> conflict, matching
                # FileApprovalStore's "approval already exists" semantics.
                raise ApprovalConflictError(
                    f"approval already exists: {request.id}"
                ) from exc
        return request

    async def approve(
        self, approval_id: str, *, expected_version: int, resolved_by: str
    ) -> ApprovalRequest:
        return await self._resolve(
            approval_id,
            target=ApprovalStatus.APPROVED,
            expected_version=expected_version,
            resolved_by=resolved_by,
            rejection_reason=_UNSET,
        )

    async def reject(
        self,
        approval_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        reason: "str | None" = None,
    ) -> ApprovalRequest:
        return await self._resolve(
            approval_id,
            target=ApprovalStatus.REJECTED,
            expected_version=expected_version,
            resolved_by=resolved_by,
            rejection_reason=reason,
        )

    async def _resolve(
        self,
        approval_id: str,
        *,
        target: ApprovalStatus,
        expected_version: int,
        resolved_by: str,
        rejection_reason: "Any",
    ) -> ApprovalRequest:
        async with self._session_factory() as session:
            async with session.begin():
                query_result = await session.execute(
                    select(ApprovalRow).where(ApprovalRow.id == approval_id)
                )
                row = query_result.scalar_one_or_none()
                if row is None:
                    raise ApprovalNotFoundError(f"approval not found: {approval_id}")
                current_status = ApprovalStatus(row.status)
                if row.version != expected_version:
                    raise ApprovalConflictError(
                        f"expected version {expected_version}, found {row.version}"
                    )
                if target not in ALLOWED_APPROVAL_TRANSITIONS.get(current_status, frozenset()):
                    raise InvalidApprovalTransitionError(
                        f"cannot transition {current_status} -> {target}"
                    )
                # ``reject`` always sets the key (even to None); ``approve``
                # leaves metadata untouched so approvals can't shadow a prior
                # rejection reason on a different request.
                if rejection_reason is not _UNSET:
                    new_metadata: "dict[str, Any]" = json.loads(row.metadata_json)
                    new_metadata[REJECTION_REASON_METADATA_KEY] = rejection_reason
                    row.metadata_json = json.dumps(new_metadata)
                now = datetime.now(timezone.utc)
                row.status = target.value
                row.resolved_at = now
                row.resolved_by = resolved_by
                row.version = row.version + 1
                await session.flush()
                return _row_to_request(row)
