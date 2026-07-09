#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyApprovalStore: DB-backed ApprovalStore (the Protocol in
agent/approval.py). Mirrors SqlAlchemyMemoryStore's structure:
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
from ...agent.approval import (
    ALLOWED_APPROVAL_TRANSITIONS,
    ApprovalRequest,
    ApprovalStatus,
    build_approval_request,
    check_dedupe_conflict,
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

    def __init__(
        self,
        *,
        session_factory: "Callable[[], AsyncSession]",
        session: "AsyncSession | None" = None,
    ) -> None:
        self._session_factory = session_factory
        # UoW mode: when set, every method uses this shared session directly and
        # does NOT open its own session or call session.begin() -- the UoW owns
        # the transaction. None means normal mode (own session + transaction).
        self._session = session

    async def _execute_in_session(self, fn):
        """Run ``fn(session)`` in own transaction (normal mode) or against the
        shared session (UoW mode). See SqlAlchemyRunStore._execute_in_session."""
        if self._session is not None:
            result = await fn(self._session)
            await self._session.flush()
            return result
        async with self._session_factory() as session:
            async with session.begin():
                return await fn(session)

    # -- read ----------------------------------------------------------

    async def get(self, approval_id: str) -> "ApprovalRequest | None":
        async def _do(session):
            result = await session.execute(
                select(ApprovalRow).where(ApprovalRow.id == approval_id)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_request(row)
        return await self._execute_in_session(_do)

    async def list_pending(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        async def _do(session):
            stmt = select(ApprovalRow).where(
                ApprovalRow.run_id == run_id,
                ApprovalRow.status == ApprovalStatus.PENDING.value,
            ).order_by(ApprovalRow.created_at)
            result = await session.execute(stmt)
            return tuple(_row_to_request(row) for row in result.scalars())
        return await self._execute_in_session(_do)

    async def list_for_run(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        # Status-agnostic counterpart to ``list_pending``: returns EVERY
        # request for the run regardless of status, ordered by created_at.
        # The resume gate (ToolExecutor._already_approved) consults this to
        # recognize a call that was approved externally without re-persisting
        # a PENDING duplicate.
        async def _do(session):
            stmt = select(ApprovalRow).where(
                ApprovalRow.run_id == run_id
            ).order_by(ApprovalRow.created_at)
            result = await session.execute(stmt)
            return tuple(_row_to_request(row) for row in result.scalars())
        return await self._execute_in_session(_do)

    # -- write ---------------------------------------------------------

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        async def _do(session):
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
        try:
            await self._execute_in_session(_do)
        except IntegrityError as exc:
            # Duplicate primary key -> conflict, matching FileApprovalStore's
            # "approval already exists" semantics. In UoW mode the IntegrityError
            # has already poisoned the shared transaction (it will roll back);
            # we still translate so callers see the domain error type.
            raise ApprovalConflictError(
                f"approval already exists: {request.id}"
            ) from exc
        return request

    async def _find_by_run_and_tool_call(
        self, run_id: str, tool_call_id: str,
    ) -> "ApprovalRequest | None":
        async def _do(session):
            result = await session.execute(
                select(ApprovalRow)
                .where(ApprovalRow.run_id == run_id, ApprovalRow.tool_call_id == tool_call_id)
                .order_by(ApprovalRow.created_at.desc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return None if row is None else _row_to_request(row)
        return await self._execute_in_session(_do)

    async def create_or_get_pending(
        self, *, run_id: str, tool_call_id: str, tool_name: str,
        reason: "str | None", arguments: "dict[str, Any]", approval_id: str,
    ) -> ApprovalRequest:
        """Dedup on (run_id, tool_call_id) (Package 3, actionable-fix-spec
        §6): a retry, a duplicate model drive, or a re-entrant pause for the
        same tool_call reuses the existing request rather than creating a
        second PENDING one. The SELECT-then-INSERT below is only the fast
        path -- ``ai_approvals``'s ``uq_approval_run_tool_call`` UNIQUE
        constraint is the actual backstop against two concurrent callers
        both passing the SELECT check and both inserting. On the (rare)
        collision, this method re-selects and returns the winner instead of
        raising, so both callers observe the SAME persisted request."""
        existing = await self._find_by_run_and_tool_call(run_id, tool_call_id)
        if existing is not None:
            check_dedupe_conflict(existing, tool_name=tool_name, arguments=arguments)
            return existing

        request = build_approval_request(
            run_id=run_id, tool_call_id=tool_call_id, tool_name=tool_name,
            reason=reason, arguments=arguments, approval_id=approval_id,
        )

        def _row() -> ApprovalRow:
            return ApprovalRow(
                id=request.id, run_id=request.run_id, tool_call_id=request.tool_call_id,
                tool_name=request.tool_name, reason=request.reason,
                arguments_json=json.dumps(dict(request.arguments)),
                status=request.status.value, version=request.version,
                created_at=request.created_at, resolved_at=request.resolved_at,
                resolved_by=request.resolved_by, metadata_json=json.dumps(dict(request.metadata)),
            )

        # NOTE on NOT using session.begin_nested() here: a SAVEPOINT that
        # releases cleanly (no conflict) was measured to NOT properly
        # participate in a LATER, unrelated failure's rollback of the
        # enclosing UnitOfWork transaction under sqlite+aiosqlite (the
        # released savepoint's row survives even though the outer
        # transaction as a whole rolls back) -- this is the documented
        # pysqlite "implicit transaction" quirk SQLAlchemy normally papers
        # over with a connect/begin event-listener workaround, which this
        # library cannot install on a caller-supplied engine. Since
        # AgentRunner's pause path depends on the approval write ACTUALLY
        # rolling back when a later checkpoint/event write in the same UoW
        # fails (see tests/ai/agent/test_runner_pause_atomic.py's rollback
        # test), that guarantee matters more than isolating the (rare --
        # RunController's one-task-per-run invariant makes a genuine
        # concurrent same-(run_id, tool_call_id) race architecturally
        # unreachable in practice) conflict path. A real conflict here
        # therefore still fails the whole enclosing transaction (propagates
        # to AgentRunner's generic except-Exception handler -> Run FAILED)
        # rather than gracefully continuing within it -- the
        # uq_approval_run_tool_call UNIQUE constraint remains the actual
        # data-integrity backstop regardless.
        try:
            if self._session is not None:
                self._session.add(_row())
                await self._session.flush()
            else:
                async def _insert(session):
                    session.add(_row())
                await self._execute_in_session(_insert)
            return request
        except IntegrityError:
            # Concurrent create_or_get_pending won the race between our
            # SELECT and INSERT (normal mode only -- see the note above for
            # why UoW mode does not attempt to recover from this here).
            existing = await self._find_by_run_and_tool_call(run_id, tool_call_id)
            if existing is not None:
                check_dedupe_conflict(existing, tool_name=tool_name, arguments=arguments)
                return existing
            raise

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
        async def _do(session):
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
        return await self._execute_in_session(_do)
