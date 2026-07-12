#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval domain models: ApprovalStatus enum, ApprovalRequest record,
ALLOWED_APPROVAL_TRANSITIONS map, the ApprovalStore Protocol, and the
build_approval_request factory.

This phase ships persistence + audit + events + external resolve; full run
pause/resume is deferred. Mirrors the frozen-dataclass + str-Enum +
transition-map + @runtime_checkable Protocol conventions used by
swarm.models / swarm.store."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


ALLOWED_APPROVAL_TRANSITIONS: "Mapping[ApprovalStatus, frozenset[ApprovalStatus]]" = {
    ApprovalStatus.PENDING: frozenset(
        {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}
    ),
    ApprovalStatus.APPROVED: frozenset(),
    ApprovalStatus.REJECTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    reason: "str | None"
    arguments: "Mapping[str, Any]"
    status: ApprovalStatus
    version: int
    created_at: datetime
    resolved_at: "datetime | None"
    resolved_by: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)


def build_approval_request(
    *,
    run_id: str,
    tool_call_id: str,
    tool_name: str,
    reason: "str | None" = None,
    arguments: "Mapping[str, Any] | None" = None,
    approval_id: "str | None" = None,
) -> ApprovalRequest:
    """Mint a PENDING ApprovalRequest (fresh UTC timestamps, version=1).

    ``arguments`` is copied into a plain dict so callers cannot mutate the
    record's state by holding onto the source mapping. ``approval_id``
    lets a caller that already minted an id --
    ToolExecutor mints one for ``RunPaused.approval_id`` before this request
    is ever persisted -- pass it through so the id reported to the caller
    matches the id actually stored. Defaults to a fresh uuid4 when omitted.
    """
    now = datetime.now(timezone.utc)
    return ApprovalRequest(
        id=approval_id if approval_id is not None else str(uuid.uuid4()),
        run_id=run_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        reason=reason,
        arguments=dict(arguments) if arguments is not None else {},
        status=ApprovalStatus.PENDING,
        version=1,
        created_at=now,
        resolved_at=None,
        resolved_by=None,
    )


def check_dedupe_conflict(
    existing: ApprovalRequest,
    *,
    tool_name: str,
    arguments: "Mapping[str, Any]",
) -> None:
    """when
    ``create_or_get_pending()`` finds an existing request for
    ``(run_id, tool_call_id)``, it must not silently hand back a request
    that was actually for a DIFFERENT call -- same dedupe key reused with
    different ``tool_name``/``arguments`` is a conflict, not a replay.
    ``reason`` is deliberately excluded per spec (informational only, not a
    call-identity field). Raises :class:`~linktools.ai.errors.ApprovalConflictError`."""
    from ..errors import ApprovalConflictError

    if existing.tool_name != tool_name or dict(existing.arguments) != dict(arguments):
        raise ApprovalConflictError(
            f"approval dedupe key (run_id={existing.run_id!r}, "
            f"tool_call_id={existing.tool_call_id!r}) already exists with "
            f"different tool_name/arguments"
        )


@runtime_checkable
class ApprovalStore(Protocol):
    """Persistence contract for ApprovalRequest.

    Method signatures are this phase's concrete resolution of the spec's
    ``(...)`` ellipses . ``approve``/``reject`` carry
    ``expected_version`` for optimistic-concurrency control; conflict /
    not-found / invalid-transition cases raise the corresponding errors from
    ``linktools.ai.errors`` (ApprovalConflictError / ApprovalNotFoundError /
    InvalidApprovalTransitionError).

    ``list_pending(run_id)`` filters status==PENDING (the pause UI's queue).
    ``list_for_run(run_id)`` is status-agnostic -- it returns every request
    for the run regardless of status, ordered by created_at. The resume gate
    (``ToolExecutor._already_approved``) consults it to recognize a call that
    was approved externally without re-persisting a PENDING duplicate.

    ``create_or_get_pending`` is the
    dedup-aware entry point the RunPaused-handling suspension path uses
    instead of a bare ``create``: a repeated tool_call_id for the same run_id
    (retry, duplicate model drive, re-entrant pause) returns the EXISTING
    pending/approved request rather than creating a second PENDING one.
    """

    async def create(self, request: ApprovalRequest) -> ApprovalRequest: ...

    async def create_or_get_pending(
        self,
        *,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        reason: "str | None",
        arguments: "Mapping[str, Any]",
        approval_id: str,
    ) -> ApprovalRequest:
        """Return the existing request for ``(run_id, tool_call_id)`` if one
        already exists (PENDING or APPROVED/REJECTED -- any status), else
        persist and return a fresh PENDING one built with ``approval_id``."""
        ...

    async def get(self, approval_id: str) -> "ApprovalRequest | None": ...

    async def approve(
        self, approval_id: str, *, expected_version: int, resolved_by: str
    ) -> ApprovalRequest: ...

    async def reject(
        self,
        approval_id: str,
        *,
        expected_version: int,
        resolved_by: str,
        reason: "str | None" = None,
    ) -> ApprovalRequest: ...

    async def list_pending(self, run_id: str) -> "tuple[ApprovalRequest, ...]": ...

    async def list_for_run(self, run_id: str) -> "tuple[ApprovalRequest, ...]": ...
