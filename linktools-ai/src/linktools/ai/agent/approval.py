#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval domain models: ApprovalStatus enum, ApprovalRequest record,
ALLOWED_APPROVAL_TRANSITIONS map, the ApprovalStore Protocol, and the
build_approval_request factory. Per spec section 28 (approval flow).

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
    ApprovalStatus.PENDING: frozenset({ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}),
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
) -> ApprovalRequest:
    """Mint a PENDING ApprovalRequest (uuid4 id, fresh UTC timestamps, version=1).

    ``arguments`` is copied into a plain dict so callers cannot mutate the
    record's state by holding onto the source mapping.
    """
    now = datetime.now(timezone.utc)
    return ApprovalRequest(
        id=str(uuid.uuid4()),
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


@runtime_checkable
class ApprovalStore(Protocol):
    """Persistence contract for ApprovalRequest.

    Method signatures are this phase's concrete resolution of the spec's
    ``(...)`` ellipses (section 28). ``approve``/``reject`` carry
    ``expected_version`` for optimistic-concurrency control; conflict /
    not-found / invalid-transition cases raise the corresponding errors from
    ``linktools.ai.errors`` (ApprovalConflictError / ApprovalNotFoundError /
    InvalidApprovalTransitionError).
    """

    async def create(self, request: ApprovalRequest) -> ApprovalRequest: ...

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
