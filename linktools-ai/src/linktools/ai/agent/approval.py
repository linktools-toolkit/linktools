#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval domain models: ApprovalStatus enum, ApprovalRequest record,
ALLOWED_APPROVAL_TRANSITIONS map, the ApprovalStore Protocol, and the
build_approval_request factory.

This module provides persistence + audit + events + external resolve for the
run pause/resume flow. Mirrors the frozen-dataclass + str-Enum +
transition-map + @runtime_checkable Protocol conventions used by
swarm.models / swarm.store."""

import hashlib
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


def compute_arguments_hash(
    tool_name: str,
    arguments: "Mapping[str, Any]",
    arguments_hash: "str | None" = None,
) -> str:
    """Stable identity hash for a tool call: SHA-256 over the canonical JSON
    of ``{"tool": tool_name, "arguments": arguments}``. Canonical encoding
    makes two argument dicts that compare equal hash identically regardless of
    key order. Computed over the REAL arguments (never the redacted audit
    copy) so two calls differing only in a secret value are distinct calls,
    not a false conflict."""
    from ..json import canonical_json

    payload = canonical_json({"tool": tool_name, "arguments": dict(arguments)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A persisted request to approve a tool call. The call's REAL arguments
    are never persisted here (they may carry secrets): only the
    ``redacted_arguments`` audit copy and the ``arguments_hash`` identity
    fingerprint are stored. The handler receives the real arguments in
    memory (on resume the model re-emits them from message history), so this
    record is for approval/audit only, never for re-driving the call."""

    id: str
    run_id: str
    tool_call_id: str
    tool_name: str
    reason: "str | None"
    redacted_arguments: "Mapping[str, Any]"
    arguments_hash: str
    status: ApprovalStatus
    version: int
    created_at: datetime
    resolved_at: "datetime | None"
    resolved_by: "str | None"
    metadata: "Mapping[str, Any]" = field(default_factory=dict)
    tenant_id: "str | None" = None
    descriptor_fingerprint: "str | None" = None
    handler_revision: "str | None" = None
    provider_revision: "str | None" = None
    policy_revision: "str | None" = None
    capability_revision: "str | None" = None
    result_processor_revision: "str | None" = None
    schema_version: int = 1
    binding: "Mapping[str, Any]" = field(default_factory=dict)
    binding_fingerprint: str = ""


def build_approval_request(
    *,
    run_id: str,
    tool_call_id: str,
    tool_name: str,
    reason: "str | None" = None,
    arguments: "Mapping[str, Any] | None" = None,
    approval_id: "str | None" = None,
    tenant_id: "str | None" = None,
    descriptor_fingerprint: "str | None" = None,
    handler_revision: "str | None" = None,
    provider_revision: "str | None" = None,
    policy_revision: "str | None" = None,
    capability_revision: "str | None" = None,
    result_processor_revision: "str | None" = None,
    arguments_hash: "str | None" = None,
) -> ApprovalRequest:
    """Mint a PENDING ApprovalRequest (fresh UTC timestamps, version=1).

    ``arguments`` is the REAL call payload (may contain secrets). It is
    redacted via :func:`redact_for_audit` before it reaches the record, and a
    fingerprint (``arguments_hash``) over the real arguments is stored so the
    call can be identified for dedupe / drift detection without persisting
    the secrets. ``approval_id`` lets a caller that already minted an id --
    ToolExecutor mints one for ``RunPaused.approval_id`` before this request
    is ever persisted -- pass it through so the id reported to the caller
    matches the id actually stored. Defaults to a fresh uuid4 when omitted.
    """
    from ..security.redact import redact_for_audit

    now = datetime.now(timezone.utc)
    raw_args = dict(arguments) if arguments is not None else {}
    from ..tool.binding import ToolExecutionBinding
    effective_arguments_hash = arguments_hash or compute_arguments_hash(tool_name, raw_args)
    revisions = (descriptor_fingerprint, handler_revision, provider_revision,
                 policy_revision, capability_revision, result_processor_revision)
    binding = None
    if all(isinstance(value, str) and value for value in revisions):
        binding = ToolExecutionBinding(schema_version=1, tool_name=tool_name,
            arguments_hash=effective_arguments_hash,
            descriptor_fingerprint=descriptor_fingerprint,
            handler_revision=handler_revision, provider_revision=provider_revision,
            policy_revision=policy_revision, capability_revision=capability_revision,
            result_processor_revision=result_processor_revision)
    return ApprovalRequest(
        id=approval_id if approval_id is not None else str(uuid.uuid4()),
        run_id=run_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        reason=reason,
        redacted_arguments=redact_for_audit(raw_args),
        arguments_hash=effective_arguments_hash,
        status=ApprovalStatus.PENDING,
        version=1,
        created_at=now,
        resolved_at=None,
        resolved_by=None,
        tenant_id=tenant_id,
        descriptor_fingerprint=descriptor_fingerprint,
        handler_revision=handler_revision,
        provider_revision=provider_revision,
        policy_revision=policy_revision,
        capability_revision=capability_revision,
        result_processor_revision=result_processor_revision,
        binding={} if binding is None else binding.to_payload(),
        binding_fingerprint="" if binding is None else binding.fingerprint(),
    )


def check_dedupe_conflict(
    existing: ApprovalRequest,
    *,
    tool_name: str,
    arguments: "Mapping[str, Any]",
    arguments_hash: "str | None" = None,
) -> None:
    """Guard ``create_or_get_pending()``: when an existing request is found for
    ``(run_id, tool_call_id)``, it must be for the SAME call -- a dedupe key
    reused with a different ``tool_name`` / arguments is a conflict, not a
    replay. Identity is compared via ``arguments_hash`` (the real arguments'
    fingerprint), never the redacted audit copy. ``reason`` is excluded
    (informational only). Raises
    :class:`~linktools.ai.errors.ApprovalConflictError`."""
    from ..errors import ApprovalConflictError

    expected_hash = arguments_hash or compute_arguments_hash(tool_name, arguments)
    if existing.tool_name != tool_name or existing.arguments_hash != expected_hash:
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
        tenant_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        reason: "str | None",
        arguments: "Mapping[str, Any]",
        approval_id: str,
        binding: "Mapping[str, Any]",
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
