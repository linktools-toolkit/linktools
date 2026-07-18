#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileApprovalStore: single-process file backend for ApprovalStore (the
Protocol in agent/approval.py). One JSON file per request at
root/requests/{approval_id}.json. Mirrors FileMemoryStore/FileRunStore's
atomic-write + path-traversal-guard patterns (see storage/file/run.py). An
``asyncio.Lock`` serializes ``approve``/``reject`` so the optimistic-version
and transition invariants hold within one process.

Rejection reason: ``ApprovalRequest`` has no dedicated field for the rejection
reason, so ``reject(..., reason=...)`` stores it under
``metadata["rejection_reason"]`` (a None reason is still recorded as that key
mapped to None, so callers can distinguish "rejected, no reason given" from
"approved"). Any pre-existing ``metadata`` is preserved; ``approve`` never
touches the metadata so it cannot shadow a prior rejection reason on a
different request.

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop.
The ``asyncio.Lock`` is held in the async wrapper and spans the
``to_thread`` call (not the other way around)."""

import asyncio
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

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
from .run import _atomic_write, _validate_id_segment

#: Key under which ``reject(reason=...)`` is recorded in the request's metadata.
REJECTION_REASON_METADATA_KEY = "rejection_reason"


class _Unset:
    """Sentinel distinguishing "approve" (don't touch metadata) from
    "reject" (always record the key, even when reason is None)."""

    __slots__ = ()


_UNSET = _Unset()


def _request_to_json(request: ApprovalRequest) -> dict:
    return {
        "id": request.id,
        "run_id": request.run_id,
        "tool_call_id": request.tool_call_id,
        "tool_name": request.tool_name,
        "reason": request.reason,
        "redacted_arguments": dict(request.redacted_arguments),
        "arguments_hash": request.arguments_hash,
        "status": request.status.value,
        "version": request.version,
        "created_at": request.created_at.isoformat(),
        "resolved_at": None
        if request.resolved_at is None
        else request.resolved_at.isoformat(),
        "resolved_by": request.resolved_by,
        "metadata": dict(request.metadata),
        "tenant_id": request.tenant_id,
        "descriptor_fingerprint": request.descriptor_fingerprint,
        "handler_revision": request.handler_revision,
        "provider_revision": request.provider_revision,
        "policy_revision": request.policy_revision,
        "capability_revision": request.capability_revision,
        "result_processor_revision": request.result_processor_revision,
        "binding": dict(request.binding),
        "binding_fingerprint": request.binding_fingerprint,
        "schema_version": request.schema_version,
    }


def _request_from_json(raw: dict) -> ApprovalRequest:
    from ...agent.approval import compute_arguments_hash
    from ...security.redact import redact_for_audit

    # Reader tolerates the pre-redaction format: an older record persisted the
    # RAW ``arguments`` (no redacted copy, no hash). On read, treat the legacy
    # payload as the redacted audit copy (run it through the redactor so a
    # secret under an obvious key is not re-surfaced in memory) and synthesize
    # the identity hash from it so dedupe still works.
    if "redacted_arguments" in raw:
        redacted = raw["redacted_arguments"]
    else:
        redacted = redact_for_audit(raw.get("arguments", {}))
    arguments_hash = raw.get("arguments_hash")
    if not arguments_hash:
        arguments_hash = compute_arguments_hash(
            raw["tool_name"], raw.get("arguments", redacted)
        )
    return ApprovalRequest(
        id=raw["id"],
        run_id=raw["run_id"],
        tool_call_id=raw["tool_call_id"],
        tool_name=raw["tool_name"],
        reason=raw["reason"],
        redacted_arguments=redacted,
        arguments_hash=arguments_hash,
        status=ApprovalStatus(raw["status"]),
        version=raw["version"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        resolved_at=None
        if raw["resolved_at"] is None
        else datetime.fromisoformat(raw["resolved_at"]),
        resolved_by=raw["resolved_by"],
        metadata=raw["metadata"],
        tenant_id=raw.get("tenant_id"),
        descriptor_fingerprint=raw.get("descriptor_fingerprint"),
        handler_revision=raw.get("handler_revision"),
        provider_revision=raw.get("provider_revision"),
        policy_revision=raw.get("policy_revision"),
        capability_revision=raw.get("capability_revision"),
        result_processor_revision=raw.get("result_processor_revision"),
        binding=raw.get("binding", {}),
        binding_fingerprint=raw.get("binding_fingerprint", ""),
        schema_version=int(raw.get("schema_version", 0)),
    )


class FileApprovalStore:
    """Single-process ApprovalStore backed by per-request JSON files.

    Requests live at ``root/requests/{approval_id}.json``. Writes are atomic
    (temp-file + ``os.replace``) and ids are validated to prevent path
    traversal. An ``asyncio.Lock`` serializes ``approve``/``reject`` so that
    optimistic-concurrency + transition invariants hold within one process.
    """

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._requests_dir = self._root / "requests"
        self._requests_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # -- paths ---------------------------------------------------------

    def _path(self, approval_id: str) -> Path:
        return (
            self._requests_dir
            / f"{_validate_id_segment(approval_id, kind='approval_id')}.json"
        )

    # -- read ----------------------------------------------------------

    def _get_sync(self, approval_id: str) -> "ApprovalRequest | None":
        path = self._path(approval_id)
        if not path.exists():
            return None
        return _request_from_json(json.loads(path.read_text()))

    async def get(self, approval_id: str) -> "ApprovalRequest | None":
        return await asyncio.to_thread(self._get_sync, approval_id)

    def _list_pending_sync(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        out: list = []
        for path in self._requests_dir.glob("*.json"):
            raw = json.loads(path.read_text())
            if raw.get("status") != ApprovalStatus.PENDING.value:
                continue
            if raw.get("run_id") != run_id:
                continue
            out.append(_request_from_json(raw))
        out.sort(key=lambda r: r.created_at)
        return tuple(out)

    async def list_pending(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        return await asyncio.to_thread(self._list_pending_sync, run_id)

    def _list_for_run_sync(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        # Status-agnostic counterpart to ``list_pending``: returns EVERY
        # request for the run regardless of status, ordered by created_at.
        # The resume gate (ToolExecutor._already_approved) consults this to
        # recognize a call that was approved externally without re-persisting
        # a PENDING duplicate.
        out: list = []
        for path in self._requests_dir.glob("*.json"):
            raw = json.loads(path.read_text())
            if raw.get("run_id") != run_id:
                continue
            out.append(_request_from_json(raw))
        out.sort(key=lambda r: r.created_at)
        return tuple(out)

    async def list_for_run(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
        return await asyncio.to_thread(self._list_for_run_sync, run_id)

    # -- write ---------------------------------------------------------

    def _create_sync(self, request: ApprovalRequest) -> ApprovalRequest:
        # _path validates the id segment, guarding against path traversal.
        path = self._path(request.id)
        if path.exists():
            raise ApprovalConflictError(f"approval already exists: {request.id}")
        _atomic_write(path, json.dumps(_request_to_json(request)).encode("utf-8"))
        return request

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        return await asyncio.to_thread(self._create_sync, request)

    def _create_or_get_pending_sync(
        self,
        *,
        tenant_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        reason: "str | None",
        arguments: dict,
        approval_id: str,
        binding: dict | None = None,
    ) -> ApprovalRequest:
        required_binding = ("descriptor_fingerprint", "handler_revision",
            "provider_revision", "policy_revision", "capability_revision",
            "result_processor_revision", "arguments_hash")
        if not binding or any(not isinstance(binding.get(k), str) or not binding[k]
                              for k in required_binding):
            raise ApprovalConflictError("approval execution binding is incomplete")
        # dedup on (run_id, tool_call_id) BEFORE
        # creating -- a retry, a duplicate model drive, or a re-entrant pause
        # for the same tool_call must reuse the existing request rather than
        # creating a second PENDING one. Held under self._lock (see
        # create_or_get_pending) so two concurrent callers can't both observe
        # "no existing" and both create.
        existing = [
            r for r in self._list_for_run_sync(run_id) if r.tool_call_id == tool_call_id
        ]
        if existing:
            # same dedupe key reused with a different
            # tool_name/arguments is a conflict, not a replay.
            check_dedupe_conflict(
                existing[-1], tool_name=tool_name, arguments=arguments,
                arguments_hash=binding.get("arguments_hash") if binding else None,
            )
            return existing[-1]
        request = build_approval_request(
            tenant_id=tenant_id,
            run_id=run_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reason=reason,
            arguments=arguments,
            approval_id=approval_id,
            **(binding or {}),
        )
        return self._create_sync(request)

    async def create_or_get_pending(
        self,
        *,
        tenant_id: str,
        run_id: str,
        tool_call_id: str,
        tool_name: str,
        reason: "str | None",
        arguments: dict,
        approval_id: str,
        binding: dict | None = None,
    ) -> ApprovalRequest:
        async with self._lock:
            return await asyncio.to_thread(
                self._create_or_get_pending_sync,
                tenant_id=tenant_id,
                run_id=run_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                reason=reason,
                arguments=arguments,
                approval_id=approval_id,
                binding=binding,
            )

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

    def _resolve_sync(
        self,
        approval_id: str,
        *,
        target: ApprovalStatus,
        expected_version: int,
        resolved_by: str,
        rejection_reason: object,
    ) -> ApprovalRequest:
        current = self._get_sync(approval_id)
        if current is None:
            raise ApprovalNotFoundError(f"approval not found: {approval_id}")
        if current.version != expected_version:
            raise ApprovalConflictError(
                f"expected version {expected_version}, found {current.version}"
            )
        if target not in ALLOWED_APPROVAL_TRANSITIONS.get(current.status, frozenset()):
            raise InvalidApprovalTransitionError(
                f"cannot transition {current.status} -> {target}"
            )
        now = datetime.now(current.created_at.tzinfo or timezone.utc)
        new_metadata = dict(current.metadata)
        # ``reject`` always sets the key (even to None); ``approve`` leaves
        # metadata untouched so approvals can't shadow a prior rejection
        # reason on a different request.
        if rejection_reason is not _UNSET:
            new_metadata[REJECTION_REASON_METADATA_KEY] = rejection_reason
        resolved = dataclasses.replace(current, status=target,
            version=current.version + 1, resolved_at=now,
            resolved_by=resolved_by, metadata=new_metadata)
        _atomic_write(
            self._path(approval_id),
            json.dumps(_request_to_json(resolved)).encode("utf-8"),
        )
        return resolved

    async def _resolve(
        self,
        approval_id: str,
        *,
        target: ApprovalStatus,
        expected_version: int,
        resolved_by: str,
        rejection_reason: object,
    ) -> ApprovalRequest:
        # The asyncio.Lock is held in the async wrapper and spans the
        # ``to_thread`` call -- serializing the read-check-mutate cycle
        # within one process while letting the event loop run during the
        # blocking I/O.
        async with self._lock:
            return await asyncio.to_thread(
                self._resolve_sync,
                approval_id,
                target=target,
                expected_version=expected_version,
                resolved_by=resolved_by,
                rejection_reason=rejection_reason,
            )
