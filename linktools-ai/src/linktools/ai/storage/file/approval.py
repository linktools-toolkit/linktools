#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileApprovalStore: single-process file backend for ApprovalStore (the
Protocol in agent_runtime/approval.py). One JSON file per request at
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
different request."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

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
        "arguments": dict(request.arguments),
        "status": request.status.value,
        "version": request.version,
        "created_at": request.created_at.isoformat(),
        "resolved_at": None if request.resolved_at is None else request.resolved_at.isoformat(),
        "resolved_by": request.resolved_by,
        "metadata": dict(request.metadata),
    }


def _request_from_json(raw: dict) -> ApprovalRequest:
    return ApprovalRequest(
        id=raw["id"],
        run_id=raw["run_id"],
        tool_call_id=raw["tool_call_id"],
        tool_name=raw["tool_name"],
        reason=raw["reason"],
        arguments=raw["arguments"],
        status=ApprovalStatus(raw["status"]),
        version=raw["version"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        resolved_at=None if raw["resolved_at"] is None else datetime.fromisoformat(raw["resolved_at"]),
        resolved_by=raw["resolved_by"],
        metadata=raw["metadata"],
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
        return self._requests_dir / f"{_validate_id_segment(approval_id, kind='approval_id')}.json"

    # -- read ----------------------------------------------------------

    async def get(self, approval_id: str) -> "ApprovalRequest | None":
        path = self._path(approval_id)
        if not path.exists():
            return None
        return _request_from_json(json.loads(path.read_text()))

    async def list_pending(self, run_id: str) -> "tuple[ApprovalRequest, ...]":
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

    # -- write ---------------------------------------------------------

    async def create(self, request: ApprovalRequest) -> ApprovalRequest:
        # _path validates the id segment, guarding against path traversal.
        path = self._path(request.id)
        if path.exists():
            raise ApprovalConflictError(f"approval already exists: {request.id}")
        _atomic_write(path, json.dumps(_request_to_json(request)).encode("utf-8"))
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
        rejection_reason: object,
    ) -> ApprovalRequest:
        async with self._lock:
            current = await self.get(approval_id)
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
            resolved = ApprovalRequest(
                id=current.id,
                run_id=current.run_id,
                tool_call_id=current.tool_call_id,
                tool_name=current.tool_name,
                reason=current.reason,
                arguments=current.arguments,
                status=target,
                version=current.version + 1,
                created_at=current.created_at,
                resolved_at=now,
                resolved_by=resolved_by,
                metadata=new_metadata,
            )
            _atomic_write(
                self._path(approval_id),
                json.dumps(_request_to_json(resolved)).encode("utf-8"),
            )
            return resolved
