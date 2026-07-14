#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileIdempotencyStore: per-record JSON files at root/{scope}/{key}.json.

Mirrors FileApprovalStore / FileRunStore patterns: atomic writes (temp-file
+ os.replace) and path-traversal guards via _validate_id_segment. An
``asyncio.Lock`` serializes the reserve/complete/fail transitions within
one process so the read-check-mutate sequences are race-free.

Lookup is by (scope, key) -- the file path is keyed on those two segments,
so reserve/get/complete/fail are all O(1). ``record.id`` is a uuid4 minted
on reserve and stored inside the JSON for diagnostics/audit; it is not used
to address the file.

Each public async method delegates to a ``_*_sync`` private method via
``asyncio.to_thread`` so blocking file I/O never runs on the event loop.
The ``asyncio.Lock`` is held in the async wrapper and spans the
``to_thread`` call (not the other way around)."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...tool.idempotency import (
    ClaimDisposition,
    ClaimResult,
    IdempotencyClaim,
    IdempotencyRecord,
    IdempotencyStatus,
)
from .run import _atomic_write, _validate_id_segment


def _dt_iso(dt: "datetime | None") -> "str | None":
    return None if dt is None else dt.isoformat()


def _record_to_json(record: IdempotencyRecord) -> dict:
    return {
        "id": record.id,
        "scope": record.scope,
        "key": record.key,
        "request_hash": record.request_hash,
        "status": record.status.value,
        "result": record.result,
        "error": record.error,
        "created_at": record.created_at.isoformat(),
        "completed_at": _dt_iso(record.completed_at),
        "owner_id": record.owner_id,
        "generation": record.generation,
        "claimed_at": _dt_iso(record.claimed_at),
        "lease_expires_at": _dt_iso(record.lease_expires_at),
    }


def _record_from_json(raw: dict) -> IdempotencyRecord:
    return IdempotencyRecord(
        id=raw["id"],
        scope=raw["scope"],
        key=raw["key"],
        request_hash=raw["request_hash"],
        status=IdempotencyStatus(raw["status"]),
        result=raw.get("result"),
        error=raw.get("error"),
        created_at=datetime.fromisoformat(raw["created_at"]),
        completed_at=None
        if raw.get("completed_at") is None
        else datetime.fromisoformat(raw["completed_at"]),
        owner_id=raw.get("owner_id"),
        generation=raw.get("generation") or 0,
        claimed_at=None
        if raw.get("claimed_at") is None
        else datetime.fromisoformat(raw["claimed_at"]),
        lease_expires_at=None
        if raw.get("lease_expires_at") is None
        else datetime.fromisoformat(raw["lease_expires_at"]),
    )


class FileIdempotencyStore:
    """Single-process IdempotencyStore backed by per-(scope, key) JSON files.

    Records live at ``root/{scope}/{key}.json``. Writes are atomic
    (temp-file + ``os.replace``) and (scope, key) segments are validated to
    prevent path traversal. An ``asyncio.Lock`` serializes the
    read-check-mutate cycles in reserve/complete/fail so the invariants
    (unique (scope, key), hash-match, status transitions) hold within one
    process."""

    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # -- paths ---------------------------------------------------------

    def _path(self, scope: str, key: str) -> Path:
        scope_segment = _validate_id_segment(scope, kind="scope")
        key_segment = _validate_id_segment(key, kind="key")
        return self._root / scope_segment / f"{key_segment}.json"

    def _read(self, scope: str, key: str) -> "IdempotencyRecord | None":
        path = self._path(scope, key)
        if not path.exists():
            return None
        return _record_from_json(json.loads(path.read_text()))

    # -- claim / complete / fail (fenced) ---------------------------------

    def _claim_sync(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        owner_id: str,
        lease_seconds: float,
    ) -> ClaimResult:
        existing = self._read(scope, key)
        now = datetime.now(timezone.utc)
        lease_at = datetime.fromtimestamp(
            now.timestamp() + lease_seconds, tz=timezone.utc
        )
        if existing is None:
            return self._persist_fresh_claim(
                scope=scope,
                key=key,
                request_hash=request_hash,
                owner_id=owner_id,
                now=now,
                lease_at=lease_at,
                generation=1,
                existing_id=None,
            )
        # Same-key different request -> conflict (no record returned).
        if existing.request_hash != request_hash:
            return ClaimResult(disposition=ClaimDisposition.CONFLICT)
        if existing.status is IdempotencyStatus.COMPLETED:
            return ClaimResult(disposition=ClaimDisposition.REPLAY, record=existing)
        if existing.status is IdempotencyStatus.RESERVED:
            lease_valid = (
                existing.lease_expires_at is not None
                and existing.lease_expires_at > now
            )
            if lease_valid and existing.owner_id == owner_id:
                # Same owner re-driving (e.g. retry) -- keep its generation.
                return ClaimResult(
                    disposition=ClaimDisposition.ACQUIRED,
                    claim=_claim_from_record(existing),
                )
            if lease_valid:
                # Another live worker owns it.
                return ClaimResult(
                    disposition=ClaimDisposition.IN_PROGRESS, record=existing
                )
            # Lease expired -- steal: new generation, new owner.
            return self._persist_fresh_claim(
                scope=scope,
                key=key,
                request_hash=request_hash,
                owner_id=owner_id,
                now=now,
                lease_at=lease_at,
                generation=existing.generation + 1,
                existing_id=existing.id,
            )
        # FAILED -> retry: new generation, new owner.
        return self._persist_fresh_claim(
            scope=scope,
            key=key,
            request_hash=request_hash,
            owner_id=owner_id,
            now=now,
            lease_at=lease_at,
            generation=existing.generation + 1,
            existing_id=existing.id,
        )

    def _persist_fresh_claim(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        owner_id: str,
        now: datetime,
        lease_at: datetime,
        generation: int,
        existing_id: "str | None",
    ) -> ClaimResult:
        record = IdempotencyRecord(
            id=existing_id or str(uuid.uuid4()),
            scope=scope,
            key=key,
            request_hash=request_hash,
            status=IdempotencyStatus.RESERVED,
            result=None,
            error=None,
            created_at=now,
            completed_at=None,
            owner_id=owner_id,
            generation=generation,
            claimed_at=now,
            lease_expires_at=lease_at,
        )
        path = self._path(scope, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(_record_to_json(record)).encode("utf-8"))
        return ClaimResult(
            disposition=ClaimDisposition.ACQUIRED,
            claim=_claim_from_record(record),
        )

    async def claim(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        owner_id: str,
        lease_seconds: float = 300.0,
    ) -> ClaimResult:
        async with self._lock:
            return await asyncio.to_thread(
                self._claim_sync,
                scope=scope,
                key=key,
                request_hash=request_hash,
                owner_id=owner_id,
                lease_seconds=lease_seconds,
            )

    def _complete_sync(self, claim: IdempotencyClaim, result: Any) -> None:
        current = self._read(claim.scope, claim.key)
        if current is None or not _fence_matches(current, claim):
            # Stale owner/generation or missing record: reject (parity with the
            # SQL backend's rowcount check) -- never silently succeed.
            from ...errors import LostIdempotencyClaimError

            raise LostIdempotencyClaimError(
                f"complete lost the claim for ({claim.scope}, {claim.key}): "
                f"owner/generation no longer match"
            )
        now = datetime.now(timezone.utc)
        updated = IdempotencyRecord(
            id=current.id,
            scope=current.scope,
            key=current.key,
            request_hash=current.request_hash,
            status=IdempotencyStatus.COMPLETED,
            result=result,
            error=None,
            created_at=current.created_at,
            completed_at=now,
            owner_id=current.owner_id,
            generation=current.generation,
            claimed_at=current.claimed_at,
            lease_expires_at=current.lease_expires_at,
        )
        _atomic_write(
            self._path(claim.scope, claim.key),
            json.dumps(_record_to_json(updated)).encode("utf-8"),
        )

    async def complete(self, claim: IdempotencyClaim, result: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self._complete_sync, claim, result)

    def _fail_sync(self, claim: IdempotencyClaim, error: str) -> None:
        current = self._read(claim.scope, claim.key)
        if current is None or not _fence_matches(current, claim):
            from ...errors import LostIdempotencyClaimError

            raise LostIdempotencyClaimError(
                f"fail lost the claim for ({claim.scope}, {claim.key}): "
                f"owner/generation no longer match"
            )
        now = datetime.now(timezone.utc)
        updated = IdempotencyRecord(
            id=current.id,
            scope=current.scope,
            key=current.key,
            request_hash=current.request_hash,
            status=IdempotencyStatus.FAILED,
            result=None,
            error=error,
            created_at=current.created_at,
            completed_at=now,
            owner_id=current.owner_id,
            generation=current.generation,
            claimed_at=current.claimed_at,
            lease_expires_at=current.lease_expires_at,
        )
        _atomic_write(
            self._path(claim.scope, claim.key),
            json.dumps(_record_to_json(updated)).encode("utf-8"),
        )

    async def fail(self, claim: IdempotencyClaim, error: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._fail_sync, claim, error)

    async def get(self, scope: str, key: str) -> "IdempotencyRecord | None":
        async with self._lock:
            return await asyncio.to_thread(self._read, scope, key)


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


def _fence_matches(record: IdempotencyRecord, claim: IdempotencyClaim) -> bool:
    """A complete/fail is valid only if the record is still RESERVED and the
    owner_id + generation match -- a stale worker (older generation, or a
    lease that was stolen) cannot overwrite a newer owner's record."""
    return (
        record.status is IdempotencyStatus.RESERVED
        and record.owner_id == claim.owner_id
        and record.generation == claim.generation
    )
