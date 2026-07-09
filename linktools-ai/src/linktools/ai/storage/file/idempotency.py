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

Per review doc §16 (Phase 4B): each public async method delegates to a
``_*_sync`` private method via ``asyncio.to_thread`` so blocking file I/O
never runs on the event loop. The ``asyncio.Lock`` is held in the async
wrapper and spans the ``to_thread`` call (not the other way around)."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...errors import IdempotencyConflictError
from ...tool.idempotency import IdempotencyRecord, IdempotencyStatus
from .run import _atomic_write, _validate_id_segment


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
        "completed_at": None if record.completed_at is None else record.completed_at.isoformat(),
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
        completed_at=None if raw.get("completed_at") is None else datetime.fromisoformat(raw["completed_at"]),
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

    # -- write ---------------------------------------------------------

    def _reserve_sync(
        self,
        scope: str,
        key: str,
        request_hash: str,
        *,
        expires_at: "datetime | None",
    ) -> "IdempotencyRecord | None":
        existing = self._read(scope, key)
        if existing is not None:
            if existing.request_hash != request_hash:
                raise IdempotencyConflictError(
                    f"idempotency key {key!r} reused with a different request"
                )
            return existing
        now = datetime.now(timezone.utc)
        record = IdempotencyRecord(
            id=str(uuid.uuid4()),
            scope=scope,
            key=key,
            request_hash=request_hash,
            status=IdempotencyStatus.RESERVED,
            result=None,
            error=None,
            created_at=now,
            completed_at=None,
        )
        # mkdir for the scope subdir is done here (not in __init__) so
        # FileIdempotencyStore(root=...) never creates empty scope dirs
        # for scopes that never reserve. Idempotent.
        path = self._path(scope, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(_record_to_json(record)).encode("utf-8"))
        return None

    async def reserve(
        self,
        scope: str,
        key: str,
        request_hash: str,
        *,
        expires_at: "datetime | None" = None,
    ) -> "IdempotencyRecord | None":
        async with self._lock:
            return await asyncio.to_thread(
                self._reserve_sync, scope, key, request_hash, expires_at=expires_at,
            )

    def _complete_sync(self, scope: str, key: str, result: Any) -> None:
        current = self._read(scope, key)
        if current is None:
            return
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
        )
        _atomic_write(self._path(scope, key), json.dumps(_record_to_json(updated)).encode("utf-8"))

    async def complete(self, scope: str, key: str, result: Any) -> None:
        async with self._lock:
            await asyncio.to_thread(self._complete_sync, scope, key, result)

    def _fail_sync(self, scope: str, key: str, error: str) -> None:
        current = self._read(scope, key)
        if current is None:
            return
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
        )
        _atomic_write(self._path(scope, key), json.dumps(_record_to_json(updated)).encode("utf-8"))

    async def fail(self, scope: str, key: str, error: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._fail_sync, scope, key, error)

    async def get(self, scope: str, key: str) -> "IdempotencyRecord | None":
        # Read-only: no lock needed. _path validates the segments, so a
        # traversal attempt raises ValueError rather than escaping root.
        return await asyncio.to_thread(self._read, scope, key)
