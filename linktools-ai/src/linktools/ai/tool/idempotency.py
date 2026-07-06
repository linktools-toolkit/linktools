#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IdempotencyStore: persistent tool-call idempotency (spec §11).

ToolExecutor consults an IdempotencyStore when an ``idempotency_key`` is
supplied to ``execute()``. The store persists reservations so tool
idempotency survives process restart -- the legacy in-process dict is gone
(review doc §11.1: "禁止仅使用进程内字典").

Lifecycle (per §11.2):

- ``reserve(scope, key, request_hash)`` is the entry point. It either:
  * creates a fresh RESERVED record and returns ``None`` -- the caller must
    then run the handler and call ``complete``/``fail``; OR
  * finds an existing record for the same ``(scope, key)`` and returns it so
    the caller can branch on ``status``:
      - COMPLETED + same hash -> return the cached result
      - RESERVED  + same hash -> raise IdempotencyInProgressError
      - FAILED    + same hash -> caller may retry (re-invoke the handler)
  * Raises IdempotencyConflictError if the existing record's hash differs
    (same key reused with different tool/args/scope).

API note: ``complete``/``fail`` key off ``(scope, key)`` rather than a
synthetic record id. The caller always knows ``(scope, key)`` (it just
passed them to ``reserve``), and ``(scope, key)`` is the natural primary
key on both backends (file path; SQL unique constraint). The review doc
sketches ``complete(record_id, ...)`` but leaves ``record_id`` undefined in
the wiring pseudocode; ``(scope, key)`` is the consistent, race-free
interpretation. ``record.id`` remains on the dataclass for diagnostics,
audit logs, and any future indexer that needs a stable handle.

The Protocol is ``@runtime_checkable`` so wiring/tests can ``isinstance``
against it. ``compute_request_hash`` centralizes the §11.3 hash formula so
both backends and the executor agree byte-for-byte."""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class IdempotencyStatus(str, Enum):
    """Three-state lifecycle of an idempotent tool call (§11.2). String enum
    so the value round-trips through JSON / SQL columns as a plain string."""

    RESERVED = "reserved"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """One persisted idempotency reservation. ``result`` is the cached tool
    result when COMPLETED (None otherwise); ``error`` is the serialized error
    message when FAILED (None otherwise). ``completed_at`` is set on either
    terminal transition (COMPLETED or FAILED) and stays None while RESERVED."""

    id: str
    scope: str
    key: str
    request_hash: str
    status: IdempotencyStatus
    result: "Any | None"
    error: "str | None"
    created_at: datetime
    completed_at: "datetime | None"


@runtime_checkable
class IdempotencyStore(Protocol):
    """Persistent idempotency storage. Two backends: FileIdempotencyStore
    (one JSON per (scope, key)) and SqlAlchemyIdempotencyStore (table
    ``ai_idempotency``). Both implement the same contract."""

    async def reserve(
        self,
        scope: str,
        key: str,
        request_hash: str,
        *,
        expires_at: "datetime | None" = None,
    ) -> "IdempotencyRecord | None":
        """Try to reserve ``(scope, key)``. Returns ``None`` if this is a
        fresh reservation (caller proceeds + ``complete``/``fail``), or the
        existing record if one already exists (caller branches on
        ``status``). Raises ``IdempotencyConflictError`` if the existing
        record carries a different ``request_hash``.

        ``expires_at`` is recorded for future TTL support; backends do not
        yet expire records automatically in this phase."""
        ...

    async def complete(self, scope: str, key: str, result: Any) -> None:
        """Mark the record at ``(scope, key)`` as COMPLETED with ``result``.
        No-op if no such record exists (defensive against races where the
        record was evicted between ``reserve`` and ``complete``)."""
        ...

    async def fail(self, scope: str, key: str, error: str) -> None:
        """Mark the record at ``(scope, key)`` as FAILED with ``error``.
        Same no-op-on-missing semantics as ``complete``."""
        ...

    async def get(self, scope: str, key: str) -> "IdempotencyRecord | None":
        """Look up by ``(scope, key)``. None if no record exists."""
        ...


def compute_request_hash(
    tool_name: str, arguments: "dict[str, Any]", scope: str
) -> str:
    """SHA-256 of ``tool_name | normalized_args | scope`` (§11.3).
    ``arguments`` are json-serialized with ``sort_keys=True`` so two dicts
    that compare equal hash identically regardless of insertion order;
    ``default=str`` keeps non-JSON-native values (Path, datetime, ...) stable
    instead of raising. Schema-version inclusion is deferred until the
    registry exposes a tool-schema version."""
    payload = f"{tool_name}|{json.dumps(arguments, sort_keys=True, default=str)}|{scope}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
