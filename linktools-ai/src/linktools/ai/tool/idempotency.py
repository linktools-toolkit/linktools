#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IdempotencyStore: persistent tool-call idempotency.

ToolExecutor consults an IdempotencyStore when an ``idempotency_key`` is
supplied to ``execute()``. The store persists reservations so tool idempotency
survives process restart -- an in-process dict would not (and is forbidden).

Lifecycle:

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
key on both backends (file path; SQL unique constraint). An alternative
``complete(record_id, ...)`` signature leaves ``record_id`` without a
source; ``(scope, key)`` is the consistent, race-free
interpretation. ``record.id`` remains on the dataclass for diagnostics,
audit logs, and any future indexer that needs a stable handle.

The Protocol is ``@runtime_checkable`` so wiring/tests can ``isinstance``
against it. ``compute_request_hash`` centralizes the hash formula so
both backends and the executor agree byte-for-byte."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable
from uuid import UUID

from ..errors import IdempotencyConfigurationError
from .policy import IdempotencyStrategy

if TYPE_CHECKING:
    from ..run.context import RunContext
    from ..tool.models import ToolDescriptor


class IdempotencyStatus(str, Enum):
    """Three-state lifecycle of an idempotent tool call. String enum
    so the value round-trips through JSON / SQL columns as a plain string."""

    RESERVED = "reserved"
    COMPLETED = "completed"
    FAILED = "failed"


class ClaimDisposition(str, Enum):
    """The outcome of a fenced ``claim()``. The caller branches on this:
    ACQUIRED -> run the handler then complete/fail with the claim token;
    REPLAY -> return the cached result; IN_PROGRESS -> raise; CONFLICT -> raise."""

    ACQUIRED = "acquired"
    REPLAY = "replay"
    IN_PROGRESS = "in_progress"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """One persisted idempotency record. Fencing fields: ``owner_id`` is the
    worker that currently owns a RESERVED record; ``generation`` increments on
    every (re)claim so a stale worker's complete/fail is rejected;
    ``lease_expires_at`` is when a RESERVED claim may be stolen (worker died).
    ``completed_at`` is set on either terminal transition."""

    id: str
    scope: str
    key: str
    request_hash: str
    status: IdempotencyStatus
    result: "Any | None"
    error: "str | None"
    created_at: datetime
    completed_at: "datetime | None"
    owner_id: "str | None" = None
    generation: int = 0
    claimed_at: "datetime | None" = None
    lease_expires_at: "datetime | None" = None


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """The fenced result of ``claim()``: a disposition + either the claim
    token (ACQUIRED) or the existing record (REPLAY/IN_PROGRESS). CONFLICT
    carries no record."""

    disposition: ClaimDisposition
    claim: "IdempotencyClaim | None" = None
    record: "IdempotencyRecord | None" = None


@runtime_checkable
class IdempotencyStore(Protocol):
    """Persistent idempotency storage. Two backends: FileIdempotencyStore
    (one JSON per (scope, key)) and SqlAlchemyIdempotencyStore (table
    ``ai_idempotency``). Both implement the same fenced-claim contract."""

    async def claim(
        self,
        *,
        scope: str,
        key: str,
        request_hash: str,
        owner_id: str,
        lease_seconds: float = 300.0,
    ) -> ClaimResult:
        """Fenced claim on ``(scope, key)``. The state machine:
        - no record -> CLAIMED generation=1, owner=owner_id -> ACQUIRED
        - COMPLETED + same hash -> REPLAY (return cached result)
        - RESERVED + same hash + lease not expired + same owner -> ACQUIRED (re-drive)
        - RESERVED + same hash + lease not expired + other owner -> IN_PROGRESS
        - RESERVED + lease expired OR FAILED -> re-claim generation+1, new owner -> ACQUIRED
        - different request_hash -> CONFLICT
        """
        ...

    async def complete(self, claim: "IdempotencyClaim", result: Any) -> None:
        """Mark the record COMPLETED with ``result``. The claim's owner_id +
        generation must match the persisted record or the call is rejected (a
        stale worker cannot overwrite a newer owner's record)."""
        ...

    async def fail(self, claim: "IdempotencyClaim", error: str) -> None:
        """Mark the record FAILED with ``error``. Same owner/generation
        fencing as ``complete``."""
        ...

    async def get(self, scope: str, key: str) -> "IdempotencyRecord | None":
        """Look up by ``(scope, key)``. None if no record exists."""
        ...


def compute_request_hash(
    tool_name: str,
    arguments: "dict[str, Any]",
    scope: str,
    *,
    schema_version: str = "1",
) -> str:
    """SHA-256 of ``tool_name | schema_version | canonical_args | scope``.
    ``arguments`` are encoded with :func:`canonical_json` so two dicts that
    compare equal hash identically regardless of insertion order, and a
    non-JSON-native value (Path, datetime, bytes, ...) raises instead of being
    silently stringified into an unstable/colliding hash.

    ``schema_version`` defaults to ``"1"`` so existing callers that don't pass
    it (or a ToolSpec whose ``schema_version`` was never bumped) hash
    identically to before this parameter was added. When a tool's input
    contract changes, bumping ``ToolSpec.schema_version`` changes the hash for
    every subsequent call, so a stale idempotency record from before the
    schema change is never mistaken for a match against the new shape."""
    from ..json import canonical_json

    payload = f"{tool_name}|{schema_version}|{canonical_json(arguments)}|{scope}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    from ..json import canonical_json

    return canonical_json(value)


def encode_business_key(value: Any) -> str:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    from ..errors import IdempotencyConfigurationError

    raise IdempotencyConfigurationError(
        "business key must be a string, integer, or UUID"
    )


@runtime_checkable
class IdempotencyKeyBuilder(Protocol):
    def build(
        self,
        *,
        descriptor: "ToolDescriptor",
        arguments: "Mapping[str, Any]",
        run_context: "RunContext | None",
        schema_version: str,
        policy: Any = None,
    ) -> "str | None": ...


class DefaultIdempotencyKeyBuilder:
    """sha256(run_id + tool_name + canonical_json(arguments) + schema_version).
    Returns None when there is no run_id (the call is not part of a persisted
    run, so idempotent replay is meaningless)."""

    def build(
        self,
        *,
        descriptor: "ToolDescriptor",
        arguments: "Mapping[str, Any]",
        run_context: "RunContext | None",
        schema_version: str,
        policy: Any = None,
    ) -> "str | None":
        run_id = getattr(run_context, "run_id", None) if run_context else None
        strategy = getattr(
            policy, "idempotency_strategy", IdempotencyStrategy.EXACT_CALL
        )
        if not run_id:
            raise IdempotencyConfigurationError("idempotent tool calls require run_id")
        if strategy == IdempotencyStrategy.BUSINESS_KEY:
            field = getattr(policy, "idempotency_key_field", None)
            if not field or field not in arguments:
                raise IdempotencyConfigurationError(
                    "business-key idempotency requires a configured key field"
                )
            tenant = getattr(run_context, "tenant_id", None) or ""
            workspace = getattr(run_context, "workspace", None) or ""
            identity = [
                tenant,
                workspace,
                descriptor.name,
                encode_business_key(arguments[field]),
                schema_version,
            ]
        else:
            identity = [
                run_id,
                descriptor.name,
                _canonical_json(dict(arguments)),
                schema_version,
            ]
        payload = "|".join(identity).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class IdempotencyClaim:
    """A fenced claim on an idempotency record. The owner_id + generation pair
    is the fencing token: complete/fail must match or be rejected."""

    scope: str
    key: str
    request_hash: str
    owner_id: str
    generation: int
    claimed_at: datetime
    lease_expires_at: datetime
