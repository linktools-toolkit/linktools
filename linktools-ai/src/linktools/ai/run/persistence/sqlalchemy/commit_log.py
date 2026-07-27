#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run commit log: domain-owned SQL model + store for commit_id-keyed
idempotent replay of Run lifecycle commits (P5 SQL commit log).

Each Run commit (start / pause / resume / complete / fail / request_cancel /
acknowledge_cancel) records its (commit_id, operation, run_id, request_hash,
result_json) here in the SAME Storage UoW as the business writes + event
append. A retried call with the SAME commit_id + request_hash returns the
recorded result; the SAME commit_id with a DIFFERENT request_hash is a
RunCommitConflictError (a real conflict, not a silent overwrite).

The log lives in the run domain (not the storage kernel) because commit
semantics are a run-domain concept: the storage kernel provides the
transactional UoW primitive; this module owns the per-commit replay table
+ the canonical request-hash computation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping

from sqlalchemy import BINARY, Index, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from linktools.ai.storage.sqlalchemy.conventions import (
    BIGSERIAL,
    TABLE_PREFIX,
    TimestampMixin,
    sha256_hash,
    timestamp_indexes,
)
from linktools.ai.storage.sqlalchemy.models import Base

from .._replay import canonical_request_hash as _canonical_request_hash
from ....errors import LinktoolsAIError


def _json_default(value: Any) -> Any:
    """JSON encoder for the few non-JSON-native types a commit result carries.
    Delegates to the shared replay-hashing default so a result serialized here
    hashes identically to the request payload hashed by canonical_request_hash
    -- both sides must agree on every type's wire form."""
    from .._replay import _json_default as _shared

    return _shared(value)


class RunCommitConflictError(LinktoolsAIError):
    """A retried Run commit used the SAME commit_id with a DIFFERENT
    request_hash. Same-id-same-hash is an idempotent replay (returns the
    recorded result); same-id-different-hash is a conflict (the caller is
    asserting two distinct operations under one id, which the log refuses
    to silently collapse)."""


class RunCommitLogRow(TimestampMixin, Base):
    """One row per committed Run lifecycle point. ``commit_id`` is the caller-
    supplied deterministic id (e.g. ``pause:{run_id}:{approval_id}``);
    uniqueness is carried by ``commit_hash`` (sha256(commit_id)) so the wide
    commit_id column stays out of the unique index; ``request_hash`` lets a
    replay detect a same-id-different-payload conflict."""

    __tablename__ = f"{TABLE_PREFIX}run_commit_log"
    __table_args__ = (
        UniqueConstraint("commit_hash", name="uk_commit_hash"),
        # A MySQL deployment prefix-lengths `commit_id` in this index (see
        # migrations/init_schema.sql) -- kept out of the vendor-neutral core.
        Index("ix_commit_id", "commit_id"),
        Index("ix_run_id", "run_id"),
        *timestamp_indexes(),
    )

    id: Mapped[int] = mapped_column(BIGSERIAL, primary_key=True, autoincrement=True)
    commit_id: Mapped[str] = mapped_column(String(200))
    commit_hash: Mapped[bytes] = mapped_column(BINARY(32))
    operation: Mapped[str] = mapped_column(String(64))
    run_id: Mapped[str] = mapped_column(String(128))
    # SHA-256 of the canonical-serialized command. 32 raw bytes.
    request_hash: Mapped[bytes] = mapped_column(BINARY(32))
    result_json: Mapped[str] = mapped_column(Text)
    result_payload: Mapped["bytes | None"] = mapped_column(LargeBinary, nullable=True)


@dataclass(frozen=True, slots=True)
class RunCommitLogRecord:
    commit_id: str
    operation: str
    run_id: str
    request_hash: bytes
    result: Mapping[str, Any]
    result_payload: bytes | None
    created_at: datetime


def _row_to_record(row: RunCommitLogRow) -> RunCommitLogRecord:
    return RunCommitLogRecord(
        commit_id=row.commit_id,
        operation=row.operation,
        run_id=row.run_id,
        request_hash=bytes(row.request_hash),
        result=json.loads(row.result_json),
        result_payload=bytes(row.result_payload) if row.result_payload is not None else None,
        created_at=row.created_at,
    )


class SqlAlchemyRunCommitLog:
    """Read + append the commit log within a caller-owned AsyncSession (the
    surrounding Storage UoW). The log is ALWAYS written in the SAME
    transaction as the business writes + event append, so a crash mid-commit
    leaves either the whole record (business + log) or nothing -- never a
    business write without its commit-log entry (which would defeat replay).

    The store is stateless: every method takes the AsyncSession it should
    run in, so the SAME log instance serves any number of concurrent UoWs
    without holding a session-factory reference."""

    def __init__(self) -> None:
        pass

    async def find(
        self, session: AsyncSession, commit_id: str
    ) -> "RunCommitLogRecord | None":
        from sqlalchemy import select

        row = (
            await session.execute(
                select(RunCommitLogRow).where(
                RunCommitLogRow.commit_hash == sha256_hash(commit_id)
            )
            )
        ).scalar_one_or_none()
        return _row_to_record(row) if row is not None else None

    async def record(
        self,
        session: AsyncSession,
        *,
        commit_id: str,
        operation: str,
        run_id: str,
        request_hash: bytes,
        result: Mapping[str, Any],
        result_payload: bytes | None = None,
    ) -> RunCommitLogRecord:
        session.add(
            RunCommitLogRow(
                commit_id=commit_id,
                commit_hash=sha256_hash(commit_id),
                operation=operation,
                run_id=run_id,
                request_hash=request_hash,
                result_json=json.dumps(result, sort_keys=True, default=_json_default),
                result_payload=result_payload,
            )
        )
        await session.flush()
        return RunCommitLogRecord(
            commit_id=commit_id,
            operation=operation,
            run_id=run_id,
            request_hash=request_hash,
            result=result,
            result_payload=result_payload,
            created_at=datetime.now(timezone.utc),
        )


def canonical_request_hash(operation: str, payload: Mapping[str, Any]) -> bytes:
    """Re-export of the shared canonical_request_hash. Kept here so existing
    callers importing from this module keep working; the actual implementation
    lives in ``run.persistence._replay`` (shared with the Filesystem
    coordinator, which must hash identically for the same payload)."""
    return _canonical_request_hash(operation, payload)


__all__: "list[str]" = [
    "RunCommitConflictError",
    "RunCommitLogRecord",
    "RunCommitLogRow",
    "SqlAlchemyRunCommitLog",
    "canonical_request_hash",
]
