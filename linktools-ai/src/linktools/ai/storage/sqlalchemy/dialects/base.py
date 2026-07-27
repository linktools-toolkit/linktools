#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-first SQLAlchemy object dialect.

The dialect is the ONLY seam at which a downstream that brings its own engine
+ driver plugs into the storage kernel's object backend. Core does NOT branch
on a dialect name and does NOT resolve a strategy from a session_factory;
the backend is HANDED a ``SqlAlchemyObjectDialect`` at construction and uses
it for the two dialect-specific pieces of the conflict-detecting insert path:

- ``insert_current_if_absent`` -- INSERT a fresh ``storage_object_*`` row if
  no row with the same ``key_hash`` exists; return whether the insert landed.
- ``classify_integrity_error`` -- map a caught SQLAlchemy IntegrityError to
  one of the kernel-level integrity kinds so the caller can re-raise the
  original error for non-unique-constraint violations.

Core ships exactly one concrete implementation -- :class:`SqliteObjectDialect`
-- which downstreams use directly for SQLite/aiosqlite or as the reference for
their own per-vendor dialect. Core never names a real production database
product (no MySQL, no PostgreSQL); a downstream that wants one of those
vendors ships its own dialect implementation and runs the kernel's
conformance suite in its own CI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class InsertResult:
    """Outcome of an ``insert_current_if_absent`` attempt. ``inserted`` is
    True when a fresh row was committed; False when a same-``key_hash`` row
    already existed (so the caller reconciles against the existing row)."""

    inserted: bool


class IntegrityViolationKind(str, Enum):
    """The vendor-agnostic taxonomy for a caught IntegrityError. The kernel's
    object backend only cares whether a write lost a unique-key race; the
    finer-grained FOREIGN_KEY / CHECK / UNKNOWN kinds exist so a downstream
    dialect can surface structural integrity errors distinctly."""

    UNIQUE_CONFLICT = "unique_conflict"
    FOREIGN_KEY = "foreign_key"
    CHECK = "check"
    UNKNOWN = "unknown"


@runtime_checkable
class SqlAlchemyObjectDialect(Protocol):
    """Per-vendor dialect contract for the storage kernel's object backend.

    The dialect MUST be injected into the backend explicitly; the kernel
    never resolves one from a session_factory. ``name`` is informational (a
    downstream tags its own dialect; core does not branch on it)."""

    @property
    def name(self) -> str:
        ...

    async def insert_current_if_absent(
        self,
        session: "AsyncSession",
        *,
        values: "Mapping[str, Any]",
    ) -> InsertResult:
        """INSERT a fresh storage_object current-row with ``values`` iff no
        row with the same ``key_hash`` exists; return ``InsertResult(inserted=
        True)`` when the row landed, ``InsertResult(inserted=False)`` when a
        same-key_hash row beat this call. A non-unique IntegrityError must
        propagate unchanged; only the unique-key case is absorbed."""
        ...

    def classify_integrity_error(
        self,
        error: BaseException,
    ) -> IntegrityViolationKind:
        """Map a caught IntegrityError to the kernel-level kind. Implementations
        should classify the original driver-level error (``error.orig`` for a
        SQLAlchemy IntegrityError) and return ``UNKNOWN`` for anything they do
        not specifically recognize so the caller re-raises the original."""
        ...


__all__: "list[str]" = (
    "InsertResult",
    "IntegrityViolationKind",
    "SqlAlchemyObjectDialect",
)
