#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-first dialect package.

The dialect is the ONLY seam at which a downstream that brings its own engine
plugs into the storage kernel's object backend. Core does NOT branch on a
dialect name and does NOT resolve a strategy from a session_factory; the
backend is HANDED a ``SqlAlchemyObjectDialect`` at construction.

Core ships exactly one concrete implementation -- :class:`SqliteObjectDialect`
-- for SQLite/aiosqlite and as the reference for downstream dialect authors.
A downstream wanting a different vendor (its own engine + driver) implements
``SqlAlchemyObjectDialect`` itself and runs the kernel's conformance suite in
its own CI. Core never names a real production database product here."""

from .base import IntegrityViolationKind, InsertResult, SqlAlchemyObjectDialect
from .sqlite import SqliteObjectDialect

__all__: "list[str]" = (
    "InsertResult",
    "IntegrityViolationKind",
    "SqlAlchemyObjectDialect",
    "SqliteObjectDialect",
)
