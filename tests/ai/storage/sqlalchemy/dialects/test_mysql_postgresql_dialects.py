#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit coverage for MySQLDialect / PostgreSQLDialect. No live
MySQL/PostgreSQL server is available in this sandbox, so
``insert_ignore_conflict`` is exercised against a fake session that records
the statement it was handed (compiled against the real vendor dialect to
confirm the SQL shape) and returns a scripted rowcount;
``classify_integrity_error`` is exercised against synthetic exception objects
carrying the same attributes the real drivers expose.

``insert_ignore_conflict`` is generic over the target model -- it is the one
seam every create-only SQL store shares (the storage kernel's object backend
AND the artifact-record store), so one test below drives it against
``StorageObjectRow`` and another against ``ArtifactRecordRow`` to confirm
niether dialect hardcodes the kernel's table/columns."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.dialects import mysql, postgresql

from linktools.ai.storage.backends.sqlalchemy.models import StorageObjectRow
from linktools.ai.storage.sqlalchemy.dialects import (
    IntegrityViolationKind,
    MySQLDialect,
    PostgreSQLDialect,
)
from linktools.ai.storage.sqlalchemy.models import ArtifactRecordRow

_VALUES = {
    "key": "k1",
    "key_hash": b"\x00" * 32,
    "etag": "etag-1",
    "version": 1,
    "content_type": None,
    "size": 0,
    "content": b"",
    "modified_at": None,
    "metadata_json": "{}",
    "tombstone": False,
    "commit_revision": 1,
}

_ARTIFACT_VALUES = {
    "artifact_id": "art-1",
    "tenant_id": "tenant-1",
    "content_hash": "a" * 64,
    "producer_kind": "tool",
    "producer_id": None,
    "run_id": "run-1",
    "data_json": "{}",
}


class _FakeResult:
    """Fake execute() result. When ``rowcount=1``, simulates a landed INSERT:
    SQLite/PG dialects call ``.first()`` (RETURNING), MySQL reads
    ``.lastrowid`` (driver-set auto-increment). Both return the fake row_id."""

    def __init__(self, rowcount: int, *, row_id: int = 42) -> None:
        self.rowcount = rowcount
        self.lastrowid = row_id if rowcount > 0 else 0
        self._row_id = row_id

    def first(self):
        return (self._row_id,) if self.rowcount > 0 else None


class _FakeSession:
    def __init__(self, rowcount: int) -> None:
        self._rowcount = rowcount
        self.executed_stmt: Any = None

    async def execute(self, stmt: Any) -> _FakeResult:
        self.executed_stmt = stmt
        return _FakeResult(self._rowcount)


class _FakeOrig(Exception):
    def __init__(self, *, args=(), sqlstate=None, pgcode=None):
        super().__init__(*args)
        if sqlstate is not None:
            self.sqlstate = sqlstate
        if pgcode is not None:
            self.pgcode = pgcode


class _FakeIntegrityError(Exception):
    def __init__(self, orig: BaseException) -> None:
        super().__init__(str(orig))
        self.orig = orig


class TestMySQLDialect:
    def test_name(self):
        assert MySQLDialect().name == "mysql"

    @pytest.mark.asyncio
    async def test_insert_ignore_conflict_landed_on_object_row(self):
        dialect = MySQLDialect()
        session = _FakeSession(rowcount=1)

        result = await dialect.insert_ignore_conflict(
            session, model=StorageObjectRow, values=_VALUES, index_elements=["key_hash"]
        )

        assert result.inserted is True
        assert result.row_id == 42  # lastrowid populated
        compiled = str(session.executed_stmt.compile(dialect=mysql.dialect()))
        assert "ON DUPLICATE KEY UPDATE" in compiled

    @pytest.mark.asyncio
    async def test_insert_ignore_conflict_conflict_on_object_row(self):
        dialect = MySQLDialect()
        session = _FakeSession(rowcount=0)

        result = await dialect.insert_ignore_conflict(
            session, model=StorageObjectRow, values=_VALUES, index_elements=["key_hash"]
        )

        assert result.inserted is False
        assert result.row_id is None

    @pytest.mark.asyncio
    async def test_insert_ignore_conflict_landed_on_artifact_record_row(self):
        """The same dialect, unmodified, drives a different model's table --
        proving the seam is generic and not hardcoded to the kernel's
        StorageObjectRow/key_hash."""
        dialect = MySQLDialect()
        session = _FakeSession(rowcount=1)

        result = await dialect.insert_ignore_conflict(
            session,
            model=ArtifactRecordRow,
            values=_ARTIFACT_VALUES,
            index_elements=["artifact_id"],
        )

        assert result.inserted is True
        compiled = str(session.executed_stmt.compile(dialect=mysql.dialect()))
        assert "ON DUPLICATE KEY UPDATE" in compiled
        assert "artifact_records" in compiled

    def test_classify_integrity_error_unique_by_code(self):
        error = _FakeIntegrityError(_FakeOrig(args=(1062, "Duplicate entry")))
        kind = MySQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.UNIQUE_CONFLICT

    def test_classify_integrity_error_foreign_key_by_code(self):
        error = _FakeIntegrityError(_FakeOrig(args=(1451, "Cannot delete or update")))
        kind = MySQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.FOREIGN_KEY

    def test_classify_integrity_error_check_by_code(self):
        error = _FakeIntegrityError(_FakeOrig(args=(3819, "Check constraint failed")))
        kind = MySQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.CHECK

    def test_classify_integrity_error_falls_back_to_message(self):
        error = _FakeIntegrityError(_FakeOrig(args=("Duplicate entry 'x' for key",)))
        kind = MySQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.UNIQUE_CONFLICT

    def test_classify_integrity_error_unknown(self):
        error = _FakeIntegrityError(_FakeOrig(args=(9999, "something else")))
        kind = MySQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.UNKNOWN


class TestPostgreSQLDialect:
    def test_name(self):
        assert PostgreSQLDialect().name == "postgresql"

    @pytest.mark.asyncio
    async def test_insert_ignore_conflict_landed_on_object_row(self):
        dialect = PostgreSQLDialect()
        session = _FakeSession(rowcount=1)

        result = await dialect.insert_ignore_conflict(
            session, model=StorageObjectRow, values=_VALUES, index_elements=["key_hash"]
        )

        assert result.inserted is True
        assert result.row_id == 42  # RETURNING populated
        compiled = str(session.executed_stmt.compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT" in compiled
        assert "DO NOTHING" in compiled

    @pytest.mark.asyncio
    async def test_insert_ignore_conflict_conflict_on_object_row(self):
        dialect = PostgreSQLDialect()
        session = _FakeSession(rowcount=0)

        result = await dialect.insert_ignore_conflict(
            session, model=StorageObjectRow, values=_VALUES, index_elements=["key_hash"]
        )

        assert result.inserted is False
        assert result.row_id is None

    @pytest.mark.asyncio
    async def test_insert_ignore_conflict_landed_on_artifact_record_row(self):
        """The same dialect, unmodified, drives a different model's table --
        proving the seam is generic and not hardcoded to the kernel's
        StorageObjectRow/key_hash."""
        dialect = PostgreSQLDialect()
        session = _FakeSession(rowcount=1)

        result = await dialect.insert_ignore_conflict(
            session,
            model=ArtifactRecordRow,
            values=_ARTIFACT_VALUES,
            index_elements=["artifact_id"],
        )

        assert result.inserted is True
        compiled = str(session.executed_stmt.compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT" in compiled
        assert "DO NOTHING" in compiled
        assert "artifact_records" in compiled

    def test_classify_integrity_error_unique_by_sqlstate(self):
        error = _FakeIntegrityError(_FakeOrig(sqlstate="23505"))
        kind = PostgreSQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.UNIQUE_CONFLICT

    def test_classify_integrity_error_foreign_key_by_pgcode(self):
        error = _FakeIntegrityError(_FakeOrig(pgcode="23503"))
        kind = PostgreSQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.FOREIGN_KEY

    def test_classify_integrity_error_check_by_sqlstate(self):
        error = _FakeIntegrityError(_FakeOrig(sqlstate="23514"))
        kind = PostgreSQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.CHECK

    def test_classify_integrity_error_falls_back_to_message(self):
        error = _FakeIntegrityError(_FakeOrig(args=("duplicate key value violates unique constraint",)))
        kind = PostgreSQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.UNIQUE_CONFLICT

    def test_classify_integrity_error_unknown(self):
        error = _FakeIntegrityError(_FakeOrig(args=("something else",)))
        kind = PostgreSQLDialect().classify_integrity_error(error)
        assert kind is IntegrityViolationKind.UNKNOWN
