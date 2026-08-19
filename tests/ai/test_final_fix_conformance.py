#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for close retry arbitration and SQLite pool setup."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.core import (
    OperationKind,
    OperationLedgerRecord,
    OperationStatus,
    Principal,
    ResourceKind,
    SessionStatus,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import CloseSessionRequest, DefaultSessionService
from linktools.ai.storage import create_sql_storage_context, provision_sql, validate_sql
from sqlalchemy.ext.asyncio import create_async_engine


def _operation(
    status: OperationStatus = OperationStatus.PENDING,
    *,
    result_ref: str | None = None,
    result_digest: str | None = None,
    error_code: str | None = None,
) -> OperationLedgerRecord:
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return OperationLedgerRecord(
        "operation",
        "tenant",
        ResourceKind.SESSION,
        "session",
        None,
        OperationKind.SESSION_CLOSE,
        status,
        "request",
        result_ref,
        result_digest,
        error_code,
        True,
        4,
        timestamp,
        timestamp,
    )


class _OperationRepository:
    def __init__(self, current: OperationLedgerRecord, error: AIError) -> None:
        self.current = current
        self.error = error
        self.compare_calls = 0

    async def compare_and_swap(
        self,
        operation_id: str,
        *,
        tenant_id: str,
        expected_status: OperationStatus,
        next_record: OperationLedgerRecord,
    ) -> OperationLedgerRecord:
        del operation_id, tenant_id, expected_status, next_record
        self.compare_calls += 1
        raise self.error

    async def get(
        self,
        operation_id: str,
        *,
        tenant_id: str,
    ) -> OperationLedgerRecord:
        del operation_id, tenant_id
        return self.current


def _service(repository: _OperationRepository) -> DefaultSessionService:
    service = object.__new__(DefaultSessionService)
    service._conversation = SimpleNamespace(operations=repository)
    return service


@pytest.mark.asyncio
async def test_close_completion_rethrows_valid_pending_conflict() -> None:
    repository = _OperationRepository(
        _operation(),
        AIError(ErrorCode.STORAGE_CONFLICT),
    )
    service = _service(repository)

    with pytest.raises(AIError) as raised:
        await service._complete_close_operation(
            _operation(),
            "tenant",
            "session",
            "closed-digest",
        )

    assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    assert repository.compare_calls == 1


@pytest.mark.asyncio
async def test_close_completion_accepts_exact_succeeded_replay() -> None:
    current = _operation(
        OperationStatus.SUCCEEDED,
        result_ref="session",
        result_digest="closed-digest",
    )
    repository = _OperationRepository(
        current,
        AIError(ErrorCode.STORAGE_CONFLICT),
    )
    service = _service(repository)

    await service._complete_close_operation(
        _operation(),
        "tenant",
        "session",
        "closed-digest",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "expected_code"),
    (
        (
            _operation(result_ref="session"),
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
        (
            _operation(OperationStatus.RUNNING),
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
        (
            _operation(OperationStatus.FAILED),
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
        (
            replace(_operation(), sequence=5),
            ErrorCode.STORAGE_INTEGRITY_ERROR,
        ),
    ),
)
async def test_close_completion_rejects_invalid_conflict_state(
    current: OperationLedgerRecord,
    expected_code: ErrorCode,
) -> None:
    repository = _OperationRepository(
        current,
        AIError(ErrorCode.STORAGE_CONFLICT),
    )
    service = _service(repository)

    with pytest.raises(AIError) as raised:
        await service._complete_close_operation(
            _operation(),
            "tenant",
            "session",
            "closed-digest",
        )

    assert raised.value.code is expected_code


class _SessionRepository:
    def __init__(
        self,
        records: tuple[object, ...],
        error: AIError | None,
        transition_result: object | None = None,
    ) -> None:
        self.records = list(records)
        self.error = error
        self.transition_result = transition_result
        self.transition_calls = 0

    async def get(self, session_id: str, *, tenant_id: str) -> object:
        del session_id, tenant_id
        return self.records.pop(0) if len(self.records) > 1 else self.records[0]

    async def transition_status(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.transition_calls += 1
        if self.error is not None and self.transition_calls == 1:
            raise self.error
        return self.transition_result


def _force_close_service(
    record: object,
    sessions: _SessionRepository,
    operation: OperationLedgerRecord,
) -> DefaultSessionService:
    service = object.__new__(DefaultSessionService)
    service._conversation = SimpleNamespace(
        sessions=sessions,
        operations=SimpleNamespace(),
    )

    async def authorized(
        session_id: str,
        principal: Principal,
        action: object,
    ) -> object:
        del session_id, principal, action
        return record

    async def begin_close(
        operation_id: str,
        tenant_id: str,
        session_id: str,
        request_digest: str,
    ) -> OperationLedgerRecord:
        del operation_id, tenant_id, session_id, request_digest
        return operation

    service._authorized = authorized
    service._begin_close_operation = begin_close
    return service


def _open_session() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session",
        tenant_id="tenant",
        status=SessionStatus.OPEN,
        active_execution_id=None,
        continuation=None,
        revision=0,
    )


@pytest.mark.asyncio
async def test_force_close_does_not_retry_storage_conflict() -> None:
    record = _open_session()
    sessions = _SessionRepository(
        (record,),
        AIError(ErrorCode.STORAGE_CONFLICT),
    )
    operation = _operation()
    service = _force_close_service(record, sessions, operation)
    request = CloseSessionRequest(
        Principal("principal", "tenant"),
        "close-key",
        force=True,
    )

    with pytest.raises(AIError) as raised:
        await service._close("session", request)

    assert raised.value.code is ErrorCode.STORAGE_CONFLICT
    assert sessions.transition_calls == 1


@pytest.mark.asyncio
async def test_force_close_rethrows_session_conflict_when_still_open() -> None:
    record = _open_session()
    sessions = _SessionRepository(
        (record, record),
        AIError(ErrorCode.SESSION_CONFLICT),
    )
    service = _force_close_service(record, sessions, _operation())
    request = CloseSessionRequest(
        Principal("principal", "tenant"),
        "close-key",
        force=True,
    )

    with pytest.raises(AIError) as raised:
        await service._close("session", request)

    assert raised.value.code is ErrorCode.SESSION_CONFLICT
    assert sessions.transition_calls == 1


@pytest.mark.asyncio
async def test_force_close_continues_after_session_conflict_progress() -> None:
    opening = _open_session()
    closing = SimpleNamespace(
        **{**vars(opening), "status": SessionStatus.CLOSING}
    )
    closed = SimpleNamespace(
        **{**vars(closing), "status": SessionStatus.CLOSED, "revision": 1}
    )
    sessions = _SessionRepository(
        (opening, closing, closing),
        AIError(ErrorCode.SESSION_CONFLICT),
        transition_result=closed,
    )
    service = _force_close_service(opening, sessions, _operation())

    async def complete_close(
        operation: OperationLedgerRecord,
        tenant_id: str,
        result_ref: str,
        result_digest: str,
    ) -> None:
        del operation, tenant_id, result_ref, result_digest

    async def view(record: object, principal: Principal) -> str:
        del record, principal
        return "closed"

    async def release(
        session_id: str,
        tenant_id: str,
        continuation: object,
    ) -> None:
        del session_id, tenant_id, continuation

    service._complete_close_operation = complete_close
    service._view = view
    service._request_session_release = release
    request = CloseSessionRequest(
        Principal("principal", "tenant"),
        "close-key",
        force=True,
    )

    assert await service._close("session", request) == "closed"
    assert sessions.transition_calls == 2


@pytest.mark.asyncio
async def test_sqlite_pragmas_apply_to_preexisting_pool_connections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import event

    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}",
        pool_size=2,
        max_overflow=0,
    )
    prewarmed = [await engine.connect(), await engine.connect()]
    listen_calls = 0
    original_listen = event.listen

    def listen(*args: object, **kwargs: object) -> None:
        nonlocal listen_calls
        listen_calls += 1
        original_listen(*args, **kwargs)

    monkeypatch.setattr(event, "listen", listen)
    context = create_sql_storage_context(engine)
    await context.initialize()
    await context.initialize()
    for connection in prewarmed:
        await connection.close()
    from sqlalchemy import MetaData

    await provision_sql(engine, MetaData())
    await validate_sql(engine, MetaData())

    checked_out = [await engine.connect(), await engine.connect()]
    try:
        for connection in checked_out:
            foreign_keys = await connection.exec_driver_sql("PRAGMA foreign_keys")
            busy_timeout = await connection.exec_driver_sql("PRAGMA busy_timeout")
            assert foreign_keys.scalar_one() == 1
            assert busy_timeout.scalar_one() == 5000
    finally:
        for connection in checked_out:
            await connection.close()
        await context.close()
        async with engine.connect() as connection:
            foreign_keys = await connection.exec_driver_sql("PRAGMA foreign_keys")
            assert foreign_keys.scalar_one() == 1
        await engine.dispose()

    assert listen_calls == 1


def test_sqlite_checkout_configuration_propagates_pragma_failure() -> None:
    from linktools.ai.storage._dialects import configure_sqlite_connection

    class _Cursor:
        closed = False

        def execute(self, statement: str) -> None:
            del statement
            raise RuntimeError("pragma failed")

        def close(self) -> None:
            self.closed = True

    class _Connection:
        def __init__(self, cursor: _Cursor) -> None:
            self._cursor = cursor

        def cursor(self) -> _Cursor:
            return self._cursor

    cursor = _Cursor()
    with pytest.raises(RuntimeError, match="pragma failed"):
        configure_sqlite_connection(_Connection(cursor), object(), object())
    assert cursor.closed
