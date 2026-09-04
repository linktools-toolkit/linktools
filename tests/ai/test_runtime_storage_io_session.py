#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-specific Runtime storage I/O contracts."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.core import (
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    ResourceKind,
    SessionStatus,
    canonical_sha256,
)
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._contracts import SessionRecord
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

pytestmark = pytest.mark.asyncio


def _session(session_id: str, now: datetime) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        tenant_id="tenant",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
    )


def _fork_operation(now: datetime) -> OperationLedgerInput:
    return OperationLedgerInput(
        operation_id="fork-operation",
        tenant_id="tenant",
        resource_kind=ResourceKind.SESSION,
        resource_id="target",
        execution_id=None,
        operation_kind=OperationKind.SESSION_FORK,
        status=OperationStatus.SUCCEEDED,
        request_digest=canonical_sha256({"action": "fork", "target": "target"}),
        result_ref="target",
        result_digest=canonical_sha256({"session_id": "target", "revision": 0}),
        error_code=None,
        compactable=True,
        created_at=now,
        updated_at=now,
    )


def _listen_sql(engine: AsyncEngine, statements: list[tuple[str, object]]) -> object:
    def capture_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append((statement, parameters))

    event.listen(engine.sync_engine, "before_cursor_execute", capture_sql)
    return capture_sql


def _record_selects(statements: list[tuple[str, object]]) -> list[tuple[str, object]]:
    return [
        (statement.upper(), parameters)
        for statement, parameters in statements
        if "AI_STATE_RECORDS" in statement.upper()
        and statement.lstrip().upper().startswith("SELECT")
    ]


def _parameter_count(parameters: object) -> int:
    if isinstance(parameters, (tuple, list, dict)):
        return len(parameters)
    return 0


async def _build_fork_state(tmp_path: Path, namespace: str) -> tuple[
    AsyncEngine,
    RuntimeState,
    SessionRecord,
    OperationLedgerInput,
]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'{namespace}.db'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace=namespace, tenant_id="tenant")
    now = datetime.now(timezone.utc)
    source = await state.conversation.sessions.create(_session("source", now))
    return engine, state, source, _fork_operation(now)


async def test_session_list_skips_generation_probe_but_page_keeps_it(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="io-session-list", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    await state.conversation.sessions.create(_session("session", now))
    statements: list[tuple[str, object]] = []
    capture_sql = _listen_sql(engine, statements)
    try:
        values = await state.conversation.sessions.list(tenant_id="tenant")
        assert tuple(value.session_id for value in values) == ("session",)
        assert not any(
            "AI_STATE_SEQUENCES" in statement.upper()
            for statement, _parameters in statements
        )

        statements.clear()
        generation, page = await state.conversation.sessions.list_page(
            tenant_id="tenant",
            owner_principal_id=None,
            cursor=None,
            limit=10,
        )
        assert generation >= 1
        assert tuple(value.session_id for value in page.items) == ("session",)
        assert any(
            "AI_STATE_SEQUENCES" in statement.upper()
            for statement, _parameters in statements
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await state.close()
        await engine.dispose()


async def test_session_fork_batches_known_record_sql(tmp_path: Path) -> None:
    engine, state, source, operation = await _build_fork_state(
        tmp_path,
        "io-session-fork",
    )
    statements: list[tuple[str, object]] = []
    capture_sql = _listen_sql(engine, statements)
    try:
        target, replayed = await state.conversation.sessions.create_fork_with_operation(
            source.session_id,
            _session("target", operation.created_at),
            expected_source_revision=source.revision,
            operation=operation,
        )
        assert target.session_id == "target"
        assert replayed is False

        record_selects = _record_selects(statements)
        assert len(record_selects) == 2
        batched = [
            parameters
            for statement, parameters in record_selects
            if " IN " in statement
        ]
        assert len(batched) == 1
        assert _parameter_count(batched[0]) == 4
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await state.close()
        await engine.dispose()


async def test_session_fork_replay_batches_known_record_sql(tmp_path: Path) -> None:
    engine, state, source, operation = await _build_fork_state(
        tmp_path,
        "io-session-fork-replay",
    )
    target = _session("target", operation.created_at)
    await state.conversation.sessions.create_fork_with_operation(
        source.session_id,
        target,
        expected_source_revision=source.revision,
        operation=operation,
    )

    statements: list[tuple[str, object]] = []
    capture_sql = _listen_sql(engine, statements)
    try:
        replayed_target, replayed = (
            await state.conversation.sessions.create_fork_with_operation(
                source.session_id,
                target,
                expected_source_revision=source.revision,
                operation=operation,
            )
        )
        assert replayed_target.session_id == "target"
        assert replayed is True

        record_selects = _record_selects(statements)
        assert len(record_selects) == 2
        batched = [
            parameters
            for statement, parameters in record_selects
            if " IN " in statement
        ]
        assert len(batched) == 1
        assert _parameter_count(batched[0]) == 3
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await state.close()
        await engine.dispose()
