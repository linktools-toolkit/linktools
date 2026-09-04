#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-specific Runtime storage I/O contracts."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.core import SessionStatus
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state._contracts import SessionRecord
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine

pytestmark = pytest.mark.asyncio


async def test_session_list_skips_generation_probe_but_page_keeps_it(
    tmp_path: Path,
) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'sessions.db'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="io-session-list", tenant_id="tenant")
    now = datetime.now(timezone.utc)
    await state.conversation.sessions.create(
        SessionRecord(
            session_id="session",
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
    )
    statements: list[str] = []

    def capture_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture_sql)
    try:
        values = await state.conversation.sessions.list(tenant_id="tenant")
        assert tuple(value.session_id for value in values) == ("session",)
        assert not any(
            "AI_STATE_SEQUENCES" in statement.upper()
            for statement in statements
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
            for statement in statements
        )
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture_sql)
        await state.close()
        await engine.dispose()
