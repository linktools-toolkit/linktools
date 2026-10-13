#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Timeline metadata shares bounded reads without holding authorization or bodies."""

from pathlib import Path
from dataclasses import replace

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, HmacCursorSigner, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.runtime._session_timeline import project_session_timeline
from linktools.ai.runtime.state._store import active_state_scope

from .test_incremental_projection_boundaries import _storage
from .test_session_timeline import _execution, _session

pytestmark = pytest.mark.asyncio


class _Authorization(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        assert active_state_scope() is None
        return await super().authorize(principal, action, resource)


async def _seed(state: RuntimeStorage) -> None:
    await state.conversation.sessions.create(_session())
    await state.execution.executions.create(replace(_execution(
        "failed", status=ExecutionStatus.FAILED, prompt="original prompt",
        error_code="MODEL_UNAVAILABLE",
    ), result=None))
    await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id="failed", expected=None,
    )


async def _timeline(state: RuntimeStorage, owner: str = "owner"):
    return await project_session_timeline(
        state.conversation, state.execution.executions, _Authorization(),
        HmacCursorSigner("timeline", b"timeline-read-scope"), state.run_store,
        "session", principal=Principal(owner, "tenant"), limit=10,
    )


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_timeline_read_scope_preserves_turns_and_authority(
    tmp_path: Path, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="timeline-scope", tenant_id="tenant")
    try:
        await _seed(state)
        result = await _timeline(state)
        assert len(result.items) == 1
        assert result.items[0].execution_id == "failed"
        assert result.items[0].user_input["prompt"]["text"] == "original prompt"
        assert result.items[0].conversation_committed is False
        assert result.next_cursor is None
        assert active_state_scope() is None
        with pytest.raises(AIError) as error:
            await _timeline(state, "other-owner")
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


async def test_timeline_read_scope_uses_one_post_authorization_sql_snapshot(tmp_path: Path) -> None:
    state = RuntimeStorage.sqlite(tmp_path / "timeline.db")
    await state.initialize(namespace="timeline-scope", tenant_id="tenant")
    try:
        await _seed(state)
        engine = state.conversation.sessions.state_store.storage_group.context.engine
        statements = []

        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
        try:
            result = await _timeline(state)
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        assert len(result.items) == 1
        # A separate authority read, then one snapshot for both routed stores.
        assert statements.count("BEGIN") <= 2
        assert sum(value.lstrip().upper().startswith("SELECT") for value in statements) <= 6
    finally:
        await state.close()


async def test_timeline_read_scope_keeps_separate_database_routes(tmp_path: Path) -> None:
    state = RuntimeStorage.from_plan(RuntimeStoragePlan(
        execution=RuntimeStorageRoute.sqlite(tmp_path / "execution.db"),
        conversation=RuntimeStorageRoute.sqlite(tmp_path / "conversation.db"),
    ))
    await state.initialize(namespace="timeline-scope", tenant_id="tenant")
    try:
        await _seed(state)
        assert (state.conversation.sessions.state_store.storage_group is not
                state.execution.executions.state_store.storage_group)
        result = await _timeline(state)
        assert [turn.execution_id for turn in result.items] == ["failed"]
        assert active_state_scope() is None
    finally:
        await state.close()
