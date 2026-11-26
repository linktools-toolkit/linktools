#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Timeline metadata shares bounded reads without holding authorization or bodies."""

from linktools.ai.persistence import Persistence
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.core import AuthorizationAction, ExecutionStatus, HmacCursorSigner, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage, PersistencePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.history import History
from linktools.ai.history._session_timeline import project_session_timeline
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.persistence.turns import SessionTurnKey

from .test_session_timeline import _commit_empty_turn, _execution, _session
from .test_session_turn_sql import _BINDING

pytestmark = pytest.mark.asyncio


class _Authorization(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        assert active_state_scope() is None
        return await super().authorize(principal, action, resource)


async def _seed(state: Persistence) -> None:
    await state.conversation.sessions.create(_session())
    await state.execution.executions.create(_execution(
        "failed", status=ExecutionStatus.FAILED, prompt="original prompt",
        error_code="MODEL_UNAVAILABLE",
    ))
    await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id="failed", expected=None,
    )
    await _commit_empty_turn(state, await state.execution.executions.get("failed", tenant_id="tenant"))
    await state.conversation.sessions.release_execution(
        "session", tenant_id="tenant", execution_id="failed",
    )
    await state.execution.executions.create(
        _execution("active", status=ExecutionStatus.PENDING_START, prompt="pending prompt"),
    )
    await state.conversation.sessions.admit_execution(
        "session", tenant_id="tenant", execution_id="active", expected=None,
    )


async def _timeline(state: Persistence, owner: str = "owner"):
    return await project_session_timeline(
        state.conversation, state.execution.executions, _Authorization(),
        HmacCursorSigner("timeline", b"timeline-read-scope"), state.run_store,
        "session", principal=Principal(owner, "tenant"), limit=10,
    )


async def test_timeline_read_scope_preserves_turns_and_authority_in_memory() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="timeline-scope", tenant_id="tenant")
    try:
        await _seed(state)
        result = await _timeline(state)
        assert len(result.items) == 2
        assert result.items[0].execution_id == "failed"
        assert result.items[0].user_input["prompt"]["text"] == "original prompt"
        assert result.items[0].conversation_committed is True
        assert result.items[0].items == ()
        assert result.items[1].execution_id == "active"
        assert result.items[1].conversation_committed is False
        assert result.next_cursor is None
        assert active_state_scope() is None
        with pytest.raises(AIError) as error:
            await _timeline(state, "other-owner")
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


class _NativeAuthorization(_Authorization):
    def __init__(self, markers: list[tuple[str, object]]) -> None:
        self.markers = markers

    async def authorize(self, principal, action, resource):
        assert action is AuthorizationAction.SESSION_READ and resource.id == "session"
        self.markers.append(("auth", principal.principal_id))
        if principal.principal_id != "owner":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await super().authorize(principal, action, resource)


async def _seed_native(state: Persistence) -> None:
    turns = state.session_turns
    assert turns is not None
    await turns.create("session", config={})

    async def accept(turn_id: str, prompt: str) -> None:
        session = await turns.get_session("session")
        assert session is not None
        assert await turns.accept(
            SessionTurnKey("session", turn_id), expected_revision=session["revision"],
            context_from=1, context_through=session["last_turn_no"], request_key=turn_id,
            request={"prompt": prompt, "principal": {"principal_id": "owner", "tenant_id": "tenant"}},
            binding=_BINDING, lease_seconds=600,
        )

    await accept("failed", "original prompt")
    assert await turns.finish(
        SessionTurnKey("session", "failed"), generation=1, turn_no=1, state="failed",
        result={"error_code": ErrorCode.MODEL_UNAVAILABLE.value},
    )
    await accept("active", "pending prompt")


@pytest.mark.parametrize("split", (False, True))
async def test_native_timeline_reads_only_authorized_conversation_route(tmp_path: Path, split: bool) -> None:
    conversation = StorageRoute.sqlite(tmp_path / "conversation.db")
    execution = StorageRoute.sqlite(tmp_path / "execution.db") if split else conversation
    storage = PersistenceStorage.from_plan(PersistencePlan(execution=execution, conversation=conversation))
    markers: list[tuple[str, object]] = []
    async with (
        Persistence.open("timeline-scope", storage=storage, tenant_id="tenant") as state,
        History.open(persistence=state, authorization=_NativeAuthorization(markers)) as history,
    ):
        await _seed_native(state)
        owner = Principal("owner", "tenant")
        conversation_engine = state.conversation.histories.state_store.storage_group.context.engine.sync_engine
        execution_engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine
        assert (conversation_engine is not execution_engine) is split
        # Warm shared archive descriptors; then count physical business reads.
        expected = await history.turn_session("session", principal=owner, limit=10)

        def capture(connection, _cursor, statement, _parameters, _context, _many):
            if connection.engine in (conversation_engine, execution_engine):
                markers.append(("sql", (connection.engine, statement.lstrip().upper())))

        markers.clear()
        event.listen(Engine, "before_cursor_execute", capture)
        try:
            with pytest.raises(AIError) as denied:
                await history.turn_session("session", principal=Principal("other-owner", "tenant"), limit=10)
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert markers == [("auth", "other-owner")]
            markers.clear()

            result = await history.turn_session("session", principal=owner, limit=10)
            assert markers[0] == ("auth", "owner")
        finally:
            event.remove(Engine, "before_cursor_execute", capture)

        assert result == expected
        [failed] = result["timeline"]["items"]
        assert failed["turn_id"] == "failed"
        assert failed["status"] == "FAILED" and failed["conversation_committed"] is True
        assert failed["user_input"]["prompt"] == "original prompt"
        assert failed["error_code"] == ErrorCode.MODEL_UNAVAILABLE.value
        assert failed["items"] == [] and failed["process"] == [] and failed["output"] is None
        assert result["timeline"]["next_cursor"] is None
        assert result["session"]["active_turn_id"] == "active"
        queries = [entry for kind, entry in markers if kind == "sql"]
        pending = await history.turn_result("session", "active", principal=owner)
        assert pending["status"] == "PENDING_START" and pending["conversation_committed"] is False
        assert pending["user_input"]["prompt"] == "pending prompt"
        assert active_state_scope() is None

        statements = [statement for _, statement in queries]
        assert all(engine is conversation_engine for engine, _ in queries)
        selects = [statement for statement in statements if statement.startswith("SELECT")]
        assert len(selects) == 2
        assert "FROM AI_SESSIONS" in selects[0] and "FROM AI_SESSION_TURNS" in selects[1]
        assert statements.count("BEGIN") <= 2
