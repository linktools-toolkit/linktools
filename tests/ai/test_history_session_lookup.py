#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Known-ID session discovery preserves the caller's owned-session scope."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event

from linktools.ai.core import AuthorizationAction, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage

from .test_session_timeline import _session

pytestmark = pytest.mark.asyncio
_OWNER = Principal("owner", "tenant")


def _reader(state, *, authorization=None):
    return History(
        SimpleNamespace(), tenant_id="tenant", sessions=state.conversation.sessions,
        executions=state.execution.executions,
        authorization=authorization or TenantAuthorizationPolicy(),
    )


@pytest.mark.parametrize("scope", ("owned", "missing", "other_owner"))
async def test_session_lookup_uses_point_reads_within_the_list_scope(
    monkeypatch: pytest.MonkeyPatch, scope: str,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-lookup", tenant_id="tenant")
    actions = []

    class Authorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            actions.append((action, resource.id, resource.owner_principal_id))
            await super().authorize(principal, action, resource)

    async def no_scan_or_write(*args, **kwargs):
        raise AssertionError("known-ID lookup must not scan pages or mutate storage")

    try:
        if scope != "missing":
            await state.conversation.sessions.create(replace(
                _session(), owner_principal_id="other" if scope == "other_owner" else "owner",
            ))
        with monkeypatch.context() as patch:
            patch.setattr(state.conversation.sessions, "list_page", no_scan_or_write)
            patch.setattr(state.conversation.sessions.state_store, "mutate", no_scan_or_write)
            view = await _reader(state, authorization=Authorization()).lookup_session(
                "session", principal=_OWNER,
            )
        expected = [(AuthorizationAction.SESSION_READ, "list", None)]
        if scope == "owned":
            assert view.session_id == "session"
            assert view.active_execution_id is None
            expected.append((AuthorizationAction.SESSION_READ, "session", "owner"))
        else:
            assert view is None
        assert actions == expected
    finally:
        await state.close()


@pytest.mark.parametrize("scope", ("owned", "missing", "other_owner"))
async def test_native_session_state_is_an_authorized_point_read(tmp_path: Path, scope: str) -> None:
    class Authorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            assert action is AuthorizationAction.SESSION_READ
            assert resource.id == "session"
            assert resource.owner_principal_id is None
            if scope == "other_owner":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async with Persistence.open(
        "session-lookup", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "state.db"),
    ) as state:
        if scope != "missing":
            await state.session_turns.create("session", config={})
        async with History.open(persistence=state, authorization=Authorization()) as history:
            engine = state._stores[PersistenceDomain.CONVERSATION].context.engine.sync_engine
            statements = []

            def capture(_connection, _cursor, sql, _parameters, _context, _many):
                if sql.split(None, 1)[0].upper() not in {"BEGIN", "PRAGMA"}:
                    statements.append(sql)

            event.listen(engine, "before_cursor_execute", capture)
            try:
                if scope == "owned":
                    view = await history.turn_session_state("session", principal=_OWNER)
                    assert view["session_id"] == "session"
                    assert view["active_turn_id"] is None
                else:
                    with pytest.raises(AIError) as denied:
                        await history.turn_session_state("session", principal=_OWNER)
                    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            if scope == "other_owner":
                assert not statements
            else:
                sql, = statements
                assert sql.startswith("SELECT")
                assert "ai_sessions.session_key = " in sql
                assert "ORDER BY" not in sql


@pytest.mark.parametrize("resource_id", ("list", "session"))
@pytest.mark.parametrize("code", (ErrorCode.AUTHORIZATION_DENIED, ErrorCode.RUNTIME_DEPENDENCY_NOT_READY))
async def test_session_lookup_preserves_policy_errors(resource_id: str, code: ErrorCode) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-lookup", tenant_id="tenant")
    failure = AIError(code)

    class Authorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if resource.id == resource_id:
                raise failure

    try:
        await state.conversation.sessions.create(_session())
        with pytest.raises(AIError) as error:
            await _reader(state, authorization=Authorization()).lookup_session("session", principal=_OWNER)
        assert error.value is failure
    finally:
        await state.close()


async def test_session_lookup_authorizes_scope_before_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-lookup", tenant_id="tenant")

    class Authorization:
        async def authorize(self, principal, action, resource):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async def no_read(*args, **kwargs):
        raise AssertionError("discovery must be authorized before reading the ID")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(state.conversation.sessions, "get", no_read)
            with pytest.raises(AIError) as error:
                await _reader(state, authorization=Authorization()).lookup_session("absent", principal=_OWNER)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


async def test_session_lookup_rejects_another_reader_tenant() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-lookup", tenant_id="tenant")

    class Authorization:
        async def authorize(self, principal, action, resource):
            pass

    try:
        with pytest.raises(AIError) as error:
            await _reader(state, authorization=Authorization()).lookup_session(
                "absent", principal=Principal("owner", "other-tenant"),
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


@pytest.mark.parametrize("change", ("removed", "owner", "identity", "metadata", "failure"))
async def test_session_lookup_rechecks_the_authorized_record(
    monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-lookup", tenant_id="tenant")
    authorized = False
    failure = AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    class Authorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            nonlocal authorized
            await super().authorize(principal, action, resource)
            if resource.id == "session":
                authorized = True

    try:
        original = await state.conversation.sessions.create(_session())

        async def get(session_id, *, tenant_id):
            assert (session_id, tenant_id) == ("session", "tenant")
            if not authorized:
                return original
            if change == "removed":
                return None
            if change == "owner":
                return replace(original, owner_principal_id="other")
            if change == "identity":
                return replace(original, session_id="other")
            if change == "failure":
                raise failure
            return replace(original, metadata={"source": "current"})

        with monkeypatch.context() as patch:
            patch.setattr(state.conversation.sessions, "get", get)
            reader = _reader(state, authorization=Authorization())
            if change in {"owner", "identity", "failure"}:
                with pytest.raises(AIError) as error:
                    await reader.lookup_session("session", principal=_OWNER)
                expected = {
                    "owner": ErrorCode.AUTHORIZATION_DENIED,
                    "identity": ErrorCode.STORAGE_INTEGRITY_ERROR,
                    "failure": ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                }
                assert error.value.code is expected[change]
                if change == "failure":
                    assert error.value is failure
            else:
                view = await reader.lookup_session("session", principal=_OWNER)
                if change == "removed":
                    assert view is None
                else:
                    assert view.metadata == {"source": "current"}
    finally:
        await state.close()


@pytest.mark.parametrize("corrupt", (False, True))
async def test_session_lookup_preserves_initial_repository_failures(
    monkeypatch: pytest.MonkeyPatch, corrupt: bool,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="session-lookup", tenant_id="tenant")

    async def get(session_id, *, tenant_id):
        if corrupt:
            return replace(_session(), session_id="wrong-id", owner_principal_id="other")
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(state.conversation.sessions, "get", get)
            with pytest.raises(AIError) as error:
                await _reader(state).lookup_session("session", principal=_OWNER)
        assert error.value.code is (
            ErrorCode.STORAGE_INTEGRITY_ERROR if corrupt else ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        )
    finally:
        await state.close()
