#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Session listings follow a host's authenticated identifier scope."""

import pytest
from sqlalchemy import event

from linktools.ai.core import AuthorizationAction, Principal, ResourceKind, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage


@pytest.mark.asyncio
async def test_native_session_list_uses_verified_prefix_before_reading_sqlite(tmp_path):
    alice, bob = Principal("alice", "tenant"), Principal("bob", "tenant")
    prefixes = {"alice": "共享/a/", "bob": "共享/b/"}

    class Authorization(TenantAuthorizationPolicy):
        async def session_list_prefix(self, principal):
            return prefixes.get(principal.principal_id)

        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if (action is AuthorizationAction.SESSION_READ and resource.kind is ResourceKind.SESSION
                    and not resource.id.startswith(prefixes.get(principal.principal_id, "\x00"))):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    class ExplicitOnly(TenantAuthorizationPolicy):
        pass

    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("session-list", tenant_id="tenant", storage=storage) as state:
        for session_id in ("共享/a/zzz", "共享/b/only", "共享/a/aaa"):
            await state.session_turns.create(session_id, config={})
        engine = state._stores[PersistenceDomain.CONVERSATION].context.engine.sync_engine
        reads = []

        def capture(_connection, _cursor, sql, parameters, _context, _many):
            if sql.startswith("SELECT") and "ai_sessions" in sql:
                reads.append((sql, parameters))

        event.listen(engine, "before_cursor_execute", capture)
        try:
            async with History.open(persistence=state, authorization=Authorization()) as history:
                first = await history.turn_sessions(principal=alice, limit=1)
                assert [item["session_id"] for item in first["items"]] == ["共享/a/aaa"]
                assert first["next_cursor"]
                sql, params = reads.pop()
                assert "session_key >= " in sql and "session_key < " in sql
                assert "共享/a/".encode() in params and "共享/a0".encode() in params
                assert not reads
                second = await history.turn_sessions(principal=alice, cursor=first["next_cursor"], limit=1)
                assert [item["session_id"] for item in second["items"]] == ["共享/a/zzz"]
                assert second["next_cursor"] is None
                assert [item["session_id"] for item in (await history.turn_sessions(principal=bob))["items"]] == [
                    "共享/b/only",
                ]
                reads.clear()
                with pytest.raises(AIError) as denied:
                    await history.turn_sessions(principal=Principal("unknown", "tenant"))
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                with pytest.raises(AIError) as foreign_cursor:
                    await history.turn_sessions(principal=bob, cursor=first["next_cursor"])
                assert foreign_cursor.value.code is ErrorCode.CURSOR_INVALID
                prefixes["alice"] = "另/"
                with pytest.raises(AIError) as outdated_cursor:
                    await history.turn_sessions(principal=alice, cursor=first["next_cursor"])
                assert outdated_cursor.value.code is ErrorCode.CURSOR_INVALID
                assert not reads
            async with History.open(persistence=state, authorization=ExplicitOnly()) as history:
                with pytest.raises(AIError) as no_scope:
                    await history.turn_sessions(principal=alice)
                assert no_scope.value.code is ErrorCode.AUTHORIZATION_DENIED
                assert not reads
                explicit = await history.turn_sessions(principal=alice, session_ids=("共享/a/zzz",))
                assert [item["session_id"] for item in explicit["items"]] == ["共享/a/zzz"]
        finally:
            event.remove(engine, "before_cursor_execute", capture)
