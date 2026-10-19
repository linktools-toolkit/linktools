#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Display facts have bounded freshness; authorization decisions never do."""

from types import SimpleNamespace

import pytest
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeStorage
from linktools.ai.storage import InMemoryCacheBackend, _write_cache
import linktools.ai.runtime._session as session_module

from .test_session_display_cache import _PRINCIPAL, _change_metadata, _prepare, _service

pytestmark = pytest.mark.asyncio


@pytest.fixture
def clock(monkeypatch):
    value = SimpleNamespace(now=100.0)
    monkeypatch.setattr(session_module, "time", SimpleNamespace(time=lambda: value.now))
    monkeypatch.setattr(_write_cache, "time", lambda: value.now)
    return value


async def test_cached_display_facts_skip_sql_but_recheck_policy_and_local_writes(tmp_path, clock):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    state = RuntimeStorage.sqlite(tmp_path / "state.db", cache=cache)
    await _prepare(state, body=True)
    allowed = True
    calls = 0

    class Policy:
        async def authorize(self, principal, action, resource):
            nonlocal calls
            calls += 1
            if not allowed:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    engine = state.conversation.sessions.state_store.storage_group.context.engine
    statements = []

    def capture(_connection, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        service = _service(state, cache, Policy())
        detail = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        statements.clear()
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == detail
        assert statements == []
        assert calls == 3
        allowed = False
        with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert calls == 4 and statements == []
        allowed = True
        await _change_metadata(state, "changed")
        fresh = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert fresh.session.revision > detail.session.revision
        assert fresh.session.metadata == {"value": "changed"}
        statements.clear()
        await service.refresh("session", principal=_PRINCIPAL, max_age=0)
        assert statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await state.close()


async def test_other_owner_membership_deletion_is_bounded_by_original_three_seconds(tmp_path, clock):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    path = tmp_path / "state.db"
    state = RuntimeStorage.sqlite(path, cache=cache)
    await _prepare(state, body=True)
    other = RuntimeStorage.sqlite(path)
    await other.initialize(namespace="display", tenant_id="tenant")
    try:
        service = _service(state, cache)
        detail = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)

        async def remove(transaction):
            records = await transaction.scan_records()
            await transaction.delete_records(tuple(
                record.key_digest for record in records if record.kind == "session"
            ))

        await other.conversation.sessions.state_store.mutate(remove)
        clock.now += 2.999
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == detail
        clock.now += .002
        with pytest.raises(AIError, match="AUTHORIZATION_DENIED"):
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
    finally:
        await other.close()
        await state.close()


async def test_newer_whole_display_survives_older_cached_session_revision(tmp_path, clock):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    path = tmp_path / "state.db"
    state = RuntimeStorage.sqlite(path, cache=cache)
    await _prepare(state, body=True)
    other = RuntimeStorage.sqlite(path)
    await other.initialize(namespace="display", tenant_id="tenant")
    engine = state.conversation.sessions.state_store.storage_group.context.engine
    statements = []

    def capture(_connection, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        service = _service(state, cache)
        old = await service.get("session", principal=_PRINCIPAL)
        await _change_metadata(other, "newer")
        fresh = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert fresh.session.revision > old.revision
        assert fresh.session.metadata == {"value": "newer"}
        await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        statements.clear()
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == fresh
        assert statements == []
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await other.close()
        await state.close()


async def test_cached_execution_absence_does_not_reject_new_source_display(tmp_path, clock):
    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    path = tmp_path / "state.db"
    state = RuntimeStorage.sqlite(path, cache=cache)
    await state.initialize(
            namespace="display", tenant_id="tenant", module_caches={RuntimeDomain.CONVERSATION: cache},
        )
    other = RuntimeStorage.sqlite(path)
    try:
        assert await state.execution.executions.get_many(("active",), tenant_id="tenant") == {}
        await _prepare(other, body=True)
        service = _service(state, cache)
        detail = await service.refresh("session", principal=_PRINCIPAL, max_age=3)
        assert [turn.execution_id for turn in detail.timeline.items] == ["active"]
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == detail

        async def remove(transaction):
            records = await transaction.scan_records()
            await transaction.delete_records(tuple(
                record.key_digest for record in records if record.kind == "execution"
            ))

        await other.execution.executions.state_store.mutate(remove)
        clock.now += 3.001
        assert await service.refresh("session", principal=_PRINCIPAL, max_age=3) == detail

        async def remove_retained_turn(transaction):
            records = await transaction.scan_records()
            await transaction.delete_records(tuple(
                record.key_digest for record in records if record.kind == "session_turn_commit"
            ))

        await other.conversation.sessions.state_store.mutate(remove_retained_turn)
        clock.now += 3.001
        with pytest.raises(AIError, match="STORAGE_INTEGRITY_ERROR"):
            await service.refresh("session", principal=_PRINCIPAL, max_age=3)
    finally:
        await other.close()
        await state.close()


@pytest.mark.parametrize("limit", (1, 2, 10))
@pytest.mark.parametrize("with_history", (False, True))
async def test_cached_canonical_page_projects_live_pending_without_freezing_progress(
    tmp_path, clock, limit, with_history,
):
    from dataclasses import replace
    from linktools.ai.core import ExecutionStatus
    from .test_session_timeline import _execution, _session

    cache = InMemoryCacheBackend(max_bytes=1_000_000)
    state = RuntimeStorage.sqlite(tmp_path / "state.db", cache=cache)
    if with_history:
        await _prepare(state, body=True)
    else:
        await state.initialize(
            namespace="display", tenant_id="tenant", module_caches={RuntimeDomain.CONVERSATION: cache},
        )
        await state.conversation.sessions.create(_session())
    pending = _execution("pending", status=ExecutionStatus.STARTED, prompt="live question")
    service = _service(state, cache)

    async def observe(session_id, *, tenant_id):
        return pending

    service._pending_session_execution = observe
    engine = state.conversation.sessions.state_store.storage_group.context.engine
    statements = []

    def capture(_connection, _cursor, statement, *_args):
        if statement.lstrip().upper().startswith("SELECT"):
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", capture)
    try:
        detail = await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=limit)
        assert detail.timeline.items[-1].execution_id == "pending"
        await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=limit)
        pending = replace(pending, status=ExecutionStatus.CANCELLING, safe_error_details={"progress": "new"})
        statements.clear()
        changed = await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=limit)
        assert statements == []
        assert changed.timeline.items[-1].status is ExecutionStatus.CANCELLING
        assert changed.timeline.items[-1].safe_error_details == {"progress": "new"}
        assert len(changed.timeline.items) == min(limit, 2 if with_history else 1)
        if changed.timeline.next_cursor is not None:
            older = await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=limit,
                                          cursor=changed.timeline.next_cursor)
            assert [turn.execution_id for turn in older.timeline.items] == ["active"]
        pending = replace(pending, execution_id="replacement")
        replacement = await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=limit)
        assert replacement.timeline.items[-1].execution_id == "replacement"
        pending = None
        settled = await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=limit)
        assert settled.session.active_execution_id is None
        assert [turn.execution_id for turn in settled.timeline.items] == (["active"] if with_history else [])
        with pytest.raises(AIError):
            await service.refresh("session", principal=_PRINCIPAL, max_age=3, limit=0)
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await state.close()


async def test_pending_to_terminal_history_repairs_old_head_and_then_reads_without_sql(tmp_path):
    from .test_pending_session_projection import _runtimes

    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache) as (runtimes, entered, finish, block):
        owner, _ = runtimes[0]
        reader, state = runtimes[1]
        session = await owner.agents.get().create_session("session")
        block()
        execution = await session.start("pending question")
        try:
            await entered.wait()
            for _ in range(3):
                pending = await reader.sessions.refresh("session", principal=reader.default_principal, max_age=3)
                assert pending.session.active_execution_id == execution.execution_id
                assert not pending.timeline.items[-1].conversation_committed
            finish.set()
            await execution.wait(timeout_seconds=5)
            await reader.executions.result(execution.execution_id, principal=reader.default_principal)
            engine = state.conversation.sessions.state_store.storage_group.context.engine
            statements = []

            def capture(_connection, _cursor, statement, *_args):
                if statement.lstrip().upper().startswith("SELECT"):
                    statements.append(statement)

            event.listen(engine.sync_engine, "before_cursor_execute", capture)
            try:
                terminal = await reader.sessions.refresh("session", principal=reader.default_principal, max_age=3)
                assert terminal.session.active_execution_id is None
                assert terminal.timeline.items[-1].conversation_committed
                # The first validation may replace older source membership facts.
                await reader.sessions.refresh("session", principal=reader.default_principal, max_age=3)
                statements.clear()
                assert await reader.sessions.refresh("session", principal=reader.default_principal, max_age=3) == terminal
                assert statements == []
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", capture)
        finally:
            finish.set()
