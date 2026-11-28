#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session occupancy consumes membership, not executable content."""

import asyncio
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.core import ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

from .test_session_timeline import _execution, _session


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ('active', 'terminal', 'missing', 'other-session', 'child', 'foreign-tenant'))
async def test_inspection_keeps_admission_validation_without_payload_reads(monkeypatch, case):
    storage = PersistenceStorage.in_memory()
    async with Persistence.open('inspect', tenant_id='tenant', storage=storage) as state:
        execution = _execution('active', status=ExecutionStatus.FAILED if case == 'terminal' else ExecutionStatus.STARTED, prompt='private body')
        if case == 'other-session':
            execution = replace(execution, session_id='other')
        elif case == 'child':
            # Simulate malformed membership independently of authoring lineage constraints.
            object.__setattr__(execution, 'parent_execution_id', 'parent')
        if case != 'missing':
            await state.execution.executions.create(execution)
        await state.conversation.sessions.create(replace(_session(), active_execution_id='active'))

        async def no_details(*args, **kwargs):
            pytest.fail('occupancy must not restore executable content')

        monkeypatch.setattr(ExecutionRepositoryImpl, 'get', no_details)
        monkeypatch.setattr(ExecutionContent, 'expand_policies', no_details)
        monkeypatch.setattr(ExecutionContent, 'read_payloads', no_details)
        async with History.open(persistence=state) as history:
            principal = Principal('owner', 'other' if case == 'foreign-tenant' else 'tenant')
            if case in {'missing', 'other-session', 'child', 'foreign-tenant'}:
                with pytest.raises(AIError) as caught:
                    await history.inspect_session('session', principal=principal)
                assert caught.value.code is (ErrorCode.AUTHORIZATION_DENIED if case == 'foreign-tenant' else ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                view = await history.inspect_session('session', principal=principal)
                assert view.active_execution_id == (None if case == 'terminal' else 'active')



class _HostAuthorization(TenantAuthorizationPolicy):
    """Host authorization, with the Runtime's tenant boundary intact."""


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("active", "terminal", "missing", "other-session", "standalone", "foreign-tenant"))
async def test_native_inspection_reads_authoritative_occupancy_without_loading_turn_content(tmp_path, monkeypatch, case):
    namespace = "native-inspect"
    principal = Principal("owner", "tenant")
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(_messages, _info):
        calls.append("provider")
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("public answer")])

    features = RuntimeFeatures(namespace)
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "inspect.db")
    async with Runtime.open(namespace, storage=storage, features=features, tenant_id="tenant",
                            authorization=_HostAuthorization()) as runtime:
        sessions = runtime.sessions
        await sessions.create_turn_session("session", principal=principal)
        turns = runtime._state.session_turns
        if case in {"active", "terminal"}:
            turn = await sessions.start_turn("session", "private input", agent="default", principal=principal,
                                        idempotency_key="first")
            await asyncio.wait_for(entered.wait(), 5)
            expected_id = turn.key.turn_id
            if case == "terminal":
                release.set()
                assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
                expected_id = None
        elif case in {"missing", "other-session", "standalone"}:
            if case == "other-session":
                release.set()
                await sessions.create_turn_session("other", principal=principal)
                turn = await sessions.start_turn("other", "private input", agent="default", principal=principal,
                                            idempotency_key="other")
                assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
                expected_id = turn.key.turn_id
            elif case == "standalone":
                release.set()
                run = await runtime.agents.get().start("private input", principal=principal)
                assert (await run.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
                expected_id = run.execution_id
            else:
                expected_id = "missing-turn"
            async with turns._context.engine.begin() as connection:
                result = await connection.execute(turns._sessions.update().filter_by(
                    **turns._session_values("session"),
                ).values(active_turn_id=expected_id.encode()))
                assert result.rowcount == 1
        else:
            expected_id = None
        state = runtime._state
        reader = state.session_turn_reader
        statements = []

        def sql(_connection, _cursor, statement, _parameters, _context, _many):
            statements.append(statement)

        async def no_details(*args, **kwargs):
            pytest.fail("native occupancy cannot read executable content")

        try:
            with monkeypatch.context() as patch:
                patch.setattr(ExecutionRepositoryImpl, "get", no_details)
                patch.setattr(ExecutionContent, "expand_policies", no_details)
                patch.setattr(ExecutionContent, "read_payloads", no_details)
                patch.setattr(reader, "get_turn", no_details)
                patch.setattr(reader, "get_history_turn", no_details)
                patch.setattr(reader, "read_scalar_input", no_details)
                engine = reader._context.engine.sync_engine
                event.listen(engine, "before_cursor_execute", sql)
                try:
                    if case == "foreign-tenant":
                        with pytest.raises(AIError) as denied:
                            await sessions.inspect("session", principal=Principal("owner", "other"))
                        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                    else:
                        observed = await sessions.inspect("session", principal=principal)
                        assert observed["active_turn_id"] == expected_id
                finally:
                    event.remove(engine, "before_cursor_execute", sql)
            selects = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
            if case == "foreign-tenant":
                assert selects == []
            else:
                [select] = selects
                assert "ai_sessions" in select and "ai_session_turns" not in select
            if case in {"missing", "other-session", "standalone"}:
                # A damaged owner is still occupied. SQL's authoritative head
                # must never let a different turn claim the session.
                before = len(calls)
                with pytest.raises(AIError) as busy:
                    await sessions.start_turn("session", "do not dispatch", principal=principal, agent="default",
                                         idempotency_key="competing")
                assert busy.value.code is ErrorCode.SESSION_BUSY and len(calls) == before
                with pytest.raises(AIError) as missing:
                    await runtime.history.turn_result("session", expected_id, principal=principal)
                assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
        finally:
            release.set()
