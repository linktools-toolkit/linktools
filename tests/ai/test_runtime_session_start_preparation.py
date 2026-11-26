#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-call Session preparation retains authorization and source admission CAS."""

import asyncio
from contextvars import ContextVar
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import AuthorizationAction, CommitPolicy, HmacCursorSigner, Principal, SessionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._session import RuntimeSessionService
from linktools.ai.session import ResumeSessionRequest, SessionManager
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

pytestmark = pytest.mark.asyncio


def _features(model):
    features = RuntimeFeatures("session-start-preparation")
    features.agent("default", model="default", allow_tools=())
    return model_features(_Models(_UsageFunctionModel(model)), features)


async def _answer(messages, info):
    return ModelResponse(parts=[TextPart("answer")])


async def _delete_session(state):
    async def remove(transaction):
        records = await transaction.scan_records()
        await transaction.delete_records(tuple(r.key for r in records if r.kind in {"session", "conversation_history", "transcript_head"}))
    await state.conversation.sessions.state_store.mutate(remove)


@pytest.mark.parametrize("entry", ("loaded", "lazy", "direct"))
async def test_sql_session_start_reads_one_authorized_session_row(tmp_path, entry):
    class HostAuthorization(TenantAuthorizationPolicy):
        """Authorize native session access explicitly."""

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.from_root(tmp_path)) as state:
        async with Runtime.open(
            persistence=state, features=_features(_answer), metrics=None, authorization=HostAuthorization(),
        ) as runtime:
            principal = runtime.default_principal
            agent = runtime.agents.get()
            await agent.create_session("session")
            if entry == "loaded":
                session = await runtime.sessions.get("session", principal=principal)
            engine = state.conversation.sessions.state_store.storage_group.context.engine
            selects = []
            admission_task = asyncio.current_task()

            def capture(_connection, _cursor, statement, *_args):
                sql = statement.lstrip().upper()
                if asyncio.current_task() is admission_task and sql.startswith("SELECT") and "AI_SESSIONS" in sql:
                    selects.append(statement)

            event.listen(engine.sync_engine, "before_cursor_execute", capture)
            try:
                if entry == "loaded":
                    async with session.start("question", agent=agent, idempotency_key="request") as (turn, _events):
                        assert len(selects) == 1
                elif entry == "lazy":
                    turn = await agent.session("session").start("question", idempotency_key="request")
                    assert len(selects) == 1
                else:
                    turn = await agent.start("question", session_id="session", idempotency_key="request")
                    assert len(selects) == 1
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", capture)
            result = await turn.wait_final(timeout_seconds=5)
            assert result.view["output"] == {"text": "answer"}


@pytest.mark.parametrize("bound_agent", (False, True), ids=("agent-id", "bound-agent"))
async def test_canonical_sql_planning_reads_one_authorized_snapshot(tmp_path, bound_agent):
    from linktools.ai.persistence.turns import SessionTurnKey
    from linktools.ai.runtime.turn_clients import TurnObserver

    reads = []
    selects = []
    active = ContextVar("canonical_planning_admission", default=False)

    class Policy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if active.get() and action is AuthorizationAction.SESSION_READ:
                reads.append((principal, resource.id))

    def capture(_connection, _cursor, statement, *_args):
        sql = statement.lstrip().upper()
        if active.get() and sql.startswith("SELECT") and "AI_SESSIONS" in sql:
            selects.append(statement)

    storage = PersistenceStorage.from_root(tmp_path)
    async with Persistence.open("session-start-preparation", storage=storage) as state:
        async with Runtime.open(
            persistence=state, features=_features(_answer), metrics=None, authorization=Policy(),
        ) as runtime:
            principal = runtime.default_principal
            await runtime.sessions.create_turn_session("session", principal=principal)
            agent = runtime.agents.get() if bound_agent else "default"
            engine = state.conversation.sessions.state_store.storage_group.context.engine
            event.listen(engine.sync_engine, "before_cursor_execute", capture)
            try:
                with TurnObserver(runtime, "session", principal, local_workspace=True) as observer:
                    token = active.set(True)
                    try:
                        turn_id = await runtime.sessions.start_turn(
                            "session", "question", agent=agent, principal=principal,
                            idempotency_key="request", planning=True,
                        )
                    finally:
                        active.reset(token)
                    async with asyncio.timeout(5):
                        async for frame in observer.events(turn_id):
                            if frame["type"] == "terminal":
                                break
                assert frame["type"] == "terminal"
                assert frame["turn_id"] == turn_id and frame["state"] == "succeeded"
                assert frame["output"] == {"text": "answer"}
                assert reads == [(principal, "session")]
                assert len(selects) == 1
                row = await state.session_turns.get_turn(SessionTurnKey("session", turn_id))
                assert row["request_json"]["planning"] is True
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.parametrize("entry", ("loaded", "lazy", "direct"))
async def test_non_sql_bound_session_planning_retains_execution_result(entry):
    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=_features(_answer), metrics=None) as runtime:
            await runtime.agents.get().create_session("session")
            kwargs = {"idempotency_key": "request"}
            if entry == "loaded":
                target = await runtime.sessions.get("session", principal=runtime.default_principal)
            elif entry == "lazy":
                target = runtime.agents.get().session("session")
            else:
                target = runtime.agents.get()
                kwargs["session_id"] = "session"
            result = await target.plan("question", **kwargs)
            assert result.result.output == {"text": "answer"}


@pytest.mark.parametrize("entry", ("missing-lazy", "deleted-loaded"))
async def test_session_start_preserves_create_and_reauthorizes_created_record(monkeypatch, entry):
    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=_features(_answer), metrics=None) as runtime:
            if entry == "deleted-loaded":
                await runtime.agents.get().create_session("session")
                target = await runtime.sessions.get("session", principal=runtime.default_principal)
                await _delete_session(state)
            else:
                target = runtime.agents.get().session("session")
            original = SessionManager.read_authorized
            reads = []

            async def read_authorized(self, *args, **kwargs):
                result = await original(self, *args, **kwargs)
                reads.append(result)
                return result

            monkeypatch.setattr(SessionManager, "read_authorized", read_authorized)
            execution = await target.start("question", idempotency_key="request")
            assert (await execution.wait(timeout_seconds=5)).result.output == {"text": "answer"}
            assert len(reads) == 1 and reads[0].session_id == "session"
            assert reads[0].owner_principal_id == runtime.default_principal.principal_id


@pytest.mark.parametrize("change", ("owner-before-start", "owner-in-policy", "owner-in-run-policy", "closed", "agent", "agent-in-policy", "cwd-in-policy"))
@pytest.mark.parametrize("terminal", (False, True))
async def test_session_start_rechecks_live_authority_status_and_binding(tmp_path, change, terminal):
    calls = 0
    armed = False

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return await _answer(messages, info)

    class Policy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            nonlocal armed
            await super().authorize(principal, action, resource)
            if armed and (
                change in {"owner-in-policy", "agent-in-policy", "cwd-in-policy"} and action is AuthorizationAction.SESSION_READ
                or change == "owner-in-run-policy" and action is AuthorizationAction.EXECUTION_RUN
            ):
                armed = False
                changes = {"owner_principal_id": "other"}
                if change == "agent-in-policy":
                    changes = {"agent_id": "other"}
                elif change == "cwd-in-policy":
                    changes = {"cwd": "changed"}
                await update(**changes)

    features = _features(model)
    if terminal:
        features.capability(ExecutionCapability(retain_history=True,
            commit=CommitPolicy.terminal(buffer=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))), scope="runtime")
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024) if terminal else None)
    async with Persistence.open("session-start-preparation", storage=storage) as state:
        async with Runtime.open(persistence=state, features=features, metrics=None, authorization=Policy()) as runtime:
            await runtime.agents.get().create_session("session")
            target = await runtime.sessions.get("session", principal=runtime.default_principal)

            async def update(**changes):
                record = await state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
                if "owner_principal_id" in changes or "agent_id" in changes:
                    # Session identity is immutable: model a real deletion and
                    # recreation, not an invalid repository identity update.
                    await _delete_session(state)
                    await state.conversation.sessions.create(replace(record, **changes))
                else:
                    await state.conversation.sessions.compare_and_swap("session", tenant_id=runtime.tenant_id,
                        expected_revision=record.revision, next_record=replace(record, revision=record.revision + 1, **changes))

            if change == "owner-before-start":
                await update(owner_principal_id="other")
            elif change == "closed":
                await update(status=SessionStatus.CLOSED)
            elif change == "agent":
                await update(agent_id="other")
            armed = True
            with pytest.raises(AIError) as denied:
                await target.start("question", idempotency_key="request")
            expected = {"closed": ErrorCode.SESSION_CONFLICT, "agent": ErrorCode.SESSION_BINDING_MISMATCH,
                        "agent-in-policy": ErrorCode.SESSION_BINDING_MISMATCH, "cwd-in-policy": ErrorCode.SESSION_CONFLICT}
            assert denied.value.code is expected.get(change, ErrorCode.AUTHORIZATION_DENIED)
            assert calls == 0


async def test_generic_session_resume_still_requires_existing_record():
    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.in_memory()) as state:
        manager = SessionManager(state.conversation, state.execution.executions, TenantAuthorizationPolicy(),
            object(), HmacCursorSigner("session", b"session-key"))
        with pytest.raises(AIError) as missing:
            await manager.resume("default", "binding", "missing", ResumeSessionRequest(Principal("owner", state.tenant_id), "question", "request", None, "run", False, False))
        assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert await state.conversation.sessions.get("missing", tenant_id=state.tenant_id) is None


async def test_two_runtimes_with_same_native_snapshot_admit_only_one_turn(tmp_path):
    arrived = 0
    both_authorized = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    class Policy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            nonlocal arrived
            await super().authorize(principal, action, resource)
            if action is AuthorizationAction.EXECUTION_RUN:
                arrived += 1
                if arrived == 2:
                    both_authorized.set()
                await asyncio.wait_for(both_authorized.wait(), 5)

    async def model(messages, info):
        nonlocal calls
        calls += 1
        await release.wait()
        return await _answer(messages, info)

    storage = PersistenceStorage.from_root(tmp_path / "shared")
    async with Runtime.open(
        "session-start-preparation", storage=storage, features=_features(model), metrics=None, authorization=Policy(),
    ) as first:
        await first.agents.get().create_session("session")
        async with Runtime.open(
            "session-start-preparation", storage=storage, features=_features(model), metrics=None, authorization=Policy(),
        ) as second:
            try:
                outcomes = await asyncio.wait_for(asyncio.gather(
                    first.agents.get().session("session").start("first", idempotency_key="first"),
                    second.agents.get().session("session").start("second", idempotency_key="second"),
                    return_exceptions=True,
                ), 10)
                errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
                assert len(errors) == 1 and isinstance(errors[0], AIError)
                assert errors[0].code is ErrorCode.SESSION_CONFLICT
            finally:
                release.set()
            winner = next(outcome for outcome in outcomes if not isinstance(outcome, BaseException))
            assert (await winner.wait_final(timeout_seconds=5)).view["output"] == {"text": "answer"}
            assert calls == 1


async def test_created_session_view_cannot_replace_fresh_read_authorization(monkeypatch):
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return await _answer(messages, info)

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=_features(model), metrics=None) as runtime:
            original = RuntimeSessionService.create

            async def create_then_replace(self, *args, **kwargs):
                view = await original(self, *args, **kwargs)
                record = await state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
                await _delete_session(state)
                await state.conversation.sessions.create(replace(record, owner_principal_id="other"))
                return view

            monkeypatch.setattr(RuntimeSessionService, "create", create_then_replace)
            with pytest.raises(AIError) as denied:
                await runtime.agents.get().session("session").start("question", idempotency_key="request")
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert calls == 0


async def test_execution_authorization_failure_is_not_session_creation_retry(monkeypatch):
    class Policy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if action is AuthorizationAction.EXECUTION_RUN:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=_features(_answer), metrics=None, authorization=Policy()) as runtime:
            target = await runtime.agents.get().create_session("session")

            async def unexpected_create(*args, **kwargs):
                pytest.fail("execution authorization must not retry session creation")

            monkeypatch.setattr(RuntimeSessionService, "create", unexpected_create)
            with pytest.raises(AIError) as denied:
                await target.start("question", idempotency_key="request")
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            record = await state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert record.active_execution_id is None


async def test_prepared_open_session_does_not_override_fresh_source_closed_status(monkeypatch):
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return await _answer(messages, info)

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=_features(model), metrics=None) as runtime:
            target = await runtime.agents.get().create_session("session")
            original = RuntimeSessionService._prepare_resume_session

            async def read_then_close(self, *args, **kwargs):
                record = await original(self, *args, **kwargs)
                await state.conversation.sessions.compare_and_swap("session", tenant_id=runtime.tenant_id,
                    expected_revision=record.revision,
                    next_record=replace(record, revision=record.revision + 1, status=SessionStatus.CLOSED))
                return record

            monkeypatch.setattr(RuntimeSessionService, "_prepare_resume_session", read_then_close)
            with pytest.raises(AIError) as closed:
                await target.start("question", idempotency_key="request")
            assert closed.value.code is ErrorCode.SESSION_CONFLICT
            assert calls == 0
            record = await state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert record.status is SessionStatus.CLOSED and record.active_execution_id is None


async def test_session_start_reuses_authorized_source_and_commit_receipt(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        return await _answer(messages, info)

    async with Persistence.open("session-start-receipt", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=_features(model), metrics=None) as runtime:
            target = await runtime.agents.get().create_session("session")
            service = runtime._execution_service
            load_session = service._load_session
            load_execution = service.load_source_execution
            launch = service._launch_started_impl
            launching = ContextVar("launch_source_read", default=False)
            session_reads = []
            launch_reads = []

            async def read_session(*args, **kwargs):
                session_reads.append(args[0])
                return await load_session(*args, **kwargs)

            async def read_execution(*args, **kwargs):
                if launching.get():
                    launch_reads.append(args[0])
                return await load_execution(*args, **kwargs)

            async def launch_started(*args, **kwargs):
                token = launching.set(True)
                try:
                    return await launch(*args, **kwargs)
                finally:
                    launching.reset(token)

            monkeypatch.setattr(service, "_load_session", read_session)
            monkeypatch.setattr(service, "load_source_execution", read_execution)
            monkeypatch.setattr(service, "_launch_started_impl", launch_started)
            execution = await target.start("question", idempotency_key="request")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert session_reads == []
                assert launch_reads == []
                replay = await target.start("question", idempotency_key="request")
                assert replay.execution_id == execution.execution_id
                assert launch_reads == [execution.execution_id]
            finally:
                release.set()
            assert (await execution.wait(timeout_seconds=5)).result.output == {"text": "answer"}


async def test_reconciled_unknown_session_refreshes_source_before_start(monkeypatch):
    async with Persistence.open("session-start-reconciled", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=_features(_answer), metrics=None) as runtime:
            target = await runtime.agents.get().create_session("session")
            service = runtime._execution_service
            load_session = service._load_session
            source_reads = []

            async def settle(session_id, *, tenant_id):
                record = await state.conversation.sessions.get(session_id, tenant_id=tenant_id)
                await _delete_session(state)
                await state.conversation.sessions.create(replace(record, owner_principal_id="replacement"))
                return True

            async def read_session(*args, **kwargs):
                source_reads.append(args[0])
                return await load_session(*args, **kwargs)

            monkeypatch.setattr(service._runtime_bridge.pending_turns, "settle_session", settle)
            monkeypatch.setattr(service, "_load_session", read_session)
            with pytest.raises(AIError) as denied:
                await target.start("question", idempotency_key="request")
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert source_reads == ["session"]


async def test_session_start_authorizes_source_owner_after_remote_cached_replacement(tmp_path, monkeypatch):
    from linktools.ai.storage import _write_cache

    monkeypatch.setattr(_write_cache, "time", lambda: 1000.0)
    decisions = []
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return await _answer(messages, info)

    class Policy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            if action is AuthorizationAction.SESSION_READ:
                decisions.append(resource.owner_principal_id)
            await super().authorize(principal, action, resource)

    from .test_session_admission import _session

    path = tmp_path / "state.db"
    storage = PersistenceStorage.sqlite(path, cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("session-start-source-owner", storage=PersistenceStorage.sqlite(path)) as writer:
        await writer.conversation.sessions.create(replace(_session(), owner_principal_id="runtime", agent_id="default"))
        async with Persistence.open("session-start-source-owner", storage=storage) as reader:
            async with Runtime.open(persistence=reader, features=_features(model), metrics=None, authorization=Policy()) as runtime:
                target = runtime.agents.get().session("session")
                original = await reader.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
                await _delete_session(writer)
                await writer.conversation.sessions.create(replace(original, owner_principal_id="replacement"))
                cached = await reader.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
                assert cached.owner_principal_id == original.owner_principal_id
                with pytest.raises(AIError) as denied:
                    await target.start("question", idempotency_key="request")
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                assert decisions and set(decisions) == {"replacement"}
                assert calls == 0


@pytest.mark.parametrize("settled", (False, True))
async def test_session_start_identity_capture_is_call_local_and_invalidated_by_settle(monkeypatch, settled):
    from linktools.ai.runtime._execution import DefaultExecutionService
    from linktools.ai.runtime._pending_turn import _PendingExecutionTurns

    captures = []
    refreshes = []
    original_capture = DefaultExecutionService._read_session_start
    original_refresh = DefaultExecutionService._load_session
    original_settle = _PendingExecutionTurns.settle_session

    async def read_capture(self, *args, **kwargs):
        capture = await original_capture(self, *args, **kwargs)
        captures.append(capture)
        return capture

    async def read_fresh_session(self, session_id, **kwargs):
        refreshes.append(session_id)
        return await original_refresh(self, session_id, **kwargs)

    async def settle(self, *args, **kwargs):
        result = await original_settle(self, *args, **kwargs)
        return settled or result

    monkeypatch.setattr(DefaultExecutionService, "_read_session_start", read_capture)
    monkeypatch.setattr(DefaultExecutionService, "_load_session", read_fresh_session)
    monkeypatch.setattr(_PendingExecutionTurns, "settle_session", settle)
    features = _features(_answer)
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.in_memory()) as state:
        async with Runtime.open(persistence=state, features=features, metrics=None) as runtime:
            target = await runtime.agents.get().create_session("session")
            handle = await target.start("question", idempotency_key="request")
            await handle.wait(timeout_seconds=5)
            assert len(captures) == 1 and captures[0].identity is None
            assert refreshes == (["session"] if settled else [])
            first_capture = captures.pop()
            refreshes.clear()
            replay = await target.start("question", idempotency_key="request")
            assert replay.execution_id == handle.execution_id
            assert len(captures) == 1 and captures[0] is not first_capture
            assert captures[0].identity is not None
            assert captures[0].identity.resource_id == handle.execution_id
            assert refreshes == (["session"] if settled else [])
