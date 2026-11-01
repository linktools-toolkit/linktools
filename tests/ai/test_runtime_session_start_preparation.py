#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-call Session preparation retains authorization and source admission CAS."""

import asyncio
from contextvars import ContextVar
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event

from linktools.ai.core import AuthorizationAction, HmacCursorSigner, Principal, SessionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._session import RuntimeSessionService
from linktools.ai.session import ResumeSessionRequest, SessionManager

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
        await transaction.delete_records(tuple(r.key_digest for r in records if r.kind in {"session", "conversation_history", "transcript_head"}))
    await state.conversation.sessions.state_store.mutate(remove)


@pytest.mark.parametrize("entry", ("loaded", "lazy", "direct"))
@pytest.mark.parametrize("method", ("start", "run", "plan"))
async def test_session_start_preparation_reads_one_authorized_snapshot_per_call(tmp_path, monkeypatch, entry, method):
    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.attach(persistence=state, features=_features(_answer), metrics=None) as runtime:
            await runtime.agents.get().create_session("session")
            engine = state.conversation.sessions.state_store.storage_group.context.engine
            active = ContextVar("session_authorized_read", default=False)
            selects = []
            original = SessionManager.read_authorized

            async def read_authorized(self, *args, **kwargs):
                token = active.set(True)
                try:
                    return await original(self, *args, **kwargs)
                finally:
                    active.reset(token)

            def capture(_connection, _cursor, statement, *_args):
                if active.get() and statement.lstrip().upper().startswith("SELECT"):
                    selects.append(statement)

            monkeypatch.setattr(SessionManager, "read_authorized", read_authorized)
            event.listen(engine.sync_engine, "before_cursor_execute", capture)
            try:
                if entry == "loaded":
                    target = await runtime.sessions.get("session", principal=runtime.default_principal)
                elif entry == "lazy":
                    target = runtime.agents.get().session("session")
                else:
                    target = runtime.agents.get()
                kwargs = {"idempotency_key": "request"}
                if entry == "direct":
                    kwargs["session_id"] = "session"
                result = await getattr(target, method)("question", **kwargs)
                if method == "start":
                    result = await result.wait(timeout_seconds=5)
                assert result.result.output == {"text": "answer"}
                assert len(selects) == (4 if entry == "loaded" else 2)
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", capture)


@pytest.mark.parametrize("entry", ("missing-lazy", "deleted-loaded"))
async def test_session_start_preserves_create_and_reauthorizes_created_record(tmp_path, monkeypatch, entry):
    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.attach(persistence=state, features=_features(_answer), metrics=None) as runtime:
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


@pytest.mark.parametrize("change", ("owner-before-start", "owner-in-policy", "closed", "agent"))
async def test_session_start_rechecks_live_authority_status_and_binding(tmp_path, change):
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
            if armed and change == "owner-in-policy" and action is AuthorizationAction.SESSION_READ:
                armed = False
                await update(owner_principal_id="other")

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.attach(persistence=state, features=_features(model), metrics=None, authorization=Policy()) as runtime:
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
            expected = {"closed": ErrorCode.SESSION_CONFLICT, "agent": ErrorCode.SESSION_BINDING_MISMATCH}
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


async def test_two_runtimes_with_same_prepared_session_keep_source_admission_cas(tmp_path, monkeypatch):
    arrived = 0
    prepared = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    original = RuntimeSessionService._prepare_resume_session

    async def read_prepared(self, *args, **kwargs):
        nonlocal arrived
        record = await original(self, *args, **kwargs)
        arrived += 1
        if arrived == 2:
            prepared.set()
        await asyncio.wait_for(prepared.wait(), 5)
        return record

    async def model(messages, info):
        nonlocal calls
        calls += 1
        await release.wait()
        return await _answer(messages, info)

    path = tmp_path / "shared.db"
    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(path)) as first_state:
        async with Runtime.attach(persistence=first_state, features=_features(model), metrics=None) as first:
            await first.agents.get().create_session("session")
            async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(path)) as second_state:
                async with Runtime.attach(persistence=second_state, features=_features(model), metrics=None) as second:
                    monkeypatch.setattr(RuntimeSessionService, "_prepare_resume_session", read_prepared)
                    try:
                        outcomes = await asyncio.wait_for(asyncio.gather(
                            first.agents.get().session("session").start("first", idempotency_key="first"),
                            second.agents.get().session("session").start("second", idempotency_key="second"),
                            return_exceptions=True), 10)
                        errors = [result for result in outcomes if isinstance(result, BaseException)]
                        assert len(errors) == 1 and isinstance(errors[0], AIError)
                        assert errors[0].code is ErrorCode.SESSION_BUSY
                    finally:
                        release.set()
                    winner = next(result for result in outcomes if not isinstance(result, BaseException))
                    assert (await winner.wait(timeout_seconds=5)).result.output == {"text": "answer"}
                    assert calls == 1


async def test_created_session_view_cannot_replace_fresh_read_authorization(tmp_path, monkeypatch):
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return await _answer(messages, info)

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.attach(persistence=state, features=_features(model), metrics=None) as runtime:
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


async def test_execution_authorization_failure_is_not_session_creation_retry(tmp_path, monkeypatch):
    class Policy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if action is AuthorizationAction.EXECUTION_RUN:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.attach(persistence=state, features=_features(_answer), metrics=None, authorization=Policy()) as runtime:
            target = await runtime.agents.get().create_session("session")

            async def unexpected_create(*args, **kwargs):
                pytest.fail("execution authorization must not retry session creation")

            monkeypatch.setattr(RuntimeSessionService, "create", unexpected_create)
            with pytest.raises(AIError) as denied:
                await target.start("question", idempotency_key="request")
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            record = await state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert record.active_execution_id is None


async def test_prepared_open_session_does_not_override_fresh_source_closed_status(tmp_path, monkeypatch):
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return await _answer(messages, info)

    async with Persistence.open("session-start-preparation", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        async with Runtime.attach(persistence=state, features=_features(model), metrics=None) as runtime:
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
