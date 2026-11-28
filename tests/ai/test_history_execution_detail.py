#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-bound execution observations retain authorization and result fences."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.core import AuthorizationAction, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import ExecutionDetail, History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend, StoredPayload

from .test_session_timeline import _commit_empty_turn, _execution, _session

pytestmark = pytest.mark.asyncio
_OWNER = Principal("owner", "tenant")
_TERMINAL = {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}


def _reader(state, *, authorization=None, executions=None, pending_execution=None):
    return History(
        SimpleNamespace(), tenant_id="tenant",
        sessions=state.conversation.sessions,
        executions=executions or state.execution.executions,
        authorization=authorization or TenantAuthorizationPolicy(),
        pending_execution=pending_execution,
    )


async def _seed(state, status=ExecutionStatus.SUCCEEDED, *, admitted=True, committed=True):
    await state.conversation.sessions.create(_session())
    error_code = (
        ErrorCode.EXECUTION_CANCELLED.value if status is ExecutionStatus.CANCELLED
        else ErrorCode.MODEL_API_ERROR.value if status in {
            ExecutionStatus.FAILED, ExecutionStatus.RECOVERY_REQUIRED,
        } else None
    )
    record = _execution("execution", status=status, prompt="question", error_code=error_code)
    record = replace(record, started_at=record.created_at)
    await state.execution.executions.create_execution(record)
    if admitted:
        await state.conversation.sessions.admit_execution(
            record.session_id, tenant_id="tenant",
            execution_id=record.execution_id, expected=None,
        )
    if committed and status in _TERMINAL:
        await _commit_empty_turn(state, record)
    return record


@pytest.mark.parametrize("status", (
    ExecutionStatus.STARTED, ExecutionStatus.RECOVERY_REQUIRED,
    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
))
async def test_execution_detail_projects_session_state_and_committed_result(
    status: ExecutionStatus,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        record = await _seed(state, status)
        reader = _reader(state)
        detail = await reader.execution_detail("execution", session_id="session", principal=_OWNER)
        assert isinstance(detail, ExecutionDetail)
        assert detail.execution.execution_id == record.execution_id
        assert detail.execution.session_id == "session"
        assert detail.execution.status is status
        assert detail.execution.usage is None
        if status in _TERMINAL:
            assert detail.result == await reader.result("execution", principal=_OWNER)
            assert detail.result.status is detail.execution.status
        else:
            assert detail.result is None
        if status is ExecutionStatus.RECOVERY_REQUIRED:
            assert detail.execution.error_code == ErrorCode.MODEL_API_ERROR.value
            with pytest.raises(AIError) as error:
                await reader.result("execution", principal=_OWNER)
            assert error.value.code is ErrorCode.MODEL_API_ERROR
    finally:
        await state.close()


@pytest.mark.parametrize("case", (
    "missing_session", "missing_execution", "different_session", "different_principal",
    "session_denied", "execution_denied",
))
async def test_execution_detail_requires_both_permissions_and_the_requested_binding(case: str) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        await _seed(state)
        await state.conversation.sessions.create(replace(_session(), session_id="other"))

        class Authorization(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                await super().authorize(principal, action, resource)
                if (case == "session_denied" and action is AuthorizationAction.SESSION_READ
                        or case == "execution_denied" and action is AuthorizationAction.EXECUTION_READ):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        with pytest.raises(AIError) as error:
            await _reader(state, authorization=Authorization()).execution_detail(
                "absent" if case == "missing_execution" else "execution",
                session_id=("absent" if case == "missing_session" else
                            "other" if case == "different_session" else "session"),
                principal=Principal("other", "tenant") if case == "different_principal" else _OWNER,
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


async def test_execution_detail_rechecks_session_ownership_after_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        await _seed(state)
        sessions = state.conversation.sessions
        get_session = sessions.get
        authorized = False

        async def get(session_id, *, tenant_id):
            session = await get_session(session_id, tenant_id=tenant_id)
            return replace(session, owner_principal_id="new-owner") if authorized else session

        class Authorization(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                nonlocal authorized
                await super().authorize(principal, action, resource)
                if action is AuthorizationAction.SESSION_READ:
                    authorized = True

        # Native session writes reject owner changes; model an inconsistent reader.
        with monkeypatch.context() as patch:
            patch.setattr(sessions, "get", get)
            with pytest.raises(AIError) as error:
                await _reader(state, authorization=Authorization()).execution_detail(
                    "execution", session_id="session", principal=_OWNER,
                )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


async def test_execution_detail_uses_the_post_authorization_terminal_record() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        original = await _seed(state, ExecutionStatus.STARTED)
        terminal = replace(
            original, status=ExecutionStatus.SUCCEEDED, revision=original.revision + 1,
            agent_run_seq=1,
            result=_execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="question").result,
        )

        class Authorization(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                await super().authorize(principal, action, resource)
                if action is AuthorizationAction.EXECUTION_READ:
                    await _commit_empty_turn(state, terminal)
                    await state.execution.executions.compare_and_swap(
                        "execution", tenant_id="tenant", expected_revision=original.revision,
                        next_record=terminal,
                    )

        detail = await _reader(state, authorization=Authorization()).execution_detail(
            "execution", session_id="session", principal=_OWNER,
        )
        assert detail.execution.status is ExecutionStatus.SUCCEEDED
        assert detail.result.status is ExecutionStatus.SUCCEEDED
        assert detail.result.output == {"text": "ok"}
    finally:
        await state.close()


@pytest.mark.parametrize("admitted, expected_error", (
    (False, ErrorCode.STORAGE_INTEGRITY_ERROR),
    (True, ErrorCode.STORAGE_RECOVERY_REQUIRED),
))
async def test_execution_detail_preserves_retained_turn_integrity(
    admitted: bool, expected_error: ErrorCode,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        await _seed(state, admitted=admitted, committed=False)
        with pytest.raises(AIError) as error:
            await _reader(state).execution_detail("execution", session_id="session", principal=_OWNER)
        assert error.value.code is expected_error
    finally:
        await state.close()


async def test_execution_detail_reads_validated_pending_metadata_without_a_result() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        record = _execution("execution", status=ExecutionStatus.PENDING_START, prompt="question")

        async def pending(execution_id, *, tenant_id):
            assert (execution_id, tenant_id) == ("execution", "tenant")
            return record

        detail = await _reader(state, pending_execution=pending).execution_detail(
            "execution", session_id="session", principal=_OWNER,
        )
        assert detail.execution.status is ExecutionStatus.PENDING_START
        assert detail.result is None
    finally:
        await state.close()


async def test_execution_detail_uses_custom_result_owner_without_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        record = await _seed(state)
        repository = state.execution.executions

        class Executions:
            state_store = repository.state_store
            get = repository.get
            get_header = repository.get_header

            async def get_result(self, execution_id, *, tenant_id):
                assert (execution_id, tenant_id) == ("execution", "tenant")
                return replace(record.result, output=StoredPayload.inline_json({"source": "custom"}))

        async def no_writes(*args, **kwargs):
            raise AssertionError("execution detail is read-only")

        with monkeypatch.context() as patch:
            patch.setattr(repository.state_store, "mutate", no_writes)
            patch.setattr(state.conversation.sessions.state_store, "mutate", no_writes)
            detail = await _reader(state, executions=Executions()).execution_detail(
                "execution", session_id="session", principal=_OWNER,
            )
        assert detail.result.output == {"source": "custom"}
    finally:
        await state.close()


@pytest.mark.parametrize("source", ("changed_output", "nonterminal", "different_owner", "missing"))
async def test_execution_detail_never_returns_a_stale_terminal_record(
    monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        from linktools.ai.core import ResourceKind, ResourceRef
        from linktools.ai.persistence.codec import decode_execution_metadata, encode_execution_content, encode_envelope

        original = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="question")
        original = replace(original, started_at=original.created_at)
        await state.conversation.sessions.create(_session())
        repository = state.execution.executions
        header = ResourceRef(ResourceKind.EXECUTION, "execution", "tenant")
        # The stale observation is independent of the source writer's immutable terminal state.
        cached_metadata = decode_execution_metadata(encode_envelope({
            "type": "execution_record", "payload": encode_execution_content(original, {}),
        }), tenant_id="tenant")
        if source == "changed_output":
            fresh = replace(original, result=replace(
                original.result, output=StoredPayload.inline_json({"source": "fresh"}),
            ))
        elif source == "nonterminal":
            fresh = replace(original, status=ExecutionStatus.STARTED, result=None, completed_at=None)
        elif source == "different_owner":
            fresh = replace(original, principal_id="other")
        else:
            fresh = None

        read_metadata = repository.get_metadata_record
        if fresh is not None:
            await repository.create_execution(fresh)
            await state.conversation.sessions.admit_execution(
                "session", tenant_id="tenant", execution_id="execution", expected=None,
            )
            if fresh.status in _TERMINAL:
                await _commit_empty_turn(state, fresh)

        metadata_reads = []
        async def get_metadata(execution_id, *, tenant_id, fresh=False):
            assert (execution_id, tenant_id) == ("execution", "tenant")
            metadata_reads.append(fresh)
            return (await read_metadata(execution_id, tenant_id=tenant_id, fresh=True)
                    if fresh else cached_metadata)

        async def get_owned_header(execution_id, *, tenant_id):
            assert (execution_id, tenant_id) == ("execution", "tenant")
            return replace(header, owner_principal_id=original.principal_id)

        with monkeypatch.context() as patch:
            patch.setattr(repository, "get_metadata_record", get_metadata)
            if source in {"different_owner", "missing"}:
                # Custom execution headers can impose an owner beyond tenant scope.
                patch.setattr(repository, "get_header", get_owned_header)
            reader = _reader(state)
            if source in {"different_owner", "missing"}:
                with pytest.raises(AIError) as error:
                    await reader.execution_detail("execution", session_id="session", principal=_OWNER)
                assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
                assert metadata_reads == [False, True]
            else:
                detail = await reader.execution_detail("execution", session_id="session", principal=_OWNER)
                if source == "nonterminal":
                    assert detail.execution.status is ExecutionStatus.STARTED
                    assert detail.result is None
                    with pytest.raises(AIError) as error:
                        await reader.result("execution", principal=_OWNER)
                    assert error.value.code is ErrorCode.EXECUTION_NOT_READY
                else:
                    assert detail.execution.status is detail.result.status is ExecutionStatus.SUCCEEDED
                    assert detail.result.output == {"source": "fresh"}
                    assert (await reader.result("execution", principal=_OWNER)).output == {"source": "fresh"}
    finally:
        await state.close()


class _SqlTurnAuthorization(TenantAuthorizationPolicy):
    """Production-style host authorization, with an independently denied turn."""

    def __init__(self):
        super().__init__()
        self.actions = []
        self.deny_execution = False

    async def authorize(self, principal, action, resource):
        self.actions.append(action)
        await super().authorize(principal, action, resource)
        if self.deny_execution and action is AuthorizationAction.EXECUTION_READ:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


@pytest.mark.parametrize("outcome", ("active", "producer-stopped", "success", "failure", "cancel"))
async def test_sqlite_turn_detail_reads_authoritative_session_result(tmp_path: Path, outcome: str) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart

    from linktools.ai.runtime import Runtime, RuntimeFeatures

    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise AIError(ErrorCode.MODEL_UNAVAILABLE)
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("detail")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "detail.db")
    policy = _SqlTurnAuthorization()
    async with (
        Persistence.open("execution-detail", storage=storage, tenant_id=_OWNER.tenant_id) as state,
        Runtime.open(persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
                     authorization=policy) as runtime,
    ):
        await runtime.sessions.create_turn_session("session", principal=_OWNER)
        await runtime.sessions.create_turn_session("other", principal=_OWNER)
        turn = await runtime.agents.get().session("session").start(
            "question", principal=_OWNER, idempotency_key="request",
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            active = await turn.result()
            assert active.key == turn.key
            assert active.status is ExecutionStatus.PENDING_START
            assert active.view["output"] is None and active.view["conversation_committed"] is False
            assert (await runtime.history.turn_session_state("session", principal=_OWNER))[
                "active_turn_id"
            ] == turn.key.turn_id
            with pytest.raises(AIError) as binding:
                await runtime.history.turn_result("other", turn.key.turn_id, principal=_OWNER)
            assert binding.value.code is ErrorCode.AUTHORIZATION_DENIED
            if outcome == "producer-stopped":
                row = await state.session_turns.get_turn(turn.key)
                with pytest.raises(AIError) as stopped:
                    await runtime.history.turn_result(
                        "session", turn.key.turn_id, principal=_OWNER, producer_stopped_generation=row["generation"],
                    )
                assert stopped.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
                assert await state.session_turns.get_turn(turn.key) == row
            elif outcome == "cancel":
                assert await turn.cancel()
        finally:
            release.set()

        terminal = await turn.wait_final(timeout_seconds=5)
        assert terminal.status is {
            "failure": ExecutionStatus.FAILED, "cancel": ExecutionStatus.CANCELLED,
        }.get(outcome, ExecutionStatus.SUCCEEDED)
        assert terminal.view["conversation_committed"] is True
        assert terminal.view["output"] == ({"text": "done"} if terminal.status is ExecutionStatus.SUCCEEDED else None)
        if outcome in ("failure", "cancel"):
            assert terminal.view["error_code"] == {
                "failure": ErrorCode.MODEL_UNAVAILABLE, "cancel": ErrorCode.EXECUTION_CANCELLED,
            }[outcome].value
        assert (await runtime.history.turn_session_state("session", principal=_OWNER))["active_turn_id"] is None
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=_OWNER.tenant_id) is None

    async with (
        Persistence.open("execution-detail", storage=storage, tenant_id=_OWNER.tenant_id, read_only=True) as state,
        History.open(persistence=state, authorization=policy) as history,
    ):
        assert await history.turn_result("session", turn.key.turn_id, principal=_OWNER) == terminal.view
    assert calls == 1


async def test_native_turn_detail_rechecks_read_policies_and_does_not_freeze_active_result(tmp_path: Path) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from sqlalchemy import event

    from linktools.ai.runtime import Runtime, RuntimeFeatures

    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    entered, release = asyncio.Event(), asyncio.Event()

    async def model(_messages, _info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("detail")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "detail.db", cache=InMemoryCacheBackend(max_bytes=2**20))
    policy = _SqlTurnAuthorization()
    async with (
        Persistence.open("execution-detail", storage=storage, tenant_id=_OWNER.tenant_id) as state,
        Runtime.open(persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
                     authorization=policy) as runtime,
    ):
        await runtime.sessions.create_turn_session("session", principal=_OWNER)
        turn = await runtime.agents.get().session("session").start(
            "question", principal=_OWNER, idempotency_key="request",
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            before = await turn.result()
            assert before.status is ExecutionStatus.PENDING_START
            policy.actions.clear()
            statements = []
            engine = state.session_turns._context.engine.sync_engine

            def capture(_connection, _cursor, statement, *_args):
                statements.append(statement.lstrip().upper())

            event.listen(engine, "before_cursor_execute", capture)
            try:
                assert (await turn.result()).view == before.view
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            assert policy.actions == [AuthorizationAction.SESSION_READ, AuthorizationAction.EXECUTION_READ]
            assert any(stmt.startswith("SELECT") for stmt in statements)
            assert not any(stmt.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")) for stmt in statements)
            assert not any("AI_MESSAGES" in stmt for stmt in statements)
            policy.deny_execution = True
            try:
                with pytest.raises(AIError) as denied:
                    await turn.result()
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            finally:
                policy.deny_execution = False
        finally:
            release.set()

        terminal = await turn.wait_final(timeout_seconds=5)
        assert terminal.status is ExecutionStatus.SUCCEEDED
        assert terminal.view["output"] == {"text": "done"}
        assert (await turn.result()).view == terminal.view
