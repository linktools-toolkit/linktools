#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-bound execution observations retain authorization and result fences."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.core import AuthorizationAction, ExecutionStatus, Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import ExecutionDetail, History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.state import active_state_transaction
from linktools.ai.storage import InMemoryCacheBackend, StoredPayload

from .test_incremental_projection_boundaries import _storage
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


async def _seed(state, status=ExecutionStatus.SUCCEEDED, *, committed=True):
    await state.conversation.sessions.create(_session())
    error_code = (
        ErrorCode.EXECUTION_CANCELLED.value if status is ExecutionStatus.CANCELLED
        else ErrorCode.MODEL_API_ERROR.value if status in {
            ExecutionStatus.FAILED, ExecutionStatus.RECOVERY_REQUIRED,
        } else None
    )
    record = _execution("execution", status=status, prompt="question", error_code=error_code)
    record = replace(record, started_at=record.created_at)
    await state.execution.executions.create_with_history_head(record)
    if committed and status in _TERMINAL:
        await _commit_empty_turn(state, record)
    return record


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("status", (
    ExecutionStatus.STARTED, ExecutionStatus.RECOVERY_REQUIRED,
    ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED,
))
async def test_execution_detail_projects_session_state_and_committed_result(
    tmp_path: Path, backend: str, status: ExecutionStatus,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
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


async def test_execution_detail_rechecks_session_ownership_after_authorization() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        await _seed(state)
        sessions = state.conversation.sessions

        class Authorization(TenantAuthorizationPolicy):
            async def authorize(self, principal, action, resource):
                await super().authorize(principal, action, resource)
                if action is AuthorizationAction.SESSION_READ:
                    session = await sessions.get("session", tenant_id="tenant")
                    await sessions.compare_and_swap(
                        "session", tenant_id="tenant", expected_revision=session.revision,
                        next_record=replace(session, owner_principal_id="new-owner", revision=session.revision + 1),
                    )

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


async def test_execution_detail_preserves_retained_turn_integrity() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        await _seed(state, committed=False)
        with pytest.raises(AIError) as error:
            await _reader(state).execution_detail("execution", session_id="session", principal=_OWNER)
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
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


async def test_nonterminal_execution_detail_reuses_cache_but_rechecks_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sqlalchemy import event
    from linktools.ai.persistence import _sql
    from linktools.ai.storage import _write_cache

    captured_at = _write_cache.time()
    monkeypatch.setattr(_write_cache, "time", lambda: captured_at)
    monkeypatch.setattr(_sql, "time", lambda: captured_at)

    cache = InMemoryCacheBackend(max_bytes=2**20)
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "detail.db", cache=cache))
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    statements = []
    actions = []

    class Authorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            actions.append(action)
            await super().authorize(principal, action, resource)

    try:
        await _seed(state, ExecutionStatus.STARTED)
        reader = _reader(state, authorization=Authorization())
        await reader.execution_detail("execution", session_id="session", principal=_OWNER)
        actions.clear()
        engine = state.execution.executions.state_store.storage_group.context.engine.sync_engine

        def capture(_connection, _cursor, statement, *_args):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            detail = await reader.execution_detail("execution", session_id="session", principal=_OWNER)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert detail.execution.status is ExecutionStatus.STARTED
        assert detail.result is None
        assert actions == [AuthorizationAction.SESSION_READ, AuthorizationAction.EXECUTION_READ]
        assert statements == []
    finally:
        await state.close()


@pytest.mark.parametrize("source", ("changed_output", "nonterminal", "different_owner", "missing"))
async def test_execution_detail_never_returns_a_stale_terminal_record(
    monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="execution-detail", tenant_id="tenant")
    try:
        original = await _seed(state)
        repository = state.execution.executions
        if source == "changed_output":
            fresh = replace(original, result=replace(
                original.result, output=StoredPayload.inline_json({"source": "fresh"}),
            ))
        elif source == "nonterminal":
            fresh = replace(original, status=ExecutionStatus.STARTED, result=None)
        elif source == "different_owner":
            fresh = replace(original, principal_id="other")
        else:
            fresh = None

        async def get(execution_id, *, tenant_id):
            assert (execution_id, tenant_id) == ("execution", "tenant")
            return fresh if active_state_transaction(repository.state_store) is not None else original

        with monkeypatch.context() as patch:
            patch.setattr(repository, "get", get)
            reader = _reader(state)
            if source in {"different_owner", "missing"}:
                with pytest.raises(AIError) as error:
                    await reader.execution_detail("execution", session_id="session", principal=_OWNER)
                assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
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
