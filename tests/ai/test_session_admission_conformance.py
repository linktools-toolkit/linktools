#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Session admission and ownership contracts."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from types import SimpleNamespace
from linktools.ai.core import (
    ExecutionStatus,
    Page,
    Principal,
    ResourceKind,
    SessionStatus,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.runtime import ExecutionRequest, RuntimeState
from linktools.ai.runtime._execution import CancelEffectOutcome, DefaultExecutionService
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import ConversationCursor, SessionRecord
from sqlalchemy.ext.asyncio import create_async_engine


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        binding_digest="b" * 64,
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
    )


async def _admission_result(state: RuntimeState, execution_id: str) -> str:
    try:
        record = await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id=execution_id,
            expected=None,
        )
    except AIError as error:
        return error.code.value
    return record.active_execution_id or ""


@pytest.mark.asyncio
async def test_memory_admission_is_atomic_and_cas_preserves_token() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-admission-memory", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        results = await asyncio.gather(
            _admission_result(state, "execution-a"),
            _admission_result(state, "execution-b"),
        )
        assert sorted(results) == ["SESSION_BUSY", "execution-a"] or sorted(results) == [
            "SESSION_BUSY",
            "execution-b",
        ]
        current = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert current is not None
        assert current.revision == 0
        assert current.resource_generation == 0
        owner = current.active_execution_id
        assert owner is not None

        updated = await state.conversation.sessions.compare_and_swap(
            "session",
            tenant_id="tenant",
            expected_revision=0,
            next_record=replace(
                current,
                revision=1,
                resource_generation=1,
                metadata={"key": "value"},
                active_execution_id="stale-caller-value",
            ),
        )
        assert updated.active_execution_id == owner

        await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id="late-owner",
        )
        after_late_release = await state.conversation.sessions.get(
            "session",
            tenant_id="tenant",
        )
        assert after_late_release is not None
        assert after_late_release.active_execution_id == owner
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_sql_admission_is_atomic_and_token_survives_reopen(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'runtime.db'}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="session-admission-sql", tenant_id="tenant")
    await state.conversation.sessions.create(_session())
    try:
        results = await asyncio.gather(
            _admission_result(state, "execution-a"),
            _admission_result(state, "execution-b"),
        )
        assert sorted(results) == ["SESSION_BUSY", "execution-a"] or sorted(results) == [
            "SESSION_BUSY",
            "execution-b",
        ]
        current = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert current is not None
        assert current.revision == 0
        owner = current.active_execution_id
        assert owner is not None
        await state.close()

        reopened = RuntimeState.sql(engine)
        await reopened.initialize(namespace="session-admission-sql", tenant_id="tenant")
        try:
            persisted = await reopened.conversation.sessions.get(
                "session",
                tenant_id="tenant",
            )
            assert persisted is not None
            assert persisted.active_execution_id == owner
            assert persisted.revision == 0
        finally:
            await reopened.close()
    finally:
        if state.ready:
            await state.close()
        await engine.dispose()


@pytest.mark.asyncio
async def test_closing_session_can_commit_owned_continuation_then_close() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-admission-close", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="execution",
            expected=None,
        )
        closing = await state.conversation.sessions.transition_status(
            "session",
            tenant_id="tenant",
            expected=frozenset({SessionStatus.OPEN}),
            next_status=SessionStatus.CLOSING,
        )
        committed = await state.conversation.sessions.advance_continuation(
            "session",
            tenant_id="tenant",
            execution_id="execution",
            expected=None,
            next_cursor=ConversationCursor("turn"),
        )
        assert committed.status is SessionStatus.CLOSING
        assert committed.active_execution_id == "execution"
        assert committed.revision == closing.revision + 1
        await state.conversation.sessions.release_execution(
            "session",
            tenant_id="tenant",
            execution_id="execution",
        )
        closed = await state.conversation.sessions.transition_status(
            "session",
            tenant_id="tenant",
            expected=frozenset({SessionStatus.CLOSING}),
            next_status=SessionStatus.CLOSED,
            closed_at=datetime.now(timezone.utc),
            require_no_active=True,
        )
        assert closed.active_execution_id is None
        assert closed.continuation == ConversationCursor("turn")
    finally:
        await state.close()


class _History:
    async def trace(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def transcript(self, execution_id: str, *, tenant_id: str, cursor: str | None, limit: int) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)


class _RejectingBackend:
    def __init__(self, repository: object | None = None) -> None:
        self.aborted: list[str] = []
        self.committed: list[str] = []
        self._repository = repository

    async def commit_terminal_checkpoint(
        self,
        commit: object,
        *,
        session_id: "str | None",
    ) -> object:
        del session_id
        execution = commit.execution
        self.committed.append(execution.execution_id)
        repository = self._repository
        if repository is not None:
            await repository.compare_and_swap(
                execution.execution_id,
                tenant_id=execution.tenant_id,
                expected_revision=commit.expected_revision,
                next_record=execution,
            )
        return SimpleNamespace(execution=execution, result=commit.result)

    async def prepare_start(self, request: object, execution: object, identity: object) -> None:
        del request, execution, identity
        raise AIError(ErrorCode.SESSION_BUSY)

    async def abort_start(self, execution: object) -> None:
        self.aborted.append(execution.execution_id)

    async def launch(self, request: object, execution: object) -> None:
        del request, execution
        raise AssertionError("rejected start must not launch")

    async def cancel(self, execution: object) -> CancelEffectOutcome:
        del execution
        return CancelEffectOutcome.CONFIRMED

    def worker_failure(self, execution_id: str, *, tenant_id: str) -> AIError | None:
        del execution_id, tenant_id
        return None


@pytest.mark.asyncio
async def test_rejected_admission_terminalizes_pending_start() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-admission-rejection", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        backend = _RejectingBackend(state.execution.executions)
        service = DefaultExecutionService(
            state.execution,
            state._object_store(RuntimeDomain.EXECUTION),
            TenantAuthorizationPolicy(),
            sessions=state.conversation.sessions,
            backend=backend,
            history_reader=_History(),
        )
        from linktools.ai.runtime.state import RuntimeStateCommands
        from linktools.ai.runtime.state._repositories import (
            ConversationHistoryRepositoryImpl,
            EventRepositoryImpl,
            OperationLedgerRepository,
        )

        service.bind_terminal_committer(
            RuntimeStateCommands(
                state.execution.executions,
                namespace="session-admission-rejection",
                events=EventRepositoryImpl(
                    state.execution.events.state_store,
                    namespace="session-admission-rejection",
                    tenant_id="tenant",
                ),
                operations=OperationLedgerRepository(
                    state.execution.operations.state_store,
                    namespace="session-admission-rejection",
                    tenant_id="tenant",
                    domain=RuntimeDomain.EXECUTION,
                ),
                conversation=state.conversation.sessions,
                recovery=state.recovery.checkpoints,
                conversation_history=ConversationHistoryRepositoryImpl(
                    state.conversation.histories.state_store,
                    namespace="session-admission-rejection",
                    tenant_id="tenant",
                ),
            )
        )
        with pytest.raises(AIError) as error:
            await service.run_for_session(
                "b" * 64,
                "session",
                ExecutionRequest(
                    user_prompt="hello",
                    principal=Principal("owner", "tenant"),
                    idempotency_key="rejected-start",
                ),
            )
        assert error.value.code.value == "SESSION_BUSY"
        executions = await state.execution.executions.list_by_session(
            "session",
            tenant_id="tenant",
        )
        assert len(executions) == 1
        execution = executions[0]
        assert execution.status is ExecutionStatus.FAILED
        assert execution.error_code == "SESSION_BUSY"
        assert backend.aborted == [execution.execution_id]
        identities = await state.execution.idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            execution.execution_id,
            tenant_id="tenant",
        )
        assert len(identities) == 1
        assert identities[0].status.value == "FAILED"
    finally:
        await state.close()
