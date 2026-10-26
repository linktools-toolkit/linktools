#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable Session admission and ownership contracts."""

from linktools.ai.persistence import Persistence
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.agent import AgentBinding, AgentCompiler
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
from linktools.ai.model import create_model
from linktools.ai.execution import ExecutionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.execution.events import LiveExecutionEventBroker
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.observe import Observation
from linktools.ai.runtime._execution import (
    CancelEffectOutcome,
    DefaultExecutionService,
    _ExecutionRuntimeBridge,
)
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.contracts import SessionRecord
from linktools.ai.spec import AgentSpec
from sqlalchemy.ext.asyncio import create_async_engine


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id="session",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
    )


async def _admission_result(state: Persistence, execution_id: str) -> str:
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
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
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
        owner = current.active_execution_id
        assert owner is not None

        updated = await state.conversation.sessions.compare_and_swap(
            "session",
            tenant_id="tenant",
            expected_revision=0,
            next_record=replace(
                current,
                revision=1,
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
    state = PersistenceStorage.sql(engine)
    state = Persistence(state)
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

        reopened = PersistenceStorage.sql(engine)
        reopened = Persistence(reopened)
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


def _binding() -> AgentBinding:
    spec = AgentSpec("agent", model="model")
    models = {"model": create_model(protocol="openai", model="gpt-test")}
    compiler = AgentCompiler(
        models=models, candidates=(), agents={spec.id: spec},
    )
    return compiler.bind(compiler.compile(spec))


class _BindingCatalog:
    def binding(self, digest: str) -> AgentBinding:
        binding = _binding()
        assert digest == binding.binding_digest
        return binding


class _History:
    async def history(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
        message_seq: int | None = None,
        part_index: int | None = None,
    ) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def trace(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
        agent_run_seq: int | None = None,
        model_request_seq: int | None = None,
        step_index: int | None = None,
        tool_call_id: str | None = None,
    ) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)

    async def transcript(
        self,
        execution_id: str,
        *,
        tenant_id: str,
        cursor: str | None,
        limit: int,
    ) -> Page[object]:
        del execution_id, tenant_id, cursor, limit
        return Page((), None)


class _RejectingBackend:
    def __init__(
        self,
        repository: object | None = None,
        idempotency: object | None = None,
    ) -> None:
        self.aborted: list[str] = []
        self.committed: list[str] = []
        self._repository = repository
        self._idempotency = idempotency

    async def commit_terminal_checkpoint(
        self,
        commit: object,
        *,
        session_id: "str | None",
        expected_execution: object | None = None,
        capture_metric: object = None,
    ) -> object:
        del session_id, expected_execution, capture_metric
        execution = commit.execution
        self.committed.append(execution.execution_id)
        repository = self._repository
        if repository is not None:
            await repository.compare_and_swap(
                execution.execution_id,
                tenant_id="tenant",
                expected_revision=commit.expected_revision,
                next_record=execution,
            )
        update = commit.idempotency
        idempotency = self._idempotency
        if update is not None and idempotency is not None:
            current = await idempotency.get(
                update.scope,
                update.idempotency_key_digest,
                tenant_id="tenant",
            )
            assert current is not None
            await idempotency.compare_and_swap(
                update.scope,
                update.idempotency_key_digest,
                tenant_id="tenant",
                expected_status=update.expected_status,
                next_record=replace(
                    current,
                    status=update.next_status,
                    result_digest=update.result_digest,
                    error_code=update.error_code,
                    updated_at=execution.updated_at,
                ),
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

    async def cancel(self, execution: object, *, force: bool = False) -> CancelEffectOutcome:
        del execution
        return CancelEffectOutcome.CONFIRMED

    def worker_failure(self, execution_id: str, *, tenant_id: str) -> AIError | None:
        del execution_id, tenant_id
        return None


@pytest.mark.asyncio
async def test_rejected_admission_terminalizes_pending_start() -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="session-admission-rejection", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session())
        backend = _RejectingBackend(
            state.execution.executions,
            state.execution.idempotency,
        )
        runtime_bridge = _ExecutionRuntimeBridge()
        runtime_bridge.bind(backend)  # type: ignore[arg-type]
        service = DefaultExecutionService(
            state.execution,
            state.object_store(PersistenceDomain.EXECUTION),
            TenantAuthorizationPolicy(),
            conversation=True, sessions=state.conversation.sessions,
            catalog=_BindingCatalog(),
            compiler=object(),
            runtime_bridge=runtime_bridge,
            live_broker=LiveExecutionEventBroker(),
            history_reader=_History(),
        )
        with pytest.raises(AIError) as error:
            await service.start_for_session(
                "agent",
                _binding().binding_digest,
                "session",
                ExecutionRequest(
                    user_prompt="hello",
                    principal=Principal("owner", "tenant"),
                    idempotency_key="rejected-start",
                    memory_scope=None,
                    mode="run",
                    planning=False,
                    thinking=False,
                ),
            )
        assert error.value.code is ErrorCode.SESSION_BUSY
        executions = await state.execution.executions.list_by_session(
            "session",
            tenant_id="tenant",
        )
        assert len(executions) == 1
        execution = executions[0]
        assert execution.status is ExecutionStatus.FAILED
        assert execution.error_code == ErrorCode.SESSION_BUSY.value
        assert backend.aborted == [execution.execution_id]
        identities = await state.execution.idempotency.list_by_resource(
            ResourceKind.EXECUTION,
            execution.execution_id,
            tenant_id="tenant",
        )
        assert len(identities) == 1
        assert identities[0].status.value == "FAILED"

        with pytest.raises(AIError) as replay_error:
            await service.start_for_session(
                "agent",
                _binding().binding_digest,
                "session",
                ExecutionRequest(
                    user_prompt="hello",
                    principal=Principal("owner", "tenant"),
                    idempotency_key="rejected-start",
                    memory_scope=None,
                    mode="run",
                    planning=False,
                    thinking=False,
                ),
            )
        assert replay_error.value.code is ErrorCode.SESSION_BUSY
        assert backend.aborted == [execution.execution_id]
        replayed = await state.execution.executions.list_by_session(
            "session",
            tenant_id="tenant",
        )
        assert [item.execution_id for item in replayed] == [execution.execution_id]
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("commit_phase", "cancel_metric"),
    (("normal", False), ("normal", True), ("after", False), ("after", True), ("before", False)),
)
async def test_rejected_start_preserves_admission_truth_and_cleanup_before_metrics(
    commit_phase: str, cancel_metric: bool,
) -> None:
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
    await state.initialize(namespace="rejected-start-metric", tenant_id="tenant")
    entered = asyncio.Event()
    release = asyncio.Event()
    attempted: list[Observation] = []

    class Recorder:
        async def try_record(self, observation: Observation) -> bool:
            attempted.append(observation)
            entered.set()
            await release.wait()
            raise OSError("metric sink failed")

    class Backend(_RejectingBackend):
        async def commit_terminal_checkpoint(self, commit: object, *, session_id: str | None,
                                             expected_execution: object = None,
                                             capture_metric: object = None) -> object:
            if commit_phase == "before":
                asyncio.current_task().cancel()
                await asyncio.sleep(0)
            result = await super().commit_terminal_checkpoint(
                commit, session_id=session_id, expected_execution=expected_execution,
            )
            metrics = object.__new__(LocalExecutionBackend)
            metrics._metric_recorder = Recorder()
            metrics._namespace = "rejected-start-metric"
            metrics._tenant_id = "tenant"
            await metrics._record_committed_terminal(result, session_id=session_id, capture_metric=capture_metric)
            if commit_phase == "after":
                asyncio.current_task().cancel()
                await asyncio.sleep(0)
            return result

    backend = Backend(state.execution.executions, state.execution.idempotency)
    bridge = _ExecutionRuntimeBridge()
    bridge.bind(backend)
    service = DefaultExecutionService(
        state.execution, state.object_store(PersistenceDomain.EXECUTION), TenantAuthorizationPolicy(),
        conversation=True, sessions=state.conversation.sessions, catalog=_BindingCatalog(), compiler=object(),
        runtime_bridge=bridge, live_broker=LiveExecutionEventBroker(), history_reader=_History(),
    )
    caller: asyncio.Task[object] | None = None
    try:
        await state.conversation.sessions.create(_session())
        request = ExecutionRequest(
            user_prompt="hello", principal=Principal("owner", "tenant"),
            idempotency_key="rejected-metric", memory_scope=None,
            mode="run", planning=False, thinking=False,
        )
        caller = asyncio.create_task(service.start_for_session(
            "agent", _binding().binding_digest, "session", request,
        ))
        if commit_phase == "before":
            with pytest.raises(asyncio.CancelledError):
                await caller
            executions = await state.execution.executions.list_by_session("session", tenant_id="tenant")
            assert len(executions) == 1
            assert executions[0].status is ExecutionStatus.PENDING_START
            assert not backend.aborted
            assert not attempted
            assert service._session_locks == {}
            return
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert len(attempted) == 1
        execution_id = attempted[0].correlation["linktools.execution_id"]
        assert backend.aborted == [execution_id]
        assert service._session_locks == {}
        assert attempted[0].status == "FAILED"
        assert attempted[0].error_code == ErrorCode.SESSION_BUSY.value
        if cancel_metric:
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
        else:
            release.set()
            if commit_phase == "after":
                with pytest.raises(asyncio.CancelledError):
                    await caller
            else:
                with pytest.raises(AIError) as original:
                    await caller
                assert original.value.code is ErrorCode.SESSION_BUSY
        terminal = await state.execution.executions.get(execution_id, tenant_id="tenant")
        assert terminal.status is ExecutionStatus.FAILED
        assert terminal.error_code == ErrorCode.SESSION_BUSY.value
    finally:
        release.set()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
        await state.close()
