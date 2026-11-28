#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""First attempts claim authority only after prepared input is durable."""

import asyncio
from pathlib import Path

import pytest
from linktools.ai.persistence import Persistence
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy, ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.capability import SessionCapability, TraceCapability
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage._database import SqlStorageContext
from pydantic_ai.messages import ModelResponse, TextPart

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, native_checkpoint


class _SessionHostAuthorization(TenantAuthorizationPolicy):
    pass


def _session_features(model, checkpointed: bool) -> RuntimeFeatures:
    features = RuntimeFeatures("startup-session")
    features.capability(SessionCapability(checkpoints=checkpointed), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    return model_features(_Models(_UsageFunctionModel(model)), features)


@pytest.mark.asyncio
@pytest.mark.parametrize(("checkpointed", "trace"), (
    (False, None), (True, None), (False, CommitPolicy.sync()),
    (False, CommitPolicy.batch()), (False, CommitPolicy.terminal()),
))
async def test_first_provider_has_durable_prepared_input_and_original_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointed: bool, trace: CommitPolicy | None,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup', defaults={"execution": {"retain_history": True}})
    if trace is not None:
        group.capability(TraceCapability(commit=trace), scope="runtime")
    checkpoint = native_checkpoint() if checkpointed else None
    if checkpoint is not None:
        group.capability(checkpoint, scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = PersistenceStorage.sqlite(tmp_path / 'state.db')
    async with Persistence.open('startup-boundaries', storage=state) as persistence_35, Runtime.open(persistence=persistence_35, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        agent = runtime.agents.get()
        original = SqlStorageContext.run_mutation
        commits = []
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get('domain'))
            return result
        with monkeypatch.context() as patch:
            patch.setattr(SqlStorageContext, 'run_mutation', count)
            execution = await agent.start('question', idempotency_key='same')
            try:
                await asyncio.wait_for(entered.wait(), 5)
                current = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                head = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert current.status is ExecutionStatus.STARTED
                assert current.revision == 3 and current.agent_run_seq == 1
                assert current.input_context is not None
                async with Persistence.open('startup-boundaries', storage=PersistenceStorage.sqlite(tmp_path / 'state.db'), read_only=True) as observer:
                    visible = await observer.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                    checkpoint_record = await observer.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                    assert visible.status is ExecutionStatus.STARTED and visible.input_context == current.input_context
                    assert checkpoint_record.recovery.agent_run_id is not None
                assert current.producer_generation == 2 and current.producer_claim_id is not None
                if checkpoint is not None:
                    runs = await checkpoint.store.list_runs()
                    assert len(runs) == 1
                    assert await checkpoint.store.latest_snapshot(run_id=runs[0].run_id) is not None
                replay = await agent.start('question', idempotency_key='same')
                assert replay.execution_id == execution.execution_id
                assert len(calls) == 1
            finally:
                release.set()
                terminal = await execution.wait()
            assert terminal.result.status is ExecutionStatus.SUCCEEDED
            if trace is not None:
                assert (await execution.model_interactions()).items


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('memory', 'local_sqlite', 'sqlite'))
@pytest.mark.parametrize('phase', ('before', 'after'))
@pytest.mark.parametrize('boundary', ('admission', 'activation'))
@pytest.mark.parametrize('checkpointed', (False, True))
async def test_startup_unknown_commit_never_repeats_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str, checkpointed: bool, boundary: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence.contracts import (
        ExecutionRecord,
        ExecutionRecoveryDetail,
        ExecutionStartReservationResult,
    )
    from linktools.ai.persistence.state import Observed, active_state_scope
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-ack', defaults={"execution": {"retain_history": True}})
    if checkpointed:
        group.capability(native_checkpoint(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = (PersistenceStorage.in_memory() if backend == 'memory' else
             PersistenceStorage.from_root(tmp_path / 'state', exclusive_writer=True) if backend == 'local_sqlite' else
             PersistenceStorage.sqlite(tmp_path / 'state.db'))
    async with Persistence.open('startup-ack', storage=state) as persistence_85, Runtime.open(persistence=persistence_85, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        agent = runtime.agents.get()
        group_type = type(runtime._state.execution.executions.state_store.storage_group)
        original = group_type.mutate
        faults = []
        def matches(value):
            if boundary == 'admission':
                return isinstance(value, ExecutionStartReservationResult) and value.created
            current = value.value if isinstance(value, Observed) else value
            return isinstance(current, ExecutionRecord) and current.recovery is not None and current.recovery.agent_run_id is not None
        async def mutate(group, stores, callback):
            outer = active_state_scope() is None
            async def guarded(transaction):
                value = await callback(transaction)
                if outer and phase == 'before' and not faults and matches(value):
                    faults.append(True)
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return value
            result = await original(group, stores, guarded)
            if outer and phase == 'after' and not faults and matches(result):
                faults.append(True)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(group_type, 'mutate', mutate)
            execution = await agent.start('question', idempotency_key='same')
            assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert faults == [True] and len(calls) == 1
        replay = await agent.start('question', idempotency_key='same')
        assert replay.execution_id == execution.execution_id
        assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointed", (False, True))
async def test_two_runtimes_share_one_first_attempt_after_parallel_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointed: bool,
) -> None:
    from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner

    prepared, release = asyncio.Event(), asyncio.Event()
    preparations, owners, calls = [], [], []

    async def model(messages, _info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    original = _RegenerableTurnRunner.admit

    async def admit(runner, key, **kwargs):
        preparations.append((runner, kwargs["revision"]))
        if len(preparations) == 2:
            prepared.set()
        await release.wait()
        admitted = await original(runner, key, **kwargs)
        if admitted is not None:
            owners.append(runner)
        return admitted

    monkeypatch.setattr(_RegenerableTurnRunner, "admit", admit)
    features = _session_features(model, checkpointed)
    path = tmp_path / "state.db"
    async with (
        Persistence.open("startup-race", storage=PersistenceStorage.sqlite(path)) as first_state,
        Runtime.open(persistence=first_state, features=features, authorization=_SessionHostAuthorization()) as first,
        Persistence.open("startup-race", storage=PersistenceStorage.sqlite(path)) as second_state,
        Runtime.open(persistence=second_state, features=features, authorization=_SessionHostAuthorization()) as second,
    ):
        principal = first.default_principal
        await first.sessions.create_turn_session("session", principal=principal)
        starts = asyncio.gather(*(runtime.sessions.start(
            "session", "question", agent="default", principal=principal, idempotency_key="same",
        ) for runtime in (first, second)))
        try:
            await asyncio.wait_for(prepared.wait(), 5)
        finally:
            release.set()
        one, two = await asyncio.wait_for(starts, 5)
        assert one.key == two.key
        assert preparations[0][0] is not preparations[1][0]
        assert preparations[0][1] == preparations[1][1]
        owner, = owners
        producer, replay = (one, two) if owner is first.sessions._turn_runner else (two, one)
        assert (await producer.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        assert (await replay.result()).status is ExecutionStatus.SUCCEEDED
        page = await second.history.turn_session("session", principal=principal)
        assert [item["turn_id"] for item in page["timeline"]["items"]] == [one.key.turn_id]
        assert len(calls) == 1

@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointed", (False, True))
async def test_busy_native_session_rejects_without_admitting_another_turn(tmp_path: Path, checkpointed: bool) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(messages, _info):
        calls.append(messages)
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = _session_features(model, checkpointed)
    path = tmp_path / "state.db"
    async with (
        Persistence.open("startup-busy", storage=PersistenceStorage.sqlite(path)) as first_state,
        Runtime.open(persistence=first_state, features=features, authorization=_SessionHostAuthorization()) as first,
        Persistence.open("startup-busy", storage=PersistenceStorage.sqlite(path)) as second_state,
        Runtime.open(persistence=second_state, features=features, authorization=_SessionHostAuthorization()) as second,
    ):
        principal = first.default_principal
        await first.sessions.create_turn_session("session", principal=principal)
        one = await first.sessions.start("session", "question", agent="default", principal=principal,
                                         idempotency_key="owner")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            with pytest.raises(AIError) as busy:
                await second.sessions.start("session", "other", agent="default", principal=principal,
                                            idempotency_key="rejected")
            assert busy.value.code is ErrorCode.SESSION_BUSY
            observed = await second.history.turn_session_state("session", principal=principal)
            assert observed["active_turn_id"] == one.key.turn_id and observed["last_turn_no"] == 0
        finally:
            release.set()
        assert (await one.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        two = await second.sessions.start("session", "other", agent="default", principal=principal,
                                          idempotency_key="rejected")
        assert (await two.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        page = await second.history.turn_session("session", principal=principal)
        assert [item["turn_id"] for item in page["timeline"]["items"]] == [one.key.turn_id, two.key.turn_id]
        assert len(calls) == 2

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checkpointed,remote,late",
    ((False, False, False), (True, False, False), (False, True, False), (True, True, False), (False, True, True)),
)
async def test_native_cancel_before_provider_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointed: bool, remote: bool, late: bool,
) -> None:
    from linktools.ai.runtime import _agent_executor
    from linktools.ai.runtime._regenerable_turn import _RegenerableTurnRunner

    preparing, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def model(messages, _info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    target = _agent_executor if late else _RegenerableTurnRunner
    name = "_materialize_agent" if late else "settle"
    original = getattr(target, name)

    async def paused(*args, **kwargs):
        preparing.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(target, name, paused)
    features = _session_features(model, checkpointed)
    path = tmp_path / "state.db"
    async with (
        Persistence.open("startup-cancel", storage=PersistenceStorage.sqlite(path)) as first_state,
        Runtime.open(persistence=first_state, features=features, authorization=_SessionHostAuthorization()) as first,
        Persistence.open("startup-cancel", storage=PersistenceStorage.sqlite(path)) as second_state,
        Runtime.open(persistence=second_state, features=features, authorization=_SessionHostAuthorization()) as second,
    ):
        principal = first.default_principal
        await first.sessions.create_turn_session("session", principal=principal)
        turn = await first.sessions.start("session", "question", agent="default", principal=principal,
                                          idempotency_key="same")
        try:
            await asyncio.wait_for(preparing.wait(), 5)
            observer = second if remote else first
            assert await observer.sessions.cancel_turn("session", turn.key.turn_id, principal=principal)
            if remote:
                generation = await second_state.session_turns.get_turn(turn.key)
                assert generation["state"] == "cancelling" and generation["generation"] == 1
        finally:
            release.set()
        terminal = await turn.wait_final(timeout_seconds=5)
        assert terminal.status is ExecutionStatus.CANCELLED
        assert calls == []
        observed = await second.history.turn_session("session", principal=principal)
        assert observed["session"]["active_turn_id"] is None
        assert observed["timeline"]["items"] == [terminal.view]

@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ('archive', 'context', 'timeout'))
@pytest.mark.parametrize('checkpointed', (False, True))
async def test_preparation_failure_stays_unclaimed_and_can_be_explicitly_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, checkpointed: bool,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.inputs import ExecutionContextArchive
    from dataclasses import replace
    from linktools.ai.runtime._local import LocalExecutionBackend
    from linktools.ai.persistence.state import active_state_scope
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-preparation-failure', defaults={"execution": {"retain_history": True}})
    if checkpointed:
        group.capability(native_checkpoint(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = PersistenceStorage.sqlite(tmp_path / 'state.db')
    async with Persistence.open('startup-preparation-failure', storage=state) as persistence_248, Runtime.open(persistence=persistence_248, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        agent = runtime.agents.get()
        observed = []
        async def fail(*args, **kwargs):
            assert active_state_scope() is None
            observed.append(True)
            if fault == 'timeout':
                raise TimeoutError('preparation deadline')
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE if fault == 'archive' else ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        with monkeypatch.context() as patch:
            if fault == 'archive':
                from linktools.ai.persistence._archive import ContentArchive
                prepare_content = ContentArchive.prepare_content
                prepare_context = ExecutionContextArchive.prepare
                preparing_context = False

                async def with_metadata(archive, context, **kwargs):
                    nonlocal preparing_context
                    preparing_context = True
                    try:
                        return await prepare_context(
                            archive, replace(context, session_metadata={"large": "context" * 10_000}), **kwargs,
                        )
                    finally:
                        preparing_context = False

                async def fail_input_preparation(self, values, prepared):
                    if preparing_context and any(codec == "json" for codec, _raw in values):
                        return await fail(self, values, prepared)
                    return await prepare_content(self, values, prepared)

                patch.setattr(ExecutionContextArchive, 'prepare', with_metadata)
                patch.setattr(ContentArchive, 'prepare_content', fail_input_preparation)
            elif fault == 'context':
                patch.setattr(LocalExecutionBackend, 'load_input_context', fail)
            else:
                patch.setattr(ExecutionContextArchive, 'prepare', fail)
            execution = await agent.start('question', idempotency_key='same')
            with pytest.raises(AIError):
                await asyncio.wait_for(execution.wait(), 5)
        assert observed and calls == []
        current = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert current.agent_run_seq == 0 and current.input_context is None
        await execution.cancel(force=True, idempotency_key='explicit-cancel')
        current = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert current.status is ExecutionStatus.CANCELLED


@pytest.mark.asyncio
@pytest.mark.parametrize('member', ('execution',))
async def test_prepared_attempt_member_conflict_rolls_back_authority_and_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str,
) -> None:
    from dataclasses import replace

    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence.contracts import (
        AgentAttemptClaim,
        ExecutionRecoveryDetail,
        ExecutionRecoveryState,
        RuntimePayloadRef,
    )
    from linktools.ai.persistence._record_owner import record_owner_table
    from linktools.ai.persistence._runtime_commands import StateCommands
    from linktools.ai.persistence._sql import _SqlTransaction
    from linktools.ai.storage import StoredPayload
    from sqlalchemy import update

    from .test_checkpoint_write_batches import _snapshot
    from .test_history_projection_conformance import _record

    state = Persistence(PersistenceStorage.sqlite(tmp_path / 'state.db'))
    await state.initialize(namespace='startup-member-cas', tenant_id='tenant')
    try:
        execution = replace(_record(ExecutionStatus.STARTED, 0), revision=1, durable_trace=False, recovery=ExecutionRecoveryDetail())
        await state.execution.executions.create_execution(execution)
        commands = StateCommands(state.execution.executions, namespace='startup-member-cas',
                                        events=state.execution.events, checkpoint=state.checkpoint,
                                        background_tasks=set())
        stores = (state.execution.executions.state_store,)
        before = [await _snapshot(store) for store in stores]
        original = _SqlTransaction.replace_records
        conflicts = []
        async def conflict(transaction, replacements):
            target = next((item for item in replacements if item.record.kind == member), None)
            if target is not None:
                table = transaction._table(record_owner_table(target.record.key))
                changed = await transaction._execute(update(table).where(
                    table.c.store_key == transaction._store_key,
                    table.c.key == target.record.key,
                ).values(storage_revision=target.expected_storage_revision + 1))
                assert changed.rowcount == 1
                conflicts.append(member)
            return await original(transaction, replacements)
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, 'replace_records', conflict)
            with pytest.raises(AIError) as caught:
                await commands.commit_prepared_agent_attempt_checkpoint(
                    AgentAttemptClaim(execution.execution_id, 1, 0, ExecutionRecoveryState.ADMITTED),
                    RuntimePayloadRef(StoredPayload.inline_json({'prepared': True}), None),
                )
        assert caught.value.code is ErrorCode.STORAGE_CONFLICT
        assert conflicts == [member]
        assert [await _snapshot(store) for store in stores] == before
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_deadline_expires_during_preparation_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datetime import datetime, timedelta, timezone

    from linktools.ai.core import RunBudget
    from linktools.ai.errors import ErrorCode
    from linktools.ai.inputs import ExecutionContextArchive

    preparing, release = asyncio.Event(), asyncio.Event()
    now = datetime.now(timezone.utc)
    clock = [now]
    class BudgetClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].replace(tzinfo=None) if tz is None else clock[0].astimezone(tz)
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('not expected')])
    group = RuntimeFeatures('startup-budget-review', defaults={"execution": {"retain_history": True}})
    group.agent('default', model='default', allow_tools=())
    state = PersistenceStorage.sqlite(tmp_path / 'state.db')
    original = ExecutionContextArchive.prepare
    async def prepare(archive, context, **kwargs):
        reference = await original(archive, context, **kwargs)
        preparing.set()
        await release.wait()
        return reference
    monkeypatch.setattr('linktools.ai.persistence._budget.datetime', BudgetClock)
    monkeypatch.setattr(ExecutionContextArchive, 'prepare', prepare)
    async with Persistence.open('startup-budget-review', storage=state) as persistence_412, Runtime.open(persistence=persistence_412, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        agent = runtime.agents.get()
        execution = await agent.start('question', idempotency_key='key', budget=RunBudget(deadline_at=now + timedelta(seconds=30)))
        await asyncio.wait_for(preparing.wait(), 5)
        before = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert before.agent_run_seq == 0 and before.input_context is None
        clock[0] = now + timedelta(seconds=60)
        release.set()
        result = (await asyncio.wait_for(execution.wait(), 5)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
        usage = await execution.budget_usage()
        assert usage.model_requests == usage.in_flight_model_requests == usage.unknown_model_requests == 0
        assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('partial', ['missing', 'unbound'])
async def test_admission_readback_rejects_real_session_partial(
    monkeypatch: pytest.MonkeyPatch, partial: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence.contracts import ExecutionRecord, ExecutionStartReservationResult
    from linktools.ai.persistence.state import active_state_scope

    calls, faults = [], []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('not expected')])
    capabilities = RuntimeFeatures('partial-review', defaults={"execution": {"retain_history": True}})
    capabilities.agent('default', model='default', allow_tools=())
    state = PersistenceStorage.in_memory()
    async with Persistence.open('partial-review', storage=state) as persistence_446, Runtime.open(persistence=persistence_446, features=model_features(_Models(_UsageFunctionModel(model)), capabilities)) as runtime:
        session = await runtime.agents.get().create_session('session')
        group = runtime._state.execution.executions.state_store.storage_group
        repository = runtime._state.conversation.sessions
        original = type(group).mutate
        async def mutate(owner, stores, callback):
            outer = active_state_scope() is None
            result = await original(owner, stores, callback)
            admitted = result.execution if isinstance(result, ExecutionStartReservationResult) else result
            if owner is group and outer and not faults and isinstance(admitted, ExecutionRecord) and admitted.status is ExecutionStatus.STARTED and admitted.agent_run_seq == 0:
                faults.append(admitted.execution_id)
                if partial == 'unbound':
                    await repository.release_execution('session', tenant_id=runtime.tenant_id, execution_id=admitted.execution_id)
                else:
                    async def remove(transaction):
                        key = repository._key('session', ('session',))
                        record = await transaction.get_record(key)
                        assert await transaction.delete_record(key, expected_storage_revision=record.storage_revision)
                    await repository.state_store.mutate(remove)
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(type(group), 'mutate', mutate)
            with pytest.raises(AIError) as caught:
                await session.start('question', idempotency_key='key')
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert len(faults) == 1 and calls == []
        current = await runtime._state.execution.executions.get(faults[0], tenant_id=runtime.tenant_id)
        assert current.status is ExecutionStatus.STARTED and current.agent_run_seq == 0
        value = await repository.get('session', tenant_id=runtime.tenant_id)
        assert value is None if partial == 'missing' else value.active_execution_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ('before', 'after'))
async def test_cancelled_admission_commit_never_dispatches_before_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.execution import CancelExecutionRequest, ExecutionStartReservationResult
    from linktools.ai.persistence.state import active_state_scope

    calls, faults = [], []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('not expected')])
    features = RuntimeFeatures('cancelled-admission', defaults={"execution": {"retain_history": True}})
    features.agent('default', model='default', allow_tools=())
    state = PersistenceStorage.sqlite(tmp_path / 'state.db')
    async with Persistence.open('cancelled-admission', storage=state) as persistence, Runtime.open(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        agent = runtime.agents.get()
        group = runtime._state.execution.executions.state_store.storage_group
        original = type(group).mutate
        async def mutate(owner, stores, callback):
            outer = active_state_scope() is None
            async def guarded(transaction):
                result = await callback(transaction)
                if outer and phase == 'before' and not faults and isinstance(result, ExecutionStartReservationResult) and result.created:
                    faults.append(result.execution.execution_id)
                    raise asyncio.CancelledError
                return result
            result = await original(owner, stores, guarded)
            if outer and phase == 'after' and not faults and isinstance(result, ExecutionStartReservationResult) and result.created:
                faults.append(result.execution.execution_id)
                raise asyncio.CancelledError
            return result
        with monkeypatch.context() as patch:
            patch.setattr(type(group), 'mutate', mutate)
            with pytest.raises(AIError) as caught:
                await agent.start('question', idempotency_key='cancelled')
        assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert len(faults) == 1 and calls == []
        execution_id = faults[0]
        current = await runtime._state.execution.executions.get(execution_id, tenant_id=runtime.tenant_id)
        if phase == 'before':
            assert current is None

        else:
            assert current.status is ExecutionStatus.STARTED and current.agent_run_seq == 0
            await runtime.executions.cancel(execution_id, CancelExecutionRequest(runtime.default_principal, 'settle-cancel'))
            result = await runtime.executions.result(execution_id, principal=runtime.default_principal)
            assert result.status is ExecutionStatus.CANCELLED
        assert calls == []


@pytest.mark.asyncio
async def test_prepared_attempt_receipt_matches_persisted_revision_and_payload(tmp_path: Path) -> None:
    from dataclasses import replace

    from linktools.ai.core import ExecutionStatus
    from linktools.ai.execution import ExecutionRecoveryDetail, ExecutionRecoveryState, RuntimePayloadRef
    from linktools.ai.persistence.contracts import AgentAttemptClaim
    from linktools.ai.persistence._runtime_commands import StateCommands
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.storage import StoredPayload

    from .test_history_projection_conformance import _record

    async with Persistence.open("prepared-receipt", storage=PersistenceStorage.sqlite(tmp_path / "receipt.db"),
                                tenant_id="tenant") as state:
        repository = state.execution.executions
        value = replace(_record(ExecutionStatus.STARTED, 0), revision=1, durable_trace=False,
                        recovery=ExecutionRecoveryDetail())
        await repository.create_execution(value)
        before = await repository.get_observed(value.execution_id, tenant_id="tenant")
        assert before is not None
        payload = RuntimePayloadRef(StoredPayload.inline_json({"prepared": True}), None)
        commands = StateCommands(repository, namespace="prepared-receipt", events=state.execution.events,
                                 checkpoint=state.checkpoint, background_tasks=set())
        receipt = await commands.commit_prepared_agent_attempt_checkpoint(
            AgentAttemptClaim(value.execution_id, 1, 0, ExecutionRecoveryState.ADMITTED),
            payload, observed=before,
        )
        assert receipt == await repository.get_observed(value.execution_id, tenant_id="tenant")
        assert receipt.storage_revision > before.storage_revision
        assert receipt.value.input_context == payload and receipt.value.agent_run_seq == 1
        assert receipt.value.producer_claim_id and receipt.value.producer_generation > value.producer_generation
        assert await repository.get_observed(value.execution_id, tenant_id="tenant", previous=receipt) is receipt
        forged = replace(receipt, value=replace(receipt.value, input_context=None))
        assert await repository.get_observed(value.execution_id, tenant_id="tenant", previous=forged) == receipt
        assert await repository.acquire_dependency_hold(value.execution_id, tenant_id="tenant", hold_id="hold")
        refreshed = await repository.get_observed(value.execution_id, tenant_id="tenant", previous=receipt)
        assert refreshed is not None
        assert refreshed.value.dependency_hold_ids == ("hold",)
        assert refreshed.storage_revision > receipt.storage_revision


@pytest.mark.asyncio
async def test_native_admission_rollback_does_not_dispatch_or_hold_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence._session_turn_sql import SqlSessionTurns

    calls, rejected = [], []

    async def model(messages, _info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("answer")])

    features = _session_features(model, checkpointed=True)
    async with (
        Persistence.open("startup-native-rollback", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state,
        Runtime.open(persistence=state, features=features, authorization=_SessionHostAuthorization()) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        before = await runtime.history.turn_session_state("session", principal=principal)
        original = SqlSessionTurns._accept_in_group

        async def rollback_after_write(store, transaction, prepared):
            assert await original(store, transaction, prepared)
            rejected.append(prepared.key)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        with monkeypatch.context() as patch:
            patch.setattr(SqlSessionTurns, "_accept_in_group", rollback_after_write)
            with pytest.raises(AIError) as rejected_start:
                await runtime.sessions.start(
                    "session", "question", agent="default", principal=principal, idempotency_key="same",
                )

        assert rejected_start.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        rejected_key, = rejected
        assert await state.session_turns.get_turn(rejected_key) is None
        assert await runtime.history.turn_session_state("session", principal=principal) == before
        assert calls == []
        turn = await runtime.sessions.start(
            "session", "question", agent="default", principal=principal, idempotency_key="same",
        )
        assert turn.key != rejected_key
        assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 1
