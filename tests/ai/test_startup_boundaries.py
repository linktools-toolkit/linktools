#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""First attempts claim authority only after prepared input is durable."""

import asyncio
from pathlib import Path

import pytest
from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.capability import ConversationCapability
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.storage._database import SqlStorageContext
from pydantic_ai.messages import ModelResponse, TextPart

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
async def test_first_provider_has_three_metadata_commits_and_original_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    async with Runtime.open('startup-boundaries', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        original = SqlStorageContext.run_mutation
        commits = []
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get('domain'))
            return result
        with monkeypatch.context() as patch:
            patch.setattr(SqlStorageContext, 'run_mutation', count)
            execution = await session.start('question', idempotency_key='same')
            try:
                await asyncio.wait_for(entered.wait(), 5)
                current = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                head = await state.execution.executions.get_history_head(execution.execution_id, tenant_id=runtime.tenant_id)
                assert current.status is ExecutionStatus.STARTED
                assert current.revision == 3 and current.agent_run_seq == 1
                assert current.input_context is not None
                assert head.producer_generation == 2 and head.producer_claim_id is not None
                assert len(commits) == 3
                replay = await session.start('question', idempotency_key='same')
                assert replay.execution_id == execution.execution_id
                assert len(calls) == 1
            finally:
                release.set()
                await execution.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('memory', 'filesystem', 'sqlite'))
@pytest.mark.parametrize('phase', ('before', 'after'))
async def test_startup_unknown_commit_never_repeats_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, phase: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime.state._contracts import (
        ExecutionRecord,
        RecoveryCheckpoint,
    )
    from linktools.ai.runtime.state._store import active_state_scope
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-ack')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = (RuntimeStorage.in_memory() if backend == 'memory' else
             RuntimeStorage.filesystem(tmp_path / 'state') if backend == 'filesystem' else
             RuntimeStorage.sqlite(tmp_path / 'state.db'))
    async with Runtime.open('startup-ack', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        group_type = type(state.execution.executions.state_store.storage_group)
        original = group_type.mutate
        faults = []
        def matches(value):
            return (isinstance(value, tuple) and len(value) == 2
                    and isinstance(value[0], ExecutionRecord) and isinstance(value[1], RecoveryCheckpoint))
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
            execution = await session.start('question', idempotency_key='same')
            assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        assert faults == [True] and len(calls) == 1
        replay = await session.start('question', idempotency_key='same')
        assert replay.execution_id == execution.execution_id
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_two_runtimes_share_one_first_attempt_after_parallel_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend
    prepared = asyncio.Event()
    release = asyncio.Event()
    preparations = []
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-race')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    first_state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    second_state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    async with Runtime.open('startup-race', storage=first_state, models=models, features=group) as first, Runtime.open(
        'startup-race', storage=second_state, models=models, features=group
    ) as second:
        await first.agents.get().create_session('session')
        original = LocalExecutionBackend._prepare_input_context
        async def prepare(backend, context):
            value = await original(backend, context)
            preparations.append(backend)
            if len(preparations) == 2:
                prepared.set()
            await release.wait()
            return value
        with monkeypatch.context() as patch:
            patch.setattr(LocalExecutionBackend, '_prepare_input_context', prepare)
            one = await first.agents.get().session('session').start('question', idempotency_key='same')
            two = await second.agents.get().session('session').start('question', idempotency_key='same')
            assert one.execution_id == two.execution_id
            try:
                await asyncio.wait_for(prepared.wait(), 5)
            finally:
                release.set()
            results = await asyncio.wait_for(asyncio.gather(one.wait(), two.wait()), 5)
            assert all(value.result.status is ExecutionStatus.SUCCEEDED for value in results)
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_busy_admission_keeps_rejected_key_after_owner_finishes(tmp_path: Path) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-busy')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    first_state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    second_state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    async with Runtime.open('startup-busy', storage=first_state, models=models, features=group) as first, Runtime.open(
        'startup-busy', storage=second_state, models=models, features=group
    ) as second:
        await first.agents.get().create_session('session')
        one = await first.agents.get().session('session').start('question', idempotency_key='owner')
        try:
            await asyncio.wait_for(entered.wait(), 5)
            with pytest.raises(AIError) as busy:
                await second.agents.get().session('session').start('other', idempotency_key='rejected')
            assert busy.value.code is ErrorCode.SESSION_BUSY
        finally:
            release.set()
        await one.wait()
        with pytest.raises(AIError) as replay:
            await second.agents.get().session('session').start('other', idempotency_key='rejected')
        assert replay.value.code is ErrorCode.SESSION_BUSY
        records = await second_state.execution.executions.list_by_session('session', tenant_id=second.tenant_id)
        assert len(records) == 2
        assert {value.status for value in records} == {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED}
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_cancel_during_context_preparation_never_calls_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend
    preparing = asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-cancel')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    async with Runtime.open('startup-cancel', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        async def prepare(backend, context):
            preparing.set()
            await asyncio.Future()
        with monkeypatch.context() as patch:
            patch.setattr(LocalExecutionBackend, '_prepare_input_context', prepare)
            execution = await session.start('question', idempotency_key='same')
            await asyncio.wait_for(preparing.wait(), 5)
            await execution.cancel(idempotency_key='cancel')
            result = await asyncio.wait_for(execution.wait(), 5)
        assert result.result.status is ExecutionStatus.CANCELLED
        assert calls == []
        observed = await state.conversation.sessions.get('session', tenant_id=runtime.tenant_id)
        assert observed.active_execution_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize('fault', ('object', 'context', 'timeout'))
async def test_preparation_failure_stays_unclaimed_and_can_be_explicitly_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime import _local
    from linktools.ai.runtime._local import LocalExecutionBackend
    from linktools.ai.runtime.state._store import active_state_scope
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-preparation-failure')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    async with Runtime.open('startup-preparation-failure', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        observed = []
        async def fail(*args, **kwargs):
            assert active_state_scope() is None
            observed.append(True)
            if fault == 'timeout':
                raise TimeoutError('preparation deadline')
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE if fault == 'object' else ErrorCode.EXECUTION_HISTORY_UNAVAILABLE)
        with monkeypatch.context() as patch:
            if fault == 'object':
                patch.setattr(_local, 'payload_fits_inline', lambda *_: False)
                patch.setattr(_local, 'put_runtime_object', fail)
            elif fault == 'context':
                patch.setattr(LocalExecutionBackend, 'load_input_context', fail)
            else:
                patch.setattr(LocalExecutionBackend, '_prepare_input_context', fail)
            execution = await session.start('question', idempotency_key='same')
            with pytest.raises(AIError):
                await asyncio.wait_for(execution.wait(), 5)
        assert observed and calls == []
        current = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert current.agent_run_seq == 0 and current.input_context is None
        await execution.cancel(force=True, idempotency_key='explicit-cancel')
        current = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert current.status is ExecutionStatus.CANCELLED
        assert (await state.conversation.sessions.get('session', tenant_id=runtime.tenant_id)).active_execution_id is None


@pytest.mark.asyncio
async def test_remote_cancel_during_preparation_settles_without_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend
    preparing = asyncio.Event()
    release = asyncio.Event()
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('startup-remote-cancel')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    first_state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    second_state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    async with Runtime.open('startup-remote-cancel', storage=first_state, models=models, features=group) as first, Runtime.open(
        'startup-remote-cancel', storage=second_state, models=models, features=group
    ) as second:
        await first.agents.get().create_session('session')
        original = LocalExecutionBackend._prepare_input_context
        async def prepare(backend, context):
            reference = await original(backend, context)
            preparing.set()
            await release.wait()
            return reference
        with monkeypatch.context() as patch:
            patch.setattr(LocalExecutionBackend, '_prepare_input_context', prepare)
            execution = await first.agents.get().session('session').start('question', idempotency_key='same')
            await asyncio.wait_for(preparing.wait(), 5)
            remote = await second.executions.get(execution.execution_id)
            try:
                await remote.cancel(idempotency_key='remote-cancel')
            finally:
                release.set()
            assert (await asyncio.wait_for(execution.wait(), 5)).result.status is ExecutionStatus.CANCELLED
        assert calls == []
        assert (await first_state.conversation.sessions.get('session', tenant_id=first.tenant_id)).active_execution_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize('member', ('execution', 'execution_history_head', 'recovery_checkpoint'))
async def test_prepared_attempt_member_conflict_rolls_back_authority_and_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member: str,
) -> None:
    from dataclasses import replace

    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime.state._contracts import (
        AgentAttemptClaim,
        RecoveryCheckpoint,
        RecoveryCheckpointState,
        RuntimePayloadRef,
    )
    from linktools.ai.runtime.state._runtime_commands import RuntimeStateCommands
    from linktools.ai.runtime.state._sql import _SqlTransaction
    from linktools.ai.storage import StoredPayload
    from sqlalchemy import update

    from .test_checkpoint_write_batches import _snapshot
    from .test_history_projection_conformance import _record

    state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    await state.initialize(namespace='startup-member-cas', tenant_id='tenant')
    try:
        execution = replace(_record(ExecutionStatus.STARTED, 0), revision=1, durable_trace=False)
        await state.execution.executions.create_with_history_head(execution)
        checkpoint = RecoveryCheckpoint(execution.execution_id, None, RecoveryCheckpointState.ADMITTED,
                                        0, execution.created_at, execution.updated_at)
        await state.recovery.checkpoints.create(checkpoint)
        commands = RuntimeStateCommands(state.execution.executions, namespace='startup-member-cas',
                                        events=state.execution.events, recovery=state.recovery.checkpoints,
                                        background_tasks=set())
        stores = (state.execution.executions.state_store, state.recovery.checkpoints.state_store)
        before = [await _snapshot(store) for store in stores]
        original = _SqlTransaction.replace_records
        conflicts = []
        async def conflict(transaction, replacements):
            target = next((item for item in replacements if item.record.kind == member), None)
            if target is not None:
                table = transaction._table('ai_state_records')
                await transaction._execute(update(table).where(
                    table.c.store_digest == transaction._store_hex,
                    table.c.key_digest == target.record.key_digest.hex(),
                ).values(storage_revision=target.expected_storage_revision + 1))
                conflicts.append(member)
            return await original(transaction, replacements)
        with monkeypatch.context() as patch:
            patch.setattr(_SqlTransaction, 'replace_records', conflict)
            with pytest.raises(AIError) as caught:
                await commands.commit_prepared_agent_attempt_checkpoint(
                    AgentAttemptClaim(execution.execution_id, 1, 0, 0, RecoveryCheckpointState.ADMITTED),
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
    from linktools.ai.runtime._local import LocalExecutionBackend

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
    group = RuntimeFeatures('startup-budget-review')
    group.capability(ConversationCapability(), scope="runtime")
    group.agent('default', model='default', allow_tools=())
    state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    original = LocalExecutionBackend._prepare_input_context
    async def prepare(backend, context):
        reference = await original(backend, context)
        preparing.set()
        await release.wait()
        return reference
    monkeypatch.setattr('linktools.ai.runtime.state._budget.datetime', BudgetClock)
    monkeypatch.setattr(LocalExecutionBackend, '_prepare_input_context', prepare)
    async with Runtime.open('startup-budget-review', storage=state, models=_Models(_UsageFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('question', idempotency_key='key', budget=RunBudget(deadline_at=now + timedelta(seconds=30)))
        await asyncio.wait_for(preparing.wait(), 5)
        before = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert before.agent_run_seq == 0 and before.input_context is None
        clock[0] = now + timedelta(seconds=60)
        release.set()
        result = (await asyncio.wait_for(execution.wait(), 5)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
        usage = await execution.budget_usage()
        assert usage.model_requests == usage.in_flight_model_requests == usage.unknown_model_requests == 0
        assert calls == []
        assert (await state.conversation.sessions.get('session', tenant_id=runtime.tenant_id)).active_execution_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize('partial', ['missing', 'unbound'])
async def test_admission_readback_rejects_real_session_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, partial: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime.state._contracts import ExecutionRecord
    from linktools.ai.runtime.state._store import active_state_scope

    calls, faults = [], []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('not expected')])
    capabilities = RuntimeFeatures('partial-review')
    capabilities.capability(ConversationCapability(), scope="runtime")
    capabilities.agent('default', model='default', allow_tools=())
    state = RuntimeStorage.sqlite(tmp_path / 'state.db')
    async with Runtime.open('partial-review', storage=state, models=_Models(_UsageFunctionModel(model)), features=capabilities) as runtime:
        session = await runtime.agents.get().create_session('session')
        group = state.execution.executions.state_store.storage_group
        repository = state.conversation.sessions
        original = type(group).mutate
        async def mutate(owner, stores, callback):
            outer = active_state_scope() is None
            result = await original(owner, stores, callback)
            if owner is group and outer and not faults and isinstance(result, ExecutionRecord) and result.status is ExecutionStatus.STARTED and result.agent_run_seq == 0:
                faults.append(result.execution_id)
                if partial == 'unbound':
                    await repository.release_execution('session', tenant_id=runtime.tenant_id, execution_id=result.execution_id)
                else:
                    async def remove(transaction):
                        key = repository._key('session', 'session')
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
        current = await state.execution.executions.get(faults[0], tenant_id=runtime.tenant_id)
        assert current.status is ExecutionStatus.STARTED and current.agent_run_seq == 0
        value = await repository.get('session', tenant_id=runtime.tenant_id)
        assert value is None if partial == 'missing' else value.active_execution_id is None
