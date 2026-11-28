#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Default management owners remain independent of retained conversation."""
from ._runtime_test_helpers import model_features

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import SessionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistencePlan, PersistenceStorage
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel


class _HostAuthorization(TenantAuthorizationPolicy):
    """Supply the explicit authorization owner required by native SQL sessions."""


def _features(*capabilities, retain_history=False):
    features = RuntimeFeatures('owner-contract', defaults={"execution": {"retain_history": retain_history}})
    features.agent('default', model='default', allow_tools=())
    for capability in capabilities:
        features.capability(capability, scope='runtime')
    return features


def _models():
    async def model(messages, info):
        return ModelResponse(parts=[TextPart('answer')])
    return _Models(_UsageFunctionModel(model))


@pytest.mark.asyncio
@pytest.mark.parametrize('explicit', [False, True])
async def test_default_session_owner_and_explicit_replacement_work_without_conversation(tmp_path, explicit):
    declarations = (SessionCapability(),) if explicit else ()
    async with Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(tmp_path / 'state.db')) as persistence_36, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_36, features=model_features(_models(), _features(*declarations))) as runtime:
        session = await runtime.agents.get().create_session('session')
        turn = await runtime.agents.get().start('question', session_id=session.session_id)
        result = await turn.wait_final(timeout_seconds=5)
        assert result.view['output'] == {'text': 'answer'}
        assert (await runtime.sessions.get("session", principal=runtime.default_principal)).session_id == 'session'
        steps = await persistence_36.create_step_store()
        try:
            assert await steps.latest_snapshot(run_id=turn.key.turn_id) is None
        finally:
            await steps.close()


@pytest.mark.asyncio
async def test_disabled_session_does_not_disable_independent_execution_or_cold_result(tmp_path):
    calls = []

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('answer')])

    features = _features(SessionCapability(enabled=False))
    models = _Models(_UsageFunctionModel(model))
    path = tmp_path / 'state.db'
    async with Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(path)) as persistence_56, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_56, features=model_features(models, features)) as runtime:
        agent = runtime.agents.get()
        with pytest.raises(AIError) as disabled:
            await agent.create_session('session')
        assert disabled.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        with pytest.raises(AIError) as bound:
            await agent.start('question', session_id='session')
        assert bound.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        execution = await agent.start('question', idempotency_key='request')
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.output == {'text': 'answer'}
        assert result.usage.model_requests > 0
        execution_id = execution.execution_id
    async with Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(path)) as persistence_70, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_70, features=model_features(models, features)) as runtime:
        restored = await runtime.executions.result(execution_id, principal=runtime.default_principal)
        assert restored.output == result.output and restored.usage == result.usage
        same = await runtime.agents.get().start('question', idempotency_key='request')
        assert same.execution_id == execution_id
        assert len(calls) == 1
        with pytest.raises(AIError) as conflict:
            await runtime.agents.get().start('changed', idempotency_key='request')
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
async def test_same_live_source_binds_two_runtimes_without_owner_leakage(tmp_path):
    session_policy = SessionCapability(enabled=False)
    features = _features(session_policy)
    models = _models()
    features.model("default", models)
    capture = features.borrow()
    features.capability(TraceCapability(), scope='runtime')
    first_storage = PersistenceStorage.sqlite(tmp_path / 'first.db')
    second_storage = PersistenceStorage.sqlite(tmp_path / 'second.db')
    async with Persistence.open('owner-contract', storage=first_storage) as persistence_93, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_93, features=capture) as first, Persistence.open('owner-contract', storage=second_storage) as persistence_94, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_94, features=capture) as second:
        for runtime in (first, second):
            with pytest.raises(AIError) as disabled:
                await runtime.agents.get().create_session('session')
            assert disabled.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        one = await first.agents.get().start('first', idempotency_key='shared-key')
        two = await second.agents.get().start('second', idempotency_key='shared-key')
        assert (await one.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        assert (await two.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        for storage, runtime, execution in ((first_storage, first, one), (second_storage, second, two)):
            record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert record is not None and not record.conversation
    assert session_policy.enabled is False
    async with Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(tmp_path / 'default.db')) as persistence_110, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_110, features=model_features(models, _features())) as default:
        session = await default.agents.get().create_session('session')
        assert (await default.sessions.get("session", principal=default.default_principal)).session_id == 'session'


@pytest.mark.asyncio
async def test_disabling_session_owner_blocks_existing_session_read_and_resume(tmp_path):
    path = tmp_path / 'state.db'
    async with Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(path)) as persistence_119, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_119, features=model_features(_models(), _features())) as runtime:
        await runtime.agents.get().create_session('existing')
    async with Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(path)) as persistence_122, Runtime.open(authorization=_HostAuthorization(), persistence=persistence_122, features=model_features(_models(), _features(SessionCapability(enabled=False)))) as runtime:
        with pytest.raises(AIError) as unavailable:
            await runtime.sessions.get('existing', principal=runtime.default_principal)
        assert unavailable.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        with pytest.raises(AIError) as unavailable:
            await runtime.agents.get().session('existing').start('question')
        assert unavailable.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING



@pytest.mark.asyncio
async def test_session_checkpoints_do_not_enable_step_for_standalone_execution(tmp_path):
    features = model_features(_models(), _features(SessionCapability(checkpoints=True)))
    async with (
        Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(tmp_path / 'state.db')) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        agent = runtime.agents.get()
        session = await agent.create_session('session')
        turn = await agent.start('session question', session_id=session.session_id)
        assert (await turn.wait_final(timeout_seconds=5)).view['output'] == {'text': 'answer'}
        row = await state.session_turn_reader.get_turn(turn.key)
        assert row['request_json']['session_checkpoints'] is True

        steps = await state.create_step_store()
        try:
            checkpoint = await steps.latest_snapshot(run_id=turn.key.turn_id)
            assert checkpoint is not None and checkpoint.state == 'complete'
        finally:
            await steps.close()

        execution = await agent.start('independent question')
        assert (await execution.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        record = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and not record.restart_checkpoints


@pytest.mark.asyncio
async def test_standalone_step_does_not_enable_checkpoints_for_session(tmp_path):
    features = model_features(_models(), _features(StepPersistenceCapability()))
    async with (
        Persistence.open('owner-contract', storage=PersistenceStorage.sqlite(tmp_path / 'state.db')) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        agent = runtime.agents.get()
        session = await agent.create_session('session')
        turn = await agent.start('session question', session_id=session.session_id)
        assert (await turn.wait_final(timeout_seconds=5)).view['output'] == {'text': 'answer'}
        row = await state.session_turn_reader.get_turn(turn.key)
        assert row['request_json']['session_checkpoints'] is False
        steps = await state.create_step_store()
        try:
            assert await steps.latest_snapshot(run_id=turn.key.turn_id) is None
        finally:
            await steps.close()

        execution = await agent.start('independent question')
        assert (await execution.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        record = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.restart_checkpoints


@pytest.mark.asyncio
async def test_session_checkpoints_use_the_configured_split_sql_for_plain_chat(tmp_path):
    execution = StorageRoute.sqlite(tmp_path / 'execution.db')
    conversation = StorageRoute.sqlite(tmp_path / 'conversation.db')
    storage = PersistenceStorage(PersistencePlan(conversation=conversation, execution=execution, checkpoint=execution))
    features = model_features(_models(), _features(SessionCapability(checkpoints=True)))
    async with Runtime.open('owner-contract', storage=storage, features=features,
                            authorization=_HostAuthorization()) as runtime:
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session('session', principal=principal)
        turn = await runtime.sessions.start(
            'session', 'question', agent='default', principal=principal, idempotency_key='send',
        )
        assert (await turn.wait_final(timeout_seconds=5)).view["output"] == {"text": "answer"}
        record = await runtime._state.session_turns.get_turn(turn.key)
        assert record["request_json"]["split_step"] is True



@pytest.mark.asyncio
async def test_session_checkpoints_reject_memory_without_disabling_standalone_execution():
    features = model_features(_models(), _features(SessionCapability(checkpoints=True)))
    async with Runtime.open('owner-contract', storage=PersistenceStorage.in_memory(), features=features,
                            authorization=_HostAuthorization()) as runtime:
        agent = runtime.agents.get()
        with pytest.raises(AIError) as unsupported:
            await agent.create_session('session')
        assert unsupported.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        execution = await agent.start('independent question')
        assert (await execution.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
