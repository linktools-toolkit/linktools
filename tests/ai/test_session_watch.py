#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session observers share source reads and preserve admission ownership."""

import asyncio
from contextlib import aclosing
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import Principal, SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_pending_session_projection import _runtimes

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('outcome', ['success', 'cancel', 'failure'])
async def test_watch_shares_reads_and_tracks_real_turn_release(monkeypatch, outcome):
    entered, finish = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        await finish.wait()
        if outcome == 'failure':
            raise RuntimeError('provider failed')
        return ModelResponse(parts=[TextPart('answer')])
    features = model_features(_Models(_UsageFunctionModel(model)))
    features.capability(ExecutionCapability(retain_history=True))
    storage = PersistenceStorage.in_memory()
    async with Runtime.open('watch', storage=storage, features=features) as runtime:
        session = await runtime.agents.get().create_session('session')
        other = await runtime.agents.get().create_session('other')
        reader = AsyncMock(wraps=runtime._session_service._read_session_record)
        monkeypatch.setattr(runtime._session_service, '_read_session_record', reader)
        policy = AsyncMock(wraps=runtime._session_service.require_available)
        monkeypatch.setattr(runtime._session_service, 'require_available', policy)
        monkeypatch.setattr(runtime._state.execution.executions, 'get_view_fresh',
            AsyncMock(side_effect=AssertionError('occupancy must not hydrate presentation')))
        async with aclosing(session.watch()) as one, aclosing(session.watch()) as two:
            initial = await asyncio.gather(anext(one), anext(two))
            assert all(view.active_execution_id is None for view in initial)
            assert reader.await_count == 1
            assert policy.await_count == 1
            pending = [asyncio.create_task(anext(one)), asyncio.create_task(anext(two))]
            await asyncio.sleep(.05)
            assert reader.await_count == 1
            await other.close(idempotency_key='close-other')
            await asyncio.sleep(.05)
            assert reader.await_count == 1
            execution = await session.start('hello', idempotency_key='turn')
            await asyncio.wait_for(entered.wait(), 5)
            busy = await asyncio.wait_for(asyncio.gather(*pending), 5)
            assert all(view.active_execution_id == execution.execution_id for view in busy)
            pending = [asyncio.create_task(anext(one)), asyncio.create_task(anext(two))]
            if outcome == 'cancel':
                await execution.cancel(idempotency_key='cancel')
            else:
                finish.set()
            await execution.wait(timeout_seconds=5)
            idle = await asyncio.wait_for(asyncio.gather(*pending), 5)
            assert all(view.active_execution_id is None for view in idle)
            reads = reader.await_count
            await asyncio.sleep(.05)
            assert reader.await_count == reads
            await session.close(idempotency_key='close')
            closed = await anext(one)
            if closed.status is SessionStatus.CLOSING:
                closed = await asyncio.wait_for(anext(one), 5)
            assert closed.status is SessionStatus.CLOSED


async def test_watch_pending_admission_and_completion(tmp_path):
    async with _runtimes(tmp_path, cache=InMemoryCacheBackend(max_bytes=2**24), shared=False) as (values, entered, finish, block):
        runtime, state = values[0]
        session = await runtime.agents.get().create_session('session')
        async with aclosing(session.watch()) as stream:
            assert (await anext(stream)).active_execution_id is None
            busy = asyncio.create_task(anext(stream))
            block()
            execution = await session.start('hello', idempotency_key='pending')
            await asyncio.wait_for(entered.wait(), 5)
            assert (await asyncio.wait_for(busy, 5)).active_execution_id == execution.execution_id
            assert await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
            idle = asyncio.create_task(anext(stream))
            finish.set()
            await execution.wait(timeout_seconds=5)
            assert (await asyncio.wait_for(idle, 5)).active_execution_id is None


async def test_watch_authorizes_each_observation_and_isolates_tenant():
    from ._runtime_test_helpers import RuntimeUsageModels
    async with Runtime.open('auth-watch', storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        session = await runtime.agents.get().create_session('session')
        for actor in (Principal('other', runtime.tenant_id, 'service'), Principal('other', 'other', 'service')):
            async with aclosing(runtime.sessions.watch('session', principal=actor)) as stream:
                with pytest.raises(AIError) as caught:
                    await anext(stream)
                assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert not runtime._session_service._watches
        async with aclosing(session.watch()) as stream:
            await anext(stream)


async def test_watch_does_not_publish_outer_transaction_rollback():
    from dataclasses import replace
    from ._runtime_test_helpers import RuntimeUsageModels
    storage = PersistenceStorage.in_memory()
    async with Runtime.open('watch-rollback', storage=storage,
                            features=model_features(RuntimeUsageModels())) as runtime:
        session = await runtime.agents.get().create_session('session')
        repository = runtime._state.conversation.sessions
        async with aclosing(session.watch()) as stream:
            await anext(stream)
            waiting = asyncio.create_task(anext(stream))
            async def mutate(transaction):
                current = await repository.get('session', tenant_id=runtime.tenant_id)
                await repository.compare_and_swap('session', tenant_id=runtime.tenant_id,
                    expected_revision=current.revision,
                    next_record=replace(current, status=SessionStatus.CLOSED, revision=current.revision+1))
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            with pytest.raises(AIError) as failure:
                await repository.state_store.mutate(mutate)
            assert failure.value.code is ErrorCode.STORAGE_CONFLICT
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(asyncio.shield(waiting), .05)
            await session.close(idempotency_key='close')
            closed = await asyncio.wait_for(waiting, 5)
            if closed.status is SessionStatus.CLOSING:
                closed = await asyncio.wait_for(anext(stream), 5)
            assert closed.status is SessionStatus.CLOSED


async def test_watch_rechecks_authorization_and_closes_with_runtime(monkeypatch):
    from ._runtime_test_helpers import RuntimeUsageModels
    async with Runtime.open('watch-auth', storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        session = await runtime.agents.get().create_session('session')
        async with aclosing(session.watch()) as stream:
            await anext(stream)
            class Deny:
                async def authorize(self, *args):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            monkeypatch.setattr(runtime._session_service, '_runtime_authorization', Deny())
            await session.close(idempotency_key='close')
            with pytest.raises(AIError) as caught:
                await anext(stream)
            assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    async with Runtime.open('watch-close', storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        session = await runtime.agents.get().create_session('session')
        stream = session.watch()
        await anext(stream)
    with pytest.raises(AIError) as caught:
        await anext(stream)
    assert caught.value.code is ErrorCode.STORAGE_CLOSED
    await stream.aclose()


@pytest.mark.parametrize('kind', ['retry', 'fork'])
async def test_watch_observes_execution_reentry_without_session_start(kind):
    from linktools.ai.execution import RetryExecutionRequest, ForkExecutionRequest
    entered, finish = asyncio.Event(), asyncio.Event()
    count = 0
    async def model(messages, info):
        nonlocal count
        count += 1
        if count == 1 and kind == 'retry':
            raise RuntimeError('initial attempt fails')
        if count > 1:
            entered.set()
            await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])
    features = model_features(_Models(_UsageFunctionModel(model)))
    features.capability(ExecutionCapability(retain_history=True))
    async with Runtime.open('watch-reentry', storage=PersistenceStorage.in_memory(), features=features) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('initial', idempotency_key='initial')
        await execution.wait(timeout_seconds=5)
        async with aclosing(session.watch()) as stream:
            assert (await anext(stream)).active_execution_id is None
            waiting = asyncio.create_task(anext(stream))
            request = (RetryExecutionRequest if kind=='retry' else ForkExecutionRequest)('again', runtime.default_principal, kind)
            again = await getattr(runtime.executions, kind)(execution.execution_id, request)
            await asyncio.wait_for(entered.wait(), 5)
            assert (await asyncio.wait_for(waiting, 5)).active_execution_id == again.execution_id
            waiting = asyncio.create_task(anext(stream))
            finish.set()
            await again.wait(timeout_seconds=5)
            assert (await asyncio.wait_for(waiting, 5)).active_execution_id is None


async def test_watch_rejects_missing_durable_admission():
    from dataclasses import replace
    from ._runtime_test_helpers import RuntimeUsageModels
    async with Runtime.open('watch-integrity', storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        session = await runtime.agents.get().create_session('session')
        repository = runtime._state.conversation.sessions
        current = await repository.get('session', tenant_id=runtime.tenant_id)
        await repository.compare_and_swap('session', tenant_id=runtime.tenant_id,
            expected_revision=current.revision, next_record=replace(current, active_execution_id='missing'))
        async with aclosing(session.watch()) as stream:
            with pytest.raises(AIError) as caught:
                await anext(stream)
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_memory_commit_observer_failure_does_not_change_commit():
    from dataclasses import replace
    from ._runtime_test_helpers import RuntimeUsageModels
    async with Runtime.open('watch-observer', storage=PersistenceStorage.in_memory(),
                            features=model_features(RuntimeUsageModels())) as runtime:
        await runtime.agents.get().create_session('session')
        repository = runtime._state.conversation.sessions
        observed = []
        def broken(committed):
            observed.append(committed)
            raise RuntimeError('observer failure')
        async def mutate(transaction):
            transaction.observe_mutation(broken)
            current = await repository.get('session', tenant_id=runtime.tenant_id)
            return await repository.compare_and_swap('session', tenant_id=runtime.tenant_id,
                expected_revision=current.revision, next_record=replace(current, metadata={'saved': True}))
        saved = await repository.state_store.mutate(mutate)
        assert saved.metadata == {'saved': True}
        assert observed == [True]
        assert (await repository.get('session', tenant_id=runtime.tenant_id)).metadata == {'saved': True}


async def test_host_invalidations_follow_sync_turn_without_reading_state(monkeypatch):
    from linktools.ai.core import CommitPolicy, ExecutionStatus

    entered, finish = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = model_features(_Models(_UsageFunctionModel(model)))
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()))
    storage = PersistenceStorage.in_memory()
    invalidated, failed = asyncio.Event(), asyncio.Event()
    async with Runtime.open("invalidations", storage=storage, features=features) as runtime:
        session = await runtime.agents.get().create_session("session")
        other = await runtime.agents.get().create_session("other")
        state_capture = AsyncMock(side_effect=AssertionError("host invalidations must not read a view"))
        monkeypatch.setattr(runtime._session_service, "_observe_changes", state_capture)
        unsubscribe = runtime.sessions.observe_changes("session", invalidated.set)

        def fail_once():
            remove_bad()
            failed.set()
            raise RuntimeError("host callback failure does not stop other observers")

        remove_bad = runtime.sessions.observe_changes("session", fail_once)
        await other.close(idempotency_key="other")
        assert not invalidated.is_set() and not failed.is_set()

        execution = await session.start("hello", idempotency_key="send")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await asyncio.wait_for(invalidated.wait(), 5)
            await asyncio.wait_for(failed.wait(), 5)
            invalidated.clear()
        finally:
            finish.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        await asyncio.wait_for(invalidated.wait(), 5)
        unsubscribe()
        invalidated.clear()
        unsubscribed = runtime.sessions.observe_changes("session", invalidated.set)
        unsubscribe()  # A stale unsubscribe must not remove the newer observer.
        await session.close(idempotency_key="close")
        assert invalidated.is_set()
        invalidated.clear()
        assert state_capture.await_count == 0
        repository = runtime._state.conversation.sessions

    # Runtime teardown owns every registration, be it still open or removed.
    repository.notify_session_change("session", tenant_id=runtime.tenant_id)
    assert not invalidated.is_set()
    unsubscribed()
    with pytest.raises(AIError) as closed:
        runtime.sessions.observe_changes("session", invalidated.set)
    assert closed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
