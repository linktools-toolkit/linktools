#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in sessions keep authority independent of optional context retention."""

import asyncio
from copy import deepcopy

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, RuntimeFeatures
from linktools.ai.core import Principal, SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ListSessionRequest, Runtime, RuntimeContext, RuntimeStorage, RuntimeStoragePlan
from linktools.ai.storage import StorageRoute
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.core import CommitPolicy

from ._runtime_test_helpers import _Models, _UsageFunctionModel


_OWNER = Principal('owner', 'tenant')


def _features(*, retain=False):
    features = RuntimeFeatures('builtin-session')
    features.agent('default', model='default', allow_tools=())
    if retain:
        features.capability(ExecutionCapability(retain_history=True), scope='runtime')
    return features


def _storage(path, backend='sqlite', cache=None):
    if backend == 'split_sqlite':
        return RuntimeStorage.from_plan(RuntimeStoragePlan(
            conversation=StorageRoute.sqlite(path / 'conversation.db', cache=cache),
            execution=StorageRoute.sqlite(path / 'execution.db', cache=cache),
        ))
    return (RuntimeStorage.sqlite(path / 'state.db', cache=cache) if backend == 'sqlite'
            else RuntimeStorage.filesystem(path / 'state', cache=cache))


def _open(path, model, *, retain=False, backend='sqlite', cache=None):
    return Runtime.open('builtin-session', storage=_storage(path, backend, cache),
                        models=_Models(_UsageFunctionModel(model)),
                        features=_features(retain=retain),
                        context=RuntimeContext(None, tenant_id='tenant'))


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['sqlite', 'filesystem'])
async def test_builtin_metadata_lifecycle_survives_feature_and_cache_changes(tmp_path, backend):
    async def model(messages, info):
        raise AssertionError('metadata operations must not execute a model')

    async with _open(tmp_path, model, backend=backend) as runtime:
        session = await runtime.agents.get().create_session('session', principal=_OWNER,
                                                          metadata={'title': 'first'}, idempotency_key='create')
        replay = await runtime.agents.get().create_session('session', principal=_OWNER,
                                                         metadata={'title': 'first'}, idempotency_key='create')
        assert replay.session_id == session.session_id
        view = await runtime.sessions.reconcile('session', principal=_OWNER)
        updated = await session.update(expected_revision=view.revision, metadata={'title': 'second'},
                                       idempotency_key='update')
        with pytest.raises(AIError) as stale:
            await session.update(expected_revision=view.revision, metadata={'title': 'stale'},
                                 idempotency_key='stale')
        assert stale.value.code is ErrorCode.SESSION_REVISION_CONFLICT
        assert updated.metadata['title'] == 'second'
        with pytest.raises(AIError) as foreign:
            await runtime.sessions.get('session', principal=Principal('owner', 'other-tenant'))
        assert foreign.value.code is ErrorCode.AUTHORIZATION_DENIED

    async with _open(tmp_path, model, backend=backend, retain=True,
                     cache=InMemoryCacheBackend(max_bytes=1_000_000)) as runtime:
        page = await runtime.sessions.list(ListSessionRequest(principal=_OWNER))
        assert [(item.session_id, item.metadata['title']) for item in page.items] == [('session', 'second')]
        session = await runtime.sessions.get('session', principal=_OWNER)
        assert (await session.close(idempotency_key='close')).status is SessionStatus.CLOSED

    async with _open(tmp_path, model, backend=backend) as runtime:
        assert (await runtime.sessions.reconcile('session', principal=_OWNER)).status is SessionStatus.CLOSED


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ['sqlite', 'filesystem', 'split_sqlite'])
async def test_unretained_first_turn_releases_session_but_cannot_resume_after_reopen(tmp_path, backend):
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart('unretained answer')])

    async with _open(tmp_path, model, backend=backend) as runtime:
        session = await runtime.agents.get().create_session('session', principal=_OWNER)
        execution = await session.start('first', idempotency_key='first')
        assert (await execution.wait(timeout_seconds=5)).result.status.value == 'SUCCEEDED'
        view = await runtime.sessions.reconcile('session', principal=_OWNER)
        assert view.active_execution_id is None
        assert view.history_quality == 'unavailable'

    async with _open(tmp_path, model, backend=backend, retain=True) as runtime:
        session = await runtime.sessions.get('session', principal=_OWNER)
        assert (await runtime.sessions.reconcile('session', principal=_OWNER)).history_quality == 'unavailable'
        for operation in (lambda: session.start('must not forget', idempotency_key='second'),
                          lambda: session.fork('fork', idempotency_key='fork')):
            with pytest.raises(AIError) as missing:
                await operation()
            assert missing.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
        assert len(calls) == 1
        assert (await session.close()).status is SessionStatus.CLOSED


@pytest.mark.asyncio
async def test_new_unretained_turn_invalidates_older_retained_cursor(tmp_path):
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart('answer')])

    async with _open(tmp_path, model, retain=True) as runtime:
        session = await runtime.agents.get().create_session('session', principal=_OWNER)
        first = await session.start('retained question')
        await first.wait(timeout_seconds=5)
        assert (await session.history()).items

    async with _open(tmp_path, model) as runtime:
        session = await runtime.sessions.get('session', principal=_OWNER)
        assert (await session.history()).items
        second = await session.start('unretained question')
        assert (await second.wait(timeout_seconds=5)).result.status.value == 'SUCCEEDED'
        assert len(calls[1]) > len(calls[0])
        assert (await runtime.sessions.reconcile('session', principal=_OWNER)).history_quality == 'unavailable'

    async with _open(tmp_path, model, retain=True) as runtime:
        session = await runtime.sessions.get('session', principal=_OWNER)
        assert (await session.history()).items
        historical = await runtime.executions.get(first.execution_id, principal=_OWNER)
        for operation in (lambda: session.start('do not reuse old cursor'), lambda: session.fork('fork'),
                          lambda: historical.fork('do not bypass unavailable session')):
            with pytest.raises(AIError) as missing:
                await operation()
            assert missing.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
        assert len(calls) == 2


class _ClaimCache(InMemoryCacheBackend):
    def __init__(self):
        super().__init__(max_bytes=4_000_000)
        self.claims = 0

    async def claim_turn(self, *args, **kwargs):
        self.claims += 1
        return await super().claim_turn(*args, **kwargs)


@pytest.mark.asyncio
async def test_coordination_does_not_opt_default_execution_into_terminal_admission(tmp_path):
    cache = _ClaimCache()
    features = _features()
    features.capability(ExecutionCapability(retain_history=True, coordination=cache), scope="runtime")
    storage = _storage(tmp_path, cache=cache)
    entered, finish = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])

    async with Runtime.open('builtin-session', storage=storage,
                            models=_Models(_UsageFunctionModel(model)),
                            features=features,
                            context=RuntimeContext(None, tenant_id='tenant')) as runtime:
        session = await runtime.agents.get().create_session('session', principal=_OWNER)
        execution = await session.start('question')
        try:
            await asyncio.wait_for(entered.wait(), 5)
            record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id='tenant')
            assert record is not None and record.status.value == 'STARTED'
        finally:
            finish.set()
        assert (await execution.wait(timeout_seconds=5)).result.status.value == 'SUCCEEDED'
        assert (await session.history()).items
    assert cache.claims >= 1



@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['failed', 'cancelled'])
async def test_unretained_no_response_failure_preserves_prior_context(tmp_path, outcome):
    calls = []
    entered = asyncio.Event()
    phase = 'seed'

    async def model(messages, info):
        calls.append(deepcopy(messages))
        if phase == 'interrupt':
            entered.set()
            if outcome == 'failed':
                raise RuntimeError('model failed before response')
            await asyncio.Event().wait()
        return ModelResponse(parts=[TextPart('retained answer')])

    async with _open(tmp_path, model, retain=True) as runtime:
        session = await runtime.agents.get().create_session('session', principal=_OWNER)
        first = await session.start('retained question')
        await first.wait(timeout_seconds=5)

    phase = 'interrupt'
    async with _open(tmp_path, model) as runtime:
        session = await runtime.sessions.get('session', principal=_OWNER)
        interrupted = await session.start('aborted question')
        await asyncio.wait_for(entered.wait(), 5)
        if outcome == 'cancelled':
            await interrupted.cancel(idempotency_key='cancel')
        result = (await interrupted.wait(timeout_seconds=5)).result
        assert result.status.value == outcome.upper()
        view = await runtime.sessions.reconcile('session', principal=_OWNER)
        assert view.active_execution_id is None
        assert view.history_quality != 'unavailable'

    phase = 'continue'
    async with _open(tmp_path, model, retain=True) as runtime:
        session = await runtime.sessions.get('session', principal=_OWNER)
        final = await session.start('continue question')
        assert (await final.wait(timeout_seconds=5)).result.status.value == 'SUCCEEDED'
    assert 'retained question' in repr(calls[-1])
    assert 'aborted question' not in repr(calls[-1])


@pytest.mark.asyncio
async def test_unavailable_guard_does_not_depend_on_retained_execution_header(tmp_path, monkeypatch):
    async def model(messages, info):
        return ModelResponse(parts=[TextPart('answer')])

    async with _open(tmp_path, model) as runtime:
        session = await runtime.agents.get().create_session('session', principal=_OWNER)
        execution = await session.start('first')
        await execution.wait(timeout_seconds=5)

    storage = _storage(tmp_path)
    async with Runtime.open('builtin-session', storage=storage,
                            models=_Models(_UsageFunctionModel(model)), features=_features(retain=True),
                            context=RuntimeContext(None, tenant_id='tenant')) as runtime:
        session = await runtime.sessions.get('session', principal=_OWNER)

        async def no_execution_evidence(*args, **kwargs):
            raise AssertionError('missing-context authority must survive execution pruning')

        monkeypatch.setattr(runtime._state.execution.executions, 'get', no_execution_evidence)
        for operation in (lambda: session.start('second'), lambda: session.fork('fork')):
            with pytest.raises(AIError) as missing:
                await operation()
            assert missing.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE


@pytest.mark.asyncio
async def test_retained_history_can_fork_without_current_capture_feature(tmp_path):
    calls = []

    async def model(messages, info):
        calls.append(deepcopy(messages))
        return ModelResponse(parts=[TextPart('answer')])

    async with _open(tmp_path, model, retain=True) as runtime:
        source = await runtime.agents.get().create_session('source', principal=_OWNER)
        await (await source.start('retained source question')).wait(timeout_seconds=5)

    async with _open(tmp_path, model) as runtime:
        source = await runtime.sessions.get('source', principal=_OWNER)
        fork = await source.fork('fork', idempotency_key='fork')
        assert (await fork.history()).items
        await (await fork.start('fork question')).wait(timeout_seconds=5)
        assert (await runtime.sessions.reconcile('fork', principal=_OWNER)).history_quality == 'unavailable'
        assert (await runtime.sessions.reconcile('source', principal=_OWNER)).history_quality != 'unavailable'
    assert 'retained source question' in repr(calls[-1])


@pytest.mark.asyncio
async def test_reconcile_terminal_before_split_session_release_preserves_unavailable(tmp_path, monkeypatch):
    finish_model = asyncio.Event()
    release_entered = asyncio.Event()
    allow_release = asyncio.Event()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        await finish_model.wait()
        return ModelResponse(parts=[TextPart('unretained answer')])

    storage = _storage(tmp_path, 'split_sqlite')
    async with Runtime.open('builtin-session', storage=storage,
                            models=_Models(_UsageFunctionModel(model)), features=_features(),
                            context=RuntimeContext(None, tenant_id='tenant')) as runtime:
        session = await runtime.agents.get().create_session('session', principal=_OWNER)
        mutate = runtime._state.conversation.sessions.state_store.mutate
        held = False
        execution = None

        async def hold_first_release(callback):
            nonlocal held
            if execution is not None and not held:
                terminal = await runtime._state.execution.executions.get(execution.execution_id, tenant_id='tenant')
                if terminal.status.value == 'SUCCEEDED':
                    held = True
                    release_entered.set()
                    await allow_release.wait()
            return await mutate(callback)

        monkeypatch.setattr(runtime._state.conversation.sessions.state_store, 'mutate', hold_first_release)
        execution = await session.start('first')
        finish_model.set()
        try:
            await asyncio.wait_for(release_entered.wait(), 5)
            terminal = await runtime._state.execution.executions.get(execution.execution_id, tenant_id='tenant')
            assert terminal.status.value == 'SUCCEEDED'
            before = await runtime._state.conversation.sessions.get('session', tenant_id='tenant')
            assert before.active_execution_id == execution.execution_id
            reconciled = await runtime.sessions.reconcile('session', principal=_OWNER)
            assert reconciled.active_execution_id is None
            assert reconciled.history_quality == 'unavailable'
            with pytest.raises(AIError) as missing:
                await session.start('must not race past release')
            assert missing.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
            assert calls == 1
        finally:
            allow_release.set()
        assert (await execution.wait(timeout_seconds=5)).result.status.value == 'SUCCEEDED'
        assert (await runtime.sessions.reconcile('session', principal=_OWNER)).history_quality == 'unavailable'


@pytest.mark.asyncio
async def test_sync_peer_coordinates_with_terminal_owner_before_pending_publication(tmp_path):
    claimed, allow_owner, entered, finish = (asyncio.Event() for _ in range(4))

    class HeldClaimCache(InMemoryCacheBackend):
        held = False

        async def claim_turn(self, *args, **kwargs):
            result = await super().claim_turn(*args, **kwargs)
            if not self.held:
                self.held = True
                claimed.set()
                await allow_owner.wait()
            return result

    cache = HeldClaimCache(max_bytes=4_000_000)
    calls = []

    async def model(messages, info):
        calls.append(messages)
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart('answer')])

    terminal = RuntimeFeatures('builtin-session')
    terminal.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal(), coordination=cache), scope="runtime")
    terminal.agent('default', model='default', allow_tools=())
    sync = _features()
    sync.capability(ExecutionCapability(retain_history=True, coordination=cache), scope="runtime")
    models = _Models(_UsageFunctionModel(model))
    storage = _storage(tmp_path, cache=cache)
    async with Runtime.open('builtin-session', storage=storage, models=models,
                            features=terminal) as owner, Runtime.open(
            'builtin-session', storage=_storage(tmp_path, cache=cache), models=models,
            features=sync) as peer:
        session = await owner.agents.get().create_session('session')
        remote = peer.agents.get().session('session')
        first = asyncio.create_task(session.start('question', idempotency_key='request'))
        await asyncio.wait_for(claimed.wait(), 5)
        duplicate = asyncio.create_task(remote.start('question', idempotency_key='request'))
        try:
            # A claimed but unpublished owner must prevent a sync peer from
            # entering its model through an independent durable admission.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(entered.wait(), .15)
            allow_owner.set()
            execution = await asyncio.wait_for(first, 5)
            same = await asyncio.wait_for(duplicate, 5)
            await asyncio.wait_for(entered.wait(), 5)
            assert same.execution_id == execution.execution_id
            assert len(calls) == 1
            assert await owner._state.execution.executions.get(
                execution.execution_id, tenant_id=owner.tenant_id) is None
            with pytest.raises(AIError) as conflict:
                await remote.start('different', idempotency_key='request')
            assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
            with pytest.raises(AIError) as busy:
                await remote.start('next', idempotency_key='next')
            assert busy.value.code is ErrorCode.SESSION_BUSY
        finally:
            allow_owner.set()
            finish.set()
            await asyncio.gather(first, duplicate, return_exceptions=True)
        assert (await execution.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        assert (await same.wait(timeout_seconds=5)).result.output == {'text': 'answer'}
        assert len(calls) == 1
