#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured Runtime policies and offline response declarations have one owner."""

from dataclasses import FrozenInstanceError, replace
import inspect

import pytest
from pydantic_ai.capabilities import CombinedCapability, PrefixTools

from linktools.ai.capability import RuntimeFeatures, ToolResponsesCapability
from linktools.ai.core import AuthorizationAction, PromptLimits, ResourceKind, ResourceRef, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeContext, RuntimeHistory, RuntimeStorage

from .test_tool_response_fixture import _manifest, _store


class Policy:
    def __init__(self):
        self.denied = False
        self.calls = []

    def __bool__(self):
        return False

    async def authorize(self, principal, action, resource):
        self.calls.append((principal, action, resource))
        if self.denied:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


@pytest.mark.asyncio
async def test_capture_borrows_dynamic_policy_and_freezes_limits():
    policy = Policy()
    limits = PromptLimits(max_binary_input_parts=1)
    capture = await RuntimeFeatures('policy', authorization=policy, limits=limits).capture()
    assert capture.authorization is policy
    assert capture.limits is limits
    assert 'authorization=' not in repr(capture)
    assert replace(capture, authorization=Policy()) == capture
    with pytest.raises(FrozenInstanceError):
        capture.limits = PromptLimits()
    with pytest.raises(FrozenInstanceError):
        limits.max_binary_input_parts = 2
    policy.denied = True
    with pytest.raises(AIError) as caught:
        await capture.authorization.authorize(service_principal('tenant', 'actor'), AuthorizationAction.EXECUTION_READ,
                                              ResourceRef(ResourceKind.EXECUTION, 'execution', 'tenant'))
    assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED


def test_removed_open_keywords_have_no_aliases():
    for name in ('authorization', 'limits', 'tool_responses'):
        assert name not in inspect.signature(Runtime.open).parameters
        with pytest.raises(TypeError):
            Runtime.open('test', models=ModelRegistry(), storage=RuntimeStorage.in_memory(), **{name: None})
    assert 'authorization' not in inspect.signature(RuntimeHistory.open).parameters
    with pytest.raises(TypeError):
        RuntimeHistory.open('test', storage=RuntimeStorage.in_memory(), authorization=None)
    with pytest.raises(TypeError):
        RuntimeFeatures('test', limits=object())


@pytest.mark.asyncio
async def test_runtime_uses_captured_policy_and_limits_even_when_filter_rejects_all():
    policy = Policy()
    limits = PromptLimits(max_binary_input_parts=1)
    features = RuntimeFeatures('policy', authorization=policy, limits=limits, filter=lambda *_: False)
    async with Runtime.open('test', models=ModelRegistry(), storage=RuntimeStorage.in_memory(), features=features) as runtime:
        assert runtime._execution_service._authorization is policy
        assert runtime._session_service._authorization is policy
        assert runtime._graph_service._authorization is policy
        assert runtime._execution_service._input_materializer._limits is limits


@pytest.mark.asyncio
@pytest.mark.parametrize('tenant', ('one', 'two'))
async def test_default_policy_resolves_per_open_tenant(tenant):
    capture = await RuntimeFeatures('reused').capture()
    async with Runtime.open('test', models=ModelRegistry(), storage=RuntimeStorage.in_memory(),
                            context=RuntimeContext(None, tenant_id=tenant), features=capture) as runtime:
        policy = runtime._execution_service._authorization
        ref = ResourceRef(ResourceKind.EXECUTION, 'execution', tenant)
        await policy.authorize(service_principal(tenant, 'actor'), AuthorizationAction.EXECUTION_READ, ref)
        with pytest.raises(AIError) as caught:
            await policy.authorize(service_principal('foreign', 'actor'), AuthorizationAction.EXECUTION_READ, ref)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert capture.authorization is None


@pytest.mark.asyncio
async def test_history_uses_current_capture_policy(tmp_path):
    storage = RuntimeStorage.from_root(tmp_path)
    async with Runtime.open('test', models=ModelRegistry(), storage=storage):
        pass
    policy = Policy()
    capture = await RuntimeFeatures('history', authorization=policy).capture()
    async with RuntimeHistory.open('test', storage=RuntimeStorage.from_root(tmp_path), features=capture) as history:
        assert history._authorization is policy
        assert history._service._authorization is policy


@pytest.mark.asyncio
async def test_response_capability_scope_uniqueness_and_bootstrap_consumption():
    store, fixture = await _store(_manifest())
    try:
        feature = ToolResponsesCapability(fixture)
        assert feature.fixture is fixture
        with pytest.raises(AttributeError):
            feature.fixture = fixture
        with pytest.raises(AIError) as caught:
            RuntimeFeatures('wrong-scope').capability(feature, scope="agent")
        assert caught.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
        features = RuntimeFeatures('responses')
        features.capability(feature, scope='runtime')
        with pytest.raises(AIError) as caught:
            features.capability(ToolResponsesCapability(fixture), scope='runtime')
        assert caught.value.code is ErrorCode.CAPABILITY_CONFLICT
        async with Runtime.open('test', models=ModelRegistry(), storage=RuntimeStorage.in_memory(), features=features) as runtime:
            assert runtime._execution_service._tool_responses is fixture
        wrapped = RuntimeFeatures('wrapped')
        wrapped.capability(PrefixTools(CombinedCapability([ToolResponsesCapability(fixture)]), prefix='test'), id='wrapped.responses', scope='runtime')
        async with Runtime.open('wrapped', models=ModelRegistry(), storage=RuntimeStorage.in_memory(), features=wrapped) as runtime:
            assert runtime._execution_service._tool_responses is fixture
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_captured_limits_reject_oversized_input_before_execution():
    from pydantic_ai import BinaryContent

    features = RuntimeFeatures('limits', limits=PromptLimits(max_binary_input_parts=1))
    async with Runtime.open('test', models=ModelRegistry(), storage=RuntimeStorage.in_memory(), features=features) as runtime:
        with pytest.raises(AIError) as caught:
            await runtime._execution_service._input_materializer.materialize(
                (BinaryContent(b'one', media_type='text/plain'), BinaryContent(b'two', media_type='text/plain')), (),
            )
        assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_restored_history_forwards_current_features(monkeypatch, tmp_path):
    from linktools.ai.runtime import RestoredRuntime

    sentinel = object()
    features = RuntimeFeatures('restored', authorization=Policy())
    seen = {}

    def open_history(namespace, **kwargs):
        seen.update(namespace=namespace, **kwargs)
        return sentinel

    monkeypatch.setattr(RuntimeHistory, 'open', open_history)
    restored = RestoredRuntime('digest', 'namespace', 'tenant', tmp_path, None, 'generation')
    assert restored.open_history(features=features) is sentinel
    assert seen['features'] is not features
    assert seen['features'].authorization is features.authorization
    assert seen['tenant_id'] == 'tenant'
    assert 'authorization' not in seen


@pytest.mark.asyncio
@pytest.mark.parametrize('captured', (False, True))
async def test_restored_history_ignores_source_routes_and_cache(tmp_path, monkeypatch, captured):
    import shutil

    from linktools.ai.runtime import (
        CheckpointCapability, ConversationCapability, RuntimeDomain, RuntimeSnapshot,
        RuntimeStorageRoute, TraceCapability,
    )
    from linktools.ai.runtime.state import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore

    from .test_explicit_runtime_features import _FeatureModels, _features
    from .test_runtime_snapshot_strict_decode import _SnapshotGuard
    from .test_public_persistence_configuration import _Cache

    source = tmp_path / 'source'
    routes = {domain: RuntimeStorageRoute.filesystem(source / 'modules' / domain.value, transaction_root=source)
              for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.CHECKPOINT, RuntimeDomain.TRACE)}
    cache = _Cache()
    group = _features(ConversationCapability(storage=routes[RuntimeDomain.CONVERSATION], cache=cache),
                      CheckpointCapability(storage=routes[RuntimeDomain.CHECKPOINT]),
                      TraceCapability(storage=routes[RuntimeDomain.TRACE], cache=cache))
    policy = Policy()
    group._authorization = policy
    capture = await group.capture()
    async with Runtime.open('restore-policy', models=_FeatureModels(), storage=RuntimeStorage.filesystem(source),
                            features=capture) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('question')
        await execution.wait(timeout_seconds=5)
        execution_id, principal = execution.execution_id, runtime.default_principal
        expected = (await execution.model_interactions()).items
    objects = InMemoryObjectStore('snapshot')
    limits = SnapshotLimits(max_entries=10000, max_bytes=16 * 1024 * 1024)
    ref = await RuntimeSnapshot.create('restore-policy', tenant_id=principal.tenant_id,
                                       storage=RuntimeStorage.filesystem(source), object_store=objects,
                                       features=capture, exclusive=_SnapshotGuard(), limits=limits)
    restored = await RuntimeSnapshot.restore(ref, object_store=objects, target=tmp_path / 'restored',
                                            namespace='restore-policy', tenant_id=principal.tenant_id, limits=limits)
    shutil.rmtree(source)

    async def forbidden_cache(*args, **kwargs):
        pytest.fail('restored history must not reuse original module caches')

    monkeypatch.setattr(cache, 'get', forbidden_cache)
    monkeypatch.setattr(cache, 'put', forbidden_cache)

    async def forbidden_capture():
        pytest.fail('restored history must not capture unavailable source declarations')

    if not captured:
        monkeypatch.setattr(group, 'capture', forbidden_capture)
    async with restored.open_history(features=capture if captured else group) as history:
        assert history._authorization is policy
        assert (await history.result(execution_id, principal=principal)).output == {'text': 'done'}
        assert len((await history.model_interactions(execution_id, principal=principal)).items) == len(expected)
        assert (await history.transcript(execution_id, principal=principal, include_content=True)).items
        policy.denied = True
        with pytest.raises(AIError) as caught:
            await history.result(execution_id, principal=principal)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert not source.exists()
