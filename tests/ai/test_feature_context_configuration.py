#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence

from linktools.ai.core import ExecutionContext
"""Captured Runtime policies and offline response declarations have one owner."""
from ._runtime_test_helpers import model_features

from dataclasses import FrozenInstanceError, replace

import pytest
from pydantic_ai.capabilities import CombinedCapability, PrefixTools

from linktools.ai.capability import ToolResponsesCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import AuthorizationAction, PromptLimits, ResourceKind, ResourceRef, service_principal
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage

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
async def test_runtime_borrows_dynamic_policy_and_capture_freezes_limits():
    policy = Policy()
    limits = PromptLimits(max_binary_input_parts=1)
    capture = await RuntimeFeatures('policy', limits=limits).capture()
    assert capture.limits is limits
    with pytest.raises(FrozenInstanceError):
        capture.limits = PromptLimits()
    with pytest.raises(FrozenInstanceError):
        limits.max_binary_input_parts = 2
    async with Runtime.open(
        'policy', storage=PersistenceStorage.in_memory(),
        features=capture, authorization=policy,
    ) as runtime:
        assert (await runtime.history.list_sessions(principal=runtime.default_principal)).items == ()
        assert policy.calls
        policy.denied = True
        with pytest.raises(AIError) as caught:
            await runtime.history.list_sessions(principal=runtime.default_principal)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED


def test_runtime_configuration_rejects_misplaced_agent_options():
    for name in ('limits', 'tool_responses'):
        with pytest.raises(TypeError):
            Runtime.open('test', storage=PersistenceStorage.in_memory(), **{name: None})
    with pytest.raises(TypeError):
        RuntimeFeatures('test', limits=object())


@pytest.mark.asyncio
async def test_runtime_uses_host_policy_and_captured_limits_when_filter_rejects_all():
    policy = Policy()
    limits = PromptLimits(max_binary_input_parts=1)
    features = RuntimeFeatures('policy', limits=limits, filter=lambda *_: False)
    async with Persistence.open('test', storage=PersistenceStorage.in_memory()) as persistence_75, Runtime.attach(persistence=persistence_75, features=features, authorization=policy) as runtime:
        assert runtime._execution_service._authorization is policy
        assert runtime._session_service._authorization is policy
        assert runtime._execution_service._input_materializer._limits is limits
        assert (await runtime.history.list_sessions(principal=runtime.default_principal)).items == ()
        policy.denied = True
        with pytest.raises(AIError) as rejected:
            await runtime.history.list_sessions(principal=runtime.default_principal)
        assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize('tenant', ('one', 'two'))
async def test_default_policy_resolves_per_open_tenant(tenant):
    capture = await RuntimeFeatures('reused').capture()
    async with Persistence.open('test', storage=PersistenceStorage.in_memory(), tenant_id=tenant) as persistence_86, Runtime.attach(persistence=persistence_86, context=ExecutionContext(None), features=capture) as runtime:
        policy = runtime._execution_service._authorization
        ref = ResourceRef(ResourceKind.EXECUTION, 'execution', tenant)
        await policy.authorize(service_principal(tenant, 'actor'), AuthorizationAction.EXECUTION_READ, ref)
        with pytest.raises(AIError) as caught:
            await policy.authorize(service_principal('foreign', 'actor'), AuthorizationAction.EXECUTION_READ, ref)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_history_uses_current_host_policy(tmp_path):
    storage = PersistenceStorage.from_root(tmp_path)
    async with Persistence.open('test', storage=storage):
        pass
    policy = Policy()
    async with Persistence.open('test', storage=PersistenceStorage.from_root(tmp_path), tenant_id="default", read_only=True) as history_persistence_106, History.open(persistence=history_persistence_106, authorization=policy) as history:
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
        async with Persistence.open('test', storage=PersistenceStorage.in_memory()) as persistence_125, Runtime.attach(persistence=persistence_125, features=features) as runtime:
            assert runtime._execution_service._tool_responses is fixture
        wrapped = RuntimeFeatures('wrapped')
        wrapped.capability(PrefixTools(CombinedCapability([ToolResponsesCapability(fixture)]), prefix='test'), id='wrapped.responses', scope='runtime')
        async with Persistence.open('wrapped', storage=PersistenceStorage.in_memory()) as persistence_129, Runtime.attach(persistence=persistence_129, features=wrapped) as runtime:
            assert runtime._execution_service._tool_responses is fixture
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_captured_limits_reject_oversized_input_before_execution():
    from pydantic_ai import BinaryContent

    features = RuntimeFeatures('limits', limits=PromptLimits(max_binary_input_parts=1))
    async with Persistence.open('test', storage=PersistenceStorage.in_memory()) as persistence_140, Runtime.attach(persistence=persistence_140, features=features) as runtime:
        with pytest.raises(AIError) as caught:
            await runtime._execution_service._input_materializer.materialize(
                (BinaryContent(b'one', media_type='text/plain'), BinaryContent(b'two', media_type='text/plain')), (),
            )
        assert caught.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_restored_history_enforces_current_policy(tmp_path):
    from linktools.ai.snapshot import RestoredSnapshot

    policy = Policy()
    async with Persistence.open('namespace', storage=PersistenceStorage.from_root(tmp_path), tenant_id='tenant'):
        pass
    restored = RestoredSnapshot('digest', 'namespace', 'tenant', tmp_path, None, 'generation')
    async with restored.open_history(authorization=policy) as history:
        principal = service_principal('tenant', 'reader')
        assert (await history.list_sessions(principal=principal)).items == ()
        assert policy.calls
        policy.denied = True
        with pytest.raises(AIError) as rejected:
            await history.list_sessions(principal=principal)
        assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize('captured', (False, True))
async def test_restored_history_ignores_source_routes_and_cache(tmp_path, monkeypatch, captured):
    import shutil

    from linktools.ai.capability import CheckpointCapability, ExecutionCapability, TraceCapability
    from linktools.ai.snapshot import Snapshot
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.storage import StorageRoute
    from linktools.ai.persistence.snapshots import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore

    from .test_explicit_runtime_features import _FeatureModels, _features
    from .test_runtime_snapshot_strict_decode import _SnapshotGuard
    from .test_public_persistence_configuration import _Cache

    source = tmp_path / 'source'
    routes = {domain: StorageRoute.filesystem(source / 'modules' / domain.value, transaction_root=source)
              for domain in (PersistenceDomain.CONVERSATION, PersistenceDomain.CHECKPOINT, PersistenceDomain.TRACE)}
    def source_storage() -> PersistenceStorage:
        return PersistenceStorage(replace(PersistenceStorage.filesystem(source).plan, conversation=routes[PersistenceDomain.CONVERSATION], trace=routes[PersistenceDomain.TRACE], checkpoint=routes[PersistenceDomain.CHECKPOINT]))

    cache = _Cache()
    routes[PersistenceDomain.TRACE] = routes[PersistenceDomain.TRACE].with_cache(cache)
    group = _features(ExecutionCapability(retain_history=True),
                      CheckpointCapability(),
                      TraceCapability())
    policy = Policy()
    group.model("default", _FeatureModels())
    capture = await group.capture()
    async with Persistence.open('restore-policy', storage=source_storage()) as persistence_198, Runtime.attach(persistence=persistence_198, features=capture, authorization=policy) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('question')
        await execution.wait(timeout_seconds=5)
        execution_id, principal = execution.execution_id, runtime.default_principal
        expected = (await execution.model_interactions()).items
    objects = InMemoryObjectStore('snapshot')
    limits = SnapshotLimits(max_entries=10000, max_bytes=16 * 1024 * 1024)
    async with Persistence.open('restore-policy', storage=source_storage(), tenant_id=principal.tenant_id, read_only=True) as persistence:
        ref = await Snapshot.create(persistence=persistence, object_store=objects,
                                    exclusive=_SnapshotGuard(), limits=limits)
    restored = await Snapshot.restore(ref, object_store=objects, target=tmp_path / 'restored',
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
    async with restored.open_history(authorization=policy) as history:
        assert history._authorization is policy
        assert (await history.result(execution_id, principal=principal)).output == {'text': 'done'}
        assert len((await history.model_interactions(execution_id, principal=principal)).items) == len(expected)
        assert (await history.transcript(execution_id, principal=principal, include_content=True)).items
        policy.denied = True
        with pytest.raises(AIError) as caught:
            await history.result(execution_id, principal=principal)
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert not source.exists()
