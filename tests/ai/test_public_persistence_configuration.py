#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public capability configuration owns module routes, caches and read-only use."""

import asyncio
import hashlib
from dataclasses import replace
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics
from linktools.ai.capability import ExecutionCapability, CheckpointCapability, ConversationCapability, TraceCapability
from linktools.ai.runtime import Runtime, RuntimeDomain, RuntimeHistory, RuntimeSnapshot, RuntimeStorage
from linktools.ai.storage import StorageRoute
from linktools.ai.runtime.state import SnapshotLimits
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore, ObjectRef
from linktools.ai.core import CommitPolicy

from linktools.ai.runtime._metrics_service import MetricsService

from .test_explicit_runtime_features import _FeatureModels, _features
from .test_runtime_snapshot_strict_decode import _SnapshotGuard

pytestmark = pytest.mark.asyncio


class _Cache(InMemoryCacheBackend):
    def __init__(self) -> None:
        super().__init__(max_bytes=2**20)
        self.reads = 0
        self.writes = 0

    async def get(self, key: str) -> bytes | None:
        self.reads += 1
        return await super().get(key)

    async def put(self, key: str, content: bytes) -> None:
        self.writes += 1
        await super().put(key, content)


async def test_module_filesystem_routes_reopen_history_without_models_or_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "state"
    routes = {
        domain: StorageRoute.filesystem(root / "modules" / domain.value, transaction_root=root)
        for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.CHECKPOINT, RuntimeDomain.TRACE)
    }
    features = await _features(
        ConversationCapability(),
        CheckpointCapability(storage=routes[RuntimeDomain.CHECKPOINT]),
        TraceCapability(storage=routes[RuntimeDomain.TRACE]),
    ).capture()
    def root_storage() -> RuntimeStorage:
        return RuntimeStorage(replace(RuntimeStorage.filesystem(root).plan, conversation=routes[RuntimeDomain.CONVERSATION]))

    state = root_storage()
    async with Runtime.open("module-routes", metrics=Metrics.in_memory(), storage=state, models=_FeatureModels(), features=features) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        principal = runtime.default_principal
        execution_id = execution.execution_id
        expected = (await execution.model_interactions()).items
        assert expected
        for domain, route in routes.items():
            assert runtime._state.plan.route(domain) == route
            assert list(route.path.rglob("manifest.json"))

    async def unexpected_hook(*args, **kwargs):
        pytest.fail("read-only history must not invoke execution hooks")

    def unexpected_recording(*args, **kwargs):
        pytest.fail("read-only history must not open a metric recorder or model")

    monkeypatch.setattr(TraceCapability, "before_run", unexpected_hook)
    monkeypatch.setattr(CheckpointCapability, "before_model_request", unexpected_hook)
    monkeypatch.setattr(MetricsService, "__init__", unexpected_recording)
    monkeypatch.setattr(_FeatureModels, "materialize", unexpected_recording)
    reopened = root_storage()
    async with RuntimeHistory.open("module-routes", storage=reopened, features=features) as history:
        assert (await history.result(execution_id, principal=principal)).output == {"text": "done"}
        interactions = (await history.model_interactions(execution_id, principal=principal)).items
        assert [(item.model_request_seq, item.status) for item in interactions] == [
            (item.model_request_seq, item.status) for item in expected
        ]
        assert (await history.transcript(execution_id, principal=principal, include_content=True)).items


    limits = SnapshotLimits(max_entries=10000, max_bytes=16 * 1024 * 1024)
    targets = (InMemoryObjectStore("snapshot-first"), InMemoryObjectStore("snapshot-second"))
    snapshots = [await RuntimeSnapshot.create(
        "module-routes", tenant_id=principal.tenant_id,
        storage=root_storage(), object_store=target, features=features,
        exclusive=_SnapshotGuard(), limits=limits,
    ) for target in targets]
    assert (snapshots[0].key, snapshots[0].digest, snapshots[0].size) == (
        snapshots[1].key, snapshots[1].digest, snapshots[1].size,
    )
    restored = await RuntimeSnapshot.restore(
        snapshots[0], object_store=targets[0], target=tmp_path / "restored",
        namespace="module-routes", tenant_id=principal.tenant_id, limits=limits,
    )
    async with restored.open_history() as history:
        assert (await history.result(execution_id, principal=principal)).output == {"text": "done"}
        interactions = (await history.model_interactions(execution_id, principal=principal)).items
        assert [(item.model_request_seq, item.status) for item in interactions] == [
            (item.model_request_seq, item.status) for item in expected
        ]
        assert (await history.transcript(execution_id, principal=principal, include_content=True)).items


@pytest.mark.parametrize("trace_cache_mode", ("inherit", "none", "separate", "shared"))
async def test_module_resource_cache_inherits_or_explicitly_overrides_root(
    tmp_path: Path, trace_cache_mode: str,
) -> None:
    root_cache, module_cache = _Cache(), _Cache()
    root = tmp_path / "state"
    state = RuntimeStorage.filesystem(root, cache=root_cache)
    trace_cache = {"inherit": root_cache, "none": None,
                   "separate": module_cache, "shared": root_cache}[trace_cache_mode]
    trace = (TraceCapability() if trace_cache_mode == "inherit" else TraceCapability(
        storage=state.plan.route(RuntimeDomain.TRACE).with_cache(trace_cache),
    ))
    features = _features(ConversationCapability(), trace)
    async with Runtime.open("module-cache", storage=state, models=_FeatureModels(), features=features) as runtime:
        before_root = (root_cache.reads, root_cache.writes)
        before_module = (module_cache.reads, module_cache.writes)
        assert await runtime._state.conversation.sessions.get("missing", tenant_id="default") is None
        assert await runtime._state.conversation.sessions.get("missing", tenant_id="default") is None
        assert root_cache.reads > before_root[0]
        assert root_cache.writes > before_root[1]
        assert (module_cache.reads, module_cache.writes) == before_module
        content = b"module cache body"

        async def chunks() -> AsyncIterator[bytes]:
            yield content

        for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.TRACE):
            key = "cache-probe-" + domain.value
            digest = hashlib.sha256(content).hexdigest()
            await runtime._state.object_store(domain).put(
                key, chunks(), expected_size=len(content), expected_digest=digest,
            )
            before = [(cache.reads, cache.writes) for cache in (root_cache, module_cache)]
            ref = ObjectRef("runtime", key, digest, len(content))
            assert await runtime._state.read_object(domain, ref) == content
            assert await runtime._state.read_object(domain, ref) == content
            expected_cache = root_cache if domain is RuntimeDomain.CONVERSATION else trace_cache
            for cache, counters in zip((root_cache, module_cache), before):
                if cache is expected_cache:
                    assert cache.reads > counters[0]
                    assert cache.writes > counters[1]
                else:
                    assert (cache.reads, cache.writes) == counters


@pytest.mark.parametrize("feature_type", (CheckpointCapability, TraceCapability))
async def test_atomic_module_route_rejects_a_different_transaction_group(
    tmp_path: Path, feature_type: type,
) -> None:
    root = tmp_path / "state"
    feature = feature_type(storage=StorageRoute.filesystem(tmp_path / "separate"))
    with pytest.raises(AIError) as invalid:
        async with Runtime.open(
            "different-group", storage=RuntimeStorage.filesystem(root),
            models=_FeatureModels(), features=_features(feature),
        ):
            pytest.fail("an atomic module cannot silently split its transactions")
    assert invalid.value.code in {ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, ErrorCode.STORAGE_DEPENDENCY_NOT_READY}


async def test_split_conversation_route_uses_durable_admission_with_shared_cache(tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        yield "done"

    cache = InMemoryCacheBackend(max_bytes=2**20)
    state = RuntimeStorage.sqlite(tmp_path / "execution.db", cache=cache)
    route = StorageRoute.sqlite(tmp_path / "conversation.db")
    state = RuntimeStorage(replace(state.plan, conversation=route))
    features = _features(ConversationCapability())
    features.capability(ExecutionCapability(commit=CommitPolicy.terminal(), coordination=cache), scope="runtime")
    async with Runtime.open(
        "split-conversation", storage=state,
        models=_FeatureModels(FunctionModel(stream_function=model)), features=features,
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="turn")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert record is not None
            assert record.status is ExecutionStatus.STARTED
        finally:
            release.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        replay = await session.start("question", idempotency_key="turn")
        assert replay.execution_id == execution.execution_id
