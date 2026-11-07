#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Public capability configuration owns module routes, caches and read-only use."""
from ._runtime_test_helpers import model_features

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
from linktools.ai.capability import ExecutionCapability, TraceCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.history import History
from linktools.ai.snapshot import Snapshot
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.storage import StorageRoute
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore, ObjectRef
from linktools.ai.core import CommitPolicy

from linktools.ai.observe.service import MetricsService

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


@pytest.mark.parametrize("checkpointed", (False, True))
async def test_module_sqlite_routes_reopen_history_without_models_or_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpointed: bool,
) -> None:
    root = tmp_path / "state"
    routes = {
        domain: StorageRoute.sqlite(root / "runtime.db", exclusive_writer=True)
        for domain in (PersistenceDomain.CONVERSATION, PersistenceDomain.CHECKPOINT, PersistenceDomain.TRACE)
    }
    declarations = _features(
        ExecutionCapability(retain_history=True),
        *((native_checkpoint(),) if checkpointed else ()),
        TraceCapability(),
    )
    declarations.model("default", _FeatureModels())
    features = declarations.borrow()
    def root_storage() -> PersistenceStorage:
        return PersistenceStorage(replace(PersistenceStorage.from_root(root, exclusive_writer=True).plan, conversation=routes[PersistenceDomain.CONVERSATION], trace=routes[PersistenceDomain.TRACE], checkpoint=routes[PersistenceDomain.CHECKPOINT]))

    state = root_storage()
    async with Persistence.open('module-routes', storage=state) as persistence_67, Runtime.attach(persistence=persistence_67, metrics=Metrics.in_memory(), features=features) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        principal = runtime.default_principal
        execution_id = execution.execution_id
        expected = (await execution.model_interactions()).items
        assert expected
        for domain, route in routes.items():
            assert runtime._state.plan.route(domain) == route
            assert route.path.is_file()

    async def unexpected_hook(*args, **kwargs):
        pytest.fail("read-only history must not invoke execution hooks")

    def unexpected_recording(*args, **kwargs):
        pytest.fail("read-only history must not open a metric recorder or model")

    monkeypatch.setattr(TraceCapability, "before_run", unexpected_hook)
    monkeypatch.setattr(StepPersistence, "before_model_request", unexpected_hook)
    monkeypatch.setattr(MetricsService, "__init__", unexpected_recording)
    monkeypatch.setattr(_FeatureModels, "materialize", unexpected_recording)
    reopened = root_storage()
    async with Persistence.open("module-routes", storage=reopened, read_only=True) as persistence, History.open(persistence=persistence) as history:
        assert (await history.result(execution_id, principal=principal)).output == {"text": "done"}
        interactions = (await history.model_interactions(execution_id, principal=principal)).items
        assert [(item.model_request_seq, item.status) for item in interactions] == [
            (item.model_request_seq, item.status) for item in expected
        ]
        assert (await history.transcript(execution_id, principal=principal, include_content=True)).items


    limits = SnapshotLimits(max_entries=10000, max_bytes=16 * 1024 * 1024)
    targets = (InMemoryObjectStore("snapshot-first"), InMemoryObjectStore("snapshot-second"))
    snapshots = []
    if checkpointed:
        async with Persistence.open("module-routes", storage=root_storage(), tenant_id=principal.tenant_id,
                                    read_only=True) as persistence:
            with pytest.raises(AIError) as rejected:
                await Snapshot.create(persistence=persistence, object_store=targets[0],
                                      exclusive=_SnapshotGuard(), limits=limits)
        assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
        return
    for target in targets:
        async with Persistence.open("module-routes", storage=root_storage(), tenant_id=principal.tenant_id, read_only=True) as persistence:
            snapshots.append(await Snapshot.create(
                persistence=persistence, object_store=target, exclusive=_SnapshotGuard(), limits=limits,
            ))
    assert (snapshots[0].key, snapshots[0].digest, snapshots[0].size) == (
        snapshots[1].key, snapshots[1].digest, snapshots[1].size,
    )
    restored = await Snapshot.restore(
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
    state = PersistenceStorage.from_root(root, cache=root_cache, exclusive_writer=True)
    trace_cache = {"inherit": root_cache, "none": None,
                   "separate": module_cache, "shared": root_cache}[trace_cache_mode]
    if trace_cache_mode != "inherit":
        state = PersistenceStorage(replace(state.plan, trace=state.plan.route(PersistenceDomain.TRACE).with_cache(trace_cache)))
    trace = TraceCapability()
    features = _features(ExecutionCapability(retain_history=True), trace)
    async with Persistence.open('module-cache', storage=state) as persistence_135, Runtime.attach(persistence=persistence_135, features=model_features(_FeatureModels(), features)) as runtime:
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

        for domain in (PersistenceDomain.CONVERSATION, PersistenceDomain.TRACE):
            key = "cache-probe-" + domain.value
            digest = hashlib.sha256(content).hexdigest()
            await runtime._state.object_store(domain).put(
                key, chunks(), expected_size=len(content), expected_digest=digest,
            )
            before = [(cache.reads, cache.writes) for cache in (root_cache, module_cache)]
            ref = ObjectRef("runtime", key, digest, len(content))
            assert await runtime._state.read_object(domain, ref) == content
            assert await runtime._state.read_object(domain, ref) == content
            expected_cache = root_cache if domain is PersistenceDomain.CONVERSATION else trace_cache
            for cache, counters in zip((root_cache, module_cache), before):
                if cache is expected_cache:
                    assert cache.reads > counters[0]
                    assert cache.writes > counters[1]
                else:
                    assert (cache.reads, cache.writes) == counters


async def test_trace_route_rejects_a_different_transaction_group(tmp_path: Path) -> None:
    root = tmp_path / "state"
    storage = PersistenceStorage(replace(
        PersistenceStorage.from_root(root, exclusive_writer=True).plan,
        trace=StorageRoute.sqlite(tmp_path / "separate.db", exclusive_writer=True),
    ))
    with pytest.raises(AIError) as invalid:
        async with Persistence.open("different-group", storage=storage) as persistence, Runtime.attach(
            persistence=persistence, features=model_features(_FeatureModels(), _features(TraceCapability())),
        ):
            pytest.fail("Trace cannot silently split its source transaction")
    assert invalid.value.code in {ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, ErrorCode.STORAGE_DEPENDENCY_NOT_READY}


async def test_native_checkpoint_accepts_an_independent_durable_store(tmp_path: Path) -> None:
    from pydantic_ai_harness.step_persistence import SqliteStepStore

    database = tmp_path / "independent-steps.db"
    checkpoint = native_checkpoint(database=database)
    storage = PersistenceStorage.sqlite(tmp_path / "control.db")
    async with Persistence.open("independent-steps", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_FeatureModels(), _features(checkpoint)),
    ) as runtime:
        execution = await runtime.agents.get().start("question")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        runs = await checkpoint.store.list_runs()
        assert len(runs) == 1
        expected = await checkpoint.store.latest_snapshot(run_id=runs[0].run_id)
        assert expected is not None and expected.state == "complete"
    cold = SqliteStepStore(database=database)
    assert await cold.latest_snapshot(run_id=runs[0].run_id) == expected


async def test_split_conversation_route_uses_durable_admission_with_shared_cache(tmp_path: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        yield "done"

    cache = InMemoryCacheBackend(max_bytes=2**20)
    state = PersistenceStorage.sqlite(tmp_path / "execution.db", cache=cache)
    route = StorageRoute.sqlite(tmp_path / "conversation.db")
    state = PersistenceStorage(replace(state.plan, conversation=route), coordination=cache)
    features = _features()
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    async with Persistence.open('split-conversation', storage=state) as persistence_196, Runtime.attach(persistence=persistence_196, features=model_features(_FeatureModels(FunctionModel(stream_function=model)), features)) as runtime:
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
