#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-owned providers supply independent producer scopes without a dataset."""

import asyncio

import pytest

from linktools.ai.core import ExecutionContext, TaskStatus
from linktools.ai.observe import MetricFlushResult, MetricRecorderProvider, MetricRecorderStatus, Metrics, Observation
from linktools.ai.observe.service import MetricsService
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.task import Task, TaskGraph, TaskNode, TaskNodeContext
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _definitions, _FeatureModels
from .test_metrics_recording import _observation


class _HostRecorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.accepting = True
        self.close_failures = 0
        self.closed = False

    async def try_record(self, observation: Observation) -> bool:
        if not self.accepting:
            return False
        self.observations.append(observation)
        return True

    def status(self) -> MetricRecorderStatus:
        # A successful host publication is not confirmation of a SQL sink write.
        return MetricRecorderStatus(True, self.accepting, len(self.observations), 0, 0, 0, len(self.observations), 0)

    async def flush(self, *, timeout_seconds: float = 5.0) -> MetricFlushResult:
        return MetricFlushResult(True, self.status())

    async def close(self) -> None:
        self.accepting = False
        if self.close_failures:
            self.close_failures -= 1
            raise OSError("host scope cleanup incomplete")
        self.closed = True


class _HostProvider:
    def __init__(self) -> None:
        self.scopes: list[_HostRecorder] = []

    def recorder(self) -> _HostRecorder:
        scope = _HostRecorder()
        self.scopes.append(scope)
        return scope

    async def close(self) -> None:
        pytest.fail("producers must not close the shared provider or client")


async def _work(context: TaskNodeContext[None]) -> str:
    return "done"


@pytest.mark.asyncio
async def test_provider_only_agent_and_bound_task_preserve_observation_identity_and_context() -> None:
    provider = _HostProvider()
    assert isinstance(provider, MetricRecorderProvider)
    assert isinstance(Metrics.in_memory(), MetricRecorderProvider)
    task = Task("work", _work, effect_policy="none")
    async with Runtime.open(
        "provider-source", tenant_id="tenant", storage=PersistenceStorage.in_memory(),
        features=model_features(_FeatureModels(), _definitions()), metrics=provider,
        context=ExecutionContext(None, correlation={"request": "root"}, metric_dimensions={"environment": "test"}),
    ) as runtime:
        result = await runtime.agents.get().run("question", timeout_seconds=5)
        engine = await runtime.tasks.bind(task)
        run = await engine.start(TaskGraph("graph", (TaskNode("node", task=task),)), idempotency_key="graph")
        assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        assert len(provider.scopes) == 2
        agent, tasks = provider.scopes
        assert agent is not tasks
        assert {item.kind for item in agent.observations} >= {"linktools.model.request", "linktools.execution.terminal"}
        assert tasks.observations
        observations = agent.observations + tasks.observations
        assert all(item.source_namespace == "provider-source" and item.tenant_id == "tenant" for item in observations)
        assert all(item.dimensions["context.environment"] == "test" for item in observations)
        ids = [item.observation_id for item in observations]
        assert len(ids) == len(set(ids))
        terminals = [item for item in agent.observations if item.kind == "linktools.execution.terminal"]
        assert terminals[0].correlation["linktools.execution_id"] == result.result.execution_id
        assert (await runtime.metrics.flush()).completed
        assert runtime.metrics.status().persisted == 0
        assert runtime.metrics.status().write_failures == 0
    assert all(scope.closed for scope in provider.scopes)


@pytest.mark.asyncio
async def test_shared_provider_runtime_close_does_not_close_peer_scope() -> None:
    provider = _HostProvider()
    async with Runtime.open("peer", storage=PersistenceStorage.in_memory(), features=model_features(_FeatureModels(), _definitions()), metrics=provider) as peer:
        async with Runtime.open("first", storage=PersistenceStorage.in_memory(), features=model_features(_FeatureModels(), _definitions()), metrics=provider) as first:
            await first.agents.get().run("first", timeout_seconds=5)
        assert provider.scopes[1].closed
        assert not provider.scopes[0].closed
        await peer.agents.get().run("peer", timeout_seconds=5)
        assert provider.scopes[0].observations
    assert all(scope.closed for scope in provider.scopes)


@pytest.mark.asyncio
async def test_independent_task_engines_and_views_borrow_provider() -> None:
    provider = _HostProvider()
    task = Task("work", _work, effect_policy="none")
    async with Persistence.open("tasks", storage=PersistenceStorage.in_memory()) as persistence:
        async with TaskEngine.open(task, persistence=persistence, metrics=provider) as peer:
            async with TaskEngine.open(task, persistence=persistence, metrics=provider) as first:
                view = first.with_definitions()
                await view.close()
                assert len(provider.scopes) == 2
                assert not provider.scopes[1].closed
                run = await first.start(TaskGraph("first", (TaskNode("node", task=task),)), idempotency_key="first")
                assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
            assert provider.scopes[1].closed and not provider.scopes[0].closed
            run = await peer.start(TaskGraph("peer", (TaskNode("node", task=task),)), idempotency_key="peer")
            assert (await run.wait(timeout_seconds=5)).result.status is TaskStatus.SUCCEEDED
        assert persistence.ready
    assert all(scope.closed for scope in provider.scopes)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_kind", ("runtime", "task"))
async def test_provider_scope_close_failure_is_retryable(owner_kind: str) -> None:
    provider = _HostProvider()
    async with Persistence.open("retry-close", storage=PersistenceStorage.in_memory()) as persistence:
        if owner_kind == "runtime":
            owner = Runtime.attach(persistence=persistence, features=model_features(_FeatureModels(), _definitions()), metrics=provider)
            await owner.__aenter__()
        else:
            owner = TaskEngine(persistence=persistence, metrics=provider)
        scope = provider.scopes[0]
        scope.close_failures = 1
        try:
            with pytest.raises(OSError, match="cleanup incomplete"):
                await owner.close()
            assert not scope.closed
            assert persistence.ready
            await owner.close()
            assert scope.closed
        finally:
            await owner.close()


@pytest.mark.asyncio
async def test_provider_preserves_stable_observation_and_enriches_without_dataset_access() -> None:
    provider = _HostProvider()
    service = MetricsService(provider)
    service.configure_runtime_dimensions({"region": "west"})
    observation = _observation("stable")
    try:
        assert await service.try_record(observation)
        stored = provider.scopes[0].observations[0]
        assert stored.observation_id == observation.observation_id
        assert stored.occurred_at == observation.occurred_at
        assert stored.source_namespace == observation.source_namespace
        assert stored.tenant_id == observation.tenant_id
        assert stored.dimensions == {"context.region": "west"}
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_task_constructor_failure_leaves_unstarted_recorder_without_background_work(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.task.engine import _engine

    provider = _HostProvider()
    def fail(*args, **kwargs):
        raise RuntimeError("task dependency unavailable")
    async with Persistence.open("initialization", storage=PersistenceStorage.in_memory()) as persistence:
        before = asyncio.all_tasks()
        with monkeypatch.context() as patch:
            patch.setattr(_engine, "TaskExecutionService", fail)
            with pytest.raises(RuntimeError, match="task dependency unavailable"):
                TaskEngine(persistence=persistence, metrics=provider)
        assert asyncio.all_tasks() == before
        assert len(provider.scopes) == 1
        assert not provider.scopes[0].observations
        async with TaskEngine.open(persistence=persistence, metrics=provider):
            assert len(provider.scopes) == 2
        assert provider.scopes[1].closed
        assert persistence.ready


@pytest.mark.asyncio
async def test_runtime_initialization_failure_closes_only_its_provider_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.runtime import _factory

    provider = _HostProvider()
    def fail(*args, **kwargs):
        raise RuntimeError("runtime dependency unavailable")
    async with Persistence.open("initialize-runtime", storage=PersistenceStorage.in_memory()) as persistence:
        with monkeypatch.context() as patch:
            patch.setattr(_factory, "DefaultExecutionHistoryService", fail)
            with pytest.raises(RuntimeError, match="runtime dependency unavailable"):
                async with Runtime.attach(
                    persistence=persistence, features=model_features(_FeatureModels(), _definitions()), metrics=provider,
                ):
                    pytest.fail("initialization failure was ignored")
        assert len(provider.scopes) == 1
        assert provider.scopes[0].closed
        assert persistence.ready
        async with Runtime.attach(
            persistence=persistence, features=model_features(_FeatureModels(), _definitions()), metrics=provider,
        ) as runtime:
            await runtime.agents.get().run("retry", timeout_seconds=5)
        assert provider.scopes[1].closed


@pytest.mark.asyncio
async def test_cancelled_runtime_close_preserves_provider_cleanup_until_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _HostProvider()
    entered, release = asyncio.Event(), asyncio.Event()
    async with Persistence.open("cancel-close", storage=PersistenceStorage.in_memory()) as persistence:
        runtime = Runtime.attach(persistence=persistence, features=model_features(_FeatureModels(), _definitions()), metrics=provider)
        await runtime.__aenter__()
        scope = provider.scopes[0]
        original_close = scope.close
        async def blocking_close() -> None:
            entered.set()
            await release.wait()
            await original_close()
        monkeypatch.setattr(scope, "close", blocking_close)
        closing = asyncio.create_task(runtime.close())
        try:
            await asyncio.wait_for(entered.wait(), 2)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert not scope.closed
            assert persistence.ready
        finally:
            release.set()
            await runtime.close()
        assert scope.closed
