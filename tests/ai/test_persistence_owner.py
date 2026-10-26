#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neutral storage ownership and scope contracts."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionContext
from linktools.ai.persistence import Persistence, PersistenceStorage


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "filesystem", "sqlite"])
async def test_persistence_owner_binds_one_scope_and_closes_resources(
    tmp_path: Path, backend: str,
) -> None:
    if backend == "memory":
        storage = PersistenceStorage.in_memory()
    elif backend == "filesystem":
        storage = PersistenceStorage.filesystem(tmp_path / "state")
    else:
        storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite")
    async with Persistence.open("audit", storage=storage, tenant_id="tenant") as persistence:
        assert persistence.ready
        assert persistence.namespace == "audit"
        assert persistence.tenant_id == "tenant"
        assert persistence.execution.executions.tenant_id == "tenant"
        assert persistence.task.admissions is not None
    assert not persistence.ready


@pytest.mark.asyncio
async def test_persistence_owner_closes_after_body_failure() -> None:
    with pytest.raises(ValueError, match="body failed"):
        async with Persistence.open("audit", storage=PersistenceStorage.in_memory()) as persistence:
            raise ValueError("body failed")
    assert not persistence.ready


@pytest.mark.asyncio
async def test_persistence_owner_closes_after_body_cancellation() -> None:
    entered = asyncio.Event()
    holders: list[Persistence] = []

    async def run() -> None:
        async with Persistence.open("audit", storage=PersistenceStorage.in_memory()) as persistence:
            holders.append(persistence)
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not holders[0].ready


def test_persistence_owner_execution_context_preserves_application_and_observation_metadata() -> None:
    application = object()
    context = ExecutionContext(application, correlation={"request": "one"}, metric_dimensions={"site": "office"})
    child = context.overlay({"request": "two"})
    assert child.app is application
    assert child.correlation == {"request": "two"}
    assert context.correlation == {"request": "one"}
    assert child.metric_dimensions == {"site": "office"}


@pytest.mark.asyncio
async def test_persistence_owner_runtime_borrows_scope_without_closing_it() -> None:
    from linktools.ai.runtime import Runtime

    async with Persistence.open("audit", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as persistence:
        async with Runtime.attach(persistence=persistence, context=ExecutionContext("application"), auto_recover=False) as runtime:
            assert runtime.namespace == "audit"
            assert runtime.tenant_id == "tenant"
            assert runtime.context.app == "application"
        assert persistence.ready
        async with Runtime.attach(persistence=persistence, auto_recover=False):
            pass
        assert persistence.ready


@pytest.mark.asyncio
async def test_persistence_owner_runtime_failure_does_not_close_borrowed_storage() -> None:
    from linktools.ai.runtime import Runtime

    async with Persistence.open("audit", storage=PersistenceStorage.in_memory()) as persistence:
        with pytest.raises(ValueError, match="body failed"):
            async with Runtime.attach(persistence=persistence, auto_recover=False):
                raise ValueError("body failed")
        assert persistence.ready
        with pytest.raises(TypeError):
            async with Runtime.attach(persistence=persistence, features=object(), auto_recover=False):
                pytest.fail("invalid features admitted")
        assert persistence.ready


@pytest.mark.asyncio
async def test_persistence_owner_runtime_cancellation_leaves_borrowed_storage_open() -> None:
    from linktools.ai.runtime import Runtime

    async with Persistence.open("audit", storage=PersistenceStorage.in_memory()) as persistence:
        entered = asyncio.Event()

        async def run() -> None:
            async with Runtime.attach(persistence=persistence, auto_recover=False):
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert persistence.ready


@pytest.mark.asyncio
async def test_persistence_owner_exposes_borrowed_coordination_to_independent_readers() -> None:
    from linktools.ai.storage import InMemoryCacheBackend

    backend = InMemoryCacheBackend(max_bytes=1048576)
    async with Persistence.open("audit", storage=PersistenceStorage.in_memory(coordination=backend)) as persistence:
        assert persistence.coordination_backend is backend


@pytest.mark.asyncio
async def test_persistence_owner_runtime_composition_failure_preserves_borrowed_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.runtime import _factory

    def fail_execution(*args: object, **kwargs: object) -> None:
        raise ValueError("execution construction failed")

    async with Persistence.open("audit", storage=PersistenceStorage.in_memory()) as persistence:
        monkeypatch.setattr(_factory, "DefaultExecutionService", fail_execution)
        with pytest.raises(ValueError, match="execution construction failed"):
            async with Runtime.attach(features=RuntimeFeatures(), persistence=persistence, auto_recover=False):
                pytest.fail("failed execution service admitted")
        assert persistence.ready
        assert persistence.execution.executions.tenant_id == "default"


@pytest.mark.asyncio
async def test_persistence_owner_isolates_runtime_callbacks_revisions_and_close() -> None:
    from collections.abc import Mapping
    from contextlib import AsyncExitStack
    from copy import copy
    from hashlib import sha256

    from pydantic_ai.capabilities import AbstractCapability
    from pydantic_ai.tools import RunContext

    from linktools.ai.capability import AgentContext
    from linktools.ai.core import CommitPolicy, ExecutionStatus, JsonValue
    from linktools.ai.persistence.capabilities import CapabilityWriter
    from linktools.ai.persistence.state import StateTransaction, StoredRecord
    from linktools.ai.runtime import Runtime
    from ._runtime_test_helpers import RuntimeUsageModels, model_features

    class Audit(AbstractCapability[AgentContext[None]]):
        id = "audit"
        commit = CommitPolicy.terminal()

        def __init__(self, revision: int) -> None:
            self.revision = revision
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.writer: CapabilityWriter | None = None

        def bind_persistence(self, writer: CapabilityWriter) -> "Audit":
            bound = copy(self)
            bound.writer = writer
            return bound

        async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
            assert self.writer is not None
            await self.writer.submit(key="audit", payload={
                "execution": ctx.deps.execution_id, "revision": self.revision,
            })
            self.entered.set()
            await self.release.wait()

        async def persist_in_transaction(self, tx: StateTransaction, payload: Mapping[str, JsonValue]) -> None:
            key = sha256(str(payload["execution"]).encode()).digest()
            await tx.insert_record(StoredRecord(
                key, key, None, "audit", str(payload["execution"]), None,
                0, None, 0, None, dict(payload),
            ))

    async with Persistence.open("audit", storage=PersistenceStorage.in_memory()) as persistence:
        first, second = Audit(1), Audit(2)
        first_features, second_features = model_features(RuntimeUsageModels()), model_features(RuntimeUsageModels())
        first_features.capability(first, revision=1)
        second_features.capability(second, revision=2)
        async with AsyncExitStack() as stack:
            second_runtime = await stack.enter_async_context(Runtime.attach(
                persistence=persistence, features=second_features, auto_recover=False,
            ))
            async with Runtime.attach(persistence=persistence, features=first_features, auto_recover=False) as first_runtime:
                assert first_runtime._run_store is not second_runtime._run_store
                assert first_runtime._run_store is not persistence.run_store
                assert first_runtime._run_store.capability_revisions == {"audit": 1}
                assert second_runtime._run_store.capability_revisions == {"audit": 2}
                second_execution = await second_runtime.agents.get().start("second")
                await asyncio.wait_for(second.entered.wait(), 5)
                first_execution = await first_runtime.agents.get().start("first")
                await asyncio.wait_for(first.entered.wait(), 5)
                first.release.set()
                assert (await first_execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            assert persistence.ready
            assert persistence.run_store.capability_writes is None
            assert not persistence.run_store.capability_revisions
            store = await persistence.capability_store("audit")
            pending_key = sha256(second_execution.execution_id.encode()).digest()
            assert await store.read(lambda tx: tx.get_record(pending_key)) is None
            second.release.set()
            assert (await second_execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            third = await second_runtime.agents.get().start("after peer close")
            assert (await third.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
            store = await persistence.capability_store("audit")
            for execution, revision in ((first_execution, 1), (second_execution, 2), (third, 2)):
                key = sha256(execution.execution_id.encode()).digest()
                record = await store.read(lambda tx, key=key: tx.get_record(key))
                assert record is not None
                assert record.data["revision"] == revision


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["agent", "task", "evaluation"])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_producer_scope_settles_cancelled_cleanup_before_storage_closes(
    tmp_path: Path, producer: str, backend: str,
) -> None:
    from contextlib import AsyncExitStack
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.task_engine import TaskEngine
    from linktools.ai.evaluation_engine import Evaluations

    entered, release = asyncio.Event(), asyncio.Event()
    observed: list[bool] = []
    holders: list[Persistence] = []
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "close.sqlite"))

    async def run() -> None:
        async with Persistence.open("producer-cleanup", storage=storage) as persistence:
            holders.append(persistence)
            async with AsyncExitStack() as stack:
                if producer == "agent":
                    owner = await stack.enter_async_context(Runtime.attach(
                        persistence=persistence, features=RuntimeFeatures(), auto_recover=False,
                    ))
                    target, attribute = owner._agent_lifecycle, "close"
                else:
                    engine = await stack.enter_async_context(TaskEngine.open(persistence=persistence))
                    if producer == "evaluation":
                        owner = await stack.enter_async_context(Evaluations.open(engine))
                        target, attribute = owner, "_close_trial_scopes"
                    else:
                        target, attribute = engine._graph_service, "preflight_close"
                original = getattr(target, attribute)

                async def delayed_cleanup() -> None:
                    entered.set()
                    await release.wait()
                    observed.append(persistence.ready)
                    await original()

                setattr(target, attribute, delayed_cleanup)

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 5)
    try:
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert holders[0].ready
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    assert observed == [True]
    assert not holders[0].ready
