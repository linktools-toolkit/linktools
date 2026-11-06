#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned declarations retain dependencies through producer cleanup."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.capabilities import CombinedCapability, PrefixTools
from pydantic_ai_harness.step_persistence import InMemoryStepStore, RunRecord, StepPersistence
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import AssetCapability, StepPersistenceCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageOverlay

from ._runtime_test_helpers import RuntimeUsageModels, model_features

pytestmark = pytest.mark.asyncio


def _assets(backend: InMemoryAssetBackend | None = None) -> AssetStore:
    return AssetStore(StorageOverlay(InMemoryAssetBackend() if backend is None else backend))


def _features(*capabilities: object) -> RuntimeFeatures:
    features = model_features(RuntimeUsageModels(), RuntimeFeatures())
    for capability in capabilities:
        features.capability(capability)
    return features


@pytest.mark.parametrize("attach", (False, True))
@pytest.mark.parametrize("initialized", (False, True))
async def test_runtime_owns_declared_assets_but_not_attached_persistence(attach: bool, initialized: bool) -> None:
    assets = _assets()
    if initialized:
        await assets.initialize()
    features = _features(AssetCapability(assets))
    async with Persistence.open("assets", storage=PersistenceStorage.in_memory()) as persistence:
        owner = (Runtime.attach(persistence=persistence, features=features) if attach else
                 Runtime.open("assets", storage=PersistenceStorage.in_memory(), features=features))
        async with owner as runtime:
            assert assets.ready
            assert runtime.features.asset_reader is not None
            assert (await runtime.agents.get().run("question")).result.output == {"text": "done"}
        assert not assets.ready
        assert persistence.ready


async def test_captured_assets_are_borrowed_and_runtime_reuses_capture() -> None:
    backend = InMemoryAssetBackend()
    assets = AssetStore(StorageOverlay(backend, writer=backend))
    try:
        await assets.initialize()
        key = AssetKey("rule", "shared.md")
        await assets.put(key, b"shared instructions")
        capture = await _features(AssetCapability(assets)).capture()
        async with Runtime.open("first", storage=PersistenceStorage.in_memory(), features=capture) as first:
            async with Runtime.open("second", storage=PersistenceStorage.in_memory(), features=capture) as second:
                assert first.features is second.features is capture
                assert assets.ready
            assert assets.ready
            assert await capture.asset_reader.get(key) == b"shared instructions"
            assert (await first.agents.get().run("still live")).result.output == {"text": "done"}
        assert assets.ready
    finally:
        await assets.close()
    assert not assets.ready


async def test_owned_asset_declaration_rejects_second_runtime_without_closing_first() -> None:
    assets = _assets()
    features = _features(AssetCapability(assets))
    async with Runtime.open("first", storage=PersistenceStorage.in_memory(), features=features) as first:
        with pytest.raises(AIError, match="already belongs"):
            async with Runtime.open("second", storage=PersistenceStorage.in_memory(), features=features):
                pytest.fail("ownership was acquired twice")
        assert assets.ready
        assert (await first.agents.get().run("still live")).result.output == {"text": "done"}
    assert not assets.ready


@pytest.mark.parametrize("cancelled", (False, True))
async def test_owned_assets_close_after_capture_failure_or_cancellation(cancelled: bool) -> None:
    assets = _assets()
    capability = AssetCapability(assets)
    entered = asyncio.Event()

    class Loader:
        async def load(self, context: object) -> tuple[object, ...]:
            entered.set()
            if cancelled:
                await asyncio.Event().wait()
            raise ValueError("capture failed")

    capability.loader("custom", Loader())
    runtime = Runtime.open("failed", storage=PersistenceStorage.in_memory(), features=_features(capability))
    task = asyncio.create_task(runtime.__aenter__())
    await entered.wait()
    if cancelled:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancelled else ValueError):
        await task
    assert not assets.ready
    await runtime.close()


async def test_asset_close_failure_preserves_owner_and_can_retry() -> None:
    class Backend(InMemoryAssetBackend):
        fail = True

        async def close(self) -> None:
            if self.fail:
                raise ValueError("asset close failed")
            await super().close()

    backend = Backend()
    assets = _assets(backend)
    features = _features(AssetCapability(assets))
    runtime = Runtime.open("retry", storage=PersistenceStorage.in_memory(), features=features)
    await runtime.__aenter__()
    with pytest.raises(ValueError, match="asset close failed"):
        await runtime.close()
    with pytest.raises(AIError, match="already belongs"):
        async with Runtime.open("second", storage=PersistenceStorage.in_memory(), features=features):
            pytest.fail("failed cleanup released ownership")
    backend.fail = False
    await runtime.close()
    assert not assets.ready


@pytest.mark.parametrize("sql", (False, True))
async def test_default_step_capability_uses_execution_storage_and_borrows_engine(tmp_path: Path, sql: bool) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}") if sql else None
    if engine is not None:
        await provision_database(engine)
    storage = PersistenceStorage.sql(engine) if sql else PersistenceStorage.from_root(tmp_path)
    try:
        async with Runtime.open("steps", storage=storage, features=_features(StepPersistenceCapability())) as runtime:
            assert (await runtime.agents.get().run("question")).result.output == {"text": "done"}
        async with Persistence.open("steps", storage=storage) as persistence:
            steps = await persistence.create_step_store()
            try:
                runs = await steps.list_runs()
                assert runs
                assert all([await steps.latest_snapshot(run_id=run.run_id) for run in runs])
            finally:
                await steps.close()
        assert not (tmp_path / "steps.db").exists()
        if engine is not None:
            async with engine.connect() as connection:
                assert await connection.scalar(text("SELECT 1")) == 1
    finally:
        if engine is not None:
            await engine.dispose()


async def test_explicit_step_store_is_borrowed_inside_native_containers() -> None:
    class Store(InMemoryStepStore):
        closed = False

        async def close(self) -> None:
            self.closed = True

    steps = Store()
    feature = PrefixTools(CombinedCapability([StepPersistenceCapability(store=steps)]), prefix="test_", id="wrapped-steps")
    async with Runtime.open("borrowed", storage=PersistenceStorage.in_memory(), features=_features(feature)) as runtime:
        await runtime.agents.get().run("question")
    assert not steps.closed
    assert await steps.list_runs()
    await steps.register_run(RunRecord(run_id="after-runtime"))


async def test_default_step_capability_rejects_memory_instead_of_inventing_durability() -> None:
    with pytest.raises(AIError) as failure:
        async with Runtime.open("memory", storage=PersistenceStorage.in_memory(),
                                features=_features(StepPersistenceCapability())):
            pytest.fail("missing SQL route was accepted")
    assert failure.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING


async def test_project_and_native_step_declarations_conflict() -> None:
    with pytest.raises(AIError) as failure:
        async with Runtime.open("duplicate", storage=PersistenceStorage.in_memory(), features=_features(
            StepPersistenceCapability(store=InMemoryStepStore()), StepPersistence(store=InMemoryStepStore()),
        )):
            pytest.fail("duplicate persistence was accepted")
    assert failure.value.code is ErrorCode.CAPABILITY_CONFLICT


async def test_created_step_adapter_closes_after_later_composition_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    closed = asyncio.Event()
    create = Persistence.create_step_store

    async def create_steps(persistence: Persistence, **kwargs: object):
        steps = await create(persistence, **kwargs)
        close = steps.close

        async def close_steps() -> None:
            await close()
            closed.set()

        steps.close = close_steps
        return steps

    async def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("run store failed")

    monkeypatch.setattr(Persistence, "create_step_store", create_steps)
    monkeypatch.setattr(Persistence, "create_agent_run_store", fail)
    async with Persistence.open("partial", storage=PersistenceStorage.from_root(tmp_path)) as persistence:
        with pytest.raises(ValueError, match="run store failed"):
            async with Runtime.attach(persistence=persistence, features=_features(StepPersistenceCapability())):
                pytest.fail("failed composition entered")
        assert closed.is_set()
        assert persistence.ready


async def test_producer_close_failure_keeps_owned_dependencies_until_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from linktools.ai.persistence.runs import RuntimeAgentRunStore

    events: list[str] = []

    class Backend(InMemoryAssetBackend):
        async def close(self) -> None:
            events.append("assets")
            await super().close()

    assets = _assets(Backend())
    create = Persistence.create_step_store
    preflight = RuntimeAgentRunStore.preflight_close
    fail = True

    async def create_steps(persistence: Persistence, **kwargs: object):
        steps = await create(persistence, **kwargs)
        close = steps.close

        async def close_steps() -> None:
            events.append("steps")
            await close()

        steps.close = close_steps
        return steps

    async def preflight_close(store: RuntimeAgentRunStore) -> None:
        if fail:
            raise ValueError("producer close failed")
        events.append("producer")
        await preflight(store)

    monkeypatch.setattr(Persistence, "create_step_store", create_steps)
    monkeypatch.setattr(RuntimeAgentRunStore, "preflight_close", preflight_close)
    async with Persistence.open("close-order", storage=PersistenceStorage.from_root(tmp_path)) as persistence:
        runtime = Runtime.attach(persistence=persistence, features=_features(
            AssetCapability(assets), StepPersistenceCapability(),
        ))
        await runtime.__aenter__()
        await runtime.agents.get().run("question")
        with pytest.raises(ValueError, match="producer close failed"):
            await runtime.close()
        assert events == []
        assert assets.ready and persistence.ready
        fail = False
        await runtime.close()
        assert events == ["producer", "steps", "assets"]
        assert not assets.ready and persistence.ready


async def test_owned_assets_clean_partial_initialization() -> None:
    class Backend(InMemoryAssetBackend):
        allocated = False

        async def initialize(self) -> None:
            self.allocated = True
            raise ValueError("initialize failed")

        async def close(self) -> None:
            self.allocated = False

    backend = Backend()
    assets = _assets(backend)
    runtime = Runtime.open("partial-assets", storage=PersistenceStorage.in_memory(),
                           features=_features(AssetCapability(assets)))
    with pytest.raises(ValueError, match="initialize failed"):
        await runtime.__aenter__()
    assert not assets.ready and not backend.allocated
    await runtime.close()


async def test_step_declaration_requires_runtime_binding() -> None:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    with pytest.raises(AIError) as failure:
        agent = Agent(TestModel(), capabilities=[StepPersistenceCapability()])
        await agent.run("question")
    assert failure.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


async def test_default_steps_recover_after_runtime_handoff_and_reopen(tmp_path: Path) -> None:
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.core import ExecutionStatus
    from ._runtime_test_helpers import _Models

    entered = asyncio.Event()

    async def blocked(messages, info):
        entered.set()
        await asyncio.Event().wait()
        yield "unreachable"

    declaration = StepPersistenceCapability()
    features = _features(declaration)
    features.model("default", _Models(FunctionModel(stream_function=blocked)))
    storage = PersistenceStorage.from_root(tmp_path)
    async with Runtime.open("handoff", storage=storage, features=features) as owner:
        execution = await owner.agents.get().start("question", idempotency_key="input")
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(owner.handoff(), 5)
    assert declaration.store is None

    features.model("default", RuntimeUsageModels())
    async with Runtime.open("handoff", storage=storage, features=features, auto_recover=False) as successor:
        recovered = await successor.executions.recover(
            execution.execution_id, principal=successor.default_principal, idempotency_key="recover",
        )
        assert recovered.execution_id == execution.execution_id
        await successor.executions.wait(
            execution.execution_id, principal=successor.default_principal, timeout_seconds=5,
        )
        result = await successor.executions.inspect(execution.execution_id, principal=successor.default_principal)
        assert result.status is ExecutionStatus.SUCCEEDED
    assert declaration.store is None
