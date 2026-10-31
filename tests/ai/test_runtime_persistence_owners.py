#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime producer admission and its borrowed persistence lifetime."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _FeatureModels


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_recover", (False, True))
async def test_filesystem_persistence_rejects_second_agent_producer_before_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, auto_recover: bool,
) -> None:
    async with Persistence.open("producer", storage=PersistenceStorage.filesystem(tmp_path)) as persistence:
        first = Runtime.attach(persistence=persistence, features=model_features(_FeatureModels()))
        second = Runtime.attach(
            persistence=persistence, features=model_features(_FeatureModels()), auto_recover=auto_recover,
        )
        calls = 0
        original = persistence.create_agent_run_store

        async def create():
            nonlocal calls
            calls += 1
            return await original()

        await first.__aenter__()
        monkeypatch.setattr(persistence, "create_agent_run_store", create)
        try:
            with pytest.raises(AIError) as caught:
                await second.__aenter__()
            assert caught.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert calls == 0
            assert persistence.ready
        finally:
            await second.close()
            await first.close()
        async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
            assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_shared_persistence_still_accepts_independent_agent_runtimes(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.from_root(tmp_path)
    async with Persistence.open("shared-producer", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())), Runtime.attach(
            persistence=persistence, features=model_features(_FeatureModels()),
        ):
            assert persistence.ready


@pytest.mark.asyncio
async def test_task_only_runtimes_do_not_take_filesystem_agent_exclusivity(tmp_path: Path) -> None:
    async with Persistence.open("task-producer", storage=PersistenceStorage.filesystem(tmp_path)) as persistence:
        async with Runtime.attach(persistence=persistence), Runtime.attach(persistence=persistence), Runtime.attach(
            persistence=persistence, features=model_features(_FeatureModels()),
        ):
            assert persistence.ready


@pytest.mark.asyncio
async def test_persistence_close_rejects_live_runtime_without_closing_resources() -> None:
    async with Persistence.open("borrowed-producer", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence):
            with pytest.raises(AIError) as caught:
                await persistence.close()
            assert caught.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert persistence.ready
        assert persistence.ready


@pytest.mark.asyncio
async def test_failed_borrowed_runtime_cleanup_retains_exclusive_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open("retry-producer", storage=PersistenceStorage.filesystem(tmp_path)) as persistence:
        first = Runtime.attach(persistence=persistence, features=model_features(_FeatureModels()))
        await first.__aenter__()
        original = first._close_agent
        calls = 0

        async def close_agent() -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("producer cleanup failed")
            await original()

        monkeypatch.setattr(first, "_close_agent", close_agent)
        try:
            with pytest.raises(RuntimeError, match="producer cleanup failed"):
                await first.close()
            with pytest.raises(AIError):
                await persistence.close()
            assert persistence.ready
            with pytest.raises(AIError):
                async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
                    pytest.fail("second producer was admitted")
        finally:
            await first.close()
        assert calls == 2
        async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
            pass


@pytest.mark.asyncio
async def test_owned_runtime_retains_permit_until_resource_close_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("owned-producer", storage=PersistenceStorage.from_root(tmp_path))
    await runtime.__aenter__()
    engine = await runtime.tasks.bind()
    persistence = engine.persistence
    original = persistence._run_close_actions
    calls = 0

    async def close_resources() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("physical close failed")
        await original()

    monkeypatch.setattr(persistence, "_run_close_actions", close_resources)
    try:
        with pytest.raises(RuntimeError, match="physical close failed"):
            await runtime.close()
        assert not persistence.ready
        assert len(persistence._runtime_owners) == 1
        with pytest.raises(AIError):
            Runtime.attach(persistence=persistence)
    finally:
        await runtime.close()
    assert calls == 2
    assert not persistence._runtime_owners


@pytest.mark.asyncio
async def test_owned_runtime_cannot_close_a_live_peer_runtime() -> None:
    owner = Runtime.open("owned-peer", storage=PersistenceStorage.in_memory())
    await owner.__aenter__()
    owner_engine = await owner.tasks.bind()
    persistence = owner_engine.persistence
    peer = Runtime.attach(persistence=persistence)
    await peer.__aenter__()
    try:
        with pytest.raises(AIError) as caught:
            await owner.close()
        assert caught.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert persistence.ready and len(persistence._runtime_owners) == 2
        assert (await peer.tasks.bind()).persistence is persistence
    finally:
        await peer.close()
        await owner.close()
    assert not persistence.ready and not persistence._runtime_owners


@pytest.mark.asyncio
async def test_cancelled_close_keeps_runtime_permission_until_owned_cleanup_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with Persistence.open("cancel-close", storage=PersistenceStorage.filesystem(tmp_path)) as persistence:
        runtime = Runtime.attach(persistence=persistence, features=model_features(_FeatureModels()))
        await runtime.__aenter__()
        entered, release = asyncio.Event(), asyncio.Event()
        original = runtime._close_agent

        async def close_agent() -> None:
            entered.set()
            await release.wait()
            await original()

        monkeypatch.setattr(runtime, "_close_agent", close_agent)
        closing = asyncio.create_task(runtime.close())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert persistence.ready and len(persistence._runtime_owners) == 1
            with pytest.raises(AIError):
                await persistence.close()
            with pytest.raises(AIError):
                async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
                    pytest.fail("second producer was admitted during cleanup")
        finally:
            release.set()
            await runtime.close()
        assert not persistence._runtime_owners


@pytest.mark.asyncio
async def test_composition_cancellation_releases_only_its_own_runtime_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    original = RuntimeFeatures.capture
    features = model_features(_FeatureModels())

    async def capture(current: RuntimeFeatures):
        if current is features:
            entered.set()
            await release.wait()
        return await original(current)

    monkeypatch.setattr(RuntimeFeatures, "capture", capture)
    async with Persistence.open("cancel-compose", storage=PersistenceStorage.filesystem(tmp_path)) as persistence:
        async with Runtime.attach(persistence=persistence):
            runtime = Runtime.attach(persistence=persistence, features=features)
            starting = asyncio.create_task(runtime.__aenter__())
            try:
                await asyncio.wait_for(entered.wait(), 5)
                with pytest.raises(AIError):
                    await persistence.close()
                with pytest.raises(AIError):
                    async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
                        pytest.fail("second producer was admitted during composition")
                assert len(persistence._runtime_owners) == 2
                starting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await starting
                assert persistence.ready and len(persistence._runtime_owners) == 1
            finally:
                release.set()
                await runtime.close()
            async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
                pass


@pytest.mark.asyncio
async def test_precreated_runtime_cannot_attach_after_persistence_close() -> None:
    persistence = Persistence(PersistenceStorage.in_memory())
    await persistence.initialize(namespace="closed-admission", tenant_id="default")
    runtime = Runtime.attach(persistence=persistence)
    await persistence.close()
    with pytest.raises(AIError) as caught:
        await runtime.__aenter__()
    assert caught.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    assert not persistence._runtime_owners


@pytest.mark.asyncio
async def test_failed_composition_and_failed_cleanup_keep_runtime_permission_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._factory as factory
    from linktools.ai.inputs import ExecutionInputMaterializer

    history_reader = factory._execution_history_reader
    close_input = ExecutionInputMaterializer.close
    build_calls = 0
    close_calls = 0

    def fail_build_once(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            raise RuntimeError("composition failed")
        return history_reader(*args, **kwargs)

    async def fail_cleanup_once(current: ExecutionInputMaterializer) -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("input cleanup failed")
        await close_input(current)

    monkeypatch.setattr(factory, "_execution_history_reader", fail_build_once)
    monkeypatch.setattr(ExecutionInputMaterializer, "close", fail_cleanup_once)
    async with Persistence.open("failed-compose", storage=PersistenceStorage.filesystem(tmp_path)) as persistence:
        runtime = Runtime.attach(persistence=persistence, features=model_features(_FeatureModels()))
        try:
            with pytest.raises(RuntimeError, match="composition failed") as caught:
                await runtime.__aenter__()
            assert str(caught.value.__cause__) == "input cleanup failed"
            assert persistence.ready and len(persistence._runtime_owners) == 1
            with pytest.raises(AIError):
                await persistence.close()
            with pytest.raises(AIError):
                async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
                    pytest.fail("second producer was admitted after incomplete initialization cleanup")
            assert build_calls == close_calls == 1
        finally:
            await runtime.close()
        assert close_calls == 2 and not persistence._runtime_owners
        async with Runtime.attach(persistence=persistence, features=model_features(_FeatureModels())):
            assert build_calls == 2
