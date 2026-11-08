#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime input backends are acquired only with an immediate cleanup owner."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from linktools.ai.capability import FileSystem
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.runs import RuntimeAgentRunStore
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.workspace import LocalWorkspaceBackend, Workspace

if TYPE_CHECKING:
    from linktools.ai.observe import MetricRecorder
    from linktools.ai.workspace import WorkspaceResource


class _OwnedBackend(LocalWorkspaceBackend):
    def __init__(self, root: Path, resource: Path, *, close_failures: int = 0) -> None:
        super().__init__(root)
        self.resource = resource.open("wb")
        self.close_calls = 0
        self.close_failures = close_failures

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_failures:
            self.close_failures -= 1
            raise RuntimeError("input backend close failed")
        await super().close()
        self.resource.close()


class _FailingMetrics:
    def recorder(self) -> "MetricRecorder":
        raise ValueError("metrics initialization failed")


def _features(
    tmp_path: Path, instances: list[_OwnedBackend], *, workspace: bool = True,
    close_failures: int = 0,
) -> RuntimeFeatures[None]:
    def factory(root: Path, *, resources: "tuple[WorkspaceResource, ...]" = ()) -> LocalWorkspaceBackend:
        assert resources == ()
        backend = _OwnedBackend(root, tmp_path / "owned-resource", close_failures=close_failures)
        instances.append(backend)
        return backend

    features: RuntimeFeatures[None] = RuntimeFeatures(workspace_factory=factory)
    if workspace:
        features.capability(FileSystem(Workspace.load(tmp_path)))
    return features


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace", (False, True))
async def test_runtime_allocates_input_backend_only_for_workspace(tmp_path: Path, workspace: bool) -> None:
    instances: list[_OwnedBackend] = []
    runtime = Runtime.open("input-owner", storage=PersistenceStorage.in_memory(),
                           features=_features(tmp_path, instances, workspace=workspace))
    try:
        async with runtime:
            assert len(instances) == int(workspace)
            if instances:
                assert not instances[0].resource.closed
        await runtime.close()
        assert all(value.close_calls == 1 and value.resource.closed for value in instances)
    finally:
        for value in instances:
            value.resource.close()


@pytest.mark.asyncio
async def test_runtime_compile_failure_does_not_allocate_input_backend(tmp_path: Path) -> None:
    instances: list[_OwnedBackend] = []
    features = _features(tmp_path, instances)
    features.agent("default", model="missing")
    runtime = Runtime.open("input-owner", storage=PersistenceStorage.in_memory(), features=features)
    try:
        with pytest.raises(AIError) as raised:
            await runtime.__aenter__()
        assert raised.value.code is ErrorCode.MODEL_CONNECTION_NOT_FOUND
        await runtime.close()
        assert instances == []
    finally:
        for value in instances:
            value.resource.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_failures", (0, 1))
async def test_runtime_failed_entry_keeps_input_cleanup_retryable(tmp_path: Path, close_failures: int) -> None:
    instances: list[_OwnedBackend] = []
    async with Persistence.open("input-owner", storage=PersistenceStorage.in_memory()) as persistence:
        runtime = Runtime.attach(persistence=persistence,
            features=_features(tmp_path, instances, close_failures=close_failures), metrics=_FailingMetrics())
        try:
            with pytest.raises(ValueError, match="metrics initialization failed") as raised:
                await runtime.__aenter__()
            assert len(instances) == 1
            backend = instances[0]
            assert persistence.ready
            if close_failures:
                assert isinstance(raised.value.__cause__, RuntimeError)
                assert not backend.resource.closed
            else:
                assert backend.resource.closed
            await runtime.close()
            assert backend.close_calls == close_failures + 1 and backend.resource.closed
            assert persistence.ready
        finally:
            for value in instances:
                value.resource.close()


@pytest.mark.asyncio
async def test_runtime_cancelled_entry_closes_acquired_input_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[_OwnedBackend] = []
    entered = asyncio.Event()

    async def prepare_archives(store: RuntimeAgentRunStore) -> None:
        del store
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(RuntimeAgentRunStore, "prepare_archives", prepare_archives)
    async with Persistence.open("input-owner", storage=PersistenceStorage.in_memory()) as persistence:
        runtime = Runtime.attach(persistence=persistence, features=_features(tmp_path, instances))
        task = asyncio.create_task(runtime.__aenter__())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert len(instances) == 1 and not instances[0].resource.closed
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert instances[0].close_calls == 1 and instances[0].resource.closed
            await runtime.close()
            assert persistence.ready
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            await runtime.close()
            for value in instances:
                value.resource.close()


@pytest.mark.asyncio
async def test_runtime_input_backend_requires_close_support(tmp_path: Path) -> None:
    def factory(root: Path, *, resources: "tuple[WorkspaceResource, ...]" = ()) -> object:
        del root, resources
        return object()

    features: RuntimeFeatures[None] = RuntimeFeatures(workspace_factory=factory)
    features.capability(FileSystem(Workspace.load(tmp_path)))
    runtime = Runtime.open("input-owner", storage=PersistenceStorage.in_memory(), features=features)
    with pytest.raises(AIError) as raised:
        await runtime.__aenter__()
    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    assert raised.value.safe_details["reason"] == "workspace_cleanup_required"
    await runtime.close()
