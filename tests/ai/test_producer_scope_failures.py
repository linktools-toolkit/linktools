#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Producer recovery admission and failure cleanup preserve their boundaries."""

import pytest

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.task import RecoverGraphRequest
from .test_graph_bound_recovery import _graph_recovery


@pytest.mark.asyncio
async def test_graph_recovery_denied_actor_never_reaches_producer() -> None:
    async with _graph_recovery() as (_runtime, storage, service, backend, launcher, _authorization):
        before = await storage.execution.executions.get("execution-agent", tenant_id="default")
        with pytest.raises(AIError) as denied:
            await service.recover("bound-graph", RecoverGraphRequest(Principal("unapproved", "default"), "denied"))
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert backend.operations == []
        assert launcher.launches == []
        assert await storage.execution.executions.get("execution-agent", tenant_id="default") == before


@pytest.mark.asyncio
async def test_constructor_failure_waits_for_cleanup_and_preserves_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from dataclasses import replace
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.runtime import Runtime
    from linktools.ai.runtime import _factory, _runtime_service

    entered, release = asyncio.Event(), asyncio.Event()
    holders: list[Persistence] = []
    ready: list[bool] = []
    original_compose = _factory.compose_runtime_components

    async def compose(**kwargs: object) -> _factory._RuntimeComponents:
        components = await original_compose(**kwargs)

        async def close() -> None:
            entered.set()
            await release.wait()
            ready.append(holders[0].ready)
            await components.close_callback()
            raise RuntimeError("secondary cleanup error")

        return replace(components, close_callback=close)

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("original constructor failure")

    monkeypatch.setattr(_factory, "compose_runtime_components", compose)
    monkeypatch.setattr(_runtime_service, "Runtime", fail)

    async def run() -> None:
        async with Persistence.open("constructor-failure", storage=PersistenceStorage.in_memory()) as p:
            holders.append(p)
            async with Runtime.open(persistence=p, auto_recover=False):
                pytest.fail("constructor unexpectedly succeeded")

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 5)
    try:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert holders[0].ready
    finally:
        release.set()
    with pytest.raises(ValueError, match="original constructor failure"):
        await asyncio.wait_for(task, 5)
    assert ready == [True]
    assert not holders[0].ready
