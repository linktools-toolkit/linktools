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
@pytest.mark.parametrize("cancel_cleanup", [False, True])
async def test_constructor_failure_waits_for_cleanup_and_preserves_original(
    monkeypatch: pytest.MonkeyPatch, cancel_cleanup: bool,
) -> None:
    import asyncio
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.runtime import _factory

    entered, release = asyncio.Event(), asyncio.Event()
    holders: list[Persistence] = []
    runtimes: list[Runtime] = []
    primary = ValueError("original constructor failure")
    secondary = RuntimeError("secondary cleanup error")
    ready: list[bool] = []
    original_close = _factory._AgentLifecycle.close

    async def close(lifecycle: _factory._AgentLifecycle) -> None:
        entered.set()
        await release.wait()
        ready.append(holders[0].ready)
        await original_close(lifecycle)
        raise secondary

    def fail(*args: object, **kwargs: object) -> None:
        raise primary

    monkeypatch.setattr(_factory._AgentLifecycle, "close", close)
    monkeypatch.setattr(Runtime, "_bind_agent_components", fail)

    async def run() -> None:
        persistence = Persistence(PersistenceStorage.in_memory())
        await persistence.initialize(namespace="constructor-failure", tenant_id="default")
        holders.append(persistence)
        runtime = Runtime.attach(features=RuntimeFeatures(), persistence=persistence, auto_recover=False)
        runtimes.append(runtime)
        async with runtime:
            pytest.fail("constructor unexpectedly succeeded")

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 5)
    try:
        if cancel_cleanup:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
        assert not task.done()
        assert holders[0].ready
    finally:
        release.set()
    with pytest.raises(ValueError, match="original constructor failure") as failed:
        await asyncio.wait_for(task, 5)
    assert failed.value is primary
    if cancel_cleanup:
        assert isinstance(failed.value.__cause__, asyncio.CancelledError)
    else:
        assert failed.value.__cause__ is secondary
    assert ready == [True]
    owners = tuple(holders[0]._runtime_owners)
    assert len(owners) == 1 and holders[0].ready
    with pytest.raises(AIError) as blocked:
        await holders[0].close()
    assert blocked.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    assert holders[0].ready and tuple(holders[0]._runtime_owners) == owners
    monkeypatch.setattr(_factory._AgentLifecycle, "close", original_close)
    await asyncio.wait_for(runtimes[0].close(), 5)
    assert holders[0].ready and not holders[0]._runtime_owners
    await holders[0].close()
    assert not holders[0].ready
