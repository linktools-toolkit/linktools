#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History and portable snapshots borrow one independently owned persistence."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from linktools.ai.core import Principal
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.history import History
from linktools.ai.snapshot import Snapshot
from linktools.ai.storage import InMemoryObjectStore


class _OfflineGuard:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
async def test_history_snapshot_borrow_persistence_without_execution_producers(
    tmp_path: Path, backend: str,
) -> None:
    storage = (PersistenceStorage.filesystem(tmp_path / "source") if backend == "filesystem"
               else PersistenceStorage.sqlite(tmp_path / "source.sqlite"))
    principal = Principal("reader", "tenant", "service")
    objects = InMemoryObjectStore("portable")
    limits = SnapshotLimits(10000, 10 * 1024 * 1024)
    async with Persistence.open("history", storage=storage, tenant_id="tenant"):
        pass
    async with Persistence.open("history", storage=storage, tenant_id="tenant", read_only=True) as persistence:
        async with History.open(persistence=persistence) as history:
            assert (await history.list_sessions(principal=principal)).items == ()
        assert persistence.ready
        reference = await Snapshot.create(
            persistence=persistence, object_store=objects,
            exclusive=_OfflineGuard(), limits=limits,
        )
        assert persistence.ready
        restored = await Snapshot.restore(
            reference, object_store=objects, target=tmp_path / "restored",
            namespace="history", tenant_id="tenant", limits=limits,
        )
        async with restored.open_history() as history:
            assert (await history.list_sessions(principal=principal)).items == ()
        assert persistence.ready
    assert not persistence.ready


@pytest.mark.asyncio
async def test_history_body_failure_does_not_close_borrowed_persistence() -> None:
    async with Persistence.open("history", storage=PersistenceStorage.in_memory()) as persistence:
        with pytest.raises(ValueError, match="reader failed"):
            async with History.open(persistence=persistence):
                raise ValueError("reader failed")
        assert persistence.ready
        async with History.open(persistence=persistence) as history:
            assert (await history.list_sessions(principal=Principal("reader", "default", "service"))).items == ()


@pytest.mark.asyncio
async def test_history_cancellation_keeps_borrowed_persistence_usable() -> None:
    import asyncio

    entered = asyncio.Event()
    async with Persistence.open("history", storage=PersistenceStorage.in_memory()) as persistence:
        async def read_until_cancelled() -> None:
            async with History.open(persistence=persistence):
                entered.set()
                await asyncio.Event().wait()

        reader = asyncio.create_task(read_until_cancelled())
        await entered.wait()
        reader.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reader
        assert persistence.ready
        async with History.open(persistence=persistence) as history:
            assert (await history.list_sessions(principal=Principal("reader", "default", "service"))).items == ()


@pytest.mark.asyncio
async def test_history_retains_tenant_authorization_without_a_runtime() -> None:
    from linktools.ai.errors import AIError, ErrorCode

    async with Persistence.open("history", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as persistence:
        async with History.open(persistence=persistence) as history:
            with pytest.raises(AIError) as rejected:
                await history.list_sessions(principal=Principal("reader", "other", "service"))
            assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_snapshot_limit_failure_keeps_borrowed_persistence_usable(tmp_path: Path) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open("history", storage=storage):
        pass
    async with Persistence.open("history", storage=storage, read_only=True) as persistence:
        with pytest.raises(AIError) as rejected:
            await Snapshot.create(
                persistence=persistence, object_store=InMemoryObjectStore("portable"),
                exclusive=_OfflineGuard(), limits=SnapshotLimits(1, 1),
            )
        assert rejected.value.code is ErrorCode.SNAPSHOT_UNSUPPORTED
        assert persistence.ready


@pytest.mark.asyncio
async def test_independent_history_reads_only_current_shared_pending_owner() -> None:
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.persistence.coordination import PendingTurnCoordination
    from linktools.ai.storage import InMemoryCacheBackend
    from .test_history_projection_conformance import _record

    cache = InMemoryCacheBackend(max_bytes=2**20)
    writer = PendingTurnCoordination.select(cache, namespace="history", tenant_id="tenant")
    assert writer is not None
    record = _record(ExecutionStatus.STARTED, 1)
    claim = await writer.claim(
        session_key="session", request_key="request", request_digest="digest",
        execution_id=record.execution_id, owner_id="producer", lease_seconds=30,
    )
    assert claim is not None and claim.lease is not None
    await writer.publish(record, claim)
    principal = Principal(record.principal_id, "tenant", record.principal_kind)
    storage = PersistenceStorage.in_memory(coordination=cache)
    async with Persistence.open("history", storage=storage, tenant_id="tenant") as persistence:
        async with History.open(persistence=persistence) as history:
            info = await history.inspect_execution(record.execution_id, principal=principal, include_usage=False)
            assert info.status is ExecutionStatus.STARTED
            assert await writer.release(claim.lease)
            with pytest.raises(AIError) as rejected:
                await history.inspect_execution(record.execution_id, principal=principal, include_usage=False)
            assert rejected.value.code is ErrorCode.AUTHORIZATION_DENIED
        await cache.put("borrowed-cache", b"still open")
        assert await cache.get("borrowed-cache") == b"still open"


@pytest.mark.asyncio
async def test_history_closes_only_its_own_model_subscriptions() -> None:
    class Subscription:
        generation = 0
        closed = False

        async def wait(self, after_generation: int) -> int:
            return self.generation

        async def close(self) -> None:
            self.closed = True

    class ReadService:
        async def subscribe_model_interactions(self, execution_id: str, *, principal: Principal):
            return sources[execution_id]

    sources = {"one": Subscription(), "two": Subscription()}
    service = ReadService()
    first = History(service, tenant_id="tenant")
    second = History(service, tenant_id="tenant")
    principal = Principal("reader", "tenant", "service")
    first_subscription = await first.subscribe_model_interactions("one", principal=principal)
    await second.subscribe_model_interactions("two", principal=principal)
    await first.close()
    assert sources["one"].closed
    assert not sources["two"].closed
    await first_subscription.close()
    await second.close()
    assert sources["two"].closed


@pytest.mark.asyncio
async def test_history_shutdown_finishes_subscription_cleanup_after_cancellation() -> None:
    import asyncio

    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    class Subscription:
        generation = 0
        closed = False

        async def wait(self, after_generation: int) -> int:
            return self.generation

        async def close(self) -> None:
            close_started.set()
            await finish_close.wait()
            self.closed = True

    source = Subscription()

    class ReadService:
        async def subscribe_model_interactions(self, execution_id: str, *, principal: Principal):
            return source

    history = History(ReadService(), tenant_id="tenant")
    await history.subscribe_model_interactions("one", principal=Principal("reader", "tenant", "service"))
    closing = asyncio.create_task(history.close())
    await close_started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    finish_close.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert source.closed
    await history.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_independent_history_does_not_claim_another_producers_staging(
    tmp_path: Path, backend: str,
) -> None:
    import asyncio

    from pydantic_ai.messages import ModelResponse, TextPart

    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime import Runtime, RuntimeFeatures

    from ._runtime_test_helpers import _UsageFunctionModel, model_features
    from .test_live_history_readback_integration import _Models

    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("history-reader-visibility")
    features.agent("default", model="default", allow_tools=())
    storage = (PersistenceStorage.in_memory() if backend == "memory"
               else PersistenceStorage.sqlite(tmp_path / "state.sqlite"))
    async with (
        Persistence.open("history-reader-visibility", storage=storage) as persistence,
        Runtime.attach(persistence=persistence,
                     features=model_features(_Models(_UsageFunctionModel(respond)), features)) as producer,
        History.open(persistence=persistence) as independent,
    ):
        execution = await producer.agents.get().start("producer-local prompt")
        principal = producer.default_principal
        try:
            await asyncio.wait_for(entered.wait(), 5)
            local = await producer.history.capture_model_interaction_cutoffs(
                execution.execution_id, principal=principal)
            external = await independent.capture_model_interaction_cutoffs(
                execution.execution_id, principal=principal)
            assert local.local_staging_available
            assert not local.durable_history_available
            assert local.cutoffs
            assert not external.local_staging_available
            assert not external.durable_history_available
            assert external.cutoffs == external.durable_cutoffs == ()
            assert await independent.subscribe_model_interactions(
                execution.execution_id, principal=principal) is None
            contents = await producer.history.history(
                execution.execution_id, principal=principal, include_content=True)
            assert any(item.content == "producer-local prompt" for item in contents.items)
        finally:
            release.set()
            await execution.wait(timeout_seconds=5)
        with pytest.raises(AIError) as unavailable:
            await independent.history(execution.execution_id, principal=principal, include_content=True)
        assert unavailable.value.code is ErrorCode.EXECUTION_HISTORY_UNAVAILABLE
        terminal = await independent.capture_model_interaction_cutoffs(
            execution.execution_id, principal=principal)
        assert not terminal.durable_history_available
        assert not terminal.local_staging_available
