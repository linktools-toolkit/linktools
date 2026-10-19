#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime shutdown joins admitted work even after its caller is cancelled."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.runtime.state import RuntimeState
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.storage import InMemoryCacheBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_mixed_pending_admission import _start, _writers
from .test_terminal_commit_shared_runtime import _group


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close_waiter", (False, True))
async def test_runtime_close_joins_pending_publication_after_start_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_close_waiter: bool,
) -> None:
    publishing, release, published = asyncio.Event(), asyncio.Event(), asyncio.Event()
    preflight_entered, storage_closed = asyncio.Event(), asyncio.Event()

    class BlockingCache(InMemoryCacheBackend):
        async def put(self, key, value):
            if key.startswith("runtime-pending-execution:") and not publishing.is_set():
                publishing.set()
                await release.wait()
                await super().put(key, value)
                published.set()
                return
            await super().put(key, value)

    async def model(messages, info):
        assert not storage_closed.is_set()
        return ModelResponse(parts=[TextPart("answer")])

    original_preflight = DefaultExecutionService.preflight_close

    async def observe_preflight(service):
        preflight_entered.set()
        await original_preflight(service)

    monkeypatch.setattr(DefaultExecutionService, "preflight_close", observe_preflight)
    storage = RuntimeStorage.sqlite(
        tmp_path / "state.db", cache=BlockingCache(max_bytes=2**20),
    )
    original_close = RuntimeState.close

    async def close_storage(owner):
        assert published.is_set(), "storage closed before admitted publication settled"
        await original_close(owner)
        storage_closed.set()

    monkeypatch.setattr(RuntimeState, "close", close_storage)
    async with Runtime.open(
        "pending-admission-close", storage=storage,
        models=_Models(_UsageFunctionModel(model)), features=_group(),
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        starting = asyncio.create_task(session.start("question", idempotency_key="request"))
        closing = None
        repeated_close = None
        try:
            await asyncio.wait_for(publishing.wait(), 5)
            starting.cancel()
            closing = asyncio.create_task(runtime.close())
            await asyncio.wait_for(preflight_entered.wait(), 5)
            assert not closing.done()
            assert not starting.done()
            assert not storage_closed.is_set()
            if cancel_close_waiter:
                closing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await closing
                repeated_close = asyncio.create_task(runtime.close())
                await asyncio.sleep(0)
                assert not repeated_close.done()
                assert not published.is_set() and not storage_closed.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(starting, 5)
            await asyncio.wait_for(repeated_close or closing, 5)
            assert published.is_set() and storage_closed.is_set()
        finally:
            release.set()
            await asyncio.gather(
                *(task for task in (starting, closing, repeated_close) if task is not None),
                return_exceptions=True,
            )


@pytest.mark.asyncio
async def test_preflight_rejects_new_admission_before_claiming() -> None:
    async with _writers() as (_state, pending, legacy):
        for service in (pending, legacy):
            await service.preflight_close()
            with pytest.raises(AIError) as raised:
                await _start(service)
            assert raised.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert service._runtime_bridge.pending_turns.coordination.claims == []
            assert service.runtime_backend().calls == 0
