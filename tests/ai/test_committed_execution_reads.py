#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consumption and next-turn admission bypass unexpired observation records."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime, RuntimeStorage, ConversationCapability, CheckpointCapability
from linktools.ai.runtime.service_api import CancelExecutionRequest
from linktools.ai.storage import InMemoryCacheBackend, _write_cache

from .test_live_history_readback_integration import _Models


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_committed_results_and_next_turn_ignore_remote_observation_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    now = [1000.0]
    monkeypatch.setattr(_write_cache, "time", lambda: now[0])
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await release.wait()
        yield "finished"

    group = RuntimeFeatures("committed-read")
    group.agent("default", model="default", allow_tools=())
    group.capability(ConversationCapability(), scope="runtime")
    # This test observes a source-admitted execution through a stale reader.
    group.capability(CheckpointCapability(), scope="runtime")
    database = tmp_path / "runtime.db"
    models = _Models(FunctionModel(stream_function=model))
    owner_storage = RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000))
    reader_storage = RuntimeStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000))
    async with Runtime.open("committed-read", models=models, storage=owner_storage, features=group) as owner, Runtime.open(
        "committed-read", models=models, storage=reader_storage, features=group,
    ) as reader:
        principal = owner.default_principal
        await owner.agents.get().create_session("session")
        first = await owner.agents.get().session("session").start("first", idempotency_key="first")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            await reader.sessions.get("session", principal=principal)
            assert (await reader.executions.inspect(first.execution_id, principal=principal)).status is ExecutionStatus.STARTED
            if cancel:
                outcome = await owner.executions.cancel(
                    first.execution_id, CancelExecutionRequest(principal, "cancel-first"),
                )
                assert outcome.cancelled is True
            release.set()
            terminal = (await asyncio.wait_for(first.wait(), 10)).result
            now[0] = 1001.0
            assert terminal.status is (ExecutionStatus.CANCELLED if cancel else ExecutionStatus.SUCCEEDED)
            # A remote owner has not invalidated this observer's accepted display capture.
            assert (await reader.executions.inspect(first.execution_id, principal=principal)).status is ExecutionStatus.STARTED
            result = await reader.executions.result(first.execution_id, principal=principal)
            assert result.status is terminal.status
            assert result.output == terminal.output
            remote = await reader.executions.get(first.execution_id, principal=principal)
            assert (await asyncio.wait_for(remote.wait(), 1)).result.status is terminal.status
            if not cancel:
                assert await reader._execution_service.result_payload_size(first.execution_id, principal=principal) > 0
            # The captured session continuation also remains hot; admission must read its source.
            second = await reader.agents.get().session("session").start("second", idempotency_key="second")
            assert (await asyncio.wait_for(second.wait(), 10)).result.status is ExecutionStatus.SUCCEEDED
            assert calls == 2
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ("result", "result_payload_size", "wait"))
async def test_terminal_consumption_reuses_confirmed_status_for_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query: str,
) -> None:
    async def model(messages, info):
        yield "finished"

    group = RuntimeFeatures("terminal-consumption")
    group.agent("default", model="default", allow_tools=())
    group.capability(ConversationCapability(), scope="runtime")
    async with Runtime.open(
        "terminal-consumption", models=_Models(FunctionModel(stream_function=model)),
        storage=RuntimeStorage.sqlite(tmp_path / "runtime.db"), features=group,
    ) as runtime:
        await runtime.agents.get().create_session("session")
        execution = await runtime.agents.get().session("session").start(
            "first", idempotency_key="first",
        )
        await execution.wait()
        service = runtime._execution_service
        load = service._load_execution
        confirmed = False

        async def load_once(execution_id, *, tenant_id):
            nonlocal confirmed
            if confirmed:
                raise AssertionError("terminal consumption already owns its confirmed status")
            value = await load(execution_id, tenant_id=tenant_id)
            assert value.status is ExecutionStatus.SUCCEEDED
            confirmed = True
            return value

        monkeypatch.setattr(service, "_load_execution", load_once)
        result = await getattr(service, query)(
            execution.execution_id, principal=runtime.default_principal,
        )
        assert confirmed
        if query == "result_payload_size":
            assert result > 0
        else:
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"text": "finished"}
