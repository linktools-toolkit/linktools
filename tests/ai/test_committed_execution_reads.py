#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Consumption and next-turn admission bypass unexpired observation records."""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.execution import CancelExecutionRequest
from linktools.ai.storage import InMemoryCacheBackend, _write_cache

from ._runtime_test_helpers import _Models, model_features, native_checkpoint


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_committed_execution_results_ignore_remote_observation_cache(
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

    group = RuntimeFeatures("committed-read", defaults={"execution": {"retain_history": True}})
    group.agent("default", model="default", allow_tools=())
    # This test observes a source-admitted execution through a stale reader.
    group.capability(native_checkpoint(), scope="runtime")
    database = tmp_path / "runtime.db"
    models = _Models(FunctionModel(stream_function=model))
    owner_storage = PersistenceStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000))
    reader_storage = PersistenceStorage.sqlite(database, cache=InMemoryCacheBackend(max_bytes=1000000))
    async with Persistence.open('committed-read', storage=owner_storage) as persistence_49, Runtime.open(persistence=persistence_49, features=model_features(models, group)) as owner, Persistence.open('committed-read', storage=reader_storage) as persistence_49, Runtime.open(persistence=persistence_49, features=model_features(models, group)) as reader:
        principal = owner.default_principal
        first = await owner.agents.get().start("first", idempotency_key="first")
        try:
            await asyncio.wait_for(entered.wait(), 10)
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
            # The observation cache's frozen clock keeps the stale view live
            # regardless of how long the runner takes to read the source.
            assert (await asyncio.wait_for(remote.wait(), 10)).result.status is terminal.status
            if not cancel:
                assert await reader._execution_service.result_payload_size(first.execution_id, principal=principal) > 0
            assert calls == 1
        finally:
            release.set()


class _SessionAuthorization(TenantAuthorizationPolicy):
    """Explicit host policy for authorized native SQL sessions."""


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", (False, True))
async def test_native_next_turn_uses_canonical_continuation_across_runtimes(tmp_path: Path, cancel: bool) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    model_histories = []

    async def model(messages, _info):
        model_histories.append((
            [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)],
            [message.text for message in messages if isinstance(message, ModelResponse)],
        ))
        if len(model_histories) == 1:
            entered.set()
            await release.wait()
            yield "first output"
        else:
            yield "second output"

    group = RuntimeFeatures("committed-session", defaults={"execution": {"retain_history": True}})
    group.agent("default", model="default", allow_tools=())
    features = model_features(_Models(FunctionModel(stream_function=model)), group)
    database = tmp_path / "session.db"
    async with AsyncExitStack() as stack:
        runtimes = []
        for _ in range(2):
            persistence = await stack.enter_async_context(Persistence.open(
                "committed-session", storage=PersistenceStorage.sqlite(
                    database, cache=InMemoryCacheBackend(max_bytes=1000000),
                ),
            ))
            runtimes.append(await stack.enter_async_context(Runtime.open(
                persistence=persistence, features=features, authorization=_SessionAuthorization(persistence.tenant_id),
            )))
        owner, reader = runtimes
        principal = owner.default_principal
        await owner.sessions.create_turn_session("session", principal=principal)
        first = await owner.sessions.start_turn(
            "session", "first", principal=principal, agent="default", idempotency_key="first",
        )
        try:
            await asyncio.wait_for(entered.wait(), 10)
            occupied = await reader.history.turn_session_state("session", principal=principal)
            assert occupied["active_turn_id"] == first.key.turn_id
            if cancel:
                assert await first.cancel()
            release.set()
            expected = ExecutionStatus.CANCELLED if cancel else ExecutionStatus.SUCCEEDED
            assert (await asyncio.wait_for(first.wait_final(), 10)).status is expected
            second = await reader.sessions.start_turn(
                "session", "second", principal=principal, agent="default", idempotency_key="second",
            )
            assert (await asyncio.wait_for(second.wait_final(), 10)).status is ExecutionStatus.SUCCEEDED
            page = await reader.history.turn_session("session", principal=principal)
            turns = page["timeline"]["items"]
            assert [(item["turn_id"], item["status"]) for item in turns] == [
                (first.key.turn_id, expected.value), (second.key.turn_id, "SUCCEEDED"),
            ]
            previous_response = (
                "[Runtime: This turn was interrupted before a final answer.]" if cancel else "first output"
            )
            assert model_histories == [(["first"], []), (["first", "second"], [previous_response])]
        finally:
            release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ("result", "result_payload_size", "wait"))
async def test_terminal_consumption_reuses_confirmed_status_for_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query: str,
) -> None:
    async def model(messages, info):
        yield "finished"

    group = RuntimeFeatures("terminal-consumption", defaults={"execution": {"retain_history": True}})
    group.agent("default", model="default", allow_tools=())
    async with Persistence.open('terminal-consumption', storage=PersistenceStorage.sqlite(tmp_path / 'runtime.db')) as persistence_96, Runtime.open(persistence=persistence_96, features=model_features(_Models(FunctionModel(stream_function=model)), group)) as runtime:
        execution = await runtime.agents.get().start("first", idempotency_key="first")
        await execution.wait()
        service = runtime._execution_service
        repository = runtime._state.execution.executions
        load = repository.get_result_record
        confirmed = False

        async def load_once(execution_id, *, tenant_id, fresh=False):
            nonlocal confirmed
            assert fresh
            if confirmed:
                raise AssertionError("terminal consumption already owns its confirmed status")
            value = await load(execution_id, tenant_id=tenant_id, fresh=fresh)
            assert value.status is ExecutionStatus.SUCCEEDED
            confirmed = True
            return value

        monkeypatch.setattr(repository, "get_result_record", load_once)
        result = await getattr(service, query)(
            execution.execution_id, principal=runtime.default_principal,
        )
        assert confirmed
        if query == "result_payload_size":
            assert result > 0
        else:
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"text": "finished"}
