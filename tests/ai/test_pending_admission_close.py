#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime shutdown waits for admitted work to yield its storage ownership."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_mixed_pending_admission import _start, _writers


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_close_waiter", (False, True))
async def test_native_close_drains_committed_admission_after_caller_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_close_waiter: bool,
) -> None:
    committed, producer_cancelled, release, producer_drained = (asyncio.Event() for _ in range(4))
    keys = []
    calls = 0

    class HostAuthorization(TenantAuthorizationPolicy):
        """Authorize the native SQL session in both runtimes."""

    async def model(_messages, _info):
        nonlocal calls
        assert producer_drained.is_set(), "shutdown must not dispatch the old producer"
        calls += 1
        return ModelResponse(parts=[TextPart("answer")])

    name = "native-admission-close"
    features = RuntimeFeatures(name, defaults={"execution": {"retain_history": True}})
    features.agent("default", model="default", allow_tools=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.from_root(tmp_path)

    def open_runtime(state):
        return Runtime.open(persistence=state, features=features, authorization=HostAuthorization())

    async with Persistence.open(name, storage=storage) as state, open_runtime(state) as runtime:
        turns = state.session_turns
        await runtime.agents.get().create_session("session")
        original_accept = turns.accept

        async def hold_committed_admission(key, **kwargs):
            assert await original_accept(key, **kwargs)
            keys.append(key)
            committed.set()
            try:
                # The caller has not received its receipt yet.
                await asyncio.Future()
            except asyncio.CancelledError:
                producer_cancelled.set()
                # Model a driver still draining the acknowledged mutation.
                await release.wait()
                assert state.ready
                producer_drained.set()
                raise

        with monkeypatch.context() as patch:
            patch.setattr(turns, "accept", hold_committed_admission)
            principal = runtime.default_principal
            sending = asyncio.create_task(runtime.sessions.start(
                "session", "question", agent=runtime.agents.get(),
                principal=principal, idempotency_key="request",
            ))
            closing = repeated = None
            try:
                await asyncio.wait_for(committed.wait(), 5)
                sending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await sending
                closing = asyncio.create_task(runtime.close())
                await asyncio.wait_for(producer_cancelled.wait(), 5)
                assert not closing.done() and not producer_drained.is_set()
                assert state.ready and calls == 0

                if cancel_close_waiter:
                    closing.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await closing
                    repeated = asyncio.create_task(runtime.close())
                    await asyncio.sleep(0)
                    assert not repeated.done() and not producer_drained.is_set()

                release.set()
                await asyncio.wait_for(repeated or closing, 5)
                assert producer_drained.is_set() and state.ready and calls == 0
                [key] = keys
                row = await turns.get_turn(key)
                assert row["state"] == "accepted" and row["generation"] == 1
                assert (await turns.get_session("session"))["active_turn_id"] == key.turn_id.encode()
            finally:
                release.set()
                await asyncio.gather(
                    *(task for task in (sending, closing, repeated) if task is not None),
                    return_exceptions=True,
                )

    async with Persistence.open(name, storage=storage) as state, open_runtime(state) as reopened:
        principal = reopened.default_principal
        duplicate = await reopened.sessions.start(
            "session", "question", agent=reopened.agents.get(), principal=principal, idempotency_key="request",
        )
        assert duplicate.key == key and calls == 0
        assert not await reopened.sessions.recover_turn("session", key.turn_id, mode="resume", principal=principal)
        turns = state.session_turns
        async with turns._context.engine.begin() as connection:
            expired = await connection.execute(turns._turns.update().filter_by(
                **turns._turn_values(key), generation=1, state="accepted",
            ).values(lease_until=0))
            assert expired.rowcount == 1
        assert await reopened.sessions.recover_turn("session", key.turn_id, mode="resume", principal=principal)
        result = await duplicate.wait_final(timeout_seconds=5)
        assert result.key == key and result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "answer"} and calls == 1
        assert (await turns.get_turn(key))["generation"] == 2
        assert (await turns.get_session("session"))["active_turn_id"] is None


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
