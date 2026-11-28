#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native request lookup and event-driven session inactivity through the SDK."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import AuthorizationAction, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.session import SessionTurnAdmission

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _Authorization(TenantAuthorizationPolicy):
    denied_turn = None

    async def authorize(self, principal, action, resource):
        await super().authorize(principal, action, resource)
        if action is AuthorizationAction.EXECUTION_READ and resource.id == self.denied_turn:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


def _features(model):
    features = RuntimeFeatures("native-session-receipts", defaults={"execution": {"retain_history": True}})
    features.agent("default", allow_tools=(), allow_capabilities=())
    return model_features(_Models(_UsageFunctionModel(model)), features)


@pytest.mark.asyncio
async def test_native_lookup_and_wait_follow_the_authoritative_turn(tmp_path, monkeypatch):
    entered, release, first_read = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def model(_messages, _info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    auth = _Authorization()
    storage = PersistenceStorage.from_root(tmp_path)
    async with Runtime.open(
        "native-session-receipts", storage=storage, features=_features(model), authorization=auth,
    ) as runtime, asyncio.timeout(10):
        principal = runtime.default_principal
        sessions = runtime.sessions
        await sessions.create_turn_session("chat", principal=principal)
        assert await sessions.lookup_admission("chat", principal=principal, idempotency_key="missing") is None
        with pytest.raises(AIError) as denied:
            await sessions.lookup_admission("missing", principal=principal, idempotency_key="send")
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        turn = await sessions.start_turn(
            "chat", "Original input", agent="default", principal=principal, idempotency_key="send",
        )
        await entered.wait()
        assert await sessions.lookup_admission(
            "chat", principal=principal, idempotency_key="send",
        ) == SessionTurnAdmission(turn, "Original input")
        auth.denied_turn = turn
        with pytest.raises(AIError) as denied:
            await sessions.lookup_admission("chat", principal=principal, idempotency_key="send")
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
        auth.denied_turn = None

        read_state = runtime.history.turn_session_state
        reads = []

        async def observe_state(*args, **kwargs):
            view = await read_state(*args, **kwargs)
            reads.append(view["active_turn_id"])
            first_read.set()
            return view

        monkeypatch.setattr(runtime.history, "turn_session_state", observe_state)
        async with asyncio.TaskGroup() as tasks:
            wait = tasks.create_task(sessions.wait_inactive("chat", principal=principal))
            await first_read.wait()
            assert reads == [turn] and not wait.done()
            # Waiting notices do not release occupancy and must not reread it.
            sessions.invalidate_turn_session("chat", turn, revision=1, phase="waiting")
            await asyncio.sleep(0)
            assert reads == [turn] and not wait.done()
            release.set()
        assert wait.result()["active_turn_id"] is None and reads == [turn, None]

    async with Runtime.open(
        "native-session-receipts", storage=storage, features=_features(model), authorization=auth,
    ) as reopened, asyncio.timeout(10):
        principal = reopened.default_principal
        assert await reopened.sessions.lookup_admission(
            "chat", principal=principal, idempotency_key="send",
        ) == SessionTurnAdmission(turn, "Original input")
        view = await reopened.sessions.wait_inactive("chat", principal=principal)
        assert view["active_turn_id"] is None and view["last_turn_no"] == 1


@pytest.mark.asyncio
async def test_closing_runtime_wakes_waiting_native_sdk_observer(tmp_path, monkeypatch):
    entered, snapshot = asyncio.Event(), asyncio.Event()

    async def model(_messages, _info):
        entered.set()
        await asyncio.Event().wait()

    async with Runtime.open(
        "native-session-receipts", storage=PersistenceStorage.from_root(tmp_path),
        features=_features(model), authorization=_Authorization(),
    ) as runtime, asyncio.timeout(10):
        principal = runtime.default_principal
        sessions = runtime.sessions
        await sessions.create_turn_session("chat", principal=principal)
        await sessions.start_turn("chat", "blocked", agent="default", principal=principal, idempotency_key="send")
        await entered.wait()
        read_state = runtime.history.turn_session_state

        async def observe_state(*args, **kwargs):
            view = await read_state(*args, **kwargs)
            snapshot.set()
            return view

        monkeypatch.setattr(runtime.history, "turn_session_state", observe_state)

        async def observe_close():
            with pytest.raises(AIError) as closed:
                await sessions.wait_inactive("chat", principal=principal)
            assert closed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert not closed.value.retryable

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(observe_close())
            await snapshot.wait()
            await runtime.close()


@pytest.mark.asyncio
async def test_native_inactivity_rechecks_remote_sqlite_without_notice_bridge(tmp_path, monkeypatch):
    entered, release, seen_busy = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(_messages, _info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    storage = PersistenceStorage.from_root(tmp_path)
    async with (
        Runtime.open("native-session-receipts", storage=storage, features=_features(model),
                     authorization=_Authorization()) as producer,
        Runtime.open("native-session-receipts", storage=storage, features=_features(model),
                     authorization=_Authorization()) as observer,
        asyncio.timeout(10),
    ):
        caller = producer.default_principal
        await producer.sessions.create_turn_session("chat", principal=caller)
        turn = await producer.sessions.start_turn(
            "chat", "remote", agent="default", principal=caller, idempotency_key="send",
        )
        await entered.wait()
        read = observer.history.turn_session_state

        async def observe_state(*args, **kwargs):
            view = await read(*args, **kwargs)
            if view["active_turn_id"] == turn.key.turn_id:
                seen_busy.set()
            return view

        monkeypatch.setattr(observer.history, "turn_session_state", observe_state)
        async with asyncio.TaskGroup() as tasks:
            waiter = tasks.create_task(observer.sessions.wait_inactive(
                "chat", principal=observer.default_principal, timeout_seconds=5,
            ))
            await seen_busy.wait()
            assert not waiter.done()
            release.set()
            await turn.wait_final(timeout_seconds=5)
        assert waiter.result()["active_turn_id"] is None
        assert waiter.result()["last_turn_no"] == 1 and calls == 1
