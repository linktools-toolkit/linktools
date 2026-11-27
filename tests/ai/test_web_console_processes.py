#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Web reads and explicit controls share the cross-process Runtime owner."""

import asyncio
import multiprocessing
from pathlib import Path

import httpx
import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from pydantic_ai.models.function import FunctionModel

from linktools.ai.history import History
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.capability import TraceCapability
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.web import create_app

from ._runtime_test_helpers import _wait_for_committed, model_features
from .test_live_history_readback_integration import _Models
from .test_web_console import _HEADERS, client


async def _get(http: httpx.AsyncClient, path: str) -> dict[str, object]:
    response = await http.get(path)
    assert response.status_code == 200, response.text
    return response.json()


class _NativeProcessPolicy(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        await super().authorize(principal, action, resource)


def _native_process_features(model, *, trace: bool = False):
    features = RuntimeFeatures("web-native-process")
    if trace:
        features.capability(TraceCapability(), scope="runtime")
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    return model_features(_Models(FunctionModel(stream_function=model)), features)


def _native_web_turn_owner(database: Path, channel, finish=None, close=None, trace: bool = False) -> None:
    async def run():
        entered = asyncio.Event()

        async def model(messages, info):
            entered.set()
            if finish is None:
                await asyncio.Event().wait()
            else:
                while not finish.is_set():
                    await asyncio.sleep(0.01)
            yield "owner answer"

        async with (
            Persistence.open("web-native-process", storage=PersistenceStorage.sqlite(database)) as state,
            Runtime.open(persistence=state, features=_native_process_features(model, trace=trace),
                         authorization=_NativeProcessPolicy()) as runtime,
        ):
            principal = runtime.default_principal
            await runtime.sessions.create_turn_session("session", principal=principal, idempotency_key="create")
            turn = await runtime.sessions.start(
                "session", "original prompt", agent="default", principal=principal, idempotency_key="send",
            )
            await entered.wait()
            channel.send(turn.key.turn_id)
            if finish is not None:
                settled = await turn.wait_final()
                assert settled.status.value == "SUCCEEDED", settled.view
                while not close.is_set():
                    await asyncio.sleep(0.01)
            else:
                await asyncio.Event().wait()

    try:
        asyncio.run(run())
    finally:
        channel.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("trace", (False, True))
async def test_web_reads_live_process_history_without_taking_ownership(tmp_path: Path, trace: bool) -> None:
    database = tmp_path / "canonical.db"
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    finish, close = ctx.Event(), ctx.Event()
    owner = ctx.Process(target=_native_web_turn_owner, args=(database, child, finish, close, trace))
    owner.start()
    child.close()
    calls = []

    async def model(messages, info):
        calls.append("unexpected replay")
        yield "unexpected replay"

    policy = _NativeProcessPolicy()
    try:
        assert await asyncio.to_thread(parent.poll, 45), "native subprocess did not enter the model"
        turn_id = parent.recv()
        path = f"/api/session?session_id=session&turn_id={turn_id}"
        metadata = "/api/session?session_id=session&include_timeline=false"
        storage = PersistenceStorage.sqlite(database)
        async with (
            Persistence.open("web-native-process", storage=storage) as state,
            Runtime.open(persistence=state, features=_native_process_features(model, trace=trace),
                         authorization=policy) as runtime,
            Persistence.open("web-native-process", storage=storage, read_only=True) as read_state,
            History.open(persistence=read_state, authorization=policy) as history,
            client(create_app(runtime=runtime, local_workspace=True)) as writable,
            client(create_app(history=history, local_workspace=True)) as readonly,
        ):
            for http in (writable, readonly):
                for _ in range(2):
                    active = await _get(http, path)
                    assert active["status"] == "PENDING_START" and not active["conversation_committed"]
                    assert active["user_input"]["prompt"] == "original prompt"
                    assert (await _get(http, metadata))["session"]["active_turn_id"] == turn_id
            assert not calls and owner.is_alive()
            finish.set()
            terminal = await _wait_for_committed(
                lambda: _get(writable, path),
                lambda view: view["conversation_committed"], timeout=10,
            )
            assert terminal["status"] == "SUCCEEDED", terminal
            for http in (writable, readonly):
                result = await _get(http, path)
                assert result["status"] == "SUCCEEDED" and result["conversation_committed"]
                assert any(item["item_kind"] == "assistant" and item["content"] == "owner answer"
                           for item in result["items"])
                assert (await _get(http, metadata))["session"]["active_turn_id"] is None
                if not trace:
                    continue
                # The lossy, optional Trace writer may publish after the Session.
                models = await _wait_for_committed(
                    lambda: _get(http, f"{path}&models=true&include_content=true"),
                    lambda value: bool(value["items"]), timeout=10,
                )
                interaction, = models["items"]
                assert interaction["status"] == "SUCCEEDED" and interaction["content_included"]
            assert not calls
            close.set()
            await asyncio.to_thread(owner.join, 10)
            assert owner.exitcode == 0
    finally:
        parent.close()
        if owner.is_alive():
            owner.kill()
        await asyncio.to_thread(owner.join, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ("recover", "stop", "live_stop"))
async def test_web_native_crashed_producer_keeps_occupancy_until_explicit_control(tmp_path: Path, control: str) -> None:
    database = tmp_path / "canonical.db"
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    owner = ctx.Process(target=_native_web_turn_owner, args=(database, child))
    owner.start()
    child.close()

    calls = []

    async def model(messages, info):
        calls.append("recovered")
        yield "recovered answer"

    store = PersistenceStorage.sqlite(database)
    try:
        assert await asyncio.to_thread(parent.poll, 45), "native subprocess did not enter the model"
        turn_id = parent.recv()
        path = f"/api/session?session_id=session&turn_id={turn_id}"
        metadata = "/api/session?session_id=session&include_timeline=false"
        async with (
            Persistence.open("web-native-process", storage=store) as state,
            Runtime.open(persistence=state, features=_native_process_features(model),
                         authorization=_NativeProcessPolicy()) as runtime,
            client(create_app(runtime=runtime, local_workspace=True)) as http,
        ):
            active = await _get(http, path)
            assert active["turn_id"] == turn_id and not active["conversation_committed"]
            assert active["user_input"]["prompt"] == "original prompt"
            assert (await _get(http, metadata))["session"]["active_turn_id"] == turn_id
            assert not calls

            if control == "live_stop":
                for _ in range(2):
                    stopped = await http.post("/api/session/cancel?session_id=session", headers=_HEADERS,
                                              json={"turn_id": turn_id, "request_id": "stop"})
                    assert stopped.status_code == 200 and stopped.json()["accepted"], stopped.text
                assert (await _get(http, path))["status"] == "CANCELLING"
                assert (await _get(http, metadata))["session"]["active_turn_id"] == turn_id

            owner.kill()
            await asyncio.to_thread(owner.join, 10)
            assert owner.exitcode is not None
            # Simulate elapsed lease time after the actual producer died.
            # Advance the old deadline, never the ownership/generation fences.
            turns = state.session_turns
            assert turns is not None
            table = turns._turns
            async with turns._context.engine.begin() as connection:
                expired = await connection.execute(table.update().where(
                    table.c.turn_id == turn_id.encode(),
                ).values(lease_until=0))
                assert expired.rowcount == 1
            for _ in range(2):
                observed = await _get(http, path)
                assert observed["status"] == "RECOVERY_REQUIRED" and not observed["conversation_committed"]
                assert (await _get(http, metadata))["session"]["active_turn_id"] == turn_id
            assert not calls
            if control == "live_stop":
                row = await turns.get_turn(SessionTurnKey("session", turn_id))
                assert row["state"] == "cancelling" and row["generation"] == 1

            if control == "stop":
                stopped = await http.post("/api/session/cancel?session_id=session", headers=_HEADERS,
                                          json={"turn_id": turn_id, "request_id": "stop"})
                assert stopped.status_code == 200 and stopped.json()["accepted"], stopped.text
                assert (await _get(http, metadata))["session"]["active_turn_id"] == turn_id

        async with (
            Persistence.open("web-native-process", storage=store) as state,
            Runtime.open(persistence=state, features=_native_process_features(model),
                         authorization=_NativeProcessPolicy()) as runtime,
            client(create_app(runtime=runtime, local_workspace=True)) as http,
        ):
            assert (await _get(http, metadata))["session"]["active_turn_id"] == turn_id
            assert not calls
            if control != "recover":
                replayed = await http.post("/api/session/cancel?session_id=session", headers=_HEADERS,
                                           json={"turn_id": turn_id, "request_id": "stop"})
                assert replayed.status_code == 200 and replayed.json()["accepted"], replayed.text
                assert (await _get(http, metadata))["session"]["active_turn_id"] == turn_id
            recovered = await http.post("/api/session/recover?session_id=session", headers=_HEADERS,
                                        json={"turn_id": turn_id, "request_id": "recover"})
            assert recovered.status_code == 200 and recovered.json()["accepted"], recovered.text
            expected = "SUCCEEDED" if control == "recover" else "CANCELLED"
            terminal = await _wait_for_committed(lambda: _get(http, path),
                lambda value: value["status"] == expected and value["conversation_committed"], timeout=10)
            assert terminal["turn_id"] == turn_id
            assert (await _get(http, metadata))["session"]["active_turn_id"] is None
            assert calls == (["recovered"] if control == "recover" else [])
            if control == "recover":
                assert any(item["item_kind"] == "assistant" and item["content"] == "recovered answer"
                           for item in terminal["items"])
    finally:
        parent.close()
        if owner.is_alive():
            owner.kill()
        await asyncio.to_thread(owner.join, 10)
