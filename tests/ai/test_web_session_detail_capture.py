#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web session metadata and timeline use the same authorized capture."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import AuthorizationAction, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._session_turn_sql import SqlSessionTurnReader
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.web import create_app

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_web_console import client


class _WorkspacePolicy(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        await super().authorize(principal, action, resource)


@asynccontextmanager
async def _runtime(tmp_path, model, policy):
    features = RuntimeFeatures("web-session-capture")
    features.agent("default", model="default", allow_tools=())
    async with (
        Persistence.open("web-session-capture", storage=PersistenceStorage.from_root(tmp_path)) as persistence,
        Runtime.open(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
                     authorization=policy) as runtime,
    ):
        await runtime.sessions.create_turn_session("session", principal=runtime.default_principal)
        yield runtime, persistence


@pytest.mark.asyncio
async def test_web_session_detail_uses_one_capture_for_pending_and_durable_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocked, entered, release = False, asyncio.Event(), asyncio.Event()

    async def model(_messages, _info):
        if blocked:
            entered.set()
            await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path, model, _WorkspacePolicy()) as (runtime, persistence):
        done = asyncio.Queue()
        unsubscribe = runtime.sessions.observe_turn_session(
            "session", lambda frame: done.put_nowait(frame["turn_id"]) if frame["type"] == "terminal" else None,
        )

        async def start(prompt, key):
            return (await runtime.sessions.start_turn(
                "session", prompt, agent="default", principal=runtime.default_principal, idempotency_key=key,
            )).key.turn_id

        first = await start("earlier", "first")
        assert await asyncio.wait_for(done.get(), 5) == first
        second = await start("later", "second")
        assert await asyncio.wait_for(done.get(), 5) == second

        calls = []
        original_session, original_page = SqlSessionTurnReader.get_session, SqlSessionTurnReader.turn_page

        async def read_session(reader, identity):
            calls.append(("metadata", identity))
            return await original_session(reader, identity)

        async def read_page(reader, identity, **kwargs):
            calls.append(("timeline", identity))
            return await original_page(reader, identity, **kwargs)

        monkeypatch.setattr(SqlSessionTurnReader, "get_session", read_session)
        monkeypatch.setattr(SqlSessionTurnReader, "turn_page", read_page)
        try:
            async with client(create_app(runtime=runtime, local_workspace=True)) as http:
                path = "/api/session?session_id=session"
                metadata = await http.get(path + "&include_timeline=false")
                assert metadata.status_code == 200, metadata.text
                assert metadata.json()["timeline"] is None
                assert calls == [("metadata", "session")]

                blocked = True
                pending = await start("current", "pending")
                await asyncio.wait_for(entered.wait(), 5)
                assert await persistence.execution.executions.get(pending, tenant_id=runtime.tenant_id) is None
                calls.clear()
                page = await http.get(path + "&limit=1")
                assert page.status_code == 200, page.text
                body = page.json()
                assert calls == [("metadata", "session"), ("timeline", "session")]
                assert body["session"]["active_turn_id"] == pending
                assert [turn["turn_id"] for turn in body["timeline"]["items"]] == [second]
                active = await http.get(path + f"&turn_id={pending}")
                assert active.status_code == 200 and active.json()["turn_id"] == pending
                assert active.json()["conversation_committed"] is False
                older = await http.get("/api/session", params={
                    "session_id": "session", "limit": 1, "cursor": body["timeline"]["next_cursor"],
                })
                assert older.status_code == 200, older.text
                assert older.json()["session"]["active_turn_id"] == pending
                assert [turn["turn_id"] for turn in older.json()["timeline"]["items"]] == [first]

                release.set()
                assert await asyncio.wait_for(done.get(), 5) == pending
                calls.clear()
                durable = await http.get(path + "&limit=1")
                assert durable.status_code == 200, durable.text
                body = durable.json()
                assert calls == [("metadata", "session"), ("timeline", "session")]
                assert body["session"]["active_turn_id"] is None
                assert body["timeline"]["items"][0]["turn_id"] == pending
                assert body["timeline"]["items"][0]["conversation_committed"] is True
        finally:
            release.set()
            unsubscribe()


@pytest.mark.asyncio
async def test_web_session_detail_still_authorizes_and_retains_metadata_only_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Policy(_WorkspacePolicy):
        async def authorize(self, principal, action, resource):
            if action is AuthorizationAction.SESSION_READ and resource.id == "session":
                checks.append(resource.id)
                if deny:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return await super().authorize(principal, action, resource)

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    checks = []
    deny = False
    async with _runtime(tmp_path, model, Policy()) as (runtime, _):
        done = asyncio.Event()
        unsubscribe = runtime.sessions.observe_turn_session(
            "session", lambda frame: done.set() if frame["type"] == "terminal" else None,
        )
        try:
            (await runtime.sessions.start_turn(
                "session", "secret", agent="default", principal=runtime.default_principal, idempotency_key="turn",
            )).key.turn_id
            await asyncio.wait_for(done.wait(), 5)
        finally:
            unsubscribe()
        checks.clear()
        deny = True
        async with client(create_app(runtime=runtime, local_workspace=True)) as http:
            path = "/api/session?session_id=session"
            denied = await http.get(path)
            assert denied.status_code == 404 and "secret" not in denied.text
            assert checks == ["session"]
            deny = False

            async def unavailable(*args, **kwargs):
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)

            monkeypatch.setattr(SqlSessionTurnReader, "turn_page", unavailable)
            unavailable_page = await http.get(path)
            assert unavailable_page.status_code == 503
            assert unavailable_page.json()["code"] == ErrorCode.SESSION_HISTORY_UNAVAILABLE.value
            assert checks == ["session", "session"]
            metadata = await http.get(path + "&include_timeline=false")
            assert metadata.status_code == 200 and metadata.json()["timeline"] is None
            assert checks == ["session", "session", "session"]
