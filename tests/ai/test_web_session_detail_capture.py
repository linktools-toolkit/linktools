#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Web session metadata and timeline use the same authorized capture."""

import asyncio
from pathlib import Path

import pytest

from linktools.ai.core import AuthorizationAction
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.web import create_app

from .test_pending_session_projection import _runtimes
from .test_web_console import client


@pytest.mark.asyncio
async def test_web_session_detail_uses_one_capture_for_pending_and_durable_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtimes(tmp_path, cache=InMemoryCacheBackend(max_bytes=2**24), shared=False) as (runtimes, entered, release, block):
        runtime, repos = runtimes[0]
        session = await runtime.agents.get().create_session("session")
        first = await session.start("earlier", idempotency_key="first")
        await first.wait()
        old_inspect = History.inspect_session
        metadata_calls = []

        async def inspect(history, session_id, *, principal):
            metadata_calls.append(session_id)
            return await old_inspect(history, session_id, principal=principal)

        monkeypatch.setattr(History, "inspect_session", inspect)
        async with client(create_app(runtime=runtime)) as http:
            path = "/api/session?session_id=session"
            metadata = await http.get(path + "&include_timeline=false")
            assert metadata.status_code == 200
            assert metadata.json()["timeline"] is None and metadata_calls == ["session"]
            metadata_calls.clear()

            block()
            pending = await session.start("current", idempotency_key="pending")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert await repos.execution.executions.get(pending.execution_id, tenant_id=runtime.tenant_id) is None
                page = await http.get(path + "&limit=1")
                assert page.status_code == 200, page.text
                body = page.json()
                assert body["session"]["active_execution_id"] == pending.execution_id
                assert [turn["execution_id"] for turn in body["timeline"]["items"]] == [pending.execution_id]
                assert body["timeline"]["items"][0]["conversation_committed"] is False
                older = await http.get("/api/session", params={"session_id": "session", "limit": 1,
                    "cursor": body["timeline"]["next_cursor"]})
                assert older.status_code == 200, older.text
                assert older.json()["session"]["active_execution_id"] == pending.execution_id
                assert [turn["execution_id"] for turn in older.json()["timeline"]["items"]] == [first.execution_id]
            finally:
                release.set()
            await pending.wait()
            durable = await http.get(path + "&limit=1")
            assert durable.status_code == 200, durable.text
            body = durable.json()
            assert body["session"]["active_execution_id"] is None
            assert body["timeline"]["items"][0]["execution_id"] == pending.execution_id
            assert body["timeline"]["items"][0]["conversation_committed"] is True
            assert metadata_calls == []


@pytest.mark.asyncio
async def test_web_session_detail_still_authorizes_and_retains_metadata_only_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _runtimes(tmp_path, cache=InMemoryCacheBackend(max_bytes=2**24), shared=False) as (runtimes, _, _, _):
        runtime, _ = runtimes[0]
        session = await runtime.agents.get().create_session("session")
        run = await session.start("secret", idempotency_key="turn")
        await run.wait()
        history = runtime.history
        auth_type = type(history._authorization)
        original_authorize = auth_type.authorize
        deny = True
        checks = []

        async def authorize(policy, principal, action, resource):
            if action is AuthorizationAction.SESSION_READ and resource.id == "session":
                checks.append(resource.id)
                if deny:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return await original_authorize(policy, principal, action, resource)

        monkeypatch.setattr(auth_type, "authorize", authorize)
        async with client(create_app(runtime=runtime)) as http:
            path = "/api/session?session_id=session"
            denied = await http.get(path)
            assert denied.status_code == 404 and "secret" not in denied.text
            assert checks == ["session"]
            deny = False

            from linktools.ai.history import _history

            async def unavailable(*args, **kwargs):
                raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)

            monkeypatch.setattr(_history, "project_captured_session_timeline", unavailable)
            unavailable_page = await http.get(path)
            assert unavailable_page.status_code == 503
            assert unavailable_page.json()["code"] == ErrorCode.SESSION_HISTORY_UNAVAILABLE.value
            assert checks == ["session", "session"]
            metadata = await http.get(path + "&include_timeline=false")
            assert metadata.status_code == 200 and metadata.json()["timeline"] is None
            assert checks == ["session", "session", "session"]
