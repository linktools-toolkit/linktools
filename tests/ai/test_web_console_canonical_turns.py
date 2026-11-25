#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The same SQL Web conversation API serves both Step capability settings."""

import asyncio
import json

import pytest

pytest.importorskip("starlette")

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.web import create_app

from ._runtime_test_helpers import RuntimeUsageModels, model_features, native_checkpoint
from .test_web_console import _HEADERS, client


@pytest.mark.asyncio
@pytest.mark.parametrize("step_enabled", (False, True))
async def test_sql_web_conversation_uses_one_send_stream_and_history_contract(tmp_path, step_enabled):
    class WorkspacePolicy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)

    features = RuntimeFeatures("web-canonical")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    if step_enabled:
        features.capability(native_checkpoint(), scope="runtime")

    async with (
        Persistence.open("web-canonical", storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=model_features(RuntimeUsageModels(), features),
                     authorization=WorkspacePolicy()) as runtime,
    ):
        app = create_app(runtime=runtime, local_workspace=True)
        async with client(app) as http:
            settings = await http.get("/api/config")
            assert settings.json()["turn_sessions"] is True
            created = await http.post("/api/sessions", headers=_HEADERS, json={
                "session_id": "chat", "title": "Chat", "request_id": "create",
            })
            assert created.status_code == 201

            frames = asyncio.Queue()
            disconnect = asyncio.Event()
            path = "/api/session/events"
            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
                "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
                "query_string": b"session_id=chat&view=chat", "headers": [],
                "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345),
            }

            async def receive():
                await disconnect.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] != "http.response.body":
                    return
                for line in message.get("body", b"").splitlines():
                    if line.startswith(b"data: "):
                        frames.put_nowait(json.loads(line[6:]))

            stream = asyncio.create_task(app(scope, receive, send))
            try:
                async with asyncio.timeout(10):
                    snapshot = await frames.get()
                    assert snapshot["type"] == "snapshot" and snapshot["session"]["active_turn_id"] is None
                    accepted = await http.post("/api/session/messages?session_id=chat", headers=_HEADERS, json={
                        "prompt": "question", "agent_id": "default", "request_id": "send",
                    })
                    assert accepted.status_code == 202 and set(accepted.json()) == {"turn_id"}
                    turn_id = accepted.json()["turn_id"]
                    seen = []
                    while True:
                        frame = await frames.get()
                        assert frame["turn_id"] == turn_id
                        seen.append(frame["type"])
                        if frame["type"] == "terminal":
                            break
                    assert seen[0] == "admitted" and "delta" in seen
                    assert frame.get("error_code") is None
            finally:
                disconnect.set()
                await stream

            result = await http.get(f"/api/session?session_id=chat&turn_id={turn_id}")
            assert result.status_code == 200
            assert result.json()["turn_id"] == turn_id
            assert result.json()["conversation_committed"] is True
