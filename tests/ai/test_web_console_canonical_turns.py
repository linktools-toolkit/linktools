#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The same SQL Web conversation API serves both Step capability settings."""

import asyncio
import json
from contextlib import AsyncExitStack

import pytest
from pydantic_ai_harness.step_persistence import StepPersistence

pytest.importorskip("starlette")

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.web import create_app

from ._runtime_test_helpers import RuntimeUsageModels, model_features
from .test_web_console import _HEADERS, client


@pytest.mark.asyncio
@pytest.mark.parametrize("step_enabled", (False, True))
async def test_sql_web_conversation_uses_one_send_stream_and_history_contract(tmp_path, step_enabled):
    class WorkspacePolicy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)

    features = RuntimeFeatures("web-canonical")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with AsyncExitStack() as stack:
        state = await stack.enter_async_context(Persistence.open(
            "web-canonical", storage=PersistenceStorage.from_root(tmp_path),
        ))
        if step_enabled:
            native = await state.create_step_store()
            stack.push_async_callback(native.close)
            features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
        runtime = await stack.enter_async_context(Runtime.open(
            persistence=state, features=model_features(RuntimeUsageModels(), features),
            authorization=WorkspacePolicy(),
        ))
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
                    assert accepted.status_code == 202, accepted.text
                    assert set(accepted.json()) == {"turn_id"}
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


@pytest.mark.asyncio
async def test_web_upload_sends_binary_to_default_agent_and_keeps_it_in_object_store(tmp_path):
    from base64 import b64encode

    from pydantic_ai.messages import BinaryContent, ModelResponse, TextPart, UserPromptPart

    from linktools.ai.execution import StoredUserInput
    from linktools.ai.persistence.codec import decode_domain
    from linktools.ai.persistence.turns import SessionTurnKey
    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    class WorkspacePolicy(TenantAuthorizationPolicy):
        pass

    seen = []
    binary = bytes(range(256)) * 2304  # 576 KiB

    async def model(messages, _info):
        seen.extend(
            (item.identifier, item.data)
            for message in messages for part in message.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content if isinstance(item, BinaryContent)
        )
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("web-upload")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    model_features(_Models(_UsageFunctionModel(model)), features)
    async with (
        Persistence.open("web-upload", storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=WorkspacePolicy()) as runtime,
        client(create_app(runtime=runtime)) as http,
    ):
        config = (await http.get("/api/config")).json()
        assert config["attachment_upload"] == {"max_files": 32}
        created = await http.post("/api/sessions", headers=_HEADERS, json={
            "session_id": "chat", "request_id": "create",
        })
        assert created.status_code == 201, created.text
        url = "/api/session/messages?session_id=chat"
        payload = {"prompt": "Read my upload", "request_id": "upload"}
        old = await http.post(url, headers=_HEADERS, json={**payload, "files": ["server-file.bin"]})
        assert old.status_code == 400 and old.json()["safe_details"]["reason"] == "use_attachment_upload"
        item = {"name": "file.bin", "media_type": "application/octet-stream"}
        broken = await http.post(url, headers=_HEADERS, json={
            **payload, "attachments": [{**item, "data": "%%%%"}],
        })
        assert broken.status_code == 400 and broken.json()["safe_details"]["reason"] == "attachment_encoding_invalid"

        request = {**payload, "attachments": [{**item, "data": b64encode(binary).decode()}]}
        accepted = await http.post(url, headers=_HEADERS, json=request)
        assert accepted.status_code == 202, accepted.text
        turn_id = accepted.json()["turn_id"]
        repeat = await http.post(url, headers=_HEADERS, json=request)
        assert repeat.status_code == 202 and repeat.json() == {"turn_id": turn_id}
        actor = runtime.default_principal
        session = await runtime.sessions.get("chat", principal=actor)
        result = await session._turn(turn_id, actor).wait_final(timeout_seconds=10)
        assert result.view["output"] == {"text": "done"}
        assert seen == [("file.bin", binary)]
        row = await state.session_turn_reader.get_turn(SessionTurnKey("chat", turn_id))
        stored = decode_domain(row["request_json"]["input"], StoredUserInput)
        assert [ref.size for ref in stored.payload.object_refs()] == [len(binary)]
        history = (await http.get(f"/api/session?session_id=chat&turn_id={turn_id}")).json()
        assert history["user_input"]["attachments"][0]["size"] == len(binary)
        assert request["attachments"][0]["data"] not in json.dumps(history)


@pytest.mark.asyncio
async def test_web_send_reports_default_agent_feature_that_prevents_admission(tmp_path):
    class HostPolicy(TenantAuthorizationPolicy):
        pass

    features = RuntimeFeatures("web-send-diagnostics")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=("worker",))
    features.agent("worker", allow_tools=(), allow_skills=(), allow_subagents=())
    model_features(RuntimeUsageModels(), features)
    async with (
        Persistence.open("web-send-diagnostics", storage=PersistenceStorage.from_root(tmp_path)) as state,
        Runtime.open(persistence=state, features=features, authorization=HostPolicy()) as runtime,
        client(create_app(runtime=runtime)) as http,
    ):
        created = await http.post("/api/sessions", headers=_HEADERS, json={
            "session_id": "chat", "request_id": "create",
        })
        assert created.status_code == 201, created.text
        rejected = await http.post("/api/session/messages?session_id=chat", headers=_HEADERS, json={
            "prompt": "question", "request_id": "send",
        })
        assert rejected.status_code == 400, rejected.text
        assert rejected.json()["safe_details"] == {
            "field": "agent", "reason": "turn_agent_features_unsupported", "features": ["subagents"],
        }
        state_view = await runtime.history.turn_session_state("chat", principal=runtime.default_principal)
        assert state_view["active_turn_id"] is None
