#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The same SQL Web conversation API serves both Step capability settings."""

import asyncio
import json
from contextlib import AsyncExitStack

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart

pytest.importorskip("starlette")

from linktools.ai.capability import SessionCapability
from linktools.ai.core import PromptLimits, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.web import create_app

from ._runtime_test_helpers import RuntimeUsageModels, _Models, _UsageFunctionModel, model_features
from .test_web_console import _HEADERS, client


@pytest.mark.asyncio
@pytest.mark.parametrize("step_enabled", (False, True))
async def test_sql_web_conversation_uses_one_send_stream_and_history_contract(tmp_path, step_enabled):
    class WorkspacePolicy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)

    async def model(_messages, _info):
        return ModelResponse(parts=[
            ThinkingPart("displayable thought", signature="private-provider-signature"), TextPart("done"),
        ])

    features = RuntimeFeatures("web-canonical", defaults={"execution": {"retain_history": True}})
    async with AsyncExitStack() as stack:
        state = await stack.enter_async_context(Persistence.open(
            "web-canonical", storage=PersistenceStorage.from_root(tmp_path),
        ))
        if step_enabled:
            features.capability(SessionCapability(checkpoints=True), scope="runtime")
        runtime = await stack.enter_async_context(Runtime.open(
            persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
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
                        "prompt": "question", "request_id": "send", "planning": step_enabled,
                    })
                    assert accepted.status_code == 202, accepted.text
                    assert set(accepted.json()) == {"turn_id"}
                    turn_id = accepted.json()["turn_id"]
                    seen = []
                    deltas = []
                    thinking_timing = None
                    while True:
                        frame = await frames.get()
                        assert frame["turn_id"] == turn_id
                        seen.append(frame["type"])
                        if frame["type"] == "delta":
                            assert "private-provider-signature" not in json.dumps(frame)
                            assert type(frame["part_start"]) is bool and frame["part_start"] is True
                            assert frame["model_key"] == "model:1:1" and frame["attempt"] == 1
                            assert type(frame["part_index"]) is int
                            assert "stream_truncated" not in frame
                            deltas.append((frame["delta_kind"], frame["text"]))
                        if frame["type"] == "thinking_completed":
                            thinking_timing = frame
                        if frame["type"] == "terminal":
                            break
                    assert seen[0] == "admitted"
                    assert deltas == [("thinking", "displayable thought"), ("text", "done")]
                    assert frame.get("error_code") is None
                    assert thinking_timing is not None
                    assert set(thinking_timing) == {
                        "type", "turn_id", "generation", "model_key", "attempt", "part_index", "elapsed_ms",
                    }
                    assert thinking_timing["model_key"] == "model:1:1"
                    assert thinking_timing["attempt"] == 1 and thinking_timing["part_index"] == 0
                    assert isinstance(thinking_timing["elapsed_ms"], int) and thinking_timing["elapsed_ms"] >= 0
                    assert seen.index("thinking_completed") < seen.index("terminal")
            finally:
                disconnect.set()
                await stream

            result = await http.get(f"/api/session?session_id=chat&turn_id={turn_id}")
            assert result.status_code == 200
            assert result.json()["turn_id"] == turn_id
            assert result.json()["conversation_committed"] is True
            stored = await state.session_turn_reader.get_turn(SessionTurnKey("chat", turn_id))
            assert stored["request_json"].get("planning", False) is step_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("limits", [None, PromptLimits(max_turn_attachment_bytes=2 * 1024 * 1024)])
async def test_web_upload_sends_binary_to_default_agent_and_keeps_it_in_object_store(tmp_path, limits):
    from base64 import b64encode

    from pydantic_ai.messages import BinaryContent, UserPromptPart

    from linktools.ai.execution import StoredUserInput
    from linktools.ai.persistence.codec import decode_domain

    class WorkspacePolicy(TenantAuthorizationPolicy):
        pass

    seen = []
    binary = bytes(range(256)) * 7168  # 1.75 MiB; Base64 exceeds the former 2 MiB HTTP limit

    async def model(messages, _info):
        seen.extend(
            (item.identifier, item.data)
            for message in messages for part in message.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content if isinstance(item, BinaryContent)
        )
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("web-upload", limits=limits, defaults={"execution": {"retain_history": True}})
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.from_root(tmp_path)
    async with (
        Persistence.open("web-upload", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=WorkspacePolicy()) as runtime,
        client(create_app(runtime=runtime)) as http,
    ):
        config = (await http.get("/api/config")).json()
        expected_limit = (limits or PromptLimits()).max_turn_attachment_bytes
        assert config["attachment_upload"] == {"max_files": 32, "max_bytes": expected_limit}
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
        if limits is not None:
            oversized = await http.post(url, headers=_HEADERS, json={
                **payload, "attachments": [{**item, "data": b64encode(bytes(expected_limit + 1)).decode()}],
            })
            assert oversized.status_code == 413 and oversized.json()["code"] == "REQUEST_BODY_TOO_LARGE"
        frames = []
        unsubscribe = runtime.sessions.observe_turn_session("chat", frames.append)
        try:
            accepted = await http.post(url, headers=_HEADERS, json=request)
            assert accepted.status_code == 202, accepted.text
            turn_id = accepted.json()["turn_id"]
            repeat = await http.post(url, headers=_HEADERS, json=request)
            assert repeat.status_code == 202 and repeat.json() == {"turn_id": turn_id}
            renamed = await http.post(url, headers=_HEADERS, json={
                **payload, "attachments": [{**request["attachments"][0], "name": "different.bin"}],
            })
            assert renamed.status_code == 409 and renamed.json()["code"] == "IDEMPOTENCY_CONFLICT"
            actor = runtime.default_principal
            session = await runtime.sessions.get("chat", principal=actor)
            result = await session._turn(turn_id, actor).wait_final(timeout_seconds=10)
        finally:
            unsubscribe()
        assert result.view["output"] == {"text": "done"}
        (identifier, seen_binary), = seen
        assert identifier != "file.bin" and seen_binary == binary
        names = {
            frame["type"]: frame["attachments"][0]["display_name"]
            for frame in frames if frame["type"] in {"admitted", "terminal"}
        }
        assert names == {"admitted": "file.bin", "terminal": "file.bin"}
        display_prompt = {"kind": "items", "items": [{"kind": "text", "text": payload["prompt"]}]}
        assert {frame["type"]: frame["prompt"] for frame in frames if frame["type"] in {"admitted", "terminal"}} == {
            "admitted": display_prompt, "terminal": display_prompt,
        }
        row = await state.session_turn_reader.get_turn(SessionTurnKey("chat", turn_id))
        stored = decode_domain(row["request_json"]["input"], StoredUserInput)
        assert [ref.size for ref in stored.payload.object_refs()] == [len(binary)]
        history = (await http.get(f"/api/session?session_id=chat&turn_id={turn_id}")).json()
        assert history["user_input"]["prompt"] == display_prompt
        attachment, = history["user_input"]["attachments"]
        assert attachment["size"] == len(binary) and attachment["display_name"] == "file.bin"
        assert set(attachment) == {"attachment_id", "media_type", "source", "position", "display_name", "size"}
        assert request["attachments"][0]["data"] not in json.dumps(history)


    async with (
        Persistence.open("web-upload", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=WorkspacePolicy()) as runtime,
        client(create_app(runtime=runtime)) as http,
    ):
        reopened = await http.get("/api/session", params={"session_id": "chat", "turn_id": turn_id})
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()["user_input"]["attachments"] == [attachment]
        assert reopened.json()["user_input"]["prompt"] == display_prompt


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


def test_native_sse_projects_tool_updates_without_unused_or_empty_fields():
    from linktools.ai.web._app import _native_turn_event_wire, _sse_frame

    scope = {"turn_id": "turn", "turn_no": 9, "generation": 2, "revision": 15}
    live = _native_turn_event_wire({
        **scope, "type": "tool", "item": {
            "kind": "tool", "key": "tool:1", "tool_name": "lookup", "status": "SUCCEEDED",
            "position": [2, 0], "elapsed_ms": 0, "argument_preview": None, "error_message": None,
        },
    })
    assert live == {
        "turn_id": "turn", "generation": 2, "type": "tool",
        "item": {"kind": "tool", "key": "tool:1", "tool_name": "lookup", "status": "SUCCEEDED", "elapsed_ms": 0},
    }
    assert _sse_frame("turn", live, compact=True) == (
        'event: turn\ndata: {"turn_id":"turn","generation":2,"type":"tool",'
        '"item":{"kind":"tool","key":"tool:1","tool_name":"lookup","status":"SUCCEEDED","elapsed_ms":0}}\n\n'
    )
    assert _native_turn_event_wire({
        **scope, "type": "progress", "item": {"kind": "model", "key": "model:1:2", "status": "RUNNING"},
    }) is None
    terminal = _native_turn_event_wire({
        **scope, "type": "terminal", "state": "succeeded", "prompt": "question",
        "attachments": [], "items": [{"item_kind": "assistant", "content": "answer"}],
        "output": {"text": "answer"}, "error_code": None,
    })
    assert terminal == {
        **scope, "type": "terminal", "state": "succeeded", "prompt": "question",
        "items": [{"item_kind": "assistant", "content": "answer"}],
    }


@pytest.mark.asyncio
async def test_native_preview_supplies_only_real_positions_and_resets_retried_stream(tmp_path, monkeypatch):
    from linktools.ai.core import ExecutionDeltaType, ExecutionEventType
    from linktools.ai.runtime._agent_executor import AgentExecutor, DurableBoundary, LiveDelta
    from linktools.ai.runtime.turn_clients import TurnObserver
    from linktools.ai.web._app import _native_turn_event_wire

    class WorkspacePolicy(TenantAuthorizationPolicy):
        pass

    execute = AgentExecutor.execute_regenerable
    position = {"agent_run_seq": 1, "model_request_seq": 7, "stream_attempt_seq": 1, "part_index": 0}
    text = ExecutionDeltaType.ASSISTANT_TEXT_DELTA

    async def exercise_stream(executor, *, event_sink, **kwargs):
        await event_sink(LiveDelta(text, "attempt one", {**position, "part_start": True,
                                                        "private": "do not project"}))
        await event_sink(LiveDelta(text, " continuation", {**position, "part_start": False}))
        await event_sink(LiveDelta(text, "attempt two", {**position, "stream_attempt_seq": 2,
                                                        "part_start": True, "stream_truncated": True}))
        await event_sink(LiveDelta(text, "stale", {**position, "part_start": False}))
        await event_sink(LiveDelta(text, "unpositioned", {"part_index": 6, "part_start": 1}))

        tool = {"call_id": "ordered", "tool_name": "read_file", "tool_position": [2, 2],
                "agent_run_seq": 1, "tool_model_request_seq": 7, "raw_arguments": "never public"}
        await event_sink(DurableBoundary(ExecutionEventType.TOOL_CALL_STARTED, tool))
        await event_sink(DurableBoundary(ExecutionEventType.TOOL_CALL_STARTED,
                                         {**tool, "call_id": "unordered", "tool_model_request_seq": None}))
        return await execute(executor, event_sink=event_sink, **kwargs)

    monkeypatch.setattr(AgentExecutor, "execute_regenerable", exercise_stream)
    features = model_features(RuntimeUsageModels())
    features.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "stream-positions", storage=PersistenceStorage.sqlite(tmp_path / "positions.db"),
        features=features, authorization=WorkspacePolicy(),
    ) as runtime:
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        with TurnObserver(runtime, "session", principal, local_workspace=False) as observer:
            turn = await runtime.sessions.start(
                "session", "question", agent="default", principal=principal, idempotency_key="one",
            )
            async with asyncio.timeout(5):
                frames = [frame async for frame in observer.events(turn.key.turn_id, append_only=False)]

    selected = [frame for frame in frames if frame.get("model_key") == "model:1:7"]
    assert [frame["type"] for frame in selected] == ["delta", "delta", "preview_reset", "delta"]
    first, continuation, reset, replacement = selected
    assert first["part_start"] is True and "stream_truncated" not in first
    assert continuation["part_start"] is False
    assert reset["attempt"] == 1 and replacement["attempt"] == 2
    assert replacement["stream_truncated"] is True and replacement["part_start"] is True
    assert "private" not in first
    assert not any(frame.get("text") == "stale" for frame in frames)
    unpositioned, = (frame for frame in frames if frame.get("text") == "unpositioned")
    assert set(unpositioned).isdisjoint({"model_key", "attempt", "part_index", "part_start"})
    ordered, unordered = (_native_turn_event_wire(frame) for frame in frames if frame["type"] == "tool")
    assert ordered["order"] == [1, 7, 2] and "position" not in ordered["item"]
    assert "order" not in unordered
    assert "raw_arguments" not in str(ordered)
