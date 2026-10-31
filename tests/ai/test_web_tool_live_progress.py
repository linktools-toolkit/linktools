#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The Web event transport exposes a tool call before that tool returns."""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

pytest.importorskip("starlette")

from linktools.ai.capability import AgentContext, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.web import create_app
from ._runtime_test_helpers import _Models, model_features


@pytest.mark.asyncio
async def test_web_sends_tool_start_while_tool_is_still_running() -> None:
    entered, release, saw_start, disconnected = (asyncio.Event() for _ in range(4))
    requests = 0

    async def model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        nonlocal requests
        requests += 1
        if requests == 1:
            yield {0: DeltaToolCall(name="wait_for_release", json_args="{}", tool_call_id="call-live")}
        else:
            yield "finished"

    async def wait_for_release(_ctx: AgentContext[None]) -> str:
        entered.set()
        await release.wait()
        return "observed result"

    features = RuntimeFeatures("web-live-tool")
    features.tool(wait_for_release, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("wait_for_release",))
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open("web-live-tool", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(FunctionModel(stream_function=model)), features)) as runtime:
            execution = await runtime.agents.get("default").start("Use the waiting tool")
            app = create_app(runtime=runtime)
            initial = False
            frames: list[str] = []

            async def receive() -> dict[str, object]:
                nonlocal initial
                if not initial:
                    initial = True
                    return {"type": "http.request", "body": b"", "more_body": False}
                await disconnected.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, object]) -> None:
                if message["type"] != "http.response.body" or not message.get("body"):
                    return
                frame = message["body"].decode()
                frames.append(frame)
                if "TOOL_CALL_STARTED" in frame:
                    envelope = json.loads(frame.split("data: ", 1)[1])
                    payload = envelope["item"]["event"]["payload"]
                    assert payload["tool_name"] == "wait_for_release" and payload["call_id"] == "call-live"
                    assert not release.is_set()
                    saw_start.set()

            path = f"/api/executions/{execution.execution_id}/events"
            scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
                     "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
                     "headers": [(b"host", b"127.0.0.1:8765")], "server": ("127.0.0.1", 8765),
                     "client": ("127.0.0.1", 12345)}
            stream = asyncio.create_task(app(scope, receive, send))
            try:
                await asyncio.wait_for(entered.wait(), 10)
                await asyncio.wait_for(saw_start.wait(), 10)
                current = await runtime.executions.inspect(execution.execution_id, principal=runtime.default_principal)
                assert current.status is ExecutionStatus.STARTED
                assert not any("TOOL_CALL_FINISHED" in frame for frame in frames)
                release.set()
                await asyncio.wait_for(execution.wait(), 10)
                await asyncio.wait_for(stream, 10)
                assert any("TOOL_CALL_FINISHED" in frame and "SUCCEEDED" in frame for frame in frames)
                assert any(frame.startswith("event: snapshot") and "SUCCEEDED" in frame for frame in frames)
            finally:
                release.set()
                disconnected.set()
                if not stream.done():
                    stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)
