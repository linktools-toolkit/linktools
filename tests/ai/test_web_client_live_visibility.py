#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completed client exchanges are readable before their execution finishes."""

from linktools.ai.core import CommitPolicy
from linktools.ai.capability import TraceCapability
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.runtime import RuntimeDomain
import asyncio
import base64
import json
from contextlib import aclosing
from pathlib import Path
from time import monotonic

import httpx2
import pytest

pytest.importorskip("starlette")

from linktools.ai.capability import AgentContext, RuntimeFeatures
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime, RuntimeStorage, TaskModelProjection
from linktools.ai.runtime.state._step_archive import (
    CapturedExecutionProjection, _AgentRunProjectionFlight,
)
from linktools.ai.web import create_app

from .test_model_client_request_capture import _registry, _response
from .test_model_client_response_capture import _DONE, _Stream
from .test_web_console_viewers import _client


@pytest.mark.asyncio
@pytest.mark.parametrize("hold_commit", (False, True))
async def test_web_client_live_visibility_follows_completed_request_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hold_commit: bool,
) -> None:
    release_response = asyncio.Event()
    tool_entered = asyncio.Event()
    release_tool = asyncio.Event()
    commit_entered = asyncio.Event()
    release_commit = asyncio.Event()
    sent: list[bytes] = []
    response_started_at = 0.0
    commit_started_at = 0.0
    wire_response = b"data: " + json.dumps({
        "id": "chatcmpl-live", "object": "chat.completion.chunk", "created": 1,
        "model": "gpt-4o", "unknown_provider_field": {"preserved": True},
        "choices": [{"index": 0, "delta": {"role": "assistant", "tool_calls": [{
            "index": 0, "id": "call-wait", "type": "function",
            "function": {"name": "wait_for_release", "arguments": "{}"},
        }]}, "finish_reason": "tool_calls"}],
    }).encode() + b"\r\n\r\n" + _DONE
    response_stream = _Stream([wire_response[:23], wire_response[23:]])

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal response_started_at
        sent.append(request.content)
        if len(sent) != 1:
            return _response(request)
        assert json.loads(request.content)["stream"] is True
        await release_response.wait()
        response_started_at = monotonic()
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=response_stream,
        )

    async def wait_for_release(_ctx: AgentContext[None]) -> str:
        tool_entered.set()
        await release_tool.wait()
        return "tool finished"

    models, clients = _registry(monkeypatch, handler)
    storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    group = RuntimeFeatures[None]("web-client-live")
    group.tool(wait_for_release, effect_policy="replay_safe")
    group.agent("default", model="default", allow_tools=("wait_for_release",))
    group.capability(TraceCapability(record_http_bodies=True, storage=storage.plan.route(RuntimeDomain.TRACE), commit=CommitPolicy.batch(buffer=InMemoryCacheBackend(max_bytes=65536), batch_size=128, flush_interval=3.0)), scope="runtime")
    async with Runtime.open(
        "web-client-live", models=models, storage=storage,
        features=group,
    ) as runtime:
        commit = runtime._state.run_store.commit_captured_execution_projection

        async def observed_commit(
            captured: CapturedExecutionProjection, flight: _AgentRunProjectionFlight,
            *, execution_id: str,
        ) -> None:
            nonlocal commit_started_at
            if any(item.model_request_seq == 1 and item.status == "SUCCEEDED"
                   for item in captured.interactions):
                commit_started_at = monotonic()
                commit_entered.set()
                if hold_commit:
                    # Hold the existing commit boundary, leaving staging and the
                    # real observer active so premature publication is detectable.
                    await release_commit.wait()
            await commit(captured, flight, execution_id=execution_id)

        monkeypatch.setattr(runtime._state.run_store, "commit_captured_execution_projection", observed_commit)
        execution = await runtime.agents.get("default").start("原始\nrequest with  spaces")
        async with _client(create_app(runtime=runtime)) as http:
            path = f"/api/executions/{execution.execution_id}/models?include_client_requests=true"

            async def observe_and_read() -> TaskModelProjection:
                async with aclosing(execution.watch(include_model_interactions=True)) as events:
                    async for envelope in events:
                        projection = envelope.event
                        if not isinstance(projection, TaskModelProjection):
                            continue
                        assert projection.visibility == "durable_history"
                        assert projection.item.model_request_seq == 1
                        assert not projection.item.content_included
                        assert not projection.item.client_requests_included
                        assert projection.item.request == {} and projection.item.response is None
                        if projection.item.status == "RUNNING":
                            # Begin the response after the RUNNING batch is durable;
                            # the following completion uses the unmodified timer.
                            release_response.set()
                            continue
                        assert projection.item.status == "SUCCEEDED"
                        assert commit_entered.is_set()
                        assert not hold_commit or release_commit.is_set()
                        assert tool_entered.is_set() and not release_tool.is_set()
                        assert response_stream.closes == 1 and len(sent) == 1

                        # One notification triggers one read, with no token polling,
                        # retry, explicit flush, or wait for execution completion.
                        response = await http.get(path)
                        assert response.status_code == 200, response.text
                        assert response.headers["cache-control"] == "private, no-store"
                        item, = response.json()["items"]
                        assert item["status"] == "SUCCEEDED"
                        assert item["client_requests_included"] and not item["content_included"]
                        assert item["request"] == {} and item["response"] is None
                        attempt, = item["client_requests"]
                        assert attempt["body_encoding"] == "base64"
                        assert base64.b64decode(attempt["body"]) == sent[0]
                        raw_response = attempt["response"]
                        assert raw_response["body_encoding"] == "base64"
                        assert base64.b64decode(raw_response["body"]) == wire_response
                        assert raw_response["status_code"] == 200
                        assert raw_response["format"] == "sse" and raw_response["complete"]
                        assert raw_response["content_encoding"] is None
                        info = await runtime.executions.inspect(
                            execution.execution_id, principal=runtime.default_principal,
                        )
                        assert info.status is ExecutionStatus.STARTED
                        return projection
                raise AssertionError("execution ended without a completed model projection")

            observed = asyncio.create_task(observe_and_read())
            try:
                await asyncio.wait_for(tool_entered.wait(), 10)
                await asyncio.wait_for(commit_entered.wait(), 10)
                # Leave the default approximately three-second batch window intact.
                # A generous lower bound distinguishes batching from eager flushes.
                assert commit_started_at - response_started_at >= 0.5
                if hold_commit:
                    done, _ = await asyncio.wait((observed,), timeout=0.05)
                    assert not done, "terminal model projection preceded durable commit"
                    response = await http.get(path)
                    assert response.status_code == 200, response.text
                    item, = response.json()["items"]
                    assert item["status"] == "RUNNING"
                    assert not item["client_requests"]
                    assert not release_tool.is_set()
                    release_commit.set()
                projection = await asyncio.wait_for(observed, 10)
                assert projection.root_execution_id == execution.execution_id
                assert len(sent) == 1
            finally:
                release_response.set()
                release_commit.set()
                release_tool.set()
                if not observed.done():
                    observed.cancel()
                await asyncio.gather(observed, return_exceptions=True)
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert len(sent) == 2
    assert clients and all(client.is_closed for client in clients)
