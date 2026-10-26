#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""A real child transport failure remains actionable after parent persistence."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx2
import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_model_client_request_capture import _binding


_PARTIAL = "unverified child fragment"


class _DisconnectedStream(httpx2.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        chunk = {
            "id": "synthetic-child", "object": "chat.completion.chunk", "created": 1,
            "model": "gpt-4o", "choices": [{"index": 0, "finish_reason": None,
                "delta": {"role": "assistant", "content": _PARTIAL}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        }
        yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
        raise httpx2.RemoteProtocolError("synthetic disconnect")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_child_stream_failure_details_survive_parent_history_and_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    streams: list[_DisconnectedStream] = []
    feedback: list[str] = []
    parent_requests = 0

    def transport(request: httpx2.Request) -> httpx2.Response:
        assert json.loads(request.content)["stream"] is True
        stream = _DisconnectedStream()
        streams.append(stream)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream,
        )

    async def parent(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal parent_requests
        parent_requests += 1
        returns = [part for message in messages for part in message.parts
                   if isinstance(part, ToolReturnPart)]
        if returns:
            assert len(returns) == 1 and returns[0].outcome == "failed"
            feedback.append(str(returns[0].content))
            return ModelResponse(parts=[TextPart("Child work failed; no findings are available.")])
        assert any(tool.name == "delegate_task" for tool in info.function_tools)
        return ModelResponse(parts=[ToolCallPart(
            "delegate_task", {"subagent_id": "child", "task": "inspect the synthetic input"},
            tool_call_id="delegate-child",
        )])

    binding, clients = _binding(monkeypatch, transport, retries=1)
    features = RuntimeFeatures("child-stream-failure")
    features.model("parent", _Models(_UsageFunctionModel(parent)))
    features.model("child", binding)
    features.agent("default", model="parent", allow_tools=(), allow_skills=(), allow_subagents=("child",))
    features.agent("child", model="child", allow_tools=(), allow_skills=(), allow_subagents=())
    features.capability(TraceCapability(), scope="runtime")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")

    def storage() -> PersistenceStorage:
        if backend == "filesystem":
            return PersistenceStorage.filesystem(tmp_path / "state")
        return PersistenceStorage.sqlite(tmp_path / "state.db")

    async with Persistence.open('child-stream-failure', storage=storage()) as persistence_89, Runtime.attach(persistence=persistence_89, features=features) as runtime:
        execution = await runtime.agents.get().start("delegate once and report any failure")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert parent_requests == 2 and len(feedback) == 1
        children = await runtime.executions.list_children(
            execution.execution_id, principal=runtime.default_principal,
        )
        assert len(children) == 1
        child = await runtime.executions.get(children[0].execution_id)
        child_result = (await child.wait(timeout_seconds=5)).result
        assert child_result.status is ExecutionStatus.FAILED
        assert child_result.safe_error_details is not None
        details = child_result.safe_error_details
        assert details["model_request_seq"] == 1
        assert details["stream_attempt_seq"] == details["stream_attempt_count"] == 2
        for key in ("model_request_seq", "stream_attempt_seq", "stream_attempt_count", "model_name", "phase", "reason"):
            if key in details:
                assert json.dumps(key) + ":" + json.dumps(details[key], separators=(",", ":")) in feedback[0]
        assert child_result.error_code in feedback[0]
        assert "Do not infer findings" in feedback[0]
        assert "Continue from the parent" not in feedback[0]
        for secret in (_PARTIAL, "capture.invalid", "connection-secret", "synthetic disconnect"):
            assert secret not in feedback[0]

        interactions = (await child.model_interactions(include_content=True)).items
        assert len(interactions) == 1 and interactions[0].response is None
        attempts = interactions[0].stream_attempts
        assert attempts is not None and [attempt.status for attempt in attempts] == ["FAILED", "FAILED"]
        assert all(_PARTIAL in json.dumps(attempt.response) for attempt in attempts)
        assert (interactions[0].usage.input_tokens, interactions[0].usage.output_tokens) == (14, 2)
        assert _PARTIAL not in str((await child.transcript(include_content=True)).items)
        parent_history = (await execution.history(include_content=True)).items
        assert [item.content for item in parent_history if item.item_kind == "tool_result"] == feedback
        parent_id, child_id = execution.execution_id, child.execution_id

    assert len(streams) == 2 and all(stream.closed for stream in streams)
    assert all(client.is_closed for client in clients)
    async with Persistence.open('child-stream-failure', storage=storage()) as persistence_127, Runtime.attach(persistence=persistence_127, features=features) as reopened:
        execution = await reopened.executions.get(parent_id)
        child = await reopened.executions.get(child_id)
        assert (await execution.history(include_content=True)).items == parent_history
        assert (await child.wait(timeout_seconds=5)).result.safe_error_details == details
        assert (await child.model_interactions(include_content=True)).items == interactions
    assert parent_requests == 2 and len(streams) == 2
