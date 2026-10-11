#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual SDK client attempts follow existing Runtime request boundaries."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import Principal, RunBudget, agent_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelClientRequest, ModelClientRequestRecorder
from linktools.ai.model import _openai
from linktools.ai.runtime import ExecutionInputContext, Runtime, RuntimeStorage
from linktools.ai.runtime._metric_capability import ModelObservationCapability
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_archive import StagingAgentRunStore

from .test_model_client_request_capture import _agents, _model, _registry, _response
from .test_model_client_response_capture import _DONE, _EVENT, _Stream, _invoke


async def _attempts(storage: RuntimeStorage, execution: Any) -> list[tuple[ModelClientRequest, ...]]:
    items = (await execution.model_interactions()).items
    archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
    records = await archive.list_model_interactions(agent_run_id=agent_run_id(
        namespace="default", tenant_id="default", execution_id=execution.execution_id,
        agent_run_seq=items[0].agent_run_seq,
    ))
    values = await archive.resolve_model_interactions(records, include_client_requests=True)
    assert all(value[3] is not None for value in values)
    return [value[3] for value in values]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_client_recorder_binds_model_dispatch_in_preexisting_task(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    sent: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        return _response(request)

    model, _clients = _model(monkeypatch, handler)
    recorder = ModelClientRequestRecorder()
    wrapped = recorder.wrap_model(model)
    ready = asyncio.Event()

    async def consume() -> None:
        await ready.wait()
        await _invoke(wrapped, streaming)

    async with model:
        task = asyncio.create_task(consume())
        with recorder:
            ready.set()
            await task
    assert [attempt.body for attempt in recorder.requests] == sent
    assert len(sent) == 1
    assert recorder.requests[0].response is not None
    assert recorder.requests[0].response.complete


@pytest.mark.asyncio
async def test_agent_dispatch_without_a_logical_request_still_fails_integrity() -> None:
    model = TestModel()
    capability = ModelObservationCapability(
        None, source_namespace="default", tenant_id="default", execution_id="execution",
        session_id=None, agent_run_id="run", agent_id="default",
    )
    context = RunContext(deps=None, model=model, usage=RunUsage(), run_step=1)
    request = await capability.before_model_request(context, ModelRequestContext(
        model=model, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters(),
    ))
    with pytest.raises(AIError) as caught:
        await request.model.request([], None, ModelRequestParameters())
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("budgeted", (False, True))
async def test_runtime_records_compaction_and_agent_dispatches_separately(
    monkeypatch: pytest.MonkeyPatch, budgeted: bool,
) -> None:
    sent: list[bytes] = []
    received: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        response = _response(request)
        received.append(response.content)
        return response

    models, _clients = _registry(monkeypatch, handler)
    chat_model = _openai.OpenAIChatModel

    def small_model(*args: Any, **kwargs: Any) -> Any:
        return chat_model(*args, profile={"context_window": 1000}, **kwargs)

    monkeypatch.setattr(_openai, "OpenAIChatModel", small_model)
    history = []
    for index in range(20):
        history.extend((
            ModelRequest(parts=[UserPromptPart(f"user-{index} " + "x" * 240)]),
            ModelResponse(parts=[TextPart(f"answer-{index} " + "y" * 240)]),
        ))
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("default", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        execution = await runtime.agents.get("default").start(
            "continue", input_context=ExecutionInputContext.from_messages(history),
            budget=RunBudget(model_requests=2) if budgeted else None,
        )
        assert (await execution.wait(timeout_seconds=10)).result.status == "SUCCEEDED"
        items = (await execution.model_interactions()).items
        assert [item.purpose for item in items] == ["compaction", "agent"]
        assert [item.model_request_seq for item in items] == [1, 2]
        if budgeted:
            usage = await execution.budget_usage()
            assert usage.model_requests == 2 and usage.in_flight_model_requests == 0
        attempts = await _attempts(storage, execution)
        assert len(attempts) == 2 and all(len(value) == 1 for value in attempts)
        assert [value[0].body for value in attempts] == sent
        assert [value[0].response.body for value in attempts] == received
        assert all(value[0].response.complete for value in attempts)
        assert [json.loads(body).get("stream", False) for body in sent] == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("outcome", ("success", "retry", "read_error", "cancel"))
async def test_runtime_flushes_original_attempts_only_at_request_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, outcome: str,
) -> None:
    partial = _EVENT.replace(b'"finish_reason": "stop"', b'"finish_reason": null')
    chunks: list[bytes | BaseException] = [partial]
    if outcome == "read_error":
        chunks.append(httpx2.ReadError("fixture stream failure"))
    elif outcome != "cancel":
        chunks.append(_DONE)
    stream = _Stream(chunks, block=outcome == "cancel")
    stage = StagingAgentRunStore.stage_model_interaction

    def stage_after_stream_close(store: StagingAgentRunStore, interaction: Any) -> None:
        if interaction.status != "RUNNING":
            assert stream.closes == 1
            assert interaction.client_requests[-1].response is not None
        stage(store, interaction)

    monkeypatch.setattr(StagingAgentRunStore, "stage_model_interaction", stage_after_stream_close)
    sent: list[bytes] = []
    failure = b'{"error":{"message":"retry once"},"unknown":true}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.content)
        if outcome == "retry" and len(sent) == 1:
            return httpx2.Response(429, headers={"content-type": "application/json"}, content=failure)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    models, _clients = _registry(monkeypatch, handler, retries=int(outcome == "retry"))
    storage = RuntimeStorage.in_memory() if backend == "memory" else RuntimeStorage.sqlite(tmp_path / "state.db")
    async with Runtime.open("default", models=models, storage=storage, capabilities=(_agents(),)) as runtime:
        execution = await runtime.agents.get("default").start("original prompt")
        if outcome == "cancel":
            await asyncio.wait_for(stream.blocked.wait(), 5)
            run_id = agent_run_id(
                namespace="default", tenant_id="default", execution_id=execution.execution_id,
                agent_run_seq=1,
            )
            staged = await storage.run_store.list_model_interactions(agent_run_id=run_id)
            assert len(staged) == 1 and staged[0].status == "RUNNING"
            assert staged[0].client_requests is None
            archive = storage.run_store.read_store(RuntimeDomain.EXECUTION)
            records = await archive.list_model_interactions(agent_run_id=run_id)
            assert all(record.client_requests is None for record in records)
            await execution.cancel()
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status == {"read_error": "FAILED", "cancel": "CANCELLED"}.get(outcome, "SUCCEEDED")
        logical_requests = await _attempts(storage, execution)
        assert len(logical_requests) == 1
        attempts = logical_requests[0]
        assert [attempt.body for attempt in attempts] == sent
        assert len(attempts) == (2 if outcome == "retry" else 1)
        if outcome == "retry":
            assert attempts[0].response.status_code == 429
            assert attempts[0].response.body == failure
        response = attempts[-1].response
        assert response is not None and response.status_code == 200
        assert response.body == partial + (_DONE if outcome in {"success", "retry"} else b"")
        assert response.complete is (outcome in {"success", "retry"})
        assert stream.closes == 1


@pytest.mark.asyncio
async def test_concurrent_subagents_keep_original_attempts_with_their_own_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, list[bytes]] = {}
    children_ready = asyncio.Event()
    child_count = 0

    async def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal child_count
        messages = json.loads(request.content)["messages"]
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        sent.setdefault(prompt, []).append(request.content)
        if prompt.startswith("child-"):
            child_count += 1
            if child_count == 2:
                children_ready.set()
            await asyncio.wait_for(children_ready.wait(), 5)
        if prompt.startswith("root-") and not any(message["role"] == "tool" for message in messages):
            delta = {"role": "assistant", "tool_calls": [{
                "index": 0, "id": "delegate-" + prompt, "type": "function",
                "function": {"name": "delegate_task", "arguments": json.dumps({
                    "subagent_id": "child", "task": prompt.replace("root-", "child-"),
                })},
            }]}
            finish = "tool_calls"
        else:
            delta = {"role": "assistant", "content": "done " + prompt}
            finish = "stop"
        body = b"data: " + json.dumps({
            "id": "chatcmpl-" + prompt, "object": "chat.completion.chunk", "created": 1,
            "model": "gpt-4o", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }).encode() + b"\n\n" + _DONE
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    models, _clients = _registry(monkeypatch, handler)
    group = CapabilityGroup("application")
    group.agent("default", model="default", allow_tools=(), allow_subagents=("child",))
    group.agent("child", model="default", allow_tools=(), allow_subagents=())
    storage = RuntimeStorage.in_memory()
    principal = Principal("owner", "default")
    async with Runtime.open("default", models=models, storage=storage, capabilities=(group,)) as runtime:
        parents = await asyncio.gather(*(runtime.agents.get("default").start(f"root-{i}", principal=principal) for i in range(2)))
        results = await asyncio.gather(*(execution.wait(timeout_seconds=15) for execution in parents))
        assert all(value.result.status == "SUCCEEDED" for value in results)
        for index, parent in enumerate(parents):
            attempts = await _attempts(storage, parent)
            assert [value.body for request in attempts for value in request] == sent[f"root-{index}"]
            children = await runtime.executions.list_children(parent.execution_id, principal=principal)
            assert len(children) == 1
            child = await runtime.executions.get(children[0].execution_id, principal=principal)
            attempts = await _attempts(storage, child)
            assert [value.body for request in attempts for value in request] == sent[f"child-{index}"]
        assert child_count == 2
