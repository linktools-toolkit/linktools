#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Original client requests remain separate from SDK message projections."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from types import TracebackType

import httpx2
import pytest
from openai import DefaultAsyncHttpxClient
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    InstructionPart, ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart,
    ToolCallPart, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from linktools.ai.capability import RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue, PromptLimits, agent_run_id
from linktools.ai.errors import AIError, ErrorCode

from linktools.ai.model import ModelClientRequestRecorder, ModelRegistry
from linktools.ai.model import _openai
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime._compaction import CompactionCapability
from linktools.ai.runtime._journal import ModelRequestJournal
from linktools.ai.runtime.state import RuntimeDomain


from linktools.ai.runtime import TraceCapability


class _ObservedHttpClient(DefaultAsyncHttpxClient):
    close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        await super().aclose()


def _registry(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[..., object],
    *,
    retries: int = 0,
) -> tuple[ModelRegistry, list[_ObservedHttpClient]]:
    clients: list[_ObservedHttpClient] = []

    def create_client(**kwargs: object) -> httpx2.AsyncClient:
        client = _ObservedHttpClient(
            transport=httpx2.MockTransport(handler), **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(_openai, "DefaultAsyncHttpxClient", create_client)
    registry = ModelRegistry.openai(
        model="gpt-4o", base_url="https://capture.invalid/v1",
        api_key="connection-secret-never-captured", max_retries=retries,
        retry_delay=0,
    )
    return registry, clients


def _model(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[..., object], *, retries: int = 0,
) -> tuple[Model, list[_ObservedHttpClient]]:
    registry, clients = _registry(monkeypatch, handler, retries=retries)
    return registry.capture().resolve("default").materialize(), clients


def _response(request: httpx2.Request) -> httpx2.Response:
    body = {
        "id": "chatcmpl-test", "object": "chat.completion", "created": 1,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"},
                     "finish_reason": "stop"}],
    }
    if not json.loads(request.content).get("stream"):
        return httpx2.Response(200, json=body)
    chunk = {
        "id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1,
        "model": "gpt-4o", "choices": [{"index": 0,
            "delta": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
    }
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"},
        text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
    )


async def _request(model: Model, content: str) -> None:
    await model.request(
        [ModelRequest(parts=[UserPromptPart(content)])], None, ModelRequestParameters(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", (False, True))
async def test_model_client_request_capture_matches_dispatch_and_retains_retries(
    monkeypatch: pytest.MonkeyPatch, streaming: bool,
) -> None:
    dispatched: list[bytes] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        dispatched.append(request.content)
        if len(dispatched) == 1:
            return httpx2.Response(429, json={"error": {"message": "retry"}})
        return _response(request)

    model, clients = _model(monkeypatch, handler, retries=1)
    messages = [
        ModelRequest(parts=[SystemPromptPart("system"), UserPromptPart("原始\nmessage")]),
        ModelResponse(parts=[ToolCallPart("lookup", '{"value": 1}', tool_call_id="call_1")]),
        ModelRequest(parts=[ToolReturnPart("lookup", "result\ntext", tool_call_id="call_1")]),
    ]
    parameters = ModelRequestParameters(instruction_parts=[InstructionPart("instruction")])
    settings = {"extra_body": {"metadata": {"after_mapping": True}}}
    async with model:
        with ModelClientRequestRecorder() as capture:
            if streaming:
                async with model.request_stream(messages, settings, parameters) as response:
                    async for _event in response:
                        pass
            else:
                await model.request(messages, settings, parameters)
        assert not clients[0].is_closed
    assert len(clients) == 1 and clients[0].is_closed
    assert tuple(item.body for item in capture.requests) == tuple(dispatched)
    assert len(capture.requests) == 2
    assert all(item.api == "openai-chat" and item.capture_error is None for item in capture.requests)
    body = json.loads(capture.requests[0].body)
    assert body["metadata"] == {"after_mapping": True}
    assert body["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "system", "content": "instruction"},
        {"role": "user", "content": "原始\nmessage"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1",
            "type": "function", "function": {"name": "lookup", "arguments": '{"value": 1}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result\ntext"},
    ]
    assert all(b"connection-secret" not in item.body for item in capture.requests)
    assert "capture.invalid" not in repr(capture.requests)


@pytest.mark.asyncio
async def test_model_client_request_capture_separates_nested_and_concurrent_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, _response)

    async def child(label: str) -> ModelClientRequestRecorder:
        with ModelClientRequestRecorder() as captured:
            await asyncio.sleep(0)
            await _request(model, label)
        return captured

    async with model:
        with ModelClientRequestRecorder() as outer:
            await _request(model, "before")
            with ModelClientRequestRecorder() as inner:
                await _request(model, "inner")
            children = await asyncio.gather(child("one"), child("two"))
            await asyncio.create_task(_request(model, "sdk-child-task"))
            await _request(model, "after")

    def prompts(capture: ModelClientRequestRecorder) -> list[str]:
        return [json.loads(item.body)["messages"][0]["content"] for item in capture.requests]

    assert prompts(outer) == ["before", "sdk-child-task", "after"]
    assert prompts(inner) == ["inner"]
    assert [prompts(item) for item in children] == [["one"], ["two"]]


@pytest.mark.asyncio
async def test_model_client_request_capture_does_not_collect_after_scope_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, _response)
    gate = asyncio.Event()

    async def delayed() -> None:
        await gate.wait()
        await _request(model, "late")

    async with model:
        with ModelClientRequestRecorder() as capture:
            task = asyncio.create_task(delayed())
        gate.set()
        await task
    assert capture.requests == ()


@pytest.mark.asyncio
async def test_model_client_request_capture_keeps_provider_failure_and_cancelled_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatched = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        if json.loads(request.content)["messages"][0]["content"] == "failure":
            return httpx2.Response(400, json={"error": {"message": "invalid"}})
        dispatched.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled dispatch must not complete")

    model, clients = _model(monkeypatch, handler)
    cancelled = ModelClientRequestRecorder()

    async def cancelled_request() -> None:
        with cancelled:
            await _request(model, "cancel")

    async with model:
        with ModelClientRequestRecorder() as failed:
            with pytest.raises(ModelHTTPError) as raised:
                await _request(model, "failure")
        assert raised.value.status_code == 400
        task = asyncio.create_task(cancelled_request())
        await dispatched.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(failed.requests) == len(cancelled.requests) == 1
    assert json.loads(cancelled.requests[0].body)["messages"][0]["content"] == "cancel"
    assert clients[0].is_closed


@pytest.mark.asyncio
async def test_model_client_request_capture_reports_unavailable_body_without_consuming_it() -> None:
    consumed = False

    async def content() -> AsyncIterator[bytes]:
        nonlocal consumed
        consumed = True
        yield b"body"

    request = httpx2.Request("POST", "https://capture.invalid/chat/completions", content=content())
    with ModelClientRequestRecorder() as capture:
        await _openai._capture_chat_request(request)
    assert not consumed
    assert len(capture.requests) == 1
    assert capture.requests[0].body is None
    assert capture.requests[0].capture_error == "body_unavailable"


@pytest.mark.asyncio
async def test_model_client_request_capture_client_lifecycle_follows_nested_agent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, clients = _model(monkeypatch, _response)
    agent = Agent(model)
    async with model:
        async with model:
            with ModelClientRequestRecorder() as first:
                await agent.run("one")
            assert len(clients) == 1 and not clients[0].is_closed
        assert not clients[0].is_closed
        with ModelClientRequestRecorder() as second:
            await agent.run("two")
    assert len(first.requests) == len(second.requests) == 1
    assert len(clients) == 1 and clients[0].is_closed
    async with agent:
        with ModelClientRequestRecorder() as reopened:
            await agent.run("three")
    assert len(reopened.requests) == 1
    assert len(clients) == 2 and all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_model_client_request_capture_preserves_final_body_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _clients = _model(monkeypatch, _response)
    original = [{"role": "developer", "content": [{"type": "text", "text": "actual"}]}]
    async with model:
        with ModelClientRequestRecorder() as recorder:
            await model.request(
                [ModelRequest(parts=[UserPromptPart("before-extra-body")])],
                {"extra_body": {"messages": original}}, ModelRequestParameters(),
            )
    assert json.loads(recorder.requests[0].body)["messages"] == original


@pytest.mark.asyncio
async def test_model_client_request_capture_does_not_close_another_active_model_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    both_dispatched = asyncio.Event()
    release_second = asyncio.Event()
    prompts: list[str] = []

    async def handler(request: httpx2.Request) -> httpx2.Response:
        label = json.loads(request.content)["messages"][0]["content"]
        prompts.append(label)
        if len(prompts) == 2:
            both_dispatched.set()
        await both_dispatched.wait()
        if label == "two":
            await release_second.wait()
        return _response(request)

    model, clients = _model(monkeypatch, handler)

    async def run(label: str) -> None:
        async with model:
            await _request(model, label)

    first = asyncio.create_task(run("one"))
    second = asyncio.create_task(run("two"))
    try:
        await asyncio.wait_for(first, 5)
        assert len(clients) == 1 and not clients[0].is_closed
    finally:
        release_second.set()
        await asyncio.wait_for(second, 5)
    assert clients[0].is_closed


class _ModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def __init__(self, model: Model) -> None:
        self.model = model

    def materialize(self) -> Model:
        return self.model


def _agents() -> RuntimeFeatures[object]:
    group = RuntimeFeatures[object]("application")
    group.agent("default", model="default", allow_tools=())
    return group


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("success", "failure", "cancel", "cancel_cleanup"))
async def test_model_client_request_capture_runtime_closes_owned_client(
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    dispatched = asyncio.Event()

    async def handler(request: httpx2.Request) -> httpx2.Response:
        dispatched.set()
        if outcome == "failure":
            return httpx2.Response(400, json={"error": {"message": "invalid"}})
        if outcome.startswith("cancel"):
            await asyncio.Event().wait()
        return _response(request)

    registry, clients = _registry(monkeypatch, handler)
    storage = RuntimeStorage.in_memory()
    features = _agents()
    features.capability(TraceCapability(), scope="runtime")
    async with Runtime.open(
        "default", models=registry, storage=storage, features=features,
    ) as runtime:
        with ModelClientRequestRecorder() as recorder:
            execution = await runtime.agents.get("default").start("hello")
            await asyncio.wait_for(dispatched.wait(), 5)
            if outcome == "cancel_cleanup":
                close = clients[0].aclose

                async def fail_close() -> None:
                    await close()
                    raise RuntimeError("cleanup")

                monkeypatch.setattr(clients[0], "aclose", fail_close)
            if outcome.startswith("cancel"):
                await execution.cancel()
            result = (await execution.wait(timeout_seconds=5)).result
        expected = {"success": ExecutionStatus.SUCCEEDED, "failure": ExecutionStatus.FAILED,
                    "cancel": ExecutionStatus.CANCELLED,
                    "cancel_cleanup": ExecutionStatus.CANCELLED}[outcome]
        assert result.status is expected
        assert recorder.requests == ()
        item = (await execution.model_interactions()).items[0]
        archive = storage.run_store.read_store(RuntimeDomain.TRACE)
        records = await archive.list_model_interactions(agent_run_id=agent_run_id(
            namespace="default", tenant_id="default", execution_id=item.execution_id, agent_run_seq=item.agent_run_seq,
        ))
        attempts = (await archive.resolve_model_interaction(records[0], include_client_requests=True))[3]
        assert attempts is not None and len(attempts) == 1
        assert len(clients) == 1 and clients[0].is_closed and clients[0].close_calls == 1


class _LifecycleModel(TestModel):
    def __init__(self, failure: str) -> None:
        super().__init__(custom_output_text="done", call_tools=[])
        self.failure = failure
        self.entries = 0
        self.exits: list[BaseException | None] = []
        self.primary = AIError(ErrorCode.MODEL_CONFIG_INVALID)

    async def __aenter__(self) -> "_LifecycleModel":
        self.entries += 1
        if self.failure == "enter":
            raise self.primary
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exits.append(exc_value)
        if self.failure in {"exit", "request_and_exit"}:
            raise AIError(ErrorCode.INTERNAL_ERROR)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if self.failure in {"request", "request_and_exit"}:
            raise self.primary
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context,
        ) as response:
            yield response


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("enter", "exit", "request", "request_and_exit"))
async def test_model_client_request_capture_runtime_preserves_custom_model_lifecycle_errors(
    failure: str,
) -> None:
    model = _LifecycleModel(failure)
    registry = ModelRegistry()
    registry.register(_ModelBinding(model))
    async with Runtime.open(
        "default", models=registry, storage=RuntimeStorage.in_memory(), features=_agents(),
    ) as runtime:
        result = (await runtime.agents.get("default").run("hello", timeout_seconds=5)).result
    assert result.status is ExecutionStatus.FAILED
    expected = ErrorCode.INTERNAL_ERROR if failure == "exit" else ErrorCode.MODEL_CONFIG_INVALID
    assert result.error_code == expected.value
    assert model.entries == 1
    assert len(model.exits) == (0 if failure == "enter" else 1)
    if failure in {"request", "request_and_exit"}:
        assert model.exits[0] is model.primary


@pytest.mark.asyncio
async def test_model_client_request_capture_runtime_leaves_borrowed_client_open() -> None:
    async with _ObservedHttpClient(transport=httpx2.MockTransport(_response)) as client:
        model = OpenAIChatModel("gpt-4o", provider=OpenAIProvider(
            base_url="https://capture.invalid/v1", api_key="test", http_client=client,
        ))
        registry = ModelRegistry()
        registry.register(_ModelBinding(model))
        async with Runtime.open(
            "default", models=registry, storage=RuntimeStorage.in_memory(), features=_agents(),
        ) as runtime:
            result = (await runtime.agents.get("default").run("hello", timeout_seconds=5)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert not client.is_closed and client.close_calls == 0
    assert client.is_closed


@pytest.mark.asyncio
async def test_model_client_request_capture_compaction_reuses_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, clients = _model(monkeypatch, _response)
    journal = ModelRequestJournal(
        source_namespace="workspace", tenant_id="tenant", execution_id="execution", agent_run_id="run",
    )
    phases: list[str] = []
    compaction_requests = []

    async def record(ctx, fact, phase, model, response, error, messages, settings, parameters, streaming, source, client_requests):
        if phase == "completed":
            compaction_requests.extend(client_requests)

    async def observer(ctx, fact, phase, selected_model, response, error) -> None:
        assert selected_model is model and not clients[0].is_closed
        assert error is None
        phases.append(phase)

    history: list[ModelMessage] = []
    for index in range(20):
        history.extend((
            ModelRequest(parts=[UserPromptPart(f"user {index} " + "x" * 240)]),
            ModelResponse(parts=[TextPart(f"answer {index} " + "y" * 240)]),
        ))
    agent = Agent(model, capabilities=[CompactionCapability(
        300, limits=PromptLimits(), journal=journal, observer=observer, request_recorder=record,
    )])
    async with model:
        with ModelClientRequestRecorder() as recorder:
            result = await agent.run("continue", message_history=history)
        assert result.output == "done"
        assert len(clients) == 1 and not clients[0].is_closed
    assert phases == ["started", "completed"]
    assert len(recorder.requests) == 1
    assert len(compaction_requests) == 1
    assert compaction_requests[0].response is not None and compaction_requests[0].response.complete
    assert clients[0].is_closed and clients[0].close_calls == 1
