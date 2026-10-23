#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transport, fresh fallback and output correction own distinct budgets."""

import asyncio
import json
from collections.abc import AsyncIterator

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError, ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.capability import TraceCapability
from linktools.ai.core import ExecutionStatus, TaskStatus
from linktools.ai.model import FallbackBinding, ModelBinding, create_model
from linktools.ai.model._binding import _RetryingModel
from linktools.ai.runtime import AgentTaskInput, Runtime, RuntimeFeatures, RuntimeStorage
from linktools.ai.task import TaskGraph, TaskNode

from ._runtime_test_helpers import _Models, _UsageFunctionModel


def _offline_failures(monkeypatch: pytest.MonkeyPatch, protocol: str, sent: list[str]) -> None:
    module = "linktools.ai.model._anthropic" if protocol == "anthropic" else "linktools.ai.model._openai"

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content)["model"])
        return httpx2.Response(503, json={
            "type": "error", "error": {"type": "overloaded_error", "message": "synthetic unavailable"},
        })

    def client(**kwargs: object) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handle), **kwargs)

    monkeypatch.setattr(module + ".DefaultAsyncHttpxClient", client)


def _features(binding: ModelBinding, *, output_retries: int = 0) -> RuntimeFeatures:
    features = RuntimeFeatures("fallback-retry-budget")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=(), output_retries=output_retries)
    features.model("default", binding)
    features.capability(TraceCapability(), scope="runtime")
    return features


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ("openai", "openai-responses", "anthropic"))
@pytest.mark.parametrize("runtime_request", (False, True))
@pytest.mark.parametrize("retries", (0, 2))
async def test_fresh_fallback_request_uses_each_candidate_budget_once(
    monkeypatch: pytest.MonkeyPatch, protocol: str, runtime_request: bool, retries: int,
) -> None:
    sent: list[str] = []
    _offline_failures(monkeypatch, protocol, sent)
    leaves = [create_model(
        protocol=protocol, model=f"candidate-{index}", api_key="synthetic",
        base_url="https://offline.invalid/v1", max_retries=retries, retry_delay=0,
    ) for index in range(3)]
    for leaf in leaves:
        async with leaf.materialize() as model:
            assert model.wrapped.provider.client.max_retries == 0
    binding = FallbackBinding(*leaves)
    if runtime_request:
        async with Runtime.open(
            "retry-budget", features=_features(binding), storage=RuntimeStorage.in_memory(),
        ) as runtime:
            execution = await runtime.agents.get().start("synthetic")
            result = (await execution.wait(timeout_seconds=15)).result
            assert result.status is ExecutionStatus.FAILED
            assert result.error_code == "MODEL_API_ERROR"
            assert len((await execution.model_interactions()).items) == 1
    else:
        async with binding.materialize() as model:
            with pytest.raises(FallbackExceptionGroup) as failed:
                await model.request([], None, ModelRequestParameters())
            assert len(failed.value.exceptions) == 3
            assert all(isinstance(error, ModelHTTPError) and error.status_code == 503
                       for error in failed.value.exceptions)
    assert sent == [f"candidate-{index}" for index in range(3) for _ in range(retries + 1)]


@pytest.mark.asyncio
async def test_nested_fallback_does_not_reset_candidate_transport_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[str] = []
    _offline_failures(monkeypatch, "openai", sent)
    leaves = [create_model(
        protocol="openai", model=f"candidate-{index}", api_key="synthetic",
        base_url="https://offline.invalid/v1", max_retries=2, retry_delay=0,
    ) for index in range(3)]
    binding = FallbackBinding(
        FallbackBinding(*leaves[:2]), leaves[2], fallback_on=(ModelAPIError, FallbackExceptionGroup),
    )
    async with Runtime.open(
        "retry-budget", features=_features(binding), storage=RuntimeStorage.in_memory(),
    ) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        assert (await execution.wait(timeout_seconds=15)).result.status is ExecutionStatus.FAILED
    assert sent == [f"candidate-{index}" for index in range(3) for _ in range(3)]


@pytest.mark.asyncio
async def test_cancellation_during_candidate_backoff_stops_retry_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    waiting = asyncio.Event()
    calls: list[str] = []

    async def backoff(self: _RetryingModel, error: ModelAPIError, backoff: float) -> float:
        del self, error, backoff
        waiting.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled backoff must not complete")

    monkeypatch.setattr(_RetryingModel, "_wait_for_retry", backoff)

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls.append("primary")
        raise ModelHTTPError(503, "primary")

    def secondary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls.append("secondary")
        return ModelResponse([TextPart("must not run")])

    binding = FallbackBinding(
        _Models(_RetryingModel(_UsageFunctionModel(primary), 2, 0)), _Models(_UsageFunctionModel(secondary)),
    )
    async with Runtime.open(
        "retry-budget", features=_features(binding), storage=RuntimeStorage.in_memory(),
    ) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        await asyncio.wait_for(waiting.wait(), 5)
        await execution.cancel()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
    assert calls == ["primary"]


@pytest.mark.asyncio
async def test_committed_stream_error_does_not_retry_or_fall_back() -> None:
    calls: list[str] = []

    async def primary(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        calls.append("primary")
        yield "committed text "
        raise ModelHTTPError(503, "primary")

    async def secondary(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        calls.append("secondary")
        yield "must not run"

    binding = FallbackBinding(
        _Models(_RetryingModel(FunctionModel(stream_function=primary), 2, 0)),
        _Models(_RetryingModel(FunctionModel(stream_function=secondary), 2, 0)),
    )
    async with Runtime.open(
        "retry-budget", features=_features(binding), storage=RuntimeStorage.in_memory(),
    ) as runtime:
        execution = await runtime.agents.get().start("synthetic")
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.FAILED
        assert len((await execution.model_interactions()).items) == 1
    assert calls == ["primary"]


class _Count(BaseModel):
    count: int


@pytest.mark.asyncio
@pytest.mark.parametrize("as_task", (False, True))
async def test_output_correction_has_its_own_bounded_requests_without_restarting_execution(as_task: bool) -> None:
    calls: list[str] = []

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls.append("primary")
        raise ModelHTTPError(503, "primary")

    def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        calls.append("secondary")
        return ModelResponse([ToolCallPart(info.output_tools[0].name, {"count": "invalid"})])

    binding = FallbackBinding(
        _Models(_RetryingModel(_UsageFunctionModel(primary), 2, 0)),
        _Models(_RetryingModel(_UsageFunctionModel(invalid), 2, 0)),
    )
    async with Runtime.open(
        "retry-budget", features=_features(binding, output_retries=2), storage=RuntimeStorage.in_memory(),
    ) as runtime:
        if as_task:
            definition = runtime.tasks.from_agent("one-task", runtime.agents.get())
            run = await runtime.tasks.bind(definition).start(TaskGraph("probe", (
                TaskNode("one", task=definition, input=AgentTaskInput("synthetic"), output_type=_Count),
            )), idempotency_key="one")
            graph = (await run.wait(timeout_seconds=15, include_content=True)).result
            assert graph.status is TaskStatus.FAILED
            executions = [item.execution_id for item in graph.node_states if item.execution_id]
            assert len(executions) == 1
            interactions = (await runtime.executions.model_interactions(
                executions[0], principal=runtime.default_principal,
            )).items
        else:
            execution = await runtime.agents.get().start("synthetic", output=_Count)
            result = (await execution.wait(timeout_seconds=15)).result
            assert result.status is ExecutionStatus.FAILED
            interactions = (await execution.model_interactions()).items
        assert len(interactions) == 3
        assert [item.output_retry_index for item in interactions] == [None, 1, 2]
    assert calls == (["primary"] * 3 + ["secondary"]) * 3
