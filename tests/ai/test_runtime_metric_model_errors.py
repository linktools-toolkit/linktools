#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model metric error classification and cancellation semantics."""

import asyncio

import httpx
import pytest
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import model_binding_error
from linktools.ai.observe import Observation
from linktools.ai.runtime._agent_executor import _execution_error
from linktools.ai.runtime._metric_capability import (
    ModelObservationCapability,
    _model_error_code,
)
from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, RunCancelled
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    async def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_error", ("pydantic", "openai"))
@pytest.mark.parametrize(
    ("status_code", "expected"),
    (
        (408, ErrorCode.MODEL_TIMEOUT),
        (429, ErrorCode.MODEL_RATE_LIMITED),
        (503, ErrorCode.MODEL_UNAVAILABLE),
        (400, ErrorCode.MODEL_REQUEST_REJECTED),
        (302, ErrorCode.MODEL_API_ERROR),
    ),
)
async def test_provider_http_failures_keep_execution_and_observation_codes(
    provider_error: str,
    status_code: int,
    expected: ErrorCode,
) -> None:
    if provider_error == "pydantic":
        error = ModelHTTPError(
            status_code,
            "model",
            body={"secret": "provider response"},
            headers={"Retry-After": "3"},
        )
    else:
        error = APIStatusError(
            "provider secret",
            response=httpx.Response(
                status_code,
                request=httpx.Request("POST", "https://provider.invalid"),
                headers={"Retry-After": "3"},
            ),
            body={"secret": "provider response"},
        )
    observation = await _observe_failure(error)
    projected = model_binding_error(error)
    assert projected is not None
    execution = _execution_error(
        error, usage_limits=UsageLimits(), run_usage=RunUsage(),
    )
    assert projected.code is execution.code is expected
    assert projected.retryable == execution.retryable
    assert projected.safe_details == execution.safe_details
    assert observation.status == "FAILED"
    assert observation.error_code == expected.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (
            APITimeoutError(request=httpx.Request("POST", "https://provider.invalid")),
            ErrorCode.MODEL_TIMEOUT,
        ),
        (
            APIConnectionError(request=httpx.Request("POST", "https://provider.invalid")),
            ErrorCode.MODEL_UNAVAILABLE,
        ),
        (
            APIError(
                "provider secret",
                request=httpx.Request("POST", "https://provider.invalid"),
                body={"secret": "provider response"},
            ),
            ErrorCode.MODEL_API_ERROR,
        ),
        (ModelAPIError("model", "provider secret"), ErrorCode.MODEL_API_ERROR),
    ),
)
async def test_provider_transport_failures_keep_execution_and_observation_codes(
    error: Exception,
    expected: ErrorCode,
) -> None:
    observation = await _observe_failure(error)
    execution = _execution_error(
        error, usage_limits=UsageLimits(), run_usage=RunUsage(),
    )
    assert execution.code is expected
    assert observation.status == "FAILED"
    assert observation.error_code == expected.value


def test_model_metric_preserves_runtime_ai_error_code() -> None:
    assert (
        _model_error_code(AIError(ErrorCode.STORAGE_UNAVAILABLE))
        == ErrorCode.STORAGE_UNAVAILABLE.value
    )


def test_model_metric_unknown_error_matches_runtime_internal_error() -> None:
    assert _model_error_code(RuntimeError("provider wrapper bug")) == ErrorCode.INTERNAL_ERROR.value


@pytest.mark.asyncio
async def test_model_cancellation_records_cancelled_observation() -> None:
    observation = await _observe_failure(RunCancelled("cancelled by application"))
    assert observation.status == "CANCELLED"
    assert observation.error_code == ErrorCode.EXECUTION_CANCELLED.value


async def _observe_failure(error: Exception) -> Observation:
    recorder = _Recorder()
    capability = ModelObservationCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        agent_run_id="agent-run",
        agent_id="agent",
    )

    async def handler(_request: object) -> object:
        raise error

    model = TestModel()
    context = RunContext(
        deps=type("Deps", (), {"correlation": {}})(),
        model=model,
        usage=RunUsage(),
        run_id="run",
        run_step=1,
    )
    request_context = ModelRequestContext(
        model=model,
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    with pytest.raises(type(error)) as captured:
        await capability.wrap_model_request(
            context,
            request_context=request_context,
            handler=handler,  # type: ignore[arg-type]
        )

    assert captured.value is error
    assert len(recorder.observations) == 1
    return recorder.observations[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("original_error", (None, RuntimeError("model failed")))
async def test_model_metric_cancellation_follows_completed_request_handoff(
    original_error: Exception | None,
) -> None:
    entered = asyncio.Event()
    phases: list[str] = []
    observations: list[Observation] = []

    class BlockingRecorder(_Recorder):
        async def try_record(self, observation: Observation) -> bool:
            observations.append(observation)
            entered.set()
            await asyncio.Event().wait()
            return True

    capability = ModelObservationCapability(
        BlockingRecorder(), source_namespace="workspace", tenant_id="tenant",
        execution_id="execution", session_id=None, agent_run_id="agent-run",
        agent_id="agent",
    )

    async def record_event(fact: object, *, phase: str, **kwargs: object) -> None:
        phases.append("durable:" + phase)

    async def publish_event(fact: object, *, phase: str, **kwargs: object) -> None:
        phases.append("published:" + phase)

    capability._record_request_event = record_event
    capability._publish_request_event = publish_event
    model = TestModel()
    context = RunContext(
        deps=type("Deps", (), {"correlation": {}})(), model=model,
        usage=RunUsage(), run_id="run", run_step=1,
    )
    request = ModelRequestContext(
        model=model, messages=[], model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    async def handler(_request: object) -> ModelResponse:
        if original_error is not None:
            raise original_error
        return ModelResponse(parts=())

    task = asyncio.create_task(capability.wrap_model_request(
        context, request_context=request, handler=handler,
    ))
    await asyncio.wait_for(entered.wait(), timeout=1)
    terminal_phase = "completed" if original_error is None else "failed"
    assert phases[-2:] == ["durable:" + terminal_phase, "published:" + terminal_phase]
    assert observations[0].status == ("SUCCEEDED" if original_error is None else "FAILED")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert phases[-2:] == ["durable:" + terminal_phase, "published:" + terminal_phase]
