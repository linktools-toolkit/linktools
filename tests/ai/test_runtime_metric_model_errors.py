#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model metric error classification and cancellation semantics."""

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Observation
from linktools.ai.runtime._metric_capability import (
    RuntimeModelObservationCapability,
    _http_error_code,
    _model_error_code,
)
from pydantic_ai import RunContext
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


def test_model_metric_http_error_classification_is_canonical() -> None:
    assert _http_error_code(408) is ErrorCode.MODEL_TIMEOUT
    assert _http_error_code(429) is ErrorCode.MODEL_RATE_LIMITED
    assert _http_error_code(500) is ErrorCode.MODEL_UNAVAILABLE
    assert _http_error_code(503) is ErrorCode.MODEL_UNAVAILABLE
    assert _http_error_code(400) is ErrorCode.MODEL_REQUEST_REJECTED
    assert _http_error_code(499) is ErrorCode.MODEL_REQUEST_REJECTED
    assert _http_error_code(302) is ErrorCode.MODEL_API_ERROR


def test_model_metric_preserves_runtime_ai_error_code() -> None:
    assert (
        _model_error_code(AIError(ErrorCode.STORAGE_UNAVAILABLE))
        == ErrorCode.STORAGE_UNAVAILABLE.value
    )


def test_model_metric_unknown_error_matches_runtime_internal_error() -> None:
    assert _model_error_code(RuntimeError("provider wrapper bug")) == ErrorCode.INTERNAL_ERROR.value


@pytest.mark.asyncio
async def test_model_cancellation_records_cancelled_observation() -> None:
    recorder = _Recorder()
    capability = RuntimeModelObservationCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id="step-run",
        agent_id="agent",
    )

    async def handler(_request: object) -> object:
        raise RunCancelled("cancelled by application")

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

    with pytest.raises(RunCancelled):
        await capability.wrap_model_request(
            context,
            request_context=request_context,
            handler=handler,  # type: ignore[arg-type]
        )

    assert len(recorder.observations) == 1
    assert recorder.observations[0].status == "CANCELLED"
    assert recorder.observations[0].error_code == ErrorCode.EXECUTION_CANCELLED.value
