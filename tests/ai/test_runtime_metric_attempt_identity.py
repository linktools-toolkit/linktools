#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual model attempts keep unique metric identities across re-execution."""

import pytest
from linktools.ai.observe import Observation
from linktools.ai.runtime._metric_capability import RuntimeModelObservationCapability
from pydantic_ai import RunContext
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


@pytest.mark.asyncio
async def test_actual_model_attempts_never_reuse_observation_identity() -> None:
    recorder = _Recorder()
    capability = RuntimeModelObservationCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id="durable-step-run",
        agent_id="agent",
    )

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

    async def handler(_request: object) -> ModelResponse:
        return ModelResponse(parts=())

    await capability.wrap_model_request(
        context,
        request_context=request_context,
        handler=handler,  # type: ignore[arg-type]
    )
    await capability.wrap_model_request(
        context,
        request_context=request_context,
        handler=handler,  # type: ignore[arg-type]
    )

    assert len(recorder.observations) == 2
    assert recorder.observations[0].observation_id != recorder.observations[1].observation_id
