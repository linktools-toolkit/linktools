#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Actual model attempts keep unique metric identities across re-execution."""

import pytest
from linktools.ai.observe import Observation
from linktools.ai.runtime._metric_capability import _RuntimeModelMetricCapability
from pydantic_ai.messages import ModelResponse


class _Recorder:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def try_record(self, observation: Observation) -> bool:
        self.observations.append(observation)
        return True


@pytest.mark.asyncio
async def test_actual_model_attempts_never_reuse_observation_identity() -> None:
    recorder = _Recorder()
    capability = _RuntimeModelMetricCapability(
        recorder,
        source_namespace="workspace",
        tenant_id="tenant",
        execution_id="execution",
        session_id=None,
        step_run_id="durable-step-run",
        agent_id="agent",
        provider="test",
        model_identity="test:model",
        route_id="default",
    )

    async def handler(_request: object) -> ModelResponse:
        return ModelResponse(parts=())

    await capability.wrap_model_request(
        None,
        request_context=object(),  # type: ignore[arg-type]
        handler=handler,  # type: ignore[arg-type]
    )
    await capability.wrap_model_request(
        None,
        request_context=object(),  # type: ignore[arg-type]
        handler=handler,  # type: ignore[arg-type]
    )

    assert len(recorder.observations) == 2
    assert recorder.observations[0].observation_id != recorder.observations[1].observation_id
