#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic-specific automatic Model metric producer."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from time import monotonic_ns

from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIError as OpenAIAPIError,
    APIStatusError as OpenAIAPIStatusError,
    APITimeoutError as OpenAIAPITimeoutError,
)
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext

from ..errors import AIError, ErrorCode
from ..observe import MetricMeasurement, MetricRecorder, Observation


class _RuntimeModelMetricCapability(AbstractCapability[None]):
    """Observe actual model request attempts without mutating semantics."""

    def __init__(
        self,
        recorder: MetricRecorder,
        *,
        source_namespace: str,
        tenant_id: str,
        execution_id: str,
        session_id: str | None,
        step_run_id: str,
        agent_id: str,
        provider: str,
        model_identity: str,
        route_id: str,
    ) -> None:
        self._recorder = recorder
        self._source_namespace = source_namespace
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        self._session_id = session_id
        self._step_run_id = step_run_id
        self._agent_id = agent_id
        self._provider = provider
        self._model_identity = model_identity
        self._route_id = route_id

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        del ctx
        attempt_id = uuid.uuid4().hex
        started = monotonic_ns()
        try:
            response = await handler(request_context)
        except asyncio.CancelledError:
            self._record_model(
                attempt_id,
                started,
                status="CANCELLED",
                error_code=None,
                measurements=(),
            )
            raise
        except Exception as error:
            self._record_model(
                attempt_id,
                started,
                status="FAILED",
                error_code=_model_error_code(error),
                measurements=(),
            )
            raise
        self._record_model(
            attempt_id,
            started,
            status="SUCCEEDED",
            error_code=None,
            measurements=_provider_usage_measurements(response),
        )
        return response

    def _record_model(
        self,
        attempt_id: str,
        started: int,
        *,
        status: str,
        error_code: str | None,
        measurements: tuple[MetricMeasurement, ...],
    ) -> None:
        try:
            observation = Observation(
                version=1,
                observation_id=attempt_id,
                kind="linktools.model.request",
                occurred_at=datetime.now(timezone.utc),
                source_namespace=self._source_namespace,
                tenant_id=self._tenant_id,
                status=status,
                error_code=error_code,
                correlation=self._correlation(),
                dimensions={
                    "agent_id": self._agent_id,
                    "provider": self._provider,
                    "model_identity": self._model_identity,
                    "route_id": self._route_id,
                },
                measurements=(
                    MetricMeasurement("latency_ns", 1, monotonic_ns() - started),
                    *measurements,
                ),
            )
            self._recorder.try_record(observation)
        except Exception:
            return

    def _correlation(self) -> Mapping[str, str | int]:
        values: dict[str, str | int] = {
            "execution_id": self._execution_id,
            "step_run_id": self._step_run_id,
        }
        if self._session_id is not None:
            values["session_id"] = self._session_id
        return values


def _provider_usage_measurements(
    response: ModelResponse,
) -> tuple[MetricMeasurement, ...]:
    usage = response.usage
    return (
        MetricMeasurement("input_tokens", 1, usage.input_tokens),
        MetricMeasurement("output_tokens", 1, usage.output_tokens),
        MetricMeasurement("cache_read_tokens", 1, usage.cache_read_tokens),
        MetricMeasurement("cache_write_tokens", 1, usage.cache_write_tokens),
    )


def _model_error_code(error: Exception) -> str:
    if isinstance(error, AIError):
        return error.code.value
    if isinstance(error, ModelHTTPError):
        return _http_error_code(error.status_code).value
    if isinstance(error, OpenAIAPITimeoutError):
        return ErrorCode.MODEL_TIMEOUT.value
    if isinstance(error, OpenAIAPIConnectionError):
        return ErrorCode.MODEL_UNAVAILABLE.value
    if isinstance(error, OpenAIAPIStatusError):
        return _http_error_code(error.status_code).value
    if isinstance(error, (ModelAPIError, OpenAIAPIError)):
        return ErrorCode.MODEL_API_ERROR.value
    if isinstance(error, UnexpectedModelBehavior):
        return ErrorCode.MODEL_RESPONSE_INVALID.value
    return ErrorCode.MODEL_API_ERROR.value


def _http_error_code(status_code: int) -> ErrorCode:
    if status_code == 408:
        return ErrorCode.MODEL_TIMEOUT
    if status_code == 429:
        return ErrorCode.MODEL_RATE_LIMITED
    if status_code >= 500:
        return ErrorCode.MODEL_UNAVAILABLE
    if 400 <= status_code < 500:
        return ErrorCode.MODEL_REQUEST_REJECTED
    return ErrorCode.MODEL_API_ERROR


__all__: list[str] = []
