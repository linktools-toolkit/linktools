#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic-specific automatic Model metric producer."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from linktools.core import environ
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIError as OpenAIAPIError,
    APIStatusError as OpenAIAPIStatusError,
    APITimeoutError as OpenAIAPITimeoutError,
)
from pydantic import ValidationError
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
)
from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    ContentFilterError,
    ModelAPIError,
    ModelHTTPError,
    RunCancelled,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.usage import UsageLimitExceeded

from ..capability import AgentContext
from ..errors import AIError, ErrorCode
from ..observe import MetricMeasurement, MetricRecorder, Observation
from ._metrics import (
    _bind_metric_execution_context,
    _metric_correlation,
)
from ._journal import ModelRequestFact, ModelRequestJournal

_logger = environ.get_logger("ai.runtime.model_metrics")


class RuntimeModelObservationCapability(AbstractCapability[AgentContext[object]]):
    """Observe actual logical model handler invocations without changing them."""

    def __init__(
        self,
        recorder: MetricRecorder | None,
        *,
        source_namespace: str,
        tenant_id: str,
        execution_id: str,
        session_id: str | None,
        step_run_id: str,
        agent_id: str,
        journal: ModelRequestJournal | None = None,
    ) -> None:
        self.id = "linktools.ai.model-observation"
        self._recorder = recorder
        self._source_namespace = source_namespace
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        self._session_id = session_id
        self._step_run_id = step_run_id
        self._agent_id = agent_id
        self._journal = journal or ModelRequestJournal(
            source_namespace=source_namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            step_run_id=step_run_id,
        )

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def before_run(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
    ) -> None:
        if self._recorder is None:
            return
        _bind_metric_execution_context(
            self._recorder,
            self._execution_id,
            ctx.deps.correlation,
        )

    async def before_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self._journal.begin(
            ctx.run_step,
            purpose="agent",
            output_retry_index=None if ctx.retry <= 0 else ctx.retry,
        )
        return request_context

    async def wrap_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        selected_model = request_context.model
        run_context = ctx.deps
        fact = self._journal.latest_for_step(ctx.run_step)
        if fact is None or fact.duration_ns is not None:
            if fact is not None:
                self._journal.consume(fact.request_sequence)
            fact = self._journal.begin(
                ctx.run_step,
                purpose="agent",
                output_retry_index=None if ctx.retry <= 0 else ctx.retry,
            )
        assert fact is not None
        request_sequence = fact.request_sequence
        try:
            response = await handler(request_context)
        except asyncio.CancelledError:
            fact = self._journal.finish(request_sequence, status="CANCELLED")
            self._record_model(
                run_context,
                fact,
                model=selected_model,
                response=None,
                status="CANCELLED",
                error_code=None,
                measurements=(),
            )
            raise
        except RunCancelled as error:
            fact = self._journal.finish(request_sequence, status="CANCELLED")
            self._record_model(
                run_context,
                fact,
                model=selected_model,
                response=None,
                status="CANCELLED",
                error_code=_model_error_code(error),
                measurements=(),
            )
            raise
        except Exception as error:
            fact = self._journal.finish(request_sequence, status="FAILED")
            self._record_model(
                run_context,
                fact,
                model=selected_model,
                response=None,
                status="FAILED",
                error_code=_model_error_code(error),
                measurements=(),
            )
            raise
        fact = self._journal.finish(request_sequence, status="SUCCEEDED")
        self._record_model(
            run_context,
            fact,
            model=selected_model,
            response=response,
            status="SUCCEEDED",
            error_code=None,
            measurements=_provider_usage_measurements(response),
        )
        return response

    async def after_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del request_context
        self._consume_current_request(ctx.run_step)
        return response

    async def on_model_request_error(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        del request_context
        self._consume_current_request(ctx.run_step)
        raise error

    async def on_run_error(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        error: BaseException,
    ) -> AgentRunResult[object]:
        self._consume_current_request(ctx.run_step)
        raise error

    def _consume_current_request(self, step_index: int) -> None:
        fact = self._journal.latest_for_step(step_index)
        if fact is not None:
            self._journal.consume(fact.request_sequence)

    async def record_external_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        fact: ModelRequestFact,
        phase: str,
        model: Model,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None:
        if phase == "started":
            return
        if phase not in {"completed", "failed", "cancelled"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if phase == "completed":
            if response is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._record_model(
                ctx.deps,
                fact,
                model=model,
                response=response,
                status="SUCCEEDED",
                error_code=None,
                measurements=_provider_usage_measurements(response),
            )
            return
        exception = error if isinstance(error, Exception) else None
        self._record_model(
            ctx.deps,
            fact,
            model=model,
            response=None,
            status="CANCELLED" if phase == "cancelled" else "FAILED",
            error_code=None if exception is None else _model_error_code(exception),
            measurements=(),
        )

    def _record_model(
        self,
        run_context: AgentContext[object] | None,
        fact: ModelRequestFact,
        *,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        measurements: tuple[MetricMeasurement, ...],
    ) -> None:
        if self._recorder is None:
            return
        try:
            selected_model_id = str(model.model_id)
            selected_model_system = str(model.system)
            selected_model_name = str(model.model_name)
            response_provider = (
                "" if response is None or response.provider_name is None
                else str(response.provider_name)
            )
            response_model = "" if response is None else str(response.model_name)
            dimensions = {
                "agent_id": self._agent_id,
                "provider": selected_model_system,
                "model_identity": selected_model_name,
                "route_id": selected_model_id,
                "selected_model_id": selected_model_id,
                "selected_model_system": selected_model_system,
                "selected_model_name": selected_model_name,
            }
            if response_provider:
                dimensions["response_provider_name"] = response_provider
            if response_model:
                dimensions["response_model_name"] = response_model
            observation = Observation(
                version=1,
                observation_id=fact.observation_id,
                kind="linktools.model.request",
                occurred_at=datetime.now(timezone.utc),
                source_namespace=self._source_namespace,
                tenant_id=self._tenant_id,
                status=status,
                error_code=error_code,
                correlation=_metric_correlation(
                    None if run_context is None else run_context.correlation,
                    execution_id=self._execution_id,
                    session_id=self._session_id,
                    step_run_id=self._step_run_id,
                    request_sequence=fact.request_sequence,
                    request_purpose=fact.purpose,
                    output_retry_index=fact.output_retry_index,
                ),
                dimensions=dimensions,
                measurements=(
                    MetricMeasurement("latency_ns", 1, fact.duration_ns or 0),
                    *measurements,
                ),
            )
            self._recorder.try_record(observation)
        except Exception:
            _logger.exception("model metric observation rejected")


def _provider_usage_measurements(
    response: ModelResponse,
) -> tuple[MetricMeasurement, ...]:
    usage = response.usage
    values = [
        ("input_tokens", usage.input_tokens),
        ("output_tokens", usage.output_tokens),
        ("cache_read_tokens", usage.cache_read_tokens),
        ("cache_write_tokens", usage.cache_write_tokens),
    ]
    measurements = [
        MetricMeasurement(name, 1, value)
        for name, value in values
        if value > 0
    ]
    if usage.input_tokens > 0 and usage.output_tokens > 0:
        measurements.insert(
            2,
            MetricMeasurement(
                "total_tokens",
                1,
                usage.input_tokens + usage.output_tokens,
            ),
        )
    return tuple(measurements)


def _model_error_code(error: Exception) -> str:
    if isinstance(error, AIError):
        return error.code.value
    if isinstance(error, UsageLimitExceeded):
        return ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
    if isinstance(error, RunCancelled):
        return ErrorCode.EXECUTION_CANCELLED.value
    if isinstance(error, ConcurrencyLimitExceeded):
        return ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED.value
    if isinstance(error, ContentFilterError):
        return ErrorCode.MODEL_CONTENT_FILTERED.value
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
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED.value
    if isinstance(error, UserError):
        return ErrorCode.INTERNAL_ERROR.value
    return ErrorCode.INTERNAL_ERROR.value


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
