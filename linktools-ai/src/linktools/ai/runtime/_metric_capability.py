#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic-specific automatic Model metric producer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Protocol

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
)
from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    ContentFilterError,
    RunCancelled,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import FinalResultEvent, FinishReason, ModelMessage, ModelResponse, ModelResponseState, ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.usage import RequestUsage, UsageLimitExceeded

from ..capability import AgentContext
from ..core import ExecutionEventType, JsonValue
from ..errors import AIError, ErrorCode
from ..model import model_binding_error
from ..observe import MetricMeasurement, MetricRecorder, Observation
from ._budget import RunBudgetContext
from ._journal import ModelRequestFact, ModelRequestJournal, _await_request_handoff
from ._metrics import (
    _bind_metric_execution_context,
    _metric_correlation,
)

_logger = environ.get_logger("ai.runtime.model_metrics")

ModelRequestEventSink = Callable[[ExecutionEventType, JsonValue], Awaitable[None]]


class ModelInteractionRecorder(Protocol):
    async def commit_history_boundary(self) -> None: ...

    def begin_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None = None,
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> None: ...

    def prepare_model_interaction(
        self,
        fact: ModelRequestFact,
        model: Model,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        model_id: str | None = None,
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> None: ...

    def finish_model_interaction(
        self,
        fact: ModelRequestFact,
        *,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        duration_ns: int,
        usage: object | None,
    ) -> None: ...

    async def record_model_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
        include_observation: bool,
    ) -> None: ...


class _ProviderStreamedResponse(StreamedResponse):
    """Observe errors at provider pulls without intercepting consumer failures."""

    def __init__(self, response: StreamedResponse, failed: Callable[[Exception], None]) -> None:
        self._response = response
        self._failed = failed
        self._iterator: AsyncIterator[ModelResponseStreamEvent] | None = None
        super().__init__(model_request_parameters=response.model_request_parameters)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._iterator is None:
            self._iterator = self._get_event_iterator()
        return self._iterator

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        try:
            iterator = self._response.__aiter__()
        except Exception as error:
            self._failed(error)
            raise
        while True:
            try:
                event = await iterator.__anext__()
            except StopAsyncIteration:
                return
            except Exception as error:
                self._failed(error)
                raise
            yield event

    def get(self) -> ModelResponse:
        return self._response.get()

    @property
    def usage(self) -> RequestUsage:
        return self._response.usage

    @property
    def model_name(self) -> str:
        return self._response.model_name

    @property
    def provider_name(self) -> str | None:
        return self._response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._response.timestamp

    @property
    def final_result_event(self) -> FinalResultEvent | None:
        return self._response.final_result_event

    @final_result_event.setter
    def final_result_event(self, value: FinalResultEvent | None) -> None:
        self._response.final_result_event = value

    @property
    def provider_response_id(self) -> str | None:
        return self._response.provider_response_id

    @provider_response_id.setter
    def provider_response_id(self, value: str | None) -> None:
        self._response.provider_response_id = value

    @property
    def provider_details(self) -> dict[str, Any] | None:
        return self._response.provider_details

    @provider_details.setter
    def provider_details(self, value: dict[str, Any] | None) -> None:
        self._response.provider_details = value

    @property
    def finish_reason(self) -> FinishReason | None:
        return self._response.finish_reason

    @finish_reason.setter
    def finish_reason(self, value: FinishReason | None) -> None:
        self._response.finish_reason = value

    @property
    def state(self) -> ModelResponseState:
        return self._response.state

    @state.setter
    def state(self, value: ModelResponseState) -> None:
        self._response.state = value

    @property
    def metadata(self) -> dict[str, Any] | None:
        return self._response.metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        self._response.metadata = value

    @property
    def cancelled(self) -> bool:
        return self._response.cancelled

    async def cancel(self) -> None:
        await self._response.cancel()

    async def close_stream(self) -> None:
        await self._response.close_stream()

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        return self._response.get_stream_cancel_errors()

    def time_to_first_chunk(self, request_start: float) -> float | None:
        return self._response.time_to_first_chunk(request_start)


class _PreparedRequestModel(WrapperModel):
    """Freeze provider input after SDK preparation and before provider execution."""

    def __init__(
        self,
        wrapped: Model,
        prepare: Callable[
            [Sequence[ModelMessage], ModelSettings | None, ModelRequestParameters, bool],
            Awaitable[None],
        ],
        failed: Callable[[Exception], None],
    ) -> None:
        super().__init__(wrapped)
        self._prepare = prepare
        self._failed = failed

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        await self._prepare(messages, model_settings, model_request_parameters, False)
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: PydanticRunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        await self._prepare(messages, model_settings, model_request_parameters, True)
        try:
            manager = self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context,
            )
            response = await manager.__aenter__()
        except Exception as error:
            self._failed(error)
            raise
        try:
            yield _ProviderStreamedResponse(response, self._failed)
        except BaseException as error:
            # Context managers receive consumer failures too. Forward their
            # suppression semantics without treating those failures as provider errors.
            if not await manager.__aexit__(type(error), error, error.__traceback__):
                raise
        else:
            try:
                await manager.__aexit__(None, None, None)
            except Exception as error:
                self._failed(error)
                raise


class ModelObservationCapability(AbstractCapability[AgentContext[object]]):
    """Observe actual logical model handler invocations without changing them."""

    def __init__(
        self,
        recorder: MetricRecorder | None,
        *,
        source_namespace: str,
        tenant_id: str,
        execution_id: str,
        agent_run_seq: int = 1,
        session_id: str | None,
        agent_run_id: str,
        agent_id: str,
        journal: ModelRequestJournal | None = None,
        interaction_recorder: ModelInteractionRecorder | None = None,
        event_sink: ModelRequestEventSink | None = None,
        budget: RunBudgetContext | None = None,
    ) -> None:
        self.id = "linktools.ai.model-observation"
        self._recorder = recorder
        self._source_namespace = source_namespace
        self._tenant_id = tenant_id
        self._execution_id = execution_id
        self._agent_run_seq = agent_run_seq
        self._session_id = session_id
        self._agent_run_id = agent_run_id
        self._agent_id = agent_id
        self._journal = journal or ModelRequestJournal(
            source_namespace=source_namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
            agent_run_id=agent_run_id,
        )
        self._interaction_recorder = interaction_recorder
        self._event_sink = event_sink
        self._budget = budget
        self._prepared_models: dict[int, Model] = {}
        self._provider_errors: dict[int, Exception] = {}

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
        model = request_context.model
        request_sequence: int | None = None

        async def prepare(
            messages: Sequence[ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
            streaming: bool,
        ) -> None:
            nonlocal request_sequence
            fact = self._journal.latest_for_step(ctx.run_step)
            if fact is None or fact.status is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request_sequence = fact.model_request_seq
            if fact.model_request_seq in self._prepared_models:
                return
            self._stage_request(
                fact,
                messages=messages,
                model_settings=settings,
                parameters=parameters,
                streaming=streaming,
                model=model,
                model_id=request_context.model_id,
                prepared=True,
            )
            self._prepared_models[fact.model_request_seq] = model
            if self._interaction_recorder is not None:
                await self._interaction_recorder.commit_history_boundary()

        def provider_failed(error: Exception) -> None:
            if request_sequence is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._provider_errors.setdefault(request_sequence, error)

        request_context.model = _PreparedRequestModel(model, prepare, provider_failed)
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
        fact = self._journal.begin(
            ctx.run_step,
            purpose="agent",
            output_retry_index=None if ctx.retry <= 0 else ctx.retry,
        )
        model_request_seq = fact.model_request_seq
        try:
            self._stage_request(fact, request_context)
            try:
                await self._record_request_event(fact, phase="started")
                await self._publish_request_event(fact, phase="started")
            except asyncio.CancelledError:
                await self._complete_request(
                    fact,
                    run_context,
                    selected_model,
                    status="CANCELLED",
                    response=None,
                    error_code=None,
                    usage=None,
                    phase="cancelled",
                )
                raise
            except RunCancelled as error:
                interrupted = await self._complete_request(
                    fact,
                    run_context,
                    selected_model,
                    status="CANCELLED",
                    response=None,
                    error_code=_model_error_code(error),
                    usage=None,
                    phase="cancelled",
                )
                if interrupted:
                    raise asyncio.CancelledError from error
                raise
            except Exception as error:
                interrupted = await self._complete_request(
                    fact,
                    run_context,
                    selected_model,
                    status="FAILED",
                    response=None,
                    error_code=_model_error_code(error),
                    usage=None,
                    phase="failed",
                )
                if interrupted:
                    raise asyncio.CancelledError from error
                if isinstance(error, AIError):
                    raise
                raise AIError(
                    ErrorCode.INTERNAL_ERROR,
                    safe_details={"phase": "model_request_started_event"},
                ) from error

            try:
                response = (
                    await handler(request_context) if self._budget is None else
                    await self._budget.run_model(
                        fact.observation_id, lambda: handler(request_context),
                    )
                )
            except asyncio.CancelledError:
                provider_error = self._provider_errors.get(model_request_seq)
                failed = provider_error is not None and not isinstance(provider_error, RunCancelled)
                await self._complete_request(
                    fact,
                    run_context,
                    selected_model,
                    status="FAILED" if failed else "CANCELLED",
                    response=None,
                    error_code=None if provider_error is None else _model_error_code(provider_error),
                    usage=None,
                    phase="failed" if failed else "cancelled",
                )
                raise
            except RunCancelled as error:
                interrupted = await self._complete_request(
                    fact,
                    run_context,
                    selected_model,
                    status="CANCELLED",
                    response=None,
                    error_code=_model_error_code(error),
                    usage=None,
                    phase="cancelled",
                )
                if interrupted:
                    raise asyncio.CancelledError from error
                raise
            except Exception as error:
                interrupted = await self._complete_request(
                    fact,
                    run_context,
                    selected_model,
                    status="FAILED",
                    response=None,
                    error_code=_model_error_code(error),
                    usage=None,
                    phase="failed",
                )
                if interrupted:
                    raise asyncio.CancelledError from error
                raise

            interrupted = await self._complete_request(
                fact,
                run_context,
                selected_model,
                status="SUCCEEDED",
                response=response,
                error_code=None,
                usage=response.usage,
                phase="completed",
            )
            if interrupted:
                raise asyncio.CancelledError
            return response
        finally:
            self._prepared_models.pop(model_request_seq, None)
            self._provider_errors.pop(model_request_seq, None)
            self._journal.consume(model_request_seq)

    async def after_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx, request_context
        return response

    async def on_model_request_error(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        del ctx, request_context
        raise error

    async def record_external_model_request(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        fact: ModelRequestFact,
        phase: str,
        model: Model,
        response: ModelResponse | None,
        error: BaseException | None,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool = False,
        source_messages: Sequence[ModelMessage] | None = None,
    ) -> None:
        if phase == "started":
            self._stage_request(
                fact,
                messages=messages,
                model_settings=model_settings,
                parameters=parameters,
                streaming=streaming,
                model=model,
                model_id=str(getattr(model, "model_id", "")) or None,
                source_messages=source_messages,
            )
            preparing = False
            try:
                await self._record_request_event(fact, phase="started")
                await self._publish_request_event(fact, phase="started")
                preparing = True
                self._stage_request(
                    fact,
                    messages=messages,
                    model_settings=model_settings,
                    parameters=parameters,
                    streaming=streaming,
                    model=model,
                    model_id=str(getattr(model, "model_id", "")) or None,
                    source_messages=source_messages,
                    prepared=True,
                )
                if self._interaction_recorder is not None:
                    await self._interaction_recorder.commit_history_boundary()
            except asyncio.CancelledError:
                await self._complete_request(
                    fact,
                    ctx.deps,
                    model,
                    status="CANCELLED",
                    response=None,
                    error_code=None,
                    usage=None,
                    phase="cancelled",
                )
                raise
            except RunCancelled as error:
                interrupted = await self._complete_request(
                    fact,
                    ctx.deps,
                    model,
                    status="CANCELLED",
                    response=None,
                    error_code=_model_error_code(error),
                    usage=None,
                    phase="cancelled",
                )
                if interrupted:
                    raise asyncio.CancelledError from error
                raise
            except Exception as error:
                interrupted = await self._complete_request(
                    fact,
                    ctx.deps,
                    model,
                    status="FAILED",
                    response=None,
                    error_code=_model_error_code(error),
                    usage=None,
                    phase="failed",
                )
                if interrupted:
                    raise asyncio.CancelledError from error
                if isinstance(error, AIError) or preparing:
                    raise
                raise AIError(
                    ErrorCode.INTERNAL_ERROR,
                    safe_details={"phase": "model_request_started_event"},
                ) from error
            return
        if phase not in {"completed", "failed", "cancelled"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if phase == "completed":
            if response is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            status = "SUCCEEDED"
            error_code = None
            usage = response.usage
            measurements = _provider_usage_measurements(response)
        else:
            exception = error if isinstance(error, Exception) else None
            error_code = None if exception is None else _model_error_code(exception)
            status = "CANCELLED" if phase == "cancelled" else "FAILED"
            usage = None
            measurements = ()
        self._finish_request(
            fact,
            model,
            response if phase == "completed" else None,
            status,
            error_code,
            usage,
        )
        self._record_model(
            ctx.deps,
            fact,
            model=model,
            response=response if phase == "completed" else None,
            status=status,
            error_code=error_code,
            measurements=measurements,
        )
        await self._record_request_event(
            fact,
            phase=phase,
            response=response if phase == "completed" else None,
            error_code=error_code,
        )
        await self._publish_request_event(
            fact,
            phase=phase,
            response=response if phase == "completed" else None,
            error_code=error_code,
        )

    async def _record_request_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
    ) -> None:
        recorder = self._interaction_recorder
        if recorder is None:
            return
        await recorder.record_model_event(
            fact,
            phase=phase,
            response=response,
            error_code=error_code,
            include_observation=self._recorder is not None,
        )

    async def _publish_request_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
    ) -> None:
        if self._event_sink is None:
            return
        if phase == "started":
            event_type = ExecutionEventType.MODEL_REQUEST_STARTED
            status = "RUNNING"
        elif phase in {"completed", "failed", "cancelled"}:
            event_type = ExecutionEventType.MODEL_REQUEST_FINISHED
            status = fact.status
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload: dict[str, JsonValue] = {
            "execution_id": self._execution_id,
            "agent_run_seq": self._agent_run_seq,
            "model_request_seq": fact.model_request_seq,
            "step_index": fact.step_index,
            "purpose": fact.purpose,
            "output_retry_index": fact.output_retry_index,
            "status": status,
            "started_at": fact.started_at.isoformat(),
            "finished_at": (
                None if fact.finished_at is None else fact.finished_at.isoformat()
            ),
            "duration_ns": fact.duration_ns,
            "error_code": error_code,
            "usage": _event_usage(response),
        }
        try:
            await self._event_sink(event_type, payload)
        except (asyncio.CancelledError, RunCancelled, AIError):
            raise
        except Exception as error:
            raise AIError(
                ErrorCode.INTERNAL_ERROR,
                safe_details={"phase": "model_request_event_publication"},
            ) from error

    async def _complete_request(
        self,
        fact: ModelRequestFact,
        run_context: AgentContext[object] | None,
        model: Model,
        *,
        status: str,
        response: ModelResponse | None,
        error_code: str | None,
        usage: object | None,
        phase: str,
    ) -> bool:
        model = self._prepared_models.get(fact.model_request_seq, model)
        finished = self._journal.finish(fact.model_request_seq, status=status)
        self._finish_request(
            finished,
            model,
            response,
            status,
            error_code,
            usage,
        )
        measurements = (
            _provider_usage_measurements(response)
            if status == "SUCCEEDED" and response is not None
            else ()
        )
        self._record_model(
            run_context,
            finished,
            model=model,
            response=response,
            status=status,
            error_code=error_code,
            measurements=measurements,
        )

        async def handoff() -> None:
            await self._record_request_event(
                finished,
                phase=phase,
                response=response,
                error_code=error_code,
            )
            await self._publish_request_event(
                finished,
                phase=phase,
                response=response,
                error_code=error_code,
            )

        return await _await_request_handoff(handoff())

    def _stage_request(
        self,
        fact: ModelRequestFact,
        request_context: ModelRequestContext | None = None,
        *,
        messages: Sequence[ModelMessage] | None = None,
        model_settings: ModelSettings | None = None,
        parameters: ModelRequestParameters | None = None,
        streaming: bool | None = None,
        model: Model | None = None,
        model_id: str | None = None,
        source_messages: Sequence[ModelMessage] | None = None,
        prepared: bool = False,
    ) -> None:
        recorder = self._interaction_recorder
        if recorder is None:
            return
        if request_context is not None:
            messages = request_context.messages
            model_settings = request_context.model_settings
            parameters = request_context.model_request_parameters
            streaming = bool(getattr(request_context, "streaming", False))
            model = request_context.model
            model_id = request_context.model_id
        if messages is None or parameters is None or model is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        stage = (
            recorder.prepare_model_interaction
            if prepared else recorder.begin_model_interaction
        )
        stage(
            fact,
            model,
            messages,
            model_settings,
            parameters,
            bool(streaming),
            model_id,
            source_messages,
        )

    def _finish_request(
        self,
        fact: ModelRequestFact,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        usage: object | None,
    ) -> None:
        recorder = self._interaction_recorder
        if recorder is None:
            return
        recorder.finish_model_interaction(
            fact,
            model=model,
            response=response,
            status=status,
            error_code=error_code,
            duration_ns=fact.duration_ns or 0,
            usage=usage,
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
                    agent_run_id=self._agent_run_id,
                    model_request_seq=fact.model_request_seq,
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


def _event_usage(response: ModelResponse | None) -> JsonValue | None:
    if response is None:
        return None
    usage = response.usage
    return {
        "model_requests": 1,
        "tool_calls": 0,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


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
    provider_error = model_binding_error(error)
    if provider_error is not None:
        return provider_error.code.value
    if isinstance(error, UnexpectedModelBehavior):
        return ErrorCode.MODEL_RESPONSE_INVALID.value
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED.value
    if isinstance(error, UserError):
        return ErrorCode.INTERNAL_ERROR.value
    return ErrorCode.INTERNAL_ERROR.value


__all__: list[str] = []
