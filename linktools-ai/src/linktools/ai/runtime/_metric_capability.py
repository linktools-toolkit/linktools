#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic-specific automatic Model metric producer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from copy import copy
from datetime import datetime, timezone
from typing import Any, Protocol

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    WrapNodeRunHandler,
    CapabilityOrdering,
    WrapModelRequestHandler,
)
from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    FallbackExceptionGroup,
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
from pydantic_ai.usage import RequestUsage, RunUsage, UsageLimitExceeded

from ..capability import AgentContext
from ._trace_capture import record_http_bodies_enabled
from ..core import ExecutionEventType, JsonValue
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..model import (
    ModelClientRequest, ModelClientRequestRecorder, ModelStreamAttempt, model_binding_error,
    model_request_error as project_model_request_error,
)
from ..observe import MetricMeasurement, MetricRecorder, Observation
from ._budget import RunBudgetContext
from ..observe.journal import ModelRequestFact, ModelRequestJournal, await_request_handoff
from ..model.messages import project_model_response_attempts, model_response_usage, model_attempts_usage, model_attempts_details
from ..observe.service import (
    bind_metric_execution_context,
    metric_correlation,
)

_logger = environ.get_logger("ai.runtime.model_metrics")

ModelRequestEventSink = Callable[[ExecutionEventType, JsonValue], Awaitable[None]]
_StreamObserver = Callable[[str, BaseException | None], None]


class ModelInteractionRecorder(Protocol):
    def observe_stream_attempt(self, fact: ModelRequestFact, attempt: ModelStreamAttempt) -> None: ...
    def response_message_seq(self, model_request_seq: int) -> int | None: ...

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
        client_requests: tuple[ModelClientRequest, ...] | None = None,
        stream_attempts: tuple[ModelStreamAttempt, ...] | None = None,
    ) -> None: ...

    async def record_model_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
        usage: RequestUsage | RunUsage | None = None,
        include_observation: bool,
    ) -> None: ...


class _StreamDeadline(Protocol):
    def iterate(
        self, iterator: AsyncIterator[ModelResponseStreamEvent],
    ) -> AsyncIterator[ModelResponseStreamEvent]: ...


class _ProviderStreamedResponse(StreamedResponse):
    """Observe errors at provider pulls without intercepting consumer failures."""

    def __init__(
        self, response: StreamedResponse, failed: Callable[[Exception], None], observed: _StreamObserver,
        project_response: Callable[[ModelResponse], ModelResponse] | None = None,
    ) -> None:
        self._response = response
        self._project_response = project_response or (lambda response: response)
        self._failed = failed
        self._observed = observed
        super().__init__(model_request_parameters=response.model_request_parameters)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Native fallback replays its first event through this SDK-owned cache.
        if self._event_iterator is None:
            self._event_iterator = self._get_event_iterator()
        return self._event_iterator

    def _enforce_request_deadline(self, deadline: _StreamDeadline) -> None:
        # The inner stream already synthesizes SDK events; wrap pulls, not event processing.
        self._event_iterator = deadline.iterate(self.__aiter__())

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        try:
            iterator = self._response.__aiter__()
        except asyncio.CancelledError as error:
            self._observed("iteration_error", error)
            raise
        except Exception as error:
            self._observed("iteration_error", error)
            self._failed(error)
            raise
        while True:
            try:
                event = await iterator.__anext__()
            except StopAsyncIteration:
                self._observed("exhausted", None)
                return
            except asyncio.CancelledError as error:
                self._observed("iteration_error", error)
                raise
            except Exception as error:
                self._observed("iteration_error", error)
                self._failed(error)
                raise
            yield event

    def get(self) -> ModelResponse:
        try:
            return self._project_response(self._response.get())
        except Exception as error:
            self._failed(error)
            raise

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
        self._observed("cancel_requested", None)
        try:
            await self._response.cancel()
        except Exception as error:
            self._failed(error)
            raise

    async def close_stream(self) -> None:
        self._observed("close_started", None)
        try:
            await self._response.close_stream()
        except BaseException as error:
            self._observed("close_error", error)
            if isinstance(error, Exception):
                self._failed(error)
            raise
        finally:
            self._observed("close_finished", None)

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        return self._response.get_stream_cancel_errors()

    def time_to_first_chunk(self, request_start: float) -> float | None:
        return self._response.time_to_first_chunk(request_start)


class PreparedRequestModel(WrapperModel):
    """Freeze SDK input after preparation and before provider execution."""

    def __init__(
        self,
        wrapped: Model,
        prepare: Callable[
            [Sequence[ModelMessage], ModelSettings | None, ModelRequestParameters, bool],
            Awaitable[Model],
        ],
        failed: Callable[[Exception], None],
        observed: _StreamObserver | None = None,
        project_response: Callable[[ModelResponse], ModelResponse] | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._prepare = prepare
        self._failed = failed
        self._observed = observed or (lambda event, error: None)
        self._project_response = project_response or (lambda response: response)
        self._response_observed = False

    @property
    def response_observed(self) -> bool:
        return self._response_observed

    def reset_response_observation(self) -> None:
        self._response_observed = False

    def project_response(self, response: ModelResponse) -> ModelResponse:
        self._response_observed = True
        return self._project_response(response)


    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model = await self._prepare(messages, model_settings, model_request_parameters, False)
        return self.project_response(await model.request(messages, model_settings, model_request_parameters))

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: PydanticRunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        model = await self._prepare(messages, model_settings, model_request_parameters, True)
        try:
            manager = model.request_stream(
                messages, model_settings, model_request_parameters, run_context,
            )
            response = await manager.__aenter__()
        except asyncio.CancelledError as error:
            self._observed("entry_error", error)
            raise
        except Exception as error:
            self._observed("entry_error", error)
            self._failed(error)
            raise
        self._observed("entered", None)
        try:
            yield _ProviderStreamedResponse(response, self._failed, self._observed, self.project_response)
        except BaseException as error:
            # Context managers receive consumer failures too. Forward their
            # suppression semantics without treating those failures as provider errors.
            try:
                suppressed = await manager.__aexit__(type(error), error, error.__traceback__)
            except asyncio.CancelledError as exit_error:
                if exit_error is not error:
                    self._observed("exit_error", exit_error)
                raise
            except Exception as exit_error:
                if exit_error is not error:
                    self._observed("exit_error", exit_error)
                    self._failed(exit_error)
                raise
            if not suppressed:
                raise
        else:
            try:
                await manager.__aexit__(None, None, None)
            except asyncio.CancelledError as error:
                self._observed("exit_error", error)
                raise
            except Exception as error:
                self._observed("exit_error", error)
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
        self._prepared_models: dict[int, PreparedRequestModel] = {}
        self._client_recorders: dict[int, ModelClientRequestRecorder] = {}
        self._provider_errors: dict[int, Exception] = {}
        self._pending_model_failure: Callable[[Sequence[ModelMessage]], Awaitable[None]] | None = None

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def wrap_node_run(
        self, ctx: PydanticRunContext[AgentContext[object]], *,
        node: AgentNode[AgentContext[object]], handler: WrapNodeRunHandler[AgentContext[object]],
    ) -> NodeResult[AgentContext[object]]:
        try:
            return await handler(node)
        finally:
            await self._finish_pending_model_failure(ctx.messages)

    async def on_node_run_error(
        self, ctx: PydanticRunContext[AgentContext[object]], *,
        node: AgentNode[AgentContext[object]], error: Exception,
    ) -> NodeResult[AgentContext[object]]:
        del node
        # Finalize before an outer recovery hook starts another shared-usage run.
        await self._finish_pending_model_failure(ctx.messages)
        raise error

    async def _finish_pending_model_failure(self, messages: Sequence[ModelMessage]) -> None:
        finish = self._pending_model_failure
        self._pending_model_failure = None
        if finish is not None:
            await finish(messages)

    async def before_run(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
    ) -> None:
        if self._recorder is None:
            return
        bind_metric_execution_context(
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
        stream_attempt_index: int | None = None

        async def prepare(
            messages: Sequence[ModelMessage],
            settings: ModelSettings | None,
            parameters: ModelRequestParameters,
            streaming: bool,
        ) -> Model:
            nonlocal request_sequence
            compaction = self._journal.latest_for_step(ctx.run_step, purpose="compaction")
            if compaction is not None and compaction.status is None:
                # SDKs may prepare the agent wrapper before running compaction.
                # The summary's external recorder owns its input and completion.
                return model
            fact = self._journal.latest_for_step(ctx.run_step)
            if fact is None or fact.status is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            request_sequence = fact.model_request_seq
            if fact.model_request_seq in self._prepared_models:
                return self._client_recorders[fact.model_request_seq].wrap_model(model)
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
            # A summary may have reused this wrapper before the logical request.
            prepared_model.reset_response_observation()
            self._prepared_models[fact.model_request_seq] = prepared_model
            if self._interaction_recorder is not None:
                await self._interaction_recorder.commit_history_boundary()
            return self._client_recorders[fact.model_request_seq].wrap_model(model)

        def provider_failed(error: Exception) -> None:
            if request_sequence is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            recorder = self._client_recorders.get(request_sequence)
            # SDK streaming failures surface in the consumer, bypassing its
            # non-stream error-recovery ledger. Consume only this native group
            # here, then hide its type so SDK cannot count it a second time.
            if isinstance(error, FallbackExceptionGroup):
                if request_sequence not in self._provider_errors:
                    usage = _failed_attempt_usage(error)
                    if usage is not None:
                        ctx.usage.incr(usage)
                    if ctx.usage_limits is not None:
                        try:
                            ctx.usage_limits.check_tokens(ctx.usage)
                            ctx.usage_limits.check_cost(ctx.usage, warn_if_cost_unavailable=False)
                        except UsageLimitExceeded as exceeded:
                            exceeded.__cause__ = error
                            error = exceeded
                    self._provider_errors[request_sequence] = error
                else:
                    error = self._provider_errors[request_sequence]
            safe_error = model_request_error(
                error, content_omission_reason=recorder.body_omission_reason if recorder else None,
                fact=self._journal.latest_for_step(ctx.run_step),
                stream_attempts=None if recorder is None else recorder.stream_attempts,
            )
            self._provider_errors.setdefault(request_sequence, safe_error)
            if safe_error is not error:
                raise safe_error from None

        def stream_observed(event: str, error: BaseException | None) -> None:
            nonlocal stream_attempt_index
            recorder = self._client_recorders.get(request_sequence) if request_sequence is not None else None
            if recorder is None or recorder.stream_attempts is not None:
                # The retry adapter observes each SDK stream against its own HTTP range.
                return
            if event in {"entered", "entry_error"}:
                stream_attempt_index = len(recorder.requests) - 1
            if stream_attempt_index is not None:
                recorder.observe_stream(event, attempt_index=stream_attempt_index, error=error)

        def project_response(response: ModelResponse) -> ModelResponse:
            recorder = self._client_recorders.get(request_sequence) if request_sequence is not None else None
            return project_model_response_attempts(
                response, content_omission_reason=recorder.body_omission_reason if recorder else None,
            )

        prepared_model = PreparedRequestModel(
            model, prepare, provider_failed, stream_observed, project_response,
        )
        request_context.model = prepared_model
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
        history = ctx.messages
        history_identities = {id(message) for message in history}
        usage_before = copy(ctx.usage)
        deferred_failure = False
        fact = self._journal.begin(
            ctx.run_step,
            purpose="agent",
            output_retry_index=None if ctx.retry <= 0 else ctx.retry,
        )
        model_request_seq = fact.model_request_seq
        async def stream_attempt_observed(attempt: ModelStreamAttempt) -> None:
            if self._interaction_recorder is not None:
                self._interaction_recorder.observe_stream_attempt(fact, attempt)
            if self._event_sink is None:
                return
            event_type = (ExecutionEventType.MODEL_ATTEMPT_STARTED if attempt.status == "RUNNING"
                          else ExecutionEventType.MODEL_ATTEMPT_FINISHED)
            payload: dict[str, JsonValue] = {
                "execution_id": self._execution_id, "agent_run_seq": self._agent_run_seq,
                "model_request_seq": fact.model_request_seq,
                "stream_attempt_seq": attempt.stream_attempt_seq, "status": attempt.status,
                "replaces_stream_attempt_seq": attempt.replaces_stream_attempt_seq,
                "retry_scheduled": attempt.retry_scheduled,
                "started_at": attempt.started_at.isoformat(),
                "finished_at": None if attempt.finished_at is None else attempt.finished_at.isoformat(),
                "duration_ns": attempt.duration_ns,
                "error_code": None if attempt.error is None else _model_error_code(attempt.error),
                "error_type": None if attempt.error is None else environ.logging.redact(
                    f"{type(attempt.error).__module__}.{type(attempt.error).__qualname__}"),
                "usage": _event_usage(None, usage=attempt.usage),
            }
            await self._event_sink(event_type, payload)

        with ModelClientRequestRecorder(
            record_http_bodies=record_http_bodies_enabled(ctx.capabilities.values()),
            stream_attempt_observer=stream_attempt_observed,
        ) as client_recorder:
            self._client_recorders[model_request_seq] = client_recorder
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
                    prepared_model = self._prepared_models.get(model_request_seq)
                    observed_response = prepared_model is not None and prepared_model.response_observed
                    if failed and (_fallback_error(provider_error) is not None or observed_response):
                        # SDK commits the interrupted continuation after cancelling
                        # this wrapper. Finish this same fact at its public node boundary.
                        async def finish_failed_request(messages: Sequence[ModelMessage]) -> None:
                            try:
                                prepared_model = self._prepared_models.get(model_request_seq)
                                observed_response = prepared_model is not None and prepared_model.response_observed
                                partial = messages[-1] if observed_response and messages is history and messages else None
                                if (not isinstance(partial, ModelResponse) or id(partial) in history_identities
                                    or partial.state != "interrupted" or partial.run_id != ctx.run_id
                                    or partial.conversation_id != ctx.conversation_id):
                                    partial = None
                                usage = _failed_attempt_usage(provider_error)
                                if partial is not None:
                                    partial = project_model_response_attempts(
                                        partial, content_omission_reason=client_recorder.body_omission_reason,
                                    )
                                if partial is not None or usage is not None:
                                    # Use this SDK's native counter/cost and resumed-seed
                                    # semantics without re-merging stream segments.
                                    usage = ctx.usage - usage_before
                                observed_error = provider_error
                                if ctx.usage_limits is not None:
                                    try:
                                        ctx.usage_limits.check_tokens(ctx.usage)
                                        ctx.usage_limits.check_cost(ctx.usage, warn_if_cost_unavailable=False)
                                    except UsageLimitExceeded as exceeded:
                                        observed_error = exceeded
                                interrupted = await self._complete_request(
                                    fact, run_context, selected_model, status="FAILED", response=partial,
                                    error_code=_model_error_code(observed_error), usage=usage, phase="failed",
                                    usage_includes_unreturned=True,
                                )
                                if interrupted:
                                    raise asyncio.CancelledError
                            finally:
                                self._client_recorders.pop(model_request_seq, None)
                                self._prepared_models.pop(model_request_seq, None)
                                self._provider_errors.pop(model_request_seq, None)
                                self._journal.consume(model_request_seq)

                        self._pending_model_failure = finish_failed_request
                        deferred_failure = True
                        raise
                    await self._complete_request(
                        fact,
                        run_context,
                        selected_model,
                        status="FAILED" if failed else "CANCELLED",
                        response=None,
                        error_code=None if provider_error is None else _model_error_code(provider_error),
                        usage=_failed_attempt_usage(provider_error),
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
                    safe_error = model_request_error(
                        error, content_omission_reason=client_recorder.body_omission_reason, fact=fact,
                        stream_attempts=client_recorder.stream_attempts,
                    )
                    interrupted = await self._complete_request(
                        fact,
                        run_context,
                        selected_model,
                        status="FAILED",
                        response=None,
                        error_code=_model_error_code(safe_error),
                        usage=_failed_attempt_usage(error),
                        phase="failed",
                    )
                    if interrupted:
                        raise asyncio.CancelledError from error
                    if safe_error is not error:
                        raise safe_error from None
                    raise

                # Freeze the resolved SDK identity before publishing the completed request.
                response.run_id = response.run_id or ctx.run_id
                response.conversation_id = response.conversation_id or ctx.conversation_id
                interrupted = await self._complete_request(
                    fact,
                    run_context,
                    selected_model,
                    status="SUCCEEDED",
                    response=response,
                    error_code=None,
                    usage=model_response_usage(response),
                    phase="completed",
                )
                if interrupted:
                    raise asyncio.CancelledError
                return response
            finally:
                if not deferred_failure:
                    self._client_recorders.pop(model_request_seq, None)
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
        del request_context
        fact = self._journal.latest_for_step(ctx.run_step)
        recorder = None if fact is None else self._client_recorders.get(fact.model_request_seq)
        return project_model_response_attempts(
            response, content_omission_reason=recorder.body_omission_reason if recorder else None,
        )

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
        client_requests: tuple[ModelClientRequest, ...] | None = None,
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
                    client_requests=client_requests,
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
                    client_requests=client_requests,
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
                    client_requests=client_requests,
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
            usage = model_response_usage(response)
            measurements = _provider_usage_measurements(response)
        else:
            exception = error if isinstance(error, Exception) else None
            error_code = None if exception is None else _model_error_code(exception)
            status = "CANCELLED" if phase == "cancelled" else "FAILED"
            usage = _failed_attempt_usage(error)
            measurements = _usage_measurements(usage) if usage is not None else ()
        self._finish_request(
            fact,
            model,
            response if phase == "completed" else None,
            status,
            error_code,
            usage,
            client_requests=client_requests,
        )
        observation = self._capture_model(
            ctx.deps,
            fact,
            model=model,
            response=response if phase == "completed" else None,
            status=status,
            error_code=error_code,
            measurements=measurements,
        )

        async def handoff() -> None:
            await self._record_request_event(
                fact,
                phase=phase,
                response=response if phase == "completed" else None,
                error_code=error_code,
                usage=usage,
            )
            await self._publish_request_event(
                fact,
                phase=phase,
                response=response if phase == "completed" else None,
                error_code=error_code,
                usage=usage,
            )

        interrupted = await await_request_handoff(handoff())
        await self._record_model(observation)
        if interrupted:
            raise asyncio.CancelledError

    async def _record_request_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
        usage: RequestUsage | RunUsage | None = None,
    ) -> None:
        recorder = self._interaction_recorder
        if recorder is None:
            return
        await recorder.record_model_event(
            fact,
            phase=phase,
            response=response,
            error_code=error_code,
            usage=usage,
            include_observation=self._recorder is not None,
        )

    async def _publish_request_event(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None = None,
        error_code: str | None = None,
        usage: RequestUsage | RunUsage | None = None,
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
            "usage": _event_usage(response, usage=usage),
        }
        client_recorder = self._client_recorders.get(fact.model_request_seq)
        attempts = None if client_recorder is None else client_recorder.stream_attempts
        if phase != "started" and attempts:
            payload["stream_attempt_seq"] = attempts[-1].stream_attempt_seq
            if phase == "completed" and self._interaction_recorder is not None:
                message_seq = self._interaction_recorder.response_message_seq(fact.model_request_seq)
                if message_seq is not None:
                    payload["message_seq"] = message_seq
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
        usage: RequestUsage | RunUsage | None,
        phase: str,
        client_requests: tuple[ModelClientRequest, ...] | None = None,
        usage_includes_unreturned: bool = False,
    ) -> bool:
        model = self._prepared_models.get(fact.model_request_seq, model)
        client_recorder = self._client_recorders.get(fact.model_request_seq)
        unreturned = None if client_recorder is None else client_recorder.unreturned_stream_usage
        if not usage_includes_unreturned and unreturned is not None:
            usage = copy(usage) if usage is not None else RequestUsage()
            usage.incr(unreturned)
        finished = self._journal.finish(fact.model_request_seq, status=status)
        self._finish_request(
            finished,
            model,
            response,
            status,
            error_code,
            usage,
            client_requests=client_requests,
        )
        measurements = (
            _usage_measurements(usage) if usage is not None else ()
        )
        observation = self._capture_model(
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
                usage=usage,
            )
            await self._publish_request_event(
                finished,
                phase=phase,
                response=response,
                error_code=error_code,
                usage=usage,
            )

        interrupted = await await_request_handoff(handoff())
        await self._record_model(observation)
        return interrupted

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
        usage: RequestUsage | RunUsage | None,
        *,
        client_requests: tuple[ModelClientRequest, ...] | None = None,
    ) -> None:
        recorder = self._interaction_recorder
        if recorder is None:
            return
        if client_requests is None:
            client_recorder = self._client_recorders.get(fact.model_request_seq)
            if client_recorder is not None:
                client_requests = client_recorder.requests
        recorder.finish_model_interaction(
            fact,
            model=model,
            response=response,
            status=status,
            error_code=error_code,
            duration_ns=fact.duration_ns or 0,
            usage=usage,
            client_requests=client_requests,
            stream_attempts=(self._client_recorders[fact.model_request_seq].stream_attempts
                             if fact.model_request_seq in self._client_recorders else None),
        )

    def _capture_model(
        self,
        run_context: AgentContext[object] | None,
        fact: ModelRequestFact,
        *,
        model: Model,
        response: ModelResponse | None,
        status: str,
        error_code: str | None,
        measurements: tuple[MetricMeasurement, ...],
    ) -> Observation | None:
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
            return Observation(
                version=1,
                observation_id=fact.observation_id,
                kind="linktools.model.request",
                occurred_at=datetime.now(timezone.utc),
                source_namespace=self._source_namespace,
                tenant_id=self._tenant_id,
                status=status,
                error_code=error_code,
                correlation=metric_correlation(
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
        except Exception:
            _logger.exception("model metric observation rejected")

    async def _record_model(self, observation: Observation | None) -> None:
        if self._recorder is None or observation is None:
            return
        try:
            await self._recorder.try_record(observation)
        except Exception:
            _logger.exception("model metric observation rejected")


def _event_usage(
    response: ModelResponse | None, *, usage: RequestUsage | RunUsage | None = None,
) -> JsonValue | None:
    if usage is None:
        if response is None:
            return None
        usage = model_response_usage(response)
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
    return _usage_measurements(model_response_usage(response))


def _usage_measurements(usage: RequestUsage | RunUsage) -> tuple[MetricMeasurement, ...]:
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


def _fallback_error(error: BaseException) -> FallbackExceptionGroup | None:
    """Find native attempts retained by an explicit SDK exception cause."""
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, FallbackExceptionGroup):
            return error
        if error.__cause__ is None:
            return None
        error = error.__cause__
    return None


def _failed_attempt_usage(error: BaseException | None) -> RequestUsage | None:
    group = _fallback_error(error) if error is not None else None
    return model_attempts_usage(getattr(group, "attempts", None)) if group is not None else None


def model_request_error(
    error: Exception, *, content_omission_reason: str | None, fact: ModelRequestFact | None = None,
    stream_attempts: tuple[ModelStreamAttempt, ...] | None = None,
) -> Exception:
    """Apply request retention without attributing consumer errors to transport."""
    if isinstance(error, RunCancelled):
        return error
    group = _fallback_error(error)
    if content_omission_reason is None and group is None and not stream_attempts:
        return error
    details: dict[str, JsonValue] = {}
    attempts = getattr(group, "attempts", None)
    if attempts is not None:
        details["failed_attempts"] = model_attempts_details(
            attempts, content_omission_reason=content_omission_reason,
        )
        if fact is not None:
            details["model_request_seq"] = fact.model_request_seq
            details["purpose"] = fact.purpose
    if stream_attempts:
        details["stream_attempt_seq"] = stream_attempts[-1].stream_attempt_seq
        details["stream_attempt_count"] = len(stream_attempts)
        if fact is not None:
            details["model_request_seq"] = fact.model_request_seq
    return project_model_request_error(
        error, content_omission_reason=content_omission_reason,
        code=ErrorCode(_model_error_code(error)), safe_details=details,
    )


def _fallback_provider_only(error: BaseException) -> bool:
    if isinstance(error, FallbackExceptionGroup):
        return all(_fallback_provider_only(child) for child in error.exceptions)
    return isinstance(error, Exception) and model_binding_error(
        error, diagnostics=ErrorDiagnostics.from_exception(error, content_omission_reason="classification_only"),
    ) is not None


def _model_error_code(error: Exception) -> str:
    if isinstance(error, AIError):
        return error.code.value
    if isinstance(error, FallbackExceptionGroup) and _fallback_provider_only(error):
        return ErrorCode.MODEL_API_ERROR.value
    if isinstance(error, UsageLimitExceeded):
        return ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED.value
    if isinstance(error, RunCancelled):
        return ErrorCode.EXECUTION_CANCELLED.value
    if isinstance(error, ConcurrencyLimitExceeded):
        return ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED.value
    if isinstance(error, ContentFilterError):
        return ErrorCode.MODEL_CONTENT_FILTERED.value
    provider_error = model_binding_error(
        error, diagnostics=ErrorDiagnostics.from_exception(error, content_omission_reason="classification_only"),
    )
    if provider_error is not None:
        return provider_error.code.value
    if isinstance(error, UnexpectedModelBehavior):
        return ErrorCode.MODEL_RESPONSE_INVALID.value
    if isinstance(error, ValidationError):
        return ErrorCode.OUTPUT_VALIDATION_FAILED.value
    if isinstance(error, UserError):
        return ErrorCode.INTERNAL_ERROR.value
    return ErrorCode.INTERNAL_ERROR.value


__all__ = [
    "PreparedRequestModel", "ModelInteractionRecorder", "ModelObservationCapability",
    "ModelRequestEventSink", "model_request_error",
]
