#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Look ahead once without replaying SDK event processing or provider state."""

from __future__ import annotations

import asyncio
from contextlib import AbstractContextManager, AsyncExitStack, nullcontext
from copy import deepcopy
from dataclasses import replace
from time import monotonic_ns
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol, TypeVar, cast
from linktools.core import environ
from pydantic_ai import messages as sdk_messages
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.messages import (
    FinalResultEvent,
    ModelMessage,
    NativeToolCallPart,
    NativeToolReturnPart,
    FinishReason,
    ModelResponse,
    ModelResponseState,
    ModelResponseStreamEvent,
)
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.usage import RequestUsage

from ._client_request import ModelClientRequestRecorder
from ._stream_attempt import ModelStreamAttempt

_logger = environ.get_logger("ai.model.stream")
_SDK_MODEL_ATTEMPT = getattr(sdk_messages, "ModelRequestAttempt", None)
T = TypeVar("T")


class _StreamContextExit(BaseException):
    """Internal wake-up for a suspended source, never a provider exception."""

    def __init__(self, error: BaseException | None) -> None:
        super().__init__()
        self.error = error


class _PullDeadline(Protocol):
    def iterate(self, iterator: AsyncIterator[T]) -> AsyncIterator[T]: ...


class _PrimedStreamedResponse(StreamedResponse):
    """One lookahead event; SDK processing and provider state stay on inner."""

    def __init__(
        self,
        inner: StreamedResponse,
        iterator: AsyncIterator[ModelResponseStreamEvent],
        first: ModelResponseStreamEvent | None,
    ) -> None:
        self._response = inner
        self._source_iterator = iterator
        self._first = first
        super().__init__(inner.model_request_parameters)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Keep the SDK iterator cache authoritative, including native fallback lookahead.
        if self._event_iterator is None:
            self._event_iterator = self._get_event_iterator()
        return self._event_iterator

    def _enforce_request_deadline(self, deadline: _PullDeadline) -> None:
        # Preserve the SDK deadline when bypassing its event synthesis for already-processed events.
        self._event_iterator = deadline.iterate(self.__aiter__())

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._first is not None:
            first, self._first = self._first, None
            yield first
        async for event in self._source_iterator:
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
    def failed_attempts(self) -> list[object] | None:
        return getattr(self._response, "failed_attempts", None)

    @failed_attempts.setter
    def failed_attempts(self, value: list[object] | None) -> None:
        self._response.failed_attempts = value

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


class _RetryingStreamedResponse(_PrimedStreamedResponse):
    """Replace a failed provider attempt, never the surrounding agent execution."""

    def __init__(
        self, model: Model, messages: list[ModelMessage], settings: ModelSettings | None,
        parameters: ModelRequestParameters, run_context: RunContext[Any] | None,
        *, max_retries: int, initial_backoff: float,
        retryable: Callable[[ModelAPIError], bool],
        wait: Callable[[ModelAPIError, float], Awaitable[float]],
        recorder: ModelClientRequestRecorder | None,
    ) -> None:
        placeholder = CompletedStreamedResponse(
            ModelResponse(parts=[], model_name=model.model_name, state="incomplete"),
            model_request_parameters=parameters,
        )
        self._model = model
        self._messages = messages
        self._settings = settings
        self._run_context = run_context
        self._provider_owned_continuation = bool(parameters.native_tools) or any(
            isinstance(message, ModelResponse) and message.state == "suspended" for message in messages
        )
        self._max_retries = max_retries
        self._backoff = initial_backoff
        self._retryable = retryable
        self._wait = wait
        self._recorder = recorder
        self._attempt_committed = False
        self._cancel_requested = False
        self._consumer_error: BaseException | None = None
        self._context_exiting = False
        self._consumer_error_suppressed = False
        self._prior_usage: list[RequestUsage] = []
        self._prior_attempts: list[object] = []
        self._fallback_attempts: list[object] = []
        self._local_attempt_seq = 0
        self._active_http_index: int | None = None
        self._legacy_usage_accounted = _SDK_MODEL_ATTEMPT is None and run_context is not None
        super().__init__(placeholder, self._iterate_attempts(parameters), None)

    async def prime(self) -> None:
        try:
            self._first = await anext(self._source_iterator, None)
        except BaseException:
            # No response was handed to the SDK, so its response ledger cannot
            # account for these billed attempts. Once handed over, it alone counts them.
            known = self.known_failed_usage()
            if known is not None:
                if self._run_context is not None:
                    self._run_context.usage.incr(known)
                if self._recorder is not None:
                    self._recorder.record_unreturned_stream_usage(known)
            raise

    def known_failed_usage(self) -> RequestUsage | None:
        usages = [] if self._legacy_usage_accounted else [*self._prior_usage]
        current = _known_usage(self._response.get())
        if current is not None:
            usages.append(current)
        if not usages:
            return None
        result = RequestUsage()
        for usage in usages:
            result.incr(usage)
        return result

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._cancel_requested:
            return
        if self._first is not None:
            first, self._first = self._first, None
            self._attempt_committed = True
            yield first
        async for event in self._source_iterator:
            if self._cancel_requested:
                return
            self._attempt_committed = True
            yield event

    async def aclose(self, consumer_error: BaseException | None = None) -> bool:
        if self._context_exiting:
            return False
        self._context_exiting = True
        self._consumer_error = consumer_error
        try:
            await self._source_iterator.athrow(_StreamContextExit(consumer_error))
        except StopAsyncIteration:
            pass
        except BaseException as close_error:
            if consumer_error is None or self._consumer_error_suppressed:
                raise
            if close_error is not consumer_error:
                raise consumer_error from close_error
            return False
        return self._consumer_error_suppressed

    async def cancel(self) -> None:
        self._cancel_requested = True
        self._observe_sdk("cancel_requested")
        with self._scope():
            await self._response.cancel()

    async def close_stream(self) -> None:
        self._cancel_requested = True
        self._observe_sdk("close_started")
        try:
            with self._scope():
                await self._response.close_stream()
        except BaseException as error:
            self._observe_sdk("close_error", error)
            raise
        finally:
            self._observe_sdk("close_finished")

    def _observe_sdk(self, event: str, error: BaseException | None = None) -> None:
        if self._recorder is not None and self._active_http_index is not None:
            self._recorder.observe_stream(event, attempt_index=self._active_http_index, error=error)

    def _scope(self) -> AbstractContextManager[object]:
        return nullcontext() if self._recorder is None else self._recorder.bind()

    async def _observe(self, attempt: ModelStreamAttempt) -> None:
        if self._recorder is not None:
            await self._recorder.observe_stream_attempt(attempt)

    def _http_position(self) -> int:
        return len(self._recorder.requests) if self._recorder is not None else 0

    def _can_replace(self, response: ModelResponse | None) -> bool:
        if self._provider_owned_continuation or (response is not None and (
            response.state == "suspended" or any(
                isinstance(part, (NativeToolCallPart, NativeToolReturnPart)) for part in response.parts
            )
        )):
            return False
        if not self._attempt_committed:
            return True
        if self._recorder is None or not self._recorder.supports_stream_replacement:
            return False
        return response is not None

    async def _iterate_attempts(self, parameters: ModelRequestParameters) -> AsyncIterator[ModelResponseStreamEvent]:
        previous_seq = None
        for retry_index in range(self._max_retries + 1):
            if self._cancel_requested:
                raise asyncio.CancelledError
            self._attempt_committed = False
            self._local_attempt_seq += 1
            seq = (self._recorder.next_stream_attempt_seq() if self._recorder is not None
                   else self._local_attempt_seq)
            started_at, started_ns = datetime.now(timezone.utc), monotonic_ns()
            start = self._http_position()
            self._active_http_index = None
            await self._observe(ModelStreamAttempt(
                seq, "RUNNING", started_at, None, None, start, start, None, None, None, False, previous_seq,
            ))
            self._response = CompletedStreamedResponse(
                ModelResponse(parts=[], model_name=self._model.model_name, state="incomplete"),
                model_request_parameters=parameters,
            )
            current: StreamedResponse | None = None
            failure: BaseException | None = None
            retry = False
            exhausted = False
            operation_error: BaseException | None = None
            try:
                async with AsyncExitStack() as stack:
                    try:
                        with self._scope():
                            current = await stack.enter_async_context(self._model.request_stream(
                                self._messages, self._settings, parameters, self._run_context,
                            ))
                        self._response = current
                        if self._http_position() > start:
                            self._active_http_index = self._http_position() - 1
                        self._observe_sdk("entered")
                        iterator = aiter(current)
                        while True:
                            with self._scope():
                                event = await anext(iterator, None)
                            if event is None:
                                exhausted = True
                                self._observe_sdk("exhausted")
                                break
                            if self._prior_usage:
                                self._check_known_limits(_known_usage(current.get()))
                            try:
                                yield event
                            except _StreamContextExit as context_exit:
                                exit_error = context_exit.error
                            else:
                                continue
                            # Leave the control handler before raising, so its signal
                            # never becomes the caller's exception context.
                            if exit_error is not None:
                                raise exit_error
                            break
                    except BaseException as error:
                        operation_error = error
                        if self._active_http_index is None and self._http_position() > start:
                            self._active_http_index = self._http_position() - 1
                        if not self._context_exiting and not isinstance(error, GeneratorExit):
                            self._observe_sdk("entry_error" if current is None else "iteration_error", error)
                        raise
            except BaseException as error:
                failure = operation_error or error
                if error is not operation_error:
                    self._observe_sdk("exit_error", error)
                if isinstance(operation_error, GeneratorExit) and error is not operation_error:
                    failure = self._consumer_error or error
                partial = None if current is None else current.get()
                retry = (
                    error is failure and isinstance(error, ModelAPIError)
                    and not isinstance(operation_error, GeneratorExit)
                    and not self._context_exiting and not self._cancel_requested and not exhausted
                    and retry_index < self._max_retries and self._retryable(error)
                    and self._can_replace(partial)
                )
            if (self._context_exiting and self._consumer_error is not None
                    and operation_error is self._consumer_error and failure is None):
                self._consumer_error_suppressed = True
            partial = None if current is None else deepcopy(current.get())
            usage = _known_usage(partial)
            if retry:
                try:
                    self._check_known_limits(usage)
                except Exception as limit_error:
                    limit_error.__cause__ = failure
                    failure, retry = limit_error, False
            status = ("CANCELLED" if failure is None and self._context_exiting and not exhausted else
                      "SUCCEEDED" if failure is None else
                      "CANCELLED" if isinstance(failure, (asyncio.CancelledError, GeneratorExit)) else "FAILED")
            await self._observe(ModelStreamAttempt(
                seq, status, started_at, datetime.now(timezone.utc), monotonic_ns() - started_ns,
                start, self._http_position(), partial, usage, failure, retry,
            ))
            if failure is None:
                return
            if not retry:
                raise failure
            self._remember_failed_attempt(partial, usage, failure, started_at, monotonic_ns() - started_ns)
            previous_seq = seq
            self._response = CompletedStreamedResponse(
                ModelResponse(parts=[], model_name=self._model.model_name, state="incomplete"),
                model_request_parameters=parameters,
            )
            self._backoff = await self._wait(cast(ModelAPIError, failure), self._backoff)

    def _check_known_limits(self, current_usage: RequestUsage | None) -> None:
        if self._run_context is None or self._run_context.usage_limits is None:
            return
        provisional = deepcopy(self._run_context.usage)
        if not self._legacy_usage_accounted:
            for previous_usage in self._prior_usage:
                provisional.incr(previous_usage)
        if current_usage is not None:
            provisional.incr(current_usage)
        self._run_context.usage_limits.check_tokens(provisional)
        self._run_context.usage_limits.check_cost(provisional, warn_if_cost_unavailable=False)

    def _remember_failed_attempt(
        self, response: ModelResponse | None, usage: RequestUsage | None,
        error: BaseException, started_at: datetime, duration_ns: int,
    ) -> None:
        if usage is not None:
            self._prior_usage.append(usage)
            if self._legacy_usage_accounted:
                self._run_context.usage.incr(usage)
                if self._recorder is not None:
                    self._recorder.record_unreturned_stream_usage(usage)
        if _SDK_MODEL_ATTEMPT is not None:
            self._prior_attempts.append(_SDK_MODEL_ATTEMPT(
                model_name=self._model.model_name,
                provider_name=None if response is None else response.provider_name,
                outcome="error", error=f"{type(error).__module__}.{type(error).__qualname__}",
                timestamp=started_at, duration=timedelta(microseconds=duration_ns / 1000), usage=usage,
            ))

    @property
    def usage(self) -> RequestUsage:
        result = deepcopy(self._response.usage)
        if _SDK_MODEL_ATTEMPT is None and not self._legacy_usage_accounted:
            if self._prior_usage:
                result = _known_usage(self._response.get()) or result
            for usage in self._prior_usage:
                result.incr(usage)
        return result

    @property
    def failed_attempts(self) -> list[object] | None:
        attempts = [*self._fallback_attempts,
                    *(getattr(self._response.get(), "failed_attempts", None) or ()),
                    *self._prior_attempts]
        return list({id(attempt): attempt for attempt in attempts}.values()) or None

    @failed_attempts.setter
    def failed_attempts(self, value: list[object] | None) -> None:
        # Native fallback prepends its attempts to this public list before writing it back.
        self._fallback_attempts = [
            item for item in value or ()
            if all(item is not local for local in self._prior_attempts)
        ]
        self._response.failed_attempts = self._fallback_attempts or None

    def get(self) -> ModelResponse:
        response = self._response.get()
        if _SDK_MODEL_ATTEMPT is not None:
            return replace(response, failed_attempts=self.failed_attempts)
        return replace(response, usage=self.usage)


def _known_usage(response: ModelResponse | None) -> RequestUsage | None:
    if response is None:
        return None
    usage = deepcopy(response.usage)
    positive = any(getattr(usage, field, 0) for field in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "audio_seconds",
    ))
    # Public stream usage does not distinguish an unreported in-flight zero
    # from an explicitly reported zero on every supported SDK.
    if not positive and usage.cost is None:
        return None
    # Price each billed attempt before any compatibility aggregation.
    if usage.cost is None and response.model_name:
        try:
            usage.cost = response.cost().total_price
        except (LookupError, ValueError):
            pass
        except Exception as error:
            _logger.warning("model attempt pricing unavailable: error_type=%s", type(error).__name__)
    return usage
