#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit model-binding composition using the SDK's native fallback policy."""

import inspect
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import partial, wraps
from time import perf_counter_ns
from typing import NoReturn

import anyio
from pydantic import TypeAdapter
from pydantic_ai import messages as sdk_messages
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.fallback import FallbackModel, FallbackOn, ResponseRejected
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, RunUsage

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._contract import ModelBinding
from ._input import normalize_input_mime_types


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FallbackBinding:
    """Compose ordered bindings without taking over SDK fallback or lifecycle.

    The SDK interprets ``fallback_on``, including annotated response handlers.
    A failed pinned continuation cannot rewind and replay the candidate chain.
    The contract describes candidate model semantics, not callable
    implementation identity. Each candidate still owns its input constraints
    and thinking profile.
    """

    bindings: tuple[ModelBinding, ...]
    fallback_on: FallbackOn = field(repr=False)

    def __init__(
        self, *bindings: ModelBinding, fallback_on: FallbackOn = (ModelAPIError,),
    ) -> None:
        if not bindings:
            raise ValueError("fallback binding requires at least one candidate")
        if any(isinstance(binding, FallbackBinding) for binding in bindings):
            raise ValueError("nested fallback bindings are not supported; choose one policy for ordered candidates")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "fallback_on", fallback_on)

    @property
    def provider(self) -> str:
        return "fallback"

    @property
    def model_identity(self) -> str:
        return "fallback:" + ",".join(binding.model_identity for binding in self.bindings)

    @property
    def contract(self) -> dict[str, JsonValue]:
        candidates: list[JsonValue] = []
        input_mime_types: set[str] = set()
        for binding in self.bindings:
            contract = dict(binding.contract)
            candidates.append({
                "provider": binding.provider,
                "model_identity": binding.model_identity,
                "contract": contract,
            })
            input_mime_types.update(normalize_input_mime_types(contract.get("input_mime_types", [])))
        return {
            "provider": self.provider,
            "model_identity": self.model_identity,
            "candidates": candidates,
            "input_mime_types": sorted(input_mime_types),
        }

    def materialize(
        self, *, prepare: Callable[[ModelBinding], Model] | None = None,
    ) -> Model:
        """Build a fresh native composition; the caller owns its async context.

        Runtime can prepare each leaf candidate before SDK composition,
        preserving candidate-specific capabilities without inspecting wrappers.
        """
        models = [prepare(binding) if prepare is not None else binding.materialize()
                  for binding in self.bindings]
        return _PinnedFallbackModel(models[0], *models[1:], fallback_on=self.fallback_on)


@dataclass
class _ContinuationRequest:
    response: ModelResponse
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_ns: int = field(default_factory=perf_counter_ns)
    candidate: Model | None = None


class _ContinuationCandidate(WrapperModel):
    """Remember the actual candidate only within its current continuation call."""

    def __init__(self, model: Model, continuation: ContextVar[_ContinuationRequest | None]) -> None:
        super().__init__(model)
        self._continuation = continuation

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if continuation := self._continuation.get():
            continuation.candidate = self.wrapped
        return await self.wrapped.request(messages, model_settings, model_request_parameters)


class _PinnedFallbackModel(FallbackModel):
    """Keep native selection while making a pinned failure terminal for this request."""

    def __init__(self, *models: Model, fallback_on: FallbackOn) -> None:
        self._continuation: ContextVar[_ContinuationRequest | None] = ContextVar(
            "fallback_continuation", default=None,
        )
        conditions = (fallback_on,) if callable(fallback_on) else fallback_on
        if not isinstance(conditions, Sequence) or not conditions or isinstance(conditions, (str, bytes)):
            # Native validation still owns unsupported and empty policies.
            super().__init__(*models, fallback_on=fallback_on)
            return
        super().__init__(*models, fallback_on=[
            self._stop_continuation, *(self._wrap_condition(condition) for condition in conditions),
        ])
        self.models = [_ContinuationCandidate(model, self._continuation) for model in self.models]

    def _stop_continuation(self, error: Exception) -> bool:
        if self._continuation.get() is not None:
            raise error
        return False

    def _wrap_condition(
        self, condition: type[Exception] | Callable[..., object],
    ) -> type[Exception] | Callable[..., object]:
        if isinstance(condition, type) and issubclass(condition, Exception):
            return condition
        if not callable(condition):
            return condition
        target = condition
        while isinstance(target, partial):
            target = target.func
        if not inspect.isroutine(target):
            target = target.__call__

        @wraps(target)
        async def check(value: object) -> object:
            result = condition(value)
            if inspect.isawaitable(result):
                result = await result
            continuation = self._continuation.get()
            if isinstance(value, ModelResponse) and continuation is not None and result:
                await self._reject_continuation(value, continuation)
            return result

        # Copy binding/partial parameter positions without classifying annotations;
        # the SDK remains the owner of response-vs-exception handler detection.
        with suppress(ValueError):
            check.__signature__ = inspect.signature(condition)
        return check

    async def _reject_continuation(
        self, response: ModelResponse, continuation: _ContinuationRequest,
    ) -> NoReturn:
        group = FallbackExceptionGroup("Pinned continuation response rejected", [ResponseRejected(1)])
        previous = continuation.response
        try:
            attempt_type = getattr(sdk_messages, "ModelRequestAttempt", None)
            if attempt_type is not None:
                usage = copy(response.usage)
                if usage.cost is None:
                    with suppress(Exception):
                        usage.cost = response.cost().total_price
                if previous.provider_response_id and previous.provider_response_id == response.provider_response_id:
                    usage = _usage_increment(usage, previous.usage)
                group.attempts = [*(response.failed_attempts or []), attempt_type(
                    # Missing leaf identity names the real composition boundary,
                    # not a candidate inferred from provider metadata.
                    model_name=response.model_name or previous.model_name or self.model_name,
                    provider_name=response.provider_name or previous.provider_name,
                    outcome="rejected", timestamp=continuation.timestamp,
                    duration=timedelta(microseconds=(perf_counter_ns() - continuation.started_ns) / 1000),
                    usage=usage,
                )]
        finally:
            if (
                response.state == "suspended" and response.provider_response_id
                and response.provider_response_id != previous.provider_response_id
                and continuation.candidate is not None
            ):
                # The caller already knows the old job. Only a newly rejected
                # job needs local teardown, directed to the actual candidate.
                with anyio.CancelScope(shield=True), suppress(Exception):
                    await continuation.candidate.cancel_suspended_response(response)
        raise group

    def _request_continuation(self, messages: list[ModelMessage]) -> _ContinuationRequest | None:
        if messages and isinstance(last := messages[-1], ModelResponse) and last.state == "suspended":
            return _ContinuationRequest(last)
        return None

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        token = self._continuation.set(self._request_continuation(messages))
        try:
            return await super().request(messages, model_settings, model_request_parameters)
        finally:
            self._continuation.reset(token)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[object] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with AsyncExitStack() as stack:
            token = self._continuation.set(self._request_continuation(messages))
            try:
                response = await stack.enter_async_context(super().request_stream(
                    messages, model_settings, model_request_parameters, run_context,
                ))
            finally:
                # Streaming consumers may close the context from another task.
                # Every native fallback decision happens during entry, not yield.
                self._continuation.reset(token)
            yield response


def _usage_increment(usage: RequestUsage, previous: RequestUsage) -> RequestUsage | None:
    if usage.unmeasured_requests or previous.unmeasured_requests:
        # An omitted cumulative observation cannot establish any increment.
        return None
    current, baseline = RunUsage(), RunUsage()
    current.incr(usage)
    baseline.incr(previous)
    increment = current - baseline
    increment.cost = usage.cost - previous.cost if usage.cost is not None and previous.cost is not None else None
    values = TypeAdapter(RunUsage).dump_python(increment, exclude={"requests", "tool_calls"})
    counters = {
        "input_tokens", "output_tokens", "cache_write_tokens", "cache_write_1h_tokens", "cache_read_tokens",
        "output_reasoning_tokens", "input_audio_tokens", "cache_audio_read_tokens", "output_audio_tokens",
        "audio_seconds", "cost",
    }
    if any(isinstance(value, (int, float, Decimal)) and value < 0
           for name, value in values.items() if name in counters):
        adapter = TypeAdapter(RequestUsage)
        raise AIError(ErrorCode.MODEL_RESPONSE_INVALID, retryable=False, safe_details={
            "reason": "continuation_usage_inconsistent",
            "usage_omission_reason": "cumulative_usage_decreased",
            "previous_usage": adapter.dump_python(previous, mode="json", include=counters),
            "current_usage": adapter.dump_python(usage, mode="json", include=counters),
        })
    return RequestUsage(**values)


__all__ = ["FallbackBinding"]
