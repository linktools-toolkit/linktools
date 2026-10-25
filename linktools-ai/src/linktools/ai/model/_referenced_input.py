#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ephemeral provider views for authorized referenced input."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime
from typing import Any, TYPE_CHECKING

from pydantic_ai.messages import (
    BinaryContent, FinalResultEvent, FinishReason, ModelMessage, ModelRequest,
    ModelResponse, ModelResponseState, ModelResponseStreamEvent, UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from ..errors import AIError, ErrorCode
from ._client_request import ModelClientRequestRecorder
from ._contract import ModelBinding, ModelInputValidator
from ._errors import model_request_error
from ._input import validate_model_inputs
from ._referenced_content import ReferencedContent
from ._response import project_model_attempts, project_model_response_attempts

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelRequestAttempt
    from ._stream import _PullDeadline


class ReferencedInputModel(WrapperModel):
    """Lower references only in provider calls, leaving SDK history unchanged.

    ``load`` owns source resolution, authorization and immutable byte validation.
    Provider operations borrow the actual selected model; its normal SDK async
    context remains responsible for entering and closing it.
    Native callers use that model's ModelInputValidator contract. An explicit
    binding is reserved for compositions which already own that exact model's
    construction, such as Runtime's candidate preparation.

    Unknown model contracts, including unprepared native fallback compositions,
    fail closed on references. A native Model alone does not grant permission.
    ``preserve_errors`` is for candidate dispatch whose enclosing owner must let
    SDK fallback policy inspect the original failure before applying retention.
    """

    def __init__(
        self, model: Model, *, load: Callable[[ReferencedContent], Awaitable[bytes]],
        binding: ModelBinding | None = None, preserve_errors: bool = False,
    ) -> None:
        super().__init__(model)
        self._load = load
        self._binding = binding
        self._preserve_errors = preserve_errors

    def validate_inputs(self, value: object) -> None:
        """Use the selected provider's contract, never an unrelated old binding."""
        if self._binding is not None:
            validate_model_inputs(self._binding, value)
        elif (validator := self._input_validator()) is not None:
            validator.validate_inputs(value)
        else:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, retryable=False,
                          safe_details={"reason": "model_input_contract_unknown"})

    def _input_validator(self) -> ModelInputValidator | None:
        model = self.wrapped
        seen: set[int] = set()
        while id(model) not in seen:
            seen.add(id(model))
            if isinstance(model, ModelInputValidator):
                return model
            if not isinstance(model, WrapperModel):
                break
            model = model.wrapped
        return None

    @classmethod
    def content_omission_reason(cls, messages: Sequence[ModelMessage]) -> str | None:
        return "referenced_content_lifecycle" if any(
            isinstance(item, ReferencedContent)
            for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content
        ) else None

    def _validate_request(self, messages: list[ModelMessage]) -> None:
        if (self.content_omission_reason(messages) is not None or self._binding is not None
                or self._input_validator() is not None):
            self.validate_inputs(messages)

    def project_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Project host-owned message parts before ephemeral input lowering."""
        return messages

    async def _provider_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        messages = self.project_messages(messages)
        if self.content_omission_reason(messages) is None:
            return messages
        ModelClientRequestRecorder.omit_current_http_bodies("referenced_content_lifecycle")
        result: list[ModelMessage] = []
        for message in messages:
            if not isinstance(message, ModelRequest):
                result.append(message)
                continue
            parts = []
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                    parts.append(part)
                    continue
                content = []
                for item in part.content:
                    if isinstance(item, ReferencedContent):
                        data = await self._load(item)
                        if item.representation == "text":
                            try:
                                item = data.decode("utf-8")
                            except UnicodeDecodeError:
                                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, retryable=False,
                                              safe_details={"reason": "text_input_not_utf8"}) from None
                        else:
                            item = BinaryContent(data, media_type=item.media_type, identifier=item.identifier)
                    content.append(item)
                parts.append(replace(part, content=content))
            result.append(replace(message, parts=parts))
        self.validate_inputs(result)
        return result

    def _safe_error(self, error: Exception, messages: Sequence[ModelMessage]) -> Exception:
        if self._preserve_errors:
            return error
        return model_request_error(error, content_omission_reason=self.content_omission_reason(messages))

    def _project_response(self, response: ModelResponse, messages: Sequence[ModelMessage]) -> ModelResponse:
        omission = self.content_omission_reason(messages)
        if self._preserve_errors or omission is None:
            return response
        return project_model_response_attempts(response, content_omission_reason=omission)

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self._validate_request(messages)
        try:
            response = await self.wrapped.request(
                await self._provider_messages(messages), model_settings, model_request_parameters,
            )
            return self._project_response(response, messages)
        except Exception as error:
            safe = self._safe_error(error, messages)
            if safe is not error:
                raise safe from None
            raise

    async def count_tokens(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        self._validate_request(messages)
        try:
            return await self.wrapped.count_tokens(
                await self._provider_messages(messages), model_settings, model_request_parameters,
            )
        except Exception as error:
            safe = self._safe_error(error, messages)
            if safe is not error:
                raise safe from None
            raise

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None,
    ) -> ModelResponse:
        self._validate_request(request_context.messages)
        try:
            response = await self.wrapped.compact_messages(
                replace(request_context, messages=await self._provider_messages(request_context.messages)),
                instructions=instructions,
            )
            return self._project_response(response, request_context.messages)
        except Exception as error:
            safe = self._safe_error(error, request_context.messages)
            if safe is not error:
                raise safe from None
            raise

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        self._validate_request(messages)
        try:
            manager = self.wrapped.request_stream(
                await self._provider_messages(messages), model_settings, model_request_parameters, run_context,
            )
            response = await manager.__aenter__()
        except Exception as error:
            safe = self._safe_error(error, messages)
            if safe is not error:
                raise safe from None
            raise
        try:
            yield _ReferencedStreamedResponse(
                response, lambda error: self._safe_error(error, messages),
                content_omission_reason=None if self._preserve_errors else self.content_omission_reason(messages),
            )
        except BaseException as consumer_error:
            try:
                suppressed = await manager.__aexit__(
                    type(consumer_error), consumer_error, consumer_error.__traceback__,
                )
            except Exception as error:
                # Passing a consumer error through __aexit__ does not turn it
                # into a provider failure or change its public classification.
                if error is consumer_error:
                    raise
                safe = self._safe_error(error, messages)
                if safe is not error:
                    raise safe from None
                raise
            if not suppressed:
                raise
        else:
            try:
                await manager.__aexit__(None, None, None)
            except Exception as error:
                safe = self._safe_error(error, messages)
                if safe is not error:
                    raise safe from None
                raise


class _ReferencedStreamedResponse(StreamedResponse):
    """Apply retention to provider pulls and closure, not consumer exceptions."""

    def __init__(
        self, response: StreamedResponse, safe_error: Callable[[Exception], Exception], *,
        content_omission_reason: str | None,
    ) -> None:
        self._response = response
        self._safe_error = safe_error
        self._content_omission_reason = content_omission_reason
        super().__init__(model_request_parameters=response.model_request_parameters)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Already-processed events share the SDK cache used by native fallback.
        if self._event_iterator is None:
            self._event_iterator = self._get_event_iterator()
        return self._event_iterator

    def _enforce_request_deadline(self, deadline: _PullDeadline) -> None:
        self._event_iterator = deadline.iterate(self.__aiter__())

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        try:
            async for event in self._response:
                yield event
        except Exception as error:
            safe = self._safe_error(error)
            if safe is not error:
                raise safe from None
            raise

    def get(self) -> ModelResponse:
        try:
            response = self._response.get()
            return response if self._content_omission_reason is None else project_model_response_attempts(
                response, content_omission_reason=self._content_omission_reason,
            )
        except Exception as error:
            safe = self._safe_error(error)
            if safe is not error:
                raise safe from None
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
    def failed_attempts(self) -> list[ModelRequestAttempt] | None:
        attempts = self._response.failed_attempts
        if attempts is None or self._content_omission_reason is None:
            return attempts
        return project_model_attempts(attempts, content_omission_reason=self._content_omission_reason)

    @failed_attempts.setter
    def failed_attempts(self, value: list[ModelRequestAttempt] | None) -> None:
        self._response.failed_attempts = value

    @property
    def cancelled(self) -> bool:
        return self._response.cancelled

    async def cancel(self) -> None:
        try:
            await self._response.cancel()
        except Exception as error:
            safe = self._safe_error(error)
            if safe is not error:
                raise safe from None
            raise

    async def close_stream(self) -> None:
        try:
            await self._response.close_stream()
        except Exception as error:
            safe = self._safe_error(error)
            if safe is not error:
                raise safe from None
            raise

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        return self._response.get_stream_cancel_errors()

    def time_to_first_chunk(self, request_start: float) -> float | None:
        return self._response.time_to_first_chunk(request_start)


__all__ = ["ReferencedInputModel"]
