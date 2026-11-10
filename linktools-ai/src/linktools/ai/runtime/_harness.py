#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness adapters for Runtime-owned durable state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities.abstract import ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, StructuredContent, SystemPromptPart, TextContent, ToolCallPart, ToolReturn, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition

from ..capability import AgentContext, ReferencedInputsCapability, ToolCallIdentity
from ..errors import AIError, ErrorCode
from ..model import ModelBinding, ModelClientRequestRecorder, ReferencedContent, ReferencedInputModel, ReferencedInputOrigin, model_input_content
from ..spec import CapabilityRef
from ..model.messages import RetiredReferencedContent, bind_referenced_input_origin
from ..model.messages import project_provider_messages, without_history_system_prompts

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_logger = environ.get_logger("ai.runtime.harness")



def bind_tool_reference_origins(ctx: RunContext[AgentContext[object]], value: object) -> object:
    """Bind newly produced references to this invocation, without shared state."""
    native = value.return_value if isinstance(value, ToolReturn) else value
    items = list(native) if isinstance(native, list) else [native]
    if (isinstance(value, ToolReturn) and isinstance(value.content, Sequence)
            and not isinstance(value.content, str)):
        items.extend(value.content)
    if not any(isinstance(item, ReferencedContent) and item.origin is None for item in items):
        return value
    identity = ToolCallIdentity.from_context(ctx)
    origin = ReferencedInputOrigin(
        namespace=identity.namespace, principal=identity.principal,
        execution_id=identity.execution_id, agent_run_id=identity.agent_run_id,
        agent_run_seq=identity.agent_run_seq, model_request_seq=identity.model_request_seq,
        tool_call_id=identity.tool_call_id, tool_name=identity.tool_name,
    )
    return bind_referenced_input_origin(value, origin)


class BindReferencedInputOrigins(AbstractCapability[AgentContext[object]]):
    """Retain native tool producer facts before SDK message/checkpoint projection."""

    id = "linktools.ai.referenced-input-origins"

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    async def wrap_tool_execute(
        self, ctx: RunContext[AgentContext[object]], *, call: ToolCallPart,
        tool_def: ToolDefinition, args: ValidatedToolArgs, handler: WrapToolExecuteHandler,
    ) -> Any:
        try:
            result = await handler(args)
        except SkipToolExecution as error:
            error.result = bind_tool_reference_origins(ctx, error.result)
            raise
        return bind_tool_reference_origins(ctx, result)


class SummaryProviderModel(ReferencedInputModel):
    """Project durable summaries before the shared provider-input boundary."""

    def __init__(
        self, model: Model, *, context: AgentContext[object] | None = None,
        referenced_inputs: (Mapping[CapabilityRef, ReferencedInputsCapability[object]]
                            | Callable[[], Mapping[CapabilityRef, ReferencedInputsCapability[object]]] | None) = None,
        binding: ModelBinding | None = None,
        preserve_errors: bool = False,
    ) -> None:
        sources = referenced_inputs if callable(referenced_inputs) else dict(referenced_inputs or {})
        self._reference_sources = sources
        self._reference_context = context

        async def load_many(contents: tuple[ReferencedContent, ...]) -> tuple[bytes, ...]:
            available = sources() if callable(sources) else sources
            if not available or context is None:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                              safe_details={"reason": "referenced_inputs_unavailable"})
            return await ReferencedInputsCapability.load_selected(context, contents, available)

        async def load(content: ReferencedContent) -> bytes:
            return (await load_many((content,)))[0]

        super().__init__(model, load=load, load_many=load_many, binding=binding,
                         preserve_errors=preserve_errors)

    async def load_references(self, messages: Sequence[ModelMessage]) -> tuple[bytes, ...]:
        for message in messages:
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and any(
                    isinstance(item, ReferencedContent) and item.origin is None
                    for item in model_input_content(part)
                ):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED, retryable=False,
                                  safe_details={"reason": "referenced_input_origin_unknown"})
        return await super().load_references(messages)

    def validate_inputs(self, value: object) -> None:
        if isinstance(value, Sequence) and self.content_omission_reason(value) is not None:
            available = self._reference_sources() if callable(self._reference_sources) else self._reference_sources
            if not available or self._reference_context is None:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                              safe_details={"reason": "referenced_inputs_unavailable"})
            if any(
                item.source not in available
                for message in value if isinstance(message, ModelRequest)
                for part in message.parts
                for item in model_input_content(part) if isinstance(item, ReferencedContent)
            ):
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                              safe_details={"reason": "referenced_input_source_mismatch"})
        super().validate_inputs(value)

    def project_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        projected = project_provider_messages(messages)
        result = []
        for message in projected:
            if not isinstance(message, ModelRequest):
                result.append(message)
                continue
            parts = []
            for part in message.parts:
                if not isinstance(part, ToolReturnPart):
                    parts.append(part)
                    continue
                content = [
                    item.content if (
                        isinstance(item, RetiredReferencedContent)
                        or isinstance(item, TextContent) and not isinstance(item, StructuredContent)
                        and isinstance(item.metadata, dict)
                        and item.metadata.get("source") == "linktools.attachment"
                    ) else item
                    for item in model_input_content(part)
                ]
                parts.append(replace(part, content=content if isinstance(part.content, list) else content[0]))
            result.append(replace(message, parts=parts))
        return result


def prepare_input_retention(messages: list[ModelMessage]) -> None:
    """Mark the current logical request before candidate or nested dispatch."""
    if SummaryProviderModel.content_omission_reason(messages) is not None:
        ModelClientRequestRecorder.omit_current_http_bodies("referenced_content_lifecycle")


class FallbackProviderModel(WrapperModel):
    """Keep logical retention outside SDK dispatch and hydration inside candidates."""

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        prepare_input_retention(messages)
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        prepare_input_retention(messages)
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context,
        ) as response:
            yield response


class ReinjectRuntimeSystemPrompt(AbstractCapability[Any]):
    """Refresh authority through public SDK hooks without erasing context summaries."""

    id = "linktools.ai.reinject-system-prompt"

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if ctx.agent is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = without_history_system_prompts(request_context.messages)
        current = await ctx.agent.system_prompt_parts(
            deps=ctx.deps, model=ctx.model, message_history=messages,
            prompt=ctx.prompt, usage=ctx.usage, model_settings=ctx.model_settings,
        )
        ctx.messages[:] = _prepend_system_parts(
            without_history_system_prompts(ctx.messages), current,
        )
        request_context.messages = _prepend_system_parts(messages, current)
        return request_context


def _prepend_system_parts(
    messages: list[ModelMessage], parts: Sequence[SystemPromptPart],
) -> list[ModelMessage]:
    if not parts:
        return messages
    if messages and isinstance(messages[0], ModelRequest):
        return [replace(messages[0], parts=[*parts, *messages[0].parts]), *messages[1:]]
    return [ModelRequest(parts=list(parts)), *messages]





__all__ = [
    "BindReferencedInputOrigins", "bind_tool_reference_origins",
    "ReinjectRuntimeSystemPrompt",
    "SummaryProviderModel",
    "FallbackProviderModel",
    "prepare_input_retention",
]
