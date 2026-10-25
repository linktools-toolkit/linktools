#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness adapters for Runtime-owned durable state."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, cast, TYPE_CHECKING

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import FallbackExceptionGroup
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextContent, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.planning import (
    PlanItem as HarnessPlanItem,
)
from pydantic_ai_harness.planning import (
    TaskStatus,
)

from ..capability import AgentContext, ReferencedInputsCapability, ToolCallRetry
from ..errors import AIError, ErrorCode
from ..model import ModelBinding, ModelClientRequestRecorder, ReferencedContent, validate_model_inputs
from ..spec import CapabilityRef
from ._plan import PlanItem, RuntimePlanStore
from ._message import model_input_content, project_provider_messages, without_history_system_prompts
from ._metric_capability import model_request_error
from ._user_content_codec import RetiredReferencedContent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_logger = environ.get_logger("ai.runtime.harness")



class SummaryProviderModel(WrapperModel):
    """Keep SDK persistence separate from the provider's supported part union."""

    def __init__(
        self, model: Model, *, context: AgentContext[object] | None = None,
        referenced_inputs: (Mapping[CapabilityRef, ReferencedInputsCapability[object]]
                            | Callable[[], Mapping[CapabilityRef, ReferencedInputsCapability[object]]] | None) = None,
        binding: ModelBinding | None = None,
        preserve_errors: bool = False,
    ) -> None:
        super().__init__(model)
        self._context = context
        self._referenced_inputs = referenced_inputs if callable(referenced_inputs) else dict(referenced_inputs or {})
        self._binding = binding
        self._preserve_errors = preserve_errors

    async def _provider_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        projected = project_provider_messages(messages)
        if self._binding is not None:
            validate_model_inputs(self._binding, projected)
        references = [
            item for message in projected if isinstance(message, ModelRequest)
            for part in message.parts
            for item in model_input_content(part) if isinstance(item, ReferencedContent)
        ]
        sources = self._referenced_inputs() if callable(self._referenced_inputs) else self._referenced_inputs
        if references and (not sources or self._context is None or self._binding is None):
            raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING,
                          safe_details={"reason": "referenced_inputs_unavailable"})
        if any(item.source not in sources for item in references):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                          safe_details={"reason": "referenced_input_source_mismatch"})
        if references:
            ModelClientRequestRecorder.omit_current_http_bodies("referenced_content_lifecycle")
        result: list[ModelMessage] = []
        for message in projected:
            if not isinstance(message, ModelRequest):
                result.append(message)
                continue
            parts = []
            for part in message.parts:
                if (not isinstance(part, (UserPromptPart, ToolReturnPart))
                    or isinstance(part, UserPromptPart) and isinstance(part.content, str)):
                    parts.append(part)
                    continue
                content = []
                for item in model_input_content(part):
                    if isinstance(item, ReferencedContent):
                        data = await sources[item.source].load(self._context, item)
                        if item.representation == "text":
                            try:
                                item = data.decode("utf-8")
                            except UnicodeDecodeError:
                                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, retryable=False,
                                              safe_details={"reason": "text_input_not_utf8"}) from None
                        else:
                            item = BinaryContent(data, media_type=item.media_type, identifier=item.identifier)
                    elif (isinstance(part, ToolReturnPart) and (
                        isinstance(item, RetiredReferencedContent)
                        or isinstance(item, TextContent) and isinstance(item.metadata, dict)
                        and item.metadata.get("source") == "linktools.attachment"
                    )):
                        # Durable retirement metadata must never become native result text.
                        item = item.content
                    content.append(item)
                value = (content if isinstance(part, UserPromptPart) or isinstance(part.content, list)
                         else content[0])
                parts.append(replace(part, content=value))
            result.append(replace(message, parts=parts))
        if self._binding is not None:
            validate_model_inputs(self._binding, result)
        return result

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        try:
            return await self.wrapped.request(
                await self._provider_messages(messages), model_settings, model_request_parameters,
            )
        except Exception as error:
            if self._preserve_errors or isinstance(error, FallbackExceptionGroup):
                raise
            safe_error = model_request_error(error, content_omission_reason=self.content_omission_reason(messages))
            if safe_error is not error:
                raise safe_error from None
            raise

    async def count_tokens(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        try:
            return await self.wrapped.count_tokens(
                await self._provider_messages(messages), model_settings, model_request_parameters,
            )
        except Exception as error:
            if self._preserve_errors or isinstance(error, FallbackExceptionGroup):
                raise
            safe_error = model_request_error(error, content_omission_reason=self.content_omission_reason(messages))
            if safe_error is not error:
                raise safe_error from None
            raise

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None,
    ) -> ModelResponse:
        try:
            return await self.wrapped.compact_messages(
                replace(request_context, messages=await self._provider_messages(request_context.messages)),
                instructions=instructions,
            )
        except Exception as error:
            if self._preserve_errors or isinstance(error, FallbackExceptionGroup):
                raise
            safe_error = model_request_error(
                error, content_omission_reason=self.content_omission_reason(request_context.messages),
            )
            if safe_error is not error:
                raise safe_error from None
            raise

    @classmethod
    def content_omission_reason(cls, messages: Sequence[ModelMessage]) -> str | None:
        return "referenced_content_lifecycle" if any(
            isinstance(item, ReferencedContent)
            for message in messages if isinstance(message, ModelRequest)
            for part in message.parts
            for item in model_input_content(part)
        ) else None

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(
            await self._provider_messages(messages), model_settings, model_request_parameters, run_context,
        ) as response:
            yield response


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


class HarnessPlanStoreAdapter:
    """Expose the Runtime plan store through Harness' public PlanStore contract."""

    def __init__(self, store: RuntimePlanStore) -> None:
        if not isinstance(store, RuntimePlanStore):
            raise TypeError("store must be RuntimePlanStore")
        self._store = store

    async def get_items(self) -> list[HarnessPlanItem]:
        return _harness_plan_items(await self._store.get_items())

    async def set_items(self, items: list[HarnessPlanItem]) -> None:
        await self._store.write_plan(_runtime_plan_items(items))

    async def get_item(self, item_id: str) -> HarnessPlanItem | None:
        return next(
            (item for item in await self.get_items() if item.id == item_id), None
        )

    async def add_item(self, item: HarnessPlanItem) -> HarnessPlanItem:
        candidate = item.model_copy(deep=True)

        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            if any(value.id == candidate.id for value in values):
                raise ValueError(
                    f"A step with id {candidate.id!r} is already in this plan."
                )
            values.append(candidate.model_copy(deep=True))
            return _runtime_plan_items(values)

        await self._store.edit_items(edit)
        return candidate.model_copy(deep=True)

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> HarnessPlanItem | None:
        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            selected = next((item for item in values if item.id == item_id), None)
            if selected is None:
                return current
            if content is not None:
                selected.content = content
            if status is not None:
                selected.status = status
            if active_form is not None:
                selected.active_form = active_form
            if parent_id is not None:
                selected.parent_id = parent_id
            if depends_on is not None:
                selected.depends_on = list(depends_on)
            return _runtime_plan_items(values)

        previous, current, _revision = await self._store.edit_items(edit)
        if not any(item.id == item_id for item in _harness_plan_items(previous)):
            return None
        selected = next(
            (item for item in _harness_plan_items(current) if item.id == item_id),
            None,
        )
        if selected is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if active_form is not None:
            selected.active_form = active_form
        if depends_on is not None:
            selected.depends_on = list(depends_on)
        return selected.model_copy(deep=True)

    async def remove_item(self, item_id: str) -> bool:
        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            return _runtime_plan_items(
                [item for item in values if item.id != item_id]
            )

        previous, _current, _revision = await self._store.edit_items(edit)
        return any(item.id == item_id for item in _harness_plan_items(previous))


def _harness_plan_items(items: Sequence[PlanItem]) -> list[HarnessPlanItem]:
    return [
        HarnessPlanItem(
            id=_plan_item_id(index),
            content=item.content,
            status=TaskStatus(item.status),
        )
        for index, item in enumerate(items)
    ]


def _runtime_plan_items(items: list[HarnessPlanItem]) -> list[PlanItem]:
    if not isinstance(items, list):
        raise ToolCallRetry(
            "Plan items are invalid for this runtime. Correct the plan and retry."
        )
    values: list[PlanItem] = []
    for item in items:
        if not isinstance(item, HarnessPlanItem):
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        if (
            not isinstance(item.parent_id, (str, type(None)))
            or not isinstance(item.depends_on, list)
            or any(not isinstance(value, str) for value in item.depends_on)
            or item.parent_id is not None
            or item.depends_on
        ):
            raise ToolCallRetry(
                "Subtasks and dependencies are not enabled. Use a flat plan and retry."
            )
        if not isinstance(item.status, TaskStatus):
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        if item.status is TaskStatus.blocked:
            raise ToolCallRetry(
                "Subtasks and dependencies are not enabled. Use a flat plan and retry."
            )
        if not isinstance(item.content, str) or not item.content.strip():
            raise ToolCallRetry(
                "Each plan item must have non-empty content. Fill every item and retry."
            )
        if item.status not in {
            TaskStatus.pending,
            TaskStatus.in_progress,
            TaskStatus.completed,
            TaskStatus.cancelled,
        }:
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        values.append(PlanItem(item.content, cast(str, item.status.value)))
    return values


def _plan_item_id(index: int) -> str:
    return f"item-{index + 1}"



__all__ = [
    "ReinjectRuntimeSystemPrompt",
    "SummaryProviderModel",
    "FallbackProviderModel",
    "prepare_input_retention",
    "HarnessPlanStoreAdapter",
]
