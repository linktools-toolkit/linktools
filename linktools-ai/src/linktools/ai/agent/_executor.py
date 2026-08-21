#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI execution adapter for frozen AgentDefinitions."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from linktools.core import environ
from pydantic_ai import AgentRunResultEvent, ModelSettings
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimitExceeded, UsageLimits
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.step_persistence import StepStore

from ..capability import CapabilityMaterializationContext, SKILL_TOOL_NAMES
from ..core import ExecutionDeltaType, ExecutionEventType, JsonValue, UsageMetrics, canonical_sha256
from ..errors import AIError, ErrorCode
from ..spec import AgentUsageLimits
from ._builder import build_pydantic_agent
from ._capabilities import (
    PLANNING_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
    AgentRunScope,
    SubagentDelegate,
    ToolOperationBridge,
    compose_platform_capabilities,
    tool_allowed_in_planning,
    tool_name_allowed,
)
from ._definition import AgentDefinition


@dataclass(frozen=True, slots=True)
class LiveDelta:
    """An ephemeral model presentation update."""

    kind: ExecutionDeltaType
    content: str


@dataclass(frozen=True, slots=True)
class DurableBoundary:
    """A semantic boundary that is safe to persist in the execution audit."""

    kind: ExecutionEventType
    payload: JsonValue


AgentEmission = LiveDelta | DurableBoundary
EventSink = Callable[[AgentEmission], Awaitable[None]]


class UsageSink(Protocol):
    async def __call__(self, usage: UsageMetrics) -> None: ...


_logger = environ.get_logger("ai.agent.executor")


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    run_id: str
    output: JsonValue
    messages: "list[ModelMessage]"
    usage: UsageMetrics


class AgentExecutor:
    """Execute one immutable definition without loading declarations or committing runtime state."""

    def __init__(self, *, execution_root: Path) -> None:
        self._execution_root = execution_root.expanduser().resolve()

    async def execute(
        self,
        definition: AgentDefinition,
        user_prompt: str,
        history: "list[ModelMessage]",
        conversation_id: str,
        *,
        step_store: StepStore,
        step_run_id: str,
        segment_sequence: int,
        history_id: "str | None" = None,
        capability_context: CapabilityMaterializationContext,
        memory_scope: "str | None" = None,
        memory_store: "SearchableMemoryStore | None" = None,
        platform_tool_names: "tuple[str, ...]" = (),
        planning: bool = False,
        thinking: bool = False,
        parent_step_run_id: "str | None" = None,
        subagent_delegate: "SubagentDelegate | None" = None,
        event_sink: EventSink,
        usage_sink: "UsageSink | None" = None,
        tool_operations: "ToolOperationBridge | None" = None,
    ) -> AgentExecutionResult:
        if not isinstance(planning, bool) or not isinstance(thinking, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        run_usage = RunUsage()
        usage_limits = _to_usage_limits(definition.spec.usage_limits)
        result: AgentExecutionResult | None = None
        try:
            result = await self._execute(
                definition,
                user_prompt,
                history,
                conversation_id,
                step_store=step_store,
                step_run_id=step_run_id,
                segment_sequence=segment_sequence,
                history_id=history_id,
                capability_context=capability_context,
                memory_scope=memory_scope,
                memory_store=memory_store,
                platform_tool_names=platform_tool_names,
                planning=planning,
                thinking=thinking,
                parent_step_run_id=parent_step_run_id,
                subagent_delegate=subagent_delegate,
                event_sink=event_sink,
                run_usage=run_usage,
                usage_limits=usage_limits,
                tool_operations=tool_operations,
            )
            return result
        except UsageLimitExceeded as error:
            raise AIError(
                ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED,
                retryable=False,
                safe_details={
                    "limits": _limit_details(usage_limits),
                    "usage": _usage_details(run_usage),
                },
            ) from error
        finally:
            if usage_sink is not None:
                usage = result.usage if result is not None else _usage_metrics(run_usage)
                await usage_sink(usage)

    async def _execute(
        self,
        definition: AgentDefinition,
        user_prompt: str,
        history: "list[ModelMessage]",
        conversation_id: str,
        *,
        step_store: StepStore,
        step_run_id: str,
        segment_sequence: int,
        history_id: "str | None",
        capability_context: CapabilityMaterializationContext,
        memory_scope: "str | None",
        memory_store: "SearchableMemoryStore | None",
        platform_tool_names: "tuple[str, ...]",
        planning: bool,
        thinking: bool,
        parent_step_run_id: "str | None",
        subagent_delegate: "SubagentDelegate | None",
        event_sink: EventSink,
        run_usage: RunUsage,
        usage_limits: UsageLimits,
        tool_operations: "ToolOperationBridge | None",
    ) -> AgentExecutionResult:
        if not self._execution_root.is_dir():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if await step_store.get_run(run_id=step_run_id) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        model = definition.model.materialize()
        if thinking and not model.profile.get("supports_thinking", False):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "thinking", "reason": "model_not_supported"},
            )
        agent = build_pydantic_agent(definition, model=model)
        materialized: "list[PydanticAgentCapability[None]]" = []
        for binding in definition.effective_capabilities:
            materialized.extend(await binding.materialize(capability_context))
        control_tool_names = _control_tool_names(
            definition,
            planning=planning,
            subagent_available=subagent_delegate is not None,
        )
        scope = AgentRunScope(
            root=self._execution_root,
            agent_name=definition.spec.id,
            conversation_id=conversation_id,
            step_run_id=step_run_id,
            segment_sequence=segment_sequence,
            history_id=history_id,
            memory_scope=memory_scope,
            step_store=step_store,
            memory_store=memory_store,
            platform_tool_names=platform_tool_names,
            planning=planning,
            control_tool_names=control_tool_names,
            parent_step_run_id=parent_step_run_id,
            subagent_delegate=subagent_delegate,
            tool_operations=tool_operations,
        )
        platform = await compose_platform_capabilities(
            scope,
            model_factory=lambda _value: model,
            parent_model=model,
        )
        capabilities = tuple(materialized) + platform
        capabilities = (
            *capabilities,
            _ToolPresentation(
                definition.spec.allow_tools,
                planning=planning,
                control_tool_names=control_tool_names,
            ),
        )
        _logger.debug(
            "agent execution started: agent=%s definition=%s step=%s capability_count=%s planning=%s thinking=%s",
            definition.spec.id,
            definition.digest,
            step_run_id,
            len(capabilities),
            planning,
            thinking,
        )
        final_result = None
        model_settings = ModelSettings(thinking=True) if thinking else None
        async with agent.run_stream_events(
            user_prompt,
            message_history=history or None,
            conversation_id=conversation_id,
            run_id=step_run_id,
            usage_limits=usage_limits,
            usage=run_usage,
            capabilities=capabilities,
            model_settings=model_settings,
        ) as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    final_result = event.result
                    continue
                emission = _map_event(event)
                if emission is not None:
                    await event_sink(emission)
        if final_result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        run = await step_store.get_run(run_id=step_run_id)
        snapshot = await step_store.latest_snapshot(run_id=step_run_id)
        unresolved = await step_store.list_unresolved_tool_effects(run_id=step_run_id)
        if run is None or snapshot is None or unresolved or run.conversation_id != conversation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = final_result.output
        if not isinstance(output, definition.output_type):
            raise AIError(ErrorCode.OUTPUT_VALIDATION_FAILED)
        payload = cast(dict[str, JsonValue], output.model_dump(mode="json"))
        _logger.debug("agent execution completed: definition=%s step=%s", definition.digest, step_run_id)
        usage = _usage_metrics(run_usage)
        return AgentExecutionResult(final_result.run_id, payload, final_result.all_messages(), usage)


class _ToolPresentation(AbstractCapability[None]):
    def __init__(
        self,
        allow_tools: "tuple[str, ...]",
        *,
        planning: bool,
        control_tool_names: "frozenset[str]",
    ) -> None:
        self._allow_tools = allow_tools
        self._planning = planning
        self._control_tool_names = control_tool_names

    async def prepare_tools(
        self,
        _ctx: RunContext[None],
        tool_defs: "list[ToolDefinition]",
    ) -> "list[ToolDefinition]":
        names = [tool.name for tool in tool_defs]
        if len(names) != len(set(names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        selected: list[ToolDefinition] = []
        for tool in tool_defs:
            if tool.name not in self._control_tool_names and not _function_tool_allowed(
                tool.name,
                self._allow_tools,
            ):
                continue
            if self._planning and not tool_allowed_in_planning(
                tool,
                control_tool_names=self._control_tool_names,
            ):
                continue
            selected.append(tool)
        return selected

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


def _control_tool_names(
    definition: AgentDefinition,
    *,
    planning: bool,
    subagent_available: bool,
) -> "frozenset[str]":
    names: set[str] = set()
    if any(binding.provider == "skill" for binding in definition.effective_capabilities):
        names.update(SKILL_TOOL_NAMES)
    if subagent_available:
        names.update(SUBAGENT_TOOL_NAMES)
    if planning:
        names.update(PLANNING_TOOL_NAMES)
    return frozenset(names)


def _function_tool_allowed(name: str, allow_tools: "tuple[str, ...]") -> bool:
    if tool_name_allowed(name, allow_tools):
        return True
    if not name.startswith("mcp__"):
        return False
    server, separator, tool = name[5:].partition("__")
    if not separator or not server or not tool:
        return False
    selector = f"mcp__{server}"
    return selector in allow_tools or f"{selector}__*" in allow_tools


def _map_event(event: object) -> "AgentEmission | None":
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
        return LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, event.part.content)
    if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart) and event.part.content:
        return LiveDelta(ExecutionDeltaType.ASSISTANT_THINKING_DELTA, event.part.content)
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
        return LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, event.delta.content_delta)
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
        return LiveDelta(ExecutionDeltaType.ASSISTANT_THINKING_DELTA, event.delta.content_delta)
    if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
        text = event.part.content
        return DurableBoundary(
            ExecutionEventType.ASSISTANT_PART_COMPLETED,
            {"part_type": "text", "digest": canonical_sha256(text), "characters": len(text)},
        )
    if isinstance(event, PartEndEvent) and isinstance(event.part, ThinkingPart):
        text = event.part.content
        return DurableBoundary(
            ExecutionEventType.ASSISTANT_PART_COMPLETED,
            {"part_type": "thinking", "digest": canonical_sha256(text), "characters": len(text)},
        )
    if isinstance(event, FunctionToolCallEvent):
        part = event.part
        return DurableBoundary(
            ExecutionEventType.TOOL_CALL_STARTED,
            {
                "call_id": part.tool_call_id,
                "tool_name": part.tool_name,
                "arguments_digest": canonical_sha256(part.args_as_dict()),
            },
        )
    if isinstance(event, FunctionToolResultEvent):
        part = event.part
        if isinstance(part, ToolReturnPart):
            success = part.outcome == "success"
            return DurableBoundary(
                ExecutionEventType.TOOL_CALL_FINISHED,
                {
                    "call_id": part.tool_call_id,
                    "tool_name": part.tool_name,
                    "result_digest": canonical_sha256(str(part.content)) if success else None,
                    "status": "SUCCEEDED" if success else "FAILED",
                },
            )
        if isinstance(part, RetryPromptPart):
            return DurableBoundary(
                ExecutionEventType.TOOL_CALL_FINISHED,
                {
                    "call_id": part.tool_call_id,
                    "tool_name": part.tool_name or "unknown",
                    "result_digest": None,
                    "status": "FAILED",
                    "safe_error_code": ErrorCode.OUTPUT_VALIDATION_FAILED.value,
                },
            )
    return None


def _to_usage_limits(value: AgentUsageLimits | None) -> UsageLimits:
    if value is None:
        return UsageLimits(
            cost_limit=None,
            request_limit=None,
            tool_calls_limit=None,
            input_tokens_limit=None,
            output_tokens_limit=None,
            total_tokens_limit=None,
        )
    return UsageLimits(
        cost_limit=None,
        request_limit=value.model_requests,
        tool_calls_limit=value.tool_calls,
        input_tokens_limit=value.input_tokens,
        output_tokens_limit=value.output_tokens,
        total_tokens_limit=value.total_tokens,
    )


def _usage_metrics(value: RunUsage) -> UsageMetrics:
    return UsageMetrics(
        model_requests=value.requests,
        tool_calls=value.tool_calls,
        input_tokens=value.input_tokens,
        output_tokens=value.output_tokens,
        cache_read_tokens=value.cache_read_tokens,
        cache_write_tokens=value.cache_write_tokens,
    )


def _limit_details(value: UsageLimits) -> dict[str, int | None]:
    return {
        "model_requests": value.request_limit,
        "tool_calls": value.tool_calls_limit,
        "input_tokens": value.input_tokens_limit,
        "output_tokens": value.output_tokens_limit,
        "total_tokens": value.total_tokens_limit,
    }


def _usage_details(value: RunUsage) -> dict[str, int]:
    metrics = _usage_metrics(value)
    return {
        "model_requests": metrics.model_requests,
        "tool_calls": metrics.tool_calls,
        "input_tokens": metrics.input_tokens,
        "output_tokens": metrics.output_tokens,
        "total_tokens": metrics.total_tokens,
    }


__all__ = [
    "AgentExecutionResult",
    "AgentEmission",
    "AgentExecutor",
    "DurableBoundary",
    "EventSink",
    "LiveDelta",
    "UsageSink",
]
