#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI execution adapter for frozen AgentDefinitions."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from linktools.core import environ
from pydantic import BaseModel
from pydantic_ai import Agent, AgentRunResultEvent, TextOutput
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
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.step_persistence import StepStore

from ..capability import CapabilityRuntimeContext
from ..core import ExecutionEventType, JsonValue, canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelMaterializer
from ._capabilities import (
    AgentRunScope,
    compose_platform_capabilities,
    tool_name_allowed,
)
from ._definition import AgentDefinition
from ._output import AssistantTextOutput

EventSink = Callable[[ExecutionEventType, JsonValue], Awaitable[None]]

_logger = environ.get_logger("ai.agent.executor")


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    run_id: str
    output: JsonValue
    messages: "list[ModelMessage]"


class AgentExecutor:
    """Execute one immutable definition without loading declarations or committing runtime state."""

    def __init__(self, materializer: ModelMaterializer, *, execution_root: Path) -> None:
        self._materializer = materializer
        self._execution_root = execution_root.expanduser().resolve()

    async def execute(
        self,
        definition: AgentDefinition,
        prompt: str,
        history: "list[ModelMessage]",
        conversation_id: str,
        *,
        step_store: StepStore,
        step_run_id: str,
        segment_sequence: int,
        capability_context: CapabilityRuntimeContext,
        memory_namespace: "str | None" = None,
        memory_store: "SearchableMemoryStore | None" = None,
        platform_tool_names: "tuple[str, ...]" = (),
        parent_step_run_id: "str | None" = None,
        event_sink: EventSink,
    ) -> AgentExecutionResult:
        if not self._execution_root.is_dir():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if await step_store.get_run(run_id=step_run_id) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        model = self._materializer.materialize(definition.model_route, definition.model_connection)
        agent = self._build_agent(definition, model)
        materialized: list[PydanticAgentCapability[None]] = []
        for binding in definition.effective_capabilities:
            materialized.extend(await binding.materialize(capability_context))
        scope = AgentRunScope(
            root=self._execution_root,
            agent_name=definition.spec.id,
            conversation_id=conversation_id,
            step_run_id=step_run_id,
            segment_sequence=segment_sequence,
            memory_namespace=memory_namespace,
            step_store=step_store,
            memory_store=memory_store,
            platform_tool_names=platform_tool_names,
            parent_step_run_id=parent_step_run_id,
        )
        platform = await compose_platform_capabilities(
            scope,
            model_factory=lambda _value: model,
            parent_model=model,
        )
        capabilities = tuple(materialized) + platform
        capabilities = (*capabilities, _AllowlistPresentation(definition.spec.allow_tools))
        _logger.debug(
            "agent execution started: agent=%s definition=%s step=%s capability_count=%s",
            definition.spec.id,
            definition.digest,
            step_run_id,
            len(capabilities),
        )
        final_result = None
        async with agent.run_stream_events(
            prompt,
            message_history=history or None,
            conversation_id=conversation_id,
            capabilities=capabilities,
        ) as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    final_result = event.result
                    continue
                mapped = _map_event(event)
                if mapped is not None:
                    await event_sink(mapped[0], mapped[1])
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
        return AgentExecutionResult(final_result.run_id, payload, final_result.all_messages())

    def _build_agent(self, definition: AgentDefinition, model: "Model | str") -> "Agent[None, object]":
        output_type: type[BaseModel] | TextOutput = definition.output_type
        if output_type is AssistantTextOutput:
            output_type = TextOutput(_assistant_text_output)
        return cast(
            "Agent[None, object]",
            Agent(
                model,
                name=definition.spec.id,
                system_prompt=definition.prompt.system,
                instructions="\n".join((*definition.spec.instructions, *definition.prompt.instructions)),
                output_type=output_type,
            ),
        )


class _AllowlistPresentation(AbstractCapability[None]):
    def __init__(self, allow_tools: "tuple[str, ...]") -> None:
        self._allow_tools = allow_tools

    async def prepare_tools(self, _ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        selected = [tool for tool in tool_defs if _function_tool_allowed(tool.name, self._allow_tools)]
        names = [tool.name for tool in selected]
        if len(names) != len(set(names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        return selected

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


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


def _assistant_text_output(value: str) -> AssistantTextOutput:
    return AssistantTextOutput(text=value)


def _map_event(
    event: object,
) -> "tuple[ExecutionEventType, JsonValue] | None":
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
        return ExecutionEventType.ASSISTANT_TEXT_DELTA, {"text": event.part.content}
    if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart) and event.part.content:
        return ExecutionEventType.ASSISTANT_THINKING_DELTA, {"text": event.part.content}
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
        return ExecutionEventType.ASSISTANT_TEXT_DELTA, {"text": event.delta.content_delta}
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
        return ExecutionEventType.ASSISTANT_THINKING_DELTA, {"text": event.delta.content_delta}
    if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
        text = event.part.content
        return ExecutionEventType.ASSISTANT_TEXT_END, {"text_digest": canonical_sha256(text), "characters": len(text)}
    if isinstance(event, FunctionToolCallEvent):
        part = event.part
        return ExecutionEventType.TOOL_CALL_STARTED, {
            "call_id": part.tool_call_id,
            "tool_name": part.tool_name,
            "arguments_digest": canonical_sha256(part.args_as_dict()),
        }
    if isinstance(event, FunctionToolResultEvent):
        part = event.part
        if isinstance(part, ToolReturnPart):
            success = part.outcome == "success"
            return ExecutionEventType.TOOL_CALL_FINISHED, {
                "call_id": part.tool_call_id,
                "tool_name": part.tool_name,
                "result_digest": canonical_sha256(str(part.content)) if success else None,
                "status": "SUCCEEDED" if success else "FAILED",
            }
        if isinstance(part, RetryPromptPart):
            return ExecutionEventType.TOOL_CALL_FINISHED, {
                "call_id": part.tool_call_id,
                "tool_name": part.tool_name or "unknown",
                "result_digest": None,
                "status": "FAILED",
                "safe_error_code": ErrorCode.OUTPUT_VALIDATION_FAILED.value,
            }
    return None


__all__ = ["AgentExecutionResult", "AgentExecutor", "EventSink"]
