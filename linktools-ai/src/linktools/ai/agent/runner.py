#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official Harness and local Agent execution boundaries."""

import os
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from linktools.core import environ

from ..core.json import JsonValue
from .context import AgentBinding


class AgentRunner(Protocol):
    async def run(self, binding: AgentBinding, prompt: str) -> str: ...
    async def resume(self, binding: AgentBinding, execution_id: str, prompt: str) -> str: ...


class LocalTool(Protocol):
    async def __call__(self, **kwargs: JsonValue) -> 'dict[str, JsonValue]': ...


class LocalEventHandler(Protocol):
    def __call__(self, event: 'dict[str, JsonValue]') -> 'Awaitable[None] | None': ...


@dataclass(frozen=True, slots=True)
class LocalAgentResult:
    run_id: str
    output: str
    messages: 'list[ModelMessage]'


class LocalAgentRunner:
    """Own Pydantic AI construction while exposing a transport-neutral run port."""

    def __init__(
        self,
        root: Path,
        config: 'dict[str, JsonValue]',
        *,
        model: 'str | Model | None' = None,
        base_url: 'str | None' = None,
        api_key: 'str | None' = None,
        tools: 'tuple[LocalTool, ...]' = (),
        capabilities: 'tuple[AgentCapability[None], ...]' = (),
    ) -> None:
        self._root = root
        self._config = config
        self._model = model
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._tools = tools
        self._capabilities = capabilities
        self._agents: dict[str | None, Agent[None, str]] = {}
        self._logger = environ.get_logger("ai.agent.runner")

    async def run(
        self,
        agent_id: 'str | None',
        prompt: str,
        history: 'list[ModelMessage]',
        conversation_id: str,
        *,
        on_event: 'LocalEventHandler | None' = None,
    ) -> LocalAgentResult:
        agent = self._get_agent(agent_id)
        self._logger.info("agent runner started: agent=%s conversation=%s", agent_id or "default", conversation_id)
        final_result = None
        output_parts: list[str] = []
        async with agent.run_stream_events(
            prompt,
            message_history=history or None,
            conversation_id=conversation_id,
        ) as events:
            async for event in events:
                if isinstance(event, AgentRunResultEvent):
                    final_result = event.result
                    continue
                mapped = _map_event(event)
                if mapped is None:
                    continue
                if mapped["type"] == "text":
                    output_parts.append(str(mapped["text"]))
                if on_event is not None:
                    pending = on_event(mapped)
                    if pending is not None:
                        await pending
        if final_result is None:
            raise RuntimeError("Agent ended without a result")
        result = cast("LocalAgentResult", LocalAgentResult(final_result.run_id, "".join(output_parts) or str(final_result.output), final_result.all_messages()))
        self._logger.info("agent runner completed: agent=%s run=%s", agent_id or "default", result.run_id)
        return result

    def _get_agent(self, agent_id: 'str | None') -> 'Agent[None, str]':
        existing = self._agents.get(agent_id)
        if existing is not None:
            return existing
        model = self._model or self._config.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        if isinstance(model, str) and model != "test":
            model = OpenAIChatModel(
                model.removeprefix("openai:"),
                provider=OpenAIProvider(
                    base_url=self._base_url or _config_string(self._config, "base_url"),
                    api_key=self._api_key,
                ),
            )
        test_model = model == "test" or isinstance(model, TestModel)
        selected_tools = self._tools if test_model else ()
        selected_capabilities = () if test_model else self._capabilities
        agent = Agent(
            model,
            name=agent_id or "linktools-local",
            instructions=self._instructions(agent_id),
            tools=selected_tools,
            capabilities=selected_capabilities,
        )
        self._agents[agent_id] = cast("Agent[None, str]", agent)
        return self._agents[agent_id]

    def _instructions(self, agent_id: 'str | None') -> str:
        configured = self._config.get("instructions")
        if isinstance(configured, str) and configured.strip():
            return configured
        selected = agent_id or str(self._config.get("default_agent", "default"))
        agent_file = self._root / ".linktools" / "agents" / f"{selected}.md"
        if agent_file.is_file():
            return agent_file.read_text(encoding="utf-8")
        return "You are the Linktools local coding assistant. Be concise and practical."


def _config_string(config: 'dict[str, JsonValue]', name: str) -> 'str | None':
    value = config.get(name)
    return value if isinstance(value, str) else None


def _map_event(
    event: 'PartStartEvent | PartDeltaEvent | PartEndEvent | FunctionToolCallEvent | FunctionToolResultEvent',
) -> 'dict[str, JsonValue] | None':
    if isinstance(event, PartStartEvent):
        if isinstance(event.part, TextPart) and event.part.content:
            return {"type": "text", "text": event.part.content}
        if isinstance(event.part, ThinkingPart) and event.part.content:
            return {"type": "thinking", "text": event.part.content}
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
            return {"type": "text", "text": event.delta.content_delta}
        if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
            return {"type": "thinking", "text": event.delta.content_delta}
    elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart) and event.part.content:
        return {"type": "text_end"}
    elif isinstance(event, FunctionToolCallEvent):
        part = event.part
        return cast("dict[str, JsonValue]", {
            "type": "tool",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "phase": "start",
            "arguments": part.args_as_dict(),
        })
    elif isinstance(event, FunctionToolResultEvent):
        part = event.part
        return cast("dict[str, JsonValue]", {
            "type": "tool",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "phase": "end",
            "ok": part.outcome == "success",
            "detail": str(part.content),
        })
    return None


__all__ = ["AgentRunner", "LocalAgentResult", "LocalAgentRunner", "LocalEventHandler", "LocalTool"]
