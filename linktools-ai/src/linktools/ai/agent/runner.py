#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official Harness and local Agent execution boundaries."""

import asyncio
import os
import re
import unicodedata
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
)
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from linktools.core import environ

from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.ids import canonical_sha256
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
        self._agents: dict[tuple[str, str], Agent[None, str]] = {}
        self._agent_lock = asyncio.Lock()
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
        agent = await self._get_agent(agent_id)
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

    async def _get_agent(self, agent_id: 'str | None') -> 'Agent[None, str]':
        selected = agent_id or str(self._config.get("default_agent", "default"))
        instructions = await asyncio.to_thread(self._instructions, selected)
        model = self._model or self._config.get("model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        cache_key = (selected, canonical_sha256({
            "instructions": instructions,
            "tools": [type(tool).__qualname__ for tool in self._tools],
            "toolset_fingerprint": self._config.get("toolset_fingerprint", ""),
            "model": str(model),
            "output_schema": self._config.get("output_schema", "text"),
            "output_schema_revision": self._config.get("output_schema_revision", 1),
        }))
        async with self._agent_lock:
            existing = self._agents.get(cache_key)
            if existing is not None:
                return existing
            try:
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
                    name=selected,
                    instructions=instructions,
                    tools=selected_tools,
                    capabilities=selected_capabilities,
                )
            except Exception:
                raise
            cached = cast("Agent[None, str]", agent)
            self._agents[cache_key] = cached
            if len(self._agents) > 128:
                self._agents.pop(next(iter(self._agents)))
            return cached

    def _instructions(self, agent_id: 'str | None') -> str:
        selected = agent_id or str(self._config.get("default_agent", "default"))
        _validate_agent_id(selected)
        configured = self._config.get("instructions")
        if isinstance(configured, str) and configured.strip():
            return configured
        agents_root = (self._root / ".linktools" / "agents").resolve()
        agent_file = (agents_root / f"{selected}.md").resolve()
        try:
            agent_file.relative_to(agents_root)
        except ValueError as error:
            raise LinktoolsAIError(ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT) from error
        try:
            exists = agent_file.exists()
        except OSError as error:
            raise LinktoolsAIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        if exists:
            try:
                if not agent_file.is_file():
                    raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
                if agent_file.stat().st_size > 1024 * 1024:
                    raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID)
                return agent_file.read_text(encoding="utf-8")
            except LinktoolsAIError:
                raise
            except PermissionError as error:
                raise LinktoolsAIError(ErrorCode.STORAGE_UNAVAILABLE) from error
            except UnicodeDecodeError as error:
                raise LinktoolsAIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        return ""


def _validate_agent_id(agent_id: str) -> None:
    normalized = unicodedata.normalize("NFC", agent_id)
    if normalized != agent_id or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", agent_id):
        raise LinktoolsAIError(ErrorCode.AGENT_ID_INVALID)
    if agent_id in {".", ".."} or "/" in agent_id or "\\" in agent_id or "\x00" in agent_id:
        raise LinktoolsAIError(ErrorCode.AGENT_ID_INVALID)
    if agent_id.startswith(".") or re.match(r"^[A-Za-z]:", agent_id) or "%2f" in agent_id.lower() or "%5c" in agent_id.lower():
        raise LinktoolsAIError(ErrorCode.AGENT_ID_INVALID)


def _config_string(config: 'dict[str, JsonValue]', name: str) -> 'str | None':
    value = config.get(name)
    return value if isinstance(value, str) else None


def _map_event(
    event: 'PartStartEvent | PartDeltaEvent | PartEndEvent | FunctionToolCallEvent | FunctionToolResultEvent',
) -> 'dict[str, JsonValue] | None':
    if isinstance(event, PartStartEvent):
        if isinstance(event.part, TextPart) and event.part.content:
            return {"type": "text", "text": event.part.content}
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
            return {"type": "text", "text": event.delta.content_delta}
    elif isinstance(event, PartEndEvent) and isinstance(event.part, TextPart) and event.part.content:
        return {"type": "text_end"}
    elif isinstance(event, FunctionToolCallEvent):
        part = event.part
        return cast("dict[str, JsonValue]", {
            "type": "tool",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "phase": "start",
            "arguments_digest": canonical_sha256(part.args_as_dict()),
            "operation_id": part.tool_call_id,
            "status": "STARTED",
            "truncated": False,
        })
    elif isinstance(event, FunctionToolResultEvent):
        part = event.part
        return cast("dict[str, JsonValue]", {
            "type": "tool",
            "id": part.tool_call_id,
            "name": part.tool_name,
            "phase": "end",
            "operation_id": part.tool_call_id,
            "result_digest": canonical_sha256(part.content),
            "status": "SUCCEEDED" if part.outcome == "success" else "FAILED",
            "truncated": False,
            "safe_summary": "tool completed" if part.outcome == "success" else "tool failed",
        })
    return None


__all__ = ["AgentRunner", "LocalAgentResult", "LocalAgentRunner", "LocalEventHandler", "LocalTool"]
