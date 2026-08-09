#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI and Harness boundary for workspace execution."""

import asyncio
import os
import re
import unicodedata
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

from linktools.core import environ
from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    RetryPromptPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.step_persistence import StepStore

from ._capabilities import AgentCapabilityScope, AgentCatalogView, EmptyAgentCatalog, EmptySkillCatalog, compose_parent_capabilities
from ..capability import SkillCatalogView
from ..errors import ErrorCode, AIError
from ..core import canonical_sha256
from ..core import JsonValue


class AgentRunner(Protocol):
    async def run(
        self,
        agent_id: "str | None",
        prompt: str,
        history: "list[ModelMessage]",
        conversation_id: str,
        *,
        step_store: StepStore,
        step_run_id: str,
        segment_sequence: int,
        parent_step_run_id: "str | None" = None,
        memory_namespace: "str | None" = None,
        memory_store: "SearchableMemoryStore | None" = None,
        on_event: "EventHandler | None" = None,
    ) -> "WorkspaceAgentResult": ...


class AgentTool(Protocol):
    async def __call__(self, **kwargs: JsonValue) -> "dict[str, JsonValue]": ...


class EventHandler(Protocol):
    def __call__(self, event: "dict[str, JsonValue]") -> "Awaitable[None] | None": ...


@dataclass(frozen=True, slots=True)
class WorkspaceAgentResult:
    run_id: str
    output: str
    messages: "list[ModelMessage]"


@dataclass(frozen=True, slots=True)
class _ResolvedDefinition:
    agent_name: str
    instructions: str
    model: "str | Model"
    fingerprint: Mapping[str, JsonValue]


class WorkspaceAgentRunner:
    """Resolve one immutable Agent definition and execute its Harness segment."""

    def __init__(
        self,
        root: Path,
        config: "dict[str, JsonValue]",
        *,
        model: "str | Model | None" = None,
        base_url: "str | None" = None,
        api_key: "str | None" = None,
        tools: "tuple[AgentTool, ...]" = (),
        capabilities: "tuple[AgentCapability[None], ...]" = (),
        skill_catalog: "SkillCatalogView | None" = None,
        agent_catalog: "AgentCatalogView | None" = None,
    ) -> None:
        self._root = root.expanduser().resolve()
        self._config = config
        self._model = model
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._tools = tools
        self._workspace_capabilities = capabilities
        self._skill_catalog = skill_catalog if skill_catalog is not None else EmptySkillCatalog()
        self._agent_catalog = agent_catalog if agent_catalog is not None else EmptyAgentCatalog()
        self._agents: dict[tuple[str, str], Agent[None, str]] = {}
        self._agent_lock = asyncio.Lock()
        self._logger = environ.get_logger("ai.agent.runner")

    async def binding_digest(self, agent_id: "str | None") -> str:
        definition = await self._resolve_definition(agent_id)
        return canonical_sha256(definition.fingerprint)

    async def run(
        self,
        agent_id: "str | None",
        prompt: str,
        history: "list[ModelMessage]",
        conversation_id: str,
        *,
        step_store: StepStore,
        step_run_id: str,
        segment_sequence: int,
        parent_step_run_id: "str | None" = None,
        memory_namespace: "str | None" = None,
        memory_store: "SearchableMemoryStore | None" = None,
        on_event: "EventHandler | None" = None,
    ) -> WorkspaceAgentResult:
        definition = await self._resolve_definition(agent_id)
        existing = await step_store.get_run(run_id=step_run_id)
        if existing is not None:
            raise _SegmentAlreadyStarted(step_run_id)
        agent = await self._get_agent(definition)
        parent_model = self._materialize_model(definition.model)
        capabilities = await compose_parent_capabilities(
            AgentCapabilityScope(
                root=self._root,
                agent_name=definition.agent_name,
                conversation_id=conversation_id,
                step_run_id=step_run_id,
                segment_sequence=segment_sequence,
                memory_namespace=memory_namespace,
                step_store=step_store,
                workspace_capabilities=self._workspace_capabilities,
                skill_catalog=self._skill_catalog,
                agent_catalog=self._agent_catalog,
                memory_store=memory_store,
                context_target_tokens=_context_target_tokens(self._config),
                parent_step_run_id=parent_step_run_id,
            ),
            model_factory=lambda value: self._materialize_model(definition.model if value is None else value),
            parent_model=parent_model,
        )
        self._logger.info(
            "workspace agent segment started: agent=%s conversation=%s step=%s segment=%s",
            definition.agent_name,
            conversation_id,
            step_run_id,
            segment_sequence,
        )
        final_result = None
        output_parts: list[str] = []
        try:
            async with agent.run_stream_events(
                prompt,
                message_history=history or None,
                conversation_id=conversation_id,
                run_id=step_run_id,
                capabilities=capabilities,
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
        except AIError as error:
            if error.code is ErrorCode.STORAGE_CONFLICT and await step_store.get_run(run_id=step_run_id) is not None:
                raise _SegmentAlreadyStarted(step_run_id) from error
            raise
        if final_result is None:
            raise RuntimeError("Agent ended without a result")
        record = await step_store.get_run(run_id=step_run_id)
        snapshot = await step_store.latest_snapshot(run_id=step_run_id)
        unresolved_effects = await step_store.list_unresolved_tool_effects(run_id=step_run_id)
        if (
            record is None
            or record.conversation_id != conversation_id
            or record.metadata.get("segment_sequence") != str(segment_sequence)
            or record.metadata.get("agent_name") != definition.agent_name
            or snapshot is None
            or snapshot.run_id != step_run_id
            or snapshot.conversation_id != conversation_id
            or unresolved_effects
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = final_result.all_messages()
        for message in final_result.new_messages():
            if isinstance(message, (ModelRequest, ModelResponse)) and (
                message.run_id != final_result.run_id
                or message.conversation_id != conversation_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = WorkspaceAgentResult(final_result.run_id, "".join(output_parts) or str(final_result.output), messages)
        self._logger.info("workspace agent segment completed: agent=%s run=%s", definition.agent_name, result.run_id)
        return result

    async def _resolve_definition(self, agent_id: "str | None") -> _ResolvedDefinition:
        selected = self._selected_agent_name(agent_id)
        instructions = await asyncio.to_thread(self._instructions, selected)
        model = self._model or _config_string(self._config, "model") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        model_identity = model if isinstance(model, str) else f"{type(model).__module__}.{type(model).__qualname__}"
        fingerprint: dict[str, JsonValue] = {
            "agent_name": selected,
            "instructions_digest": canonical_sha256(instructions),
            "model": model_identity,
            "provider_endpoint": _endpoint_identity(self._base_url or _config_string(self._config, "base_url")),
            "tools": [f"{type(tool).__module__}.{type(tool).__qualname__}" for tool in self._tools],
            "toolset_fingerprint": _config_string(self._config, "toolset_fingerprint") or "",
            "output_schema": _config_string(self._config, "output_schema") or "text",
            "output_schema_revision": _config_int(self._config, "output_schema_revision", 1),
            "workspace_config": _stable_config(self._config),
        }
        return _ResolvedDefinition(selected, instructions, model, fingerprint)

    async def _get_agent(self, definition: _ResolvedDefinition) -> "Agent[None, str]":
        cache_key = (definition.agent_name, canonical_sha256(definition.fingerprint))
        async with self._agent_lock:
            existing = self._agents.get(cache_key)
            if existing is not None:
                return existing
            model: "str | Model" = self._materialize_model(definition.model)
            test_model = isinstance(model, TestModel)
            selected_tools = self._tools if test_model else ()
            agent = Agent(
                model,
                name=definition.agent_name,
                instructions=definition.instructions,
                tools=selected_tools,
            )
            cached = cast("Agent[None, str]", agent)
            self._agents[cache_key] = cached
            if len(self._agents) > 128:
                self._agents.pop(next(iter(self._agents)))
            return cached

    def _materialize_model(self, model: "str | Model") -> "str | Model":
        if model == "test":
            return TestModel(call_tools=[])
        if isinstance(model, TestModel):
            call_tools = [] if model.call_tools == "all" else list(model.call_tools)
            return TestModel(
                call_tools=call_tools,
                custom_output_text=model.custom_output_text,
                custom_output_args=model.custom_output_args,
                seed=model.seed,
                model_name=model.model_name,
                profile=model.profile,
                settings=model.settings,
            )
        if not isinstance(model, str):
            return model
        return OpenAIChatModel(
            model.removeprefix("openai:"),
            provider=OpenAIProvider(
                base_url=self._base_url or _config_string(self._config, "base_url"),
                api_key=self._api_key,
            ),
        )

    def _instructions(self, agent_id: str) -> str:
        _validate_agent_id(agent_id)
        configured = self._config.get("instructions")
        return configured if isinstance(configured, str) else ""

    def _selected_agent_name(self, agent_id: "str | None") -> str:
        selected = agent_id or str(self._config.get("default_agent", "default"))
        _validate_agent_id(selected)
        return selected


class _SegmentAlreadyStarted(RuntimeError):
    def __init__(self, step_run_id: str) -> None:
        super().__init__(step_run_id)
        self.step_run_id = step_run_id


def _validate_agent_id(agent_id: str) -> None:
    normalized = unicodedata.normalize("NFC", agent_id)
    if normalized != agent_id or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", agent_id):
        raise AIError(ErrorCode.AGENT_ID_INVALID)
    if agent_id in {".", ".."} or "/" in agent_id or "\\" in agent_id or "\x00" in agent_id:
        raise AIError(ErrorCode.AGENT_ID_INVALID)
    if agent_id.startswith(".") or re.match(r"^[A-Za-z]:", agent_id) or "%2f" in agent_id.lower() or "%5c" in agent_id.lower():
        raise AIError(ErrorCode.AGENT_ID_INVALID)


def _config_string(config: Mapping[str, JsonValue], name: str) -> "str | None":
    value = config.get(name)
    return value if isinstance(value, str) else None


def _config_int(config: Mapping[str, JsonValue], name: str, default: int) -> int:
    value = config.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _context_target_tokens(config: Mapping[str, JsonValue]) -> "int | None":
    value = config.get("context_target_tokens")
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


def _stable_config(config: Mapping[str, JsonValue]) -> "dict[str, JsonValue]":
    return {
        key: value
        for key, value in config.items()
        if not any(secret in key.lower() for secret in ("key", "secret", "token", "password", "credential"))
    }


def _endpoint_identity(value: "str | None") -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return "invalid"
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or ''}{parsed.path}"


def _map_event(
    event: "PartStartEvent | PartDeltaEvent | PartEndEvent | FunctionToolCallEvent | FunctionToolResultEvent",
) -> "dict[str, JsonValue] | None":
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content:
        return {"type": "text", "text": event.part.content}
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
        return {"type": "text", "text": event.delta.content_delta}
    if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart) and event.part.content:
        return {"type": "text_end"}
    if isinstance(event, FunctionToolCallEvent):
        part = event.part
        return cast("dict[str, JsonValue]", {"type": "tool", "phase": "start", "id": part.tool_call_id, "name": part.tool_name, "arguments_digest": canonical_sha256(part.args_as_dict()), "operation_id": part.tool_call_id, "status": "STARTED", "truncated": False})
    if isinstance(event, FunctionToolResultEvent):
        part = event.part
        if isinstance(part, ToolReturnPart):
            success = part.outcome == "success"
            return cast("dict[str, JsonValue]", {"type": "tool", "phase": "end", "id": part.tool_call_id, "name": part.tool_name, "operation_id": part.tool_call_id, "result_digest": canonical_sha256(str(part.content)), "status": "SUCCEEDED" if success else "FAILED", "truncated": False, "safe_summary": "tool completed" if success else "tool failed"})
        if isinstance(part, RetryPromptPart):
            return cast("dict[str, JsonValue]", {"type": "tool", "phase": "end", "id": part.tool_call_id, "name": part.tool_name or "unknown", "operation_id": part.tool_call_id, "result_digest": canonical_sha256(str(part.content)), "status": "FAILED", "truncated": False, "safe_summary": "tool retry requested"})
    return None


__all__ = ["AgentRunner", "AgentTool", "EventHandler", "WorkspaceAgentResult", "WorkspaceAgentRunner"]
