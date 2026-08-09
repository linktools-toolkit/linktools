#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI and Harness boundary for workspace execution."""

import asyncio
import os
import re
import unicodedata
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

from linktools.core import environ
from pydantic import BaseModel
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
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.step_persistence import StepStore

from ..capability import (
    MCPToolProvider,
    SkillCatalogSnapshot,
    SkillCatalogView,
    SkillDescriptor,
)
from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelRoute
from ._binding import BindingExecutionPlan
from ._capabilities import (
    AgentCapabilityScope,
    AgentCatalogView,
    EmptyAgentCatalog,
    EmptySkillCatalog,
    compose_parent_capabilities,
)


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


class ModelMaterializer(Protocol):
    def materialize(self, route: ModelRoute) -> Model: ...


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


class BindingAgentRunner:
    """Execute a frozen binding through the same Harness segment pipeline."""

    def __init__(
        self,
        plan: BindingExecutionPlan,
        materializer: ModelMaterializer,
        *,
        materialized_model: "Model | None" = None,
        agent_catalog: "AgentCatalogView | None" = None,
        mcp_provider: "MCPToolProvider | None" = None,
        capabilities: "tuple[AgentCapability[None], ...]" = (),
        memory_store: "SearchableMemoryStore | None" = None,
    ) -> None:
        self._plan = plan
        self._materializer = materializer
        self._model = materialized_model
        self._agent_catalog = agent_catalog if agent_catalog is not None else EmptyAgentCatalog()
        self._mcp_provider = mcp_provider
        self._capabilities = capabilities
        self._memory_store = memory_store
        self._agent: "Agent[None, object] | None" = None
        self._lock = asyncio.Lock()
        self._logger = environ.get_logger("ai.agent.binding")

    async def run(
        self,
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
        toolsets: "Sequence[AbstractToolset[None]]" = (),
    ) -> WorkspaceAgentResult:
        existing = await step_store.get_run(run_id=step_run_id)
        if existing is not None:
            raise _SegmentAlreadyStarted(step_run_id)
        if self._plan.mcp_servers and self._mcp_provider is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        _validate_toolsets(toolsets, required=bool(self._plan.mcp_servers))
        agent = await self._get_agent()
        model = self._model
        if model is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        skill_catalog = SkillCatalogSnapshot(
            tuple(SkillDescriptor(item.id, item.revision, "") for item in self._plan.skills),
            self._plan.skills,
        )
        scope = AgentCapabilityScope(
            root=Path.cwd(),
            agent_name=self._plan.binding.spec.id,
            conversation_id=conversation_id,
            step_run_id=step_run_id,
            segment_sequence=segment_sequence,
            memory_namespace=memory_namespace,
            step_store=step_store,
            workspace_capabilities=self._capabilities,
            skill_catalog=skill_catalog,
            agent_catalog=self._agent_catalog,
            memory_store=memory_store if memory_store is not None else self._memory_store,
            parent_step_run_id=parent_step_run_id,
        )
        composed = await compose_parent_capabilities(scope, model_factory=lambda _value: model, parent_model=model)
        self._logger.info("binding agent segment started: agent=%s execution_step=%s binding=%s", self._plan.binding.spec.id, step_run_id, self._plan.binding.digest)
        final_result = None
        output_parts: list[str] = []
        async with agent.run_stream_events(
            prompt,
            message_history=history or None,
            conversation_id=conversation_id,
            run_id=step_run_id,
            capabilities=composed,
            toolsets=tuple(toolsets),
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
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record = await step_store.get_run(run_id=step_run_id)
        snapshot = await step_store.latest_snapshot(run_id=step_run_id)
        unresolved_effects = await step_store.list_unresolved_tool_effects(run_id=step_run_id)
        if record is None or snapshot is None or unresolved_effects or record.conversation_id != conversation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = final_result.output
        if not isinstance(output, BaseModel):
            raise AIError(ErrorCode.OUTPUT_VALIDATION_FAILED)
        encoded = canonical_json_bytes(output.model_dump(mode="json")).decode("utf-8")
        self._logger.info("binding agent segment completed: agent=%s run=%s", self._plan.binding.spec.id, final_result.run_id)
        return WorkspaceAgentResult(final_result.run_id, encoded or "null", final_result.all_messages())

    async def _get_agent(self) -> "Agent[None, object]":
        async with self._lock:
            if self._agent is not None:
                return self._agent
            model = self._model or self._materializer.materialize(self._plan.model_route)
            self._model = model
            agent = Agent(
                model,
                name=self._plan.binding.spec.id,
                system_prompt=self._plan.binding.prompt.system,
                instructions="\n".join((*self._plan.binding.spec.instructions, *self._plan.binding.prompt.instructions)),
                output_type=self._plan.output_type,
            )
            self._agent = cast("Agent[None, object]", agent)
            return self._agent


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


def _validate_toolsets(toolsets: Sequence[AbstractToolset[None]], *, required: bool) -> None:
    if required and not toolsets:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "MCP provider returned no run-scoped toolsets")
    ids = [toolset.id for toolset in toolsets]
    if any(not isinstance(toolset_id, str) or not toolset_id.strip() for toolset_id in ids):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "MCP toolset id is required")
    if len(set(ids)) != len(ids):
        raise AIError(ErrorCode.STORAGE_CONFLICT, "MCP toolset ids must be unique per run")


__all__ = ["AgentRunner", "AgentTool", "BindingAgentRunner", "EventHandler", "ModelMaterializer", "WorkspaceAgentResult", "WorkspaceAgentRunner"]
