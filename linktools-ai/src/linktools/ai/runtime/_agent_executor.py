#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned Pydantic AI execution driver."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from linktools.core import environ
from openai import APIError as OpenAIAPIError
from pydantic import ValidationError
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import AgentRunResultEvent, ModelSettings, TextOutput, Tool
from pydantic_ai.capabilities import AbstractCapability, ReinjectSystemPrompt
from pydantic_ai.exceptions import (
    ConcurrencyLimitExceeded,
    ContentFilterError,
    ModelAPIError,
    ModelHTTPError,
    RunCancelled,
    UnexpectedModelBehavior,
    UserError,
)
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
from pydantic_ai.tools import RunContext as PydanticRunContext, ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimitExceeded, UsageLimits
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.planning import PlanStore
from pydantic_ai_harness.step_persistence import StepStore

from ..agent import AgentBinding, AgentDefinition, AssistantTextOutput
from ..capability import (
    RunContext,
    SKILL_TOOL_NAMES,
    SkillCapability,
    materialize_mcp_servers,
    mcp_selector_server,
    mcp_server_selector,
    workspace_capabilities,
    workspace_tool_class,
)
from ..core import (
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionMode,
    JsonValue,
    ResourceKind,
    ResourceRef,
    ThinkingValue,
    UsageMetrics,
    canonical_sha256,
    normalize_json_value,
)
from ..errors import AIError, ErrorCode
from ._capabilities import (
    MEMORY_READ_TOOL_NAMES,
    MEMORY_TOOL_NAMES,
    PLANNING_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
    SubagentDelegate,
    ToolOperationBridge,
    _tool_effect_policy,
    compose_platform_capabilities,
    select_runtime_tool_names,
    tool_allowed_in_planning,
    tool_is_control,
    tool_name_allowed,
)
from ._input import _RuntimeUserPrompt, _restore_user_prompt

_logger = environ.get_logger("ai.runtime.agent_executor")
_RUNTIME_RESERVED_TOOL_NAMES = frozenset(
    (*SKILL_TOOL_NAMES, *MEMORY_TOOL_NAMES, *PLANNING_TOOL_NAMES, *SUBAGENT_TOOL_NAMES)
)
_WORKSPACE_CAPABILITY_IDS = frozenset({"workspace-filesystem", "workspace-shell"})


@dataclass(frozen=True, slots=True)
class LiveDelta:
    kind: ExecutionDeltaType
    content: str


@dataclass(frozen=True, slots=True)
class DurableBoundary:
    kind: ExecutionEventType
    payload: JsonValue


AgentEmission = LiveDelta | DurableBoundary
EventSink = Callable[[AgentEmission], Awaitable[None]]


class UsageSink(Protocol):
    async def __call__(self, usage: UsageMetrics) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    run_id: str
    output: JsonValue
    messages: list[ModelMessage]
    usage: UsageMetrics


@dataclass(frozen=True, slots=True)
class _RunScope:
    binding: AgentBinding
    context: RunContext[object]
    user_prompt: _RuntimeUserPrompt
    history: list[ModelMessage]
    conversation_id: str
    step_store: StepStore
    step_run_id: str
    segment_sequence: int
    history_id: str | None = None
    memory_scope: str | None = None
    memory_store: SearchableMemoryStore | None = None
    plan_store_resolver: Callable[[PydanticRunContext[object]], PlanStore] | None = None
    mode: ExecutionMode = "run"
    planning: bool = False
    thinking: ThinkingValue = False
    parent_step_run_id: str | None = None
    subagent_available: bool = False
    subagent_delegate: SubagentDelegate | None = None
    event_sink: EventSink | None = None
    usage_sink: UsageSink | None = None
    tool_operations: ToolOperationBridge | None = None
    background_tasks: set[asyncio.Task[object]] = field(default_factory=set, compare=False)
    replace_history_system_prompt: bool = False
    context_target_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"run", "plan"} or not isinstance(self.planning, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.mode == "plan" and not self.planning:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(self.subagent_available, bool) or not isinstance(self.replace_history_system_prompt, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.event_sink is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


class AgentExecutor:
    """Execute one exact Agent binding inside a Runtime-owned execution scope."""

    def __init__(self) -> None:
        self._detached_tasks: set[asyncio.Task[Any]] = set()

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(task for task in self._detached_tasks if not task.done())

    async def execute(self, scope: _RunScope) -> AgentExecutionResult:
        binding = scope.binding
        definition = binding.definition
        run_usage = RunUsage()
        configured_limits = definition.spec.usage_limits
        usage_limits = UsageLimits(
            cost_limit=None,
            request_limit=None if configured_limits is None else configured_limits.model_requests,
            tool_calls_limit=None if configured_limits is None else configured_limits.tool_calls,
            input_tokens_limit=None if configured_limits is None else configured_limits.input_tokens,
            output_tokens_limit=None if configured_limits is None else configured_limits.output_tokens,
            total_tokens_limit=None if configured_limits is None else configured_limits.total_tokens,
        )
        result: AgentExecutionResult | None = None
        primary_error: BaseException | None = None
        try:
            try:
                result = await self._execute(scope, run_usage=run_usage, usage_limits=usage_limits)
                return result
            except asyncio.CancelledError as error:
                primary_error = error
                raise
            except AIError as error:
                primary_error = error
                raise
            except Exception as error:
                mapped = _execution_error(error, usage_limits=usage_limits, run_usage=run_usage)
                primary_error = mapped
                raise mapped from error
        finally:
            if scope.usage_sink is not None:
                usage = result.usage if result is not None else _usage_metrics(run_usage)
                if isinstance(primary_error, asyncio.CancelledError):
                    task = asyncio.create_task(
                        scope.usage_sink(usage),
                        name=f"agent-usage-{scope.step_run_id}",
                    )
                    self._detach_task(task, scope.step_run_id, scope.background_tasks)
                else:
                    try:
                        await scope.usage_sink(usage)
                    except Exception:
                        if primary_error is None:
                            raise
                        _logger.error(
                            "usage sink failed after agent execution failure: step=%s",
                            scope.step_run_id,
                            exc_info=False,
                        )

    def _detach_task(
        self,
        task: asyncio.Task[Any],
        step_run_id: str,
        background_tasks: set[asyncio.Task[object]],
    ) -> None:
        background_tasks.add(cast("asyncio.Task[object]", task))
        self._detached_tasks.add(task)

        def consume(done: asyncio.Task[Any]) -> None:
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except BaseException:
                _logger.exception("detached usage sink failed: step=%s", step_run_id)
            finally:
                background_tasks.discard(cast("asyncio.Task[object]", done))
                self._detached_tasks.discard(done)

        task.add_done_callback(consume)

    async def _execute(
        self,
        scope: _RunScope,
        *,
        run_usage: RunUsage,
        usage_limits: UsageLimits,
    ) -> AgentExecutionResult:
        binding = scope.binding
        definition = binding.definition
        if await scope.step_store.get_run(run_id=scope.step_run_id) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        model = definition.model.materialize()
        model_settings = _thinking_settings(model, scope.thinking)
        agent, capabilities, runtime_tool_names, trusted_tool_classes, trusted_mcp_selectors = await _materialize_agent(
            scope,
            model=model,
        )
        capabilities = (
            *capabilities,
            _ToolPresentation(
                definition.ordinary_tool_policy,
                static_tool_names=tuple(candidate.id for candidate in definition.selected_tools),
                mcp_policy=definition.mcp_selector_policy,
                plan_mode=scope.mode == "plan",
                trusted_tool_classes=trusted_tool_classes,
                trusted_mcp_selectors=trusted_mcp_selectors,
            ),
        )
        if scope.replace_history_system_prompt:
            capabilities = (*capabilities, ReinjectSystemPrompt(replace_existing=True))
        _logger.debug(
            "agent execution started: agent=%s definition=%s step=%s mode=%s planning=%s thinking=%s runtime_tools=%s",
            definition.spec.id,
            definition.digest,
            scope.step_run_id,
            scope.mode,
            scope.planning,
            scope.thinking,
            runtime_tool_names,
        )
        final_result = None
        async with agent.run_stream_events(
            _restore_user_prompt(scope.user_prompt),
            deps=scope.context,
            message_history=scope.history or None,
            conversation_id=scope.conversation_id,
            run_id=scope.step_run_id,
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
                    await cast(EventSink, scope.event_sink)(emission)
        if final_result is None:
            raise AIError(ErrorCode.INTERNAL_ERROR, safe_details={"phase": "agent_result"})
        run = await scope.step_store.get_run(run_id=scope.step_run_id)
        snapshot = await scope.step_store.latest_snapshot(run_id=scope.step_run_id)
        unresolved = await scope.step_store.list_unresolved_tool_effects(run_id=scope.step_run_id)
        if run is None or snapshot is None or unresolved or run.conversation_id != scope.conversation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = final_result.output
        if binding.output_binding.mode == "text":
            if not isinstance(output, AssistantTextOutput):
                raise AIError(ErrorCode.OUTPUT_VALIDATION_FAILED)
            output_payload: object = output.model_dump(mode="json")
        elif isinstance(output, Mapping):
            output_payload = dict(output)
        else:
            raise AIError(ErrorCode.OUTPUT_VALIDATION_FAILED)
        try:
            payload = normalize_json_value(output_payload)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.OUTPUT_VALIDATION_FAILED, retryable=False) from error
        usage = _usage_metrics(run_usage)
        return AgentExecutionResult(final_result.run_id, payload, final_result.all_messages(), usage)


async def _materialize_agent(
    scope: _RunScope,
    *,
    model: Model,
) -> tuple[
    PydanticAgent[RunContext[object], object],
    tuple[AbstractCapability[RunContext[object]], ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[str, ...],
]:
    definition = scope.binding.definition
    business_tools: list[Tool[RunContext[object]]] = []
    workspace_names: list[str] = []
    trusted: dict[str, str] = {}
    for candidate in definition.selected_tools:
        tool = candidate.value
        if not isinstance(tool, Tool):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        tool_class = workspace_tool_class(tool)
        if tool_class is None:
            business_tools.append(cast("Tool[RunContext[object]]", tool))
        else:
            workspace_names.append(candidate.id)
            trusted[candidate.id] = tool_class

    runtime_tool_names = select_runtime_tool_names(
        ordinary_tool_policy=definition.ordinary_tool_policy,
        memory_scope=scope.memory_scope,
        planning=scope.planning,
        subagent_available=scope.subagent_available and bool(definition.selected_subagents),
    )
    for name in runtime_tool_names:
        if name in MEMORY_TOOL_NAMES:
            trusted[name] = "memory.read" if name in MEMORY_READ_TOOL_NAMES else "memory.write"
        elif name in PLANNING_TOOL_NAMES or name in SUBAGENT_TOOL_NAMES:
            trusted[name] = "control"
        else:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    if definition.skill_specs:
        for name in SKILL_TOOL_NAMES:
            trusted[name] = "control"
    trusted_tool_classes = tuple(sorted(trusted.items()))
    trusted_mcp_selectors = tuple(
        sorted(mcp_server_selector(server.id) for server in definition.mcp_servers)
    )

    capabilities: list[AbstractCapability[RunContext[object]]] = []
    capabilities.extend(workspace_capabilities(scope.context.workspace, workspace_names))
    for candidate in definition.selected_capabilities:
        if not isinstance(candidate.value, AbstractCapability):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        capabilities.append(cast("AbstractCapability[RunContext[object]]", candidate.value))
    if definition.skill_specs:
        capabilities.append(SkillCapability(definition.skill_specs))
    if definition.mcp_servers:
        capabilities.extend(
            cast(
                "tuple[AbstractCapability[RunContext[object]], ...]",
                await materialize_mcp_servers(
                    definition.mcp_servers,
                    definition.mcp_selector_policy,
                    principal=scope.context.principal,
                    execution=ResourceRef(
                        ResourceKind.EXECUTION,
                        scope.context.execution_id,
                        scope.context.principal.tenant_id,
                    ),
                    execution_root=str(scope.context.workspace.root),
                ),
            )
        )
    platform = await compose_platform_capabilities(
        agent_name=definition.spec.id,
        conversation_id=scope.conversation_id,
        step_run_id=scope.step_run_id,
        segment_sequence=scope.segment_sequence,
        history_id=scope.history_id,
        memory_scope=scope.memory_scope,
        step_store=scope.step_store,
        memory_store=scope.memory_store,
        runtime_tool_names=runtime_tool_names,
        plan_mode=scope.mode == "plan",
        trusted_tool_classes=trusted_tool_classes,
        trusted_mcp_selectors=trusted_mcp_selectors,
        context_target_tokens=scope.context_target_tokens,
        parent_step_run_id=scope.parent_step_run_id,
        subagent_delegate=scope.subagent_delegate,
        tool_operations=scope.tool_operations,
        background_tasks=scope.background_tasks,
        plan_store_resolver=scope.plan_store_resolver,
    )
    capabilities.extend(cast("tuple[AbstractCapability[RunContext[object]], ...]", platform))

    output_type: object
    if scope.binding.output_binding.mode == "text":
        output_type = TextOutput(_assistant_text_output)
    else:
        output_type = scope.binding.output_type
    agent = cast(
        "PydanticAgent[RunContext[object], object]",
        PydanticAgent(
            model,
            name=definition.spec.id,
            system_prompt=definition.spec.system_prompt,
            instructions="\n".join(definition.spec.instructions),
            output_type=output_type,
            deps_type=RunContext,
            tools=tuple(business_tools),
        ),
    )
    return agent, tuple(capabilities), runtime_tool_names, trusted_tool_classes, trusted_mcp_selectors


def _assistant_text_output(value: str) -> AssistantTextOutput:
    return AssistantTextOutput(text=value)


def _thinking_settings(model: Model, thinking: ThinkingValue) -> ModelSettings:
    profile = model.profile
    supports = bool(profile.get("supports_thinking", False))
    always = bool(profile.get("thinking_always_enabled", False))
    if thinking is False:
        if always:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "thinking", "reason": "model_always_enabled"},
            )
        return ModelSettings(thinking=False)
    if not supports:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={"field": "thinking", "reason": "model_not_supported"},
        )
    return ModelSettings(thinking=thinking)


class _ToolPresentation(AbstractCapability[RunContext[object]]):
    def __init__(
        self,
        ordinary_policy: tuple[str, ...],
        *,
        static_tool_names: tuple[str, ...],
        mcp_policy: tuple[str, ...],
        plan_mode: bool,
        trusted_tool_classes: tuple[tuple[str, str], ...],
        trusted_mcp_selectors: tuple[str, ...],
    ) -> None:
        self._ordinary_policy = ordinary_policy
        self._static_tool_names = static_tool_names
        self._mcp_policy = mcp_policy
        self._plan_mode = plan_mode
        self._trusted_tool_classes = trusted_tool_classes
        self._trusted_mcp_selectors = trusted_mcp_selectors

    async def prepare_tools(
        self,
        _ctx: PydanticRunContext[RunContext[object]],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        names = [tool.name for tool in tool_defs]
        if len(names) != len(set(names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._validate_provenance_and_static_surface(tool_defs)
        mcp_names = {
            tool.name
            for tool in tool_defs
            if tool.capability_id in self._trusted_mcp_selectors
        }
        for selector in self._mcp_policy:
            parsed = mcp_selector_server(selector)
            if parsed is None:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            _namespace, exact_tool = parsed
            if exact_tool is not None and selector not in mcp_names:
                raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        selected: list[ToolDefinition] = []
        for tool in tool_defs:
            if not tool_is_control(tool, trusted_tool_classes=self._trusted_tool_classes):
                if tool.name.startswith("mcp__"):
                    if not _mcp_tool_allowed(tool.name, self._mcp_policy):
                        continue
                elif not tool_name_allowed(tool.name, self._ordinary_policy):
                    continue
            if self._plan_mode and not tool_allowed_in_planning(
                tool,
                trusted_tool_classes=self._trusted_tool_classes,
                trusted_mcp_selectors=self._trusted_mcp_selectors,
            ):
                continue
            selected.append(tool)
        return selected

    def _validate_provenance_and_static_surface(
        self,
        tool_defs: list[ToolDefinition],
    ) -> None:
        trusted_classes = dict(self._trusted_tool_classes)
        expected_static = frozenset(self._static_tool_names)
        actual_static: set[str] = set()
        for tool in tool_defs:
            tool_class = trusted_classes.get(tool.name)
            if tool_class is not None:
                _tool_effect_policy(
                    tool,
                    trusted_tool_classes=self._trusted_tool_classes,
                )
                if tool_class in {"filesystem.read", "filesystem.write", "shell"}:
                    actual_static.add(tool.name)
            elif tool.name in _RUNTIME_RESERVED_TOOL_NAMES:
                raise AIError(
                    ErrorCode.CAPABILITY_POLICY_CONFLICT,
                    safe_details={"tool_name": tool.name},
                )

            if tool.capability_id is None:
                actual_static.add(tool.name)
            elif tool.capability_id in _WORKSPACE_CAPABILITY_IDS:
                actual_static.add(tool.name)
                if tool.name not in trusted_classes:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

            is_mcp_name = tool.name.startswith("mcp__")
            is_mcp_owner = tool.capability_id in self._trusted_mcp_selectors
            if is_mcp_name or is_mcp_owner:
                if (
                    not is_mcp_owner
                    or tool.capability_id is None
                    or not tool.name.startswith(f"{tool.capability_id}__")
                ):
                    raise AIError(
                        ErrorCode.CAPABILITY_POLICY_CONFLICT,
                        safe_details={"tool_name": tool.name},
                    )

        if actual_static != expected_static:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                safe_details={
                    "expected_static_tools": tuple(sorted(expected_static)),
                    "actual_static_tools": tuple(sorted(actual_static)),
                },
            )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None


def _mcp_tool_allowed(name: str, selectors: tuple[str, ...]) -> bool:
    if not name.startswith("mcp__"):
        return False
    for selector in selectors:
        if selector.endswith("__*") and name.startswith(selector[:-1]):
            return True
        if "__" not in selector[5:] and name.startswith(f"{selector}__"):
            return True
        if selector == name:
            return True
    return False


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
                    "safe_error_code": ErrorCode.TOOL_RETRY_REQUIRED.value,
                },
            )
    return None


def _execution_error(
    error: Exception,
    *,
    usage_limits: UsageLimits,
    run_usage: RunUsage,
) -> AIError:
    if isinstance(error, UsageLimitExceeded):
        return AIError(
            ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED,
            retryable=False,
            safe_details={
                "limits": _limit_details(usage_limits),
                "usage": _usage_details(run_usage),
            },
        )
    if isinstance(error, RunCancelled):
        return AIError(ErrorCode.EXECUTION_CANCELLED, retryable=False)
    if isinstance(error, ConcurrencyLimitExceeded):
        return AIError(
            ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED,
            retryable=True,
        )
    if isinstance(error, ContentFilterError):
        return AIError(ErrorCode.MODEL_CONTENT_FILTERED, retryable=False)
    if isinstance(error, ModelHTTPError):
        details: dict[str, JsonValue] = {
            "model_name": error.model_name,
            "status_code": error.status_code,
        }
        retry_after = error.retry_after
        if isinstance(retry_after, (int, float, str)) and not isinstance(retry_after, bool):
            details["retry_after"] = retry_after
        if error.status_code == 408:
            code = ErrorCode.MODEL_TIMEOUT
        elif error.status_code == 429:
            code = ErrorCode.MODEL_RATE_LIMITED
        elif error.status_code >= 500:
            code = ErrorCode.MODEL_UNAVAILABLE
        elif 400 <= error.status_code < 500:
            code = ErrorCode.MODEL_REQUEST_REJECTED
        else:
            code = ErrorCode.MODEL_API_ERROR
        return AIError(code, safe_details=details)
    if isinstance(error, ModelAPIError):
        return AIError(
            ErrorCode.MODEL_API_ERROR,
            retryable=False,
            safe_details={"model_name": error.model_name},
        )
    if isinstance(error, OpenAIAPIError):
        return AIError(ErrorCode.MODEL_API_ERROR, retryable=False)
    if isinstance(error, UnexpectedModelBehavior):
        return AIError(ErrorCode.MODEL_RESPONSE_INVALID, retryable=False)
    if isinstance(error, ValidationError):
        return AIError(ErrorCode.OUTPUT_VALIDATION_FAILED, retryable=False)
    if isinstance(error, UserError):
        return AIError(
            ErrorCode.INTERNAL_ERROR,
            retryable=False,
            safe_details={"phase": "agent_execution"},
        )
    return AIError(
        ErrorCode.INTERNAL_ERROR,
        retryable=False,
        safe_details={"phase": "agent_execution"},
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
    "AgentEmission",
    "AgentExecutionResult",
    "AgentExecutor",
    "DurableBoundary",
    "EventSink",
    "LiveDelta",
    "UsageSink",
]
