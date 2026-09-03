#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned Pydantic AI execution driver."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, cast

from linktools.core import environ
from openai import (
    APIConnectionError as OpenAIAPIConnectionError,
    APIError as OpenAIAPIError,
    APIStatusError as OpenAIAPIStatusError,
    APITimeoutError as OpenAIAPITimeoutError,
)
from pydantic import ValidationError
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai import AgentRunResultEvent, ModelSettings, TextOutput, Tool
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ReinjectSystemPrompt,
    WrapperCapability,
)
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
    ToolCallPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    RunContext as PydanticRunContext,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, PreparedToolset
from pydantic_ai.usage import RunUsage, UsageLimitExceeded, UsageLimits
from pydantic_ai_harness.memory import SearchableMemoryStore
from pydantic_ai_harness.planning import PlanStore
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore

from ..agent import AgentBinding, AgentDefinition, AssistantTextOutput
from ..capability import (
    RunContext,
    SKILL_TOOL_NAMES,
    SkillCapability,
    SkillSourceRegistry,
    SubagentCapability,
    SubagentDelegate,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
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
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..observe import MiddlewarePipeline, context_for
if TYPE_CHECKING:
    from ..workspace import RepositoryInstructionResolver, RepositoryInstructions
from ._capabilities import (
    MEMORY_READ_TOOL_NAMES,
    MEMORY_TOOL_NAMES,
    PLANNING_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
    ToolOperationBridge,
    _ObservationalMiddlewareCapability,
    _WorkspaceToolGate,
    _tool_effect_policy,
    compose_platform_capabilities,
    select_runtime_tool_names,
    tool_allowed_in_planning,
    tool_is_control,
    tool_name_allowed,
)
from ._input import _RuntimeUserPrompt, _restore_user_prompt
from ._skill_adapter import _PydanticSkillCapability
from ._subagent_adapter import _PydanticSubagentCapability

_logger = environ.get_logger("ai.runtime.agent_executor")
_RUNTIME_RESERVED_TOOL_NAMES = frozenset(
    (*SKILL_TOOL_NAMES, *MEMORY_TOOL_NAMES, *PLANNING_TOOL_NAMES, *SUBAGENT_TOOL_NAMES)
)
_WORKSPACE_CAPABILITY_IDS = frozenset({"workspace-sandbox"})
_MAX_TOOL_RETRIES = sys.maxsize


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
class PendingToolApproval:
    tool_call_id: str
    tool_name: str
    arguments: JsonValue
    args_digest: str


@dataclass(frozen=True, slots=True)
class AgentExecutionPaused:
    run_id: str
    step_index: int
    paused_at: datetime
    messages: list[ModelMessage]
    usage: UsageMetrics
    approvals: tuple[PendingToolApproval, ...]


AgentExecutionOutcome = AgentExecutionResult | AgentExecutionPaused


@dataclass(frozen=True, slots=True)
class _RunScope:
    binding: AgentBinding
    context: RunContext[object]
    user_prompt: _RuntimeUserPrompt | None
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
    subagent_descriptions: Mapping[str, str | None] = field(default_factory=dict)
    subagent_delegate: SubagentDelegate | None = None
    event_sink: EventSink | None = None
    usage_sink: UsageSink | None = None
    tool_operations: ToolOperationBridge | None = None
    background_tasks: set[asyncio.Task[object]] = field(default_factory=set, compare=False)
    replace_history_system_prompt: bool = False
    context_target_tokens: int | None = None
    repository_instructions: RepositoryInstructions | None = None
    repository_instruction_history: tuple[ModelMessage, ...] = ()
    repository_instruction_marker_authority: frozenset[tuple[str, str]] = frozenset()
    deferred_tool_results: DeferredToolResults | None = None

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

    def __init__(
        self,
        skill_sources: SkillSourceRegistry,
        *,
        instruction_resolver: RepositoryInstructionResolver,
        middleware: MiddlewarePipeline,
    ) -> None:
        if not isinstance(skill_sources, SkillSourceRegistry):
            raise TypeError("skill_sources must be SkillSourceRegistry")
        if not isinstance(middleware, MiddlewarePipeline):
            raise TypeError("middleware must be MiddlewarePipeline")
        self._skill_sources = skill_sources
        self._instruction_resolver = instruction_resolver
        self._middleware = middleware
        self._detached_tasks: set[asyncio.Task[Any]] = set()

    @classmethod
    def pending_tool_calls(
        cls,
        messages: Sequence[ModelMessage],
        *,
        run_id: str,
    ) -> tuple[ToolCallPart, ...]:
        del cls
        if not isinstance(run_id, str) or not run_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        calls: list[ToolCallPart] = []
        call_ids: set[str] = set()
        terminal_ids: set[str] = set()
        for message in messages:
            if message.run_id != run_id:
                continue
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_id = part.tool_call_id
                    if not isinstance(tool_call_id, str) or not tool_call_id or tool_call_id in call_ids:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    call_ids.add(tool_call_id)
                    calls.append(part)
                elif isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    tool_call_id = part.tool_call_id
                    if tool_call_id is not None:
                        terminal_ids.add(tool_call_id)
        if not terminal_ids.issubset(call_ids):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return tuple(call for call in calls if call.tool_call_id not in terminal_ids)

    def trusted_tool_class(
        self,
        binding: AgentBinding,
        tool_name: str,
        *,
        memory_scope: str | None,
        planning: bool,
        subagent_available: bool,
    ) -> str | None:
        if not isinstance(tool_name, str) or not tool_name:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        runtime_tool_names = select_runtime_tool_names(
            ordinary_tool_policy=binding.definition.ordinary_tool_policy,
            memory_scope=memory_scope,
            planning=planning,
            subagent_available=(
                subagent_available and bool(binding.snapshot.subagents)
            ),
        )
        return dict(
            _trusted_tool_classes_for_definition(
                binding.definition,
                runtime_tool_names,
            )
        ).get(tool_name)

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        return tuple(task for task in self._detached_tasks if not task.done())

    async def execute(self, scope: _RunScope) -> AgentExecutionOutcome:
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
        result: AgentExecutionOutcome | None = None
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
    ) -> AgentExecutionOutcome:
        binding = scope.binding
        definition = binding.definition
        if await scope.step_store.get_run(run_id=scope.step_run_id) is not None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        model = definition.model.materialize()
        model_settings = _thinking_settings(model, scope.thinking)
        deferred_step_index: int | None = None

        def capture_deferred_step(step_index: int) -> None:
            nonlocal deferred_step_index
            if deferred_step_index is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            deferred_step_index = step_index

        agent, capabilities, runtime_tool_names, trusted_tool_classes, trusted_mcp_selectors = await _materialize_agent(
            scope,
            model=model,
            skill_sources=self._skill_sources,
            deferred_pause_sink=capture_deferred_step,
        )
        presentation = _ToolPresentation(
            definition.ordinary_tool_policy,
            static_tool_names=tuple(candidate.id for candidate in definition.selected_tools),
            mcp_policy=definition.mcp_selector_policy,
            plan_mode=scope.mode == "plan",
            trusted_tool_classes=trusted_tool_classes,
            trusted_mcp_selectors=trusted_mcp_selectors,
            instruction_aware=scope.repository_instructions is not None,
        )
        gate = _WorkspaceToolGate(
            execution_id=scope.context.execution_id,
            workspace_root=scope.context.workspace.root,
            repository_instruction_history=scope.repository_instruction_history,
            repository_instruction_marker_authority=scope.repository_instruction_marker_authority,
            repository_instructions=scope.repository_instructions,
            instruction_resolver=self._instruction_resolver,
            policy=scope.context.workspace.policy,
            trusted_tool_classes=trusted_tool_classes,
        )
        middleware = _ObservationalMiddlewareCapability(
            self._middleware,
            context_for(
                scope.context.principal,
                scope.context.execution_id,
                scope.context.session_id,
                scope.step_run_id,
                definition.spec.id,
            ),
        )
        capabilities = (presentation, gate, middleware, *capabilities)
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
        user_prompt = None if scope.user_prompt is None else _restore_user_prompt(scope.user_prompt)
        deferred_kwargs: dict[str, object] = {}
        if scope.deferred_tool_results is not None:
            deferred_kwargs["deferred_tool_results"] = scope.deferred_tool_results
        async with agent.run_stream_events(
            user_prompt,
            deps=scope.context,
            message_history=scope.history or None,
            conversation_id=scope.conversation_id,
            run_id=scope.step_run_id,
            usage_limits=usage_limits,
            usage=run_usage,
            capabilities=capabilities,
            model_settings=model_settings,
            **deferred_kwargs,
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
        output = final_result.output
        if isinstance(output, DeferredToolRequests):
            if not output.approvals or output.calls or output.metadata:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if deferred_step_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            messages = final_result.all_messages()
            pending = self.pending_tool_calls(messages, run_id=scope.step_run_id)
            if len(pending) != len(output.approvals):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if scope.tool_operations is not None:
                admitted_call_ids = await scope.tool_operations.existing_call_ids(
                    tuple(call.tool_call_id for call in pending)
                )
                if admitted_call_ids:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            approvals: list[PendingToolApproval] = []
            for call, requested in zip(pending, output.approvals, strict=True):
                try:
                    call_arguments = normalize_json_value(call.args_as_dict())
                    requested_arguments = normalize_json_value(requested.args_as_dict())
                except (TypeError, ValueError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                if (
                    call.tool_call_id != requested.tool_call_id
                    or call.tool_name != requested.tool_name
                    or call_arguments != requested_arguments
                ):
                    raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
                if await scope.step_store.get_tool_effect(
                    run_id=scope.step_run_id,
                    tool_call_id=call.tool_call_id,
                ) is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                approvals.append(
                    PendingToolApproval(
                        call.tool_call_id,
                        call.tool_name,
                        call_arguments,
                        canonical_sha256(call_arguments),
                    )
                )
            return AgentExecutionPaused(
                run_id=scope.step_run_id,
                step_index=deferred_step_index,
                paused_at=datetime.now(timezone.utc),
                messages=list(messages),
                usage=_usage_metrics(run_usage),
                approvals=tuple(approvals),
            )
        run = await scope.step_store.get_run(run_id=scope.step_run_id)
        snapshot = await scope.step_store.latest_snapshot(run_id=scope.step_run_id)
        unresolved = await scope.step_store.list_unresolved_tool_effects(run_id=scope.step_run_id)
        if run is None or snapshot is None or unresolved or run.conversation_id != scope.conversation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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


def _trusted_tool_classes_for_definition(
    definition: AgentDefinition,
    runtime_tool_names: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    trusted: dict[str, str] = {}
    for candidate in definition.selected_tools:
        tool = candidate.value
        if not isinstance(tool, Tool):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        tool_class = workspace_tool_class(tool)
        if tool_class is not None:
            trusted[candidate.id] = tool_class
    for name in runtime_tool_names:
        if name in MEMORY_TOOL_NAMES:
            trusted[name] = (
                "memory.read" if name in MEMORY_READ_TOOL_NAMES else "memory.write"
            )
        elif name in PLANNING_TOOL_NAMES or name in SUBAGENT_TOOL_NAMES:
            trusted[name] = "control"
        else:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    if definition.skill_definitions:
        for name in SKILL_TOOL_NAMES:
            trusted[name] = "control"
    return tuple(sorted(trusted.items()))


async def _materialize_agent(
    scope: _RunScope,
    *,
    model: Model,
    skill_sources: SkillSourceRegistry,
    deferred_pause_sink: Callable[[int], None],
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
    for candidate in definition.selected_tools:
        tool = candidate.value
        if not isinstance(tool, Tool):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if workspace_tool_class(tool) is None:
            business_tools.append(cast("Tool[RunContext[object]]", tool))
        else:
            workspace_names.append(candidate.id)

    runtime_tool_names = select_runtime_tool_names(
        ordinary_tool_policy=definition.ordinary_tool_policy,
        memory_scope=scope.memory_scope,
        planning=scope.planning,
        subagent_available=scope.subagent_available and bool(scope.binding.snapshot.subagents),
    )
    trusted_tool_classes = _trusted_tool_classes_for_definition(
        definition,
        runtime_tool_names,
    )
    trusted_mcp_selectors = tuple(
        sorted(mcp_server_selector(server.id) for server in definition.mcp_servers)
    )

    capabilities: list[AbstractCapability[RunContext[object]]] = []
    capabilities.extend(workspace_capabilities(scope.context.workspace, workspace_names))
    for candidate in definition.selected_capabilities:
        if not isinstance(candidate.value, AbstractCapability):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        capabilities.append(cast("AbstractCapability[RunContext[object]]", candidate.value))
    if definition.skill_definitions:
        capabilities.append(
            _PydanticSkillCapability(
                SkillCapability(definition.skill_definitions, skill_sources)
            )
        )
    if any(name in SUBAGENT_TOOL_NAMES for name in runtime_tool_names):
        if scope.subagent_delegate is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(
            _PydanticSubagentCapability(
                SubagentCapability(
                    scope.binding.snapshot.subagents,
                    scope.subagent_delegate,
                    scope.subagent_descriptions,
                )
            )
        )
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
        tool_operations=scope.tool_operations,
        background_tasks=scope.background_tasks,
        plan_store_resolver=scope.plan_store_resolver,
        deferred_pause_sink=deferred_pause_sink,
    )
    platform = tuple(
        _RuntimePersistenceBoundary(capability)
        if isinstance(capability, StepPersistence)
        else capability
        for capability in platform
    )
    capabilities.extend(cast("tuple[AbstractCapability[RunContext[object]], ...]", platform))

    business_output_type: object
    if scope.binding.output_binding.mode == "text":
        business_output_type = TextOutput(_assistant_text_output)
    else:
        business_output_type = scope.binding.output_type
    output_type: object = business_output_type
    if scope.context.workspace.policy.tool_permissions.requires_approval:
        output_type = [business_output_type, DeferredToolRequests]
    base_instructions = "\n".join(definition.spec.instructions)
    preload_instructions = _render_preloaded_skills(
        definition.preloaded_skill_definitions,
        max_bytes=scope.context.workspace.policy.max_preloaded_skill_bytes,
    )
    runtime_instructions = "\n\n".join(
        value for value in (base_instructions, preload_instructions) if value != ""
    )
    agent = cast(
        "PydanticAgent[RunContext[object], object]",
        PydanticAgent(
            model,
            name=definition.spec.id,
            system_prompt=definition.spec.system_prompt,
            instructions=runtime_instructions,
            output_type=output_type,
            deps_type=RunContext,
            retries={"tools": _MAX_TOOL_RETRIES},
            tools=tuple(business_tools),
        ),
    )
    return agent, tuple(capabilities), runtime_tool_names, trusted_tool_classes, trusted_mcp_selectors


def _render_preloaded_skills(
    definitions: Sequence[object],
    *,
    max_bytes: int,
) -> str:
    ordered = tuple(sorted(definitions, key=lambda value: value.spec.id))
    ids = tuple(definition.spec.id for definition in ordered)
    if len(ids) != len(set(ids)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    for skill_id in ids:
        if (
            not isinstance(skill_id, str)
            or not skill_id
            or any(character in skill_id for character in "\r\n[]")
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        try:
            skill_id.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if not ordered:
        return ""
    rendered = (
        "<preloaded-skills>\n"
        + "\n\n".join(
            f"[skill: {definition.spec.id}]\n{definition.spec.content}"
            for definition in ordered
        )
        + "\n</preloaded-skills>"
    )
    try:
        rendered_bytes = rendered.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if len(rendered_bytes) > max_bytes:
        raise AIError(ErrorCode.PROMPT_TOO_LARGE)
    return rendered


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


class _RuntimePersistenceBoundary(WrapperCapability[RunContext[object]]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            position="outermost",
            wraps=(AbstractCapability,),
        )


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
        instruction_aware: bool,
    ) -> None:
        self._ordinary_policy = ordinary_policy
        self._static_tool_names = static_tool_names
        self._mcp_policy = mcp_policy
        self._plan_mode = plan_mode
        self._trusted_tool_classes = trusted_tool_classes
        self._trusted_mcp_selectors = trusted_mcp_selectors
        self._instruction_aware = instruction_aware

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="outermost")

    def get_wrapper_toolset(
        self,
        toolset: AbstractToolset[RunContext[object]],
    ) -> AbstractToolset[RunContext[object]]:
        return PreparedToolset(toolset, self._prepare_final_tools)

    async def _prepare_final_tools(
        self,
        _ctx: PydanticRunContext[RunContext[object]],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        names = [tool.name for tool in tool_defs]
        if len(names) != len(set(names)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        self._validate_provenance_and_static_surface(tool_defs)
        trusted_classes = dict(self._trusted_tool_classes)
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
            if (
                self._instruction_aware
                and tool.name in WORKSPACE_FILESYSTEM_TOOL_NAMES
                and trusted_classes.get(tool.name) in {"filesystem.read", "filesystem.write"}
            ):
                tool = replace(tool, sequential=True)
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
                        ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                        safe_details={"tool_name": tool.name},
                    )

        if actual_static != expected_static:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                safe_details={
                    "expected_static_tools": sorted(expected_static),
                    "actual_static_tools": sorted(actual_static),
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


def _model_http_error_code(status_code: int) -> ErrorCode:
    if status_code == 408:
        return ErrorCode.MODEL_TIMEOUT
    if status_code == 429:
        return ErrorCode.MODEL_RATE_LIMITED
    if status_code >= 500:
        return ErrorCode.MODEL_UNAVAILABLE
    if 400 <= status_code < 500:
        return ErrorCode.MODEL_REQUEST_REJECTED
    return ErrorCode.MODEL_API_ERROR


def _execution_error(
    error: Exception,
    *,
    usage_limits: UsageLimits,
    run_usage: RunUsage,
) -> AIError:
    diagnostics = ErrorDiagnostics.from_exception(error)
    if isinstance(error, UsageLimitExceeded):
        return AIError(
            ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED,
            retryable=False,
            safe_details={
                "limits": _limit_details(usage_limits),
                "usage": _usage_details(run_usage),
            },
            diagnostics=diagnostics,
        )
    if isinstance(error, RunCancelled):
        return AIError(
            ErrorCode.EXECUTION_CANCELLED,
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(error, ConcurrencyLimitExceeded):
        return AIError(
            ErrorCode.EXECUTION_CONCURRENCY_LIMIT_EXCEEDED,
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(error, ContentFilterError):
        return AIError(
            ErrorCode.MODEL_CONTENT_FILTERED,
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(error, ModelHTTPError):
        details: dict[str, JsonValue] = {
            "model_name": error.model_name,
            "status_code": error.status_code,
        }
        retry_after = error.retry_after
        if isinstance(retry_after, (int, float, str)) and not isinstance(retry_after, bool):
            details["retry_after"] = retry_after
        return AIError(
            _model_http_error_code(error.status_code),
            safe_details=details,
            diagnostics=diagnostics,
        )
    if isinstance(error, ModelAPIError):
        return AIError(
            ErrorCode.MODEL_API_ERROR,
            retryable=False,
            safe_details={"model_name": error.model_name},
            diagnostics=diagnostics,
        )
    if isinstance(error, OpenAIAPITimeoutError):
        return AIError(
            ErrorCode.MODEL_TIMEOUT,
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(error, OpenAIAPIConnectionError):
        return AIError(
            ErrorCode.MODEL_UNAVAILABLE,
            retryable=True,
            diagnostics=diagnostics,
        )
    if isinstance(error, OpenAIAPIStatusError):
        return AIError(
            _model_http_error_code(error.status_code),
            safe_details={"status_code": error.status_code},
            diagnostics=diagnostics,
        )
    if isinstance(error, OpenAIAPIError):
        return AIError(
            ErrorCode.MODEL_API_ERROR,
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(error, UnexpectedModelBehavior):
        return AIError(
            ErrorCode.MODEL_RESPONSE_INVALID,
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(error, ValidationError):
        return AIError(
            ErrorCode.OUTPUT_VALIDATION_FAILED,
            retryable=False,
            diagnostics=diagnostics,
        )
    if isinstance(error, UserError):
        return AIError(
            ErrorCode.INTERNAL_ERROR,
            retryable=False,
            safe_details={"phase": "agent_execution"},
            diagnostics=diagnostics,
        )
    return AIError(
        ErrorCode.INTERNAL_ERROR,
        retryable=False,
        safe_details={"phase": "agent_execution"},
        diagnostics=diagnostics,
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
    "AgentExecutionOutcome",
    "AgentExecutionPaused",
    "AgentExecutionResult",
    "AgentExecutor",
    "PendingToolApproval",
    "DurableBoundary",
    "EventSink",
    "LiveDelta",
    "UsageSink",
]
