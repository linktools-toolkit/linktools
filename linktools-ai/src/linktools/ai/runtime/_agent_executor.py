#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned Pydantic AI execution driver."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import (
    AsyncIterable,
    Awaitable,
    Callable,
    Mapping,
)
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic_ns
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
from pydantic_ai import TextOutput, Tool
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    PrepareTools,
    ProcessEventStream,
    ReinjectSystemPrompt,
    Thinking,
    WrapModelRequestHandler,
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
    AgentStreamEvent,
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
from pydantic_ai.models import Model, ModelRequestContext, ModelResponse
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    RunContext as PydanticRunContext,
    ToolDefinition,
)
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimitExceeded, UsageLimits

from ..agent import AgentBinding, AgentDefinition, AssistantTextOutput
from ..capability import (
    AgentContext,
    CapabilityContribution,
    LinkToolsSkills,
    LinkToolsSubagents,
    SkillSourceRegistry,
    SubagentDelegate,
    tool_compaction_keep_result_from_metadata,
    tool_class_from_metadata,
    tool_context_dedupe_from_metadata,
    tool_plan_safe_from_metadata,
    validate_tool_semantic_metadata,
    workspace_capabilities,
)
from ..core import (
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionMode,
    JsonValue,
    ResourceKind,
    ResourceRef,
    ThinkingValue,
    ToolOperationStatus,
    UsageMetrics,
    canonical_sha256,
    normalize_json_value,
)
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..observe import MetricMeasurement, MetricRecorder, Observation
from ..workspace import LocalSandbox, SandboxResource, SandboxSession

if TYPE_CHECKING:
    from ..workspace import RepositoryInstructions

from ._capabilities import (
    compose_platform_capabilities,
)
from ._compaction import RuntimeCompactionPolicy
from ._input import CanonicalUserInput
from ._journal import ModelRequestJournal
from ._mcp import materialize_mcp_servers
from ._memory import MemoryStore
from ._metric_capability import RuntimeModelObservationCapability
from ._plan import RuntimePlanStore
from ._tool import ToolOperationBridge
from ._tool_boundary import (
    ManagedToolDescriptor,
    RepositoryInstructionBoundary,
    RuntimeToolBoundaryToolset,
    managed_tool_descriptor_from_metadata,
)
from ._tool_metrics import _ToolMetricContext
from ._tool_return_codec import (
    rehydrate_deferred_tool_results,
    tool_return_content_digest,
)
from .state._step_contracts import StepStore

_logger = environ.get_logger("ai.runtime.agent_executor")
_SECONDARY_ERROR_CODE_KEY = "secondary_error_code"
_PLAN_SAFE_FRAMEWORK_TOOL_KINDS = frozenset({"capability-load", "tool-search"})


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
    def __call__(self, usage: UsageMetrics) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    run_id: str
    output: JsonValue
    messages: list[ModelMessage]
    usage: UsageMetrics


AgentExecutionOutcome = AgentExecutionResult | DeferredToolRequests


@dataclass(frozen=True, slots=True)
class _RunScope:
    binding: AgentBinding
    context: AgentContext[object]
    user_prompt: CanonicalUserInput | None
    history: list[ModelMessage]
    conversation_id: str
    step_store: StepStore
    step_run_id: str
    segment_sequence: int
    history_id: str | None = None
    memory_store: MemoryStore | None = None
    plan_store_resolver: (
        Callable[[PydanticRunContext[object]], RuntimePlanStore] | None
    ) = None
    sandbox_session: "SandboxSession | None" = None
    skill_resource_paths: Mapping[str, str] = field(default_factory=dict)
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
    background_tasks: set[asyncio.Task[object]] = field(
        default_factory=set, compare=False
    )
    replace_history_system_prompt: bool = False
    context_target_tokens: int | None = None
    repository_instructions: RepositoryInstructions | None = None
    repository_instruction_boundary: RepositoryInstructionBoundary | None = None
    deferred_tool_results: DeferredToolResults | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"run", "plan"} or not isinstance(self.planning, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.mode == "plan" and not self.planning:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(self.subagent_available, bool) or not isinstance(
            self.replace_history_system_prompt, bool
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.event_sink is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


class AgentExecutor:
    """Execute one exact Agent binding inside a Runtime-owned execution scope."""

    def __init__(
        self,
        skill_sources: SkillSourceRegistry,
        *,
        metrics: MetricRecorder | None = None,
    ) -> None:
        if not isinstance(skill_sources, SkillSourceRegistry):
            raise TypeError("skill_sources must be SkillSourceRegistry")
        self._skill_sources = skill_sources
        self._metrics = metrics

    async def execute(self, scope: _RunScope) -> AgentExecutionOutcome:
        binding = scope.binding
        definition = binding.definition
        run_usage = RunUsage()
        configured_limits = definition.spec.usage_limits
        usage_limits = UsageLimits(
            cost_limit=None,
            request_limit=None
            if configured_limits is None
            else configured_limits.model_requests,
            tool_calls_limit=None
            if configured_limits is None
            else configured_limits.tool_calls,
            input_tokens_limit=None
            if configured_limits is None
            else configured_limits.input_tokens,
            output_tokens_limit=None
            if configured_limits is None
            else configured_limits.output_tokens,
            total_tokens_limit=None
            if configured_limits is None
            else configured_limits.total_tokens,
        )
        result: AgentExecutionOutcome | None = None
        primary_error: BaseException | None = None
        metric_started = monotonic_ns() if self._metrics is not None else None
        metric_id = uuid.uuid4().hex if self._metrics is not None else None
        try:
            try:
                result = await self._execute_with_sandbox(
                    scope,
                    run_usage=run_usage,
                    usage_limits=usage_limits,
                )
                return result
            except asyncio.CancelledError as error:
                primary_error = error
                raise
            except AIError as error:
                mapped = _with_sandbox_cleanup_diagnostic(error, error)
                primary_error = mapped
                if mapped is error:
                    raise
                raise mapped from error
            except Exception as error:
                mapped = _with_sandbox_cleanup_diagnostic(
                    _execution_error(
                        error,
                        usage_limits=usage_limits,
                        run_usage=run_usage,
                    ),
                    error,
                )
                primary_error = mapped
                raise mapped from error
        finally:
            self._record_agent_run(
                scope,
                metric_id=metric_id,
                metric_started=metric_started,
                result=result,
                primary_error=primary_error,
            )
            if scope.usage_sink is not None:
                usage = (
                    result.usage
                    if isinstance(result, AgentExecutionResult)
                    else _usage_metrics(run_usage)
                )
                try:
                    scope.usage_sink(usage)
                except Exception:
                    if primary_error is None:
                        raise
                    _logger.error(
                        "usage sink failed after agent execution failure: step=%s",
                        scope.step_run_id,
                        exc_info=False,
                    )

    def _record_agent_run(
        self,
        scope: _RunScope,
        *,
        metric_id: str | None,
        metric_started: int | None,
        result: AgentExecutionOutcome | None,
        primary_error: BaseException | None,
    ) -> None:
        if self._metrics is None or metric_id is None or metric_started is None:
            return
        if isinstance(primary_error, asyncio.CancelledError):
            status = "CANCELLED"
            error_code = None
        elif primary_error is not None:
            status = "FAILED"
            error_code = (
                primary_error.code.value
                if isinstance(primary_error, AIError)
                else ErrorCode.INTERNAL_ERROR.value
            )
        elif isinstance(result, DeferredToolRequests):
            status = "DEFERRED"
            error_code = None
        else:
            status = "SUCCEEDED"
            error_code = None
        correlation: dict[str, str | int] = {
            "execution_id": scope.context.execution_id,
            "step_run_id": scope.step_run_id,
        }
        if scope.context.session_id is not None:
            correlation["session_id"] = scope.context.session_id
        try:
            self._metrics.try_record(
                Observation(
                    version=1,
                    observation_id=metric_id,
                    kind="linktools.agent.run",
                    occurred_at=datetime.now(timezone.utc),
                    source_namespace=scope.context.workspace.workspace_id,
                    tenant_id=scope.context.principal.tenant_id,
                    status=status,
                    error_code=error_code,
                    correlation=correlation,
                    dimensions={"agent_id": scope.binding.definition.spec.id},
                    measurements=(
                        MetricMeasurement(
                            "latency_ns",
                            1,
                            monotonic_ns() - metric_started,
                        ),
                    ),
                )
            )
        except Exception:
            _logger.exception("agent metric observation rejected")

    async def _execute_with_sandbox(
        self,
        scope: _RunScope,
        *,
        run_usage: RunUsage,
        usage_limits: UsageLimits,
    ) -> AgentExecutionOutcome:
        selected = tuple(
            candidate.id
            for candidate in scope.binding.definition.selected_tools
            if tool_class_from_metadata(
                _frozen_tool_metadata(candidate)
            )
            in {"filesystem.read", "filesystem.write", "shell"}
        )
        resources, resource_keys = await _skill_sandbox_resources(
            scope.binding.definition,
            self._skill_sources,
        )
        if not selected and not resources:
            return await self._execute(
                scope,
                run_usage=run_usage,
                usage_limits=usage_limits,
            )
        sandbox = scope.context.workspace.sandbox
        backend = sandbox if sandbox is not None else LocalSandbox()
        session = await backend.open(
            root=scope.context.workspace.root,
            resources=resources,
        )
        try:
            resource_paths = {
                skill_id: session.resource_path(key)
                for skill_id, key in resource_keys.items()
            }
            _logger.debug(
                "workspace sandbox opened for agent run: step=%s tools=%s resources=%s",
                scope.step_run_id,
                selected,
                tuple(resource_paths),
            )
            result = await self._execute(
                replace(
                    scope,
                    sandbox_session=session,
                    skill_resource_paths=resource_paths,
                ),
                run_usage=run_usage,
                usage_limits=usage_limits,
            )
        except BaseException as primary_error:
            try:
                await _close_sandbox_session(session)
            except asyncio.CancelledError as cleanup_cancel:
                if cleanup_cancel.__cause__ is not None:
                    raise primary_error from cleanup_cancel.__cause__
                raise primary_error
            except AIError as cleanup_error:
                _logger.warning(
                    "workspace sandbox cleanup failed after agent error: "
                    "step=%s code=%s exception_type=%s",
                    scope.step_run_id,
                    cleanup_error.code.value,
                    type(cleanup_error).__name__,
                )
                raise primary_error from cleanup_error
            raise
        await _close_sandbox_session(session)
        return result

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
        deferred_step_index: int | None = None

        def capture_deferred_step(step_index: int) -> None:
            nonlocal deferred_step_index
            if deferred_step_index is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            deferred_step_index = step_index

        model_journal = ModelRequestJournal(
            source_namespace=scope.context.workspace.workspace_id,
            tenant_id=scope.context.principal.tenant_id,
            execution_id=scope.context.execution_id,
            step_run_id=scope.step_run_id,
        )
        agent, capabilities = await _materialize_agent(
            scope,
            model=model,
            skill_sources=self._skill_sources,
            deferred_pause_sink=capture_deferred_step,
            metrics=self._metrics,
            model_journal=model_journal,
        )
        capabilities = cast(
            "tuple[AbstractCapability[AgentContext[object]], ...]",
            (*capabilities, _thinking_capability(scope.thinking)),
        )
        if scope.replace_history_system_prompt:
            capabilities = (
                *capabilities,
                ReinjectSystemPrompt(
                    replace_existing=True, id="linktools.ai.reinject-system-prompt"
                ),
            )
        capabilities = (
            *capabilities,
            _event_stream_capability(cast(EventSink, scope.event_sink)),
        )
        _logger.debug(
            "agent execution started: agent=%s definition=%s step=%s "
            "mode=%s planning=%s thinking=%s selected_tools=%s",
            definition.spec.id,
            definition.digest,
            scope.step_run_id,
            scope.mode,
            scope.planning,
            scope.thinking,
            len(definition.selected_tools),
        )
        user_prompt = scope.user_prompt
        deferred_kwargs: dict[str, object] = {}
        if scope.deferred_tool_results is not None:
            deferred_kwargs["deferred_tool_results"] = rehydrate_deferred_tool_results(
                scope.deferred_tool_results
            )
        final_result = await agent.run(
            user_prompt,
            deps=scope.context,
            message_history=scope.history or None,
            conversation_id=scope.conversation_id,
            run_id=scope.step_run_id,
            usage_limits=usage_limits,
            usage=run_usage,
            capabilities=capabilities,
            **deferred_kwargs,
        )
        output = final_result.output
        if isinstance(output, DeferredToolRequests):
            if deferred_step_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _validate_deferred_requests(output)
            return output
        run = await scope.step_store.get_run(run_id=scope.step_run_id)
        snapshot = await scope.step_store.latest_snapshot(run_id=scope.step_run_id)
        operations = (
            ()
            if scope.tool_operations is None
            else await scope.tool_operations.list_operations()
        )
        unresolved = tuple(
            operation
            for operation in operations
            if operation.status
            not in {ToolOperationStatus.COMPLETED, ToolOperationStatus.FAILED}
        )
        if (
            run is None
            or snapshot is None
            or unresolved
            or run.conversation_id != scope.conversation_id
        ):
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
            raise AIError(
                ErrorCode.OUTPUT_VALIDATION_FAILED, retryable=False
            ) from error
        binding.output_binding.validate_payload(payload)
        usage = _usage_metrics(run_usage)
        return AgentExecutionResult(
            final_result.run_id, payload, final_result.all_messages(), usage
        )


async def _close_sandbox_session(session: SandboxSession) -> None:
    try:
        await session.close()
    except asyncio.CancelledError:
        raise
    except AIError as error:
        if error.code is ErrorCode.SANDBOX_CLEANUP_FAILED:
            raise
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error
    except BaseException as error:
        raise AIError(ErrorCode.SANDBOX_CLEANUP_FAILED) from error


async def _skill_sandbox_resources(
    definition: AgentDefinition,
    sources: SkillSourceRegistry,
) -> tuple[tuple[SandboxResource, ...], Mapping[str, str]]:
    resources: dict[str, SandboxResource] = {}
    resource_keys: dict[str, str] = {}
    for skill in definition.skill_definitions:
        source_ref = skill.source_ref
        if source_ref is None:
            continue
        source = sources.resolve(source_ref.source_id)
        view = await source.inspect(source_ref.root)
        if view.location.kind != "local":
            continue
        source_path = Path(view.location.path)
        key = canonical_sha256(
            {
                "source_id": source_ref.source_id,
                "root": source_ref.root,
            }
        )
        resource = SandboxResource(key=key, source=source_path)
        existing = resources.get(key)
        if existing is not None and existing.source != resource.source:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        resources[key] = resource
        resource_keys[skill.id] = key
    ordered = tuple(resources[key] for key in sorted(resources))
    return ordered, resource_keys


def _validate_deferred_requests(requests: DeferredToolRequests) -> None:
    calls = (*requests.approvals, *requests.calls)
    ids = tuple(call.tool_call_id for call in calls)
    if not calls or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    if set(requests.metadata) - set(ids):
        raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    try:
        for call in calls:
            if not call.tool_name or not isinstance(call.args_as_dict(), dict):
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            normalize_json_value(call.args_as_dict())
        normalize_json_value(requests.metadata)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT) from error


async def _materialize_agent(
    scope: _RunScope,
    *,
    model: Model,
    skill_sources: SkillSourceRegistry,
    deferred_pause_sink: Callable[[int], None],
    metrics: MetricRecorder | None,
    model_journal: ModelRequestJournal,
) -> tuple[
    PydanticAgent[AgentContext[object], object],
    tuple[AbstractCapability[AgentContext[object]], ...],
]:
    definition = scope.binding.definition
    business_tools: list[Tool[AgentContext[object]]] = []
    workspace_names: list[str] = []
    business_descriptors: dict[str, ManagedToolDescriptor] = {}
    workspace_descriptors: dict[str, ManagedToolDescriptor] = {}
    compaction_policy = RuntimeCompactionPolicy()
    for candidate in definition.selected_tools:
        source_tool = cast("Tool[AgentContext[object]]", candidate.value)
        metadata = _frozen_tool_metadata(candidate)
        tool = _tool_with_metadata(source_tool, metadata)
        tool_class = tool_class_from_metadata(metadata)
        if tool_class is None:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        descriptor = managed_tool_descriptor_from_metadata(metadata)
        if tool_class == "business":
            business_tools.append(tool)
            business_descriptors[candidate.id] = descriptor
        elif tool_class in {
            "filesystem.read",
            "filesystem.write",
            "shell",
        }:
            workspace_names.append(candidate.id)
            workspace_descriptors[candidate.id] = descriptor
        else:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    capabilities: list[AbstractCapability[AgentContext[object]]] = []
    for candidate in definition.selected_capabilities:
        if not isinstance(candidate.value, AbstractCapability):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        capability = cast(
            "AbstractCapability[AgentContext[object]]",
            candidate.value,
        )
        capabilities.append(capability)
    if definition.skill_definitions:
        skill_capability = LinkToolsSkills(
            definition.skill_definitions,
            skill_sources,
            resource_paths=scope.skill_resource_paths,
            preloaded_skill_ids=definition.spec.preload_skills,
            max_preloaded_bytes=(
                scope.context.workspace.policy.max_preloaded_skill_bytes
            ),
        )
        capabilities.append(skill_capability)
    if scope.subagent_available and scope.binding.snapshot.subagents:
        if scope.subagent_delegate is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        subagent_capability = LinkToolsSubagents(
            scope.binding.snapshot.subagents,
            scope.subagent_delegate,
            scope.subagent_descriptions,
        )
        capabilities.append(subagent_capability)
    prepare_tools = _plan_mode_prepare(
        plan_mode=scope.mode == "plan",
        compaction_policy=compaction_policy,
    )
    capabilities.append(
        PrepareTools(
            prepare_tools,
            id=(
                "linktools.plan-mode"
                if scope.mode == "plan"
                else "linktools.semantic-capture"
            ),
        )
    )

    tool_metrics = (
        None
        if metrics is None
        else _ToolMetricContext(
            metrics,
            source_namespace=scope.context.workspace.workspace_id,
            tenant_id=scope.context.principal.tenant_id,
            execution_id=scope.context.execution_id,
            session_id=scope.context.session_id,
            step_run_id=scope.step_run_id,
            agent_id=definition.spec.id,
        )
    )

    raw_toolsets: list[AbstractToolset[AgentContext[object]]] = []
    workspace_toolsets = workspace_capabilities(
        scope.context.workspace,
        workspace_names,
        session=scope.sandbox_session,
    )
    workspace_toolset_values = tuple(
        toolset
        for capability in workspace_toolsets
        if (toolset := capability.get_toolset()) is not None
    )
    if workspace_toolset_values:
        raw_toolsets.append(
            RuntimeToolBoundaryToolset(
                workspace_toolset_values,
                workspace_descriptors,
                id="linktools.workspace",
                workspace_policy=scope.context.workspace.policy.tool_permissions,
                sandbox_session=scope.sandbox_session,
                tool_operations=scope.tool_operations,
                tool_metrics=tool_metrics,
                repository_boundary=scope.repository_instruction_boundary,
                background_tasks=scope.background_tasks,
            )
        )
    if definition.mcp_servers:
        mcp_toolsets = await materialize_mcp_servers(
            definition.mcp_servers,
            definition.mcp_selector_policy,
            principal=scope.context.principal,
            execution=ResourceRef(
                ResourceKind.EXECUTION,
                scope.context.execution_id,
                scope.context.principal.tenant_id,
            ),
            execution_root=str(scope.context.workspace.root),
        )
        for materialized in mcp_toolsets:
            raw_toolsets.append(
                RuntimeToolBoundaryToolset(
                    (
                        cast(
                            "AbstractToolset[AgentContext[object]]",
                            materialized.toolset,
                        ),
                    ),
                    {},
                    id="linktools.mcp",
                    descriptor=materialized.descriptor,
                    tool_operations=scope.tool_operations,
                    tool_metrics=tool_metrics,
                    background_tasks=scope.background_tasks,
                )
            )
    if business_tools:
        raw_business = FunctionToolset(business_tools, id="linktools.business")
        raw_toolsets.insert(
            0,
            RuntimeToolBoundaryToolset(
                (raw_business,),
                business_descriptors,
                id="linktools.business",
                tool_operations=scope.tool_operations,
                tool_metrics=tool_metrics,
                background_tasks=scope.background_tasks,
            ),
        )
    model_observation = RuntimeModelObservationCapability(
        metrics,
        source_namespace=scope.context.workspace.workspace_id,
        tenant_id=scope.context.principal.tenant_id,
        execution_id=scope.context.execution_id,
        session_id=scope.context.session_id,
        step_run_id=scope.step_run_id,
        agent_id=definition.spec.id,
        journal=model_journal,
    )
    capabilities.append(model_observation)
    platform = await compose_platform_capabilities(
        agent_name=definition.spec.id,
        step_run_id=scope.step_run_id,
        execution_id=scope.context.execution_id,
        segment_sequence=scope.segment_sequence,
        history_id=scope.history_id,
        memory_scope=scope.context.memory_scope,
        step_store=scope.step_store,
        memory_store=scope.memory_store,
        ordinary_tool_policy=definition.ordinary_tool_policy,
        compaction_policy=compaction_policy,
        planning=scope.planning,
        context_target_tokens=scope.context_target_tokens,
        parent_step_run_id=scope.parent_step_run_id,
        plan_store_resolver=scope.plan_store_resolver,
        deferred_pause_sink=deferred_pause_sink,
        model_journal=model_journal,
        model_observation_enabled=metrics is not None,
        model_request_observer=model_observation.record_external_model_request,
    )
    capabilities.extend(
        cast("tuple[AbstractCapability[AgentContext[object]], ...]", platform)
    )

    business_output_type: object
    if scope.binding.output_binding.mode == "text":
        business_output_type = TextOutput(_assistant_text_output)
    else:
        business_output_type = scope.binding.output_type
    output_type: object = [business_output_type, DeferredToolRequests]
    base_instructions = "\n".join(definition.spec.instructions)
    repository_boundary = scope.repository_instruction_boundary
    if repository_boundary is None:
        repository_instructions = (
            ""
            if scope.repository_instructions is None
            else scope.repository_instructions.render()
        )
        runtime_instructions: object = "\n\n".join(
            value
            for value in (
                base_instructions,
                repository_instructions,
            )
            if value != ""
        )
    else:

        def runtime_instructions(
            _: PydanticRunContext[object],
        ) -> str:
            return "\n\n".join(
                value
                for value in (
                    base_instructions,
                    repository_boundary.render(),
                )
                if value != ""
            )

    agent = cast(
        "PydanticAgent[AgentContext[object], object]",
        PydanticAgent(
            model,
            name=definition.spec.id,
            system_prompt=definition.spec.system_prompt,
            instructions=runtime_instructions,
            output_type=output_type,
            deps_type=AgentContext,
            retries={
                "tools": definition.spec.tool_retries,
                "output": definition.spec.output_retries,
            },
            toolsets=tuple(raw_toolsets),
        ),
    )
    return agent, tuple(capabilities)


def _frozen_tool_metadata(
    candidate: "CapabilityContribution[object]",
) -> Mapping[str, object]:
    contract = candidate.semantic_contract
    metadata = contract.get("metadata")
    if not isinstance(metadata, Mapping):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    validate_tool_semantic_metadata(
        metadata,
        require_effect=True,
        require_tool_class=True,
    )
    return metadata


def _tool_with_metadata(
    tool: "Tool[AgentContext[object]]",
    metadata: Mapping[str, object],
) -> "Tool[AgentContext[object]]":
    return Tool(
        tool.function,
        takes_ctx=tool.takes_ctx,
        max_retries=tool.max_retries,
        name=tool.name,
        description=tool.description,
        prepare=tool.prepare,
        args_validator=tool.args_validator,
        docstring_format=tool.docstring_format,
        require_parameter_descriptions=tool.require_parameter_descriptions,
        strict=tool.strict,
        sequential=tool.sequential,
        requires_approval=tool.requires_approval,
        metadata=dict(metadata),
        timeout=tool.timeout,
        defer_loading=tool.defer_loading,
        include_return_schema=tool.include_return_schema,
        function_schema=tool.function_schema,
    )


def _plan_mode_prepare(
    *,
    plan_mode: bool,
    compaction_policy: RuntimeCompactionPolicy | None = None,
) -> Callable[[PydanticRunContext[AgentContext[object]], list[ToolDefinition]], Any]:
    async def prepare(
        _ctx: PydanticRunContext[AgentContext[object]],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        if compaction_policy is not None:
            keep_result_tools: set[str] = set()
            context_dedupe_by_tool: dict[str, str] = {}
            for tool_def in tool_defs:
                metadata = tool_def.metadata
                validate_tool_semantic_metadata(metadata)
                if (
                    tool_def.tool_kind in _PLAN_SAFE_FRAMEWORK_TOOL_KINDS
                    or tool_compaction_keep_result_from_metadata(metadata)
                ):
                    keep_result_tools.add(tool_def.name)
                dedupe = tool_context_dedupe_from_metadata(metadata)
                if dedupe is not None:
                    context_dedupe_by_tool[tool_def.name] = dedupe
            compaction_policy.keep_result_tools = frozenset(keep_result_tools)
            compaction_policy.context_dedupe_by_tool = context_dedupe_by_tool
        if not plan_mode:
            return tool_defs
        return [
            tool_def
            for tool_def in tool_defs
            if tool_def.tool_kind in _PLAN_SAFE_FRAMEWORK_TOOL_KINDS
            or tool_plan_safe_from_metadata(tool_def.metadata)
        ]

    return prepare


def _assistant_text_output(value: str) -> AssistantTextOutput:
    return AssistantTextOutput(text=value)


def _event_stream_capability(
    sink: EventSink,
) -> ProcessEventStream[AgentContext[object]]:
    async def forward(
        _ctx: PydanticRunContext[AgentContext[object]],
        events: AsyncIterable[AgentStreamEvent],
    ) -> None:
        async for event in events:
            emission = _map_event(event)
            if emission is None:
                event_type = type(event)
                _logger.debug(
                    "pydantic event not projected: type=%s:%s",
                    event_type.__module__,
                    event_type.__qualname__,
                )
                continue
            await sink(emission)

    return ProcessEventStream(forward, id="linktools.ai.event-stream")


class _RuntimeThinking(Thinking):
    """Validate thinking against the effective model at the final request boundary."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def wrap_model_request(
        self,
        ctx: PydanticRunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        del ctx
        _validate_thinking_model(request_context.model, cast(ThinkingValue, self.effort))
        return await handler(request_context)


def _thinking_capability(thinking: ThinkingValue) -> Thinking:
    return _RuntimeThinking(effort=thinking, id="linktools.ai.thinking")


def _validate_thinking_model(model: Model, thinking: ThinkingValue) -> None:
    profile = model.profile
    supports = bool(profile.get("supports_thinking", False))
    always = bool(profile.get("thinking_always_enabled", False))
    if thinking is False:
        if always:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "thinking", "reason": "model_always_enabled"},
            )
        return
    if not (supports or always):
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={"field": "thinking", "reason": "model_not_supported"},
        )


def _map_event(event: object) -> "AgentEmission | None":
    if (
        isinstance(event, PartStartEvent)
        and isinstance(event.part, TextPart)
        and event.part.content
    ):
        return LiveDelta(ExecutionDeltaType.ASSISTANT_TEXT_DELTA, event.part.content)
    if (
        isinstance(event, PartStartEvent)
        and isinstance(event.part, ThinkingPart)
        and event.part.content
    ):
        return LiveDelta(
            ExecutionDeltaType.ASSISTANT_THINKING_DELTA, event.part.content
        )
    if (
        isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, TextPartDelta)
        and event.delta.content_delta
    ):
        return LiveDelta(
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA, event.delta.content_delta
        )
    if (
        isinstance(event, PartDeltaEvent)
        and isinstance(event.delta, ThinkingPartDelta)
        and event.delta.content_delta
    ):
        return LiveDelta(
            ExecutionDeltaType.ASSISTANT_THINKING_DELTA, event.delta.content_delta
        )
    if isinstance(event, PartEndEvent) and isinstance(event.part, TextPart):
        text = event.part.content
        return DurableBoundary(
            ExecutionEventType.ASSISTANT_PART_COMPLETED,
            {
                "part_type": "text",
                "digest": canonical_sha256(text),
                "characters": len(text),
            },
        )
    if isinstance(event, PartEndEvent) and isinstance(event.part, ThinkingPart):
        text = event.part.content
        return DurableBoundary(
            ExecutionEventType.ASSISTANT_PART_COMPLETED,
            {
                "part_type": "thinking",
                "digest": canonical_sha256(text),
                "characters": len(text),
            },
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
                    "result_digest": (
                        tool_return_content_digest(part.content) if success else None
                    ),
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


def _sandbox_cleanup_cause(error: BaseException) -> "AIError | None":
    cause = error.__cause__
    if isinstance(cause, AIError) and cause.code is ErrorCode.SANDBOX_CLEANUP_FAILED:
        return cause
    return None


def _with_sandbox_cleanup_diagnostic(error: AIError, source: BaseException) -> AIError:
    if _sandbox_cleanup_cause(source) is None:
        return error
    details = dict(error.safe_details)
    details[_SECONDARY_ERROR_CODE_KEY] = ErrorCode.SANDBOX_CLEANUP_FAILED.value
    return AIError(
        error.code,
        str(error),
        category=error.category,
        retryable=error.retryable,
        operation_id=error.operation_id,
        safe_details=details,
        diagnostics=error.diagnostics,
    )


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
        if isinstance(retry_after, (int, float, str)) and not isinstance(
            retry_after, bool
        ):
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
    "AgentExecutionResult",
    "AgentExecutor",
    "DurableBoundary",
    "EventSink",
    "LiveDelta",
    "UsageSink",
]
