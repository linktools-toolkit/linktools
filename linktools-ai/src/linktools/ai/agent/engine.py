#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine: the Store-free per-invocation model/tool loop (FS-29).

The engine owns ONLY the prompt-build → model/tool drive → outcome path. It
 touches no run-lifecycle persistence and no coordinator or controller --
RunRecord create/transition, checkpoint/session/approval persistence, pause/
cancel/stream events, and the cross-store commit are ExecutionService's sole
job (see execution/service.py). The single public entry point is
:meth:`AgentEngine.execute_pure`, which returns a discriminated-union
``AgentExecutionOutcome`` (AgentCompleted / AgentPaused / AgentFailed /
AgentCancelled) -- never a Store write.

execute_pure drives ``agent.pydantic_agent.iter()`` and:
* runs the before_run/after_run middleware hooks (on a new, non-resuming run)
  via the wired MiddlewarePipeline -- after_run fires before the outcome is
  returned, so it always precedes the ExecutionService commit;
* builds the model prompt via :class:`~linktools.ai.agent.prompt.builder.PromptBuilder`
  (memory + knowledge sections from their async policies, then feature-
  resolved sections folded in via PromptBuilder.combine). Prior-turn history
  arrives as native pydantic-ai ``message_history`` on AgentInput (loaded by
  ExecutionService), NOT via a Store read here;
* threads the per-Run ToolContext through pydantic-ai dependency injection
  (``deps=AgentDependencies(tool_context=...)`` -> ``ctx.deps``), never a
  mutable SDK hook field;
* enforces ModelPolicy.timeout_seconds / max_tokens / budget;
* classifies exceptions via the narrow ``_EXPECTED_RUN_FAILURES`` allowlist
  (model routing/policy/output denials, ToolError, MCPToolError) -> AgentFailed;
  everything else (config/invariant/protocol violations, programming errors)
  propagates unchanged;
* publishes ONLY process-local typed execution events through the injected ``live_events``
  sink (text / tool / model_progress); state events (paused / completed /
  failed / cancelled) are the ExecutionService's job, published only AFTER the
  durable commit succeeds. Security+observability events route through the
  injected ``security_events`` sink.

Optional Observability: when ``observability`` is wired, execute_pure wraps the
loop in an outer ``agent.run`` span and the iter() drive in a nested
``agent.model`` span (parented via the tracing contextvar). When ``metrics`` is
wired, records ``counter("agent.run.completed"/"agent.run.failed")`` and
``histogram("agent.run.duration_ms")``. Both default to None (no-op)."""

import asyncio
import contextlib
import dataclasses
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from linktools.core import environ

from ..errors import (
    MCPToolError,
    ModelInvocationDeniedError,
    ModelOutputValidationError,
    ModelPolicyExceededError,
    ModelResultDeniedError,
    ModelRoutingError,
    RunPaused,
    RuntimeInitializationError,
    ToolError,
    ToolSchemaError,
)
from ..execution.domain import MessageCaptureState
from ..execution.domain import RunErrorInfo
from ..execution.domain import (
    RunStatus as ExecutionRunStatus,
    RunUsage as ExecutionRunUsage,
)
from ..execution.snapshots import (
    ModelRequestUsageObservation,
    RequestUsage,
)
from ..execution.live_events import (
    AssistantTextDelta,
    AssistantThoughtDelta,
    ToolCallCompleted,
    ToolCallFailed,
    ToolCallStarted,
    publish_execution_event,
)
from ..governance.policy.engine import ToolContext
from ..governance.security.redact import redact_exception
from ..model.recording import SemanticRecordingModel
from ..model.resolver import resolve_pricing_id
from ..observability.tracing import use_span
from ..prompt import UserPrompt
from .dependencies import AgentDependencies
from .middleware.pipeline import MiddlewarePipeline
from .models import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    model_supports_streaming,
    AgentPaused,
    AgentUsage,
    PauseRequest,
)
from .models import RunResult
from .prompt.builder import PromptBuilder

if TYPE_CHECKING:
    from ..execution.cancellation import CancellationToken
    from ..execution.context import RunContext
    from .models import AgentExecutionOutcome, AgentInput, CompiledAgent

    from pydantic_ai.toolsets import AbstractToolset

    from .assembly.assembler import AgentAssembler
    from .assembly.models import AgentAssembly
    from .tool.pydantic_ai import PydanticAIToolAdapter
    from .sandbox.protocols import Sandbox
    from .retrieval.retriever import Retriever
    from .memory.store import MemoryStore
    from .prompt.window import SessionWindowPolicy
    from ..observability.metrics import ObservabilityMetrics
    from ..observability.tracing import ObservabilitySink
    from ..execution.live_events import RunLiveEventSink, SecurityEventSink
    from ..execution.trace_collector import SemanticTraceCollector
    from ..governance.security.pipeline import SecurityPipeline
    from ..model.pricing import ModelPricing, ModelPricingProvider


logger = environ.get_logger("ai.agent.engine")


def _execution_id_from_run_context(run_ctx: object) -> str:
    direct = getattr(run_ctx, "run_id", None)
    if isinstance(direct, str):
        return direct
    dependencies = getattr(run_ctx, "deps", None)
    tool_context = getattr(dependencies, "tool_context", None)
    value = getattr(tool_context, "run_id", None)
    return value if isinstance(value, str) else ""


def _request_usage_from_provider(usage: object) -> RequestUsage:
    input_tokens = int(getattr(usage, "input_tokens", 0))
    output_tokens = int(getattr(usage, "output_tokens", 0))
    raw_total = getattr(usage, "total_tokens", None)
    total_tokens = (
        input_tokens + output_tokens
        if raw_total is None or (int(raw_total) == 0 and input_tokens + output_tokens > 0)
        else int(raw_total)
    )
    return RequestUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cache_write_tokens=int(getattr(usage, "cache_write_tokens", 0)),
        cache_read_tokens=int(getattr(usage, "cache_read_tokens", 0)),
        total_cost=getattr(usage, "total_cost", None),
    )


def _run_usage_from_provider(usage: object) -> ExecutionRunUsage:
    request = _request_usage_from_provider(usage)
    return ExecutionRunUsage(
        input_tokens=request.input_tokens,
        output_tokens=request.output_tokens,
        total_tokens=request.total_tokens,
        cache_write_tokens=request.cache_write_tokens,
        cache_read_tokens=request.cache_read_tokens,
        total_cost=request.total_cost,
    )


def _response_identity(response: object) -> "tuple[str | None, str | None]":
    provider_name = getattr(response, "provider_name", None)
    model_name = getattr(response, "model_name", None)
    return (
        provider_name if isinstance(provider_name, str) else None,
        model_name if isinstance(model_name, str) and model_name else None,
    )


def _provider_response_id(response: object) -> "str | None":
    value = getattr(response, "provider_response_id", None)
    return value if isinstance(value, str) and value else None


def _raw_stream_usage(stream: object) -> "object | None":
    accessor = getattr(stream, "usage", None)
    return accessor() if callable(accessor) else accessor


def _is_empty_unpriced_usage(usage: RequestUsage) -> bool:
    return (
        usage.input_tokens == 0
        and usage.output_tokens == 0
        and usage.cache_read_tokens == 0
        and usage.cache_write_tokens == 0
        and usage.total_cost is None
    )


def _optional_text(value: object) -> "str | None":
    return value if isinstance(value, str) and value else None


@dataclasses.dataclass(frozen=True, slots=True)
class _PartialStreamResponse:
    usage: object
    provider_name: "str | None"
    model_name: "str | None"
    provider_response_id: "str | None"


def _partial_stream_response(
    stream: object,
    raw_usage: object,
) -> _PartialStreamResponse:
    return _PartialStreamResponse(
        usage=raw_usage,
        provider_name=_optional_text(getattr(stream, "provider_name", None)),
        model_name=_optional_text(getattr(stream, "model_name", None)),
        provider_response_id=_optional_text(
            getattr(stream, "provider_response_id", None)
        ),
    )


async def _observe_failed_stream_usage(
    *,
    request_stream: object,
    request_key: str,
    observe_usage: "Callable[[str, RequestUsage, object], Awaitable[None]]",
) -> None:
    try:
        response = request_stream.get()
    except BaseException:
        raw_usage = _raw_stream_usage(request_stream)
        if raw_usage is None:
            return
        usage = _request_usage_from_provider(raw_usage)
        if _is_empty_unpriced_usage(usage):
            return
        response = _partial_stream_response(request_stream, raw_usage)
    else:
        usage = _request_usage_from_provider(response.usage)

    await observe_usage(request_key, usage, response)


def _latest_model_response(
    state: object, *, after_count: "int | None" = None
) -> "object | None":
    history = tuple(getattr(state, "message_history", ()) or ())
    if after_count is not None and len(history) <= after_count:
        return None
    for message in reversed(history):
        if hasattr(message, "usage") and hasattr(message, "model_name"):
            return message
    return None


# Only modeled execution failures become AgentFailed.
# Configuration, invariant, protocol, and programming errors propagate.
_EXPECTED_RUN_FAILURES: "tuple[type[BaseException], ...]" = (
    ModelRoutingError,
    ModelPolicyExceededError,
    ModelOutputValidationError,
    ModelInvocationDeniedError,
    ModelResultDeniedError,
    ToolError,
    MCPToolError,
)


@contextlib.asynccontextmanager
async def _noop_span():
    """Async context manager that yields ``None`` and does nothing -- the
    fallback for :meth:`AgentEngine._span` when observability is not wired,
    so the lifecycle body has a single ``async with`` shape regardless."""
    yield None


class RunUsageSink(Protocol):
    async def observe_request(
        self,
        observation: ModelRequestUsageObservation,
        *,
        pricing: "ModelPricing | None",
    ) -> "ExecutionRunUsage": ...

    def snapshot(self) -> "ExecutionRunUsage": ...


class AgentEngine:
    def __init__(
        self,
        *,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
        session_window: "SessionWindowPolicy | None" = None,
        memory_store: "MemoryStore | None" = None,
        retriever: "Retriever | None" = None,
        observability: "ObservabilitySink | None" = None,
        metrics: "ObservabilityMetrics | None" = None,
        sandbox: "Sandbox | None" = None,
        assembler: "AgentAssembler | None" = None,
        tool_adapter: "PydanticAIToolAdapter | None" = None,
        security_pipeline: "SecurityPipeline | None" = None,
        pricing_provider: "ModelPricingProvider | None" = None,
        trace_codec: "object | None" = None,
    ) -> None:
        self._middleware_pipeline = middleware_pipeline
        self._session_window = session_window
        self._memory_store = memory_store
        self._retriever = retriever
        self._observability = observability
        self._metrics = metrics
        self._sandbox = sandbox
        self._assembler = assembler
        self._tool_adapter = tool_adapter
        self._security_pipeline = security_pipeline
        self._pricing_provider = pricing_provider
        self._trace_codec = trace_codec

    def _span(self, name: str, *, attrs: "dict | None" = None):
        """Return an async context manager that opens an observability span when
        a sink is wired, or a no-op when it is not. Keeps the lifecycle body
        single-shape regardless of observability being configured."""
        if self._observability is None:
            return _noop_span()
        return use_span(self._observability, name, attributes=attrs or {})

    def _effective_memory_policy(self):
        """Explicit policy from options, else the default built from the wired
        memory store, else None (no memory injection)."""
        if self._memory_store is not None:
            from .context_policies import DefaultMemoryPolicy

            return DefaultMemoryPolicy(self._memory_store)
        return None

    def _effective_retrieval_policy(self):
        if self._retriever is not None:
            from .context_policies import DefaultRetrievalPolicy

            return DefaultRetrievalPolicy(self._retriever)
        return None

    async def _forward_model_stream(
        self,
        node: object,
        run_ctx: object,
        *,
        cancellation: "CancellationToken",
        live_events: "RunLiveEventSink",
        request_key: str,
        observe_usage: "Callable[[str, RequestUsage, object], Awaitable[None]] | None",
    ) -> str:
        """Stream one model-request node, publishing text/thinking chunks to
        ``live_events`` as they arrive. Returns the concatenated text delta so
        the caller can fold it into its running accumulator. ``node.stream`` is
        the pydantic-ai per-node async stream yielding PartStartEvent /
        PartDeltaEvent; only TextPart/ThinkingPart (and their delta kinds) are
        forwarded, all other part kinds are ignored here."""
        from pydantic_ai.messages import (
            PartDeltaEvent,
            PartStartEvent,
            TextPart,
            TextPartDelta,
            ThinkingPart,
            ThinkingPartDelta,
        )

        accumulated = ""
        request_stream = None
        usage_attempted = False
        usage_observed = False
        try:
            async with node.stream(run_ctx) as request_stream:
                async for ev in request_stream:
                    chunk = None
                    kind = None
                    if isinstance(ev, PartStartEvent) and isinstance(ev.part, TextPart):
                        chunk = ev.part.content
                        kind = "text"
                    elif isinstance(ev, PartDeltaEvent) and isinstance(
                        ev.delta, TextPartDelta
                    ):
                        chunk = ev.delta.content_delta
                        kind = "text"
                    elif isinstance(ev, PartStartEvent) and isinstance(
                        ev.part, ThinkingPart
                    ):
                        chunk = ev.part.content
                        kind = "thinking"
                    elif isinstance(ev, PartDeltaEvent) and isinstance(
                        ev.delta, ThinkingPartDelta
                    ):
                        chunk = ev.delta.content_delta
                        kind = "thinking"
                    if chunk:
                        if kind == "text":
                            accumulated += chunk
                        event_type = (
                            AssistantTextDelta
                            if kind == "text"
                            else AssistantThoughtDelta
                        )
                        await publish_execution_event(
                            live_events,
                            _execution_id_from_run_context(run_ctx),
                            event_type(
                                execution_id=_execution_id_from_run_context(run_ctx),
                                text=chunk,
                            ),
                        )
                        await cancellation.raise_if_cancelled()
                if observe_usage is not None:
                    response = request_stream.get()
                    usage_attempted = True
                    await observe_usage(
                        request_key,
                        _request_usage_from_provider(response.usage),
                        response,
                    )
                    usage_observed = True
        except BaseException as primary_error:
            if (
                observe_usage is not None
                and request_stream is not None
                and not usage_attempted
                and not usage_observed
            ):
                try:
                    await _observe_failed_stream_usage(
                        request_stream=request_stream,
                        request_key=request_key,
                        observe_usage=observe_usage,
                    )
                except BaseException as usage_error:
                    raise usage_error from primary_error
            raise
        return accumulated

    async def _forward_tool_stream(
        self,
        node: object,
        run_ctx: object,
        *,
        cancellation: "CancellationToken",
        live_events: "RunLiveEventSink",
    ) -> None:
        """Stream one call-tools node, publishing tool start/end events to
        ``live_events`` as each tool call begins and returns."""
        from pydantic_ai.messages import (
            FunctionToolCallEvent,
            FunctionToolResultEvent,
            ToolReturnPart,
        )

        async with node.stream(run_ctx) as tool_stream:
            async for ev in tool_stream:
                tool_event = None
                if isinstance(ev, FunctionToolCallEvent):
                    tool_event = ToolCallStarted(
                        execution_id=_execution_id_from_run_context(run_ctx),
                        tool_call_id=ev.part.tool_call_id,
                        tool_name=ev.part.tool_name,
                        arguments=ev.part.args,
                    )
                elif isinstance(ev, FunctionToolResultEvent):
                    execution_id = _execution_id_from_run_context(run_ctx)
                    if isinstance(ev.part, ToolReturnPart) and ev.part.outcome == "success":
                        tool_event = ToolCallCompleted(
                            execution_id=execution_id,
                            tool_call_id=ev.part.tool_call_id,
                            tool_name=ev.part.tool_name,
                            result=ev.part.content,
                        )
                    else:
                        tool_event = ToolCallFailed(
                            execution_id=execution_id,
                            tool_call_id=ev.part.tool_call_id,
                            tool_name=ev.part.tool_name,
                            error=str(getattr(ev.part, "content", "tool failed")),
                        )
                if tool_event is not None:
                    await publish_execution_event(live_events, _execution_id_from_run_context(run_ctx), tool_event)
                    await cancellation.raise_if_cancelled()

    async def _apply_window_policy(
        self,
        message_history: "tuple[Any, ...] | None",
        model: Any,
        *,
        security_events: "SecurityEventSink",
    ) -> "tuple[Any, ...] | None":
        """Trim ``message_history`` per the wired session-window policy (if any)
        and emit a ``PromptWindowApplied`` security event recording the before/
        after counts. The policy runs even when history is empty so the wiring
        is observable and a length-based policy can still record its decision.
        Returns the trimmed history (or the original when no policy is wired)."""
        window_policy = self._session_window
        if window_policy is None:
            return message_history
        before_count = len(message_history or ())
        trimmed = await window_policy.select_messages(message_history or (), model)
        from ..observability.events.payloads import PromptWindowApplied

        await security_events.emit(
            PromptWindowApplied(
                policy=type(window_policy).__name__,
                before=before_count,
                after=len(trimmed),
            )
        )
        return tuple(trimmed) or None

    async def _build_turn_prompt(
        self,
        context: "RunContext",
        user_prompt: "str | UserPrompt | None",
    ) -> "str | None":
        """Build the base prompt for a NEW turn: memory section (from the wired
        memory policy) + knowledge section (from the wired retrieval policy) +
        the user's prompt, combined by PromptBuilder. Returns None on the resume
        path (the prompt is already baked into the checkpointed history)."""
        prompt_text = (
            user_prompt.text_fallback()
            if isinstance(user_prompt, UserPrompt)
            else user_prompt
        )
        memory_section = ""
        memory_policy = self._effective_memory_policy()
        if memory_policy is not None:
            memories = await memory_policy.select_memories(context, prompt_text)
            memory_section = self._prompt_formatter().format_memory(memories) or ""
        knowledge_section = ""
        retrieval_policy = self._effective_retrieval_policy()
        if retrieval_policy is not None:
            items = await retrieval_policy.retrieve(context, prompt_text)
            knowledge_section = self._prompt_formatter().format_knowledge(items) or ""
        return PromptBuilder.build_base_prompt(
            user_prompt=prompt_text or "",
            prior_messages=(),
            memory_section=memory_section,
            knowledge_section=knowledge_section,
        )

    async def _prepare_turn_inputs(
        self,
        agent: "CompiledAgent",
        input: "AgentInput",
        context: "RunContext",
        assembly: "AgentAssembly | None",
        *,
        trace_collector: "SemanticTraceCollector | None",
        security_events: "SecurityEventSink",
    ) -> "tuple[AgentDependencies, list[AbstractToolset], AgentAssembly | None]":
        """Build the per-turn ``deps`` and ``toolsets`` for the model drive, and
        assemble the agent's declared features (if any) when ``assembly`` is not
        already supplied. Emits the ToolExposureApplied / PromptCatalogInjected
        security events for the assembled tools and prompt sections. Returns
        ``(deps, toolsets, assembly)`` -- ``assembly`` is the (possibly
        just-assembled) feature assembly, threaded back so the caller can fold
        its prompt sections into the effective prompt."""
        tool_context = ToolContext(
            run_id=context.run_id,
            session_id=context.session_id,
            tool_call_id=None,
            tenant_id=context.tenant_id,
        )
        deps = AgentDependencies(
            tool_context=tool_context,
            sandbox=self._sandbox,
        )
        toolsets: "list[AbstractToolset]" = []
        if agent.spec.features and self._assembler is None:
            raise RuntimeInitializationError(
                "AgentEngine requires an AgentAssembler for declared features"
            )
        if agent.spec.features and self._tool_adapter is None:
            raise RuntimeInitializationError(
                "AgentEngine requires a PydanticAIToolAdapter for feature tools"
            )
        if agent.spec.features:
            from .assembly.provider import AgentFeatureContext
            from .tool.invocation import ToolExecutionContext

            feature_context = AgentFeatureContext(
                agent_id=agent.spec.id,
                sandbox=deps.sandbox,
                execution_id=context.run_id,
                root_execution_id=context.root_execution_id,
                parent_execution_id=context.parent_execution_id,
                session_id=context.session_id,
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                workspace=context.workspace,
            )
            if assembly is None:
                assembly = await self._assembler.assemble(
                    agent.spec,
                    feature_context,
                )
                if environ.debug:
                    logger.debug(
                        "run %s feature assembly: agent=%s tools=%s prompt_sections=%s",
                        context.run_id,
                        agent.spec.id,
                        len(assembly.tools),
                        tuple(assembly.prompt_sections),
                    )
            tool_descriptors = {
                definition.descriptor.name: definition.descriptor
                for definition in assembly.tools
            }
            if tool_descriptors:
                deps = dataclasses.replace(deps, tool_descriptors=tool_descriptors)
            if assembly.tools:
                toolsets.append(
                    self._tool_adapter.build_toolset(
                        assembly.tools,
                        context=ToolExecutionContext(
                            execution_id=context.run_id,
                            tool_call_id="",
                            dependencies=deps,
                            run_context=context,
                            approved_tool_call_id=input.approved_tool_call_id,
                            approved_binding_fingerprint=(
                                input.approved_binding_fingerprint
                            ),
                            trace_sink=trace_collector,
                        ),
                    )
                )
        if assembly is not None:
            from ..observability.events.payloads import (
                PromptCatalogInjected,
                ToolExposureApplied,
            )

            await security_events.emit(
                ToolExposureApplied(
                    agent_id=agent.spec.id, total_tools=len(assembly.tools)
                )
            )
            for section in assembly.prompt_sections:
                await security_events.emit(
                    PromptCatalogInjected(agent_id=agent.spec.id, section=section)
                )
        return deps, toolsets, assembly

    def _wrap_iter_model(
        self,
        base_model: Any,
        context: "RunContext",
        agent: "CompiledAgent",
        *,
        trace_collector: "SemanticTraceCollector | None",
    ) -> Any:
        """Wrap the agent's model with the wired security pipeline and trace
        recording, in that order (security outermost so it sees the recorded
        model's calls). Returns the wrapped model, or ``base_model`` unchanged
        when neither is wired."""
        wrapped = None
        if self._security_pipeline is not None:
            from ..governance.security.secured_model import SecuredModel

            wrapped = SecuredModel(
                base_model,
                pipeline=self._security_pipeline,
                run_id=context.run_id,
                agent_id=agent.spec.id,
            )
        if trace_collector is not None:
            wrapped = SemanticRecordingModel(
                wrapped or base_model,
                trace_collector,
                self._trace_codec,
            )
        return wrapped or base_model

    async def _enforce_usage_policy(
        self,
        model_policy: Any,
        usage: Any,
    ) -> None:
        """Raise ModelPolicyExceededError when the run's token usage breaches
        ``max_tokens`` or ``budget``. ``usage`` is the captured cumulative usage;
        nothing happens when it is None (no usage recorded)."""
        if usage is None:
            return
        max_tokens = model_policy.max_tokens
        if max_tokens is not None:
            used = usage.input_tokens + usage.output_tokens
            if used > max_tokens:
                raise ModelPolicyExceededError(
                    f"max_tokens exceeded: used {used} > max_tokens {max_tokens}",
                    kind="max_tokens",
                )
        budget = model_policy.budget
        if budget is None:
            return
        cost = getattr(usage, "total_cost", None)
        if cost is None:
            raise ModelPolicyExceededError(
                "ModelPolicy.budget cannot be enforced because actual model cost "
                "is unknown",
                kind="budget",
            )
        if cost > budget:
            raise ModelPolicyExceededError(
                f"cost budget exceeded: {cost} > budget {budget}",
                kind="budget",
            )

    def _prompt_formatter(self):
        from .context_policies import DefaultPromptContextFormatter

        return DefaultPromptContextFormatter()

    async def execute_pure(
        self,
        agent: "CompiledAgent",
        input: "AgentInput",
        context: "RunContext",
        *,
        cancellation: "CancellationToken",
        live_events: "RunLiveEventSink",
        security_events: "SecurityEventSink",
        assembly: "AgentAssembly | None" = None,
        trace_sequence: int = 0,
        trace_collector: "SemanticTraceCollector | None" = None,
        usage_sink: "RunUsageSink | None" = None,
        extra_toolsets: "tuple[Any, ...]" = (),
    ) -> "AgentExecutionOutcome":
        """The target pure execution loop. Touches NO run_store/
        session_store/commit_coordinator/run_controller -- ExecutionService
        owns RunRecord creation/transition, checkpoint and session storage,
        and execution claim/heartbeat/fencing; this method only assembles
        the prompt, drives the model/tool loop, and returns a value
        (never a Store write).

        Every security/observability event the feature and tool layers
        emit (tool-pipeline decisions, exposure/catalog/window markers) flows
        through the injected ``security_events`` sink via a
        ``SecurityEventSinkEmitter`` -- this method holds no durable store
        reference.

        Exception classification (no ``except Exception: return FAILED``
        catch-all): an explicit Run cancellation (``cancellation.
        is_cancelled()`` true when a CancelledError surfaces) returns
        AgentCancelled; an external CancelledError re-raises after cleanup;
        the narrow ``_EXPECTED_RUN_FAILURES`` set (model routing/policy/output
        denials, runtime tool failures, MCP tool errors) is returned as
        AgentFailed. Everything else -- configuration/invariant/protocol
        violations (RuntimeInitializationError, RunInvariantError,
        AgentAssemblyError, MCPConnectionError, ToolSchemaError, ...)
        AND unknown programming errors (TypeError, AttributeError, ...) --
        propagates unchanged. The classification is an allowlist of expected
        failures, not a denylist, so a new programming bug surfaces as a real
        traceback instead of being hidden inside a clean AgentFailed outcome."""
        from pydantic_ai import Agent as PydanticAgent

        metrics = self._metrics
        run_attrs = {"run_id": context.run_id, "session_id": context.session_id}
        started = time.monotonic()

        current_run = None

        def _encoded(messages: "tuple[Any, ...]") -> "tuple[Any, ...]":
            return self._trace_codec.encode_model_messages(messages)

        def all_messages_encoded() -> "tuple[Any, ...]":
            # RESUME_CHECKPOINT source: the exact post-window-policy pause
            # context (all_messages()). Empty when the run never reached
            # agent.iter (current_run unset).
            current = current_run
            if current is None:
                return ()
            return _encoded(tuple(current.all_messages()))

        def delta_messages_encoded() -> "tuple[Any, ...]":
            # TURN_DELTA source: this run's new_messages() -- window-policy
            # immune (only messages produced this run). Empty when current_run
            # is unset; best-effort otherwise.
            current = current_run
            if current is None:
                return ()
            return _encoded(tuple(current.new_messages()))

        async def _snapshot(
            status: ExecutionRunStatus,
            *,
            final_output: Any = None,
            usage: "ExecutionRunUsage | None" = None,
            capture_state: MessageCaptureState = MessageCaptureState.COMPLETE,
        ) -> Any:
            if trace_collector is None:
                return None
            # checkpoint (all_messages) is non-empty ONLY for PAUSED; the store
            # clears it on every terminal state so no cumulative history is kept.
            checkpoint = (
                all_messages_encoded() if status is ExecutionRunStatus.PAUSED else ()
            )
            return await trace_collector.build_snapshot(
                delta_messages=delta_messages_encoded(),
                checkpoint_messages=checkpoint,
                final_output=final_output,
                status=status,
                usage=usage or ExecutionRunUsage(),
                capture_state=capture_state,
            )

        def _captured_usage() -> "ExecutionRunUsage":
            if usage_sink is None:
                current = current_run
                state = getattr(getattr(current, "ctx", None), "state", None)
                usage = getattr(state, "usage", None)
                if usage is None:
                    return ExecutionRunUsage()
                return _run_usage_from_provider(usage)
            captured = usage_sink.snapshot()
            return ExecutionRunUsage(
                input_tokens=captured.input_tokens,
                output_tokens=captured.output_tokens,
                total_tokens=captured.total_tokens,
                cache_write_tokens=captured.cache_write_tokens,
                cache_read_tokens=captured.cache_read_tokens,
                total_cost=captured.total_cost,
            )

        request_sequence = 0
        pricing_cache: "dict[str, ModelPricing | None]" = {}

        def _next_request_key() -> str:
            nonlocal request_sequence
            request_sequence += 1
            return f"{context.run_id}:{request_sequence}"

        async def _observe_model_usage(
            request_key: str,
            request_usage: RequestUsage,
            response: object,
        ) -> None:
            if usage_sink is None:
                return
            provider_name, response_model_name = _response_identity(response)
            pricing_id = resolve_pricing_id(
                provider_name=provider_name,
                response_model_name=response_model_name,
                candidates=agent.model_bundle.candidates,
            )
            pricing = None
            if pricing_id is not None and self._pricing_provider is not None:
                if pricing_id not in pricing_cache:
                    try:
                        pricing_cache[pricing_id] = (
                            await self._pricing_provider.get_pricing(pricing_id)
                        )
                    except Exception as exc:
                        logger.debug(
                            "usage pricing unavailable run_id=%s pricing_id=%s error_type=%s",
                            context.run_id,
                            pricing_id,
                            type(exc).__name__,
                        )
                        pricing_cache[pricing_id] = None
                pricing = pricing_cache[pricing_id]
            observed_key = _provider_response_id(response) or request_key
            observation = ModelRequestUsageObservation(
                request_key=observed_key,
                usage=request_usage,
                provider_name=provider_name,
                response_model_name=response_model_name,
            )
            await usage_sink.observe_request(
                observation,
                pricing=pricing,
            )

        try:
            async with self._span("agent.run", attrs=run_attrs):
                if environ.debug:
                    logger.debug(
                        "run %s started (session=%s resuming=%s)",
                        context.run_id,
                        context.session_id,
                        input.resuming,
                    )
                await cancellation.raise_if_cancelled()

                # before_run middleware fires on a NEW run only -- resume skips
                # it (the initial run already ran it).
                if not input.resuming and self._middleware_pipeline is not None:
                    await cancellation.raise_if_cancelled()
                    await self._middleware_pipeline.run_before_run(context)

                message_history = await self._apply_window_policy(
                    input.message_history or None,
                    agent.spec.model,
                    security_events=security_events,
                )

                prompt: "str | None" = None
                if not input.resuming:
                    prompt = await self._build_turn_prompt(context, input.prompt)

                deps, toolsets, assembly = await self._prepare_turn_inputs(
                    agent,
                    input,
                    context,
                    assembly,
                    trace_collector=trace_collector,
                    security_events=security_events,
                )
                toolsets.extend(extra_toolsets)

                timeout = agent.spec.model.timeout_seconds
                accumulated_text = ""
                result = None
                await cancellation.raise_if_cancelled()
                effective_prompt = PromptBuilder.combine(
                    base_prompt=prompt,
                    feature_sections=(
                        assembly.prompt_sections if assembly is not None else {}
                    ),
                    static_sections=agent.spec.instructions.sections,
                    resuming=input.resuming,
                )
                if isinstance(input.prompt, UserPrompt) and not input.resuming:
                    prompt_content = input.prompt.model_content()
                    if effective_prompt and effective_prompt != input.prompt.text_fallback():
                        prompt_content.insert(0, effective_prompt[: -len(input.prompt.text_fallback())].rstrip())
                    effective_prompt = prompt_content
                iter_model = self._wrap_iter_model(
                    agent.pydantic_agent.model,
                    context,
                    agent,
                    trace_collector=trace_collector,
                )
                # effective_prompt is None on the resume path (baked into the
                # checkpointed message_history already, per AgentInput.resuming);
                # otherwise it's the new user turn, which must be sent alongside
                # message_history -- not replaced by it -- or a continuing
                # session's new prompt is silently dropped and the model
                # receives only stale history with no new question to answer.
                run_iter = agent.pydantic_agent.iter(
                    effective_prompt,
                    message_history=message_history,
                    deps=deps,
                    toolsets=toolsets,
                    model=iter_model,
                )

                timed_out = False
                iter_started = time.monotonic()
                try:
                    async with self._span("agent.model"):
                        async with run_iter as run:
                            current_run = run
                            try:
                                while True:
                                    try:
                                        if timeout is not None:
                                            remaining = timeout - (
                                                time.monotonic() - iter_started
                                            )
                                            if remaining <= 0:
                                                timed_out = True
                                                break
                                            node = await asyncio.wait_for(
                                                run.__anext__(), remaining
                                            )
                                        else:
                                            node = await run.__anext__()
                                    except StopAsyncIteration:
                                        break
                                    except asyncio.TimeoutError:
                                        timed_out = True
                                        break
                                    except asyncio.CancelledError:
                                        if (
                                            timeout is not None
                                            and (time.monotonic() - iter_started)
                                            >= timeout
                                        ):
                                            timed_out = True
                                            break
                                        raise

                                    await cancellation.raise_if_cancelled()

                                    model = iter_model or agent.pydantic_agent.model
                                    # Only skip streaming for MODEL REQUEST nodes when
                                    # the model itself cannot stream. Tool-execution
                                    # nodes (CallToolsNode) always stream -- their
                                    # event stream (FunctionToolCallEvent /
                                    # FunctionToolResultEvent) does not depend on
                                    # the model's streaming support, only on
                                    # pydantic-ai's graph iteration.
                                    if not model_supports_streaming(
                                        model
                                    ) and PydanticAgent.is_model_request_node(node):
                                        request_key = _next_request_key()
                                        state = run.ctx.state
                                        history_count = len(
                                            tuple(
                                                getattr(
                                                    state,
                                                    "message_history",
                                                    (),
                                                )
                                                or ()
                                            )
                                        )
                                        try:
                                            node = await node.run(run.ctx)
                                        finally:
                                            if usage_sink is not None:
                                                response = _latest_model_response(
                                                    run.ctx.state,
                                                    after_count=history_count,
                                                )
                                                if response is not None:
                                                    await _observe_model_usage(
                                                        request_key,
                                                        _request_usage_from_provider(
                                                            response.usage
                                                        ),
                                                        response,
                                                    )
                                        continue

                                    if PydanticAgent.is_model_request_node(node):
                                        request_key = _next_request_key()
                                        accumulated_text += (
                                            await self._forward_model_stream(
                                                node,
                                                run.ctx,
                                                cancellation=cancellation,
                                                live_events=live_events,
                                                request_key=request_key,
                                                observe_usage=_observe_model_usage,
                                            )
                                        )
                                    elif PydanticAgent.is_call_tools_node(node):
                                        await self._forward_tool_stream(
                                            node,
                                            run.ctx,
                                            cancellation=cancellation,
                                            live_events=live_events,
                                        )
                            except RunPaused as paused:
                                if environ.debug:
                                    logger.debug(
                                        "run %s paused: tool=%s call_id=%s approval=%s reason=%s",
                                        context.run_id,
                                        paused.tool_name,
                                        paused.tool_call_id,
                                        paused.approval_id,
                                        paused.reason,
                                    )
                                snapshot = None
                                if trace_collector is not None:
                                    snapshot = await _snapshot(
                                        ExecutionRunStatus.PAUSED,
                                        usage=_captured_usage(),
                                    )
                                # The engine does NOT publish a "paused" state
                                # event: per the state-event split, state events
                                # (paused/completed/failed/cancelled) are the
                                # Coordinator's job, published only AFTER the
                                # durable commit succeeds. The engine publishes
                                # only process events (text_delta / tool_* /
                                # model_progress); the paused outcome carries
                                # the pause descriptor up to the Coordinator,
                                # which commits and then publishes "paused".
                                return AgentPaused(
                                    request=PauseRequest(
                                        approval_id=paused.approval_id,
                                        tool_call_id=paused.tool_call_id,
                                        tool_name=paused.tool_name,
                                        reason=paused.reason,
                                        arguments=paused.arguments,
                                        idempotency_key=paused.idempotency_key,
                                        binding=paused.binding,
                                    ),
                                    usage=AgentUsage(
                                        input_tokens=_captured_usage().input_tokens,
                                        output_tokens=_captured_usage().output_tokens,
                                        total_tokens=_captured_usage().total_tokens,
                                        cache_write_tokens=_captured_usage().cache_write_tokens,
                                        cache_read_tokens=_captured_usage().cache_read_tokens,
                                        total_cost=_captured_usage().total_cost,
                                    ),
                                    snapshot=snapshot,
                                )
                            else:
                                if not timed_out:
                                    result = run.result
                except asyncio.TimeoutError:
                    if (
                        timeout is not None
                        and (time.monotonic() - iter_started) >= timeout
                    ):
                        timed_out = True
                    else:
                        raise
                except asyncio.CancelledError:
                    if (
                        timeout is not None
                        and (time.monotonic() - iter_started) >= timeout
                    ):
                        timed_out = True
                    else:
                        raise
                except Exception as iter_exc:
                    if timed_out:
                        if environ.debug:
                            logger.debug(
                                "run %s masked secondary error during timeout: %s: %s",
                                context.run_id,
                                type(iter_exc).__name__,
                                iter_exc,
                            )
                    else:
                        raise

                await cancellation.raise_if_cancelled()

                if timed_out:
                    raise ModelRoutingError("model timeout")

                captured = _captured_usage()
                output = result.output if result is not None else accumulated_text
                await self._enforce_usage_policy(agent.spec.model, captured)

                run_result = RunResult(
                    output=output,
                    token_usage={
                        "input_tokens": captured.input_tokens,
                        "output_tokens": captured.output_tokens,
                    },
                )
                snapshot = None
                if trace_collector is not None:
                    snapshot = await _snapshot(
                        ExecutionRunStatus.COMPLETED,
                        final_output=output,
                        usage=ExecutionRunUsage(
                            input_tokens=captured.input_tokens,
                            output_tokens=captured.output_tokens,
                            total_tokens=captured.total_tokens,
                            cache_write_tokens=captured.cache_write_tokens,
                            cache_read_tokens=captured.cache_read_tokens,
                            total_cost=captured.total_cost,
                        ),
                    )

                if self._middleware_pipeline is not None:
                    await cancellation.raise_if_cancelled()
                    await self._middleware_pipeline.run_after_run(context, run_result)

                if metrics is not None:
                    try:
                        metrics.counter("agent.run.completed", attributes=run_attrs)
                        metrics.histogram(
                            "agent.run.duration_ms",
                            value=round((time.monotonic() - started) * 1000, 3),
                            attributes=run_attrs,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "success metrics failed for run %s", context.run_id
                        )

                if environ.debug:
                    logger.debug(
                        "run %s completed: input=%s output=%s",
                        context.run_id,
                        captured.input_tokens,
                        captured.output_tokens,
                    )
                return AgentCompleted(
                    result=run_result,
                    usage=AgentUsage(
                        input_tokens=captured.input_tokens,
                        output_tokens=captured.output_tokens,
                        total_tokens=captured.total_tokens,
                        total_cost=captured.total_cost,
                        cache_write_tokens=captured.cache_write_tokens,
                        cache_read_tokens=captured.cache_read_tokens,
                    ),
                    snapshot=snapshot,
                )
        except asyncio.CancelledError:
            if cancellation.is_cancelled():
                snapshot = None
                if trace_collector is not None:
                    snapshot = await _snapshot(
                        ExecutionRunStatus.CANCELLED,
                        usage=_captured_usage(),
                        capture_state=MessageCaptureState.PARTIAL,
                    )
                if environ.debug:
                    logger.debug("run %s cancelled", context.run_id)
                return AgentCancelled(
                    reason=None,
                    usage=AgentUsage(
                        input_tokens=_captured_usage().input_tokens,
                        output_tokens=_captured_usage().output_tokens,
                        total_tokens=_captured_usage().total_tokens,
                        total_cost=_captured_usage().total_cost,
                        cache_write_tokens=_captured_usage().cache_write_tokens,
                        cache_read_tokens=_captured_usage().cache_read_tokens,
                    ),
                    snapshot=snapshot,
                )
            raise
        except _EXPECTED_RUN_FAILURES as exc:
            # A malformed tool schema is a contract/config violation, not a
            # per-run tool failure -- even though it subclasses ToolError, it
            # must propagate instead of becoming an AgentFailed outcome.
            if isinstance(exc, ToolSchemaError):
                raise
            safe_error = redact_exception(exc)
            if metrics is not None:
                try:
                    metrics.counter(
                        "agent.run.failed",
                        attributes={
                            **run_attrs,
                            "error_type": type(exc).__name__,
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "failure metrics failed for run %s", context.run_id
                    )
            snapshot = None
            if trace_collector is not None:
                snapshot = await _snapshot(
                    ExecutionRunStatus.FAILED,
                    usage=_captured_usage(),
                    capture_state=MessageCaptureState.PARTIAL,
                )
            logger.warning(
                "run %s failed: %s: %s",
                context.run_id,
                type(exc).__name__,
                safe_error,
                exc_info=environ.debug,
            )
            return AgentFailed(
                error=RunErrorInfo(error_type=type(exc).__name__, message=safe_error),
                retryable=False,
                usage=AgentUsage(
                    input_tokens=_captured_usage().input_tokens,
                    output_tokens=_captured_usage().output_tokens,
                    total_tokens=_captured_usage().total_tokens,
                    total_cost=_captured_usage().total_cost,
                    cache_write_tokens=_captured_usage().cache_write_tokens,
                    cache_read_tokens=_captured_usage().cache_read_tokens,
                ),
                snapshot=snapshot,
            )
