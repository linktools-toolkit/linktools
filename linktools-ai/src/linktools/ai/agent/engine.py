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
* publishes ONLY process dict-events through the injected ``live_events``
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
from typing import TYPE_CHECKING, Any

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
from ..governance.policy.engine import ToolContext
from ..governance.security.redact import redact_exception
from ..model.recording import SemanticRecordingModel
from ..observability.tracing import use_span
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
    from ..model.pricing import ModelPricingProvider


logger = environ.get_logger("ai.agent.engine")


# Exception types that count as EXPECTED provider/model/tool failures for the
# pure execution loop's outcome classification. Only these are turned into an
# AgentFailed outcome; every other exception (configuration / invariant /
# protocol violations such as RuntimeInitializationError, RunInvariantError,
# AgentAssemblyError, MCPConnectionError, ModelRetryConfigurationError,
# and all unknown programming errors like TypeError/AttributeError/KeyError)
# propagates unchanged -- the spec explicitly forbids an except-Exception
# catch-all that swallows them into a FAILED outcome. ToolError is the base for
# runtime tool-execution failures; the schema-definition subfamily is carved
# out (re-raised) below because a malformed tool schema is a contract/config
# violation, not a per-run tool failure.
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
                    await live_events.publish({"type": kind, "text": chunk})
                    await cancellation.raise_if_cancelled()
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
                    tool_event = {
                        "type": "tool",
                        "name": ev.part.tool_name,
                        "phase": "start",
                        "ok": None,
                    }
                elif isinstance(ev, FunctionToolResultEvent):
                    tool_event = {
                        "type": "tool",
                        "name": ev.part.tool_name,
                        "phase": "end",
                        "ok": isinstance(ev.part, ToolReturnPart),
                    }
                if tool_event is not None:
                    await live_events.publish(tool_event)
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
        user_prompt: "str | None",
    ) -> "str | None":
        """Build the base prompt for a NEW turn: memory section (from the wired
        memory policy) + knowledge section (from the wired retrieval policy) +
        the user's prompt, combined by PromptBuilder. Returns None on the resume
        path (the prompt is already baked into the checkpointed history)."""
        memory_section = ""
        memory_policy = self._effective_memory_policy()
        if memory_policy is not None:
            memories = await memory_policy.select_memories(context, user_prompt)
            memory_section = self._prompt_formatter().format_memory(memories) or ""
        knowledge_section = ""
        retrieval_policy = self._effective_retrieval_policy()
        if retrieval_policy is not None:
            items = await retrieval_policy.retrieve(context, user_prompt)
            knowledge_section = self._prompt_formatter().format_knowledge(items) or ""
        return PromptBuilder.build_base_prompt(
            user_prompt=user_prompt,
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
        ``max_tokens`` or ``budget``. ``usage`` is the pydantic-ai RunUsage from
        the completed run; nothing happens when it is None (no usage recorded)."""
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
        if self._pricing_provider is None:
            raise ModelPolicyExceededError(
                "ModelPolicy.budget is set but no ModelPricingProvider "
                "is wired; refusing to run without a cost limit",
                kind="budget",
            )
        pricing = await self._pricing_provider.get_pricing(model_policy.primary)
        if pricing is None:
            raise ModelPolicyExceededError(
                f"ModelPolicy.budget set but model {model_policy.primary!r} has "
                f"no pricing; refusing to run without a cost limit",
                kind="budget",
            )
        cost = pricing.cost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
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
                                        node = await node.run(run.ctx)
                                        continue

                                    if PydanticAgent.is_model_request_node(node):
                                        accumulated_text += (
                                            await self._forward_model_stream(
                                                node,
                                                run.ctx,
                                                cancellation=cancellation,
                                                live_events=live_events,
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
                                    usage=AgentUsage(),
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

                output = result.output if result is not None else accumulated_text
                usage = result.usage if result is not None else None
                await self._enforce_usage_policy(agent.spec.model, usage)

                run_result = RunResult(
                    output=output,
                    token_usage={
                        "input_tokens": usage.input_tokens if usage else 0,
                        "output_tokens": usage.output_tokens if usage else 0,
                    },
                )
                snapshot = None
                if trace_collector is not None:
                    snapshot = await _snapshot(
                        ExecutionRunStatus.COMPLETED,
                        final_output=output,
                        usage=ExecutionRunUsage(
                            input_tokens=usage.input_tokens if usage else 0,
                            output_tokens=usage.output_tokens if usage else 0,
                            total_tokens=(usage.input_tokens + usage.output_tokens)
                            if usage
                            else 0,
                            cache_write_tokens=usage.cache_write_tokens if usage else 0,
                            cache_read_tokens=usage.cache_read_tokens if usage else 0,
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
                        usage.input_tokens if usage else 0,
                        usage.output_tokens if usage else 0,
                    )
                return AgentCompleted(
                    result=run_result,
                    usage=AgentUsage(
                        input_tokens=usage.input_tokens if usage else 0,
                        output_tokens=usage.output_tokens if usage else 0,
                    ),
                    snapshot=snapshot,
                )
        except asyncio.CancelledError:
            if cancellation.is_cancelled():
                snapshot = None
                if trace_collector is not None:
                    snapshot = await _snapshot(
                        ExecutionRunStatus.CANCELLED,
                        capture_state=MessageCaptureState.PARTIAL,
                    )
                if environ.debug:
                    logger.debug("run %s cancelled", context.run_id)
                return AgentCancelled(
                    reason=None, usage=AgentUsage(), snapshot=snapshot
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
                usage=AgentUsage(),
                snapshot=snapshot,
            )
