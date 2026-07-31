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

from typing import TYPE_CHECKING, Any
import asyncio
import contextlib
import dataclasses
import logging
import time
from ..errors import MCPToolError, ModelInvocationDeniedError, ModelOutputValidationError, ModelPolicyExceededError, ModelResultDeniedError, ModelRoutingError, RunPaused, RuntimeInitializationError, ToolError, ToolSchemaError
from .middleware.pipeline import MiddlewarePipeline
from ..observability.tracing import use_span
from .prompt.builder import PromptBuilder
from ..governance.policy.engine import ToolContext
from ..governance.security.redact import redact_exception
from ..execution.domain import RunErrorInfo
from .models import RunResult
from ..model.recording import SemanticRecordingModel
from ..execution.domain import RunStatus as ExecutionRunStatus, RunUsage as ExecutionRunUsage
from .dependencies import AgentDependencies
from .models import AgentCancelled, AgentCompleted, AgentFailed, model_supports_streaming, AgentPaused, AgentUsage, PauseRequest

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


_LOGGER = logging.getLogger(__name__)


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
        from pydantic_ai.messages import (
            FunctionToolCallEvent,
            FunctionToolResultEvent,
            PartDeltaEvent,
            PartStartEvent,
            TextPart,
            TextPartDelta,
            ToolReturnPart,
        )

        metrics = self._metrics
        run_attrs = {"run_id": context.run_id, "session_id": context.session_id}
        started = time.monotonic()

        current_run = None

        def resume_messages() -> "tuple[Any, ...]":
            current = current_run
            if current is None:
                return ()
            return self._trace_codec.encode_model_messages(
                tuple(current.all_messages())
            )

        async def _snapshot(
            status: ExecutionRunStatus,
            *,
            messages: "tuple[Any, ...]",
            final_output: Any = None,
            usage: "ExecutionRunUsage | None" = None,
        ) -> Any:
            if trace_collector is None:
                return None
            return await trace_collector.build_snapshot(
                resume_messages=messages,
                final_output=final_output,
                status=status,
                usage=usage or ExecutionRunUsage(),
            )

        try:
            async with self._span("agent.run", attrs=run_attrs):
                await cancellation.raise_if_cancelled()

                # before_run middleware fires on a NEW run only -- resume skips
                # it (the initial run already ran it).
                if not input.resuming and self._middleware_pipeline is not None:
                    await cancellation.raise_if_cancelled()
                    await self._middleware_pipeline.run_before_run(context)

                message_history = input.message_history or None
                window_policy = self._session_window
                if window_policy is not None:
                    # The policy is invoked even when history is empty (a fresh
                    # session) so the wiring is observable and a policy that
                    # short-circuits on length can still record the decision.
                    before_count = len(message_history or ())
                    trimmed = await window_policy.select_messages(
                        message_history or (), agent.spec.model
                    )
                    message_history = tuple(trimmed) or None
                    from ..observability.events.payloads import PromptWindowApplied

                    await security_events.emit(
                        PromptWindowApplied(
                            policy=type(window_policy).__name__,
                            before=before_count,
                            after=len(trimmed),
                        )
                    )

                user_prompt = input.prompt
                prompt: "str | None" = None
                if not input.resuming:
                    memory_section = ""
                    memory_policy = self._effective_memory_policy()
                    if memory_policy is not None:
                        memories = await memory_policy.select_memories(
                            context, user_prompt
                        )
                        memory_section = (
                            self._prompt_formatter().format_memory(memories) or ""
                        )
                    knowledge_section = ""
                    retrieval_policy = self._effective_retrieval_policy()
                    if retrieval_policy is not None:
                        items = await retrieval_policy.retrieve(context, user_prompt)
                        knowledge_section = (
                            self._prompt_formatter().format_knowledge(items) or ""
                        )
                    prompt = PromptBuilder.build_base_prompt(
                        user_prompt=user_prompt,
                        prior_messages=(),
                        memory_section=memory_section,
                        knowledge_section=knowledge_section,
                    )

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
                    tool_descriptors = {
                        definition.descriptor.name: definition.descriptor
                        for definition in assembly.tools
                    }
                    if tool_descriptors:
                        deps = dataclasses.replace(
                            deps, tool_descriptors=tool_descriptors
                        )
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

                    total = len(assembly.tools)
                    await security_events.emit(
                        ToolExposureApplied(agent_id=agent.spec.id, total_tools=total)
                    )
                    if assembly.prompt_sections:
                        for section in assembly.prompt_sections:
                            await security_events.emit(
                                PromptCatalogInjected(
                                    agent_id=agent.spec.id, section=section
                                )
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
                iter_model = None
                if self._security_pipeline is not None:
                    from ..governance.security.secured_model import SecuredModel

                    iter_model = SecuredModel(
                        agent.pydantic_agent.model,
                        pipeline=self._security_pipeline,
                        run_id=context.run_id,
                        agent_id=agent.spec.id,
                    )
                if trace_collector is not None:
                    iter_model = SemanticRecordingModel(
                        iter_model or agent.pydantic_agent.model,
                        trace_collector,
                        self._trace_codec,
                    )
                if message_history is not None:
                    run_iter = agent.pydantic_agent.iter(
                        message_history=message_history,
                        deps=deps,
                        toolsets=toolsets,
                        model=iter_model,
                    )
                else:
                    run_iter = agent.pydantic_agent.iter(
                        effective_prompt,
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
                                    if (
                                        not model_supports_streaming(model)
                                        and PydanticAgent.is_model_request_node(node)
                                    ):
                                        node = await node.run(run.ctx)
                                        continue

                                    if PydanticAgent.is_model_request_node(node):
                                        try:
                                            async with node.stream(
                                                run.ctx
                                            ) as request_stream:
                                                async for ev in request_stream:
                                                    text = None
                                                    if isinstance(
                                                        ev, PartStartEvent
                                                    ) and isinstance(ev.part, TextPart):
                                                        text = ev.part.content
                                                    elif isinstance(
                                                        ev, PartDeltaEvent
                                                    ) and isinstance(
                                                        ev.delta, TextPartDelta
                                                    ):
                                                        text = ev.delta.content_delta
                                                    if text:
                                                        accumulated_text += text
                                                        await live_events.publish(
                                                            {
                                                                "type": "text",
                                                                "text": text,
                                                            }
                                                        )
                                                        await cancellation.raise_if_cancelled()
                                        except BaseException:
                                            raise
                                    elif PydanticAgent.is_call_tools_node(node):
                                        try:
                                            async with node.stream(
                                                run.ctx
                                            ) as tool_stream:
                                                async for ev in tool_stream:
                                                    tool_event = None
                                                    if isinstance(
                                                        ev, FunctionToolCallEvent
                                                    ):
                                                        tool_event = {
                                                            "type": "tool",
                                                            "name": ev.part.tool_name,
                                                            "phase": "start",
                                                            "ok": None,
                                                        }
                                                    elif isinstance(
                                                        ev, FunctionToolResultEvent
                                                    ):
                                                        tool_event = {
                                                            "type": "tool",
                                                            "name": ev.part.tool_name,
                                                            "phase": "end",
                                                            "ok": isinstance(
                                                                ev.part, ToolReturnPart
                                                            ),
                                                        }
                                                    if tool_event is not None:
                                                        await live_events.publish(
                                                            tool_event
                                                        )
                                                        await cancellation.raise_if_cancelled()
                                        except BaseException:
                                            raise
                            except RunPaused as paused:
                                snapshot = None
                                if trace_collector is not None:
                                    snapshot = await _snapshot(
                                        ExecutionRunStatus.PAUSED,
                                        messages=self._trace_codec.encode_model_messages(tuple(run.all_messages())),
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
                except Exception:
                    if timed_out:
                        pass
                    else:
                        raise

                await cancellation.raise_if_cancelled()

                if timed_out:
                    raise ModelRoutingError("model timeout")

                output = result.output if result is not None else accumulated_text
                usage = result.usage if result is not None else None
                max_tokens = agent.spec.model.max_tokens
                if max_tokens is not None and usage is not None:
                    used = usage.input_tokens + usage.output_tokens
                    if used > max_tokens:
                        raise ModelPolicyExceededError(
                            f"max_tokens exceeded: used {used} > max_tokens {max_tokens}",
                            kind="max_tokens",
                        )
                budget = agent.spec.model.budget
                if budget is not None and usage is not None:
                    if self._pricing_provider is None:
                        raise ModelPolicyExceededError(
                            "ModelPolicy.budget is set but no ModelPricingProvider "
                            "is wired; refusing to run without a cost limit",
                            kind="budget",
                        )
                    pricing = await self._pricing_provider.get_pricing(
                        agent.spec.model.primary
                    )
                    if pricing is None:
                        raise ModelPolicyExceededError(
                            f"ModelPolicy.budget set but model "
                            f"{agent.spec.model.primary!r} has no pricing; "
                            f"refusing to run without a cost limit",
                            kind="budget",
                        )
                    cost = pricing.cost(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
                    if cost > budget:
                        raise ModelPolicyExceededError(
                            f"cost budget exceeded: {cost} > budget {budget}",
                            kind="budget",
                        )

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
                        messages=self._trace_codec.encode_model_messages(tuple(run.all_messages())),
                        final_output=output,
                        usage=ExecutionRunUsage(
                            input_tokens=usage.input_tokens if usage else 0,
                            output_tokens=usage.output_tokens if usage else 0,
                            total_tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
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
                        _LOGGER.exception(
                            "success metrics failed for run %s", context.run_id
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
                        messages=resume_messages(),
                    )
                return AgentCancelled(reason=None, usage=AgentUsage(), snapshot=snapshot)
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
                    _LOGGER.exception(
                        "failure metrics failed for run %s", context.run_id
                    )
            snapshot = None
            if trace_collector is not None:
                snapshot = await _snapshot(
                    ExecutionRunStatus.FAILED,
                    messages=resume_messages(),
                )
            return AgentFailed(
                error=RunErrorInfo(error_type=type(exc).__name__, message=safe_error),
                retryable=False,
                usage=AgentUsage(),
                snapshot=snapshot,
            )
