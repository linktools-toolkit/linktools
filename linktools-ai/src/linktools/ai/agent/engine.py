#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine: the Store-free per-invocation model/tool loop (FS-29).

The engine owns ONLY the prompt-build → model/tool drive → outcome path. It
touches NO Run-lifecycle Store (no RunStore / SessionStore / EventStore /
CheckpointStore / ApprovalStore), no commit_coordinator, no run_controller --
RunRecord create/transition, checkpoint/session/approval persistence, pause/
cancel/stream events, and the cross-store commit are RunCoordinator's sole
job (see run/coordinator.py). The single public entry point is
:meth:`AgentEngine.execute_pure`, which returns a discriminated-union
``AgentExecutionOutcome`` (AgentCompleted / AgentPaused / AgentFailed /
AgentCancelled) -- never a Store write.

execute_pure drives ``agent.pydantic_agent.iter()`` and:
* runs the before_run/after_run middleware hooks (on a new, non-resuming run)
  via the wired MiddlewarePipeline -- after_run fires before the outcome is
  returned, so it always precedes the RunCoordinator commit;
* builds the model prompt via :class:`~linktools.ai.prompt.builder.PromptBuilder`
  (memory + knowledge sections from their async policies, then capability-
  resolved sections folded in via PromptBuilder.combine). Prior-turn history
  arrives as native pydantic-ai ``message_history`` on AgentInput (loaded by
  RunCoordinator), NOT via a Store read here;
* threads the per-Run ToolContext through pydantic-ai dependency injection
  (``deps=AgentDependencies(tool_context=...)`` -> ``ctx.deps``), never a
  mutable capability field;
* enforces ModelPolicy.timeout_seconds / max_tokens / budget;
* classifies exceptions via the narrow ``_EXPECTED_RUN_FAILURES`` allowlist
  (model routing/policy/output denials, ToolError, MCPToolError) -> AgentFailed;
  everything else (config/invariant/protocol violations, programming errors)
  propagates unchanged;
* publishes ONLY process dict-events through the injected ``live_events``
  sink (text / tool / model_progress); state events (paused / completed /
  failed / cancelled) are the RunCoordinator's job, published only AFTER the
  durable commit succeeds. Security+observability events route through the
  injected ``security_events`` sink (no direct EventStore reference).

Optional Observability: when ``observability`` is wired, execute_pure wraps the
loop in an outer ``agent.run`` span and the iter() drive in a nested
``agent.model`` span (parented via the tracing contextvar). When ``metrics`` is
wired, records ``counter("agent.run.completed"/"agent.run.failed")`` and
``histogram("agent.run.duration_ms")``. Both default to None (no-op)."""

import asyncio
import contextlib
import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

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
from ..middleware.pipeline import MiddlewarePipeline
from ..observability.tracing import use_span
from ..prompt.builder import PromptBuilder
from ..governance.policy.engine import ToolContext
from ..governance.security.redact import redact_exception
from ..run.cancellation import CancellationToken
from ..run.context import RunContext
from ..run.models import (
    RunErrorInfo,
    RunResult,
)
from ..session.recorder import SessionRecorder
from .checkpoint import serialize_messages
from .dependencies import AgentDependencies
from .models import (
    AgentCancelled,
    AgentCompleted,
    AgentExecutionOutcome,
    AgentFailed,
    model_supports_streaming,
    AgentInput,
    AgentPaused,
    CompiledAgent,
    PauseRequest,
    RunUsage,
)

if TYPE_CHECKING:

    from pydantic_ai.toolsets import AbstractToolset

    from ..capability.resolver import CapabilityResolver
    from ..capability.models import CapabilityRuntimeOptions
    from ..sandbox.protocols import Sandbox
    from ..retrieval.retriever import Retriever
    from ..memory.store import MemoryStore
    from ..observability.metrics import ObservabilityMetrics
    from ..observability.tracing import ObservabilitySink
    from ..run.live_events import RunLiveEventSink, SecurityEventSink


_LOGGER = logging.getLogger(__name__)

# Exception types that count as EXPECTED provider/model/tool failures for the
# pure execution loop's outcome classification. Only these are turned into an
# AgentFailed outcome; every other exception (configuration / invariant /
# protocol violations such as RuntimeInitializationError, RunInvariantError,
# CapabilityResolutionError, MCPConnectionError, ModelRetryConfigurationError,
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
        memory_store: "MemoryStore | None" = None,
        retriever: "Retriever | None" = None,
        observability: "ObservabilitySink | None" = None,
        metrics: "ObservabilityMetrics | None" = None,
        sandbox: "Sandbox | None" = None,
        capability_resolver: "CapabilityResolver | None" = None,
        capability_options: "CapabilityRuntimeOptions | None" = None,
        security_pipeline: Any = None,
        baseline_policy: Any = None,
        tool_policy_provider: Any = None,
        managed_tool_executor: Any = None,
        security_audit_failure_mode: Any = "fail_closed",
        pricing_provider: Any = None,
    ) -> None:
        self._middleware_pipeline = middleware_pipeline
        self._memory_store = memory_store
        self._retriever = retriever
        self._observability = observability
        self._metrics = metrics
        self._sandbox = sandbox
        self._capability_resolver = capability_resolver
        self._capability_options = capability_options
        self._security_pipeline = security_pipeline
        self._baseline_policy = baseline_policy
        self._tool_policy_provider = tool_policy_provider
        self._tool_executor_for_managed = managed_tool_executor
        self._security_audit_failure_mode = security_audit_failure_mode
        self._pricing_provider = pricing_provider
        self._session_recorder = SessionRecorder()

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
        opts = self._capability_options
        if opts is not None and opts.memory_policy is not None:
            return opts.memory_policy
        if self._memory_store is not None:
            from .context_policies import DefaultMemoryPolicy

            return DefaultMemoryPolicy(self._memory_store)
        return None

    def _effective_retrieval_policy(self):
        opts = self._capability_options
        if opts is not None and opts.retrieval_policy is not None:
            return opts.retrieval_policy
        if self._retriever is not None:
            from .context_policies import DefaultRetrievalPolicy

            return DefaultRetrievalPolicy(self._retriever)
        return None

    def _prompt_formatter(self):
        opts = self._capability_options
        if opts is not None and opts.prompt_context_formatter is not None:
            return opts.prompt_context_formatter
        from .context_policies import DefaultPromptContextFormatter

        return DefaultPromptContextFormatter()

    async def execute_pure(
        self,
        agent: CompiledAgent,
        input: AgentInput,
        context: RunContext,
        *,
        cancellation: CancellationToken,
        live_events: "RunLiveEventSink",
        security_events: "SecurityEventSink",
    ) -> AgentExecutionOutcome:
        """The target pure execution loop. Touches NO run_store/
        session_store/commit_coordinator/run_controller -- RunCoordinator
        owns RunRecord creation/transition, checkpoint/session persistence,
        and execution claim/heartbeat/fencing; this method only assembles
        the prompt, drives the model/tool loop, and returns a value
        (never a Store write).

        Every security/observability event the capability and tool layers
        emit (tool-pipeline decisions, exposure/catalog/window markers) flows
        through the injected ``security_events`` sink via a
        ``SecurityEventSinkEmitter`` -- this method holds no EventStore
        reference.

        Exception classification (no ``except Exception: return FAILED``
        catch-all): an explicit Run cancellation (``cancellation.
        is_cancelled()`` true when a CancelledError surfaces) returns
        AgentCancelled; an external CancelledError re-raises after cleanup;
        the narrow ``_EXPECTED_RUN_FAILURES`` set (model routing/policy/output
        denials, runtime tool failures, MCP tool errors) is returned as
        AgentFailed. Everything else -- configuration/invariant/protocol
        violations (RuntimeInitializationError, RunInvariantError,
        CapabilityResolutionError, MCPConnectionError, ToolSchemaError, ...)
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

        try:
            async with self._span("agent.run", attrs=run_attrs):
                await cancellation.raise_if_cancelled()

                # before_run middleware fires on a NEW run only -- resume skips
                # it (the initial run already ran it).
                if not input.resuming and self._middleware_pipeline is not None:
                    await cancellation.raise_if_cancelled()
                    await self._middleware_pipeline.run_before_run(context)

                message_history = input.message_history or None
                window_policy = (
                    self._capability_options.session_window_policy
                    if self._capability_options is not None
                    else None
                )
                if window_policy is not None:
                    # The policy is invoked even when history is empty (a fresh
                    # session) so the wiring is observable and a policy that
                    # short-circuits on length can still record the decision.
                    before_count = len(message_history or ())
                    trimmed = await window_policy.select_messages(
                        message_history or (), agent.spec.model
                    )
                    message_history = tuple(trimmed) or None
                    from ..events.payloads import PromptWindowApplied

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
                cap_bundle = None
                has_resolver = self._capability_resolver is not None
                builtin_flag = getattr(
                    self._capability_options, "enable_builtin_tools", None
                )
                needs_default = (
                    agent.spec.tools is None
                    and deps.sandbox is not None
                    and builtin_flag is not False
                )
                from ..capability.models import requires_capability_resolver

                requires_tools = requires_capability_resolver(
                    tools=(agent.spec.tools if not needs_default else ("builtin",)),
                    sandbox=deps.sandbox,
                )
                if requires_tools and not has_resolver:
                    raise RuntimeInitializationError(
                        "AgentEngine requires a CapabilityResolver to resolve tools"
                    )
                if requires_tools and self._tool_executor_for_managed is None:
                    raise RuntimeInitializationError(
                        "AgentEngine requires a GovernedToolInvoker for managed tool execution"
                    )
                if requires_tools:
                    from ..capability.exposure import CapabilityToolExposurePolicy
                    from ..capability.provider import CapabilityContext

                    exposure = (
                        self._capability_options.tool_exposure
                        if self._capability_options is not None
                        else CapabilityToolExposurePolicy()
                    )
                    from ..run.live_events import SecurityEventSinkEmitter

                    # The security/observability emitter for this Run is built
                    # from the injected SecurityEventSink (durable, RunCoordinator-
                    # owned) -- NOT from a held EventStore. Capabilities/tools/
                    # governance depend only on this emitter (SecurityEventEmitter
                    # shape), with no direct EventStore reference.
                    security_emitter = SecurityEventSinkEmitter(security_events)

                    cap_ctx = CapabilityContext(
                        agent_id=agent.spec.id,
                        exposure_policy=exposure,
                        sandbox=deps.sandbox,
                        run_id=context.run_id,
                        root_run_id=context.root_run_id,
                        parent_run_id=context.parent_run_id,
                        session_id=context.session_id,
                        security_event_emitter=security_emitter,
                        user_id=context.user_id,
                        tenant_id=context.tenant_id,
                        workspace=context.workspace,
                    )
                    if needs_default:
                        from ..agent.spec import ToolRef as _TR

                        effective_spec = dataclasses.replace(
                            agent.spec,
                            tools=(_TR(kind="builtin", name="*"),),
                        )
                    else:
                        effective_spec = agent.spec
                    cap_bundle = await self._capability_resolver.resolve(
                        effective_spec, cap_ctx
                    )
                    from ..capability.resolver import _contribution_descriptors

                    descriptor_lookup = {
                        d.name: d
                        for contrib in cap_bundle.tool_contributions
                        for d in _contribution_descriptors(contrib)
                    }
                    if descriptor_lookup:
                        deps = dataclasses.replace(
                            deps, descriptor_lookup=descriptor_lookup
                        )
                    effective_pipeline = self._security_pipeline
                    if cap_bundle.tool_contributions:
                        from ..tool.pydantic import (
                            ManagedToolsetWrapper,
                            build_managed_toolset,
                        )

                        wrap_kw = dict(
                            security_pipeline=effective_pipeline,
                            tool_executor=self._tool_executor_for_managed,
                            policy_provider=self._tool_policy_provider,
                            baseline_policy=self._baseline_policy,
                            run_context=context,
                            security_audit_failure_mode=self._security_audit_failure_mode,
                            security_event_emitter=security_emitter,
                        )
                        for contrib in cap_bundle.tool_contributions:
                            for md in contrib.tools:
                                toolsets.append(
                                    ManagedToolsetWrapper(
                                        build_managed_toolset(md),
                                        descriptors={md.descriptor.name: md.descriptor},
                                        **wrap_kw,
                                    )
                                )
                if cap_bundle is not None:
                    from ..events.payloads import (
                        PromptCatalogInjected,
                        ToolExposureApplied,
                    )

                    total = 0
                    for c in cap_bundle.tool_contributions:
                        total += len(c.tools)
                    await security_events.emit(
                        ToolExposureApplied(agent_id=agent.spec.id, total_tools=total)
                    )
                    if cap_bundle.prompt_sections:
                        for section in cap_bundle.prompt_sections:
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
                    capability_sections=(
                        cap_bundle.prompt_sections if cap_bundle is not None else {}
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
                                    if not model_supports_streaming(model) and hasattr(node, "run"):
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
                                checkpoint_payload = serialize_messages(
                                    run.all_messages()
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
                                    messages=self._session_recorder.paused_messages(
                                        run_id=context.run_id
                                    ),
                                    checkpoint_payload=checkpoint_payload,
                                    usage=RunUsage(),
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

                messages_to_append = self._session_recorder.completed_messages(
                    user_prompt=input.prompt, output=output, run_id=context.run_id
                )
                checkpoint_payload = serialize_messages(run.all_messages())
                run_result = RunResult(
                    output=output,
                    token_usage={
                        "input_tokens": usage.input_tokens if usage else 0,
                        "output_tokens": usage.output_tokens if usage else 0,
                    },
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
                    messages=messages_to_append,
                    checkpoint_payload=checkpoint_payload,
                    usage=RunUsage(
                        input_tokens=usage.input_tokens if usage else 0,
                        output_tokens=usage.output_tokens if usage else 0,
                    ),
                )
        except asyncio.CancelledError:
            if cancellation.is_cancelled():
                return AgentCancelled(reason=None, usage=RunUsage())
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
            return AgentFailed(
                error=RunErrorInfo(error_type=type(exc).__name__, message=safe_error),
                retryable=False,
                usage=RunUsage(),
            )
