#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentRunner: owns the per-invocation lifecycle -- Run state transitions,
Session history load/append, runner-driven Middleware hooks (before_run/after_run/
on_error), Event publication, Checkpoint save. The 4 pydantic-ai-intercepted hooks
fire via MiddlewareCapability, enabled by passing deps=AgentDependencies(...) to
agent.pydantic_agent.iter() -- the per-Run ToolContext travels through
pydantic-ai's dependency injection (ctx.deps), not a mutable capability field.

One execute() async generator is the SINGLE lifecycle. run() and run_stream()
both delegate to it.

* ``execute()`` drives ``agent.pydantic_agent.iter()`` and yields the dict-event
  shape the CLI REPL consumes:
    - ``{"type": "text", "text": <delta>}`` -- incremental answer text.
    - ``{"type": "tool", "name": <tool>, "phase": "start"|"end", "ok": <bool|None>}``
    - ``{"type": "paused", "run_id": ..., "approval_id": ...}``
  It owns EVERY shared concern: RunRecord create + transition, RunStarted event,
  before_run middleware, prompt build (session history + Memory + Knowledge),
  per-Run ToolContext via AgentDependencies, checkpoint save, SUCCEEDED/FAILED/
  CANCELLED/WAITING_APPROVAL transitions, timeout (wait_for around the model
  call), max_tokens budget, after_run/on_error middleware, RunCompleted/RunFailed
  events, and (when wired) observability spans + metrics.

* ``run()`` is a collector: it consumes execute() entirely, detects the paused
  event (and re-raises RunPaused so non-streaming callers get the signal), then
  reads the final RunResult back from the RunStore (execute() already populated
  it via the SUCCEEDED transition).

* ``run_stream()`` is a thin pass-through: ``async for event in execute(): yield``.

Message-history adaptation is a text-join MVP (SessionMessage.content values
prepended to the prompt). Checkpoint payload uses pydantic-ai's
ModelMessagesTypeAdapter via serialize_messages().

Optional Memory + Knowledge injection: when ``memory_store`` and/or
``retriever`` are wired, ``execute()`` queries them with the user prompt and
prepends ``## Memory`` / ``## Knowledge`` sections to the prompt sent to the
model. Both default to None, so existing callers see no change. Final prompt
order (when both are set and non-empty): ``## Knowledge`` on top, then
``## Memory``, then session history, then the user prompt.

Optional Observability: when ``observability`` is wired, ``execute()``
wraps the lifecycle in an outer ``agent.run`` span and the iter() drive in a
nested ``agent.model`` span (parented via the tracing contextvar). When
``metrics`` is wired, records ``counter("agent.run.completed"/"agent.run.failed")``
and ``histogram("agent.run.duration_ms")``. Both default to None, so the
default-None path is a no-op -- no spans opened, no metrics recorded."""

import asyncio
import uuid
import contextlib
import dataclasses
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..errors import (
    InvalidRunTransitionError,
    ModelPolicyExceededError,
    ModelRoutingError,
    RunConflictError,
    RunInvariantError,
    RunPaused,
)
from ..events.payloads import (
    RunFailed,
    RunStarted,
)
from ..events.context import EventContext, append_event
from ..events.store import EventStore
from ..middleware.pipeline import MiddlewarePipeline
from ..observability.tracing import use_span
from ..policy.engine import ToolContext
from ..security.redact import redact_exception
from ..run.cancellation import CancellationToken
from ..run.checkpoint import CheckpointStore
from ..run.commit import CompleteRunCommand, PauseRunCommand
from ..run.context import RunContext
from ..run.controller import RunController
from ..run.lifecycle import mark_cancelled, mark_failed
from ..run.models import (
    RunErrorInfo,
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
)
from ..run.store import RunStore
from ..session.models import MessageRole, NewSessionMessage
from ..session.store import SessionStore
from .checkpoint import serialize_messages
from .dependencies import AgentDependencies
from .models import CompiledAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets import AbstractToolset

    from ..capability.assembler import CapabilityAssembler
    from ..capability.models import CapabilityRuntimeOptions
    from ..execution.protocols import ExecutionBackend
    from ..knowledge.retriever import Retriever
    from ..session.models import SessionMessage
    from ..memory.store import MemoryStore
    from ..observability.metrics import ObservabilityMetrics
    from ..observability.tracing import ObservabilitySink


_LOGGER = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _noop_span():
    """Async context manager that yields ``None`` and does nothing -- the
    fallback for :meth:`AgentRunner._span` when observability is not wired,
    so the lifecycle body has a single ``async with`` shape regardless."""
    yield None


def _format_session_history(messages: "Sequence[SessionMessage]") -> str:
    """Render prior session messages into the MODEL prompt with role prefixes,
    so an assistant turn is not disguised as user content. This is injected
    into the model prompt only -- the persisted USER session message is always
    the caller's original prompt, never this rendering."""
    lines: "list[str]" = []
    for message in messages:
        role = message.role.value.upper()
        content = message.content
        if not isinstance(content, str):
            content = repr(content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class AgentRunner:
    def __init__(
        self,
        *,
        run_store: RunStore,
        session_store: SessionStore,
        event_store: EventStore,
        checkpoint_store: CheckpointStore,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
        memory_store: "MemoryStore | None" = None,
        retriever: "Retriever | None" = None,
        observability: "ObservabilitySink | None" = None,
        metrics: "ObservabilityMetrics | None" = None,
        run_controller: "RunController | None" = None,
        execution: "ExecutionBackend | None" = None,
        capability_assembler: "CapabilityAssembler | None" = None,
        capability_options: "CapabilityRuntimeOptions | None" = None,
        security_pipeline: Any = None,
        baseline_policy: Any = None,
        tool_policy_provider: Any = None,
        managed_tool_executor: Any = None,
        security_audit_failure_mode: Any = "fail_closed",
        commit_coordinator: Any,
        pricing_provider: Any = None,
    ) -> None:
        self._run_store = run_store
        self._session_store = session_store
        self._event_store = event_store
        self._checkpoint_store = checkpoint_store
        self._middleware_pipeline = middleware_pipeline
        self._memory_store = memory_store
        self._retriever = retriever
        self._observability = observability
        self._metrics = metrics
        # Real cancellation. When wired, execute()
        # registers its driving asyncio.Task + a fresh CancellationToken with
        # the controller so Runtime.cancel(run_id) can actually stop the run
        # (sets the token -> next raise_if_cancelled() check aborts; also
        # task.cancel() to interrupt a hanging await inside the model call).
        # When None (default), cancellation works purely through asyncio's
        # CancelledError path -- the existing behavior, no token checks at
        # execution points. Default-None preserves every existing test.
        self._run_controller = run_controller
        # The per-Runtime ExecutionBackend. execute()
        # publishes this to AgentDependencies.execution and constructs the
        # builtin file/terminal FunctionToolset from it at execution time,
        # passing it via ``agent.iter(prompt, toolsets=[...])``. ``None``
        # (default) means the compiled agent exposes no builtin tools -- a
        # conversational-only run. Holding the backend on the runner (not the
        # compiler) is what decouples AgentCompiler from the filesystem.
        self._execution = execution
        # Capability Runtime: when an AgentSpec declares non-
        # empty tools and an assembler is wired, execute() resolves those tools
        # into prompt sections + toolsets via the capability providers. Default
        # None means empty tools resolve to the default builtin toolset when an
        # execution backend is present.
        self._capability_assembler = capability_assembler
        self._capability_options = capability_options
        self._security_pipeline = security_pipeline
        self._baseline_policy = baseline_policy
        self._tool_policy_provider = tool_policy_provider
        # The ToolExecutor every managed tool delegates to. Constructor-injected
        # (Runtime.build passes the compiler's executor) so the runner is fully
        # wired at construction -- no post-build private-field mutation.
        self._tool_executor_for_managed = managed_tool_executor
        self._security_audit_failure_mode = security_audit_failure_mode
        # RunCommitCoordinator (required). The runner NEVER writes cross-store
        # pause/complete state directly: it builds a Pause/CompleteRunCommand
        # and delegates to the coordinator, which owns the atomic commit (one
        # transaction for SQL, a journaled sequence for File). Runtime.build
        # always wires the storage-appropriate coordinator.
        self._commit_coordinator = commit_coordinator
        # Optional ModelPricingProvider: when ModelPolicy.budget is set, the
        # runner computes the per-response cost and refuses to exceed it (and
        # refuses to run at all if budget is set without pricing -- fail-closed).
        self._pricing_provider = pricing_provider

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

    def _record_success_metrics(
        self,
        *,
        attributes: "dict[str, Any]",
        started: float,
        metrics: "ObservabilityMetrics | None",
    ) -> None:
        """Best-effort success metrics. The run is already committed (SUCCEEDED)
        by the time this runs, so a metrics failure is logged and swallowed --
        never allowed to flip a committed run to FAILED or replace the
        caller's result."""
        if metrics is None:
            return
        try:
            metrics.counter("agent.run.completed", attributes=attributes)
            metrics.histogram(
                "agent.run.duration_ms",
                value=round((time.monotonic() - started) * 1000, 3),
                attributes=attributes,
            )
        except Exception:  # noqa: BLE001
            _LOGGER.exception(
                "success metrics failed for run %s", attributes.get("run_id")
            )

    async def execute(
        self,
        agent: CompiledAgent,
        request: RunInput,
        context: RunContext,
        *,
        message_history: "Sequence[ModelMessage] | None" = None,
    ) -> "AsyncIterator[dict]":
        """The SINGLE lifecycle. Drives ``agent.pydantic_agent.iter()`` and
        yields RuntimeEvent dicts (text deltas, tool events, paused).

        When ``message_history`` is provided (the resume path), the new-run
        setup is skipped entirely -- no ``create()``, no PENDING -> RUNNING
        transition, no RunStarted event, no before_run middleware, no prompt
        build (the prompt is already baked into the checkpointed history).
        Instead, ``iter()`` is driven with ``message_history=<deserialized>``
        so the pydantic-ai graph resumes from the checkpointed state.

        Both ``run()`` and ``run_stream()`` delegate here. The external API
        (run returns RunResult, run_stream yields events) is preserved by
        their respective collectors."""
        # Lazily imported: pydantic-ai symbols used inside the iter() drive.
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

        # -- Setup: create record + transition RUNNING (or read version for resume).
        # The terminal transitions reuse the version returned
        # here -- no hardcoded expected_version anywhere in this file.
        if message_history is None:
            now = datetime.now(timezone.utc)
            record = RunRecord(
                id=context.run_id,
                root_run_id=context.root_run_id,
                parent_run_id=context.parent_run_id,
                session_id=context.session_id,
                runnable_id=context.runnable_id,
                runnable_type=context.runnable_type,
                status=RunStatus.PENDING,
                input=request,
                result=None,
                error=None,
                version=1,
                created_at=now,
                started_at=None,
                finished_at=None,
            )
            created = await self._run_store.create(record)
            running = await self._run_store.transition(
                context.run_id,
                RunStatus.RUNNING,
                expected_version=created.version,
            )
            running_version = running.version
        else:
            # Resume: Runtime.resume already transitioned WAITING_APPROVAL ->
            # RUNNING; capture the current version for terminal transitions.
            current = await self._run_store.get(context.run_id)
            running_version = current.version

        # Fence this execution in persistent storage when the backend supports
        # it. Older/custom stores remain compatible, but production stores must
        # claim before any model or Tool side effect begins.
        execution_token = uuid.uuid4().hex
        claim_execution = getattr(self._run_store, "claim_execution", None)
        if claim_execution is not None:
            worker_id = f"agent-worker:{uuid.uuid4().hex}"
            await claim_execution(
                context.run_id,
                worker_id=worker_id,
                execution_token=execution_token,
            )
        heartbeat_task = None
        heartbeat_execution = getattr(self._run_store, "heartbeat_execution", None)
        if heartbeat_execution is not None and claim_execution is not None:
            async def _heartbeat():
                while True:
                    await asyncio.sleep(10.0)
                    try:
                        await heartbeat_execution(
                            context.run_id,
                            worker_id=worker_id,
                            execution_token=execution_token,
                        )
                    except Exception:
                        # Lost fencing is fail-closed: stop renewing; the
                        # owner can no longer safely commit side effects.
                        if token is not None:
                            token.cancel()
                        return
            heartbeat_task = asyncio.create_task(_heartbeat())

        # Real cancellation. When a RunController is wired,
        # register the driving asyncio.Task + a fresh CancellationToken so
        # Runtime.cancel(run_id) can actually stop this run: the token is
        # checked at the model-call execution points below, and the task is
        # cancelled to interrupt any hanging await inside the model call.
        # When ``run_controller`` is None (default), ``token`` stays None and
        # every check below is skipped -- cancellation then works purely via
        # asyncio's CancelledError path (cooperative cancellation only).
        token: "CancellationToken | None" = None
        if self._run_controller is not None:
            token = CancellationToken()
            current_task = asyncio.current_task()
            if current_task is not None:
                await self._run_controller.register(
                    context.run_id,
                    current_task,
                    token,
                )

        started = time.monotonic()
        # ``paused_signal`` is captured INSIDE the iter() context (where
        # ``run.all_messages()`` is valid for checkpoint serialization) and
        # yielded OUTSIDE all context managers below. Yielding from inside the
        # ``async with run_iter as run:`` block would leave the iter() cancel
        # scope open across the yield -- when a non-streaming consumer (run())
        # raises RunPaused without driving the generator to completion, the
        # scope would be finalized during GC in a different task and anyio
        # would raise "Attempted to exit cancel scope in a different task". By
        # capturing the signal and yielding after every ``async with`` has
        # exited, the generator holds no open context when it suspends -- safe
        # for either consumer to abandon.
        paused_signal: "RunPaused | None" = None

        try:
            async with self._span("agent.run", attrs=run_attrs):
                # -- New-run setup: RunStarted event + before_run middleware.
                # Resume path skips both -- the initial run already did them.
                if message_history is None:
                    await append_event(
                        self._event_store,
                        EventContext.from_run_context(context),
                        RunStarted(
                            run_id=context.run_id, runnable_id=context.runnable_id
                        ),
                    )
                    if self._middleware_pipeline is not None:
                        await self._middleware_pipeline.run_before_run(context)

                prior_messages = await self._session_store.list_messages(
                    context.session_id
                )
                # Prompt window policy: trim session history before it
                # is folded into the prompt. Opt-in via CapabilityRuntimeOptions;
                # None (default) leaves prior_messages untouched.
                window_policy = (
                    self._capability_options.session_window_policy
                    if self._capability_options is not None
                    else None
                )
                if window_policy is not None:
                    from ..events.payloads import PromptWindowApplied

                    before_count = len(prior_messages)
                    prior_messages = list(
                        await window_policy.select_messages(
                            prior_messages, agent.spec.model
                        )
                    )
                    await append_event(
                        self._event_store,
                        EventContext.from_run_context(context),
                        PromptWindowApplied(
                            policy=type(window_policy).__name__,
                            before=before_count,
                            after=len(prior_messages),
                        ),
                    )

                # -- Prompt build. Resume path skips this entirely -- the
                # prompt is baked into the checkpointed message_history.
                #
                # Two distinct values: ``user_prompt`` is the caller's ORIGINAL
                # input (what gets persisted as the USER session message); the
                # concatenated ``prompt`` below is the MODEL prompt (history +
                # memory + knowledge + user) -- never persisted verbatim, so
                # internal runtime context cannot leak into session history.
                user_prompt = request.prompt
                prompt: "str | None" = None
                if message_history is None:
                    history_text = _format_session_history(prior_messages)
                    prompt = (
                        f"{history_text}\n{user_prompt}"
                        if history_text
                        else user_prompt
                    )

                    # Memory + Knowledge injection via substitutable policies.
                    # Each fires only when its policy is wired (explicit on
                    # options, or the runner's default built from a wired store/
                    # retriever) AND yields a non-empty section. Memory is
                    # injected first, then knowledge -- both prepend, so the
                    # final top-to-bottom order is: Knowledge, Memory, history,
                    # user prompt.
                    memory_policy = self._effective_memory_policy()
                    if memory_policy is not None:
                        memories = await memory_policy.select_memories(
                            context, user_prompt
                        )
                        section = self._prompt_formatter().format_memory(memories)
                        if section:
                            prompt = f"{section}\n{prompt}"
                    retrieval_policy = self._effective_retrieval_policy()
                    if retrieval_policy is not None:
                        items = await retrieval_policy.retrieve(context, user_prompt)
                        section = self._prompt_formatter().format_knowledge(items)
                        if section:
                            prompt = f"{section}\n{prompt}"

                # -- Model call: agent.pydantic_agent.iter() drives the graph.
                # The per-Run ToolContext travels to capabilities via pydantic-ai
                # DI: ``deps=`` becomes ``ctx.deps.tool_context`` inside every
                # capability hook (safe concurrent reuse
                # of one CompiledAgent across many Runs).
                tool_context = ToolContext(
                    run_id=context.run_id,
                    session_id=context.session_id,
                    tool_call_id=None,
                    tenant_id=context.tenant_id,
                )
                # The builtin file/terminal toolset is
                # constructed HERE, at execution time, from the per-Runtime
                # ExecutionBackend -- not baked into the compiled Agent. This
                # is what makes AgentCompiler stateless (no filesystem surface):
                # the same CompiledAgent can be reused across Runs that target
                # different working directories, and a conversational-only run
                # (no execution backend) simply passes ``toolsets=[]`` so the
                # model has no file/terminal tools available. ``deps.execution``
                # is the same backend -- capabilities that need it read
                # ``ctx.deps.execution``.
                deps = AgentDependencies(
                    tool_context=tool_context,
                    execution=self._execution,
                )
                toolsets: "list[AbstractToolset]" = []
                capability_prompt = ""
                cap_bundle = None
                # All tools go through the capability assembler + ManagedToolAdapter.
                # tools=None + execution -> default builtin via assembler (no raw bypass).
                # tools=() -> no tools. tools explicit -> only declared.
                has_assembler = self._capability_assembler is not None
                builtin_flag = getattr(self._capability_options, "enable_builtin_tools", None)
                needs_default = (
                    agent.spec.tools is None
                    and deps.execution is not None
                    and builtin_flag is not False
                )
                # Eager fail-fast: a spec that needs tools -- default builtin
                # when an execution backend is present, or explicitly declared
                # tools -- must have BOTH an assembler and a managed executor
                # wired before any resolution work. tools=() needs neither and
                # never raises; tools=None without execution is a model-only run.
                from ..capability.models import requires_capability_assembler

                requires_tools = requires_capability_assembler(
                    tools=(agent.spec.tools if not needs_default else ("builtin",)),
                    execution=deps.execution,
                )
                if requires_tools and not has_assembler:
                    from ..errors import RuntimeInitializationError

                    raise RuntimeInitializationError(
                        "AgentRunner requires a CapabilityAssembler to resolve tools"
                    )
                if requires_tools and self._tool_executor_for_managed is None:
                    from ..errors import RuntimeInitializationError

                    raise RuntimeInitializationError(
                        "AgentRunner requires a ToolExecutor for managed tool execution"
                    )
                if requires_tools:
                    from ..capability.exposure import CapabilityToolExposurePolicy
                    from ..capability.provider import CapabilityContext

                    exposure = (
                        self._capability_options.tool_exposure
                        if self._capability_options is not None
                        else CapabilityToolExposurePolicy()
                    )
                    from ..security.emitter import EventStoreSecurityEventEmitter

                    cap_ctx = CapabilityContext(
                        agent_id=agent.spec.id,
                        exposure_policy=exposure,
                        execution=deps.execution,
                        run_id=context.run_id,
                        root_run_id=context.root_run_id,
                        parent_run_id=context.parent_run_id,
                        session_id=context.session_id,
                        event_store=self._event_store,
                        security_event_emitter=EventStoreSecurityEventEmitter(
                            self._event_store,
                            context=context,
                            failure_mode=self._security_audit_failure_mode,
                        ),
                        user_id=context.user_id,
                        tenant_id=context.tenant_id,
                        workspace=context.workspace,
                    )
                    # When tools is None, synthesize a default builtin:* ref.
                    # dataclasses.replace() so any future AgentSpec field is
                    # carried over automatically -- never hand-reconstruct the
                    # spec field by field (a new field would silently be
                    # dropped and the default path would diverge from the
                    # declared spec).
                    if needs_default:
                        from ..agent.spec import ToolRef as _TR

                        effective_spec = dataclasses.replace(
                            agent.spec,
                            tools=(_TR(kind="builtin", name="*"),),
                        )
                    else:
                        effective_spec = agent.spec
                    cap_bundle = await self._capability_assembler.assemble(
                        effective_spec, cap_ctx
                    )
                    # Publish this run's descriptor lookup so PolicyCapability
                    # (the global before-every-tool-call hook, independent of
                    # whether a tool is ManagedToolsetWrapper-wrapped) can
                    # classify calls by category/risk/mutating too -- not just
                    # by tool name.
                    # Build the per-run descriptor lookup from every
                    # contribution's tool descriptors via
                    # ``_contribution_descriptors`` -- the single helper that
                    # reads the per-tool definitions, so PolicyCapability can
                    # recognize every managed tool (reading only one shape
                    # would leave the lookup empty for the other).
                    from ..capability.assembler import _contribution_descriptors

                    descriptor_lookup = {
                        d.name: d
                        for contrib in cap_bundle.tool_contributions
                        for d in _contribution_descriptors(contrib)
                    }
                    if descriptor_lookup:
                        deps = dataclasses.replace(
                            deps, descriptor_lookup=descriptor_lookup
                        )
                    # Every ToolContribution ALWAYS goes through
                    # ManagedToolsetWrapper -> ManagedToolAdapter ->
                    # ToolExecutor.execute, whether or not a security object is
                    # configured. The managed path owns more than security --
                    # timeout, retry, idempotency, stable errors, events, call
                    # id -- so disabling the baseline must NOT route tools back
                    # to an unmanaged toolset.
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
                            event_store=self._event_store,
                            security_audit_failure_mode=self._security_audit_failure_mode,
                            security_event_emitter=cap_ctx.security_event_emitter,
                        )
                        for contrib in cap_bundle.tool_contributions:
                            # Each tool gets its own wrapped toolset built from
                            # its explicit handler. When the definition carries
                            # parameters_json_schema (e.g. a **kwargs MCP
                            # forwarding handler), use it so the model sees the
                            # right parameters; otherwise pydantic-ai derives it
                            # from the handler signature.
                            for md in contrib.tools:
                                toolsets.append(
                                    ManagedToolsetWrapper(
                                        build_managed_toolset(md),
                                        descriptors={md.descriptor.name: md.descriptor},
                                        **wrap_kw,
                                    )
                                )
                            # empty contribution (no tools) -> nothing to expose
                # ToolExposureApplied + PromptCatalog events only fire when the
                # capability assembler ran (cap_bundle exists).
                if cap_bundle is not None:
                    from ..events.payloads import (
                        PromptCatalogInjected,
                        ToolExposureApplied,
                    )

                    # Descriptor-only tool count (no toolset introspection): the
                    # same source the assembler used for conflict/cap checks.
                    total = 0
                    for c in cap_bundle.tool_contributions:
                        total += len(c.tools)
                    await append_event(
                        self._event_store,
                        EventContext.from_run_context(context),
                        ToolExposureApplied(agent_id=agent.spec.id, total_tools=total),
                    )
                    if cap_bundle.prompt_sections:
                        capability_prompt = "\n\n".join(
                            cap_bundle.prompt_sections.values()
                        )
                        for section in cap_bundle.prompt_sections:
                            await append_event(
                                self._event_store,
                                EventContext.from_run_context(context),
                                PromptCatalogInjected(
                                    agent_id=agent.spec.id, section=section
                                ),
                            )
                # ModelPolicy.timeout_seconds is enforced by wrapping
                # each graph step (``run.__anext__()``) in asyncio.wait_for with
                # the REMAINING budget. The model call happens at this await
                # point (for stream-less models / the non-streaming path), so
                # wait_for can interrupt a hanging model call. timeout_seconds
                # None means no timeout wrapper is needed.
                timeout = agent.spec.model.timeout_seconds

                accumulated_text = ""
                result = None
                # check the cancellation token BEFORE the model call.
                # raise_if_cancelled() is a no-op when the token is not set
                # (or when no controller is wired -- token is None), so this
                # is observationally invisible on the default path.
                if token is not None:
                    await token.raise_if_cancelled()
                # On resume, prompt is None (the prompt is baked into the
                # checkpointed message_history) and capability_prompt must NOT
                # be re-concatenated (it already lives in the history). Building
                # ``capability_prompt + "\n\n" + prompt`` here would be str+None.
                if message_history is not None:
                    effective_prompt = None
                elif capability_prompt:
                    effective_prompt = f"{capability_prompt}\n\n{prompt}"
                else:
                    effective_prompt = prompt
                # WP-13: wrap the model so the security pipeline fires
                # before_model/after_model around EVERY model request (a tool
                # loop drives several), not just once around the whole run.
                # Passed per-call via iter(model=...) so the shared compiled
                # agent is never mutated.
                iter_model = None
                if self._security_pipeline is not None:
                    from ..security.secured_model import SecuredModel

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

                # ``timed_out`` is set whenever the wait_for budget is exhausted
                # (either proactively before the call, or reactively when
                # wait_for fires / pydantic-ai's iter() cleanup re-raises the
                # cancellation). After the iter() context exits, the flag gates
                # a single ``raise ModelRoutingError("model timeout")``. This is
                # necessary because pydantic-ai's iter() __aexit__ can mask the
                # TimeoutError from wait_for with its own CancelledError during
                # graph-run cleanup -- the flag survives that masking so the
                # generic-except handler still records FAILED + "model timeout"
                # (not CANCELLED).
                timed_out = False
                iter_started = time.monotonic()
                try:
                    async with self._span("agent.model"):
                        async with run_iter as run:
                            try:
                                while True:
                                    # Advance to the next node, optionally under
                                    # a total-timeout budget. wait_for cancels
                                    # ``__anext__`` on timeout; we then break the
                                    # loop and let the iter() context exit before
                                    # raising ModelRoutingError (so __aexit__
                                    # cleanup happens with the flag already set,
                                    # and any cleanup-propagated CancelledError is
                                    # absorbed back into the timeout path).
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
                                        # wait_for's timeout cancellation can
                                        # surface as CancelledError rather than
                                        # TimeoutError (pydantic-ai's graph drive
                                        # re-raises the stored _node_error). If
                                        # the deadline has passed, treat as
                                        # timeout; otherwise it's a real cancel
                                        # (propagate to the outer handler).
                                        if (
                                            timeout is not None
                                            and (time.monotonic() - iter_started)
                                            >= timeout
                                        ):
                                            timed_out = True
                                            break
                                        raise

                                    # Stream events from this node. For a model
                                    # request node, surface incremental text
                                    # deltas; for a call-tools node, surface
                                    # tool-call/tool-result events. Both node
                                    # types are guarded by try/except so a model
                                    # without a stream_function (non-streaming
                                    # run() path) degrades gracefully -- the
                                    # node has already run via __anext__, we
                                    # just skip emitting per-delta events and
                                    # the final result.output is used.
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
                                                        yield {
                                                            "type": "text",
                                                            "text": text,
                                                        }
                                        except AssertionError:
                                            # pydantic-ai raises AssertionError
                                            # ("FunctionModel must receive a
                                            # stream_function ...") when the
                                            # model cannot stream -- the model
                                            # call already happened via
                                            # __anext__, so skip per-delta
                                            # events and use result.output. Any
                                            # other exception is a genuine
                                            # stream error and propagates.
                                            pass
                                    elif PydanticAgent.is_call_tools_node(node):
                                        try:
                                            async with node.stream(
                                                run.ctx
                                            ) as tool_stream:
                                                async for ev in tool_stream:
                                                    if isinstance(
                                                        ev, FunctionToolCallEvent
                                                    ):
                                                        yield {
                                                            "type": "tool",
                                                            "name": ev.part.tool_name,
                                                            "phase": "start",
                                                            "ok": None,
                                                        }
                                                    elif isinstance(
                                                        ev, FunctionToolResultEvent
                                                    ):
                                                        yield {
                                                            "type": "tool",
                                                            "name": ev.part.tool_name,
                                                            "phase": "end",
                                                            "ok": isinstance(
                                                                ev.part, ToolReturnPart
                                                            ),
                                                        }
                                        except AssertionError:
                                            # Same non-streaming-model signal as
                                            # above -- tools already ran via
                                            # __anext__; genuine errors propagate.
                                            pass
                            except RunPaused as paused:
                                # Pause path (canonical surface): persist the
                                # ApprovalRequest, save a real checkpoint of the
                                # partial message history, transition to
                                # WAITING_APPROVAL, emit ApprovalRequested +
                                # RunPaused events -- all INSIDE the iter()
                                # context so ``run.all_messages()`` works. The
                                # paused-event yield itself is postponed to
                                # AFTER all context managers exit (see the
                                # comment on ``paused_signal`` above) so a
                                # non-streaming consumer can abandon the
                                # generator without triggering a cross-task
                                # cancel-scope exit.
                                #
                                # ToolExecutor
                                # no longer persists the ApprovalRequest itself
                                # -- ``paused`` carries every field needed to
                                # build it, and THIS handler is the one that
                                # calls ``ApprovalStore.create_or_get_pending``
                                # (deduping on (run_id, tool_call_id)),
                                # so the approval write joins the same atomicity
                                # story as checkpoint/transition/event below.
                                #
                                # atomicity: when a UnitOfWork factory is
                                # wired (SqlAlchemy), approval + checkpoint +
                                # transition + events share ONE transaction --
                                # they commit together on clean exit or rollback
                                # together if ANY of them raises (which then
                                # propagates to the outer generic-except ->
                                # FAILED). When no factory is wired (File),
                                # cross-store transactions are impossible, so
                                # the path keeps its non-atomic best-effort
                                # shape: checkpoint + transition still
                                # propagate; leaving them partial is forbidden,
                                # but the approval write and RunPaused/
                                # ApprovalRequested event appends are
                                # best-effort.
                                # create_or_get_pending may return an
                                # EXISTING approval (dedup on tool_call_id)
                                # whose id differs from the fresh
                                # ``paused.approval_id`` ToolExecutor minted.
                                # Resolve the id FIRST, then build the
                                # checkpoint/event payloads from the resolved
                                # id -- otherwise a dedup hit would leave the
                                # checkpoint/events pointing at an id that was
                                # never actually persisted. Mutating
                                # ``paused.approval_id`` in place means the
                                # final ``paused`` yield (below, after every
                                # context manager exits) automatically reports
                                # the resolved id too.
                                # Pause commit is coordinator-owned: the runner
                                # builds a PauseRunCommand and delegates. The
                                # coordinator persists the approval (resolving
                                # the id -- create_or_get_pending may dedup on
                                # tool_call_id and return an existing id), the
                                # checkpoint, the WAITING_APPROVAL transition,
                                # and the ApprovalRequested + RunPaused events
                                # as one atomic commit. The resolved approval id
                                # is written back so the paused-event yield
                                # below reports the persisted id.
                                _commit = await self._commit_coordinator.pause(
                                    PauseRunCommand(
                                        run_id=context.run_id,
                                        expected_version=running_version,
                                        approval_request={
                                            "approval_id": paused.approval_id,
                                            "tool_call_id": paused.tool_call_id,
                                            "tool_name": paused.tool_name or "",
                                            "reason": paused.reason,
                                            "arguments": paused.arguments,
                                        },
                                        checkpoint_payload=serialize_messages(
                                            run.all_messages()
                                        ),
                                        event_context=EventContext.from_run_context(
                                            context
                                        ),
                                        commit_id=f"pause:{context.run_id}:{paused.approval_id}",
                                    )
                                )
                                paused.approval_id = _commit.approval_id
                                paused_signal = paused
                            else:
                                if not timed_out:
                                    result = run.result
                except asyncio.TimeoutError:
                    # TimeoutError escaping the iter() context (either from
                    # wait_for directly or from iter() __aexit__ cleanup).
                    # Only treat as model timeout when the deadline actually
                    # passed; an unrelated TimeoutError propagates.
                    if (
                        timeout is not None
                        and (time.monotonic() - iter_started) >= timeout
                    ):
                        timed_out = True
                    else:
                        raise
                except asyncio.CancelledError:
                    # CancelledError escaping the iter() context (typically
                    # iter() __aexit__ re-raising the wait_for cancellation
                    # during graph-run cleanup). If the deadline passed, treat
                    # as timeout; otherwise it's a real cancel -- propagate to
                    # execute()'s outer CancelledError handler so the run ends
                    # up CANCELLED, not FAILED.
                    if (
                        timeout is not None
                        and (time.monotonic() - iter_started) >= timeout
                    ):
                        timed_out = True
                    else:
                        raise
                except Exception:
                    # iter() __aexit__ cleanup can raise anyio errors
                    # (ClosedResourceError etc.) after a wait_for cancellation.
                    # If we already flagged a timeout, absorb the cleanup error
                    # into the ModelRoutingError path -- otherwise re-raise so
                    # genuine model errors propagate to the outer handler.
                    if timed_out:
                        pass
                    else:
                        raise

                # check the cancellation token AFTER the model call
                # completes. If Runtime.cancel flipped the token while the
                # iter() drive was in flight (but the underlying asyncio.Task
                # cancel hasn't surfaced yet), this raises CancelledError here
                # -- caught by the outer handler below which transitions
                # CANCELLING -> CANCELLED. No-op when no controller is wired.
                if token is not None:
                    await token.raise_if_cancelled()

                # If the timeout budget was exhausted, raise ModelRoutingError
                # so the outer generic-except handler records FAILED with a
                # descriptive "model timeout" message (). This sits
                # OUTSIDE the iter() context so __aexit__ has already run.
                if timed_out:
                    raise ModelRoutingError("model timeout")

                # -- Post-iter success path. Skipped on the pause path
                # (paused_signal set); the paused-event yield happens after
                # every ``async with`` has exited (below). result is the
                # AgentRunResult from iter(); prefer result.output (the parsed
                # final output -- str, dict, BaseModel, whatever output_schema
                # produces). Only fall back to accumulated_text when result is
                # somehow None (covers edge cases where iter() ended without a
                # final result and the streamed text deltas ARE the answer).
                if paused_signal is None and not timed_out:
                    if result is not None:
                        output = result.output
                    else:
                        output = accumulated_text

                    # max_tokens + cost-budget enforcement. usage is read once so
                    # the checks and RunResult.token_usage share one snapshot.
                    usage = result.usage if result is not None else None
                    max_tokens = agent.spec.model.max_tokens
                    if max_tokens is not None and usage is not None:
                        used = usage.input_tokens + usage.output_tokens
                        if used > max_tokens:
                            raise ModelPolicyExceededError(
                                f"max_tokens exceeded: used {used} > max_tokens {max_tokens}",
                                kind="max_tokens",
                            )
                    # ModelPolicy.budget is a Decimal cost limit. A budget set
                    # without a pricing provider is a configuration error
                    # (fail-closed); with pricing, the cost of this response
                    # must not exceed the budget.
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

                    # Build the complete-turn payload shared by both commit
                    # paths: USER prompt + ASSISTANT output, plus the checkpoint
                    # of the full message history. ``run`` is still bound here
                    # (Python preserves the ``with ... as run`` target after the
                    # block); the iter() context has exited cleanly but the
                    # AgentRun still serves message history.
                    user_content = request.prompt
                    messages_to_append = [
                        NewSessionMessage(
                            role=MessageRole.ASSISTANT,
                            content=str(output),
                            run_id=context.run_id,
                        ),
                    ]
                    if user_content:
                        messages_to_append.insert(
                            0,
                            NewSessionMessage(
                                role=MessageRole.USER,
                                content=user_content,
                                run_id=context.run_id,
                            ),
                        )
                    checkpoint_payload = serialize_messages(run.all_messages())

                    run_result = RunResult(
                        output=output,
                        token_usage={
                            "input_tokens": usage.input_tokens if usage else 0,
                            "output_tokens": usage.output_tokens if usage else 0,
                        },
                    )

                    # after_run is business lifecycle: run it BEFORE the commit
                    # so an after_run failure takes the normal FAILED path
                    # instead of corrupting an already-SUCCEEDED run.
                    if self._middleware_pipeline is not None:
                        await self._middleware_pipeline.run_after_run(
                            context, run_result
                        )

                    # Cross-store commit is coordinator-owned: the runner builds
                    # a CompleteRunCommand and delegates. The coordinator
                    # persists the session turn (USER + ASSISTANT), the
                    # checkpoint, the SUCCEEDED transition (with the result),
                    # and the RunCompleted event as one atomic commit. The
                    # runner never writes those stores directly.
                    await self._commit_coordinator.complete(
                        CompleteRunCommand(
                            run_id=context.run_id,
                            session_id=context.session_id,
                            expected_version=running_version,
                            messages=tuple(messages_to_append),
                            checkpoint_payload=checkpoint_payload,
                            result=run_result,
                            event_context=EventContext.from_run_context(context),
                            commit_id=f"complete:{context.run_id}:{running_version}",
                        )
                    )

                    # Success metrics are best-effort observation only -- a
                    # metrics failure must never flip a committed run to FAILED.
                    self._record_success_metrics(
                        attributes=run_attrs, started=started, metrics=metrics
                    )
        except asyncio.CancelledError:
            # in-flight cancel path. CancelledError
            # surfaces at the current await point (model call / node.stream() /
            # event append / middleware / token check). Caught BEFORE the generic
            # ``except Exception`` (CancelledError is a BaseException since
            # Python 3.8). The run is transitioned to CANCELLING then CANCELLED
            # -- CANCELLING distinguishes "cancel requested" from "actually
            # stopped", and the run only reaches CANCELLED once this
            # handler has actually drained.
            #
            # Defensive against the controller-driven path: when Runtime.cancel
            # beat us to it, the store is ALREADY in CANCELLING and the
            # RUNNING -> CANCELLING transition is rejected. We re-read the
            # record to capture its live version for the CANCELLING -> CANCELLED
            # step (expected_version always comes from the store, never
            # hardcoded).
            #
            # Both transitions are best-effort ONLY because asyncio requires
            # CancelledError to propagate -- letting a transition error escape
            # would replace the cancellation with a different exception type and
            # break the cancel machinery. The warning keeps failures visible
            # rather than silent (the run may already be terminal, e.g. a
            # concurrent cancel beat this handler).
            try:
                try:
                    cancelling = await self._run_store.transition(
                        context.run_id,
                        RunStatus.CANCELLING,
                        expected_version=running_version,
                    )
                except (InvalidRunTransitionError, RunConflictError):
                    # Runtime.cancel already moved the store to CANCELLING
                    # (controller-driven path): either the version no longer
                    # matches (RunConflictError) or the source state is no
                    # longer RUNNING (InvalidRunTransitionError). Re-read to
                    # capture the live version for the CANCELLED transition.
                    # If the run is not in CANCELLING (e.g. concurrent
                    # terminal transition), bail out -- nothing more we can do
                    # here.
                    current = await self._run_store.get(context.run_id)
                    if current is None or current.status is not RunStatus.CANCELLING:
                        raise
                    cancelling = current
                await mark_cancelled(
                    self._run_store,
                    context.run_id,
                    expected_version=cancelling.version,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition run %s to CANCELLED on cancel: %s",
                    context.run_id,
                    exc,
                )
            raise
        except Exception as exc:
            # Generic error path: transition FAILED, run on_error, append
            # RunFailed, record metrics. Re-raise so the caller sees the error.
            #
            # Terminal/commit-point guard: if the run already left RUNNING -- it
            # reached a commit point (SUCCEEDED / WAITING_APPROVAL / PAUSED) or
            # is already terminal (a post-commit error such as a metrics or
            # critical-event failure) -- do NOT fabricate a contradictory FAILED
            # transition or RunFailed event. Re-raise the original error; the
            # committed state stays authoritative and recovery re-attempts any
            # missed post-commit write.
            current = await self._run_store.get(context.run_id)
            if current is not None and current.status is not RunStatus.RUNNING:
                raise
            safe_error = redact_exception(exc)
            error_info = RunErrorInfo(error_type=type(exc).__name__, message=safe_error)
            # Run status update. The FAILED transition is kept
            # best-effort ONLY because we are already in the failing path:
            # letting the transition error escape would replace the ORIGINAL
            # exc (the actual cause) with a version-mismatch/store error,
            # losing the cause for the caller. The warning keeps the
            # transition failure visible rather than silent; a typical failure
            # is a version mismatch from a concurrent terminal transition.
            try:
                await mark_failed(
                    self._run_store,
                    context.run_id,
                    expected_version=running_version,
                    error=error_info,
                )
            except Exception as transition_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition run %s to FAILED: %s",
                    context.run_id,
                    transition_exc,
                )
            if self._middleware_pipeline is not None:
                # on_error is best-effort observation: its failure must not
                # replace the ORIGINAL exc the caller needs to see.
                try:
                    await self._middleware_pipeline.run_on_error(context, exc)
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "on_error middleware failed for run %s",
                        context.run_id,
                    )
            # best-effort audit - non-critical path: the run is already failing
            # (FAILED transition attempted above), so a missing RunFailed event
            # is an observability gap, not state corruption.
            try:
                await append_event(
                    self._event_store,
                    EventContext.from_run_context(context),
                    RunFailed(
                        run_id=context.run_id,
                        error_type=type(exc).__name__,
                        message=safe_error,
                    ),
                )
            except Exception as event_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to append RunFailed event for run %s: %s",
                    context.run_id,
                    event_exc,
                )
            if metrics is not None:
                try:
                    metrics.counter(
                        "agent.run.failed",
                        attributes={
                            "run_id": context.run_id,
                            "session_id": context.session_id,
                            "error_type": type(exc).__name__,
                        },
                    )
                except Exception:  # noqa: BLE001
                    _LOGGER.exception(
                        "failure metrics failed for run %s", context.run_id
                    )
            raise
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass
            # drop the in-flight registration so the controller does
            # not retain a reference to the (now-finished) asyncio.Task. The
            # unregister is in ``finally`` so it runs on every exit path --
            # success, pause-yield, cancel, and error. Idempotent (no-op when
            # nothing was registered, e.g. the default-None controller path).
            if self._run_controller is not None:
                await self._run_controller.unregister(context.run_id)

        # -- Paused-event yield: OUTSIDE every ``async with`` and the outer
        # try/except, so the generator holds no open context manager when it
        # suspends here. A non-streaming consumer (run()) raises RunPaused and
        # abandons the generator; GC finalizes the frame without triggering
        # any __aexit__ (so no cross-task cancel-scope exit). A streaming
        # consumer (run_stream()) receives the event and lets the generator
        # exhaust naturally on the next drive.
        if paused_signal is not None:
            yield {
                "type": "paused",
                "run_id": paused_signal.run_id,
                "approval_id": paused_signal.approval_id,
            }

    async def run(
        self,
        agent: CompiledAgent,
        request: RunInput,
        context: RunContext,
    ) -> RunResult:
        """Non-streaming entry point. Consumes execute() in full, re-raises
        RunPaused when the lifecycle yields a pause signal, and reads the
        final RunResult back from the RunStore (execute() already populated it
        via the SUCCEEDED transition). All lifecycle concerns -- events,
        middleware, transitions, timeout, budget, observability -- live in
        execute(); run() is purely a collector.

        ``contextlib.aclosing`` guarantees the generator is finalized in THIS
        task when run() exits early on a pause -- execute() yields the paused
        event with no open context manager, but aclosing is belt-and-suspenders
        so any future change to execute()'s suspension points stays safe."""
        paused_event: "dict | None" = None
        async with contextlib.aclosing(self.execute(agent, request, context)) as gen:
            async for event in gen:
                if event["type"] == "paused":
                    paused_event = event
                    break

        if paused_event is not None:
            raise RunPaused(
                run_id=paused_event["run_id"],
                approval_id=paused_event["approval_id"],
            )

        # execute() completed without pausing -- the SUCCEEDED transition
        # stored the RunResult. Read it back rather than threading state out
        # of the generator, so the result is the store's authoritative copy
        # (single source of truth).
        record = await self._run_store.get(context.run_id)
        if record is not None and record.result is not None:
            return record.result
        # execute() must always leave a terminal record with a result. If it
        # did not, that is a runtime invariant violation -- raise rather than
        # fabricate an empty success that masks the bug.
        raise RunInvariantError(
            f"run {context.run_id} completed without a persisted result"
        )

    async def run_stream(
        self,
        agent: CompiledAgent,
        request: RunInput,
        context: RunContext,
        message_history: "Sequence[ModelMessage] | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming entry point. A thin pass-through to execute(): every
        event the lifecycle yields (text deltas, tool events, paused) is
        forwarded to the consumer unchanged. Resume (Runtime.resume) supplies
        ``message_history=<deserialized checkpoint>`` so the pydantic-ai graph
        picks up from the paused state."""
        async for event in self.execute(
            agent,
            request,
            context,
            message_history=message_history,
        ):
            yield event
