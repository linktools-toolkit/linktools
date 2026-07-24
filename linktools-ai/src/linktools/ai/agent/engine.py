#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentEngine: owns the per-invocation lifecycle -- Run state transitions,
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

Prompt template composition itself is owned by
:class:`~linktools.ai.prompt.builder.PromptBuilder` (the prompt domain, per
): ``execute()`` fetches the memory + knowledge sections via their
async policies and hands user_prompt / prior_messages / those sections to
``PromptBuilder.build_base_prompt()``, then folds the capability-resolved
prompt sections in via ``PromptBuilder.combine()``. Final model prompt order
(when all are set and non-empty): capability catalog, ``## Knowledge``,
``## Memory``, session history, user prompt. Both memory + knowledge default
to None, so existing callers see no change.

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
from ..events.context import EventStreamContext, append_event
from ..events.store import EventStore
from ..middleware.pipeline import MiddlewarePipeline
from ..observability.tracing import use_span
from ..prompt.builder import PromptBuilder
from ..governance.policy.engine import ToolContext
from ..governance.security.redact import redact_exception
from ..run.cancellation import CancellationToken
from ..run.commit import CompleteRunCommand, PauseRunCommand
from ..run.context import RunContext
from ..run.controller import RunController
from ..run.events_bus import RunEventBus
from ..run.lifecycle import create_and_start_run, mark_cancelled, mark_failed
from ..run.models import (
    RunErrorInfo,
    RunInput,
    RunResult,
    RunStatus,
)
from ..run.store import RunStore
from ..session.recorder import SessionRecorder
from ..session.store import SessionStore
from .checkpoint import serialize_messages
from .dependencies import AgentDependencies
from .models import (
    AgentExecutionOutcome,
    AgentExecutionStatus,
    AgentInput,
    CompiledAgent,
    PauseRequest,
    RunUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets import AbstractToolset

    from ..capability.resolver import CapabilityResolver
    from ..capability.models import CapabilityRuntimeOptions
    from ..sandbox.protocols import Sandbox
    from ..retrieval.retriever import Retriever
    from ..memory.store import MemoryStore
    from ..observability.metrics import ObservabilityMetrics
    from ..observability.tracing import ObservabilitySink
    from ..run.dispatch import RunDispatchRequest


_LOGGER = logging.getLogger(__name__)


class _PendingCommit:
    """Mutable out-parameter: when passed to ``execute()`` as ``commit_sink``,
    the success/pause paths stash their Complete/PauseRunCommand here INSTEAD
    OF calling ``commit_coordinator.complete()``/``.pause()`` themselves,
    letting the caller (``_drive_paused_or_result``) perform that one Store-
    writing call after the generator has already fully drained -- a safe
    reordering since nothing else runs concurrently at that point. ``None``
    (the default, every direct-engine caller/test) preserves execute()'s
    original behavior exactly: it commits internally, as it always has."""

    __slots__ = (
        "complete_command",
        "pause_command",
        "record_metrics",
        "on_cancelled",
        "on_failed",
        "running_version",
    )

    def __init__(self) -> None:
        self.complete_command: "CompleteRunCommand | None" = None
        self.pause_command: "PauseRunCommand | None" = None
        # Closure capturing execute()'s local run_attrs/started/metrics --
        # avoids duplicating those fields onto this sink. Invoked by the
        # caller immediately after it performs the complete commit.
        self.record_metrics: "Any" = None
        # Closures capturing execute()'s local context/running_version/exc
        # (plus self._run_store/_middleware_pipeline/_event_store/_metrics)
        # for the cancel/fail terminal transitions. Invoked by the caller
        # from its OWN except block, catching the SAME exception these
        # closures were built to handle, immediately before re-raising it.
        self.on_cancelled: "Any" = None
        self.on_failed: "Any" = None
        # The RUNNING version execute() read/created with -- the collector
        # needs it to drive the fail-transition if the post-drain commit
        # itself raises (a case execute()'s own fail-closure cannot cover,
        # since the generator has already exited cleanly by then).
        self.running_version: "int | None" = None


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
        run_store: RunStore,
        session_store: SessionStore,
        event_store: EventStore,
        middleware_pipeline: "MiddlewarePipeline | None" = None,
        memory_store: "MemoryStore | None" = None,
        retriever: "Retriever | None" = None,
        observability: "ObservabilitySink | None" = None,
        metrics: "ObservabilityMetrics | None" = None,
        run_controller: "RunController | None" = None,
        sandbox: "Sandbox | None" = None,
        capability_resolver: "CapabilityResolver | None" = None,
        capability_options: "CapabilityRuntimeOptions | None" = None,
        security_pipeline: Any = None,
        baseline_policy: Any = None,
        tool_policy_provider: Any = None,
        managed_tool_executor: Any = None,
        security_audit_failure_mode: Any = "fail_closed",
        commit_coordinator: Any,
        pricing_provider: Any = None,
        event_bus: "RunEventBus | None" = None,
    ) -> None:
        self._run_store = run_store
        self._session_store = session_store
        self._event_store = event_store
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
        # The per-Runtime Sandbox. execute()
        # publishes this to AgentDependencies.sandbox and constructs the
        # builtin file/terminal FunctionToolset from it at execution time,
        # passing it via ``agent.iter(prompt, toolsets=[...])``. ``None``
        # (default) means the compiled agent exposes no builtin tools -- a
        # conversational-only run. Holding the backend on the runner (not the
        # compiler) is what decouples AgentCompiler from the filesystem.
        self._sandbox = sandbox
        # Capability Runtime: when an AgentSpec declares non-
        # empty tools and an resolver is wired, execute() resolves those tools
        # into prompt sections + toolsets via the capability providers. Default
        # None means empty tools resolve to the default builtin toolset when a
        # sandbox is present.
        self._capability_resolver = capability_resolver
        self._capability_options = capability_options
        self._security_pipeline = security_pipeline
        self._baseline_policy = baseline_policy
        self._tool_policy_provider = tool_policy_provider
        # The GovernedToolInvoker every managed tool delegates to. Constructor-injected
        # (build_runtime passes the compiler's executor) so the runner is fully
        # wired at construction -- no post-build private-field mutation.
        self._tool_executor_for_managed = managed_tool_executor
        self._security_audit_failure_mode = security_audit_failure_mode
        # RunCommitCoordinator (required). The runner NEVER writes cross-store
        # pause/complete state directly: it builds a Pause/CompleteRunCommand
        # and delegates to the coordinator, which owns the atomic commit (one
        # transaction for SQL, a journaled sequence for File). build_runtime
        # always wires the storage-appropriate coordinator.
        self._commit_coordinator = commit_coordinator
        # Optional ModelPricingProvider: when ModelPolicy.budget is set, the
        # runner computes the per-response cost and refuses to exceed it (and
        # refuses to run at all if budget is set without pricing -- fail-closed).
        self._pricing_provider = pricing_provider
        # Optional live fan-out of the SAME text/tool/paused event dicts
        # execute() already yields, published in ADDITION to (not instead of)
        # those yields. A future run_stream() can be reimplemented as a pure
        # RunEventBus subscriber once every caller relies on the bus instead
        # of iterating execute()'s own generator -- not done yet, so both
        # paths currently carry identical events. None (default) is a no-op.
        self._event_bus = event_bus
        # Message-format conversion for a completed turn. Not injectable
        # today (stateless, no config yet) -- constructed here so a future
        # increment can inject a configured instance without touching every
        # call site.
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
        commit_sink: "_PendingCommit | None" = None,
        emit_run_started: bool = True,
        running_version: "int | None" = None,
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

        # -- Setup: read the existing record's version, or create + transition
        # RUNNING when none exists yet. RunCoordinator (the production entry
        # point) now creates + starts new-run records itself via
        # ``run.lifecycle.create_and_start_run`` before ever calling here --
        # this get-or-create fallback exists so a direct AgentEngine caller
        # (bypassing RunCoordinator entirely, e.g. most engine-level tests)
        # still works unchanged. Resume follows the same get path: Runtime.resume
        # already transitioned WAITING_APPROVAL -> RUNNING before calling here.
        # The terminal transitions reuse the version read/returned here -- no
        # hardcoded expected_version anywhere in this file. When the caller
        # (RunCoordinator, which already create_and_start_run'd the record)
        # supplies running_version, skip the get-or-create entirely.
        if running_version is None:
            existing = await self._run_store.get(context.run_id)
            if existing is None:
                running = await create_and_start_run(
                    self._run_store, context=context, request=request
                )
                running_version = running.version
            else:
                running_version = existing.version
        if commit_sink is not None:
            # The collector needs the RUNNING version to drive the fail-
            # transition if the post-drain commit itself raises.
            commit_sink.running_version = running_version

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
                # ``emit_run_started=False`` (WP9 step 3): RunCoordinator
                # already emitted the event itself before calling here, right
                # after create_and_start_run -- the same relocation pattern
                # as RunRecord creation. before_run middleware is NOT a Store
                # dependency, so it always stays here regardless.
                if message_history is None:
                    if emit_run_started:
                        await append_event(
                            self._event_store,
                            EventStreamContext.from_run_context(context),
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
                        EventStreamContext.from_run_context(context),
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
                # Template composition itself lives in PromptBuilder
                # (linktools.ai.prompt); this block only fetches the memory and
                # knowledge sections via their (async) policies, then hands the
                # parts to the builder.
                user_prompt = request.prompt
                prompt: "str | None" = None
                if message_history is None:
                    # Memory + Knowledge injection via substitutable policies.
                    # Each fires only when its policy is wired (explicit on
                    # options, or the runner's default built from a wired store/
                    # retriever) AND yields a non-empty section.
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
                        prior_messages=prior_messages,
                        memory_section=memory_section,
                        knowledge_section=knowledge_section,
                    )

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
                # Sandbox -- not baked into the compiled Agent. This
                # is what makes AgentCompiler stateless (no filesystem surface):
                # the same CompiledAgent can be reused across Runs that target
                # different working directories, and a conversational-only run
                # (no sandbox wired) simply passes ``toolsets=[]`` so the
                # model has no file/terminal tools available. ``deps.sandbox``
                # is the same backend -- capabilities that need it read
                # ``ctx.deps.sandbox``.
                deps = AgentDependencies(
                    tool_context=tool_context,
                    sandbox=self._sandbox,
                )
                toolsets: "list[AbstractToolset]" = []
                cap_bundle = None
                # All tools go through the capability resolver + ManagedToolAdapter.
                # tools=None + sandbox -> default builtin via resolver (no raw bypass).
                # tools=() -> no tools. tools explicit -> only declared.
                has_resolver = self._capability_resolver is not None
                builtin_flag = getattr(self._capability_options, "enable_builtin_tools", None)
                needs_default = (
                    agent.spec.tools is None
                    and deps.sandbox is not None
                    and builtin_flag is not False
                )
                # Eager fail-fast: a spec that needs tools -- default builtin
                # when a sandbox is present, or explicitly declared
                # tools -- must have BOTH an resolver and a managed executor
                # wired before any resolution work. tools=() needs neither and
                # never raises; tools=None without a sandbox is a model-only run.
                from ..capability.models import requires_capability_resolver

                requires_tools = requires_capability_resolver(
                    tools=(agent.spec.tools if not needs_default else ("builtin",)),
                    sandbox=deps.sandbox,
                )
                if requires_tools and not has_resolver:
                    from ..errors import RuntimeInitializationError

                    raise RuntimeInitializationError(
                        "AgentEngine requires a CapabilityResolver to resolve tools"
                    )
                if requires_tools and self._tool_executor_for_managed is None:
                    from ..errors import RuntimeInitializationError

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
                    from ..governance.security.emitter import EventStoreSecurityEventEmitter

                    cap_ctx = CapabilityContext(
                        agent_id=agent.spec.id,
                        exposure_policy=exposure,
                        sandbox=deps.sandbox,
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
                    cap_bundle = await self._capability_resolver.resolve(
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
                    # Every ToolContribution ALWAYS goes through
                    # ManagedToolsetWrapper -> ManagedToolAdapter ->
                    # GovernedToolInvoker.execute, whether or not a security object is
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
                # capability resolver ran (cap_bundle exists).
                if cap_bundle is not None:
                    from ..events.payloads import (
                        PromptCatalogInjected,
                        ToolExposureApplied,
                    )

                    # Descriptor-only tool count (no toolset introspection): the
                    # same source the resolver used for conflict/cap checks.
                    total = 0
                    for c in cap_bundle.tool_contributions:
                        total += len(c.tools)
                    await append_event(
                        self._event_store,
                        EventStreamContext.from_run_context(context),
                        ToolExposureApplied(agent_id=agent.spec.id, total_tools=total),
                    )
                    if cap_bundle.prompt_sections:
                        for section in cap_bundle.prompt_sections:
                            await append_event(
                                self._event_store,
                                EventStreamContext.from_run_context(context),
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
                # PromptBuilder.combine folds the capability-resolved prompt
                # sections (if any) in front of the base prompt. On resume it
                # returns None -- the prompt is baked into the checkpointed
                # message_history and must not be re-fed alongside it.
                effective_prompt = PromptBuilder.combine(
                    base_prompt=prompt,
                    capability_sections=(
                        cap_bundle.prompt_sections if cap_bundle is not None else {}
                    ),
                    static_sections=agent.spec.instructions.sections,
                    resuming=message_history is not None,
                )
                # Wrap the model so the security pipeline fires
                # before_model/after_model around EVERY model request (a tool
                # loop drives several), not just once around the whole run.
                # Passed per-call via iter(model=...) so the shared compiled
                # agent is never mutated.
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
                                                        text_event = {
                                                            "type": "text",
                                                            "text": text,
                                                        }
                                                        if self._event_bus is not None:
                                                            await self._event_bus.publish(
                                                                context.run_id, text_event
                                                            )
                                                        yield text_event
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
                                                        if self._event_bus is not None:
                                                            await self._event_bus.publish(
                                                                context.run_id, tool_event
                                                            )
                                                        yield tool_event
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
                                # GovernedToolInvoker
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
                                # ``paused.approval_id`` GovernedToolInvoker minted.
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
                                pause_command = PauseRunCommand(
                                    run_id=context.run_id,
                                    expected_version=running_version,
                                    approval_request={
                                        "tenant_id": context.tenant_id,
                                        "approval_id": paused.approval_id,
                                        "tool_call_id": paused.tool_call_id,
                                        "tool_name": paused.tool_name or "",
                                        "reason": paused.reason,
                                        "arguments": paused.arguments,
                                        **paused.binding,
                                    },
                                    checkpoint_payload=serialize_messages(
                                        run.all_messages()
                                    ),
                                    event_context=EventStreamContext.from_run_context(
                                        context
                                    ),
                                    commit_id=f"pause:{context.run_id}:{paused.approval_id}",
                                )
                                if commit_sink is not None:
                                    # The caller (_drive_paused_or_result)
                                    # performs the actual pause commit after
                                    # this generator has fully drained, then
                                    # patches the resolved approval_id onto
                                    # the returned paused event -- safe for
                                    # the same reason as the success path.
                                    # ``paused.approval_id`` stays the
                                    # UNRESOLVED id here; the caller fixes it
                                    # up once it has the commit's response.
                                    commit_sink.pause_command = pause_command
                                else:
                                    _commit = await self._commit_coordinator.pause(
                                        pause_command
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
                    messages_to_append = self._session_recorder.build_turn_messages(
                        user_prompt=request.prompt,
                        output=output,
                        run_id=context.run_id,
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
                    complete_command = CompleteRunCommand(
                        run_id=context.run_id,
                        session_id=context.session_id,
                        expected_version=running_version,
                        messages=tuple(messages_to_append),
                        checkpoint_payload=checkpoint_payload,
                        result=run_result,
                        event_context=EventStreamContext.from_run_context(context),
                        commit_id=f"complete:{context.run_id}:{running_version}",
                    )
                    if commit_sink is not None:
                        # The caller (_drive_paused_or_result) performs the
                        # actual commit once this generator has fully drained
                        # -- safe, since nothing else runs concurrently at
                        # that point. Success metrics move with it (best-
                        # effort observation only; a metrics failure must
                        # never flip a committed run to FAILED, regardless of
                        # which side of the boundary records it).
                        commit_sink.complete_command = complete_command
                        commit_sink.record_metrics = lambda: self._record_success_metrics(
                            attributes=run_attrs, started=started, metrics=metrics
                        )
                    else:
                        await self._commit_coordinator.complete(complete_command)
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
            async def _do_cancel_transition() -> None:
                await self._apply_cancel_transition(context, running_version)

            # WP9 step 3: when commit_sink is wired, the caller
            # (_drive_paused_or_result) performs this Store transition itself
            # after catching the SAME CancelledError this raise re-propagates
            # -- the closure carries every local this needs (context,
            # running_version), so nothing has to be threaded out by hand.
            # commit_sink=None (every direct-engine caller/test) preserves
            # execute()'s original behavior exactly: it transitions inline.
            if commit_sink is not None:
                commit_sink.on_cancelled = _do_cancel_transition
            else:
                await _do_cancel_transition()
            raise
        except Exception as exc:
            # Generic error path: transition FAILED, run on_error, append
            # RunFailed, record metrics. Re-raise so the caller sees the error.
            #
            # Python implicitly unbinds ``exc`` at the end of this except
            # block (to avoid a traceback reference cycle) -- captured into a
            # plain local BEFORE the closure below so a LATER invocation
            # (commit_sink.on_failed, called from a different frame after
            # this block has already exited) does not hit a NameError on a
            # free variable that no longer exists by the time it runs.
            failure = exc

            async def _do_fail_transition() -> None:
                await self._apply_fail_transition(
                    context, running_version, failure, metrics=metrics
                )

            # WP9 step 3: same relocation pattern as the cancel path above --
            # commit_sink=None (every direct-engine caller/test) preserves
            # execute()'s original behavior exactly.
            if commit_sink is not None:
                commit_sink.on_failed = _do_fail_transition
            else:
                await _do_fail_transition()
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
            paused_event = {
                "type": "paused",
                "run_id": paused_signal.run_id,
                "approval_id": paused_signal.approval_id,
            }
            if self._event_bus is not None:
                await self._event_bus.publish(context.run_id, paused_event)
            yield paused_event

    async def _apply_cancel_transition(
        self, context: RunContext, running_version: int
    ) -> None:
        """Best-effort CANCELLING -> CANCELLED transition on the cancel path.
        Idempotent against the controller-driven path (Runtime.cancel may have
        already moved the store to CANCELLING): re-reads the live version on a
        version/source mismatch and bails out if the run is no longer
        CANCELLING. Swallows transition errors (asyncio requires
        CancelledError to propagate; a transition error must not replace it).
        Extracted from execute()'s cancel handler so a collector that catches
        the SAME CancelledError can drive the identical transition."""
        try:
            try:
                cancelling = await self._run_store.transition(
                    context.run_id,
                    RunStatus.CANCELLING,
                    expected_version=running_version,
                )
            except (InvalidRunTransitionError, RunConflictError):
                current = await self._run_store.get(context.run_id)
                if (
                    current is None
                    or current.status is not RunStatus.CANCELLING
                ):
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

    async def _apply_fail_transition(
        self,
        context: RunContext,
        running_version: int,
        failure: BaseException,
        *,
        metrics: "ObservabilityMetrics | None" = None,
    ) -> None:
        """Best-effort FAILED transition + on_error + RunFailed + metrics on
        the generic-error path. Terminal/commit-point guard: if the run already
        left RUNNING (reached SUCCEEDED/WAITING_APPROVAL/PAUSED or is already
        terminal), the committed state stays authoritative and this is a no-op.
        Every side effect is best-effort (the ORIGINAL failure must propagate,
        not be replaced by a transition/middleware/event/metrics error).
        Extracted from execute()'s generic-error handler so a collector that
        catches the SAME exception can drive the identical transition."""
        current = await self._run_store.get(context.run_id)
        if current is not None and current.status is not RunStatus.RUNNING:
            return
        safe_error = redact_exception(failure)
        error_info = RunErrorInfo(
            error_type=type(failure).__name__, message=safe_error
        )
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
            try:
                await self._middleware_pipeline.run_on_error(context, failure)
            except Exception:  # noqa: BLE001
                _LOGGER.exception(
                    "on_error middleware failed for run %s",
                    context.run_id,
                )
        try:
            await append_event(
                self._event_store,
                EventStreamContext.from_run_context(context),
                RunFailed(
                    run_id=context.run_id,
                    error_type=type(failure).__name__,
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
                        "error_type": type(failure).__name__,
                    },
                )
            except Exception:  # noqa: BLE001
                _LOGGER.exception(
                    "failure metrics failed for run %s", context.run_id
                )

    async def _fail_committed_run(
        self,
        context: RunContext,
        commit_sink: "_PendingCommit",
        failure: BaseException,
    ) -> None:
        """Drive the fail-transition when a post-drain commit (pause/complete)
        itself raises. execute() has already exited cleanly, so its own
        fail-closure cannot cover this case; the collector drives the identical
        transition using the RUNNING version execute() stashed on the sink.
        The terminal/commit-point guard inside :meth:`_apply_fail_transition`
        keeps an already-committed state authoritative (a partially-successful
        commit is not contradicted)."""
        running_version = commit_sink.running_version
        if running_version is None:
            # No version stashed -- nothing safe to transition against. This
            # should not happen (execute() sets it whenever a sink is wired);
            # leave state untouched rather than guessing a version.
            return
        await self._apply_fail_transition(
            context, running_version, failure, metrics=self._metrics
        )

    async def _drive_paused_or_result(
        self,
        agent: CompiledAgent,
        request: RunInput,
        context: RunContext,
        *,
        message_history: "Sequence[ModelMessage] | None" = None,
        emit_run_started: bool = True,
        running_version: "int | None" = None,
    ) -> "tuple[dict | None, RunResult | None]":
        """Shared collector behind both ``run()`` and ``execute_outcome()``:
        drains execute() to completion and reads the committed RunResult back
        from the RunStore (execute() already persisted it via the SUCCEEDED
        transition -- this is the single source of truth, not threaded state).
        Returns ``(paused_event, None)`` on pause, ``(None, result)`` on
        success.

        Any exception this raises (CancelledError or a genuine model/runtime
        error) is the EXACT SAME exception execute() raised -- never
        translated, wrapped, or swallowed -- so ``run()`` keeps its existing
        raise-based contract byte-for-byte and ``execute_outcome()`` (which
        wraps this call in its OWN try/except) can still classify the
        original exception type.

        All four terminal paths now move their Store-writing commit/
        transition OUT of execute() via a ``commit_sink`` (WP9 step 3):
        execute() stashes a command or closure instead of performing the
        Store write inline, and this method performs it -- for success/pause,
        AFTER the generator has already fully drained (nothing else runs
        concurrently, so the reordering is behavior-preserving); for cancel/
        fail, from this method's OWN except blocks, immediately before re-
        raising the SAME exception execute() raised (the closures execute()
        hands over already capture context/running_version/the exception
        itself, so no state needs to be threaded out by hand). This is the
        single location (for BOTH ``run()`` and ``execute_outcome()``) every
        terminal commit now happens from, rather than each of execute()'s 3
        former callers duplicating it. The pause commit additionally patches
        the RESOLVED approval id (``create_or_get_pending`` may dedup onto an
        existing id) onto the returned event before handing it back.

        ``contextlib.aclosing`` guarantees the generator is finalized in THIS
        task when the caller exits early on a pause -- execute() yields the
        paused event with no open context manager, but aclosing is belt-and-
        suspenders so any future change to execute()'s suspension points
        stays safe."""
        commit_sink = _PendingCommit()
        paused_event: "dict | None" = None
        try:
            async with contextlib.aclosing(
                self.execute(
                    agent,
                    request,
                    context,
                    message_history=message_history,
                    commit_sink=commit_sink,
                    emit_run_started=emit_run_started,
                    running_version=running_version,
                )
            ) as gen:
                async for event in gen:
                    if event["type"] == "paused":
                        paused_event = event
                        break
        except asyncio.CancelledError:
            if commit_sink.on_cancelled is not None:
                await commit_sink.on_cancelled()
            raise
        except Exception:
            if commit_sink.on_failed is not None:
                await commit_sink.on_failed()
            raise

        if paused_event is not None:
            if commit_sink.pause_command is not None:
                try:
                    _commit = await self._commit_coordinator.pause(
                        commit_sink.pause_command
                    )
                except Exception as commit_exc:  # noqa: BLE001
                    # The pause commit raised (e.g. a critical event-append
                    # failure). execute() has already exited cleanly, so its
                    # fail-closure cannot cover this -- drive the identical
                    # fail-transition here, then re-raise so the caller sees
                    # the original cause.
                    await self._fail_committed_run(context, commit_sink, commit_exc)
                    raise
                paused_event["approval_id"] = _commit.approval_id
            return paused_event, None

        # execute() must always leave a pending commit on a non-paused drain.
        # If it did not, that is a runtime invariant violation -- raise rather
        # than fabricate an empty success that masks the bug.
        if commit_sink.complete_command is None:
            raise RunInvariantError(
                f"run {context.run_id} completed without a pending commit"
            )
        try:
            committed = await self._commit_coordinator.complete(
                commit_sink.complete_command
            )
        except Exception as commit_exc:  # noqa: BLE001
            await self._fail_committed_run(context, commit_sink, commit_exc)
            raise
        if commit_sink.record_metrics is not None:
            commit_sink.record_metrics()
        if committed.result is None:
            raise RunInvariantError(
                f"run {context.run_id} committed without a persisted result"
            )
        return None, committed.result

    async def run(
        self,
        agent: CompiledAgent,
        request: RunInput,
        context: RunContext,
        *,
        emit_run_started: bool = True,
        running_version: "int | None" = None,
    ) -> RunResult:
        """Non-streaming entry point. Consumes execute() in full (via
        :meth:`_drive_paused_or_result`), re-raises RunPaused when the
        lifecycle yields a pause signal, and returns the final RunResult. All
        lifecycle concerns -- events, middleware, transitions, timeout,
        budget, observability -- live in execute(); run() is purely a
        collector.

        ``emit_run_started=False`` (WP9 step 3): the caller (RunCoordinator)
        already emitted the RunStarted event itself before calling here --
        the default ``True`` preserves execute()'s original behavior for
        every direct-engine caller/test that doesn't pass this.

        ``running_version`` (WP9 step 3): when the caller (RunCoordinator)
        already create_and_start_run'd the record, it passes the RUNNING
        version here so execute() skips its own get-or-create. ``None`` (the
        default) preserves execute()'s original get-or-create for every
        direct-engine caller/test."""
        paused_event, result = await self._drive_paused_or_result(
            agent,
            request,
            context,
            emit_run_started=emit_run_started,
            running_version=running_version,
        )
        if paused_event is not None:
            raise RunPaused(
                run_id=paused_event["run_id"],
                approval_id=paused_event["approval_id"],
            )
        return result

    async def dispatch(self, request: "RunDispatchRequest") -> RunResult:
        """RunDispatcher adapter: lets Swarm/Subagent execution depend on the
        narrow RunDispatcher Protocol instead of importing AgentEngine
        directly."""
        return await self.run(request.agent, request.input, request.context)

    async def execute_outcome(
        self,
        *,
        context: RunContext,
        agent: CompiledAgent,
        input: AgentInput,
        cancellation: "CancellationToken | None" = None,
        message_history: "Sequence[ModelMessage] | None" = None,
        emit_run_started: bool = True,
        running_version: "int | None" = None,
    ) -> AgentExecutionOutcome:
        """Section 12.4's target ``AgentEngine.execute()`` surface: a single
        ``AgentExecutionOutcome`` instead of the legacy async-generator-of-
        dict-events shape. Shares :meth:`_drive_paused_or_result` with
        ``run()`` (the single collector both now depend on) but wraps it in
        its OWN try/except to translate a raised exception into a FAILED/
        CANCELLED outcome instead of letting it propagate -- that generator
        is UNCHANGED and still owns the RunStore/Session/Event/Checkpoint
        writes internally, so this method matches the target CALL signature
        and return SHAPE without satisfying "AgentEngine must not depend on
        any Store" (spec 12.2) on its own; ``RunCoordinator`` (and its
        resume/subagent/swarm callers) can depend on the outcome shape
        immediately, ahead of the larger, separate increment that removes
        AgentEngine's Store dependencies for good.

        ``cancellation`` is accepted for signature conformance; the existing
        generator's own ``RunController``-based cancellation (registered by
        the constructor, not per-call) is what drives cancellation here."""
        request = RunInput(prompt=input.prompt, metadata=input.metadata)
        try:
            paused_event, result = await self._drive_paused_or_result(
                agent,
                request,
                context,
                message_history=message_history,
                emit_run_started=emit_run_started,
                running_version=running_version,
            )
        except asyncio.CancelledError:
            return AgentExecutionOutcome(status=AgentExecutionStatus.CANCELLED)
        except Exception as exc:  # noqa: BLE001 - reported via the outcome, not raised
            safe_error = redact_exception(exc)
            return AgentExecutionOutcome(
                status=AgentExecutionStatus.FAILED,
                error=RunErrorInfo(error_type=type(exc).__name__, message=safe_error),
            )

        if paused_event is not None:
            return AgentExecutionOutcome(
                status=AgentExecutionStatus.PAUSED,
                pause_request=PauseRequest(approval_id=paused_event["approval_id"]),
            )

        token_usage = result.token_usage
        return AgentExecutionOutcome(
            status=AgentExecutionStatus.COMPLETED,
            result=result,
            usage=RunUsage(
                input_tokens=token_usage.get("input_tokens", 0),
                output_tokens=token_usage.get("output_tokens", 0),
            ),
        )

    async def run_stream(
        self,
        agent: CompiledAgent,
        request: RunInput,
        context: RunContext,
        message_history: "Sequence[ModelMessage] | None" = None,
        *,
        emit_run_started: bool = True,
        running_version: "int | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming entry point. Resume (Runtime.resume) supplies
        ``message_history=<deserialized checkpoint>`` so the pydantic-ai graph
        picks up from the paused state.

        With no ``event_bus`` wired (the default -- every direct-engine
        caller/test), this is a thin pass-through to execute(): every event
        the lifecycle yields (text deltas, tool events, paused) is forwarded
        unchanged, raising on model/runtime failure exactly as before.

        With an ``event_bus`` wired (the production ``build_runtime`` path),
        ``execute_outcome()`` drives the lifecycle in a background task while
        this generator drains the SAME live events off the bus -- the seam
        that lets execute()'s own generator eventually be retired. Model
        failure/cancellation are reported as a final status event
        (``{"type": "failed"/"cancelled", ...}``) instead of a raised
        exception, per the Outcome model (spec section 12.3). A genuine bug
        in ``execute_outcome()`` itself (e.g. RunInvariantError) still
        propagates as a raised exception -- that is not a normal run
        outcome."""
        if self._event_bus is None:
            async for event in self.execute(
                agent,
                request,
                context,
                message_history=message_history,
                emit_run_started=emit_run_started,
                running_version=running_version,
            ):
                yield event
            return

        bus = self._event_bus
        run_id = context.run_id
        bus.open(run_id)
        task = asyncio.create_task(
            self.execute_outcome(
                context=context,
                agent=agent,
                input=AgentInput(prompt=request.prompt, metadata=request.metadata),
                message_history=message_history,
                emit_run_started=emit_run_started,
                running_version=running_version,
            )
        )

        async def _close_bus_when_done() -> None:
            try:
                await task
            finally:
                bus.close(run_id)

        closer = asyncio.create_task(_close_bus_when_done())
        try:
            async for event in bus.subscribe(run_id):
                yield event
        finally:
            await closer

        outcome = task.result()
        if outcome.status is AgentExecutionStatus.FAILED:
            error = outcome.error
            yield {
                "type": "failed",
                "run_id": run_id,
                "error_type": error.error_type if error is not None else "RuntimeError",
                "message": error.message if error is not None else "",
            }
        elif outcome.status is AgentExecutionStatus.CANCELLED:
            yield {"type": "cancelled", "run_id": run_id}
