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
import contextlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..errors import InvalidRunTransitionError, ModelPolicyExceededError, ModelRoutingError, RunConflictError, RunPaused
from ..events.payloads import (
    ApprovalRequested,
    RunCompleted,
    RunFailed,
    RunPaused as RunPausedEvent,
    RunStarted,
)
from ..events.store import EventStore
from ..middleware.pipeline import MiddlewarePipeline
from ..observability.tracing import use_span
from ..policy.engine import ToolContext
from ..run.cancellation import CancellationToken
from ..run.checkpoint import CheckpointStore
from ..run.context import RunContext
from ..run.controller import RunController
from ..run.models import (
    RunCheckpoint,
    RunErrorInfo,
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
)
from ..run.store import RunStore
from ..session.models import MessageRole, NewSessionMessage
from ..session.store import SessionStore
from .checkpoint_io import serialize_messages
from .dependencies import AgentDependencies
from .models import CompiledAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence
    from contextlib import AbstractAsyncContextManager

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.toolsets import AbstractToolset

    from ..agent.approval import ApprovalStore
    from ..capability.assembler import CapabilityAssembler
    from ..capability.options import CapabilityRuntimeOptions
    from ..execution.protocols import ExecutionBackend
    from ..knowledge.retriever import Retriever
    from ..memory.store import MemoryStore
    from ..observability.metrics import ObservabilityMetrics
    from ..observability.tracing import ObservabilitySink
    from ..storage.sqlalchemy.facade import _UnitOfWork


_LOGGER = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _noop_span():
    """Async context manager that yields ``None`` and does nothing -- the
    fallback for :meth:`AgentRunner._span` when observability is not wired,
    so the lifecycle body has a single ``async with`` shape regardless."""
    yield None


class AgentRunner:
    def __init__(self, *, run_store: RunStore, session_store: SessionStore,
                 event_store: EventStore, checkpoint_store: CheckpointStore,
                 middleware_pipeline: "MiddlewarePipeline | None" = None,
                 memory_store: "MemoryStore | None" = None,
                 retriever: "Retriever | None" = None,
                 observability: "ObservabilitySink | None" = None,
                 metrics: "ObservabilityMetrics | None" = None,
                 uow_factory: "Callable[[], AbstractAsyncContextManager[_UnitOfWork]] | None" = None,
                 run_controller: "RunController | None" = None,
                 execution: "ExecutionBackend | None" = None,
                 approval_store: "ApprovalStore | None" = None,
                 capability_assembler: "CapabilityAssembler | None" = None,
                 capability_options: "CapabilityRuntimeOptions | None" = None,
                 security_pipeline: Any = None,
                 baseline_policy: Any = None,
                 tool_policy_provider: Any = None,
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
        # Cross-store UnitOfWork factory (SqlAlchemy only). When wired, the
        # pause path wraps checkpoint-save + Run-transition + event-append in
        # one shared AsyncSession + one transaction, so they commit/rollback
        # atomically. None for FileStorage -- File cannot
        # promise cross-store transactions, so the pause path keeps its
        # best-effort non-atomic shape.
        self._uow_factory = uow_factory
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
        # conversational-only run, byte-for-byte identical to the prior
        # ``workdir=None`` path. Holding the backend on the runner (not the
        # compiler) is what decouples AgentCompiler from the filesystem.
        self._execution = execution
        # File-mode approval persistence for
        # the pause path (SqlAlchemy mode reaches the approval store via
        # ``tx.approvals`` inside the UoW instead). None (default) means a
        # RunPaused with a tool_call_id simply cannot persist its approval in
        # File mode without this wired -- Runtime.build() always wires it
        # from storage.approvals.
        self._approval_store = approval_store
        # Capability Runtime: when an AgentSpec declares non-
        # empty tools and an assembler is wired, execute() resolves those tools
        # into prompt sections + toolsets via the capability providers. Default
        # None preserves the legacy behavior (empty tools -> default builtin
        # toolset when an execution backend is present).
        self._capability_assembler = capability_assembler
        self._capability_options = capability_options
        self._security_pipeline = security_pipeline
        self._baseline_policy = baseline_policy
        self._tool_policy_provider = tool_policy_provider

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
                id=context.run_id, root_run_id=context.root_run_id,
                parent_run_id=context.parent_run_id, session_id=context.session_id,
                runnable_id=context.runnable_id, runnable_type=context.runnable_type,
                status=RunStatus.PENDING, input=request, result=None, error=None,
                version=1, created_at=now, started_at=None, finished_at=None,
            )
            created = await self._run_store.create(record)
            running = await self._run_store.transition(
                context.run_id, RunStatus.RUNNING, expected_version=created.version,
            )
            running_version = running.version
        else:
            # Resume: Runtime.resume already transitioned WAITING_APPROVAL ->
            # RUNNING; capture the current version for terminal transitions.
            current = await self._run_store.get(context.run_id)
            running_version = current.version

        # Real cancellation. When a RunController is wired,
        # register the driving asyncio.Task + a fresh CancellationToken so
        # Runtime.cancel(run_id) can actually stop this run: the token is
        # checked at the model-call execution points below, and the task is
        # cancelled to interrupt any hanging await inside the model call.
        # When ``run_controller`` is None (default), ``token`` stays None and
        # every check below is skipped -- cancellation then works purely via
        # asyncio's CancelledError path (the prior behavior, so the
        # default-None path is observationally identical to before).
        token: "CancellationToken | None" = None
        if self._run_controller is not None:
            token = CancellationToken()
            current_task = asyncio.current_task()
            if current_task is not None:
                await self._run_controller.register(
                    context.run_id, current_task, token,
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
                    await self._event_store.append(
                        stream_id=context.run_id,
                        run_id=context.run_id,
                        root_run_id=context.root_run_id,
                        parent_run_id=context.parent_run_id,
                        session_id=context.session_id,
                        runnable_id=context.runnable_id,
                        payload=RunStarted(
                            run_id=context.run_id,
                            runnable_id=context.runnable_id),
                    )
                    if self._middleware_pipeline is not None:
                        await self._middleware_pipeline.run_before_run(context)

                prior_messages = await self._session_store.list_messages(
                    context.session_id)
                # Prompt window policy: trim session history before it
                # is folded into the prompt. Opt-in via CapabilityRuntimeOptions;
                # None (default) leaves prior_messages untouched.
                window_policy = (
                    self._capability_options.session_window_policy
                    if self._capability_options is not None else None
                )
                if window_policy is not None:
                    from ..events.payloads import PromptWindowApplied
                    before_count = len(prior_messages)
                    prior_messages = list(await window_policy.select_messages(
                        prior_messages, agent.spec.model))
                    await self._event_store.append(
                        stream_id=context.run_id, run_id=context.run_id,
                        root_run_id=context.root_run_id, parent_run_id=context.parent_run_id,
                        session_id=context.session_id, runnable_id=context.runnable_id,
                        payload=PromptWindowApplied(
                            policy=type(window_policy).__name__,
                            before=before_count, after=len(prior_messages)),
                    )

                # -- Prompt build. Resume path skips this entirely -- the
                # prompt is baked into the checkpointed message_history.
                prompt: "str | None" = None
                if message_history is None:
                    history_text = "\n".join(str(m.content) for m in prior_messages)
                    prompt = (f"{history_text}\n{request.prompt}"
                              if history_text else request.prompt)

                    # Memory + Knowledge injection via substitutable policies.
                    # Each fires only when its policy is wired (explicit on
                    # options, or the runner's default built from a wired store/
                    # retriever) AND yields a non-empty section. Memory is
                    # injected first, then knowledge -- both prepend, so the
                    # final top-to-bottom order is: Knowledge, Memory, history,
                    # user prompt.
                    memory_policy = self._effective_memory_policy()
                    if memory_policy is not None:
                        memories = await memory_policy.select_memories(context, request.prompt)
                        section = self._prompt_formatter().format_memory(memories)
                        if section:
                            prompt = f"{section}\n{prompt}"
                    retrieval_policy = self._effective_retrieval_policy()
                    if retrieval_policy is not None:
                        items = await retrieval_policy.retrieve(context, request.prompt)
                        section = self._prompt_formatter().format_knowledge(items)
                        if section:
                            prompt = f"{section}\n{prompt}"

                # -- Model call: agent.pydantic_agent.iter() drives the graph.
                # The per-Run ToolContext travels to capabilities via pydantic-ai
                # DI: ``deps=`` becomes ``ctx.deps.tool_context`` inside every
                # capability hook (safe concurrent reuse
                # of one CompiledAgent across many Runs).
                tool_context = ToolContext(
                    run_id=context.run_id, session_id=context.session_id,
                    tool_call_id=None)
                # The builtin file/terminal toolset is
                # constructed HERE, at execution time, from the per-Runtime
                # ExecutionBackend -- not baked into the compiled Agent. This
                # is what makes AgentCompiler stateless (no filesystem surface):
                # the same CompiledAgent can be reused across Runs that target
                # different working directories, and a conversational-only run
                # (no execution backend) simply passes ``toolsets=[]`` so the
                # model has no file/terminal tools available. ``deps.execution``
                # is the same backend -- capabilities that need it read
                # ``ctx.deps.execution`` (future work).
                deps = AgentDependencies(
                    tool_context=tool_context, execution=self._execution,
                )
                toolsets: "list[AbstractToolset]" = []
                capability_prompt = ""
                if agent.spec.tools and self._capability_assembler is not None:
                    # Declared tools: resolve via the capability assembler. Empty
                    # tools fall through to the legacy default below.
                    from ..capability.policy import CapabilityToolExposurePolicy
                    from ..capability.provider import CapabilityContext, toolset_names as _count_tool_names

                    exposure = (
                        self._capability_options.tool_exposure
                        if self._capability_options is not None
                        else CapabilityToolExposurePolicy()
                    )
                    cap_ctx = CapabilityContext(
                        agent_id=agent.spec.id, exposure_policy=exposure,
                        execution=deps.execution, run_id=context.run_id,
                        root_run_id=context.root_run_id, session_id=context.session_id,
                        event_store=self._event_store,
                        user_id=context.user_id, tenant_id=context.tenant_id,
                        workspace=context.workspace,
                    )
                    cap_bundle = await self._capability_assembler.assemble(agent.spec, cap_ctx)
                    # Managed path: when a SecurityPipeline or baseline is
                    # configured, wrap each tool through ManagedToolAdapter so
                    # every call passes through the unified governance chain
                    # (policy, pipeline, baseline, timeout, events). Otherwise
                    # use the legacy direct-toolset path (backward compat).
                    use_managed = (self._security_pipeline is not None
                                   or self._baseline_policy is not None)
                    if use_managed and cap_bundle.tool_contributions:
                        import inspect as _inspect
                        from pydantic_ai.toolsets import FunctionToolset as _FTS
                        from ..tool.auto_descriptor import extract_handler
                        from ..tool.managed import ManagedToolAdapter
                        from ..tool.managed_toolset import ManagedToolsetWrapper
                        managed_ts = _FTS()
                        for contrib in cap_bundle.tool_contributions:
                            # Try per-handler extraction (FunctionToolset path).
                            any_extracted = False
                            for desc in contrib.descriptors:
                                handler = extract_handler(contrib.toolset, desc.name)
                                if handler is None:
                                    continue
                                any_extracted = True
                                _adapter = ManagedToolAdapter(
                                    descriptor=desc, handler=handler,
                                    policy_provider=self._tool_policy_provider,
                                    security_pipeline=self._security_pipeline,
                                    baseline_policy=self._baseline_policy,
                                    run_context=context,
                                )
                                async def _managed_invoke(_a=_adapter, **kw):
                                    return await _a.invoke(**kw)
                                _managed_invoke.__name__ = desc.name
                                try:
                                    _managed_invoke.__signature__ = _inspect.signature(handler)
                                except (ValueError, TypeError):
                                    pass
                                managed_ts.add_function(_managed_invoke)
                            # If no handlers extracted (opaque toolset like MCP),
                            # wrap the entire toolset at the call_tool level.
                            if not any_extracted and contrib.descriptors:
                                wrapper = ManagedToolsetWrapper(
                                    contrib.toolset,
                                    descriptor=contrib.descriptors[0],
                                    security_pipeline=self._security_pipeline,
                                    run_context=context,
                                )
                                toolsets.append(wrapper)
                        if managed_ts.tools:
                            toolsets.append(managed_ts)
                    else:
                        toolsets.extend(cap_bundle.toolsets)
                    from ..events.payloads import PromptCatalogInjected, ToolExposureApplied
                    await self._event_store.append(
                        stream_id=context.run_id, run_id=context.run_id,
                        root_run_id=context.root_run_id, parent_run_id=context.parent_run_id,
                        session_id=context.session_id, runnable_id=context.runnable_id,
                        payload=ToolExposureApplied(
                            agent_id=agent.spec.id,
                            total_tools=len(_count_tool_names(toolsets)),
                        ),
                    )
                    if cap_bundle.prompt_sections:
                        capability_prompt = "\n\n".join(cap_bundle.prompt_sections.values())
                        for section in cap_bundle.prompt_sections:
                            await self._event_store.append(
                                stream_id=context.run_id, run_id=context.run_id,
                                root_run_id=context.root_run_id, parent_run_id=context.parent_run_id,
                                session_id=context.session_id, runnable_id=context.runnable_id,
                                payload=PromptCatalogInjected(
                                    agent_id=agent.spec.id, section=section),
                            )
                elif agent.spec.tools is None and deps.execution is not None:
                    # tools unset (None) + execution backend -> default builtin
                    # toolset. An explicit empty tuple (() = "no tools") or a
                    # non-empty tuple without an assembler leaves toolsets empty.
                    from ..execution.toolset import (
                        BuiltinToolContext,
                        build_builtin_toolset,
                    )
                    toolsets.append(build_builtin_toolset(BuiltinToolContext(
                        backend=deps.execution,
                        enabled_tools={"file", "terminal"},
                    )))
                # ModelPolicy.timeout_seconds is enforced by wrapping
                # each graph step (``run.__anext__()``) in asyncio.wait_for with
                # the REMAINING budget. The model call happens at this await
                # point (for stream-less models / the non-streaming path), so
                # wait_for can interrupt a hanging model call. timeout_seconds
                # left at None reproduces the prior path (no wait_for).
                timeout = agent.spec.model.timeout_seconds

                accumulated_text = ""
                result = None
                # check the cancellation token BEFORE the model call.
                # raise_if_cancelled() is a no-op when the token is not set
                # (or when no controller is wired -- token is None), so this
                # is observationally invisible on the default path.
                if token is not None:
                    await token.raise_if_cancelled()
                effective_prompt = (
                    capability_prompt + "\n\n" + prompt if capability_prompt else prompt
                )
                if message_history is not None:
                    run_iter = agent.pydantic_agent.iter(
                        message_history=message_history, deps=deps,
                        toolsets=toolsets)
                else:
                    run_iter = agent.pydantic_agent.iter(
                        effective_prompt, deps=deps, toolsets=toolsets)

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
                                            remaining = (timeout
                                                         - (time.monotonic() - iter_started))
                                            if remaining <= 0:
                                                timed_out = True
                                                break
                                            node = await asyncio.wait_for(
                                                run.__anext__(), remaining)
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
                                        if (timeout is not None
                                                and (time.monotonic() - iter_started) >= timeout):
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
                                            async with node.stream(run.ctx) as request_stream:
                                                async for ev in request_stream:
                                                    text = None
                                                    if (isinstance(ev, PartStartEvent)
                                                            and isinstance(ev.part, TextPart)):
                                                        text = ev.part.content
                                                    elif (isinstance(ev, PartDeltaEvent)
                                                          and isinstance(ev.delta, TextPartDelta)):
                                                        text = ev.delta.content_delta
                                                    if text:
                                                        accumulated_text += text
                                                        yield {"type": "text", "text": text}
                                        except Exception:
                                            # Model lacks stream_function or
                                            # otherwise can't stream -- skip.
                                            # Model call already happened via
                                            # __anext__; final result.output
                                            # carries the answer.
                                            pass
                                    elif PydanticAgent.is_call_tools_node(node):
                                        try:
                                            async with node.stream(run.ctx) as tool_stream:
                                                async for ev in tool_stream:
                                                    if isinstance(ev, FunctionToolCallEvent):
                                                        yield {
                                                            "type": "tool",
                                                            "name": ev.part.tool_name,
                                                            "phase": "start", "ok": None,
                                                        }
                                                    elif isinstance(ev, FunctionToolResultEvent):
                                                        yield {
                                                            "type": "tool",
                                                            "name": ev.part.tool_name,
                                                            "phase": "end",
                                                            "ok": isinstance(ev.part, ToolReturnPart),
                                                        }
                                        except Exception:
                                            # Same degradation as above -- the
                                            # tools already ran via __anext__.
                                            pass
                            except RunPaused as paused:
                                # Pause path (canonical surface): persist the
                                # ApprovalRequest, save a real checkpoint of the
                                # partial message history, transition to
                                # WAITING_APPROVAL, emit ApprovalRequested +
                                # RunPaused events -- all INSIDE the iter()
                                # context so ``run.all_messages()`` works. The
                                # paused-event yield itself is deferred to
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
                                if self._uow_factory is not None:
                                    # Atomic (SqlAlchemy): all writes bind to
                                    # one AsyncSession + one transaction. Any
                                    # failure rolls back everything AND
                                    # propagates to the outer generic-except
                                    # handler so the Run ends up FAILED rather
                                    # than left in a half-paused state.
                                    async with self._uow_factory() as tx:
                                        if paused.tool_call_id is not None:
                                            approval = await tx.approvals.create_or_get_pending(
                                                run_id=paused.run_id,
                                                tool_call_id=paused.tool_call_id,
                                                tool_name=paused.tool_name or "",
                                                reason=paused.reason,
                                                arguments=paused.arguments,
                                                approval_id=paused.approval_id,
                                            )
                                            paused.approval_id = approval.id
                                        checkpoint = RunCheckpoint(
                                            id=str(uuid.uuid4()), run_id=context.run_id,
                                            sequence=1, format="pydantic-ai-v1",
                                            schema_version=1,
                                            payload=serialize_messages(run.all_messages()),
                                            created_at=datetime.now(timezone.utc),
                                            metadata={"approval_id": paused.approval_id},
                                        )
                                        approval_requested_payload = ApprovalRequested(
                                            approval_id=paused.approval_id,
                                            tool_name=paused.tool_name or "",
                                            reason=paused.reason or "",
                                        )
                                        paused_payload = RunPausedEvent(
                                            run_id=context.run_id,
                                            reason=f"approval required: {paused.approval_id}",
                                        )
                                        await tx.checkpoints.save(checkpoint)
                                        # WAITING_APPROVAL
                                        # transition MUST propagate. If it
                                        # fails the run cannot be paused --
                                        # rolling back + propagating avoids
                                        # leaving the checkpoint saved but the
                                        # run still RUNNING (the inconsistent
                                        # state forbidden by atomicity).
                                        await tx.runs.transition(
                                            context.run_id,
                                            RunStatus.WAITING_APPROVAL,
                                            expected_version=running_version,
                                        )
                                        await tx.events.append(
                                            stream_id=context.run_id,
                                            run_id=context.run_id,
                                            root_run_id=context.root_run_id,
                                            parent_run_id=context.parent_run_id,
                                            session_id=context.session_id,
                                            runnable_id=context.runnable_id,
                                            payload=approval_requested_payload,
                                        )
                                        await tx.events.append(
                                            stream_id=context.run_id,
                                            run_id=context.run_id,
                                            root_run_id=context.root_run_id,
                                            parent_run_id=context.parent_run_id,
                                            session_id=context.session_id,
                                            runnable_id=context.runnable_id,
                                            payload=paused_payload,
                                        )
                                else:
                                    # File mode: non-atomic best-effort.
                                    # Cross-store transactions are unavailable,
                                    # so checkpoint + transition propagate
                                    # (masking them is forbidden) but the approval
                                    # write and event appends stay best-effort
                                    # -- the run is already WAITING_APPROVAL, so
                                    # a missing approval/event is a recovery
                                    # gap, not state corruption.
                                    if (self._approval_store is not None
                                            and paused.tool_call_id is not None):
                                        try:
                                            approval = await self._approval_store.create_or_get_pending(
                                                run_id=paused.run_id,
                                                tool_call_id=paused.tool_call_id,
                                                tool_name=paused.tool_name or "",
                                                reason=paused.reason,
                                                arguments=paused.arguments,
                                                approval_id=paused.approval_id,
                                            )
                                            paused.approval_id = approval.id
                                        except Exception as exc:  # noqa: BLE001
                                            _LOGGER.warning(
                                                "failed to persist ApprovalRequest for run %s: %s",
                                                context.run_id, exc,
                                            )
                                    checkpoint = RunCheckpoint(
                                        id=str(uuid.uuid4()), run_id=context.run_id,
                                        sequence=1, format="pydantic-ai-v1",
                                        schema_version=1,
                                        payload=serialize_messages(run.all_messages()),
                                        created_at=datetime.now(timezone.utc),
                                        metadata={"approval_id": paused.approval_id},
                                    )
                                    approval_requested_payload = ApprovalRequested(
                                        approval_id=paused.approval_id,
                                        tool_name=paused.tool_name or "",
                                        reason=paused.reason or "",
                                    )
                                    paused_payload = RunPausedEvent(
                                        run_id=context.run_id,
                                        reason=f"approval required: {paused.approval_id}",
                                    )
                                    await self._checkpoint_store.save(checkpoint)
                                    await self._run_store.transition(
                                        context.run_id, RunStatus.WAITING_APPROVAL,
                                        expected_version=running_version,
                                    )
                                    try:
                                        await self._event_store.append(
                                            stream_id=context.run_id,
                                            run_id=context.run_id,
                                            root_run_id=context.root_run_id,
                                            parent_run_id=context.parent_run_id,
                                            session_id=context.session_id,
                                            runnable_id=context.runnable_id,
                                            payload=approval_requested_payload,
                                        )
                                        await self._event_store.append(
                                            stream_id=context.run_id,
                                            run_id=context.run_id,
                                            root_run_id=context.root_run_id,
                                            parent_run_id=context.parent_run_id,
                                            session_id=context.session_id,
                                            runnable_id=context.runnable_id,
                                            payload=paused_payload,
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        _LOGGER.warning(
                                            "failed to append RunPaused event for run %s: %s",
                                            context.run_id, exc,
                                        )
                                paused_signal = paused
                            else:
                                if not timed_out:
                                    result = run.result
                except asyncio.TimeoutError:
                    # TimeoutError escaping the iter() context (either from
                    # wait_for directly or from iter() __aexit__ cleanup).
                    # Only treat as model timeout when the deadline actually
                    # passed; an unrelated TimeoutError propagates.
                    if timeout is not None and (time.monotonic() - iter_started) >= timeout:
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
                    if (timeout is not None
                            and (time.monotonic() - iter_started) >= timeout):
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

                    # max_tokens enforcement. usage is read once so the
                    # check and the RunResult.token_usage share the same
                    # snapshot. budget (cost-per-token) is declared on
                    # ModelPolicy but deferred -- only the token-count limit is
                    # enforced here.
                    usage = result.usage if result is not None else None
                    max_tokens = agent.spec.model.max_tokens
                    if max_tokens is not None and usage is not None:
                        used = usage.input_tokens + usage.output_tokens
                        if used > max_tokens:
                            raise ModelPolicyExceededError(
                                f"max_tokens exceeded: used {used} > max_tokens {max_tokens}",
                                kind="max_tokens",
                            )

                    # sequence is assigned by the SessionStore itself
                    # (NewSessionMessage carries no id/sequence/created_at) --
                    # the caller no longer computes `len(prior_messages) + 1`,
                    # which could race with a concurrent Run appending to the
                    # same session.
                    await self._session_store.append_messages(context.session_id, (
                        NewSessionMessage(
                            role=MessageRole.ASSISTANT,
                            content=str(output), run_id=context.run_id,
                        ),
                    ))

                    # ``run`` is still bound here (Python preserves the
                    # ``with ... as run`` target after the block). The iter()
                    # context has exited cleanly, but the AgentRun still
                    # serves message history.
                    await self._checkpoint_store.save(RunCheckpoint(
                        id=str(uuid.uuid4()), run_id=context.run_id, sequence=1,
                        format="pydantic-ai-v1", schema_version=1,
                        payload=serialize_messages(run.all_messages()),
                        created_at=datetime.now(timezone.utc),
                    ))

                    run_result = RunResult(
                        output=output,
                        token_usage={
                            "input_tokens": usage.input_tokens if usage else 0,
                            "output_tokens": usage.output_tokens if usage else 0,
                        },
                    )
                    await self._run_store.transition(
                        context.run_id, RunStatus.SUCCEEDED,
                        expected_version=running_version, result=run_result,
                    )

                    if self._middleware_pipeline is not None:
                        await self._middleware_pipeline.run_after_run(
                            context, run_result)

                    # Best-effort RunCompleted: on the resume path, the event
                    # stream may already hold a RunPaused event from the
                    # initial pause -- the store-assigned sequence simply
                    # appends after it (no caller sequence to collide with).
                    # best-effort audit - non-critical path:
                    # the run is already SUCCEEDED, so a missing event is an
                    # observability gap, not state corruption.
                    try:
                        await self._event_store.append(
                            stream_id=context.run_id,
                            run_id=context.run_id,
                            root_run_id=context.root_run_id,
                            parent_run_id=context.parent_run_id,
                            session_id=context.session_id,
                            runnable_id=context.runnable_id,
                            payload=RunCompleted(run_id=context.run_id),
                        )
                    except Exception as exc:  # noqa: BLE001
                        _LOGGER.warning(
                            "failed to append RunCompleted event for run %s: %s",
                            context.run_id, exc,
                        )

                    if metrics is not None:
                        metrics.counter("agent.run.completed", attributes=run_attrs)
                        metrics.histogram(
                            "agent.run.duration_ms",
                            value=round((time.monotonic() - started) * 1000, 3),
                            attributes=run_attrs,
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
                        context.run_id, RunStatus.CANCELLING,
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
                await self._run_store.transition(
                    context.run_id, RunStatus.CANCELLED,
                    expected_version=cancelling.version,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition run %s to CANCELLED on cancel: %s",
                    context.run_id, exc,
                )
            raise
        except Exception as exc:
            # Generic error path: transition FAILED, run on_error, append
            # RunFailed, record metrics. Re-raise so the caller sees the error.
            error_info = RunErrorInfo(
                error_type=type(exc).__name__, message=str(exc))
            # Run status update. The FAILED transition is kept
            # best-effort ONLY because we are already in the failing path:
            # letting the transition error escape would replace the ORIGINAL
            # exc (the actual cause) with a version-mismatch/store error,
            # losing the cause for the caller. The warning keeps the
            # transition failure visible rather than silent; a typical failure
            # is a version mismatch from a concurrent terminal transition.
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.FAILED,
                    expected_version=running_version, error=error_info,
                )
            except Exception as transition_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to transition run %s to FAILED: %s",
                    context.run_id, transition_exc,
                )
            if self._middleware_pipeline is not None:
                await self._middleware_pipeline.run_on_error(context, exc)
            # best-effort audit - non-critical path: the run is already failing
            # (FAILED transition attempted above), so a missing RunFailed event
            # is an observability gap, not state corruption.
            try:
                await self._event_store.append(
                    stream_id=context.run_id,
                    run_id=context.run_id,
                    root_run_id=context.root_run_id,
                    parent_run_id=context.parent_run_id,
                    session_id=context.session_id,
                    runnable_id=context.runnable_id,
                    payload=RunFailed(
                        run_id=context.run_id,
                        error_type=type(exc).__name__, message=str(exc)),
                )
            except Exception as event_exc:  # noqa: BLE001
                _LOGGER.warning(
                    "failed to append RunFailed event for run %s: %s",
                    context.run_id, event_exc,
                )
            if metrics is not None:
                metrics.counter("agent.run.failed", attributes={
                    "run_id": context.run_id, "session_id": context.session_id,
                    "error_type": type(exc).__name__,
                })
            raise
        finally:
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
        self, agent: CompiledAgent, request: RunInput, context: RunContext,
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
        async with contextlib.aclosing(
            self.execute(agent, request, context)
        ) as gen:
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
        # Defensive: execute() should always leave a terminal record. If
        # somehow it didn't, synthesize an empty result so run() honors its
        # return-type contract.
        return RunResult(output="")

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
            agent, request, context, message_history=message_history,
        ):
            yield event
