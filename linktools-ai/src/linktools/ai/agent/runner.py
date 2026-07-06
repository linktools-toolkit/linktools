#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentRunner: owns the per-invocation lifecycle -- Run state transitions,
Session history load/append, runner-driven Middleware hooks (before_run/after_run/
on_error), Event publication, Checkpoint save. The 4 pydantic-ai-intercepted hooks
fire via MiddlewareCapability, enabled by setting current_context on both
capabilities immediately before agent.run().

Message-history adaptation is a text-join MVP (SessionMessage.content values
prepended to the prompt). Checkpoint payload is empty bytes this phase -- real
pydantic-ai message serialization is a separate format concern.

Optional Memory + Knowledge injection (Phase 5): when ``memory_store`` and/or
``retriever`` are wired, ``run()`` queries them with the user prompt and prepends
``## Memory`` / ``## Knowledge`` sections to the prompt sent to the model. Both
default to None, so existing callers see no change. Final prompt order (when
both are set and non-empty): ``## Knowledge`` on top, then ``## Memory``, then
session history, then the user prompt -- memory is injected first (so it lands
below knowledge), knowledge second (so it lands on top).

Optional Observability (Phase 6): when ``observability`` is wired, ``run()``
wraps the lifecycle in an outer ``agent.run`` span and the model call in a
nested ``agent.model`` span (parented via the tracing contextvar). When
``metrics`` is wired, records ``counter("agent.run.completed"/"agent.run.failed")``
and ``histogram("agent.run.duration_ms")``. Both default to None, so the
default-None path is a no-op -- no spans opened, no metrics recorded, and the
existing lifecycle runs byte-for-byte as before (the ``if observability is not
None:`` guard is the sole gate)."""

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..errors import ModelPolicyExceededError, ModelRoutingError, RunPaused
from ..events.envelope import EventEnvelope
from ..events.payloads import (
    RunCompleted,
    RunFailed,
    RunPaused as RunPausedEvent,
    RunStarted,
)
from ..events.store import EventStore
from ..middleware.pipeline import MiddlewarePipeline
from ..observability.tracing import use_span
from ..policy.engine import ToolContext
from ..run.checkpoint import CheckpointStore
from ..run.context import RunContext
from ..run.models import RunCheckpoint, RunErrorInfo, RunInput, RunRecord, RunResult, RunStatus
from ..run.store import RunStore
from ..session.models import MessageRole, SessionMessage
from ..session.store import SessionStore
from .checkpoint_io import serialize_messages
from .models import CompiledAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from pydantic_ai.messages import ModelMessage

    from ..knowledge.retriever import Retriever
    from ..memory.store import MemoryStore
    from ..observability.metrics import ObservabilityMetrics
    from ..observability.tracing import ObservabilitySink


class AgentRunner:
    def __init__(self, *, run_store: RunStore, session_store: SessionStore,
                 event_store: EventStore, checkpoint_store: CheckpointStore,
                 middleware_pipeline: "MiddlewarePipeline | None" = None,
                 memory_store: "MemoryStore | None" = None,
                 retriever: "Retriever | None" = None,
                 observability: "ObservabilitySink | None" = None,
                 metrics: "ObservabilityMetrics | None" = None) -> None:
        self._run_store = run_store
        self._session_store = session_store
        self._event_store = event_store
        self._checkpoint_store = checkpoint_store
        self._middleware_pipeline = middleware_pipeline
        self._memory_store = memory_store
        self._retriever = retriever
        self._observability = observability
        self._metrics = metrics

    async def run(self, agent: CompiledAgent, request: RunInput, context: RunContext) -> RunResult:
        now = datetime.now(timezone.utc)
        record = RunRecord(
            id=context.run_id, root_run_id=context.root_run_id, parent_run_id=context.parent_run_id,
            session_id=context.session_id, runnable_id=context.runnable_id, runnable_type=context.runnable_type,
            status=RunStatus.PENDING, input=request, result=None, error=None, version=1,
            created_at=now, started_at=None, finished_at=None,
        )
        await self._run_store.create(record)
        await self._run_store.transition(context.run_id, RunStatus.RUNNING, expected_version=1)

        tool_context = ToolContext(run_id=context.run_id, session_id=context.session_id)
        agent.policy_capability.current_context = tool_context
        if agent.middleware_capability is not None:
            agent.middleware_capability.current_context = tool_context

        # Observability (Phase 6): when a sink is wired, open an outer "agent.run"
        # span around the whole lifecycle. The default-None path skips the span
        # and delegates directly -- the lifecycle body is unchanged, the sole
        # gate being the ``observability is None`` branch below.
        started = time.monotonic()
        observability = self._observability
        if observability is None:
            return await self._run_lifecycle(agent, request, context, started)
        async with use_span(
            observability, "agent.run",
            attributes={"run_id": context.run_id, "session_id": context.session_id},
        ):
            return await self._run_lifecycle(agent, request, context, started)

    async def _run_lifecycle(
        self, agent: CompiledAgent, request: RunInput, context: RunContext, started: float,
    ) -> RunResult:
        """The per-invocation lifecycle (events, middleware hooks, prompt build,
        model call, state transitions, metrics).

        Called either directly (no observability) or from inside the outer
        ``agent.run`` span. In the latter case the tracing contextvar is already
        set, so ``use_span("agent.model")`` parents to the outer span
        automatically. ``started`` is a ``time.monotonic()`` timestamp captured
        in :meth:`run` for the duration histogram."""
        observability = self._observability
        metrics = self._metrics
        run_attrs = {"run_id": context.run_id, "session_id": context.session_id}
        try:
            await self._event_store.append(self._envelope(context, sequence=1, payload=RunStarted(
                run_id=context.run_id, runnable_id=context.runnable_id)))

            if self._middleware_pipeline is not None:
                await self._middleware_pipeline.run_before_run(context)

            prior_messages = await self._session_store.list_messages(context.session_id)
            history_text = "\n".join(str(m.content) for m in prior_messages)
            prompt = f"{history_text}\n{request.prompt}" if history_text else request.prompt

            # Memory + Knowledge prompt injection (Phase 5). Each block is
            # optional and only fires when its dependency is wired AND yields a
            # non-empty section, so the default-None path is a no-op (existing
            # tests unchanged). Memory is injected first, then knowledge -- both
            # prepend, so the final order top-to-bottom is: Knowledge, Memory,
            # history, user prompt. Owner resolution prefers user_id, then
            # tenant_id, then session_id (the same identifiers Runtime.run()
            # threads into RunContext).
            if self._memory_store is not None:
                from ..knowledge.context import format_memory
                owner = context.user_id or context.tenant_id or context.session_id
                memories = await self._memory_store.search(
                    request.prompt, owner_id=owner, limit=5,
                )
                section = format_memory(memories)
                if section:
                    prompt = f"{section}\n{prompt}"
            if self._retriever is not None:
                from ..knowledge.context import KnowledgeContext
                docs = await self._retriever.search(request.prompt, limit=5)
                section = KnowledgeContext(documents=docs).format()
                if section:
                    prompt = f"{section}\n{prompt}"

            # Model call. When observability is wired, wrap in a nested
            # "agent.model" span parented to the outer "agent.run" span via the
            # tracing contextvar. When observability is None, call the model
            # directly -- no span, no overhead, identical to the pre-Phase-6
            # behavior.
            #
            # GAP-08 (spec 31): ModelPolicy.timeout_seconds wraps the model call
            # in asyncio.wait_for when set. On timeout the asyncio.TimeoutError
            # is translated to ModelRoutingError("model timeout") so the generic
            # FAILED handler below records a descriptive message; timeout_seconds
            # left at None reproduces the pre-GAP-08 path (no wait_for wrapper).
            timeout = agent.spec.model.timeout_seconds
            try:
                if observability is not None:
                    async with use_span(observability, "agent.model"):
                        if timeout is not None:
                            run_result = await asyncio.wait_for(
                                agent.pydantic_agent.run(prompt), timeout=timeout)
                        else:
                            run_result = await agent.pydantic_agent.run(prompt)
                else:
                    if timeout is not None:
                        run_result = await asyncio.wait_for(
                            agent.pydantic_agent.run(prompt), timeout=timeout)
                    else:
                        run_result = await agent.pydantic_agent.run(prompt)
            except asyncio.TimeoutError:
                raise ModelRoutingError("model timeout")
            output = run_result.output

            # GAP-08: ModelPolicy.max_tokens enforcement + token-usage capture.
            # usage is read once so the SUCCEEDED-gating check and the
            # RunResult.token_usage population share the same snapshot. budget is
            # declared on ModelPolicy but deferred -- no cost-per-token rates
            # exist yet, so only the token-count limit is enforced here.
            usage = run_result.usage
            max_tokens = agent.spec.model.max_tokens
            if max_tokens is not None:
                used = usage.input_tokens + usage.output_tokens
                if used > max_tokens:
                    raise ModelPolicyExceededError(
                        f"max_tokens exceeded: used {used} > max_tokens {max_tokens}",
                        kind="max_tokens",
                    )

            await self._session_store.append_messages(context.session_id, (
                SessionMessage(
                    id=f"{context.run_id}-response", session_id=context.session_id,
                    sequence=len(prior_messages) + 1, role=MessageRole.ASSISTANT, content=str(output),
                    run_id=context.run_id, created_at=datetime.now(timezone.utc),
                ),
            ))

            await self._checkpoint_store.save(RunCheckpoint(
                id=str(uuid.uuid4()), run_id=context.run_id, sequence=1,
                format="pydantic-ai-v1", schema_version=1,
                payload=serialize_messages(run_result.all_messages()),
                created_at=datetime.now(timezone.utc),
            ))

            # token_usage is threaded onto RunResult so the swarm layer can
            # accumulate it (GAP-09 max_total_tokens). Default-None policy leaves
            # usage populated straight off the model result (zero-cost when the
            # FunctionModel reports nothing).
            result = RunResult(
                output=output,
                token_usage={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                },
            )
            await self._run_store.transition(
                context.run_id, RunStatus.SUCCEEDED, expected_version=2, result=result)

            if self._middleware_pipeline is not None:
                result = await self._middleware_pipeline.run_after_run(context, result)

            await self._event_store.append(self._envelope(context, sequence=2, payload=RunCompleted(
                run_id=context.run_id)))

            if metrics is not None:
                metrics.counter("agent.run.completed", attributes=run_attrs)
                metrics.histogram(
                    "agent.run.duration_ms",
                    value=round((time.monotonic() - started) * 1000, 3),
                    attributes=run_attrs,
                )
            return result
        except RunPaused as paused:
            # Pause path (Task 7): transition to WAITING_APPROVAL, emit a
            # RunPaused event, and RE-RAISE so the caller gets the signal.
            # No checkpoint in v1 -- streaming is the canonical pause surface.
            # This handler sits BEFORE the generic ``except Exception`` so the
            # run is NOT marked FAILED (the bug it guards against).
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.WAITING_APPROVAL, expected_version=2)
            except Exception:
                pass
            try:
                await self._event_store.append(self._envelope(
                    context, sequence=2,
                    payload=RunPausedEvent(
                        run_id=context.run_id,
                        reason=f"approval required: {paused.approval_id}")))
            except Exception:
                pass
            raise
        except asyncio.CancelledError:
            # GAP-16: in-flight cancel path. When the caller cancels the
            # asyncio.Task driving this run, asyncio raises CancelledError at
            # the current await point. Catch it BEFORE the generic ``except
            # Exception`` (CancelledError is a BaseException, not an Exception,
            # since Python 3.8 -- so placement is for clarity, but the principle
            # holds: the run must end up CANCELLED, not FAILED), best-effort
            # transition the RunRecord to CANCELLED, then re-raise so the
            # asyncio machinery observes the cancellation. The trailing
            # ``finally`` still clears capability current_context.
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.CANCELLED, expected_version=2)
            except Exception:
                # best-effort: the run may already be terminal (e.g. a
                # concurrent Runtime.cancel beat this transition) or the
                # version assumption may be wrong -- either way the
                # cancellation must propagate.
                pass
            raise
        except Exception as exc:
            error_info = RunErrorInfo(error_type=type(exc).__name__, message=str(exc))
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.FAILED, expected_version=2, error=error_info)
            except Exception:
                pass
            if self._middleware_pipeline is not None:
                await self._middleware_pipeline.run_on_error(context, exc)
            try:
                await self._event_store.append(self._envelope(context, sequence=2, payload=RunFailed(
                    run_id=context.run_id, error_type=type(exc).__name__, message=str(exc))))
            except Exception:
                pass
            if metrics is not None:
                metrics.counter("agent.run.failed", attributes={
                    "run_id": context.run_id, "session_id": context.session_id,
                    "error_type": type(exc).__name__,
                })
            raise
        finally:
            agent.policy_capability.current_context = None
            if agent.middleware_capability is not None:
                agent.middleware_capability.current_context = None

    async def run_stream(
        self, agent: CompiledAgent, request: RunInput, context: RunContext,
        message_history: "Sequence[ModelMessage] | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming variant of :meth:`run`. Drives ``agent.pydantic_agent.iter()``
        (the pydantic-ai graph) and yields the dict-event shape the CLI REPL
        consumes as events arrive:

        * ``{"type": "text", "text": <delta>}`` -- incremental answer text.
        * ``{"type": "tool", "name": <tool>, "phase": "start"|"end", "ok": <bool|None>}``
          -- a tool call beginning / finishing (``ok`` set on ``end``).

        Unlike ``run_stream(output_type=str)`` -- which treats the first text
        output as the final result and neither continues the tool loop nor
        exposes tool events -- ``iter()`` runs every tool turn and surfaces
        progress, so a turn that consults tools streams instead of appearing
        to hang.

        Lifecycle mirrors :meth:`run` / :meth:`_run_lifecycle` step for step:
        RunRecord PENDING -> RUNNING; capabilities' current_context set;
        RunStarted event; middleware ``before_run``; prompt built from session
        history with the same Memory + Knowledge injection; SessionMessage
        append; checkpoint save; SUCCEEDED transition + ``after_run`` +
        RunCompleted; on exception FAILED transition + ``on_error`` + RunFailed
        and re-raise; current_context cleared in ``finally``.

        **Deliberate duplication.** ``run_stream`` is an ``async def``
        generator (it ``yield``s mid-lifecycle), so it cannot reuse
        :meth:`_run_lifecycle`'s non-yielding body without contortion. The
        setup/teardown is duplicated (it is short) and the duplication is
        documented here rather than factored into a shared helper, to avoid
        any behavior change to :meth:`run` -- existing ``test_runner.py`` must
        pass unchanged. Two ``run()`` features are intentionally absent from
        the streaming path: (1) observability spans (Phase 6 wraps a single
        awaited model call; streaming spans are a separate concern) and
        (2) metrics counters/histograms. ``self._observability`` and
        ``self._metrics`` are therefore not consulted here.

        **Resume (Task 8).** When ``message_history`` is provided (the
        ``Runtime.resume`` path), the new-run setup is skipped entirely --
        no ``create()``, no ``PENDING -> RUNNING`` transition, no
        ``RunStarted`` event, no ``before_run`` middleware (all already done
        in the initial run). The history/memory/knowledge prompt-build block
        is also skipped (the prompt is already baked into the checkpointed
        message history). Instead, ``iter()`` is driven with
        ``message_history=<deserialized>``, which lets the pydantic-ai graph
        resume from the checkpointed state: pending tool calls execute (the
        ToolExecutor's resume gate recognizes the APPROVED request), the model
        is called again with the full history, and the run completes normally.
        The ``running_version`` for terminal transitions (SUCCEEDED / FAILED /
        WAITING_APPROVAL) is read from the store at entry so it matches the
        version left by ``Runtime.resume``'s ``WAITING_APPROVAL -> RUNNING``
        transition."""
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

        if message_history is None:
            # New run: create record, transition PENDING -> RUNNING.
            now = datetime.now(timezone.utc)
            record = RunRecord(
                id=context.run_id, root_run_id=context.root_run_id, parent_run_id=context.parent_run_id,
                session_id=context.session_id, runnable_id=context.runnable_id, runnable_type=context.runnable_type,
                status=RunStatus.PENDING, input=request, result=None, error=None, version=1,
                created_at=now, started_at=None, finished_at=None,
            )
            await self._run_store.create(record)
            await self._run_store.transition(context.run_id, RunStatus.RUNNING, expected_version=1)
            running_version = 2
        else:
            # Resume: Runtime.resume already transitioned WAITING_APPROVAL ->
            # RUNNING; capture the current version for the final transition.
            current = await self._run_store.get(context.run_id)
            running_version = current.version

        tool_context = ToolContext(run_id=context.run_id, session_id=context.session_id)
        agent.policy_capability.current_context = tool_context
        if agent.middleware_capability is not None:
            agent.middleware_capability.current_context = tool_context

        try:
            if message_history is None:
                await self._event_store.append(self._envelope(context, sequence=1, payload=RunStarted(
                    run_id=context.run_id, runnable_id=context.runnable_id)))

                if self._middleware_pipeline is not None:
                    await self._middleware_pipeline.run_before_run(context)

            prior_messages = await self._session_store.list_messages(context.session_id)

            # Resume path (Task 8): when message_history is provided, the
            # prompt is already baked into the checkpointed history (the
            # serialize_messages round-trip) -- skip the history/memory/
            # knowledge prompt-build block entirely and drive iter() with
            # message_history below instead of a freshly-built user_prompt.
            if message_history is None:
                history_text = "\n".join(str(m.content) for m in prior_messages)
                prompt = f"{history_text}\n{request.prompt}" if history_text else request.prompt

                # Memory + Knowledge injection -- mirrors run() so the streaming
                # path sees the same prompt context as the one-shot path.
                if self._memory_store is not None:
                    from ..knowledge.context import format_memory
                    owner = context.user_id or context.tenant_id or context.session_id
                    memories = await self._memory_store.search(
                        request.prompt, owner_id=owner, limit=5,
                    )
                    section = format_memory(memories)
                    if section:
                        prompt = f"{section}\n{prompt}"
                if self._retriever is not None:
                    from ..knowledge.context import KnowledgeContext
                    docs = await self._retriever.search(request.prompt, limit=5)
                    section = KnowledgeContext(documents=docs).format()
                    if section:
                        prompt = f"{section}\n{prompt}"

            # Drive the graph with iter() and stream incremental events.
            # Pattern mirrors the legacy LlmAgent.stream() (agent.py): each
            # model-request node is streamed for text deltas, each call-tools
            # node is streamed for tool-call/result events.
            #
            # GAP-08 streaming-timeout note: ModelPolicy.timeout_seconds is
            # enforced on the one-shot ``run()`` path above. ``iter()`` returns
            # an async generator that yields across await points managed by the
            # caller, so a single ``asyncio.wait_for`` cannot wrap it without
            # restructuring this yield-based loop; streaming timeout enforcement
            # is deferred (per-node wrapping would need a per-request budget).
            accumulated_text = ""
            if message_history is not None:
                run_iter = agent.pydantic_agent.iter(message_history=message_history)
            else:
                run_iter = agent.pydantic_agent.iter(prompt)
            async with run_iter as run:
                try:
                    async for node in run:
                        if PydanticAgent.is_model_request_node(node):
                            async with node.stream(run.ctx) as request_stream:
                                async for event in request_stream:
                                    text = None
                                    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                                        text = event.part.content
                                    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                                        text = event.delta.content_delta
                                    if text:
                                        accumulated_text += text
                                        yield {"type": "text", "text": text}
                        elif PydanticAgent.is_call_tools_node(node):
                            async with node.stream(run.ctx) as tool_stream:
                                async for event in tool_stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        yield {"type": "tool", "name": event.part.tool_name,
                                               "phase": "start", "ok": None}
                                    elif isinstance(event, FunctionToolResultEvent):
                                        yield {"type": "tool", "name": event.part.tool_name,
                                               "phase": "end",
                                               "ok": isinstance(event.part, ToolReturnPart)}
                except RunPaused as paused:
                    # Pause path (Task 6 -- canonical pause surface): save a
                    # real checkpoint of the partial message history, transition
                    # to WAITING_APPROVAL, emit a RunPaused event, yield the
                    # pause signal to the caller, and return cleanly (do NOT
                    # re-raise). The handler is INSIDE the ``async with ... as
                    # run:`` block so ``run`` is bound and ``run.all_messages()``
                    # works. The ``return`` exits the generator; the outer
                    # ``finally`` still clears capability current_context.
                    messages = run.all_messages()
                    await self._checkpoint_store.save(RunCheckpoint(
                        id=str(uuid.uuid4()), run_id=context.run_id, sequence=1,
                        format="pydantic-ai-v1", schema_version=1,
                        payload=serialize_messages(messages),
                        created_at=datetime.now(timezone.utc),
                        metadata={"approval_id": paused.approval_id},
                    ))
                    try:
                        await self._run_store.transition(
                            context.run_id, RunStatus.WAITING_APPROVAL,
                            expected_version=running_version)
                    except Exception:
                        pass
                    try:
                        await self._event_store.append(self._envelope(
                            context, sequence=2,
                            payload=RunPausedEvent(
                                run_id=context.run_id,
                                reason=f"approval required: {paused.approval_id}")))
                    except Exception:
                        pass
                    yield {"type": "paused", "run_id": paused.run_id,
                           "approval_id": paused.approval_id}
                    return
                result = run.result

            # Resolve the final output: prefer result.output when it is text,
            # otherwise fall back to the accumulated streamed text (covers
            # non-text output_schema where the streamed deltas ARE the answer).
            if result is not None and isinstance(result.output, str):
                output = result.output
            else:
                output = accumulated_text

            await self._session_store.append_messages(context.session_id, (
                SessionMessage(
                    id=f"{context.run_id}-response", session_id=context.session_id,
                    sequence=len(prior_messages) + 1, role=MessageRole.ASSISTANT, content=str(output),
                    run_id=context.run_id, created_at=datetime.now(timezone.utc),
                ),
            ))

            await self._checkpoint_store.save(RunCheckpoint(
                id=str(uuid.uuid4()), run_id=context.run_id, sequence=1,
                format="pydantic-ai-v1", schema_version=1,
                payload=serialize_messages(run.all_messages()),
                created_at=datetime.now(timezone.utc),
            ))

            run_result = RunResult(output=output)
            await self._run_store.transition(
                context.run_id, RunStatus.SUCCEEDED,
                expected_version=running_version, result=run_result)

            if self._middleware_pipeline is not None:
                await self._middleware_pipeline.run_after_run(context, run_result)

            # Best-effort RunCompleted: on the resume path, sequence=2 may
            # already hold the RunPaused event from the initial pause, so a
            # conflict here must not fail the run.
            try:
                await self._event_store.append(self._envelope(context, sequence=2, payload=RunCompleted(
                    run_id=context.run_id)))
            except Exception:
                pass
        except asyncio.CancelledError:
            # GAP-16: in-flight cancel path (mirrors _run_lifecycle). When the
            # caller cancels the asyncio.Task driving this streaming generator,
            # CancelledError surfaces at the current await point inside iter()
            # / node.stream() / a tool call. Catch it BEFORE the generic
            # ``except Exception`` (CancelledError is a BaseException, not an
            # Exception), best-effort transition the RunRecord to CANCELLED
            # (using ``running_version`` captured at entry -- the version the
            # SUCCEEDED / FAILED transitions would also use), then re-raise so
            # the asyncio machinery observes the cancellation. The trailing
            # ``finally`` still clears capability current_context.
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.CANCELLED,
                    expected_version=running_version,
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            error_info = RunErrorInfo(error_type=type(exc).__name__, message=str(exc))
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.FAILED,
                    expected_version=running_version, error=error_info)
            except Exception:
                pass
            if self._middleware_pipeline is not None:
                await self._middleware_pipeline.run_on_error(context, exc)
            try:
                await self._event_store.append(self._envelope(context, sequence=2, payload=RunFailed(
                    run_id=context.run_id, error_type=type(exc).__name__, message=str(exc))))
            except Exception:
                pass
            raise
        finally:
            agent.policy_capability.current_context = None
            if agent.middleware_capability is not None:
                agent.middleware_capability.current_context = None

    def _envelope(self, context: RunContext, *, sequence: int, payload) -> EventEnvelope:
        return EventEnvelope(
            event_id=f"{context.run_id}-{sequence}", sequence=sequence,
            occurred_at=datetime.now(timezone.utc), run_id=context.run_id,
            root_run_id=context.root_run_id, parent_run_id=context.parent_run_id,
            session_id=context.session_id, runnable_id=context.runnable_id, payload=payload)
