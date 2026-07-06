#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AgentRunner: owns the per-invocation lifecycle -- Run state transitions,
Session history load/append, runner-driven Middleware hooks (before_run/after_run/
on_error), Event publication, Checkpoint save. The 4 pydantic-ai-intercepted hooks
fire via MiddlewareCapability, enabled by passing deps=AgentDependencies(...) to
agent.pydantic_agent.iter() -- the per-Run ToolContext travels through
pydantic-ai's dependency injection (ctx.deps), not a mutable capability field.

Phase 2A of the review-doc refactoring (spec §5): one execute() async generator
is the SINGLE lifecycle. run() and run_stream() both delegate to it.

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

Optional Memory + Knowledge injection (Phase 5): when ``memory_store`` and/or
``retriever`` are wired, ``execute()`` queries them with the user prompt and
prepends ``## Memory`` / ``## Knowledge`` sections to the prompt sent to the
model. Both default to None, so existing callers see no change. Final prompt
order (when both are set and non-empty): ``## Knowledge`` on top, then
``## Memory``, then session history, then the user prompt.

Optional Observability (Phase 6): when ``observability`` is wired, ``execute()``
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
from typing import TYPE_CHECKING

from ..errors import ModelPolicyExceededError, ModelRoutingError, RunPaused
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
from ..run.models import (
    RunCheckpoint,
    RunErrorInfo,
    RunInput,
    RunRecord,
    RunResult,
    RunStatus,
)
from ..run.store import RunStore
from ..session.models import MessageRole, SessionMessage
from ..session.store import SessionStore
from .checkpoint_io import serialize_messages
from .dependencies import AgentDependencies
from .models import CompiledAgent

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence
    from contextlib import AbstractAsyncContextManager

    from pydantic_ai.messages import ModelMessage

    from ..knowledge.retriever import Retriever
    from ..memory.store import MemoryStore
    from ..observability.metrics import ObservabilityMetrics
    from ..observability.tracing import ObservabilitySink
    from ..storage.facade import _UnitOfWork


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
        # atomically (review doc §10.2). None for FileStorage -- File cannot
        # promise cross-store transactions, so the pause path keeps its
        # best-effort non-atomic shape (§10.3).
        self._uow_factory = uow_factory

    def _span(self, name: str, *, attrs: "dict | None" = None):
        """Return an async context manager that opens an observability span when
        a sink is wired, or a no-op when it is not. Keeps the lifecycle body
        single-shape regardless of observability being configured."""
        if self._observability is None:
            return _noop_span()
        return use_span(self._observability, name, attributes=attrs or {})

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
        # Per review doc §6.3, the terminal transitions reuse the version returned
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

                # -- Prompt build. Resume path skips this entirely -- the
                # prompt is baked into the checkpointed message_history.
                prompt: "str | None" = None
                if message_history is None:
                    history_text = "\n".join(str(m.content) for m in prior_messages)
                    prompt = (f"{history_text}\n{request.prompt}"
                              if history_text else request.prompt)

                    # Memory + Knowledge injection (Phase 5). Each block is
                    # optional and only fires when its dependency is wired AND
                    # yields a non-empty section, so the default-None path is a
                    # no-op. Memory injected first, then knowledge -- both
                    # prepend, so the final order top-to-bottom is: Knowledge,
                    # Memory, history, user prompt. Owner resolution prefers
                    # user_id, then tenant_id, then session_id.
                    if self._memory_store is not None:
                        from ..knowledge.context import format_memory
                        owner = (context.user_id or context.tenant_id
                                 or context.session_id)
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

                # -- Model call: agent.pydantic_agent.iter() drives the graph.
                # The per-Run ToolContext travels to capabilities via pydantic-ai
                # DI: ``deps=`` becomes ``ctx.deps.tool_context`` inside every
                # capability hook (Phase 1 refactoring -- safe concurrent reuse
                # of one CompiledAgent across many Runs).
                tool_context = ToolContext(
                    run_id=context.run_id, session_id=context.session_id,
                    tool_call_id=None)
                deps = AgentDependencies(tool_context=tool_context)
                # GAP-08: ModelPolicy.timeout_seconds is enforced by wrapping
                # each graph step (``run.__anext__()``) in asyncio.wait_for with
                # the REMAINING budget. The model call happens at this await
                # point (for stream-less models / the non-streaming path), so
                # wait_for can interrupt a hanging model call. timeout_seconds
                # left at None reproduces the pre-GAP-08 path (no wait_for).
                timeout = agent.spec.model.timeout_seconds

                accumulated_text = ""
                result = None
                if message_history is not None:
                    run_iter = agent.pydantic_agent.iter(
                        message_history=message_history, deps=deps)
                else:
                    run_iter = agent.pydantic_agent.iter(prompt, deps=deps)

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
                                # Pause path (canonical surface): save a real
                                # checkpoint of the partial message history,
                                # transition to WAITING_APPROVAL, emit a
                                # RunPaused event -- all INSIDE the iter()
                                # context so ``run.all_messages()`` works. The
                                # paused-event yield itself is deferred to
                                # AFTER all context managers exit (see the
                                # comment on ``paused_signal`` above) so a
                                # non-streaming consumer can abandon the
                                # generator without triggering a cross-task
                                # cancel-scope exit.
                                #
                                # §10.2 atomicity: when a UnitOfWork factory is
                                # wired (SqlAlchemy), checkpoint + transition +
                                # event share ONE transaction -- they commit
                                # together on clean exit or rollback together
                                # if ANY of them raises (which then propagates
                                # to the outer generic-except -> FAILED). When
                                # no factory is wired (File), cross-store
                                # transactions are impossible, so the path
                                # keeps its non-atomic best-effort shape
                                # (§10.3): checkpoint + transition still
                                # propagate (§3.3 forbids leaving them partial),
                                # but the RunPaused event append is best-effort.
                                checkpoint = RunCheckpoint(
                                    id=str(uuid.uuid4()), run_id=context.run_id,
                                    sequence=1, format="pydantic-ai-v1",
                                    schema_version=1,
                                    payload=serialize_messages(run.all_messages()),
                                    created_at=datetime.now(timezone.utc),
                                    metadata={"approval_id": paused.approval_id},
                                )
                                paused_payload = RunPausedEvent(
                                    run_id=context.run_id,
                                    reason=f"approval required: {paused.approval_id}",
                                )
                                if self._uow_factory is not None:
                                    # Atomic (SqlAlchemy): all three writes bind
                                    # to one AsyncSession + one transaction. Any
                                    # failure rolls back checkpoint + transition
                                    # AND propagates to the outer generic-except
                                    # handler so the Run ends up FAILED rather
                                    # than left in a half-paused state.
                                    async with self._uow_factory() as tx:
                                        await tx.checkpoints.save(checkpoint)
                                        # §3.3 + §6.3: WAITING_APPROVAL
                                        # transition MUST propagate. If it
                                        # fails the run cannot be paused --
                                        # rolling back + propagating avoids
                                        # leaving the checkpoint saved but the
                                        # run still RUNNING (the inconsistent
                                        # state §10.2 forbids).
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
                                            payload=paused_payload,
                                        )
                                else:
                                    # File mode: non-atomic best-effort (§10.3).
                                    # Cross-store transactions are unavailable,
                                    # so checkpoint + transition propagate (§3.3
                                    # forbids masking them) but the RunPaused
                                    # event append stays best-effort -- the run
                                    # is already WAITING_APPROVAL, so a missing
                                    # event is an observability gap, not state
                                    # corruption.
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

                # If the timeout budget was exhausted, raise ModelRoutingError
                # so the outer generic-except handler records FAILED with a
                # descriptive "model timeout" message (GAP-08). This sits
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

                    # GAP-08: max_tokens enforcement. usage is read once so the
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

                    await self._session_store.append_messages(context.session_id, (
                        SessionMessage(
                            id=f"{context.run_id}-response", session_id=context.session_id,
                            sequence=len(prior_messages) + 1, role=MessageRole.ASSISTANT,
                            content=str(output), run_id=context.run_id,
                            created_at=datetime.now(timezone.utc),
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
                    # appends after it (no caller sequence to collide with --
                    # review doc §8.1). best-effort audit - non-critical path:
                    # the run is already SUCCEEDED, so a missing event is an
                    # observability gap, not state corruption (§3.3).
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
            # GAP-16: in-flight cancel path. CancelledError surfaces at the
            # current await point (model call / node.stream() / event append /
            # middleware). Caught BEFORE the generic ``except Exception``
            # (CancelledError is a BaseException since Python 3.8). Best-effort
            # transition to CANCELLED, then re-raise so the asyncio machinery
            # observes the cancellation. No capability state to clear (Phase 1
            # refactoring: deps travel via pydantic-ai DI).
            #
            # The CANCELLED transition (§3.3 "Cancel status confirm") is kept
            # best-effort ONLY because asyncio requires CancelledError to
            # propagate -- letting the transition error escape would replace
            # the cancellation with a different exception type and break the
            # cancel machinery. The warning keeps the failure visible rather
            # than silent (the run may already be terminal, e.g. a concurrent
            # Runtime.cancel beat this transition).
            try:
                await self._run_store.transition(
                    context.run_id, RunStatus.CANCELLED,
                    expected_version=running_version,
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
            # §3.3: Run status update. The FAILED transition is kept
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
            # is an observability gap, not state corruption (§3.3).
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
        # (single source of truth per review doc §3.2).
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
