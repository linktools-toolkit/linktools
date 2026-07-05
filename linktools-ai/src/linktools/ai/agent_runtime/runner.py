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

import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..events.envelope import EventEnvelope
from ..events.payloads import RunCompleted, RunFailed, RunStarted
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
from .models import CompiledAgent

if TYPE_CHECKING:
    from ..knowledge.retriever import Retriever
    from ..memory_runtime.store import MemoryStore
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
            if observability is not None:
                async with use_span(observability, "agent.model"):
                    run_result = await agent.pydantic_agent.run(prompt)
            else:
                run_result = await agent.pydantic_agent.run(prompt)
            output = run_result.output

            await self._session_store.append_messages(context.session_id, (
                SessionMessage(
                    id=f"{context.run_id}-response", session_id=context.session_id,
                    sequence=len(prior_messages) + 1, role=MessageRole.ASSISTANT, content=str(output),
                    run_id=context.run_id, created_at=datetime.now(timezone.utc),
                ),
            ))

            await self._checkpoint_store.save(RunCheckpoint(
                id=str(uuid.uuid4()), run_id=context.run_id, sequence=1,
                format="pydantic-ai-v1", schema_version=1, payload=b"",
                created_at=datetime.now(timezone.utc),
            ))

            result = RunResult(output=output)
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

    def _envelope(self, context: RunContext, *, sequence: int, payload) -> EventEnvelope:
        return EventEnvelope(
            event_id=f"{context.run_id}-{sequence}", sequence=sequence,
            occurred_at=datetime.now(timezone.utc), run_id=context.run_id,
            root_run_id=context.root_run_id, parent_run_id=context.parent_run_id,
            session_id=context.session_id, runnable_id=context.runnable_id, payload=payload)
