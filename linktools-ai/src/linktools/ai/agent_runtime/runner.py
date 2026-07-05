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
below knowledge), knowledge second (so it lands on top)."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..events.envelope import EventEnvelope
from ..events.payloads import RunCompleted, RunFailed, RunStarted
from ..events.store import EventStore
from ..middleware.pipeline import MiddlewarePipeline
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


class AgentRunner:
    def __init__(self, *, run_store: RunStore, session_store: SessionStore,
                 event_store: EventStore, checkpoint_store: CheckpointStore,
                 middleware_pipeline: "MiddlewarePipeline | None" = None,
                 memory_store: "MemoryStore | None" = None,
                 retriever: "Retriever | None" = None) -> None:
        self._run_store = run_store
        self._session_store = session_store
        self._event_store = event_store
        self._checkpoint_store = checkpoint_store
        self._middleware_pipeline = middleware_pipeline
        self._memory_store = memory_store
        self._retriever = retriever

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
