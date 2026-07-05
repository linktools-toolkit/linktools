#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface (spec section 5; deviation #3).
Runtime.build() assembles Storage + AgentCompiler + AgentRunner + SwarmRunner +
ModelRouter; Runtime.run(spec, prompt) compiles the spec, resolves (or creates)
a Session, mints a RunContext, and delegates to AgentRunner (AgentSpec) or
SwarmRunner (SwarmSpec)."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

# AsyncIterator is a typing-only alias used to annotate the streaming
# generator below; the function itself is an ``async def`` that ``yield``s.
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from .agent.compiler import AgentCompiler
from .agent.runner import AgentRunner
from .agent.spec import AgentSpec
from .errors import SessionError, SwarmError
from .middleware.pipeline import MiddlewarePipeline
from .model.router import ModelRouter
from .run.context import RunContext
from .run.models import RunInput, RunnableType
from .session.models import SessionRecord, SessionStatus
from .storage.facade import Storage
from .swarm.runner import SwarmRunner
from .swarm.spec import SwarmSpec
from .tool.executor import ToolExecutor

if TYPE_CHECKING:
    from .knowledge.retriever import Retriever


class Runtime:
    def __init__(self, *, storage: Storage, compiler: AgentCompiler,
                 runner: AgentRunner, swarm_runner: SwarmRunner,
                 model_router: ModelRouter) -> None:
        self.storage = storage
        self.compiler = compiler
        self.runner = runner
        self.swarm_runner = swarm_runner
        self.model_router = model_router

    @classmethod
    def build(cls, *, storage: Storage,
              model_router: "ModelRouter | None" = None,
              middleware_pipeline: "MiddlewarePipeline | None" = None,
              workspace_root: "str | Path | None" = None,
              retriever: "Retriever | None" = None,
              workdir: "Path | None" = None,
              tool_executor: "ToolExecutor | None" = None,
              pause_on_approval: bool = False) -> "Runtime":
        """Assemble a Runtime from optional sub-components.

        ``tool_executor`` is the override path for the Phase-6 policy rules:
        when ``None`` (default) the compiler builds its own default
        ``ToolExecutor`` whose ``PolicyEngine`` carries only ``CommandRule``
        (the one rule that needs no tool metadata). To make the rich rules
        (Permission/Risk/Approval) enforce against real tool declarations,
        the caller awaits ``build_default_policy_engine(tool_registry)`` and
        passes ``ToolExecutor(policy=that_engine)`` here. ``build`` stays
        synchronous by design -- the async policy build is the caller's job,
        so the common ``Runtime.build(...)`` call site stays simple.

        ``pause_on_approval=True`` (Task 9) is the ergonomic entry to the
        pause/resume path: when set AND no explicit ``tool_executor`` was
        supplied, ``build`` constructs a pause-enabled default executor with
        the storage's approval store wired (the compiler's own default
        executor has no store, so it could only fall through to the legacy
        raise). When ``tool_executor`` is explicit the flag is informational
        -- the caller's executor already carries its own pause/store config."""
        router = model_router or ModelRouter()
        resolved_executor = tool_executor
        if tool_executor is None and pause_on_approval:
            # Option (b): Runtime.build has access to ``storage.approvals`` so
            # it wires the full pause-enabled executor in one place. The
            # default-False path leaves executor construction to the compiler
            # (byte-for-byte unchanged behavior).
            from .policy.command import CommandRule, DEFAULT_DENIED_COMMAND_PATTERNS
            from .policy.engine import PolicyEngine

            resolved_executor = ToolExecutor(
                policy=PolicyEngine(rules=(
                    CommandRule(denied_patterns=DEFAULT_DENIED_COMMAND_PATTERNS),
                )),
                approval_store=storage.approvals,
                pause_on_approval=True,
            )
        compiler = AgentCompiler(
            model_router=router,
            middleware_pipeline=middleware_pipeline,
            workdir=workdir,
            tool_executor=resolved_executor,
        )
        # Memory is on-by-default (storage.memories is always populated by the
        # facade); Knowledge is opt-in via the ``retriever`` argument (None ->
        # no retrieval, no prompt section). Both are forwarded to SwarmRunner
        # so swarm worker Runs see the same injection as top-level Runs.
        runner = AgentRunner(
            run_store=storage.runs, session_store=storage.sessions,
            event_store=storage.events, checkpoint_store=storage.checkpoints,
            middleware_pipeline=middleware_pipeline,
            memory_store=storage.memories,
            retriever=retriever,
        )
        swarm_runner = SwarmRunner(
            swarm_store=storage.swarms,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            checkpoint_store=storage.checkpoints,
            compiler=compiler,
            memory_store=storage.memories,
            retriever=retriever,
        )
        return cls(
            storage=storage, compiler=compiler, runner=runner,
            swarm_runner=swarm_runner, model_router=router,
        )

    async def run(self, spec: "AgentSpec | SwarmSpec", prompt: str, *,
                  session_id: "str | None" = None,
                  run_id: "str | None" = None,
                  user_id: "str | None" = None,
                  tenant_id: "str | None" = None,
                  agents: "Mapping[str, AgentSpec] | None" = None):
        resolved_session_id = session_id or str(uuid.uuid4())
        if session_id is not None:
            existing = await self.storage.sessions.get(session_id)
            if existing is None:
                raise SessionError(f"session not found: {session_id}")
        else:
            now = datetime.now(timezone.utc)
            await self.storage.sessions.create(SessionRecord(
                id=resolved_session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1,
                created_at=now, updated_at=now))

        resolved_run_id = run_id or str(uuid.uuid4())

        if isinstance(spec, SwarmSpec):
            if agents is None:
                raise SwarmError(
                    "agents mapping is required to run a SwarmSpec"
                )
            context = RunContext(
                run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
                session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.SWARM,
                user_id=user_id, tenant_id=tenant_id, workspace=None)
            return await self.swarm_runner.run(
                spec, RunInput(prompt=prompt), context, agents=agents,
            )

        compiled = await self.compiler.compile(spec)
        context = RunContext(
            run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
            session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.AGENT,
            user_id=user_id, tenant_id=tenant_id, workspace=None)
        return await self.runner.run(compiled, RunInput(prompt=prompt), context)

    async def run_stream(
        self, spec: "AgentSpec | SwarmSpec", prompt: str, *,
        session_id: "str | None" = None,
        run_id: "str | None" = None,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming variant of :meth:`run`. Resolves (or creates) the Session,
        mints a RunContext, and delegates to :meth:`AgentRunner.run_stream`,
        yielding the same dict-event shape (``text`` / ``tool``) the CLI REPL
        consumes.

        Session resolution mirrors :meth:`run` exactly (explicit ``session_id``
        must exist; ``None`` mints a fresh session). Only ``AgentSpec`` is
        supported -- a ``SwarmSpec`` raises :class:`SwarmError` because swarm
        streaming is not implemented."""
        resolved_session_id = session_id or str(uuid.uuid4())
        if session_id is not None:
            existing = await self.storage.sessions.get(session_id)
            if existing is None:
                raise SessionError(f"session not found: {session_id}")
        else:
            now = datetime.now(timezone.utc)
            await self.storage.sessions.create(SessionRecord(
                id=resolved_session_id, parent_id=None, status=SessionStatus.ACTIVE, version=1,
                created_at=now, updated_at=now))

        resolved_run_id = run_id or str(uuid.uuid4())

        if isinstance(spec, SwarmSpec):
            raise SwarmError("run_stream does not support SwarmSpec")

        compiled = await self.compiler.compile(spec)
        context = RunContext(
            run_id=resolved_run_id, root_run_id=resolved_run_id, parent_run_id=None,
            session_id=resolved_session_id, runnable_id=spec.id, runnable_type=RunnableType.AGENT,
            user_id=user_id, tenant_id=tenant_id, workspace=None)
        async for event in self.runner.run_stream(compiled, RunInput(prompt=prompt), context):
            yield event

    async def resume(
        self, run_id: str, spec: "AgentSpec", *,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
    ) -> "AsyncIterator[dict]":
        """Resume a paused Run (Task 8). Loads the paused RunRecord,
        deserializes its checkpoint's message history, transitions
        WAITING_APPROVAL -> RUNNING, and re-enters
        :meth:`AgentRunner.run_stream` with ``message_history=<deserialized>``
        so the pydantic-ai graph picks up from the checkpointed state: pending
        tool calls execute (the ToolExecutor's resume gate recognizes the
        now-APPROVED request), the model is called again with the full history,
        and the run completes normally.

        Yields ``{"type": "resumed", "run_id": run_id}`` first, then the same
        dict-event shape ``run_stream`` yields (``text`` / ``tool``).

        Raises :class:`RunNotFoundError` when the run or its checkpoint does not
        exist. Raises :class:`InvalidRunTransitionError` when the run is not in
        WAITING_APPROVAL status."""
        from .agent.checkpoint_io import deserialize_messages
        from .errors import InvalidRunTransitionError, RunNotFoundError
        from .run.models import RunStatus

        record = await self.storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status != RunStatus.WAITING_APPROVAL:
            raise InvalidRunTransitionError(
                f"cannot resume run in status {record.status}"
            )
        checkpoint = await self.storage.checkpoints.latest(run_id)
        if checkpoint is None:
            raise RunNotFoundError(f"no checkpoint for run: {run_id}")
        messages = deserialize_messages(checkpoint.payload)
        await self.storage.runs.transition(
            run_id, RunStatus.RUNNING, expected_version=record.version,
        )
        compiled = await self.compiler.compile(spec)
        context = RunContext(
            run_id=run_id, root_run_id=record.root_run_id,
            parent_run_id=record.parent_run_id, session_id=record.session_id,
            runnable_id=record.runnable_id, runnable_type=record.runnable_type,
            user_id=user_id, tenant_id=tenant_id, workspace=None,
        )
        yield {"type": "resumed", "run_id": run_id}
        async for event in self.runner.run_stream(
            compiled, RunInput(prompt=""), context, message_history=messages,
        ):
            yield event
