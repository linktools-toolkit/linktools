#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime: the top-level integration surface (spec section 5; deviation #3).
Runtime.build() assembles Storage + AgentCompiler + AgentRunner + SwarmRunner +
ModelRouter; Runtime.run(spec, prompt) compiles the spec, resolves (or creates)
a Session, mints a RunContext, and delegates to AgentRunner (AgentSpec) or
SwarmRunner (SwarmSpec)."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Mapping

# AsyncIterator is a typing-only alias used to annotate the streaming
# generator below; the function itself is an ``async def`` that ``yield``s.
if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from .agent.compiler import AgentCompiler
from .agent.runner import AgentRunner
from .agent.spec import AgentSpec
from .errors import SessionError, SwarmError
from .execution.protocols import ExecutionBackend
from .middleware.pipeline import MiddlewarePipeline
from .model.router import ModelRouter
from .run.context import RunContext
from .run.controller import RunController
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
                 model_router: ModelRouter,
                 run_controller: "RunController | None" = None) -> None:
        self.storage = storage
        self.compiler = compiler
        self.runner = runner
        self.swarm_runner = swarm_runner
        self.model_router = model_router
        # Phase 3A (§7.1): tracks in-flight asyncio.Tasks so cancel(run_id)
        # can actually stop a running task, not just flip the DB status. When
        # None, cancel() falls back to the pre-Phase-3A store-only path
        # (best-effort direct -> CANCELLED). Always wired by build() so the
        # default-constructed Runtime gains real cancellation for free.
        self.run_controller = run_controller

    @classmethod
    def build(cls, *, storage: Storage,
              model_router: "ModelRouter | None" = None,
              middleware_pipeline: "MiddlewarePipeline | None" = None,
              retriever: "Retriever | None" = None,
              execution: "ExecutionBackend | None" = None,
              tool_executor: "ToolExecutor | None" = None,
              pause_on_approval: bool = False) -> "Runtime":
        """Assemble a Runtime from optional sub-components.

        ``execution`` (Package 8, actionable-fix-spec §11): pass a pre-built
        ``ExecutionBackend`` (e.g. ``LocalExecutionBackend(runtime_dir=...)``)
        to give compiled agents builtin file/terminal tools. ``None``
        (default) means conversational-only -- no builtin tools exposed.
        Runtime.build() does not construct filesystem backends from a bare
        path itself; the caller owns that decision (which backend
        implementation, where it's rooted). This replaces the previous
        ``workdir: Path`` parameter, which implicitly assumed
        ``LocalExecutionBackend`` was the only possible choice.

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
        # §17 (review-doc) / Package 8: the ExecutionBackend is never passed
        # to AgentCompiler. AgentCompiler is stateless (no filesystem
        # surface); the builtin file/terminal tools are constructed at
        # execution time from this backend inside AgentRunner.execute() and
        # passed via ``agent.iter(prompt, toolsets=[...])``. ``execution is
        # None`` means no backend -- runs expose no builtin tools
        # (conversational-only).
        compiler = AgentCompiler(
            model_router=router,
            middleware_pipeline=middleware_pipeline,
            tool_executor=resolved_executor,
        )
        # Memory is on-by-default (storage.memories is always populated by the
        # facade); Knowledge is opt-in via the ``retriever`` argument (None ->
        # no retrieval, no prompt section). Both are forwarded to SwarmRunner
        # so swarm worker Runs see the same injection as top-level Runs.
        #
        # §10.2 atomic pause: when the storage can promise cross-store
        # transactions (SqlAlchemy), thread its ``transaction()`` factory into
        # AgentRunner so the RunPaused handler wraps checkpoint + transition +
        # event in one UnitOfWork. FileStorage cannot (capabilities flag False)
        # -> None -> the pause path keeps its non-atomic best-effort shape.
        uow_factory = (
            storage.transaction
            if storage.capabilities.cross_store_transactions else None
        )
        # Phase 3A (§7): one RunController per Runtime. AgentRunner.execute()
        # registers its driving asyncio.Task + a fresh CancellationToken here
        # so Runtime.cancel(run_id) can actually stop the run (token check +
        # task.cancel()). Always wired -- there is no downside since the
        # runner's token checks are no-ops unless a controller-issued cancel
        # sets the token.
        run_controller = RunController()
        runner = AgentRunner(
            run_store=storage.runs, session_store=storage.sessions,
            event_store=storage.events, checkpoint_store=storage.checkpoints,
            middleware_pipeline=middleware_pipeline,
            memory_store=storage.memories,
            retriever=retriever,
            uow_factory=uow_factory,
            run_controller=run_controller,
            execution=execution,
            # P0-6/G1: always wired so the pause path can persist the
            # ApprovalRequest in File mode too (SqlAlchemy mode reaches it via
            # tx.approvals inside the UoW instead).
            approval_store=storage.approvals,
        )
        swarm_runner = SwarmRunner(
            swarm_store=storage.swarms,
            run_store=storage.runs,
            session_store=storage.sessions,
            event_store=storage.events,
            compiler=compiler,
            # Package 1 (actionable-fix-spec §4): SwarmRunner reuses the SAME
            # AgentRunner Runtime just assembled for top-level Agent runs --
            # it does not build its own. Swarm worker Runs therefore inherit
            # identical Tool/Policy/Middleware/UoW/ExecutionBackend/Approval
            # semantics, and (Package 2, §5) the same RunController, so
            # cancel() can actually stop an in-flight child Run.
            agent_runner=runner,
            run_controller=run_controller,
        )
        return cls(
            storage=storage, compiler=compiler, runner=runner,
            swarm_runner=swarm_runner, model_router=router,
            run_controller=run_controller,
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

    async def cancel(self, run_id: str) -> None:
        """Cancel an in-flight Run (review doc §7).

        Two paths, depending on whether a live asyncio.Task is registered with
        the RunController:

        * **In-flight task registered** -- the run is actually being driven by
          AgentRunner.execute(). Transition the store to CANCELLING (§6.1:
          distinguishes "cancel requested" from "actually cancelled"), then
          call ``run_controller.cancel(run_id)`` which (a) sets the
          CancellationToken so the runner's next execution-point check raises
          CancelledError, and (b) calls ``task.cancel()`` so any hanging await
          inside the model call also unblocks. The runner's CancelledError
          handler then transitions CANCELLING -> CANCELLED via the version
          captured from the preceding step (§6.3).

          If the record is already CANCELLING (a previous cancel() call won
          the race), skip straight to ``run_controller.cancel(run_id)`` --
          re-transitioning CANCELLING -> CANCELLING is not a legal edge in
          ALLOWED_RUN_TRANSITIONS. This makes repeated cancel() calls on the
          same in-flight run idempotent instead of raising
          InvalidRunTransitionError on the second call. A concurrent cancel()
          can also race the CANCELLING transition itself: if it loses to
          another cancel() between the terminal-status check and the
          transition call, the ``transition()`` raises RunConflictError; on
          that specific conflict, re-read the record and, if it is now
          CANCELLING (the other caller's transition landed first), fall
          through to the same idempotent ``run_controller.cancel(run_id)``
          path rather than propagating the conflict. Any other exception
          (including a fresh terminal or unexpected status) is not ours to
          interpret and is re-raised as-is.

        * **No in-flight task** (stale record from a crashed worker, a test
          seed, or any path where the runner is not driving execute()) --
          there is nothing to actually stop, so the store goes directly to
          CANCELLED. This also covers a stale CANCELLING record with no
          registered controller entry (e.g. after a process restart). This is
          the pre-Phase-3A behavior and preserves every existing
          seeded-cancel test.

        Idempotent: a Run already in a terminal status (SUCCEEDED / FAILED /
        CANCELLED) is a no-op. Raises :class:`RunNotFoundError` when the run
        does not exist."""
        from .errors import RunConflictError, RunNotFoundError
        from .run.models import RunStatus

        record = await self.storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status in (
            RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
        ):
            return  # already terminal -- no-op

        # If a live task is registered, go through CANCELLING + signal the
        # controller; the runner finishes the transition to CANCELLED via its
        # CancelledError handler. Otherwise (no task / no controller) go
        # directly to CANCELLED -- there is nothing to actually stop.
        in_flight = (
            self.run_controller is not None
            and self.run_controller.get_token(run_id) is not None
        )
        if in_flight:
            if record.status == RunStatus.CANCELLING:
                # A previous cancel() already moved this in-flight run to
                # CANCELLING -- re-signal the controller (idempotent) instead
                # of attempting the illegal CANCELLING -> CANCELLING edge.
                await self.run_controller.cancel(run_id)
                return

            try:
                await self.storage.runs.transition(
                    run_id, RunStatus.CANCELLING,
                    expected_version=record.version,
                )
            except RunConflictError:
                fresh = await self.storage.runs.get(run_id)
                if fresh is None:
                    raise RunNotFoundError(f"run not found: {run_id}")
                if fresh.status in (
                    RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED,
                ):
                    return
                if fresh.status == RunStatus.CANCELLING:
                    await self.run_controller.cancel(run_id)
                    return
                raise

            await self.run_controller.cancel(run_id)
        else:
            await self.storage.runs.transition(
                run_id, RunStatus.CANCELLED, expected_version=record.version,
            )

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
