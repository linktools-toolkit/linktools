#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared test double for RunDispatcher used by the swarm strategy/engine tests.

Implements the two-step open_child + dispatch Protocol the way
CoordinatorRunDispatcher does, but drives the agent through the Store-free
``AgentEngine.execute_pure`` (the engine's legacy full-lifecycle ``run()`` is
gone) so strategy tests can exercise real FunctionModel-backed workers without
standing up a full RunCoordinator + CoordinatorRunDispatcher.

open_child allocates the child run id + scratch session id purely (no store
writes); dispatch creates the scratch SessionRecord, builds the child RunContext
from the handle's lineage, persists the child RunRecord via the shared
``create_and_start_run`` helper, runs the compiled agent through
``execute_pure``, and interprets the discriminated-union outcome exactly like
RunCoordinator._commit_outcome. The double holds the run_store + session_store
references so tests can inspect child-run / scratch-session side effects through
``.run_store`` / ``.session_store`` (the strategy no longer carries these on its
context)."""

import asyncio
import uuid
from datetime import datetime, timezone

from linktools.ai.execution.context import RunContext
from linktools.ai.execution.dispatch import ChildRunHandle
from linktools.ai.execution.run import RunnableType
from linktools.ai.execution.preparation import RunPreparationCoordinator
from linktools.ai.execution.session import SessionRecord, SessionStatus


class StrategyTestDispatcher:
    """A RunDispatcher test double that runs workers via a real AgentEngine."""

    def __init__(self, engine, *, session_store, run_store, run_definitions) -> None:
        self._engine = engine
        self._sessions = session_store
        self._run_store = run_store
        self._preparation = RunPreparationCoordinator(run_definitions)
        # Exposed for tests that assert on child-run / session side effects.
        self.run_store = run_store
        self.session_store = session_store

    async def open_child(self, parent_context, session_policy, metadata):
        child_run_id = str(uuid.uuid4())
        if session_policy.kind == "shared":
            session_id = parent_context.session_id
            needs_create = False
        elif session_policy.session_id_format:
            session_id = session_policy.session_id_format.format(
                child_run_id=child_run_id, **metadata
            )
            needs_create = True
        else:
            session_id = str(uuid.uuid4())
            needs_create = True
        root = (
            parent_context.root_run_id
            or parent_context.run_id
            or child_run_id
        )
        return ChildRunHandle(
            run_id=child_run_id,
            session_id=session_id,
            root_run_id=root,
            parent_run_id=parent_context.run_id,
            parent_session_id=session_policy.parent_session_id,
            user_id=parent_context.user_id,
            tenant_id=parent_context.tenant_id,
            workspace=getattr(parent_context, "workspace", None),
            session_needs_create=needs_create,
        )

    async def dispatch(self, request):
        handle = request.handle
        if handle.session_needs_create:
            now = datetime.now(timezone.utc)
            await self._sessions.create(
                SessionRecord(
                    id=handle.session_id,
                    parent_id=handle.parent_session_id,
                    user_id=handle.user_id,
                    tenant_id=handle.tenant_id,
                    status=SessionStatus.ACTIVE,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
        context = RunContext(
            run_id=handle.run_id,
            root_run_id=handle.root_run_id,
            parent_run_id=handle.parent_run_id,
            session_id=handle.session_id,
            runnable_id=(
                request.metadata.get("runnable_id")
                or request.compiled_agent.spec.id
            ),
            runnable_type=RunnableType.AGENT,
            user_id=handle.user_id,
            tenant_id=handle.tenant_id,
            workspace=handle.workspace,
        )
        # Mirror RunCoordinator.dispatch_child: a worker run gets a resumable
        # snapshot so Runtime.resume(child_run_id) can restore its spec.
        await self._preparation.prepare_agent_run(
            spec=request.compiled_agent.spec, context=context
        )
        # The engine's legacy full-lifecycle ``run()`` is gone (FS-29: the
        # engine owns only the Store-free model/tool loop). Mirror what
        # RunCoordinator.dispatch_child + _commit_outcome do for the
        # Completed/Failed paths a strategy test actually exercises: create the
        # child RunRecord via the shared lifecycle helper, drive the Store-free
        # ``execute_pure``, transition the RunRecord (mark_completed /
        # mark_failed), then interpret the outcome -- Completed returns the
        # result, Failed re-raises. NOTE: this is a partial mirror, not full
        # parity -- Paused raises RunPaused WITHOUT the WAITING_APPROVAL
        # transition and Cancelled raises CancelledError WITHOUT the CANCELLED
        # transition (the real Coordinator persists both via commit_coordinator).
        # No strategy test asserts paused/cancelled child-run status today
        # (they assert SUCCEEDED at test_strategy.py), so the simplification is
        # safe; if such a test is added, wire mark_cancelled / a pause
        # transition in here for parity.
        from linktools.ai.agent.models import (
            AgentCancelled,
            AgentCompleted,
            AgentFailed,
            AgentInput,
            AgentPaused,
        )
        from linktools.ai.errors import RunPaused
        from linktools.ai.execution.cancellation import CancellationToken
        from linktools.ai.execution.lifecycle import (
            create_and_start_run,
            mark_completed,
            mark_failed,
        )
        from linktools.ai.execution.live_events import (
            NullRunLiveEventSink,
            NullSecurityEventSink,
        )

        running = await create_and_start_run(
            self._run_store, context=context, request=request.input
        )
        outcome = await self._engine.execute_pure(
            request.compiled_agent,
            AgentInput(prompt=request.input.prompt),
            context,
            cancellation=CancellationToken(),
            live_events=NullRunLiveEventSink(),
            security_events=NullSecurityEventSink(),
        )
        if isinstance(outcome, AgentCompleted):
            await mark_completed(
                self._run_store,
                context.run_id,
                expected_version=running.version,
                result=outcome.result,
            )
            return outcome.result
        if isinstance(outcome, AgentPaused):
            raise RunPaused(run_id=context.run_id)
        if isinstance(outcome, AgentFailed):
            await mark_failed(
                self._run_store,
                context.run_id,
                expected_version=running.version,
                error=outcome.error,
            )
            raise RuntimeError(
                f"worker failed ({outcome.error.error_type}): "
                f"{outcome.error.message}"
            )
        raise asyncio.CancelledError()
