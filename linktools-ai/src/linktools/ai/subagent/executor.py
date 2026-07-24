#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentExecutor: the concrete executor that runs a resolved child AgentSpec
under a parent run.

Moved out of the Runtime composition root (it lived as a builder closure before)
so the build kernel only ASSEMBLES this executor -- it never creates sessions,
child runs, or drives execution itself (the builder constructs only). The
executor owns the subagent domain flow once assembled: child-session creation,
child-run creation, skill isolation for the child, timeout enforcement,
subagent event emission, and structured error redaction.

Dependencies: the storage (sessions / events / run definitions), the compiler
(to compile the child AgentSpec), and the run dispatcher -- a late-bound handle
resolved to the real runner once the runner exists, because the runner depends
on the capability resolver, which depends on this executor (a genuine
self-reference, confined to the single bind-once seam)."""

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from ..run.lifecycle import create_and_start_run
from ..run.preparation import RunPreparationCoordinator
from ..session.models import SessionRecord, SessionStatus
from ..storage.facade import Storage
from .models import SubagentResult


class SubagentExecutor:
    """Executes a resolved child AgentSpec under a parent run. Constructed once
    by the build kernel; ``dispatcher`` is a late-bound handle the kernel binds
    to the real runner after the runner exists."""

    def __init__(
        self,
        *,
        storage: Storage,
        compiler: "Any",
        dispatcher: "Any",
    ) -> None:
        self._storage = storage
        self._compiler = compiler
        self._dispatcher = dispatcher
        # A child agent run (subagent / extension entrypoint) gets the same
        # resumable snapshot as a top-level run: if one of its tools pauses on
        # approval, Runtime.resume(child_run_id) can restore its spec + identity.
        self._preparation = RunPreparationCoordinator(storage.run_definitions)

    async def execute(
        self,
        *,
        agent_spec: "Any",
        task: str,
        context: "dict[str, Any] | None",
        parent: "Any",
        scope: "Any | None",
        timeout_seconds: "float | None",
    ) -> SubagentResult:
        from ..run.context import RunContext
        from ..run.dispatch import RunDispatchRequest
        from ..run.models import RunnableType, RunInput

        parent_run_id = parent.run_id if parent is not None else None
        parent_session_id = parent.session_id if parent is not None else None
        child_session = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await self._storage.sessions.create(
            SessionRecord(
                id=child_session,
                parent_id=parent_session_id,
                # A subagent child session inherits its parent's principal, so a
                # worker pause/resume stays within the same ownership domain.
                user_id=parent.user_id if parent is not None else None,
                tenant_id=parent.tenant_id if parent is not None else None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        child_run = str(uuid.uuid4())
        effective_root = (
            (parent.root_run_id if parent is not None else None)
            or parent_run_id
            or child_run
        )
        run_ctx = RunContext(
            run_id=child_run,
            root_run_id=effective_root,
            parent_run_id=parent_run_id,
            session_id=child_session,
            runnable_id=agent_spec.id,
            runnable_type=RunnableType.AGENT,
            user_id=parent.user_id if parent is not None else None,
            tenant_id=parent.tenant_id if parent is not None else None,
            workspace=parent.workspace if parent is not None else None,
        )
        scope_dict = None
        if scope is not None:
            scope_dict = {
                "extension_id": scope.extension_id,
                "extension_kind": scope.extension_kind,
            }

        async def _drive():
            # A child run starts OUTSIDE any skill: clear the parent's active
            # skill for the duration of the child so a subagent cannot address
            # the parent's skill via call_subagent(instruction_path=...) (skill
            # isolation). Imported lazily to avoid a build-time import cycle.
            from ..skill.private import reset_active_skill, set_active_skill

            skill_token = set_active_skill(None)
            try:
                await self._preparation.prepare_agent_run(
                    spec=agent_spec, context=run_ctx
                )
                compiled = await self._compiler.compile(agent_spec)
                # Create + start the child RunRecord here (WP9 step 5 --
                # spec 12.8: "SubagentExecutor creates and executes the child
                # Run" -- the same create_and_start_run RunCoordinator uses
                # for a top-level Run) so the dispatcher's own get-or-create
                # fallback (AgentEngine.execute(), for direct-engine callers
                # that skip this ownership entirely) is never reached here.
                await create_and_start_run(
                    self._storage.runs,
                    context=run_ctx,
                    request=RunInput(prompt=task),
                )
                return await self._dispatcher.dispatch(
                    RunDispatchRequest(
                        agent=compiled, input=RunInput(prompt=task), context=run_ctx
                    )
                )
            finally:
                reset_active_skill(skill_token)

        from ..events.payloads import (
            SubagentCompleted,
            SubagentErrored,
            SubagentStarted,
        )

        async def _evt(payload):
            from ..events.context import EventStreamContext, append_event

            await append_event(
                self._storage.events,
                EventStreamContext(
                    stream_id=child_run,
                    run_id=child_run,
                    root_run_id=effective_root,
                    parent_run_id=parent_run_id,
                    session_id=child_session,
                    runnable_id=agent_spec.id,
                ),
                payload,
            )

        from ..subagent.runner import _CURRENT_DEPTH

        token = _CURRENT_DEPTH.set(_CURRENT_DEPTH.get() + 1)
        await _evt(
            SubagentStarted(
                agent_id=agent_spec.id,
                parent_run_id=parent_run_id,
                scope=scope_dict.get("extension_id") if scope_dict else None,
            )
        )
        try:
            if timeout_seconds is not None:
                result = await asyncio.wait_for(_drive(), timeout=timeout_seconds)
            else:
                result = await _drive()
            await _evt(
                SubagentCompleted(
                    agent_id=agent_spec.id, run_id=child_run, status="succeeded"
                )
            )
            return SubagentResult(
                agent_id=agent_spec.id,
                scope=scope_dict,
                session_id=child_session,
                run_id=child_run,
                status="succeeded",
                output=getattr(result, "output", None),
            )
        except asyncio.TimeoutError:
            await _evt(
                SubagentErrored(
                    agent_id=agent_spec.id, reason=f"timeout after {timeout_seconds}s"
                )
            )
            return SubagentResult(
                agent_id=agent_spec.id,
                scope=scope_dict,
                session_id=child_session,
                run_id=child_run,
                status="failed",
                error={"reason": f"timeout after {timeout_seconds}s"},
            )
        except Exception as exc:  # child failures surface as structured errors
            from ..governance.security.redact import redact_exception

            safe_error = redact_exception(exc)
            await _evt(SubagentErrored(agent_id=agent_spec.id, reason=safe_error))
            return SubagentResult(
                agent_id=agent_spec.id,
                scope=scope_dict,
                session_id=child_session,
                run_id=child_run,
                status="failed",
                error={"error_type": type(exc).__name__, "reason": safe_error},
            )
        finally:
            _CURRENT_DEPTH.reset(token)


__all__: "list[str]" = ["SubagentExecutor"]
