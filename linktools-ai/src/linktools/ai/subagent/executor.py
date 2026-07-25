#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SubagentExecutor: the concrete executor that runs a resolved child AgentSpec
under a parent run.

Moved out of the Runtime composition root (it lived as a builder closure before)
so the build kernel only ASSEMBLES this executor -- it never creates sessions,
child runs, or drives execution itself (the builder constructs only). The
executor owns the subagent domain flow once assembled: child-run allocation via
the dispatcher, skill isolation for the child, timeout enforcement, subagent
event emission, and structured error redaction.

The executor does NOT create the child SessionRecord / RunRecord / RunDefinition
itself -- the RunDispatcher does (``open_child`` allocates the id + lineage
purely; ``dispatch`` creates the session + RunRecord + snapshot and drives the
full lifecycle). The executor only compiles the child AgentSpec and hands the
handle + compiled agent to ``dispatch``.

Dependencies: the storage (events, for subagent lifecycle events), the compiler
(to compile the child AgentSpec), and the run dispatcher -- a late-bound handle
resolved to the real runner once the runner exists, because the runner depends
on the capability resolver, which depends on this executor (a genuine
self-reference, confined to the single bind-once seam)."""

import asyncio
from typing import Any

from ..runtime.persistence.facade import Storage
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
        from ..run.dispatch import ChildSessionPolicy, RunDispatchRequest
        from ..run.models import RunInput

        parent_run_id = parent.run_id
        parent_session_id = parent.session_id

        # Allocate the child run id + scratch session via the dispatcher (the
        # sole id authority). open_child is pure -- the session + RunRecord +
        # snapshot are created only when dispatch runs, so a failure before
        # dispatch leaves no orphan. The scratch session is linked under the
        # parent's session so a worker pause/resume stays within the same
        # ownership domain.
        handle = await self._dispatcher.open_child(
            parent,
            ChildSessionPolicy(
                kind="scratch", parent_session_id=parent_session_id
            ),
            # No session_id_format -> the dispatcher mints a uuid session id;
            # the child RunRecord.runnable_id is compiled.spec.id (== agent_spec.id
            # here, no alias indirection), sourced by dispatch_child's fallback.
            {},
        )
        child_run = handle.run_id
        child_session = handle.session_id
        effective_root = handle.root_run_id

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
                compiled = await self._compiler.compile(agent_spec)
                # The dispatcher creates the child session + RunRecord +
                # resumable snapshot, then drives start/fence/execute/commit.
                return await self._dispatcher.dispatch(
                    RunDispatchRequest(
                        compiled_agent=compiled,
                        input=RunInput(prompt=task),
                        handle=handle,
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
