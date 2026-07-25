#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LateBoundRunDispatcher: the runtime's dispatch binding seam.

The subagent executor is built before the AgentEngine it eventually delegates
to exists -- the runner depends on the capability resolver, which depends on
the subagent executor: a genuine self-reference, not an accidental cycle. This
handle confines that one-time forward reference to a single bind-once seam
instead of a bare closure; every caller only ever sees the narrow
``RunDispatcher`` Protocol, never the runner or build-kernel internals."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..run.context import RunContext
    from ..run.dispatch import (
        ChildRunHandle,
        ChildSessionPolicy,
        RunDispatchRequest,
        RunDispatcher,
    )
    from ..run.models import RunResult
    from collections.abc import Mapping
    from typing import Any as _Any


class LateBoundRunDispatcher:
    """A RunDispatcher bound to its real target after construction. Built by the
    build kernel, handed to the SubagentExecutor, and bound to the real runner
    once the runner exists."""

    def __init__(self) -> None:
        self._target: "RunDispatcher | None" = None

    def bind(self, target: "RunDispatcher") -> None:
        self._target = target

    async def open_child(
        self,
        parent_context: "RunContext",
        session_policy: "ChildSessionPolicy",
        metadata: "Mapping[str, _Any]",
    ) -> "ChildRunHandle":
        if self._target is None:
            raise RuntimeError(
                "LateBoundRunDispatcher.open_child() called before bind() -- "
                "the build kernel must bind the real dispatcher before any "
                "child run can be allocated"
            )
        return await self._target.open_child(parent_context, session_policy, metadata)

    async def dispatch(self, request: "RunDispatchRequest") -> "RunResult":
        if self._target is None:
            raise RuntimeError(
                "LateBoundRunDispatcher.dispatch() called before bind() -- "
                "the build kernel must bind the real dispatcher before any "
                "subagent execution can occur"
            )
        return await self._target.dispatch(request)


class CoordinatorRunDispatcher:
    """A RunDispatcher backed by RunCoordinator instead of AgentEngine. Child
    runs dispatched from Swarm/Subagent execution go through the SAME lifecycle
    a top-level run does (atomic start, claim/heartbeat/fencing, execute_pure,
    terminal commit) rather than a parallel AgentEngine-owned path -- so a
    child run's RunRecord, checkpoint, and session writes are owned by the
    Coordinator, the sole lifecycle owner.

    Two-step contract: ``open_child`` allocates the child run id + session id +
    lineage purely (no store writes) so a caller can record the id on its own
    domain state before execution; ``dispatch`` creates the child session +
    RunRecord, prepares the snapshot, and drives the full lifecycle, reducing
    the outcome to the RunResult the RunDispatcher Protocol promises (raising
    RunPaused on a pause, propagating on failure)."""

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator

    async def open_child(
        self,
        parent_context: "RunContext",
        session_policy: "ChildSessionPolicy",
        metadata: "Mapping[str, _Any]",
    ) -> "ChildRunHandle":
        return await self._coordinator.open_child_run(
            parent_context, session_policy, metadata
        )

    async def dispatch(self, request: "RunDispatchRequest") -> "RunResult":
        return await self._coordinator.dispatch_child(request)


__all__: "list[str]" = ["LateBoundRunDispatcher", "CoordinatorRunDispatcher"]
