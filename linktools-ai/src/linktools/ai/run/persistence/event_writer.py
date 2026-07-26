#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FencedRunEventWriter: append a security event only after verifying the
caller's execution fence still owns the run (P6 fenced security event).

A security-sensitive action (a privileged tool call, a governance decision)
must not be persisted AFTER the claiming execution has lost its lease. The
writer reads the run + verifies the fence WITHIN the same transaction as the
event append, so a fence that went stale mid-action surfaces as a
RunFenceLostError before the event lands. The Coordinator routes that error
into the run's fail/fencing-loss convergence path; it does NOT record a
warning and continue (the spec is explicit: a stale fence must NOT let the
sensitive action proceed).

The Protocol stays storage-shape-only (context + fence + event); the SQL +
Filesystem adapters implement it against their own run-store + event-store.
RunCoordinator builds a SecurityEventSink per execution; Agent/Tool/Capability
consume the sink and never reach the writer or the EventStore directly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..events.context import EventStreamContext
    from ..commit import ExecutionFence


class RunFenceLostError(Exception):
    """The execution fence the caller presented no longer owns the run (the
    stored execution_token differs, or the run is no longer RUNNING under
    this claim). The security-sensitive action that triggered the append
    must NOT proceed; the Coordinator routes the run into fail or
    fencing-loss convergence."""


@runtime_checkable
class FencedRunEventWriter(Protocol):
    """Append a security event under a verified execution fence. The fence
    is checked WITHIN the same transaction as the append so a stale claim
    cannot land an event it should not have been able to produce."""

    async def append_security(
        self,
        *,
        context: "EventStreamContext",
        fence: "ExecutionFence",
        event: object,
    ) -> None:
        """Verify ``fence.token`` matches the run's stored execution_token
        (within the append transaction), then append ``event`` to the run's
        event stream. Raise RunFenceLostError on a mismatch -- do NOT
        swallow and continue."""
        ...


__all__: "list[str]" = ["FencedRunEventWriter", "RunFenceLostError"]
