#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventStreamContext + append_event. Bundles the six lineage fields every
event-store append needs, so callers stop repeating ``stream_id=...,
run_id=..., root_execution_id=..., parent_execution_id=..., session_id=...,
runnable_id=...`` at each call site. ``append_event(store, context, payload)``
is the single helper."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol

class _RunContextLike(Protocol):
    """Structural shape ``from_run_context`` reads. Defined here (rather than
    importing ``run.context.RunContext``) so the ``events`` package does not
    depend on ``run`` -- ``run`` imports ``events`` (ExecutionService owns
    lifecycle-event emission, per the run-owner work package), so a
    ``RunContext`` import in this direction would form a top-level 2-cycle.
    ``RunContext`` satisfies this Protocol structurally."""

    run_id: str
    root_execution_id: "str | None"
    parent_execution_id: "str | None"
    session_id: str
    runnable_id: str


@dataclass(frozen=True, slots=True)
class EventStreamContext:
    stream_id: str
    run_id: str
    root_execution_id: str
    parent_execution_id: "str | None"
    session_id: str
    runnable_id: str

    @classmethod
    def from_run_context(
        cls, ctx: "_RunContextLike", *, stream_id: "str | None" = None
    ) -> "EventStreamContext":
        """Build an EventStreamContext from a RunContext-like object. ``stream_id``
        defaults to the run_id (the common case -- every current caller passes
        stream_id == run_id)."""
        run_id = ctx.run_id
        root = ctx.root_execution_id or run_id
        return cls(
            stream_id=stream_id or run_id,
            run_id=run_id,
            root_execution_id=root,
            parent_execution_id=ctx.parent_execution_id,
            session_id=ctx.session_id,
            runnable_id=ctx.runnable_id,
        )


async def append_event(
    store: Any,
    context: EventStreamContext,
    payload: Any,
    *,
    metadata: "Mapping[str, Any] | None" = None,
) -> None:
    """Append ``payload`` to ``store`` under the EventStreamContext's lineage.

    ``metadata`` is optional free-form per-event metadata (e.g. a ``commit_id``
    used for commit-scoped dedup of critical events); backends persist it
    alongside the payload."""
    await store.append(
        stream_id=context.stream_id,
        run_id=context.run_id,
        root_execution_id=context.root_execution_id,
        parent_execution_id=context.parent_execution_id,
        session_id=context.session_id,
        runnable_id=context.runnable_id,
        payload=payload,
        metadata=metadata,
    )


async def append_event_once(
    store: Any,
    context: EventStreamContext,
    payload: Any,
    *,
    commit_id: str,
    metadata: "Mapping[str, Any] | None" = None,
) -> None:
    """Commit-scoped idempotent append: the store atomically reserves
    ``(stream_id, commit_id)`` and a retried call with the same commit_id
    returns the original envelope instead of re-appending. Use this for the
    state-critical events a Run commit records (RunPaused, RunCompleted,
    RunFailed, RunCancelled, RunStarted, RunResumed, ApprovalRequested) so a
    retry does not duplicate them. The store owns the dedup via its
    stable-id/commit-id index -- callers MUST NOT pre-scan events.list() to
    decide whether to append."""
    merged = dict(metadata) if metadata else {}
    merged.setdefault("commit_id", commit_id)
    await store.append_once(
        stream_id=context.stream_id,
        run_id=context.run_id,
        root_execution_id=context.root_execution_id,
        parent_execution_id=context.parent_execution_id,
        session_id=context.session_id,
        runnable_id=context.runnable_id,
        payload=payload,
        metadata=merged,
        commit_id=commit_id,
    )
