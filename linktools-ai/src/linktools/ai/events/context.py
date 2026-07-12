#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventContext + append_event (spec §12.2). Bundles the six lineage fields every
event-store append needs, so callers stop repeating ``stream_id=...,
run_id=..., root_run_id=..., parent_run_id=..., session_id=...,
runnable_id=...`` at each call site. ``append_event(store, context, payload)``
is the single helper."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..run.context import RunContext
    from .store import EventStore


@dataclass(frozen=True, slots=True)
class EventContext:
    stream_id: str
    run_id: str
    root_run_id: str
    parent_run_id: "str | None"
    session_id: str
    runnable_id: str

    @classmethod
    def from_run_context(
        cls, ctx: "RunContext", *, stream_id: "str | None" = None
    ) -> "EventContext":
        """Build an EventContext from a RunContext. ``stream_id`` defaults to the
        run_id (the common case -- every current caller passes stream_id ==
        run_id)."""
        run_id = ctx.run_id
        root = ctx.root_run_id or run_id
        return cls(
            stream_id=stream_id or run_id,
            run_id=run_id,
            root_run_id=root,
            parent_run_id=ctx.parent_run_id,
            session_id=ctx.session_id,
            runnable_id=ctx.runnable_id,
        )


async def append_event(
    store: "EventStore", context: EventContext, payload: Any
) -> None:
    """Append ``payload`` to ``store`` under the EventContext's lineage."""
    await store.append(
        stream_id=context.stream_id,
        run_id=context.run_id,
        root_run_id=context.root_run_id,
        parent_run_id=context.parent_run_id,
        session_id=context.session_id,
        runnable_id=context.runnable_id,
        payload=payload,
    )
