#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyFencedRunEventWriter: SQL implementation of FencedRunEventWriter.

Within one Storage UoW: read the RunRecord, verify the fence's token equals
the stored execution_token, append the event, commit. A stale fence raises
RunFenceLostError before the event lands, so the Coordinator can route the
run into fail/fencing-loss convergence (the security-sensitive action that
triggered the append must NOT proceed).

The writer consumes an EventStore + RunStore that share the same
SQLAlchemy Storage UoW so the read-check-append-commit cycle is one
transaction. It does not own the transaction itself; the caller (typically
RunCoordinator) opens it via Storage.transaction()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..event_writer import RunFenceLostError

if TYPE_CHECKING:
    from ....events.context import EventStreamContext
    from ....events.store import EventStore
    from ....run.store import RunStore
    from ...commit import ExecutionFence


class SqlAlchemyFencedRunEventWriter:
    """SQL implementation of FencedRunEventWriter. The constructor takes the
    Storage the writer participates in; ``append_security`` opens one UoW
    that the read + check + append share, so a stale fence cannot slip an
    event into a transaction that committed before the claim was lost."""

    def __init__(self, storage: Any) -> None:
        self._storage = storage

    async def append_security(
        self,
        *,
        context: "EventStreamContext",
        fence: "ExecutionFence",
        event: Any,
    ) -> None:
        async with self._storage.transaction() as tx:
            # Read the current RunRecord within the append transaction so the
            # fence check + the event append are atomic against concurrent
            # claim/transition writes.
            record = await tx.runs.get(context.run_id)
            if record is None:
                raise RunFenceLostError(
                    f"fenced security event for run {context.run_id!r} cannot be "
                    f"appended: the run does not exist"
                )
            stored_token = record.execution_token or ""
            if not stored_token:
                raise RunFenceLostError(
                    f"fenced security event for run {context.run_id!r} cannot be "
                    f"appended: the run has no execution token to fence against"
                )
            if stored_token != fence.token:
                raise RunFenceLostError(
                    f"fence lost for run {context.run_id!r}: stored execution "
                    f"token differs from the presented fence"
                )
            # Fence verified; append the security event in the SAME
            # transaction so it lands iff the check passed.
            from ....events.context import append_event

            await append_event(tx.events, context, event)


__all__: "list[str]" = ["SqlAlchemyFencedRunEventWriter"]
