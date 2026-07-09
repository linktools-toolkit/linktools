#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunController: tracks in-flight asyncio Tasks + their CancellationTokens.

Review doc §7.1: ``Runtime.cancel(run_id)`` must actually stop the running
task, not just update the database status. The controller is the bridge --
when AgentRunner.execute() starts driving a run, it registers the driving
``asyncio.Task`` and a fresh ``CancellationToken`` here. Runtime.cancel() then
calls ``RunController.cancel(run_id)`` which (a) flips the token so the
runner's next ``raise_if_cancelled()`` check aborts and (b) calls
``task.cancel()`` so any in-flight await (model call, tool call, stream) also
unblocks via the standard asyncio cancellation path.

Both signals are needed because the token check only fires at the documented
execution points (between model calls); a hanging await *inside* a model call
would never reach the check. ``task.cancel()`` covers that case by injecting
``CancelledError`` at whatever await point is currently suspended. Conversely,
the token covers cancellation between execution points even when no await is
in flight (the runner can observe it proactively before the next model call).

The controller is single-process only. Cross-process cancel goes through the
database (RunStatus.CANCELLING) and is observed by the worker's token-check
loop -- this object never touches the database."""

import asyncio
from typing import TYPE_CHECKING

from .cancellation import CancellationToken

if TYPE_CHECKING:
    pass


class RunController:
    """In-memory registry of in-flight Runs keyed by run_id. Not shared across
    processes. Methods are async for symmetry with the rest of the runner
    surface and so callers can ``await`` them uniformly, even though the
    internal state mutations are synchronous dict operations (the asyncio
    machinery yields control on the await regardless)."""

    def __init__(self) -> None:
        self._tasks: "dict[str, asyncio.Task]" = {}
        self._tokens: "dict[str, CancellationToken]" = {}

    async def register(
        self,
        run_id: str,
        task: "asyncio.Task",
        token: CancellationToken,
    ) -> None:
        """Associate ``task`` + ``token`` with ``run_id``. Called by
        AgentRunner.execute() at the start of the lifecycle (after the
        RUNNING transition). If a stale registration already exists for the
        same run_id (e.g. the runner restarted after a crash without
        unregistering), it is overwritten -- the new task/token pair is the
        authoritative in-flight run."""
        self._tasks[run_id] = task
        self._tokens[run_id] = token

    async def cancel(self, run_id: str) -> None:
        """Signal cancellation to the in-flight run, if any. Two effects:

        1. Sets the token -- so the runner's next ``raise_if_cancelled()``
           check raises CancelledError (covers cancellation between execution
           points, when no await is suspended).
        2. Calls ``task.cancel()`` -- so any currently-suspended await also
           unblocks with CancelledError (covers a hanging model call).

        Idempotent: a second call on an already-cancelled run is a no-op
        (``token.cancel()`` and ``task.cancel()`` are both idempotent). A
        missing registration (no in-flight task for this run_id) is also a
        no-op -- Runtime.cancel handles the database side regardless."""
        token = self._tokens.get(run_id)
        if token is not None:
            token.cancel()
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()

    async def unregister(self, run_id: str) -> None:
        """Remove the registration. Called by AgentRunner.execute() in a
        finally block so the controller does not retain references to
        finished tasks (which would prevent GC of the run's frames).
        Idempotent -- a missing registration is a no-op."""
        self._tasks.pop(run_id, None)
        self._tokens.pop(run_id, None)

    def get_token(self, run_id: str) -> "CancellationToken | None":
        """Non-async accessor for code that needs the token without awaiting
        (e.g. to pass it to a sync helper, or to check membership). Returns
        None when no in-flight task is registered for ``run_id``."""
        return self._tokens.get(run_id)
