#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CancellationToken: cooperative cancellation signal for agent runs.

Review doc §7.2: cancellation must reach the actual execution points (model
call, tool call, swarm iteration), not just flip the database status. The
token is the propagation channel -- Runtime.cancel sets it via RunController,
and AgentRunner.execute() checks it at the documented await points via
``await token.raise_if_cancelled()``.

The token is asyncio-safe (built on ``asyncio.Event``) but NOT thread-safe:
it is shared between the task driving the run and the task calling
RunController.cancel within the same event loop. Cross-process cancellation
goes through the database (CANCELLING status) and is observed by the worker
polling ``raise_if_cancelled()`` -- the token is the in-process fast path."""


import asyncio


class CancellationToken:
    """Cooperative cancellation signal. Checked at execution points.

    A token is created per Run by AgentRunner.execute() and registered with
    RunController. The token has two states -- not-set (run is in flight) and
    set (cancel requested). The runner awaits ``raise_if_cancelled()`` before
    and after the model call; RunController.cancel() calls ``cancel()`` which
    flips the state, so the next ``raise_if_cancelled()`` raises
    ``asyncio.CancelledError`` and the lifecycle's outer handler lands the run
    in CANCELLED via CANCELLING.

    Deliberately tiny: the complexity lives in *where* to check the token, not
    in the token itself. Keep this class minimal so the cancellation contract
    is obvious from the signature."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Mark the run as cancelled. Idempotent: a second call is a no-op
        (Event.set() is itself idempotent). Does NOT raise -- the runner
        observes the cancellation at its next ``raise_if_cancelled()`` check
        or via the asyncio.Task.cancel() that RunController issues in tandem."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Non-blocking check. Used by code paths that cannot afford to raise
        (e.g. cleanup that wants to branch on cancellation without aborting
        the cleanup itself)."""
        return self._event.is_set()

    async def raise_if_cancelled(self) -> None:
        """Awaitable check used at execution points. Raises
        ``asyncio.CancelledError`` if the token has been set, so the
        cancellation surfaces through the same code path as a real asyncio
        task cancellation -- the runner's outer ``except CancelledError``
        handler does the CANCELLING -> CANCELLED transition. No-op when the
        token is not set, so callers can await it unconditionally."""
        if self._event.is_set():
            raise asyncio.CancelledError("run cancelled by request")
