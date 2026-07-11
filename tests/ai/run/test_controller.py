#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A (design note contract): RunController behavior. The controller is the
bridge between Runtime.cancel and the in-flight asyncio.Task -- it sets the
CancellationToken (observed at execution points) and cancels the task
(observed at any suspended await). These tests cover register/cancel/
unregister/get_token and the no-op-on-missing-run contract."""
import asyncio
import contextlib

import pytest

from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.controller import RunController


@pytest.mark.asyncio
async def test_get_token_returns_none_for_unregistered_run():
    controller = RunController()
    assert controller.get_token("missing") is None


@pytest.mark.asyncio
async def test_register_exposes_token_via_get_token():
    """After register(), get_token() returns the same token object that was
    passed in -- AgentRunner.execute() relies on this to look up the token
    for the run when Runtime.cancel checks ``in_flight``."""
    controller = RunController()

    async def _dummy():
        return "ok"

    task = asyncio.create_task(_dummy())
    try:
        token = CancellationToken()
        await controller.register("run-1", task, token)
        assert controller.get_token("run-1") is token
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task


@pytest.mark.asyncio
async def test_cancel_sets_token_and_cancels_task():
    """cancel() has TWO effects: (1) the token is set so the runner's next
    ``raise_if_cancelled()`` check raises CancelledError, and (2) the task
    is cancelled so any suspended await also unblocks. Both signals are
    needed -- the token only fires at execution points; ``task.cancel()``
    covers the gap inside a model call."""
    controller = RunController()

    async def _blocking():
        await asyncio.Event().wait()  # never set -- task hangs here

    task = asyncio.create_task(_blocking())
    token = CancellationToken()
    await controller.register("run-blocked", task, token)
    await asyncio.sleep(0)  # let the task reach its await

    await controller.cancel("run-blocked")

    assert token.is_cancelled() is True
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_cancel_is_noop_for_missing_run():
    """Cancelling a run that was never registered is a no-op (no exception).
    Runtime.cancel may call this on a stale run_id -- the controller must not
    require a registration to exist."""
    controller = RunController()
    await controller.cancel("never-registered")  # must not raise


@pytest.mark.asyncio
async def test_cancel_is_idempotent_on_already_cancelled_task():
    """A second cancel() on a task that's already done is a no-op (no
    exception). The controller checks ``task.done()`` before calling
    ``task.cancel()`` so it does not raise InvalidStateError on a finished
    task."""
    controller = RunController()

    async def _quick():
        return "done"

    task = asyncio.create_task(_quick())
    token = CancellationToken()
    await controller.register("run-quick", task, token)
    await task  # task completes naturally
    assert task.done()

    await controller.cancel("run-quick")  # must not raise on done task
    assert token.is_cancelled() is True


@pytest.mark.asyncio
async def test_unregister_removes_registration():
    """After unregister(), get_token() returns None and cancel() is a no-op.
    AgentRunner.execute() calls this in a ``finally`` block so the controller
    does not retain references to finished tasks (which would prevent GC of
    the run's frames)."""
    controller = RunController()

    async def _dummy():
        return "ok"

    task = asyncio.create_task(_dummy())
    token = CancellationToken()
    await controller.register("run-x", task, token)
    assert controller.get_token("run-x") is token

    await controller.unregister("run-x")
    assert controller.get_token("run-x") is None

    # cancel after unregister is a no-op (no task, no token).
    await controller.cancel("run-x")
    assert token.is_cancelled() is False
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


@pytest.mark.asyncio
async def test_unregister_is_idempotent():
    """Unregistering a run that was never registered, or unregistering twice,
    is a no-op. The finally block in execute() must be safe even if register
    was never reached (e.g. controller is wired but the very first transition
    threw)."""
    controller = RunController()
    await controller.unregister("never-registered")  # must not raise
    await controller.unregister("never-registered")  # still must not raise


@pytest.mark.asyncio
async def test_register_overwrites_stale_entry():
    """If a stale registration exists (e.g. the runner restarted without
    unregistering), register() overwrites it with the new task/token pair --
    the new in-flight run is authoritative. get_token returns the new token."""
    controller = RunController()

    async def _old():
        return "old"

    async def _new():
        return "new"

    old_task = asyncio.create_task(_old())
    old_token = CancellationToken()
    await controller.register("run-restart", old_task, old_token)
    await old_task  # completes -- simulates a stale entry

    new_task = asyncio.create_task(_new())
    new_token = CancellationToken()
    await controller.register("run-restart", new_task, new_token)

    assert controller.get_token("run-restart") is new_token
    await new_task
