#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime.cancel(run_id) -- best-effort store-level cancel for single
Agent runs. Mirrors SwarmRunner.cancel's store-only approach: flips the
RunRecord to CANCELLED without cancelling any live asyncio.Task driving the run
(the caller cancels that separately; AgentRunner catches CancelledError and
lands in the same CANCELLED state)."""

import asyncio
import contextlib
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import RunConflictError, RunNotFoundError
from linktools.ai.run.cancellation import CancellationToken
from linktools.ai.run.models import (
    RunInput,
    RunRecord,
    RunnableType,
    RunStatus,
)
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage

_NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)


def _seed_run(store, run_id: str, status: RunStatus) -> None:
    """Seed a RunRecord directly into the store at the given status. The
    ALLOWED_RUN_TRANSITIONS table permits only PENDING -> RUNNING, then RUNNING
    -> {SUCCEEDED, FAILED, CANCELLED, ...}, so non-PENDING targets are reached
    by chaining two transitions -- this reproduces a realistic version (the
    terminal record ends up at version 3) without bypassing the store."""

    async def _seed():
        await store.runs.create(
            RunRecord(
                id=run_id,
                root_run_id=run_id,
                parent_run_id=None,
                session_id="session-x",
                runnable_id="agent-x",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.PENDING,
                input=RunInput(prompt="seed"),
                result=None,
                error=None,
                version=1,
                created_at=_NOW,
                started_at=None,
                finished_at=None,
            )
        )
        if status is RunStatus.PENDING:
            return
        await store.runs.transition(
            run_id,
            RunStatus.RUNNING,
            expected_version=1,
        )
        if status is RunStatus.RUNNING:
            return
        await store.runs.transition(
            run_id,
            status,
            expected_version=2,
        )

    asyncio.run(_seed())


# 1. cancel(run_id) on a RUNNING run -> transitions to CANCELLED.


def test_cancel_running_run_transitions_to_cancelled(tmp_path):
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)
    _seed_run(storage, "run-running", RunStatus.RUNNING)

    async def _cancel():
        await runtime.cancel("run-running")

    asyncio.run(_cancel())

    async def _verify():
        return await storage.runs.get("run-running")

    record = asyncio.run(_verify())
    assert record is not None
    assert record.status is RunStatus.CANCELLED


# 2. cancel(run_id) on a SUCCEEDED run -> no-op (already terminal).


def test_cancel_succeeded_run_is_noop(tmp_path):
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)
    _seed_run(storage, "run-done", RunStatus.SUCCEEDED)

    async def _cancel():
        await runtime.cancel("run-done")

    # already terminal -- cancel must not raise and must not change status.
    asyncio.run(_cancel())

    async def _verify():
        return await storage.runs.get("run-done")

    record = asyncio.run(_verify())
    assert record is not None
    assert record.status is RunStatus.SUCCEEDED


# 2b. cancel(run_id) on an already-CANCELLED run -> also no-op (idempotent).


def test_cancel_already_cancelled_run_is_noop(tmp_path):
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)
    _seed_run(storage, "run-cancelled", RunStatus.CANCELLED)

    async def _cancel():
        await runtime.cancel("run-cancelled")

    asyncio.run(_cancel())

    async def _verify():
        return await storage.runs.get("run-cancelled")

    record = asyncio.run(_verify())
    assert record is not None
    assert record.status is RunStatus.CANCELLED


# 3. cancel(run_id) on a missing run -> RunNotFoundError.


def test_cancel_missing_run_raises_not_found(tmp_path):
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)

    async def _cancel():
        await runtime.cancel("does-not-exist")

    with pytest.raises(RunNotFoundError):
        asyncio.run(_cancel())


# 4. cancel(run_id) called twice on an in-flight RUNNING run: the first call
#    does the real RUNNING -> CANCELLING transition + signals the controller;
#    the second call (a repeat cancel request racing the runner's own
#    CANCELLING -> CANCELLED finalization) must NOT attempt the illegal
#    CANCELLING -> CANCELLING edge, and must re-signal the controller instead
#    (idempotent, matches RunController.cancel's own idempotency).


def test_cancel_inflight_cancelling_run_is_idempotent(tmp_path):
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)
    _seed_run(storage, "run-1", RunStatus.RUNNING)

    async def _scenario():
        token = CancellationToken()
        task = asyncio.create_task(asyncio.sleep(60))
        await runtime._components.run_controller.register("run-1", task, token)
        try:
            await runtime.cancel("run-1")
            first = await storage.runs.get("run-1")
            assert first is not None
            assert first.status is RunStatus.CANCELLING

            # Second call must not raise InvalidRunTransitionError and must
            # leave the record in CANCELLING (the runner -- not Runtime.cancel
            # -- owns the CANCELLING -> CANCELLED finalization).
            await runtime.cancel("run-1")
            return await storage.runs.get("run-1")
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    record = asyncio.run(_scenario())
    assert record is not None
    assert record.status is RunStatus.CANCELLING


# 5. Concurrent cancel(): the CANCELLING transition itself can lose a version
#    race to another caller. When that happens, Runtime.cancel must re-read
#    the fresh record and, if it is now CANCELLING (the other caller won),
#    treat it the same as the idempotent repeat-cancel path above instead of
#    propagating RunConflictError.


def test_cancel_handles_conflict_when_fresh_status_is_cancelling(tmp_path, monkeypatch):
    storage = FileStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)
    _seed_run(storage, "run-2", RunStatus.RUNNING)

    async def _scenario():
        token = CancellationToken()
        task = asyncio.create_task(asyncio.sleep(60))
        await runtime._components.run_controller.register("run-2", task, token)

        original_transition = storage.runs.transition
        state = {"simulated": False}

        async def _flaky_transition(run_id, target, *, expected_version, **kwargs):
            if (
                not state["simulated"]
                and run_id == "run-2"
                and target is RunStatus.CANCELLING
            ):
                state["simulated"] = True
                # Simulate a concurrent winner landing RUNNING -> CANCELLING
                # first, then this caller's own CAS losing the version race.
                await original_transition(
                    run_id,
                    target,
                    expected_version=expected_version,
                    **kwargs,
                )
                raise RunConflictError("simulated concurrent cancel")
            return await original_transition(
                run_id,
                target,
                expected_version=expected_version,
                **kwargs,
            )

        monkeypatch.setattr(storage.runs, "transition", _flaky_transition)

        try:
            await runtime.cancel("run-2")
            assert token.is_cancelled()
            return await storage.runs.get("run-2")
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    record = asyncio.run(_scenario())
    assert record is not None
    assert record.status is RunStatus.CANCELLING
