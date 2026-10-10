#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cancellation and failure semantics of an owned resource context."""

import asyncio
import gc
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from linktools.ai.runtime._evaluation_scope import _EnteredTrialScope


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", (True, False))
async def test_entry_waiter_does_not_hide_cleanup_failure(cancel_waiter: bool) -> None:
    entered = asyncio.Event()
    cleanup_calls = []
    errors = []
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))

    @asynccontextmanager
    async def context() -> AsyncIterator[object]:
        try:
            entered.set()
            await asyncio.Event().wait()
            yield object()
        finally:
            cleanup_calls.append("exit")
            raise ValueError("partial entry cleanup failed")

    try:
        scope = _EnteredTrialScope(context())
        waiter = asyncio.create_task(scope.engine())
        await entered.wait()
        if cancel_waiter:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
        for _ in range(2):
            with pytest.raises(ValueError, match="partial entry cleanup failed"):
                await scope.close()
        if not cancel_waiter:
            with pytest.raises(ValueError, match="partial entry cleanup failed"):
                await waiter
        assert cleanup_calls == ["exit"]
        del scope, waiter
        gc.collect()
        await asyncio.sleep(0)
        assert errors == []
    finally:
        loop.set_exception_handler(previous)


@pytest.mark.asyncio
async def test_concurrent_close_allows_cancelled_entry_to_finish_and_exit_once() -> None:
    entered, cancelled, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    exits = []
    marker = object()

    @asynccontextmanager
    async def context() -> AsyncIterator[object]:
        try:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                await finish.wait()
            yield marker
        finally:
            exits.append(asyncio.current_task())

    scope = _EnteredTrialScope(context())
    await entered.wait()
    first = asyncio.create_task(scope.close())
    await cancelled.wait()
    second = asyncio.create_task(scope.close())
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    finish.set()
    await asyncio.gather(first, second)
    assert await scope.engine() is marker
    await scope.close()
    assert len(exits) == 1


@pytest.mark.asyncio
async def test_observed_entry_failure_does_not_become_a_second_close_failure() -> None:
    attempts = []

    @asynccontextmanager
    async def context() -> AsyncIterator[object]:
        attempts.append("entry")
        raise ValueError("resource unavailable")
        yield object()

    scope = _EnteredTrialScope(context())
    with pytest.raises(ValueError, match="resource unavailable"):
        await scope.engine()
    await scope.close()
    await scope.close()
    assert attempts == ["entry"]


@pytest.mark.asyncio
async def test_close_before_context_starts_unblocks_the_entry_waiter() -> None:
    attempts = []

    @asynccontextmanager
    async def context() -> AsyncIterator[object]:
        attempts.append("entry")
        yield object()

    scope = _EnteredTrialScope(context())
    await scope.close()
    with pytest.raises(asyncio.CancelledError):
        await scope.engine()
    await scope.close()
    assert attempts == []


@pytest.mark.asyncio
async def test_close_fences_new_borrows_and_waits_for_existing_control() -> None:
    exited = asyncio.Event()
    marker = object()

    @asynccontextmanager
    async def context() -> AsyncIterator[object]:
        try:
            yield marker
        finally:
            exited.set()

    scope = _EnteredTrialScope(context())
    await scope.engine()
    with scope.borrow_engine() as engine:
        assert engine is marker
        closing = asyncio.create_task(scope.close())
        await asyncio.sleep(0)
        assert scope.closing
        with scope.borrow_engine() as unavailable:
            assert unavailable is None
        repeated = asyncio.create_task(scope.close())
        await asyncio.sleep(0)
        assert not exited.is_set()
        assert not closing.done() and not repeated.done()
    await asyncio.gather(closing, repeated)
    assert exited.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("exception", "cancel"))
async def test_failed_control_returns_its_borrow_before_scope_close(failure: str) -> None:
    borrowed, exited = asyncio.Event(), asyncio.Event()

    @asynccontextmanager
    async def context() -> AsyncIterator[object]:
        try:
            yield object()
        finally:
            exited.set()

    scope = _EnteredTrialScope(context())
    await scope.engine()

    async def control() -> None:
        with scope.borrow_engine() as engine:
            assert engine is not None
            borrowed.set()
            if failure == "exception":
                raise ValueError("control failed")
            await asyncio.Event().wait()

    task = asyncio.create_task(control())
    await borrowed.wait()
    if failure == "cancel":
        task.cancel()
    with pytest.raises(ValueError if failure == "exception" else asyncio.CancelledError):
        await task
    await scope.close()
    assert exited.is_set()
