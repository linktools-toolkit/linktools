#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regressions for cancellation ownership at durable StepStore boundaries."""

import asyncio

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._durability import CommitObservation, DurableCommitState
from linktools.ai.runtime.state._steps import (
    RuntimeStepStore,
    _RunDurabilityFlight,
    _RunDurabilityKind,
    _RunHistoryLock,
)


@pytest.mark.asyncio
async def test_cancelled_step_flight_finishes_before_waiters_resume() -> None:
    store = object.__new__(RuntimeStepStore)
    store._background_tasks = set()
    store._durability_flights = {}
    store._history_lock = _RunHistoryLock()

    loop = asyncio.get_running_loop()
    flight = _RunDurabilityFlight(
        "run",
        "token",
        _RunDurabilityKind.TOOL_EFFECT,
        loop.create_future(),
    )
    store._durability_flights[flight.run_id] = flight

    started = asyncio.Event()
    release = asyncio.Event()

    async def operation() -> None:
        started.set()
        await release.wait()

    async def readback() -> CommitObservation[None]:
        raise AssertionError("readback is not used after a committed operation")

    async def wait_for_flight() -> None:
        await asyncio.shield(flight.completion)

    settler = asyncio.create_task(
        store._settle_durability_flight(flight, operation, readback)
    )
    await started.wait()
    settler.cancel()
    settler.cancel()
    await asyncio.sleep(0)

    assert not settler.done()
    assert store._durability_flights[flight.run_id] is flight
    assert not flight.completion.done()
    assert len(store._background_tasks) == 1

    waiter = asyncio.create_task(wait_for_flight())
    await asyncio.sleep(0)
    assert not waiter.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await settler
    await waiter

    assert flight.run_id not in store._durability_flights
    assert flight.completion.done()
    assert flight.completion.exception() is None
    assert store._background_tasks == set()


@pytest.mark.asyncio
async def test_step_flight_fences_only_after_real_unresolved_readback() -> None:
    store = object.__new__(RuntimeStepStore)
    store._background_tasks = set()
    store._durability_flights = {}
    store._history_lock = _RunHistoryLock()

    loop = asyncio.get_running_loop()
    flight = _RunDurabilityFlight(
        "run",
        "token",
        _RunDurabilityKind.TOOL_EFFECT,
        loop.create_future(),
    )
    store._durability_flights[flight.run_id] = flight

    async def operation() -> None:
        raise RuntimeError("commit failed")

    async def readback() -> CommitObservation[None]:
        return CommitObservation(
            DurableCommitState.UNRESOLVED,
            error=RuntimeError("readback unavailable"),
        )

    with pytest.raises(AIError) as raised:
        await store._settle_durability_flight(flight, operation, readback)

    assert raised.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
    assert flight.run_id in store._durability_flights
    assert flight.completion.done()
    completion_error = flight.completion.exception()
    assert isinstance(completion_error, AIError)
    assert completion_error.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
