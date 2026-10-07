#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finite observation completion preserves authority, serial callbacks and ACKs."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import pytest

from linktools.ai.errors import AIError, ErrorCode, ObservationError
from linktools.ai.runtime import WaitResult
from linktools.ai.runtime._observation import _ObservationSession, _wait


@dataclass(frozen=True)
class _Event:
    cursor: str | None
    name: str


class _Owner:
    def __init__(self) -> None:
        self.sessions: set[_ObservationSession] = set()

    async def close(self) -> None:
        for session in tuple(self.sessions):
            await session.close()

    async def wait(
        self, *, waiter: Callable[[], Awaitable[object]],
        callback: Callable[[_Event], Awaitable[None]],
        finalize: Callable[[object, str | None], Awaitable[AsyncIterator[_Event]]],
        watch: Callable[[asyncio.Event], AsyncIterator[_Event]] | None = None,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 0.2,
    ) -> WaitResult[object]:
        return await _wait(
            scope="execution", resource_id="execution", waiter=waiter,
            watch=_empty_watch if watch is None else watch, on_event=callback,
            cursor="previous", timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            register=self.sessions.add, release=self.sessions.discard,
            finalize=finalize,
        )


def _empty_watch(ready: asyncio.Event) -> AsyncIterator[_Event]:
    ready.set()

    async def events() -> AsyncIterator[_Event]:
        if False:
            yield

    return events()


async def _ignore(event: _Event) -> None:
    pass


@pytest.mark.asyncio
async def test_final_coverage_uses_serial_callback_and_last_committed_ack() -> None:
    owner = _Owner()
    result = object()
    callback_entered = asyncio.Event()
    release_callback = asyncio.Event()
    authority_returned = asyncio.Event()
    finalizer_entered = asyncio.Event()
    order = []
    active_callbacks = 0

    async def watch(ready: asyncio.Event) -> AsyncIterator[_Event]:
        ready.set()
        yield _Event("live-ack", "live")
        await asyncio.Event().wait()

    async def waiter() -> object:
        await callback_entered.wait()
        authority_returned.set()
        return result

    async def callback(event: _Event) -> None:
        nonlocal active_callbacks
        active_callbacks += 1
        assert active_callbacks == 1
        try:
            if event.name == "live":
                callback_entered.set()
                await release_callback.wait()
            order.append(event.name)
        finally:
            active_callbacks -= 1

    async def finalize(value: object, cursor: str | None) -> AsyncIterator[_Event]:
        finalizer_entered.set()
        assert value is result
        assert cursor == "live-ack"
        assert order == ["live"]

        async def events() -> AsyncIterator[_Event]:
            yield _Event(cursor, "projection")
            yield _Event("final-ack", "checkpoint")

        return events()

    waiting = asyncio.create_task(owner.wait(waiter=waiter, callback=callback,
        finalize=finalize, watch=watch))
    try:
        await asyncio.wait_for(authority_returned.wait(), 1)
        await asyncio.sleep(0)
        assert not finalizer_entered.is_set()
        release_callback.set()
        outcome = await asyncio.wait_for(waiting, 1)
        assert outcome.result is result
        assert outcome.cursor == "final-ack"
        assert outcome.observation_error is None
        assert order == ["live", "projection", "checkpoint"]
        assert not owner.sessions
    finally:
        release_callback.set()
        await owner.close()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_final_callback_failure_preserves_ack_and_closes_finite_iterator() -> None:
    owner = _Owner()
    result = object()
    closed = asyncio.Event()
    cause = ValueError("callback failed")
    received = []

    async def waiter() -> object:
        return result

    async def finalize(value: object, cursor: str | None) -> AsyncIterator[_Event]:
        assert value is result and cursor == "previous"

        async def events() -> AsyncIterator[_Event]:
            try:
                yield _Event("committed", "first")
                yield _Event("uncommitted", "second")
                yield _Event("forbidden", "third")
            finally:
                closed.set()

        return events()

    async def callback(event: _Event) -> None:
        received.append(event.name)
        if event.name == "second":
            raise cause

    with pytest.raises(ObservationError) as raised:
        await owner.wait(waiter=waiter, callback=callback, finalize=finalize)
    assert raised.value.origin == "callback"
    assert raised.value.cursor == "committed"
    assert raised.value.__cause__ is cause
    assert received == ["first", "second"]
    assert closed.is_set()
    assert not owner.sessions


@pytest.mark.asyncio
async def test_authority_ready_at_deadline_is_retained_without_starting_final_coverage() -> None:
    owner = _Owner()
    result = object()
    finalizer_started = False

    async def waiter() -> object:
        return result

    async def finalize(value: object, cursor: str | None) -> AsyncIterator[_Event]:
        nonlocal finalizer_started
        finalizer_started = True
        return _empty_watch(asyncio.Event())

    outcome = await owner.wait(waiter=waiter, callback=_ignore, finalize=finalize,
        timeout_seconds=0)
    assert outcome.result is result
    assert outcome.cursor == "previous"
    assert outcome.observation_error.safe_details["phase"] == "drain"
    assert outcome.observation_error.safe_details["coverage_complete"] is False
    assert not finalizer_started
    assert not owner.sessions


@pytest.mark.asyncio
async def test_final_coverage_has_internal_budget_even_without_business_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._observation as observation

    monkeypatch.setattr(observation, "_FINAL_DRAIN_TIMEOUT", 0.01)
    owner = _Owner()
    result = object()
    finalizer_started = asyncio.Event()
    finalizer_closed = asyncio.Event()

    async def waiter() -> object:
        return result

    async def finalize(value: object, cursor: str | None) -> AsyncIterator[_Event]:
        finalizer_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalizer_closed.set()

    outcome = await asyncio.wait_for(owner.wait(waiter=waiter, callback=_ignore,
        finalize=finalize), 1)
    assert outcome.result is result
    assert outcome.cursor == "previous"
    assert outcome.observation_error.safe_details["phase"] == "drain"
    assert finalizer_started.is_set() and finalizer_closed.is_set()
    assert not owner.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [ErrorCode.AUTHORIZATION_DENIED, ErrorCode.STORAGE_INTEGRITY_ERROR])
async def test_final_coverage_contract_failure_is_not_downgraded_to_presentation_error(code: ErrorCode) -> None:
    owner = _Owner()
    cause = AIError(code)

    async def waiter() -> object:
        return object()

    async def finalize(value: object, cursor: str | None) -> AsyncIterator[_Event]:
        raise cause

    with pytest.raises(AIError) as raised:
        await owner.wait(waiter=waiter, callback=_ignore, finalize=finalize)
    assert raised.value is cause
    assert not owner.sessions


@pytest.mark.asyncio
async def test_noncooperative_finalizer_stays_owned_and_never_calls_back_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import linktools.ai.runtime._observation as observation

    monkeypatch.setattr(observation, "_FINAL_DRAIN_TIMEOUT", 0.01)
    owner = _Owner()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    received = []

    async def waiter() -> object:
        return object()

    async def callback(event: _Event) -> None:
        received.append(event)

    async def finalize(value: object, cursor: str | None) -> AsyncIterator[_Event]:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

        async def events() -> AsyncIterator[_Event]:
            yield _Event("late", "late")

        return events()

    try:
        with pytest.raises(ObservationError) as raised:
            await asyncio.wait_for(owner.wait(waiter=waiter, callback=callback,
                finalize=finalize, close_timeout_seconds=0.01), 1)
        assert raised.value.safe_details["phase"] == "cleanup"
        assert raised.value.safe_details["cleanup_pending"] is True
        assert raised.value.cursor == "previous"
        assert cancelled.is_set()
        assert owner.sessions
        assert received == []
    finally:
        release.set()
        await owner.close()
    assert received == []
    assert not owner.sessions


@pytest.mark.asyncio
async def test_optional_live_failure_found_during_final_handover_is_retained() -> None:
    owner = _Owner()
    result = object()
    read_started = asyncio.Event()
    cause = RuntimeError("optional broker unavailable")

    async def watch(ready: asyncio.Event) -> AsyncIterator[_Event]:
        ready.set()
        read_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise ObservationError("stream", cursor="previous") from cause
        yield

    async def waiter() -> object:
        await read_started.wait()
        return result

    async def finalize(value: object, cursor: str | None) -> AsyncIterator[_Event]:
        return _empty_watch(asyncio.Event())

    outcome = await owner.wait(waiter=waiter, callback=_ignore, finalize=finalize, watch=watch)
    assert outcome.result is result
    assert outcome.observation_error is not None
    assert outcome.observation_error.origin == "stream"
    assert outcome.observation_error.cursor == "previous"
    assert outcome.observation_error.__cause__ is cause
    assert not owner.sessions
